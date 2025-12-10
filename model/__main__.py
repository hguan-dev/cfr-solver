import logging
import pathlib

import polars as pl
import sklearn
from constants import RANDOM_SEED
from models import train_models
from utils import evaluate_performance

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)


def extract_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Assumptions:
    - No ordering among suits, e.g. spades is NOT preferred over hearts.
    - Card numbers: int[1, 52] -> [hearts, diamonds, clubs, spades] * [int[2, 14]]

    Findings:
    - Pure card numbers or suits & ranks do not offer predictive power.
    """
    logger.info("Extracting features...")
    df_features = df[:, :-3]
    df_cards = df_features.drop("History")
    df_history = df_features.select("History")

    # 1. compute suits and ranks
    suit_and_rank_exprs = []
    for col in df_cards.columns:
        # create suit column
        suit_col = f"{col}_suit"
        suit_expr = (
            pl.when(pl.col(col) == 0)
            .then(0)
            .otherwise((pl.col(col) - 1) // 13)
            .alias(suit_col)
        )

        # create rank column
        rank_col = f"{col}_rank"
        rank_expr = (
            pl.when(pl.col(col) == 0)
            .then(0)
            .otherwise((((pl.col(col) - 2) % 13) + 2))
            .alias(rank_col)
        )

        suit_and_rank_exprs.extend([suit_expr, rank_expr])

    df_suit_and_rank = df_cards.select(suit_and_rank_exprs)

    # 2. compute hole card features
    df_hole_card_features = df_suit_and_rank.select(
        [
            # ordered ranks
            pl.max_horizontal("C1_rank", "C2_rank").alias("high_rank"),
            pl.min_horizontal("C1_rank", "C2_rank").alias("low_rank"),
            # rank gap
            (pl.col("C1_rank") - pl.col("C2_rank")).abs().alias("rank_gap"),
            # pair detection
            (pl.col("C1_rank") == pl.col("C2_rank")).cast(pl.UInt8).alias("is_pair"),
            # same suit
            (
                (pl.col("C1_suit") == pl.col("C2_suit"))
                & (pl.col("C1_rank") > 0)
                & (pl.col("C2_rank") > 0)
            )
            .cast(pl.UInt8)
            .alias("is_suited"),
            # connected
            ((pl.col("C1_rank") - pl.col("C2_rank")).abs() == 1)
            .cast(pl.UInt8)
            .alias("is_connected"),
            # broadway cards
            (
                (pl.col("C1_rank") >= 10)
                & (pl.col("C2_rank") >= 10)
                & (pl.col("C1_rank") > 0)
                & (pl.col("C2_rank") > 0)
            )
            .cast(pl.UInt8)
            .alias("is_broadway"),
        ]
    )

    # 3. compute community card features
    community_rank_cols = [
        col
        for col in df_suit_and_rank.columns
        if col.startswith("B") and col.endswith("_rank")
    ]
    community_suit_cols = [
        col
        for col in df_suit_and_rank.columns
        if col.startswith("B") and col.endswith("_suit")
    ]

    # count active community cards
    active_exprs = [(pl.col(col) > 0).cast(pl.UInt8) for col in community_rank_cols]
    community_active_expr = pl.sum_horizontal(active_exprs)

    # max and min community ranks
    max_expr = pl.max_horizontal([pl.col(col) for col in community_rank_cols])
    min_expr = pl.min_horizontal([pl.col(col) for col in community_rank_cols])

    # has A or K
    has_ace_expr = pl.any_horizontal([pl.col(col) == 14 for col in community_rank_cols])
    has_king_expr = pl.any_horizontal(
        [pl.col(col) == 13 for col in community_rank_cols]
    )
    has_queen_expr = pl.any_horizontal(
        [pl.col(col) == 12 for col in community_rank_cols]
    )

    # paired community detection
    paired_conds = []
    for i in range(len(community_rank_cols)):
        for j in range(i + 1, len(community_rank_cols)):
            cond = (
                pl.col(community_rank_cols[i]) == pl.col(community_rank_cols[j])
            ) & (pl.col(community_rank_cols[i]) > 0)
            paired_conds.append(cond)

    community_paired_expr = (
        pl.fold(
            acc=pl.lit(False), function=lambda acc, x: acc | x, exprs=paired_conds
        ).cast(pl.UInt8)
        if paired_conds
        else pl.lit(0).cast(pl.UInt8)
    )

    # same suit community cards
    suited_conds = []
    if len(community_suit_cols) >= 3:
        for i in range(len(community_suit_cols)):
            for j in range(i + 1, len(community_suit_cols)):
                for k in range(j + 1, len(community_suit_cols)):
                    cond = (
                        (
                            pl.col(community_suit_cols[i])
                            == pl.col(community_suit_cols[j])
                        )
                        & (
                            pl.col(community_suit_cols[j])
                            == pl.col(community_suit_cols[k])
                        )
                        & (pl.col(community_suit_cols[i]) > 0)
                    )
                    suited_conds.append(cond)

    community_suited_expr = (
        pl.fold(
            acc=pl.lit(False), function=lambda acc, x: acc | x, exprs=suited_conds
        ).cast(pl.UInt8)
        if suited_conds
        else pl.lit(0).cast(pl.UInt8)
    )

    # connectedness check: any two cards within 2 ranks of each other
    if len(community_rank_cols) >= 2:
        close_pairs = []
        for i in range(len(community_rank_cols)):
            for j in range(i + 1, len(community_rank_cols)):
                cond = (
                    (
                        (
                            pl.col(community_rank_cols[i])
                            - pl.col(community_rank_cols[j])
                        ).abs()
                        <= 2
                    )
                    & (pl.col(community_rank_cols[i]) > 0)
                    & (pl.col(community_rank_cols[j]) > 0)
                )
                close_pairs.append(cond)

        community_connected_expr = (
            pl.fold(
                acc=pl.lit(False), function=lambda acc, x: acc | x, exprs=close_pairs
            ).cast(pl.UInt8)
            if close_pairs
            else pl.lit(0).cast(pl.UInt8)
        )
    else:
        community_connected_expr = pl.lit(0).cast(pl.UInt8)

    # add all community features
    df_community_card_features = df_suit_and_rank.select(
        [
            community_active_expr.alias("community_active"),
            max_expr.alias("max_community_rank"),
            min_expr.alias("min_community_rank"),
            has_ace_expr.cast(pl.UInt8).alias("community_has_ace"),
            has_king_expr.cast(pl.UInt8).alias("community_has_king"),
            has_queen_expr.cast(pl.UInt8).alias("community_has_queen"),
            community_paired_expr.alias("community_paired"),
            community_suited_expr.alias("community_suited"),
            community_connected_expr.alias("community_connected"),
        ]
    )

    # 4. compute hole-community interaction features
    df_combine_hole_community = pl.concat(
        [df_suit_and_rank, df_hole_card_features, df_community_card_features],
        how="horizontal",
    )
    df_interaction_features = df_combine_hole_community.select(
        [
            # overcards
            (
                (pl.col("high_rank") > pl.col("max_community_rank"))
                & (pl.col("community_active") > 0)
            )
            .cast(pl.UInt8)
            .alias("has_overcards"),
            # number of overcards
            (
                (pl.col("C1_rank") > pl.col("max_community_rank")).cast(pl.Int8)
                + (pl.col("C2_rank") > pl.col("max_community_rank")).cast(pl.Int8)
            ).alias("num_overcards"),
            # pair with community
            (
                pl.any_horizontal(
                    [
                        (pl.col("C1_rank") == pl.col("max_community_rank"))
                        & (pl.col("community_active") > 0),
                        (pl.col("C2_rank") == pl.col("max_community_rank"))
                        & (pl.col("community_active") > 0),
                    ]
                )
            )
            .cast(pl.UInt8)
            .alias("pairs_top_community"),
            # potential straight or flush
            ((pl.col("community_connected") == 1) | (pl.col("community_suited") == 1))
            .cast(pl.UInt8)
            .alias("has_draw_potential"),
        ]
    )

    # 5. compute history features (updated to handle checks)
    df_history_features = df_history.select(
        [
            # history length
            pl.col("History").str.len_chars().fill_null(0).alias("history_length"),
            # action counts
            pl.col("History").str.count_matches("r").fill_null(0).alias("num_raises"),
            pl.col("History").str.count_matches("c").fill_null(0).alias("num_calls"),
            pl.col("History").str.count_matches("k").fill_null(0).alias("num_checks"),
            # subhistories
            (pl.col("History") == "crr")
            .fill_null(False)
            .cast(pl.UInt8)
            .alias("hist_crr"),
            (pl.col("History") == "krr")
            .fill_null(False)
            .cast(pl.UInt8)
            .alias("hist_krr"),
            (pl.col("History") == "ckr")
            .fill_null(False)
            .cast(pl.UInt8)
            .alias("hist_ckr"),
            (pl.col("History") == "r").fill_null(False).cast(pl.UInt8).alias("hist_r"),
            (pl.col("History") == "c").fill_null(False).cast(pl.UInt8).alias("hist_c"),
            (pl.col("History") == "rr")
            .fill_null(False)
            .cast(pl.UInt8)
            .alias("hist_rr"),
            (pl.col("History") == "cr")
            .fill_null(False)
            .cast(pl.UInt8)
            .alias("hist_cr"),
            (pl.col("History") == "k").fill_null(False).cast(pl.UInt8).alias("hist_k"),
            (pl.col("History") == "ck")
            .fill_null(False)
            .cast(pl.UInt8)
            .alias("hist_ck"),
            (pl.col("History") == "kr")
            .fill_null(False)
            .cast(pl.UInt8)
            .alias("hist_kr"),
            (pl.col("History") == "kc")
            .fill_null(False)
            .cast(pl.UInt8)
            .alias("hist_kc"),
            (pl.col("History") == "kk")
            .fill_null(False)
            .cast(pl.UInt8)
            .alias("hist_kk"),
            # last action features
            (pl.col("History").str.slice(-1, 1) == "r")
            .fill_null(False)
            .cast(pl.UInt8)
            .alias("last_action_r"),
            (pl.col("History").str.slice(-1, 1) == "c")
            .fill_null(False)
            .cast(pl.UInt8)
            .alias("last_action_c"),
            (pl.col("History").str.slice(-1, 1) == "k")
            .fill_null(False)
            .cast(pl.UInt8)
            .alias("last_action_k"),
            # Handle empty/null history specially
            (pl.col("History").is_null() | (pl.col("History").str.len_chars() == 0))
            .cast(pl.UInt8)
            .alias("last_action_none"),
            # is this a 3-bet+ pot?
            (pl.col("History").str.count_matches("r").fill_null(0) >= 2)
            .cast(pl.UInt8)
            .alias("is_3bet_pot"),
            # started with check?
            (pl.col("History").str.slice(0, 1) == "k")
            .fill_null(False)
            .cast(pl.UInt8)
            .alias("started_with_check"),
            # had check-raise?
            (pl.col("History").str.contains("kr"))
            .fill_null(False)
            .cast(pl.UInt8)
            .alias("had_check_raise"),
        ]
    )

    # aggression ratio (raise / (rase + call + check))
    df_history_features = df_history_features.with_columns(
        [
            (
                pl.col("num_raises")
                / (
                    pl.col("num_raises")
                    + pl.col("num_calls")
                    + pl.col("num_checks")
                    + 1e-10
                )
            ).alias("aggression_ratio")
        ]
    )

    # Finally, combine everything
    result_df = pl.concat(
        [
            df_suit_and_rank,
            df_hole_card_features,
            df_community_card_features,
            df_interaction_features,
            df_history_features,
        ],
        how="horizontal",
    )
    logger.info(f"result_df features: {result_df.columns}")
    logger.info(f"Number of features: {len(result_df.columns)}")
    return result_df


def extract_labels(df: pl.DataFrame) -> pl.DataFrame:
    """
    Assumptions:
    - Labels are real numbers in the interval [0, 1].
    - Sum of labels of each sample should be 1 - since they are probabilities.
    """
    return df[:, -3:]


def main() -> None:
    logger.info("=" * 120)
    logger.info("Initializing CFR Prediction Model")
    logger.info("=" * 120)

    # hard-coded data path
    data_path = pathlib.Path(".") / "strategy_output_small.csv"
    df_input = pl.read_csv(data_path)
    logger.info(f"Running on raw dataset of shape {df_input.shape}")

    # feature engineering
    x_all = extract_features(df_input)
    y_all = extract_labels(df_input)

    # train and test models
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        x_all, y_all, test_size=0.2, random_state=RANDOM_SEED
    )
    models = train_models(x_train, y_train)
    for model in models:
        logger.info("=" * 60)
        logger.info(f"Result Metrics for {model.get_name()}")
        y_pred = model.predict(x_test)
        metrics = evaluate_performance(y_test, y_pred)
        for metric_name, metric_value in metrics.items():
            logger.info(f"{metric_name} = {metric_value}")

    logger.info("=" * 120)
    logger.info("Terminating CFR Prediction Model")
    logger.info("=" * 120)


if __name__ == "__main__":
    main()
