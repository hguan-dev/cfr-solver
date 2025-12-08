import logging
import pathlib

import polars as pl
import sklearn
from constants import RANDOM_SEED
from utils import evaluate_performance

from model import train_models

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
    """
    return df.drop("History")[:, :-3]


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
    data_path = pathlib.Path(".") / "strategy_output.csv"
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
