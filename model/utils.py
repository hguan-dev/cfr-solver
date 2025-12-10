import numpy as np
import polars as pl
from constants import Actions


def kl_divergence(
    y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12
) -> np.ndarray:
    # clip to avoid zeros
    y_true_clipped = np.clip(y_true, eps, 1)
    y_pred_clipped = np.clip(y_pred, eps, 1)
    return np.sum(
        y_true_clipped * (np.log(y_true_clipped) - np.log(y_pred_clipped)), axis=1
    )


def total_variation_distance(
    y_true: np.typing.NDArray, y_pred: np.typing.NDArray
) -> np.typing.NDArray:
    return 0.5 * np.sum(np.abs(y_true - y_pred), axis=1)


def evaluate_performance(
    y_true: pl.DataFrame | np.typing.NDArray, y_pred: pl.DataFrame | np.typing.NDArray
) -> dict[str, float]:
    metrics = {}

    # conversion
    y_true_np = y_true.to_numpy() if isinstance(y_true, pl.DataFrame) else y_true
    y_pred_np = y_pred.to_numpy() if isinstance(y_pred, pl.DataFrame) else y_pred

    # distribution similarity
    metrics["avg_kl_div"] = np.mean(kl_divergence(y_true_np, y_pred_np))
    metrics["avg_tvd"] = np.mean(total_variation_distance(y_true_np, y_pred_np))

    # how off can the model be at worst
    metrics["max_prob_error"] = np.max(np.abs(y_true_np - y_pred_np))

    # per-action MAE
    for i, action in enumerate(Actions.get_all_actions()):
        action_mae = np.mean(np.abs(y_true_np[:, i] - y_pred_np[:, i]))
        metrics[f"{action}_mae"] = action_mae

    return metrics
