"""Fee estimation evaluation.

Compare ML predictions against Bitcoin Core's estimator.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


def evaluate_fee_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline_pred: np.ndarray | None = None,
) -> dict:
    """Evaluate fee estimation model."""
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) if (y_true > 0).all() else float("inf")

    result = {
        "mae": float(mae),
        "mape": float(mape),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "median_ae": float(np.median(np.abs(y_true - y_pred))),
    }

    if baseline_pred is not None:
        baseline_mae = mean_absolute_error(y_true, baseline_pred)
        result["baseline_mae"] = float(baseline_mae)
        result["improvement_pct"] = float((baseline_mae - mae) / baseline_mae * 100)

    return result


def evaluate_by_regime(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mempool_sizes: np.ndarray,
) -> dict:
    """Evaluate performance by mempool congestion regime.

    ML typically wins during congestion and weekends.
    """
    p25, p75 = np.percentile(mempool_sizes, [25, 75])

    regimes = {
        "low_congestion": mempool_sizes <= p25,
        "normal": (mempool_sizes > p25) & (mempool_sizes <= p75),
        "high_congestion": mempool_sizes > p75,
    }

    results = {}
    for regime_name, mask in regimes.items():
        if mask.sum() == 0:
            continue
        results[regime_name] = {
            "mae": float(mean_absolute_error(y_true[mask], y_pred[mask])),
            "n_samples": int(mask.sum()),
        }

    return results
