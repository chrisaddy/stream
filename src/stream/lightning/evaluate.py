"""Lightning network model evaluation."""

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


def evaluate_capacity_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Evaluate capacity prediction model and return MAE, R², median AE."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "median_ae": float(np.median(np.abs(y_true - y_pred))),
        "n_samples": len(y_true),
    }
