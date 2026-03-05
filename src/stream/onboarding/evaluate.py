"""Onboarding model evaluation: calibration, fairness, per-tier analysis."""

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
)


def evaluate_risk_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "model",
) -> dict:
    """Evaluate risk scoring model."""
    return {
        "model_name": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "n_samples": len(y_true),
    }


def calibration_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Check probability calibration for each class."""
    n_classes = y_prob.shape[1]
    results = {}

    for cls in range(n_classes):
        y_binary = (y_true == cls).astype(int)
        prob_true, prob_pred = calibration_curve(y_binary, y_prob[:, cls], n_bins=n_bins)
        results[f"class_{cls}"] = {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
        }

    return results


def per_tier_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tier_names: list[str] = None,
) -> list[dict]:
    """Analysis per risk tier."""
    if tier_names is None:
        tier_names = ["low_risk", "medium_risk", "high_risk", "blocked"]

    results = []
    for i, name in enumerate(tier_names):
        mask = y_true == i
        pred_mask = y_pred == i
        results.append(
            {
                "tier": name,
                "actual_count": int(mask.sum()),
                "predicted_count": int(pred_mask.sum()),
                "true_positive_rate": float((mask & pred_mask).sum() / max(mask.sum(), 1)),
            }
        )
    return results
