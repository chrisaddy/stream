"""Evaluation metrics for illicit detection.

Key: Use PR-AUC (not ROC-AUC) because of severe class imbalance (~10:1).
ROC-AUC overestimates performance when negatives dominate.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


def compute_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Precision-Recall AUC (average precision)."""
    return float(average_precision_score(y_true, y_prob))


def compute_pr_curve(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Full PR curve data for plotting."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
    }


def cost_sensitive_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fn_cost: float = 50000.0,  # Regulatory fine per missed illicit tx
    fp_cost: float = 50.0,  # Cost of manual review per false alarm
) -> dict:
    """Select optimal threshold based on business costs.

    FN >> FP in regulatory context: missing an illicit transaction
    can mean millions in fines, while a false alarm just means a manual review.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    best_threshold = 0.5
    min_cost = float("inf")

    for i, t in enumerate(thresholds):
        preds = (y_prob >= t).astype(int)
        fn = ((y_true == 1) & (preds == 0)).sum()
        fp = ((y_true == 0) & (preds == 1)).sum()
        cost = fn * fn_cost + fp * fp_cost

        if cost < min_cost:
            min_cost = cost
            best_threshold = float(t)

    preds = (y_prob >= best_threshold).astype(int)
    return {
        "threshold": best_threshold,
        "total_cost": min_cost,
        "fn_count": int(((y_true == 1) & (preds == 0)).sum()),
        "fp_count": int(((y_true == 0) & (preds == 1)).sum()),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
    }


def per_timestep_evaluation(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    timesteps: np.ndarray,
    threshold: float = 0.5,
) -> list[dict]:
    """Evaluate model per timestep (temporal stability check)."""
    results = []
    for ts in sorted(np.unique(timesteps)):
        mask = timesteps == ts
        if mask.sum() == 0 or y_true[mask].sum() == 0:
            continue

        y_t = y_true[mask]
        p_t = y_prob[mask]
        preds = (p_t >= threshold).astype(int)

        results.append(
            {
                "timestep": int(ts),
                "n_samples": int(mask.sum()),
                "n_illicit": int(y_t.sum()),
                "precision": float(precision_score(y_t, preds, zero_division=0)),
                "recall": float(recall_score(y_t, preds, zero_division=0)),
                "f1": float(f1_score(y_t, preds, zero_division=0)),
                "pr_auc": float(average_precision_score(y_t, p_t)) if y_t.sum() > 0 else 0.0,
            }
        )

    return results


def full_evaluation(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    timesteps: np.ndarray | None = None,
    model_name: str = "model",
) -> dict:
    """Complete evaluation suite."""
    pr_auc = compute_pr_auc(y_true, y_prob)
    pr_curve = compute_pr_curve(y_true, y_prob)
    cost_analysis = cost_sensitive_threshold(y_true, y_prob)

    result = {
        "model_name": model_name,
        "pr_auc": pr_auc,
        "pr_curve": pr_curve,
        "cost_analysis": cost_analysis,
        "n_samples": len(y_true),
        "n_illicit": int(y_true.sum()),
        "class_ratio": f"{(y_true == 0).sum()}:{(y_true == 1).sum()}",
    }

    if timesteps is not None:
        result["temporal"] = per_timestep_evaluation(
            y_true, y_prob, timesteps, cost_analysis["threshold"]
        )

    return result
