"""XGBoost model for illicit transaction detection.

Production workhorse: fast inference, SHAP-explainable, handles tabular data well.
"""

import pickle
from io import BytesIO

import numpy as np
import shap
import xgboost as xgb


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scale_pos_weight: float = 10.0,
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    early_stopping_rounds: int = 20,
) -> xgb.XGBClassifier:
    """Train XGBoost with class-weighted loss and early stopping."""
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        eval_metric=["aucpr", "logloss"],
        early_stopping_rounds=early_stopping_rounds,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    return model


def get_shap_explainer(model: xgb.XGBClassifier) -> shap.TreeExplainer:
    """Create SHAP TreeExplainer for feature attribution."""
    return shap.TreeExplainer(model)


def explain_prediction(
    explainer: shap.TreeExplainer,
    features: np.ndarray,
    feature_names: list[str] | None = None,
    top_k: int = 10,
) -> dict:
    """Get SHAP explanation for a single prediction.

    Returns dict with top_k contributing features and their SHAP values.
    """
    if features.ndim == 1:
        features = features.reshape(1, -1)

    shap_values = explainer.shap_values(features)

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(features.shape[1])]

    # Get top contributing features by absolute SHAP value
    abs_shap = np.abs(shap_values[0])
    top_indices = np.argsort(abs_shap)[-top_k:][::-1]

    contributions = []
    for idx in top_indices:
        contributions.append({
            "feature": feature_names[idx],
            "feature_index": int(idx),
            "shap_value": float(shap_values[0][idx]),
            "feature_value": float(features[0][idx]),
        })

    return {
        "base_value": float(explainer.expected_value),
        "contributions": contributions,
    }


def get_global_feature_importance(
    explainer: shap.TreeExplainer,
    X: np.ndarray,
    feature_names: list[str] | None = None,
    max_samples: int = 1000,
) -> list[dict]:
    """Global feature importance via mean |SHAP|."""
    if X.shape[0] > max_samples:
        idx = np.random.choice(X.shape[0], max_samples, replace=False)
        X = X[idx]

    shap_values = explainer.shap_values(X)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    importance = sorted(
        [{"feature": name, "importance": float(val)} for name, val in zip(feature_names, mean_abs_shap)],
        key=lambda x: x["importance"],
        reverse=True,
    )
    return importance


def serialize_model(model: xgb.XGBClassifier) -> bytes:
    """Serialize model to bytes for R2 storage."""
    buf = BytesIO()
    pickle.dump(model, buf)
    return buf.getvalue()


def deserialize_model(data: bytes) -> xgb.XGBClassifier:
    """Deserialize model from bytes."""
    buf = BytesIO(data)
    return pickle.load(buf)
