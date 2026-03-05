"""Onboarding risk scoring models.

Logistic regression baseline + calibrated XGBoost.
The screening interview asks about logistic regression — show mastery by USING it.
"""

import pickle
from io import BytesIO

import numpy as np
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def train_logistic_baseline(
    X: np.ndarray, y: np.ndarray
) -> tuple[LogisticRegression, StandardScaler]:
    """Logistic regression baseline.

    Why logistic regression first:
    - Interpretable coefficients (compliance can understand it)
    - Well-calibrated probabilities out of the box
    - Fast inference
    - If it works well enough, you don't need more complexity
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_scaled, y)

    return model, scaler


def train_calibrated_xgboost(
    X: np.ndarray, y: np.ndarray
) -> tuple[CalibratedClassifierCV, StandardScaler]:
    """Calibrated XGBoost for meaningful probability scores.

    Platt scaling ensures predicted probabilities match actual risk rates.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    base_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric="mlogloss",
    )

    # Platt scaling via CalibratedClassifierCV
    calibrated = CalibratedClassifierCV(base_model, cv=5, method="sigmoid")
    calibrated.fit(X_scaled, y)

    return calibrated, scaler


def serialize_onboarding_model(model, scaler) -> bytes:
    buf = BytesIO()
    pickle.dump({"model": model, "scaler": scaler}, buf)
    return buf.getvalue()


def deserialize_onboarding_model(data: bytes):
    buf = BytesIO(data)
    d = pickle.load(buf)
    return d["model"], d["scaler"]
