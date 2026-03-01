"""LightGBM fee estimation model.

Predicts fee rate (sat/vB) for different confirmation targets.
Uses MAE objective — robust to fee spikes.
"""

import pickle
from io import BytesIO

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import TimeSeriesSplit


def train_fee_model(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    params: dict | None = None,
) -> lgb.LGBMRegressor:
    """Train LightGBM for fee estimation with time-series split."""
    if params is None:
        params = {
            "objective": "regression_l1",  # MAE, robust to spikes
            "n_estimators": 300,
            "max_depth": 8,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "verbose": -1,
        }

    model = lgb.LGBMRegressor(**params)

    # Time-series split for validation (no shuffle!)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = list(tscv.split(X))
    train_idx, val_idx = splits[-1]

    model.fit(
        X[train_idx], y[train_idx],
        eval_set=[(X[val_idx], y[val_idx])],
        callbacks=[lgb.early_stopping(20, verbose=False)],
    )

    return model


def train_multi_target_models(
    X: np.ndarray,
    targets: dict[str, np.ndarray],
) -> dict[str, lgb.LGBMRegressor]:
    """Train separate models for each confirmation target."""
    models = {}
    for target_name, y in targets.items():
        models[target_name] = train_fee_model(X, y)
    return models


def serialize_fee_model(model: lgb.LGBMRegressor, feature_cols: list[str] | None = None) -> bytes:
    buf = BytesIO()
    pickle.dump({"model": model, "feature_cols": feature_cols}, buf)
    return buf.getvalue()


def deserialize_fee_model(data: bytes):
    """Deserialize fee model. Returns dict with model+feature_cols (new) or bare model (old)."""
    buf = BytesIO(data)
    obj = pickle.load(buf)
    # Backward compat: old pickles are bare LGBMRegressor
    if isinstance(obj, dict):
        return obj
    return {"model": obj, "feature_cols": None}
