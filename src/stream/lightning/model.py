"""Lightning Network channel capacity prediction model."""

import pickle
from io import BytesIO

import lightgbm as lgb
import numpy as np

FEATURE_COLS = [
    "degree",
    "channels",
    "avg_neighbor_capacity",
    "max_neighbor_capacity",
    "capacity_per_channel",
    "betweenness",
    "closeness",
    "avg_channel_capacity",
    "total_edge_capacity",
]


def train_capacity_model(
    features: list[dict],
    target_col: str = "capacity",
) -> lgb.LGBMRegressor:
    """Train model to predict optimal node capacity."""
    X = np.array([[f.get(c, 0) for c in FEATURE_COLS] for f in features])
    y = np.array([f.get(target_col, 0) for f in features])

    # Filter out zero-capacity nodes
    mask = y > 0
    X, y = X[mask], y[mask]

    model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        verbose=-1,
        random_state=42,
    )

    if len(X) > 20:
        split = int(len(X) * 0.8)
        model.fit(
            X[:split],
            y[:split],
            eval_set=[(X[split:], y[split:])],
            callbacks=[lgb.early_stopping(10, verbose=False)],
        )
    else:
        model.fit(X, y)

    return model


def predict_capacity(model: lgb.LGBMRegressor, features: dict) -> float:
    """Predict optimal capacity for a node."""
    X = np.array([[features.get(c, 0) for c in FEATURE_COLS]])
    return float(model.predict(X)[0])


def serialize_lightning_model(model: lgb.LGBMRegressor) -> bytes:
    buf = BytesIO()
    pickle.dump(model, buf)
    return buf.getvalue()


def deserialize_lightning_model(data: bytes) -> lgb.LGBMRegressor:
    buf = BytesIO(data)
    return pickle.load(buf)
