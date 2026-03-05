"""Fee estimation inference — bridge between trained model and live API."""

import numpy as np
import structlog

from stream.fees.features import extract_features

log = structlog.get_logger()

# Fallback feature columns for old pickles that didn't save feature_cols
DEFAULT_FEATURE_COLS = [
    "mempool_count",
    "mempool_vsize",
    "mempool_total_fee",
    "n_projected_blocks",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "tip_height",
    "first_block_size",
    "first_block_fees",
    "first_block_median_fee",
    "avg_block_size",
    "avg_block_fee",
]


def predict_fee(model_data, snapshot: dict) -> int | None:
    """Predict fee rate (sat/vB) from a live mempool snapshot.

    Args:
        model_data: Either a bare LGBMRegressor (old pickle) or
                    {"model": ..., "feature_cols": [...]} (new pickle).
        snapshot: Raw mempool snapshot dict from collect_snapshot().

    Returns:
        Predicted fee rate as int, or None on failure.
    """
    try:
        # Unpack model and feature columns
        if isinstance(model_data, dict):
            model = model_data["model"]
            feature_cols = model_data.get("feature_cols", DEFAULT_FEATURE_COLS)
        else:
            model = model_data
            feature_cols = DEFAULT_FEATURE_COLS

        features = extract_features(snapshot)

        # Build feature vector in training order, excluding rec_ columns
        X = np.array([[features.get(c, 0) for c in feature_cols]])
        pred = model.predict(X)[0]
        return max(1, int(round(pred)))
    except Exception as e:
        log.warning("Fee prediction failed", error=str(e))
        return None
