"""Feature extraction from mempool snapshots for fee estimation."""

from datetime import datetime

import numpy as np


def extract_features(snapshot: dict) -> dict:
    """Extract features from a single mempool snapshot.

    Features capture what Bitcoin Core's estimator misses:
    the current mempool state.
    """
    mempool = snapshot.get("mempool", {})
    fees = snapshot.get("recommended_fees", {})
    blocks = snapshot.get("projected_blocks", [])
    timestamp = snapshot.get("timestamp", "")

    # Parse timestamp
    if isinstance(timestamp, str) and timestamp:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    else:
        dt = datetime.utcnow()

    features = {
        # Mempool size
        "mempool_count": mempool.get("count", 0),
        "mempool_vsize": mempool.get("vsize", 0),
        "mempool_total_fee": mempool.get("total_fee", 0),

        # Fee distribution from projected blocks
        "n_projected_blocks": len(blocks),

        # Current recommendations (these are targets, not features in production)
        "rec_fastest_fee": fees.get("fastestFee", 0),
        "rec_half_hour_fee": fees.get("halfHourFee", 0),
        "rec_hour_fee": fees.get("hourFee", 0),
        "rec_economy_fee": fees.get("economyFee", 0),
        "rec_minimum_fee": fees.get("minimumFee", 0),

        # Time features
        "hour_of_day": dt.hour,
        "day_of_week": dt.weekday(),
        "is_weekend": int(dt.weekday() >= 5),

        # Block tip
        "tip_height": snapshot.get("tip_height", 0),
    }

    # Fee distribution from projected blocks
    if blocks:
        block_sizes = [b.get("blockVSize", 0) for b in blocks]
        block_fees = [b.get("totalFees", 0) for b in blocks]
        median_fees = [b.get("medianFee", 0) for b in blocks]

        features["first_block_size"] = block_sizes[0] if block_sizes else 0
        features["first_block_fees"] = block_fees[0] if block_fees else 0
        features["first_block_median_fee"] = median_fees[0] if median_fees else 0
        features["avg_block_size"] = np.mean(block_sizes) if block_sizes else 0
        features["avg_block_fee"] = np.mean(block_fees) if block_fees else 0
    else:
        features.update({
            "first_block_size": 0, "first_block_fees": 0,
            "first_block_median_fee": 0, "avg_block_size": 0, "avg_block_fee": 0,
        })

    return features


def build_feature_matrix(snapshots: list[dict]):
    """Build feature matrix from list of snapshots."""
    import pandas as pd

    rows = [extract_features(s) for s in snapshots]
    return pd.DataFrame(rows)


def build_targets(snapshots: list[dict]):
    """Extract fee rate targets from snapshots.

    Targets: recommended fee rates for different confirmation targets.
    """
    import pandas as pd

    targets = []
    for s in snapshots:
        fees = s.get("recommended_fees", {})
        targets.append({
            "target_1_block": fees.get("fastestFee", 0),
            "target_3_blocks": fees.get("halfHourFee", 0),
            "target_6_blocks": fees.get("hourFee", 0),
            "target_12_blocks": fees.get("economyFee", 0),
            "target_24_blocks": fees.get("minimumFee", 0),
        })
    return pd.DataFrame(targets)
