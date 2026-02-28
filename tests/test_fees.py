"""Test fee estimation."""

import numpy as np
import pytest

from stream.fees.features import extract_features, build_feature_matrix


@pytest.fixture
def mock_snapshot():
    return {
        "timestamp": "2024-01-15T12:00:00Z",
        "mempool": {"count": 50000, "vsize": 100000000, "total_fee": 5000000},
        "recommended_fees": {
            "fastestFee": 25,
            "halfHourFee": 20,
            "hourFee": 15,
            "economyFee": 10,
            "minimumFee": 5,
        },
        "projected_blocks": [
            {"blockVSize": 1000000, "totalFees": 500000, "medianFee": 20},
            {"blockVSize": 900000, "totalFees": 400000, "medianFee": 15},
        ],
        "tip_height": 800000,
    }


def test_extract_features(mock_snapshot):
    features = extract_features(mock_snapshot)

    assert features["mempool_count"] == 50000
    assert features["mempool_vsize"] == 100000000
    assert features["rec_fastest_fee"] == 25
    assert features["n_projected_blocks"] == 2
    assert features["first_block_size"] == 1000000
    assert 0 <= features["hour_of_day"] <= 23
    assert 0 <= features["day_of_week"] <= 6


def test_build_feature_matrix(mock_snapshot):
    snapshots = [mock_snapshot, mock_snapshot]
    df = build_feature_matrix(snapshots)

    assert len(df) == 2
    assert "mempool_count" in df.columns
    assert "rec_fastest_fee" in df.columns


def test_extract_features_empty_blocks():
    snapshot = {
        "timestamp": "2024-01-15T12:00:00Z",
        "mempool": {"count": 0, "vsize": 0, "total_fee": 0},
        "recommended_fees": {"fastestFee": 1, "halfHourFee": 1, "hourFee": 1, "economyFee": 1, "minimumFee": 1},
        "projected_blocks": [],
        "tip_height": 0,
    }
    features = extract_features(snapshot)
    assert features["n_projected_blocks"] == 0
    assert features["first_block_size"] == 0
