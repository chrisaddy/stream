"""Test temporal split and preprocessing."""

import numpy as np
import pandas as pd
import pytest

from stream.illicit.preprocess import (
    get_class_weight,
    temporal_split,
)


@pytest.fixture
def mock_dataset():
    """Create mock Elliptic-like dataset."""
    n = 1000
    features = pd.DataFrame(np.random.randn(n, 166), columns=[f"feature_{i}" for i in range(166)])
    features["feature_0"] = np.random.choice(range(1, 50), n)  # timestep

    labels = pd.Series(np.random.choice([0, 1, 2], n, p=[0.7, 0.1, 0.2]))
    timesteps = features["feature_0"].astype(int)

    return features, labels, timesteps


def test_temporal_split_no_leakage(mock_dataset):
    """Train timesteps should not appear in test set."""
    features, labels, timesteps = mock_dataset
    X_train, y_train, X_test, y_test = temporal_split(features, labels, timesteps)

    train_ts = set(timesteps[labels != 2][timesteps <= 34].unique())
    test_ts = set(timesteps[labels != 2][timesteps > 34].unique())

    assert len(train_ts & test_ts) == 0, "Temporal leakage: train and test timesteps overlap"


def test_temporal_split_excludes_unknown(mock_dataset):
    """Unknown labels (y=2) should be filtered out."""
    features, labels, timesteps = mock_dataset
    X_train, y_train, X_test, y_test = temporal_split(features, labels, timesteps)

    assert 2 not in y_train, "Unknown labels in training set"
    assert 2 not in y_test, "Unknown labels in test set"


def test_temporal_split_shapes(mock_dataset):
    """X and y should have consistent shapes."""
    features, labels, timesteps = mock_dataset
    X_train, y_train, X_test, y_test = temporal_split(features, labels, timesteps)

    assert X_train.shape[0] == len(y_train)
    assert X_test.shape[0] == len(y_test)
    assert X_train.shape[1] == 166


def test_class_weight_positive():
    """Class weight should be positive."""
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    weight = get_class_weight(y)
    assert weight > 0
    assert weight == 9.0  # 9 licit / 1 illicit


def test_class_weight_handles_no_illicit():
    """Should not crash with no illicit samples."""
    y = np.array([0, 0, 0, 0])
    weight = get_class_weight(y)
    assert weight > 0
