"""Test XGBoost model."""

import numpy as np
import pytest

from stream.illicit.xgboost_model import (
    deserialize_model,
    serialize_model,
    train_xgboost,
)


@pytest.fixture
def trained_model():
    """Train a small XGBoost model for testing."""
    np.random.seed(42)
    X_train = np.random.randn(200, 10)
    y_train = np.random.choice([0, 1], 200, p=[0.9, 0.1])
    X_test = np.random.randn(50, 10)
    y_test = np.random.choice([0, 1], 50, p=[0.9, 0.1])

    model = train_xgboost(
        X_train, y_train, X_test, y_test, n_estimators=10, early_stopping_rounds=5
    )
    return model, X_test


def test_predict_proba_range(trained_model):
    """Probabilities should be in [0, 1]."""
    model, X_test = trained_model
    probs = model.predict_proba(X_test)[:, 1]

    assert np.all(probs >= 0), "Probabilities below 0"
    assert np.all(probs <= 1), "Probabilities above 1"


def test_predict_proba_shape(trained_model):
    """Should return probabilities for each sample."""
    model, X_test = trained_model
    probs = model.predict_proba(X_test)

    assert probs.shape == (X_test.shape[0], 2)


def test_serialize_roundtrip(trained_model):
    """Serialization should preserve model."""
    model, X_test = trained_model
    original_probs = model.predict_proba(X_test)

    data = serialize_model(model)
    loaded = deserialize_model(data)
    loaded_probs = loaded.predict_proba(X_test)

    np.testing.assert_array_almost_equal(original_probs, loaded_probs)
