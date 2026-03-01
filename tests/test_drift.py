"""Unit tests for drift monitor."""

import numpy as np
import pytest

from stream.drift.monitor import (
    _score_buffer,
    compute_psi,
    record_score,
    reset_baseline,
)


@pytest.fixture(autouse=True)
def clean_drift_state():
    """Reset drift monitor state before each test."""
    _score_buffer.clear()
    reset_baseline()
    yield
    _score_buffer.clear()
    reset_baseline()


class TestComputePSI:
    def test_returns_none_without_baseline(self):
        assert compute_psi() is None

    def test_returns_none_with_few_scores(self):
        for i in range(10):
            record_score(0.5)
        assert compute_psi() is None

    def test_identical_distributions_near_zero(self):
        """PSI of identical distributions should be approximately 0."""
        # Build baseline from uniform-ish scores
        rng = np.random.RandomState(42)
        scores = rng.uniform(0.1, 0.5, 200)
        for s in scores:
            record_score(float(s))

        # Now add more scores from same distribution
        more_scores = rng.uniform(0.1, 0.5, 200)
        for s in more_scores:
            record_score(float(s))

        psi = compute_psi()
        assert psi is not None
        assert psi < 0.1  # GREEN threshold

    def test_shifted_distribution_high_psi(self):
        """PSI of very different distributions should be high."""
        rng = np.random.RandomState(42)

        # Baseline: low scores
        for s in rng.uniform(0.0, 0.2, 200):
            record_score(float(s))

        # Clear buffer but keep baseline
        _score_buffer.clear()

        # Current: high scores (shifted)
        for s in rng.uniform(0.7, 1.0, 200):
            record_score(float(s))

        psi = compute_psi()
        assert psi is not None
        assert psi > 0.25  # RED threshold


class TestRecordScore:
    def test_populates_buffer(self):
        assert len(_score_buffer) == 0
        record_score(0.5)
        assert len(_score_buffer) == 1

    def test_buffer_maxlen(self):
        for i in range(1500):
            record_score(float(i % 100) / 100)
        assert len(_score_buffer) == 1000  # maxlen
