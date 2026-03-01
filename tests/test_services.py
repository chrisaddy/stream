"""Unit tests for scoring service logic."""

import asyncio

from stream.services import (
    _heuristic_risk_score,
    get_risk_threshold,
    score_transaction,
    set_risk_threshold,
)


class TestHeuristicRiskScore:
    def test_returns_float_in_valid_range(self):
        score = _heuristic_risk_score(fee_rate=10.0, vsize=250, fee=2500)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_typical_tx_scores_low(self):
        """A typical transaction should score well below 0.7."""
        score = _heuristic_risk_score(fee_rate=15.0, vsize=250, fee=3750)
        assert score < 0.5

    def test_extreme_fee_rate_scores_higher(self):
        """Very high fee rate should increase score."""
        low = _heuristic_risk_score(fee_rate=10.0, vsize=250, fee=2500)
        high = _heuristic_risk_score(fee_rate=5000.0, vsize=250, fee=1250000)
        assert high > low

    def test_large_tx_scores_higher(self):
        """Larger transactions should get more scrutiny."""
        small = _heuristic_risk_score(fee_rate=10.0, vsize=200, fee=2000)
        large = _heuristic_risk_score(fee_rate=10.0, vsize=50000, fee=500000)
        assert large > small

    def test_deterministic_for_same_inputs(self):
        """Same inputs should produce same score."""
        s1 = _heuristic_risk_score(fee_rate=10.0, vsize=250, fee=2500)
        s2 = _heuristic_risk_score(fee_rate=10.0, vsize=250, fee=2500)
        assert s1 == s2

    def test_different_inputs_different_scores(self):
        """Different inputs should generally produce different scores."""
        s1 = _heuristic_risk_score(fee_rate=10.0, vsize=250, fee=2500)
        s2 = _heuristic_risk_score(fee_rate=20.0, vsize=500, fee=10000)
        assert s1 != s2


class TestRiskThreshold:
    def test_default_threshold(self):
        threshold = get_risk_threshold()
        assert isinstance(threshold, float)
        assert 0.05 <= threshold <= 0.9

    def test_set_and_get_roundtrip(self):
        original = get_risk_threshold()
        try:
            set_risk_threshold(0.5)
            assert get_risk_threshold() == 0.5
        finally:
            set_risk_threshold(original)

    def test_set_clamps_min(self):
        original = get_risk_threshold()
        try:
            set_risk_threshold(0.01)
            assert get_risk_threshold() == 0.05
        finally:
            set_risk_threshold(original)

    def test_set_clamps_max(self):
        original = get_risk_threshold()
        try:
            set_risk_threshold(0.99)
            assert get_risk_threshold() == 0.9
        finally:
            set_risk_threshold(original)


class TestScoreTransaction:
    def test_returns_expected_keys(self):
        result = asyncio.new_event_loop().run_until_complete(
            score_transaction("abc123", 250, 2500)
        )
        assert "risk_score" in result
        assert "risk_label" in result
        assert "model_name" in result
        assert "model_version" in result
        assert "inference_ms" in result
        assert "shap_features" in result
        assert "is_demo" in result

    def test_risk_label_matches_score(self):
        result = asyncio.new_event_loop().run_until_complete(
            score_transaction("test_tx", 250, 2500)
        )
        threshold = get_risk_threshold()
        if result["risk_score"] > threshold:
            assert result["risk_label"] == "HIGH"
        elif result["risk_score"] > threshold * 0.6:
            assert result["risk_label"] == "MED"
        else:
            assert result["risk_label"] == "LOW"
