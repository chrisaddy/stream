"""Unit tests for Pydantic validation schemas."""

import pytest
from pydantic import ValidationError

from stream.app.schemas import (
    IllicitScoreRequest,
    LightningEvaluateRequest,
    LiveScoreRequest,
    ThresholdRequest,
)


class TestIllicitScoreRequest:
    def test_valid_166_features(self):
        features = ",".join(["0.5"] * 166)
        req = IllicitScoreRequest(features=features)
        assert req.features == features

    def test_rejects_wrong_count(self):
        features = ",".join(["0.5"] * 5)
        with pytest.raises(ValidationError, match="166"):
            IllicitScoreRequest(features=features)

    def test_rejects_non_numeric(self):
        features = ",".join(["abc"] * 166)
        with pytest.raises(ValidationError, match="not a valid number"):
            IllicitScoreRequest(features=features)

    def test_rejects_empty_string(self):
        with pytest.raises(ValidationError, match="166"):
            IllicitScoreRequest(features="")


class TestLiveScoreRequest:
    def test_valid_input(self):
        req = LiveScoreRequest(vsize=250, fee=2500)
        assert req.vsize == 250
        assert req.fee == 2500

    def test_rejects_zero_vsize(self):
        with pytest.raises(ValidationError, match="positive"):
            LiveScoreRequest(vsize=0, fee=100)

    def test_rejects_negative_vsize(self):
        with pytest.raises(ValidationError, match="positive"):
            LiveScoreRequest(vsize=-1, fee=100)

    def test_rejects_negative_fee(self):
        with pytest.raises(ValidationError, match="non-negative"):
            LiveScoreRequest(vsize=250, fee=-1)

    def test_accepts_zero_fee(self):
        req = LiveScoreRequest(vsize=250, fee=0)
        assert req.fee == 0


class TestLightningEvaluateRequest:
    def test_valid_pubkey(self):
        pubkey = "02" + "a1" * 32
        req = LightningEvaluateRequest(pubkey=pubkey)
        assert req.pubkey == pubkey

    def test_rejects_short_hex(self):
        with pytest.raises(ValidationError, match="66-character"):
            LightningEvaluateRequest(pubkey="abcdef")

    def test_rejects_non_hex(self):
        with pytest.raises(ValidationError, match="66-character"):
            LightningEvaluateRequest(pubkey="zz" * 33)

    def test_rejects_empty(self):
        with pytest.raises(ValidationError, match="required"):
            LightningEvaluateRequest(pubkey="")


class TestThresholdRequest:
    def test_valid_threshold(self):
        req = ThresholdRequest(value=0.5)
        assert req.value == 0.5

    def test_valid_min_boundary(self):
        req = ThresholdRequest(value=0.05)
        assert req.value == 0.05

    def test_valid_max_boundary(self):
        req = ThresholdRequest(value=0.9)
        assert req.value == 0.9

    def test_rejects_below_min(self):
        with pytest.raises(ValidationError, match="0.05"):
            ThresholdRequest(value=0.01)

    def test_rejects_above_max(self):
        with pytest.raises(ValidationError, match="0.9"):
            ThresholdRequest(value=0.95)
