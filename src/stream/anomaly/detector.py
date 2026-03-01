"""Unsupervised anomaly detection using River Half-Space Trees."""

import structlog

log = structlog.get_logger()

_hst_model = None
_hst_n_samples = 0
_hst_window_size = 250


def _init_hst():
    """Create a Half-Space Trees anomaly detector."""
    from river.anomaly import HalfSpaceTrees

    global _hst_model
    _hst_model = HalfSpaceTrees(n_trees=25, height=8, window_size=_hst_window_size)


def score_anomaly(fee_rate: float, vsize: float, fee: float) -> float:
    """Test-then-train anomaly scoring. Returns anomaly score in [0, 1].

    Returns 0.0 during the grace period (first window_size samples).
    """
    global _hst_model, _hst_n_samples

    if _hst_model is None:
        _init_hst()

    x = {"fee_rate": fee_rate, "vsize": vsize, "fee": fee}

    # Score before learning (test-then-train)
    if _hst_n_samples >= _hst_window_size:
        score = _hst_model.score_one(x)
    else:
        score = 0.0

    _hst_model.learn_one(x)
    _hst_n_samples += 1

    return float(score)


def get_anomaly_status() -> dict:
    """Return anomaly detector status."""
    is_calibrated = _hst_n_samples >= _hst_window_size
    return {
        "enabled": True,
        "n_samples": _hst_n_samples,
        "is_calibrated": is_calibrated,
        "grace_remaining": max(0, _hst_window_size - _hst_n_samples),
    }
