"""Drift monitor — tracks prediction distribution shift via PSI + ADWIN.

Uses an in-memory ring buffer of recent scores and compares against
a baseline distribution to detect when the model's scoring behavior
has shifted significantly. ADWIN provides adaptive windowing for
change-point detection.
"""

import time as _time
from collections import deque

import numpy as np
import structlog

log = structlog.get_logger()

# Ring buffer of recent scores
_score_buffer: deque[float] = deque(maxlen=1000)

# Baseline distribution — initialized from first N scores
_baseline_hist: np.ndarray | None = None
_baseline_count: int = 0
_BASELINE_SIZE = 200  # Collect this many before computing baseline

# PSI bin edges (10 equal-width bins from 0 to 1)
_BIN_EDGES = np.linspace(0.0, 1.0, 11)

# ADWIN state
_adwin = None
_adwin_drift_detected = False
_adwin_n_samples = 0
_adwin_last_drift_at: float | None = None
_adwin_drift_history: list[dict] = []


def _adwin_update(score: float):
    """Update ADWIN with a new score, detect drift."""
    global _adwin, _adwin_drift_detected, _adwin_n_samples, _adwin_last_drift_at

    if _adwin is None:
        from river.drift import ADWIN

        _adwin = ADWIN(delta=0.002)

    _adwin.update(score)
    _adwin_n_samples += 1

    if _adwin.drift_detected:
        _adwin_drift_detected = True
        _adwin_last_drift_at = _time.time()
        _adwin_drift_history.append(
            {
                "sample": _adwin_n_samples,
                "timestamp": _adwin_last_drift_at,
            }
        )
        log.warning("ADWIN drift detected", sample=_adwin_n_samples)
    else:
        _adwin_drift_detected = False


def record_score(score: float):
    """Record a new prediction score into the drift buffer."""
    _score_buffer.append(score)

    global _baseline_hist, _baseline_count
    if _baseline_hist is None:
        _baseline_count += 1
        if _baseline_count >= _BASELINE_SIZE:
            _compute_baseline()

    # Also feed ADWIN
    try:
        _adwin_update(score)
    except Exception:
        pass


def _compute_baseline():
    """Snapshot the current buffer as the baseline distribution."""
    global _baseline_hist
    scores = np.array(list(_score_buffer))
    hist, _ = np.histogram(scores, bins=_BIN_EDGES)
    # Normalize to proportions, add small epsilon to avoid division by zero
    _baseline_hist = (hist + 1e-6) / (hist.sum() + 1e-6 * len(hist))
    log.info("Drift baseline computed", n_scores=len(scores), bins=_baseline_hist.tolist())


def reset_baseline():
    """Force baseline recomputation (e.g. after retraining)."""
    global _baseline_hist, _baseline_count
    _baseline_hist = None
    _baseline_count = 0
    log.info("Drift baseline reset — will recompute from next scores")


def compute_psi() -> float | None:
    """Compute Population Stability Index between baseline and current distribution.

    PSI < 0.1  → GREEN (no significant shift)
    PSI 0.1-0.25 → YELLOW (moderate shift)
    PSI > 0.25 → RED (significant drift)

    Returns None if baseline not yet established.
    """
    if _baseline_hist is None:
        return None

    if len(_score_buffer) < 50:
        return None

    current_scores = np.array(list(_score_buffer))
    current_hist, _ = np.histogram(current_scores, bins=_BIN_EDGES)
    current_dist = (current_hist + 1e-6) / (current_hist.sum() + 1e-6 * len(current_hist))

    # PSI = sum((current - baseline) * ln(current / baseline))
    psi = float(np.sum((current_dist - _baseline_hist) * np.log(current_dist / _baseline_hist)))
    return psi


def get_adwin_status() -> dict:
    """Return ADWIN detector status."""
    estimation = None
    width = None
    if _adwin is not None:
        try:
            estimation = float(_adwin.estimation)
            width = int(_adwin.width)
        except Exception:
            pass

    return {
        "enabled": True,
        "drift_detected": _adwin_drift_detected,
        "n_samples": _adwin_n_samples,
        "last_drift_at": _adwin_last_drift_at,
        "drift_count": len(_adwin_drift_history),
        "estimation": estimation,
        "width": width,
    }


def reset_adwin():
    """Re-initialize ADWIN (called after adaptation)."""
    global _adwin, _adwin_drift_detected, _adwin_n_samples, _adwin_last_drift_at
    from river.drift import ADWIN

    _adwin = ADWIN(delta=0.002)
    _adwin_drift_detected = False
    _adwin_n_samples = 0
    _adwin_last_drift_at = None
    log.info("ADWIN reset")


def get_status() -> dict:
    """Get current drift status with PSI, ADWIN, and distribution stats."""
    psi = compute_psi()
    n_scores = len(_score_buffer)
    has_baseline = _baseline_hist is not None

    if psi is None:
        status = "CALIBRATING"
        color = "dim"
        remaining = _BASELINE_SIZE - _baseline_count if not has_baseline else max(0, 50 - n_scores)
    elif psi < 0.1:
        status = "GREEN"
        color = "green"
        remaining = 0
    elif psi < 0.25:
        status = "YELLOW"
        color = "yellow"
        remaining = 0
    else:
        status = "RED"
        color = "red"
        remaining = 0

    # Score distribution stats
    if n_scores > 0:
        scores = np.array(list(_score_buffer))
        stats = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "median": float(np.median(scores)),
            "p95": float(np.percentile(scores, 95)),
            "n_scores": n_scores,
        }
    else:
        stats = {"mean": 0, "std": 0, "median": 0, "p95": 0, "n_scores": 0}

    return {
        "status": status,
        "color": color,
        "psi": round(psi, 4) if psi is not None else None,
        "has_baseline": has_baseline,
        "remaining": remaining,
        "stats": stats,
        "adwin": get_adwin_status(),
    }


def get_distribution() -> dict:
    """Get score histogram data for charting."""
    if len(_score_buffer) < 10:
        return {"bins": [], "current": [], "baseline": []}

    bin_centers = [
        round(float((_BIN_EDGES[i] + _BIN_EDGES[i + 1]) / 2), 2) for i in range(len(_BIN_EDGES) - 1)
    ]

    current_scores = np.array(list(_score_buffer))
    current_hist, _ = np.histogram(current_scores, bins=_BIN_EDGES)
    current_pct = (current_hist / max(current_hist.sum(), 1) * 100).tolist()

    baseline_pct = []
    if _baseline_hist is not None:
        baseline_pct = (_baseline_hist * 100).tolist()

    return {
        "bins": bin_centers,
        "current": [round(v, 1) for v in current_pct],
        "baseline": [round(v, 1) for v in baseline_pct],
    }
