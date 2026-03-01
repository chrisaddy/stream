"""Self-healing adaptation loop — drift detection triggers full model cascade reset.

State machine: STABLE → DRIFT_DETECTED → ADAPTING → STABILIZING → RECOVERED → STABLE
"""

import time
from collections import deque

import structlog

log = structlog.get_logger()

# State machine
_state = "STABLE"
_drift_signals = 0
_consecutive_threshold = 3  # Must see N consecutive drift signals before acting
_cooldown_seconds = 60.0
_last_adaptation_at: float | None = None
_post_adaptation_labels = 0
_stabilization_threshold = 10  # Labels needed after adaptation to stabilize
_adaptation_log: deque[dict] = deque(maxlen=50)

# Internal ADWIN for prediction errors
_error_adwin = None


def _init_error_adwin():
    global _error_adwin
    from river.drift import ADWIN
    _error_adwin = ADWIN(delta=0.002)


def record_prediction_error(error: float) -> dict | None:
    """Feed ADWIN with prediction error. Returns adaptation event if triggered."""
    global _state, _drift_signals, _error_adwin

    if _error_adwin is None:
        _init_error_adwin()

    _error_adwin.update(error)

    if _error_adwin.drift_detected:
        _drift_signals += 1
        _log_event("DRIFT_SIGNAL", f"Signal {_drift_signals}/{_consecutive_threshold}")

        if _drift_signals >= _consecutive_threshold:
            if _state == "STABLE" or _state == "RECOVERED":
                # Check cooldown
                if _last_adaptation_at and (time.time() - _last_adaptation_at) < _cooldown_seconds:
                    _log_event("COOLDOWN", "Adaptation blocked by cooldown")
                    return None

                _state = "DRIFT_DETECTED"
                _log_event("DRIFT_DETECTED", "Consecutive threshold reached")
                return _trigger_adaptation()
    else:
        # Reset drift signal counter on non-drift
        if _drift_signals > 0:
            _drift_signals = max(0, _drift_signals - 1)

    return None


def _trigger_adaptation() -> dict:
    """Execute full adaptation cascade: reset ADWIN, PSI, online metrics, model race."""
    global _state, _drift_signals, _last_adaptation_at, _post_adaptation_labels, _error_adwin

    _state = "ADAPTING"
    _log_event("ADAPTING", "Resetting all online components")

    # Reset ADWIN drift detector
    try:
        from stream.drift.monitor import reset_adwin, reset_baseline
        reset_adwin()
        reset_baseline()
    except Exception as e:
        _log_event("RESET_ERROR", f"ADWIN/PSI reset failed: {e}")

    # Reset online metrics
    try:
        from stream.online.metrics import reset
        reset()
    except Exception as e:
        _log_event("RESET_ERROR", f"Metrics reset failed: {e}")

    # Reset model race
    try:
        from stream.online.race import reset_race
        reset_race()
    except Exception as e:
        _log_event("RESET_ERROR", f"Race reset failed: {e}")

    # Reset error ADWIN
    _init_error_adwin()

    _drift_signals = 0
    _last_adaptation_at = time.time()
    _post_adaptation_labels = 0
    _state = "STABILIZING"
    _log_event("STABILIZING", "Waiting for post-adaptation labels")

    return {"event": "adaptation_triggered", "timestamp": _last_adaptation_at}


def check_stabilization():
    """After N post-adaptation labels, transition to RECOVERED → STABLE."""
    global _state, _post_adaptation_labels

    if _state != "STABILIZING":
        return

    _post_adaptation_labels += 1

    if _post_adaptation_labels >= _stabilization_threshold:
        _state = "RECOVERED"
        _log_event("RECOVERED", f"Stabilized after {_post_adaptation_labels} labels")
        _state = "STABLE"
        _log_event("STABLE", "System recovered")


def force_adaptation() -> dict:
    """Manual trigger for demo purposes."""
    global _state, _drift_signals
    _state = "DRIFT_DETECTED"
    _drift_signals = _consecutive_threshold
    _log_event("FORCE_TRIGGERED", "Manual adaptation triggered")
    return _trigger_adaptation()


def get_status() -> dict:
    """Return adaptation state, signals, cooldown, and recent log."""
    cooldown_remaining = 0.0
    if _last_adaptation_at:
        elapsed = time.time() - _last_adaptation_at
        cooldown_remaining = max(0, _cooldown_seconds - elapsed)

    return {
        "state": _state,
        "drift_signals": _drift_signals,
        "consecutive_threshold": _consecutive_threshold,
        "cooldown_remaining": round(cooldown_remaining, 1),
        "post_adaptation_labels": _post_adaptation_labels,
        "stabilization_threshold": _stabilization_threshold,
        "last_adaptation_at": _last_adaptation_at,
        "log": list(_adaptation_log)[-10:],
    }


def _log_event(event_type: str, message: str):
    """Append to adaptation log."""
    entry = {
        "timestamp": time.time(),
        "event": event_type,
        "message": message,
    }
    _adaptation_log.append(entry)
    log.info("Adaptation event", event=event_type, message=message)
