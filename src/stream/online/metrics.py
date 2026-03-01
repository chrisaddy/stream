"""River online metrics singleton — tracks cumulative and rolling classification metrics."""

from collections import deque

import structlog

log = structlog.get_logger()

# Cumulative metrics
_precision = None
_recall = None
_f1 = None
_rocauc = None

# Rolling metrics (window=50)
_rolling_precision = None
_rolling_recall = None
_rolling_f1 = None
_rolling_rocauc = None

# History for sparklines
_metric_history: deque[dict] = deque(maxlen=200)
_n_labels = 0
_initialized = False


def _ensure_init():
    """Lazy-initialize all River metrics."""
    global _precision, _recall, _f1, _rocauc
    global _rolling_precision, _rolling_recall, _rolling_f1, _rolling_rocauc
    global _initialized

    if _initialized:
        return

    from river.metrics import Precision, Recall, F1, ROCAUC
    from river.utils import Rolling

    _precision = Precision()
    _recall = Recall()
    _f1 = F1()
    _rocauc = ROCAUC()

    _rolling_precision = Rolling(Precision(), window_size=50)
    _rolling_recall = Rolling(Recall(), window_size=50)
    _rolling_f1 = Rolling(F1(), window_size=50)
    _rolling_rocauc = Rolling(ROCAUC(), window_size=50)

    _initialized = True


def record_label(y_true: int, y_pred_score: float, threshold: float):
    """Update all metrics with a new labeled observation."""
    global _n_labels

    _ensure_init()

    y_pred = y_pred_score >= threshold

    # Update cumulative
    _precision.update(bool(y_true), y_pred)
    _recall.update(bool(y_true), y_pred)
    _f1.update(bool(y_true), y_pred)
    try:
        _rocauc.update(bool(y_true), y_pred_score)
    except Exception:
        pass

    # Update rolling
    _rolling_precision.update(bool(y_true), y_pred)
    _rolling_recall.update(bool(y_true), y_pred)
    _rolling_f1.update(bool(y_true), y_pred)
    try:
        _rolling_rocauc.update(bool(y_true), y_pred_score)
    except Exception:
        pass

    _n_labels += 1

    # Record snapshot for history
    _metric_history.append(get_snapshot())


def get_snapshot() -> dict:
    """Return current metric values (cumulative + rolling)."""
    if not _initialized or _n_labels == 0:
        return {
            "n_labels": 0,
            "cumulative": {"precision": 0, "recall": 0, "f1": 0, "rocauc": 0},
            "rolling": {"precision": 0, "recall": 0, "f1": 0, "rocauc": 0},
        }

    def _safe_get(metric):
        try:
            return round(float(metric.get()), 4)
        except Exception:
            return 0.0

    return {
        "n_labels": _n_labels,
        "cumulative": {
            "precision": _safe_get(_precision),
            "recall": _safe_get(_recall),
            "f1": _safe_get(_f1),
            "rocauc": _safe_get(_rocauc),
        },
        "rolling": {
            "precision": _safe_get(_rolling_precision),
            "recall": _safe_get(_rolling_recall),
            "f1": _safe_get(_rolling_f1),
            "rocauc": _safe_get(_rolling_rocauc),
        },
    }


def get_history() -> list[dict]:
    """Return metric history for convergence charts."""
    return list(_metric_history)


def reset():
    """Reinitialize all metrics (called during adaptation)."""
    global _initialized, _n_labels, _metric_history
    _initialized = False
    _n_labels = 0
    _metric_history = deque(maxlen=200)
    _ensure_init()
    log.info("Online metrics reset")
