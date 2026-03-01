"""Feedback retraining pipeline — loads reviews, trains on human labels, saves model."""

import pickle
import time

import numpy as np
import structlog

from stream.db import SessionLocal
from stream.feedback.model import train_live_heuristic

log = structlog.get_logger()

# Module-level state for retraining status
_retrain_status = {
    "state": "IDLE",  # IDLE / TRAINING / COMPLETE / ERROR
    "started_at": None,
    "completed_at": None,
    "metrics": None,
    "error": None,
    "n_labels": 0,
}

# In-memory model store — loaded model sits here after retraining
_learned_model = None
_model_metadata = None

# River online model state
_river_model = None
_river_metrics = None
_river_n_samples = 0
_river_n_positive = 0
_river_n_negative = 0


def _init_river_model():
    """Create a River online learning pipeline: StandardScaler → LogisticRegression."""
    from river.compose import Pipeline
    from river.linear_model import LogisticRegression
    from river.preprocessing import StandardScaler

    global _river_model, _river_metrics
    _river_model = Pipeline(StandardScaler(), LogisticRegression(l2=0.01))
    from river.metrics import Accuracy
    _river_metrics = Accuracy()


def river_learn_one(fee_rate: float, vsize: float, fee: float, label: int):
    """Test-then-train: predict first (update metric), then learn."""
    global _river_model, _river_metrics, _river_n_samples, _river_n_positive, _river_n_negative

    if _river_model is None:
        _init_river_model()

    x = {"fee_rate": fee_rate, "vsize": vsize, "fee": fee}
    y = bool(label)

    # Test: predict before learning (for metric tracking)
    y_pred = _river_model.predict_one(x)
    if y_pred is not None:
        _river_metrics.update(y, y_pred)

    # Train
    _river_model.learn_one(x, y)
    _river_n_samples += 1
    if label:
        _river_n_positive += 1
    else:
        _river_n_negative += 1

    log.info("River learn_one", n_samples=_river_n_samples, label=label,
             n_pos=_river_n_positive, n_neg=_river_n_negative)


def river_predict_one(fee_rate: float, vsize: float, fee: float) -> float | None:
    """Return P(class=True) if model is ready (has seen both classes), else None."""
    model = get_river_model()
    if model is None:
        return None
    x = {"fee_rate": fee_rate, "vsize": vsize, "fee": fee}
    try:
        proba = model.predict_proba_one(x)
        return proba.get(True, proba.get(1, 0.0))
    except Exception:
        return None


def get_river_model():
    """Return the River model only if it has seen both classes."""
    if _river_model is None or _river_n_positive < 1 or _river_n_negative < 1:
        return None
    return _river_model


def get_river_metrics() -> dict:
    """Return River model status and metrics."""
    is_ready = _river_n_positive >= 1 and _river_n_negative >= 1
    return {
        "n_samples": _river_n_samples,
        "n_positive": _river_n_positive,
        "n_negative": _river_n_negative,
        "accuracy": float(_river_metrics.get()) if _river_metrics and _river_n_samples > 0 else 0.0,
        "is_ready": is_ready,
    }


def warm_up_river_model():
    """Replay all existing ReviewRecords through River learn_one at startup."""
    from stream.models.alerts import AlertRecord
    from stream.models.reviews import ReviewRecord

    if _river_model is None:
        _init_river_model()

    db = SessionLocal()
    try:
        reviews = db.query(ReviewRecord).order_by(ReviewRecord.reviewed_at).all()
        if not reviews:
            log.info("River warm-up: no reviews to replay")
            return

        tx_ids = [r.tx_id for r in reviews]
        alerts = db.query(AlertRecord).filter(AlertRecord.tx_id.in_(tx_ids)).all()
        alert_map = {a.tx_id: a for a in alerts}

        replayed = 0
        for review in reviews:
            alert = alert_map.get(review.tx_id)
            if not alert or not alert.raw_input:
                continue
            raw = alert.raw_input
            fee_rate = raw.get("fee_rate", 0)
            vsize = raw.get("vsize", 0)
            fee = raw.get("fee", 0)
            label = 1 if review.verdict == "true_positive" else 0
            river_learn_one(fee_rate, vsize, fee, label)
            replayed += 1

        log.info("River warm-up complete", replayed=replayed, n_samples=_river_n_samples)
    except Exception as e:
        log.warning("River warm-up failed", error=str(e))
    finally:
        db.close()


def get_retrain_status() -> dict:
    return dict(_retrain_status)


def get_learned_model():
    """Return the learned model if available, else None."""
    return _learned_model


def get_model_metadata() -> dict | None:
    return _model_metadata


def _load_labeled_data() -> tuple[np.ndarray, np.ndarray, int]:
    """Load reviews joined with alert raw_input features.

    Returns (X, y, n_labels) where X has columns [fee_rate, vsize, fee]
    and y is 1 for true_positive, 0 for false_positive.
    """
    from stream.models.alerts import AlertRecord
    from stream.models.reviews import ReviewRecord

    db = SessionLocal()
    try:
        reviews = db.query(ReviewRecord).all()
        if not reviews:
            return np.array([]), np.array([]), 0

        # Build lookup of alert raw_input by tx_id
        tx_ids = [r.tx_id for r in reviews]
        alerts = db.query(AlertRecord).filter(AlertRecord.tx_id.in_(tx_ids)).all()
        alert_map = {a.tx_id: a for a in alerts}

        rows_X = []
        rows_y = []
        for review in reviews:
            alert = alert_map.get(review.tx_id)
            if not alert or not alert.raw_input:
                continue
            raw = alert.raw_input
            fee_rate = raw.get("fee_rate", 0)
            vsize = raw.get("vsize", 0)
            fee = raw.get("fee", 0)
            rows_X.append([fee_rate, vsize, fee])
            rows_y.append(1 if review.verdict == "true_positive" else 0)

        if not rows_X:
            return np.array([]), np.array([]), len(reviews)

        return np.array(rows_X, dtype=float), np.array(rows_y, dtype=float), len(rows_X)
    finally:
        db.close()


def run_retraining() -> dict:
    """Execute the full retraining pipeline synchronously.

    Returns the retrain status dict.
    """
    global _learned_model, _model_metadata, _retrain_status

    _retrain_status["state"] = "TRAINING"
    _retrain_status["started_at"] = time.time()
    _retrain_status["error"] = None
    _retrain_status["metrics"] = None

    try:
        X, y, n_labels = _load_labeled_data()
        _retrain_status["n_labels"] = n_labels

        if len(X) < 3:
            _retrain_status["state"] = "ERROR"
            _retrain_status["error"] = f"Need at least 3 labeled alerts with raw features, got {len(X)}"
            return _retrain_status

        # Check we have both classes
        if len(np.unique(y)) < 2:
            _retrain_status["state"] = "ERROR"
            _retrain_status["error"] = "Need both TP and FP labels to train. Review more alerts with diverse verdicts."
            return _retrain_status

        result = train_live_heuristic(X, y)
        _learned_model = result["model"]
        _model_metadata = {
            "accuracy": result["accuracy"],
            "method": result["method"],
            "n_train": result["n_train"],
            "n_test": result["n_test"],
            "n_labels": n_labels,
            "trained_at": time.time(),
            "feature_names": ["fee_rate", "vsize", "fee"],
        }

        _retrain_status["state"] = "COMPLETE"
        _retrain_status["completed_at"] = time.time()
        _retrain_status["metrics"] = {
            "accuracy": result["accuracy"],
            "method": result["method"],
            "n_train": result["n_train"],
            "n_test": result["n_test"],
        }

        log.info("Retraining complete", accuracy=result["accuracy"], method=result["method"], n_labels=n_labels)

        # Try to save to R2 (best-effort)
        try:
            _save_model_to_r2(result["model"], _model_metadata)
        except Exception as e:
            log.warning("R2 upload failed (model still in memory)", error=str(e))

        return _retrain_status

    except Exception as e:
        _retrain_status["state"] = "ERROR"
        _retrain_status["error"] = str(e)
        log.error("Retraining failed", error=str(e))
        return _retrain_status


def _save_model_to_r2(model, metadata: dict):
    """Best-effort save to R2."""
    import os

    model_bytes = pickle.dumps(model)

    # Always save locally as fallback
    os.makedirs("models/live-heuristic-v2", exist_ok=True)
    with open("models/live-heuristic-v2/latest.pkl", "wb") as f:
        f.write(model_bytes)
    log.info("Model saved locally", path="models/live-heuristic-v2/latest.pkl", size_bytes=len(model_bytes))

    # Try R2
    try:
        from io import BytesIO
        from stream.config import settings

        if settings.R2_ENDPOINT_URL:
            import boto3
            s3 = boto3.client(
                "s3",
                endpoint_url=settings.R2_ENDPOINT_URL,
                aws_access_key_id=settings.R2_ACCESS_KEY_ID,
                aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
            )
            s3.upload_fileobj(BytesIO(model_bytes), settings.R2_BUCKET_NAME, "models/live-heuristic-v2/latest.pkl")
            log.info("Model uploaded to R2")
    except Exception as e:
        log.debug("R2 upload skipped", error=str(e))
