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
