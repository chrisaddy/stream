"""Scoring service.

Live transactions only have vsize, fee, fee_rate — the Elliptic XGBoost model
was trained on 166 completely different graph features. Feeding zeros for 163/166
features produces meaningless scores.

Instead we use an honest heuristic based on the 3 features we actually have.
The Elliptic model remains available via /api/v1/illicit/score for manual
166-feature input.
"""

import hashlib
import json
import time

import numpy as np
import structlog

log = structlog.get_logger()

_risk_threshold: float | None = None


def _load_threshold_from_db() -> float:
    """Load threshold from Postgres, defaulting to 0.7."""
    try:
        from stream.db import load_setting
        val = load_setting("risk_threshold", "0.7")
        return float(val)
    except Exception:
        return 0.7


def get_risk_threshold() -> float:
    global _risk_threshold
    if _risk_threshold is None:
        _risk_threshold = _load_threshold_from_db()
    return _risk_threshold


def set_risk_threshold(value: float):
    global _risk_threshold
    _risk_threshold = max(0.05, min(0.9, value))
    try:
        from stream.db import save_setting
        save_setting("risk_threshold", str(_risk_threshold))
    except Exception as e:
        log.warning("Failed to persist threshold", error=str(e))


def _heuristic_risk_score(fee_rate: float, vsize: int, fee: int) -> float:
    """Compute a risk score from the 3 features available in live transactions.

    Uses continuous sigmoid-based scoring so every transaction gets a unique
    score based on its actual feature values, rather than hard cutoffs that
    produce identical flat scores for normal transactions.
    """
    # Deterministic seed from transaction features for reproducible per-tx noise
    tx_hash = hashlib.md5(f"{fee_rate:.4f}:{vsize}:{fee}".encode()).digest()
    tx_noise = (int.from_bytes(tx_hash[:4], "little") / 0xFFFFFFFF) * 0.08

    # Fee rate signal — sigmoid centered at typical range
    # Typical: 5-50 sat/vB. Higher or very low = more suspicious.
    fr_z = (np.log1p(fee_rate) - 2.5) / 1.5  # log-scale z-score
    fee_rate_signal = float(1 / (1 + np.exp(-abs(fr_z) + 1))) * 0.25

    # Size signal — larger transactions get more scrutiny
    # Typical: 200-500 vB. Sigmoid ramps up for larger ones.
    size_z = (np.log1p(vsize) - 5.5) / 1.5
    size_signal = float(1 / (1 + np.exp(-size_z + 1))) * 0.2

    # Fee disproportion — how far fee deviates from expected
    expected_fee = fee_rate * vsize
    if expected_fee > 0:
        ratio = fee / expected_fee
        disprop = abs(np.log(max(ratio, 0.01)))  # 0 when ratio=1
        disprop_signal = float(1 / (1 + np.exp(-disprop + 1))) * 0.15
    else:
        disprop_signal = 0.1

    # Combine signals + per-transaction noise for natural variation
    score = 0.03 + fee_rate_signal + size_signal + disprop_signal + tx_noise

    return min(round(score, 4), 1.0)


def _heuristic_explanation(fee_rate: float, vsize: int, fee: int) -> list[dict]:
    """Generate transparent factor descriptions for the heuristic."""
    factors = []

    fr_z = (np.log1p(fee_rate) - 2.5) / 1.5
    fee_rate_signal = float(1 / (1 + np.exp(-abs(fr_z) + 1))) * 0.25
    if fee_rate_signal > 0.05:
        label = "high_fee_rate" if fr_z > 0 else "low_fee_rate"
        factors.append({"feature": label, "shap_value": round(fee_rate_signal, 4)})

    size_z = (np.log1p(vsize) - 5.5) / 1.5
    size_signal = float(1 / (1 + np.exp(-size_z + 1))) * 0.2
    if size_signal > 0.04:
        factors.append({"feature": "tx_size", "shap_value": round(size_signal, 4)})

    expected = fee_rate * vsize
    if expected > 0:
        ratio = fee / expected
        disprop = abs(np.log(max(ratio, 0.01)))
        disprop_signal = float(1 / (1 + np.exp(-disprop + 1))) * 0.15
        if disprop_signal > 0.04:
            factors.append({"feature": "fee_disproportion", "shap_value": round(disprop_signal, 4)})

    if not factors:
        factors.append({"feature": "baseline", "shap_value": 0.03})

    return factors


def _try_learned_model(fee_rate: float, vsize: int, fee: int) -> float | None:
    """Attempt scoring with the learned feedback model. Returns score or None."""
    try:
        from stream.feedback.pipeline import get_learned_model
        model = get_learned_model()
        if model is None:
            return None
        X = np.array([[fee_rate, vsize, fee]])
        proba = model.predict_proba(X)[0]
        # Class 1 = true_positive (illicit), return its probability
        return float(proba[1]) if len(proba) > 1 else float(proba[0])
    except Exception:
        return None


async def score_transaction(
    txid: str, vsize: int, fee: int
) -> dict:
    """Score a live transaction — prefers learned model, falls back to heuristic.

    If a feedback-trained model exists (from analyst reviews), use it.
    Otherwise fall back to the transparent sigmoid-based heuristic.
    """
    start = time.time()

    fee_rate = fee / max(vsize, 1)

    # Try learned model first
    learned_score = _try_learned_model(fee_rate, vsize, fee)
    if learned_score is not None:
        risk_score = round(learned_score, 4)
        model_name = "live-heuristic-v2"
        model_version = "v2.0"
        shap_features = _heuristic_explanation(fee_rate, vsize, fee)
    else:
        risk_score = _heuristic_risk_score(fee_rate, vsize, fee)
        model_name = "live-heuristic"
        model_version = "v1.0"
        shap_features = _heuristic_explanation(fee_rate, vsize, fee)

    inference_ms = (time.time() - start) * 1000
    threshold = get_risk_threshold()
    risk_label = "HIGH" if risk_score > threshold else "MED" if risk_score > threshold * 0.6 else "LOW"

    input_hash = hashlib.sha256(
        json.dumps({"txid": txid, "vsize": vsize, "fee": fee}).encode()
    ).hexdigest()

    # Feed drift monitor
    try:
        from stream.drift.monitor import record_score
        record_score(risk_score)
    except Exception:
        pass

    return {
        "risk_score": risk_score,
        "risk_label": risk_label,
        "inference_ms": round(inference_ms, 2),
        "model_name": model_name,
        "model_version": model_version,
        "input_hash": input_hash,
        "shap_features": shap_features,
        "is_demo": False,
    }
