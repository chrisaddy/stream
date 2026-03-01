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

_risk_threshold = 0.7


def get_risk_threshold() -> float:
    return _risk_threshold


def set_risk_threshold(value: float):
    global _risk_threshold
    _risk_threshold = max(0.05, min(0.9, value))


def _heuristic_risk_score(fee_rate: float, vsize: int, fee: int) -> float:
    """Compute a risk score from the 3 features available in live transactions.

    Factors:
    - Fee rate anomaly: unusually high or low fee rates are suspicious
    - Transaction size: very large transactions deserve scrutiny
    - Disproportionate fees: fee much higher/lower than expected for vsize
    """
    score = 0.0

    # Fee rate anomaly — typical range is 1-100 sat/vB
    if fee_rate > 500:
        score += 0.3   # extreme overpay
    elif fee_rate > 200:
        score += 0.15
    elif fee_rate < 1:
        score += 0.2   # suspiciously cheap

    # Large transaction size — typical is 200-500 vB
    if vsize > 10000:
        score += 0.2   # very large tx
    elif vsize > 5000:
        score += 0.1

    # Disproportionate fee — fee should roughly equal fee_rate * vsize
    expected_fee = fee_rate * vsize
    if expected_fee > 0:
        ratio = fee / expected_fee
        if ratio > 3 or ratio < 0.3:
            score += 0.15

    # Base noise to prevent all-zero scores
    score += 0.05

    return min(round(score, 4), 1.0)


def _heuristic_explanation(fee_rate: float, vsize: int, fee: int) -> list[dict]:
    """Generate transparent factor descriptions for the heuristic."""
    factors = []

    if fee_rate > 200:
        factors.append({"feature": "fee_rate_anomaly", "shap_value": round(0.15 + min((fee_rate - 200) / 1000, 0.15), 4)})
    elif fee_rate < 1:
        factors.append({"feature": "low_fee_rate", "shap_value": 0.2})

    if vsize > 5000:
        factors.append({"feature": "large_tx_size", "shap_value": round(min(vsize / 50000, 0.2), 4)})

    expected = fee_rate * vsize
    if expected > 0:
        ratio = fee / expected
        if ratio > 3 or ratio < 0.3:
            factors.append({"feature": "fee_disproportion", "shap_value": 0.15})

    if not factors:
        factors.append({"feature": "baseline", "shap_value": 0.05})

    return factors


async def score_transaction(
    txid: str, vsize: int, fee: int
) -> dict:
    """Score a live transaction using honest heuristic scoring.

    The Elliptic XGBoost model requires 166 graph features we don't have
    for live transactions. This uses transparent heuristics on the 3 features
    we do have: vsize, fee, and fee_rate.
    """
    start = time.time()

    fee_rate = fee / max(vsize, 1)
    risk_score = _heuristic_risk_score(fee_rate, vsize, fee)
    shap_features = _heuristic_explanation(fee_rate, vsize, fee)

    inference_ms = (time.time() - start) * 1000
    threshold = get_risk_threshold()
    risk_label = "HIGH" if risk_score > threshold else "MED" if risk_score > threshold * 0.6 else "LOW"

    input_hash = hashlib.sha256(
        json.dumps({"txid": txid, "vsize": vsize, "fee": fee}).encode()
    ).hexdigest()

    return {
        "risk_score": risk_score,
        "risk_label": risk_label,
        "inference_ms": round(inference_ms, 2),
        "model_name": "live-heuristic",
        "model_version": "v1.0",
        "input_hash": input_hash,
        "shap_features": shap_features,
        "is_demo": False,
    }
