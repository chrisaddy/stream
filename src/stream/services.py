"""Scoring service."""

import hashlib
import json
import time

import numpy as np
import structlog

from stream.app.models import get_model

log = structlog.get_logger()

RISK_THRESHOLD = 0.7


async def score_transaction(
    txid: str, vsize: int, fee: int
) -> dict:
    """Score a single transaction. Returns a result dict.

    1. Run real XGBoost inference if model loaded (with SHAP)
    2. Fall back to demo beta distribution scores
    """
    start = time.time()

    model = get_model("illicit-xgboost")
    is_demo = model is None

    if model is not None:
        try:
            fee_rate = fee / max(vsize, 1)
            features = np.zeros(166)
            features[0] = vsize
            features[1] = fee
            features[2] = fee_rate
            X = features.reshape(1, -1)
            risk_score = float(model.predict_proba(X)[:, 1][0])

            # SHAP explanation
            shap_features = _compute_shap(model, X)
        except Exception as e:
            log.warning("Model inference failed, falling back to demo", error=str(e))
            risk_score = float(np.random.beta(2, 8))
            shap_features = None
            is_demo = True
    else:
        risk_score = float(np.random.beta(2, 8))
        shap_features = None

    inference_ms = (time.time() - start) * 1000
    risk_label = "HIGH" if risk_score > 0.7 else "MED" if risk_score > 0.4 else "LOW"

    input_hash = hashlib.sha256(
        json.dumps({"txid": txid, "vsize": vsize, "fee": fee}).encode()
    ).hexdigest()

    return {
        "risk_score": round(risk_score, 4),
        "risk_label": risk_label,
        "inference_ms": round(inference_ms, 2),
        "model_name": "illicit-xgboost",
        "model_version": "v1.0",
        "input_hash": input_hash,
        "shap_features": shap_features,
        "is_demo": is_demo,
    }


def _compute_shap(model, X: np.ndarray) -> list[dict] | None:
    """Try to compute SHAP values; return top-5 features or None."""
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        if isinstance(sv, list):
            sv = sv[1]  # class-1 shap values
        vals = sv[0]
        names = [f"feature_{i}" for i in range(len(vals))]
        pairs = sorted(zip(names, vals), key=lambda p: abs(p[1]), reverse=True)[:5]
        return [{"feature": n, "shap_value": round(float(v), 4)} for n, v in pairs]
    except Exception as e:
        log.debug("SHAP computation failed", error=str(e))
        return None
