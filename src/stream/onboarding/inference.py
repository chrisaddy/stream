"""Onboarding risk inference — bridge between trained model and live API."""

import numpy as np
import structlog

log = structlog.get_logger()

# Feature order must match pd.get_dummies(drop_first=True) on training data.
# 8 numeric features + 7 one-hot dummies = 15 total.
FEATURE_ORDER = [
    "account_age_days",
    "phone_verified",
    "document_verification_score",
    "ip_country_matches_document",
    "device_fingerprint_seen_before",
    "transaction_velocity_first_24h",
    "signup_hour",
    "initial_deposit_usd",
    # One-hot dummies (drop_first=True)
    "email_domain_type_disposable",
    "email_domain_type_free",
    "document_type_national_id",
    "document_type_passport",
    "referral_source_paid",
    "referral_source_referral",
    "referral_source_social",
]

LABEL_MAP = {0: "low_risk", 1: "medium_risk", 2: "high_risk", 3: "blocked"}


def encode_form_input(
    email_domain_type: str = "corporate",
    phone_verified: int = 1,
    doc_score: float = 0.85,
    ip_match: int = 1,
    tx_velocity: int = 2,
    initial_deposit: float = 500.0,
    account_age_days: int = 30,
    device_fingerprint_seen: int = 0,
    document_type: str = "passport",
    referral_source: str = "organic",
    signup_hour: int = 12,
) -> np.ndarray:
    """Encode form inputs into the 15-feature vector expected by the model.

    Replicates pd.get_dummies(drop_first=True) encoding for a single sample.
    """
    row = {
        "account_age_days": account_age_days,
        "phone_verified": phone_verified,
        "document_verification_score": doc_score,
        "ip_country_matches_document": ip_match,
        "device_fingerprint_seen_before": device_fingerprint_seen,
        "transaction_velocity_first_24h": tx_velocity,
        "signup_hour": signup_hour,
        "initial_deposit_usd": initial_deposit,
        # One-hot: email_domain_type (baseline = "corporate")
        "email_domain_type_disposable": int(email_domain_type == "disposable"),
        "email_domain_type_free": int(email_domain_type == "free"),
        # One-hot: document_type (baseline = "drivers_license")
        "document_type_national_id": int(document_type == "national_id"),
        "document_type_passport": int(document_type == "passport"),
        # One-hot: referral_source (baseline = "organic")
        "referral_source_paid": int(referral_source == "paid"),
        "referral_source_referral": int(referral_source == "referral"),
        "referral_source_social": int(referral_source == "social"),
    }

    return np.array([[row[f] for f in FEATURE_ORDER]], dtype=float)


def predict_risk(model_data: dict, form_values: dict) -> dict:
    """Run onboarding risk prediction.

    Args:
        model_data: {"model": CalibratedClassifierCV, "scaler": StandardScaler}
        form_values: Dict with keys matching encode_form_input parameters.

    Returns:
        Dict with predicted_class, class_probabilities, risk_label, action.
    """
    try:
        model = model_data["model"]
        scaler = model_data["scaler"]

        X = encode_form_input(**form_values)
        X_scaled = scaler.transform(X)

        pred_class = int(model.predict(X_scaled)[0])
        probas = model.predict_proba(X_scaled)[0]

        risk_label = LABEL_MAP.get(pred_class, "unknown")
        action = (
            "auto_approve"
            if risk_label == "low_risk"
            else "manual_review"
            if risk_label in ("medium_risk", "high_risk")
            else "block"
        )

        return {
            "predicted_class": pred_class,
            "risk_label": risk_label,
            "class_probabilities": {LABEL_MAP[i]: round(float(p), 4) for i, p in enumerate(probas)},
            "action": action,
            "source": "ML-MODEL",
        }
    except Exception as e:
        log.warning("Onboarding model prediction failed", error=str(e))
        return None
