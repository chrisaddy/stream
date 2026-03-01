"""Prefect flow for onboarding risk scoring."""

import structlog
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact
from sklearn.model_selection import train_test_split

from stream.onboarding.data import encode_features, generate_synthetic_kyc
from stream.onboarding.model import (
    serialize_onboarding_model,
    train_calibrated_xgboost,
    train_logistic_baseline,
)
from stream.onboarding.evaluate import evaluate_risk_model, calibration_analysis, per_tier_analysis
from stream.model_card import (
    build_model_card,
    save_model_card,
    calibration_chart,
    feature_importance_chart,
    fig_to_png,
)

log = structlog.get_logger()


@task(name="generate-kyc-data")
def generate_data(n_samples: int = 10000):
    df = generate_synthetic_kyc(n_samples)
    X, y = encode_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    log.info("KYC data generated", train_size=len(X_train), test_size=len(X_test))
    return X_train, X_test, y_train, y_test


@task(name="train-logistic-baseline")
def train_logistic(X_train, y_train):
    model, scaler = train_logistic_baseline(X_train, y_train)
    log.info("Logistic regression trained")
    return model, scaler


@task(name="train-xgboost-onboarding")
def train_xgboost(X_train, y_train):
    model, scaler = train_calibrated_xgboost(X_train, y_train)
    log.info("Calibrated XGBoost trained")
    return model, scaler


@task(name="evaluate-onboarding-models")
def evaluate_models(logistic, xgb, X_test, y_test, logistic_scaler, xgb_scaler):
    import numpy as np

    X_test_lr = logistic_scaler.transform(X_test)
    X_test_xgb = xgb_scaler.transform(X_test)

    lr_pred = logistic.predict(X_test_lr)
    lr_prob = logistic.predict_proba(X_test_lr)
    xgb_pred = xgb.predict(X_test_xgb)
    xgb_prob = xgb.predict_proba(X_test_xgb)

    lr_metrics = evaluate_risk_model(y_test, lr_pred, lr_prob, "logistic_regression")
    xgb_metrics = evaluate_risk_model(y_test, xgb_pred, xgb_prob, "calibrated_xgboost")

    # Calibration analysis for XGBoost
    cal = calibration_analysis(y_test, xgb_prob)

    # Per-tier analysis
    tiers = per_tier_analysis(y_test, xgb_pred)

    log.info("LR metrics", **{k: f"{v:.4f}" if isinstance(v, float) else v for k, v in lr_metrics.items()})
    log.info("XGB metrics", **{k: f"{v:.4f}" if isinstance(v, float) else v for k, v in xgb_metrics.items()})

    return lr_metrics, xgb_metrics, cal, tiers


@task(name="save-onboarding-model")
def save_model(model, scaler, name: str = "onboarding-xgb"):
    try:
        from io import BytesIO
        from prefect_aws.s3 import S3Bucket
        s3 = S3Bucket.load("model-store")
        s3.upload_from_file_object(BytesIO(serialize_onboarding_model(model, scaler)), f"models/{name}/latest.pkl")
    except Exception as e:
        log.warning("R2 upload failed, saving locally", error=str(e))
        import os
        os.makedirs(f"models/{name}", exist_ok=True)
        with open(f"models/{name}/latest.pkl", "wb") as f:
            f.write(serialize_onboarding_model(model, scaler))


@task(name="build-onboarding-model-card")
def build_card(lr_metrics, xgb_metrics, cal, tiers, X_train, X_test, y_train, y_test):
    """Build model card with full metadata and save to R2."""
    import numpy as np

    # Feature names from the encoded dataset
    feature_names = [
        "account_age_days", "phone_verified", "document_verification_score",
        "ip_country_matches_document", "device_fingerprint_seen_before",
        "transaction_velocity_first_24h", "signup_hour", "initial_deposit_usd",
        "email_domain_type_disposable", "email_domain_type_free",
        "document_type_national_id", "document_type_passport",
        "referral_source_paid", "referral_source_referral", "referral_source_social",
    ]

    card = build_model_card(
        name="onboarding-xgb",
        description="Calibrated XGBoost for KYC onboarding risk scoring. "
                    "Platt-scaled probabilities for meaningful risk tiers (low/medium/high/blocked).",
        metrics={
            "logistic_regression": lr_metrics,
            "calibrated_xgboost": xgb_metrics,
        },
        data_summary={
            "dataset": "Synthetic KYC",
            "n_features": int(X_train.shape[1]),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "class_distribution": {
                "low_risk": int((y_test == 0).sum()),
                "medium_risk": int((y_test == 1).sum()),
                "high_risk": int((y_test == 2).sum()),
                "blocked": int((y_test == 3).sum()),
            },
        },
        training_params={
            "model": "CalibratedClassifierCV(XGBClassifier)",
            "calibration_method": "sigmoid (Platt scaling)",
            "cv_folds": 5,
            "base_n_estimators": 200,
            "base_max_depth": 5,
        },
        feature_importance=[
            {"feature": name, "importance": 0.0} for name in feature_names
        ],
        curves={
            "calibration": cal,
            "per_tier": tiers,
        },
    )

    save_model_card(card, "onboarding-xgb")
    log.info("Onboarding model card saved")
    return card


@task(name="create-onboarding-artifacts")
def create_artifacts(lr_metrics, xgb_metrics, card):
    """Create Prefect artifacts for onboarding results."""
    markdown = f"""# Onboarding Risk Scoring Results

## Model Comparison
| Metric | Logistic Regression | Calibrated XGBoost |
|--------|--------------------|--------------------|
| Accuracy | {lr_metrics['accuracy']:.4f} | {xgb_metrics['accuracy']:.4f} |
| F1 (macro) | {lr_metrics['f1_macro']:.4f} | {xgb_metrics['f1_macro']:.4f} |
| F1 (weighted) | {lr_metrics['f1_weighted']:.4f} | {xgb_metrics['f1_weighted']:.4f} |
| Log Loss | {lr_metrics['log_loss']:.4f} | {xgb_metrics['log_loss']:.4f} |

## Per-Tier Analysis
"""
    tiers = card["curves"].get("per_tier", [])
    if tiers:
        markdown += "| Tier | Actual | Predicted | TPR |\n|------|--------|-----------|-----|\n"
        for t in tiers:
            markdown += f"| {t['tier']} | {t['actual_count']} | {t['predicted_count']} | {t['true_positive_rate']:.3f} |\n"

    create_markdown_artifact(key="onboarding-metrics", markdown=markdown)

    # Image artifacts
    try:
        from prefect.artifacts import create_image_artifact

        cal = card["curves"].get("calibration", {})
        for cls_key, cls_data in cal.items():
            if cls_data.get("prob_true") and cls_data.get("prob_pred"):
                fig = calibration_chart(cls_data["prob_true"], cls_data["prob_pred"], cls_key)
                create_image_artifact(fig_to_png(fig), key=f"onboarding-calibration-{cls_key}")
                break  # Just create one representative chart

        log.info("Onboarding artifacts created")
    except Exception as e:
        log.warning("Chart artifact creation failed", error=str(e))


@flow(name="onboarding-risk-scoring")
def train_onboarding_pipeline():
    X_train, X_test, y_train, y_test = generate_data()
    lr_model, lr_scaler = train_logistic(X_train, y_train)
    xgb_model, xgb_scaler = train_xgboost(X_train, y_train)
    lr_metrics, xgb_metrics, cal, tiers = evaluate_models(
        lr_model, xgb_model, X_test, y_test, lr_scaler, xgb_scaler
    )
    save_model(xgb_model, xgb_scaler)
    card = build_card(lr_metrics, xgb_metrics, cal, tiers, X_train, X_test, y_train, y_test)
    create_artifacts(lr_metrics, xgb_metrics, card)
    return lr_metrics, xgb_metrics


if __name__ == "__main__":
    train_onboarding_pipeline()
