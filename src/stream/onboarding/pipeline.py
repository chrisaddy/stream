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
from stream.onboarding.evaluate import evaluate_risk_model

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

    log.info("LR metrics", **{k: f"{v:.4f}" if isinstance(v, float) else v for k, v in lr_metrics.items()})
    log.info("XGB metrics", **{k: f"{v:.4f}" if isinstance(v, float) else v for k, v in xgb_metrics.items()})

    return lr_metrics, xgb_metrics


@task(name="save-onboarding-model")
def save_model(model, scaler, name: str = "onboarding-xgb"):
    try:
        from prefect_aws.s3 import S3Bucket
        s3 = S3Bucket.load("model-store")
        s3.upload_from_bytes(serialize_onboarding_model(model, scaler), f"models/{name}/latest.pkl")
    except Exception as e:
        log.warning("R2 upload failed, saving locally", error=str(e))
        import os
        os.makedirs(f"models/{name}", exist_ok=True)
        with open(f"models/{name}/latest.pkl", "wb") as f:
            f.write(serialize_onboarding_model(model, scaler))


@flow(name="onboarding-risk-scoring")
def train_onboarding_pipeline():
    X_train, X_test, y_train, y_test = generate_data()
    lr_model, lr_scaler = train_logistic(X_train, y_train)
    xgb_model, xgb_scaler = train_xgboost(X_train, y_train)
    lr_metrics, xgb_metrics = evaluate_models(
        lr_model, xgb_model, X_test, y_test, lr_scaler, xgb_scaler
    )
    save_model(xgb_model, xgb_scaler)
    return lr_metrics, xgb_metrics


if __name__ == "__main__":
    train_onboarding_pipeline()
