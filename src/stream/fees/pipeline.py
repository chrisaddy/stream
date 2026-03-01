"""Prefect flow for fee estimation training."""

import pickle
import structlog
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact

from stream.fees.collector import collect_snapshot
from stream.fees.features import build_feature_matrix, build_targets
from stream.fees.model import serialize_fee_model, train_fee_model
from stream.fees.evaluate import evaluate_fee_model
from stream.model_card import (
    build_model_card,
    save_model_card,
    feature_importance_chart,
    regime_chart,
    fig_to_png,
)

log = structlog.get_logger()


@task(name="collect-mempool-data")
async def collect_data(n_snapshots: int = 100):
    """Collect mempool snapshots (or load cached)."""
    import httpx
    import asyncio

    snapshots = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        for i in range(min(n_snapshots, 10)):  # Quick collection for demo
            snapshot = await collect_snapshot(client)
            if snapshot:
                snapshots.append(snapshot)
            await asyncio.sleep(2)

    log.info("Collected snapshots", count=len(snapshots))
    return snapshots


@task(name="featurize-snapshots")
def featurize(snapshots: list[dict]):
    X = build_feature_matrix(snapshots)
    y = build_targets(snapshots)
    log.info("Features built", shape=X.shape)
    return X, y


@task(name="train-fee-model")
def train_model(X, y):
    import numpy as np

    # Train for 1-block confirmation target
    feature_cols = [c for c in X.columns if not c.startswith("rec_")]
    X_features = X[feature_cols].values
    y_target = y["target_1_block"].values

    if len(X_features) < 10:
        log.warning("Not enough data for training, using dummy model")
        return None, feature_cols, X_features, y_target

    model = train_fee_model(X_features, y_target)
    log.info("Fee model trained")
    return model, feature_cols, X_features, y_target


@task(name="evaluate-fee-model")
def evaluate_model(model, X_features, y_target, feature_cols):
    """Evaluate fee model and return metrics."""
    import numpy as np

    if model is None:
        return None

    y_pred = model.predict(X_features)
    metrics = evaluate_fee_model(y_target, y_pred)

    # Feature importance from LightGBM
    fi = []
    importances = model.feature_importances_
    for name, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
        fi.append({"feature": name, "importance": float(imp)})

    metrics["feature_importance"] = fi
    metrics["n_samples"] = len(y_target)

    log.info("Fee model evaluated", mae=f"{metrics['mae']:.4f}", rmse=f"{metrics['rmse']:.4f}")
    return metrics


@task(name="save-fee-model")
def save_model(model, name: str = "fee-lgbm"):
    if model is None:
        return
    try:
        from io import BytesIO
        from prefect_aws.s3 import S3Bucket
        s3 = S3Bucket.load("model-store")
        model_bytes = serialize_fee_model(model)
        s3.upload_from_file_object(BytesIO(model_bytes), f"models/{name}/latest.pkl")
        log.info("Fee model saved to R2")
    except Exception as e:
        log.warning("R2 upload failed, saving locally", error=str(e))
        import os
        os.makedirs(f"models/{name}", exist_ok=True)
        with open(f"models/{name}/latest.pkl", "wb") as f:
            f.write(serialize_fee_model(model))


@task(name="build-fee-model-card")
def build_card(model, metrics, feature_cols, X_features, y_target):
    """Build model card with full metadata and save to R2."""
    if model is None or metrics is None:
        log.warning("No model to build card for")
        return None

    card = build_model_card(
        name="fee-lgbm",
        description="LightGBM fee estimator predicting optimal sat/vB for 1-block confirmation. "
                    "Uses live mempool state features.",
        metrics={
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "mape": metrics["mape"],
            "median_ae": metrics["median_ae"],
        },
        data_summary={
            "dataset": "Live mempool snapshots (mempool.space)",
            "n_features": len(feature_cols),
            "n_samples": metrics["n_samples"],
            "feature_names": feature_cols,
        },
        training_params={
            "model": "LGBMRegressor",
            "objective": "regression_l1 (MAE)",
            "n_estimators": 300,
            "max_depth": 8,
            "learning_rate": 0.05,
        },
        feature_importance=metrics.get("feature_importance", [])[:20],
    )

    save_model_card(card, "fee-lgbm")
    log.info("Fee model card saved")
    return card


@task(name="create-fee-artifacts")
def create_artifacts(metrics, card):
    """Create Prefect artifacts for fee model."""
    if metrics is None:
        return

    markdown = f"""# Fee Estimation — LightGBM Results

## Metrics
| Metric | Value |
|--------|-------|
| MAE | {metrics['mae']:.4f} sat/vB |
| RMSE | {metrics['rmse']:.4f} sat/vB |
| MAPE | {metrics['mape']:.4f} |
| Median AE | {metrics['median_ae']:.4f} sat/vB |
| Samples | {metrics['n_samples']} |
"""
    create_markdown_artifact(key="fee-lgbm-metrics", markdown=markdown)

    # Image artifacts
    if card:
        try:
            from prefect.artifacts import create_image_artifact

            fi = card.get("feature_importance", [])
            if fi:
                names = [f["feature"] for f in fi]
                scores = [f["importance"] for f in fi]
                fig = feature_importance_chart(names, scores)
                create_image_artifact(fig_to_png(fig), key="fee-feature-importance")

            log.info("Fee artifacts created")
        except Exception as e:
            log.warning("Chart artifact creation failed", error=str(e))


@flow(name="fee-estimation-training")
async def train_fees_pipeline():
    snapshots = await collect_data()
    if not snapshots:
        log.error("No snapshots collected")
        return
    X, y = featurize(snapshots)
    model, feature_cols, X_features, y_target = train_model(X, y)
    metrics = evaluate_model(model, X_features, y_target, feature_cols)
    save_model(model)
    card = build_card(model, metrics, feature_cols, X_features, y_target)
    create_artifacts(metrics, card)


if __name__ == "__main__":
    import asyncio
    asyncio.run(train_fees_pipeline())
