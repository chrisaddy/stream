"""Prefect flow for fee estimation training."""

import pickle
import structlog
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact

from stream.fees.collector import collect_snapshot
from stream.fees.features import build_feature_matrix, build_targets
from stream.fees.model import serialize_fee_model, train_fee_model

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
        return None

    model = train_fee_model(X_features, y_target)
    log.info("Fee model trained")
    return model


@task(name="save-fee-model")
def save_model(model, name: str = "fee-lgbm"):
    if model is None:
        return
    try:
        from prefect_aws.s3 import S3Bucket
        s3 = S3Bucket.load("model-store")
        model_bytes = serialize_fee_model(model)
        s3.upload_from_bytes(model_bytes, f"models/{name}/latest.pkl")
        log.info("Fee model saved to R2")
    except Exception as e:
        log.warning("R2 upload failed, saving locally", error=str(e))
        import os
        os.makedirs(f"models/{name}", exist_ok=True)
        with open(f"models/{name}/latest.pkl", "wb") as f:
            f.write(serialize_fee_model(model))


@flow(name="fee-estimation-training")
async def train_fees_pipeline():
    snapshots = await collect_data()
    if not snapshots:
        log.error("No snapshots collected")
        return
    X, y = featurize(snapshots)
    model = train_model(X, y)
    save_model(model)


if __name__ == "__main__":
    import asyncio
    asyncio.run(train_fees_pipeline())
