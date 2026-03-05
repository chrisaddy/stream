"""Prefect flow for Lightning Network analysis."""

import json
import os

import httpx
import structlog
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact

from stream.lightning.data import build_graph, get_top_nodes_connectivity
from stream.lightning.evaluate import evaluate_capacity_model
from stream.lightning.features import compute_all_features
from stream.lightning.model import FEATURE_COLS, serialize_lightning_model, train_capacity_model
from stream.model_card import (
    build_model_card,
    feature_importance_chart,
    fig_to_png,
    save_model_card,
)

log = structlog.get_logger()


@task(name="fetch-lightning-topology")
async def fetch_topology():
    async with httpx.AsyncClient(timeout=30.0) as client:
        nodes = await get_top_nodes_connectivity(client, limit=200)
    if not nodes:
        log.error("No lightning nodes fetched")
        return None
    log.info("Fetched lightning nodes", count=len(nodes))
    return nodes


@task(name="build-lightning-graph")
def build_lightning_graph(nodes):
    G = build_graph(nodes)
    log.info("Graph built", nodes=G.number_of_nodes(), edges=G.number_of_edges())
    return G


@task(name="compute-lightning-features")
def compute_features(G):
    features = compute_all_features(G)
    log.info("Features computed", count=len(features))
    return features


@task(name="save-lightning-features")
def save_features(features: list[dict], name: str = "lightning-lgbm"):
    """Save computed node features to R2/local for live inference."""
    data = json.dumps(features, default=str).encode()
    key = f"models/{name}/features.json"
    try:
        from io import BytesIO

        from prefect_aws.s3 import S3Bucket

        s3 = S3Bucket.load("model-store")
        s3.upload_from_file_object(BytesIO(data), key)
        log.info("Lightning features saved to R2", count=len(features))
    except Exception as e:
        log.warning("R2 features upload failed, saving locally", error=str(e))
        os.makedirs(f"models/{name}", exist_ok=True)
        with open(f"models/{name}/features.json", "wb") as f:
            f.write(data)


@task(name="train-lightning-model")
def train_model(features):
    model = train_capacity_model(features)
    log.info("Lightning model trained")
    return model


@task(name="evaluate-lightning-model")
def evaluate_model(model, features):
    """Evaluate lightning model and return metrics."""
    import numpy as np

    X = np.array([[f.get(c, 0) for c in FEATURE_COLS] for f in features])
    y = np.array([f.get("capacity", 0) for f in features])

    mask = y > 0
    X, y = X[mask], y[mask]

    if len(X) == 0:
        return None

    y_pred = model.predict(X)
    metrics = evaluate_capacity_model(y, y_pred)

    # Feature importance from LightGBM
    fi = []
    importances = model.feature_importances_
    for name, imp in sorted(zip(FEATURE_COLS, importances), key=lambda x: x[1], reverse=True):
        fi.append({"feature": name, "importance": float(imp)})

    metrics["feature_importance"] = fi

    log.info("Lightning model evaluated", mae=f"{metrics['mae']:.2f}", r2=f"{metrics['r2']:.4f}")
    return metrics


@task(name="save-lightning-model")
def save_model(model, name: str = "lightning-lgbm"):
    try:
        from io import BytesIO

        from prefect_aws.s3 import S3Bucket

        s3 = S3Bucket.load("model-store")
        s3.upload_from_file_object(
            BytesIO(serialize_lightning_model(model)), f"models/{name}/latest.pkl"
        )
    except Exception as e:
        log.warning("R2 upload failed, saving locally", error=str(e))
        import os

        os.makedirs(f"models/{name}", exist_ok=True)
        with open(f"models/{name}/latest.pkl", "wb") as f:
            f.write(serialize_lightning_model(model))


@task(name="build-lightning-model-card")
def build_card(model, metrics, features):
    """Build model card with full metadata and save to R2."""
    if metrics is None:
        log.warning("No metrics to build card")
        return None

    card = build_model_card(
        name="lightning-lgbm",
        description="LightGBM regressor predicting optimal Lightning node capacity "
        "based on network topology features.",
        metrics={
            "mae": metrics["mae"],
            "r2": metrics["r2"],
            "median_ae": metrics["median_ae"],
            "n_samples": metrics["n_samples"],
        },
        data_summary={
            "dataset": "Lightning Network topology (mempool.space API)",
            "n_features": len(FEATURE_COLS),
            "n_nodes": len(features),
            "feature_names": FEATURE_COLS,
        },
        training_params={
            "model": "LGBMRegressor",
            "objective": "regression_l1",
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
        },
        feature_importance=metrics.get("feature_importance", []),
    )

    save_model_card(card, "lightning-lgbm")
    log.info("Lightning model card saved")
    return card


@task(name="create-lightning-artifacts")
def create_artifacts(metrics, card):
    """Create Prefect artifacts for lightning model."""
    if metrics is None:
        return

    markdown = f"""# Lightning Network — Capacity Prediction Results

## Metrics
| Metric | Value |
|--------|-------|
| MAE | {metrics["mae"]:.2f} |
| R² | {metrics["r2"]:.4f} |
| Median AE | {metrics["median_ae"]:.2f} |
| Samples | {metrics["n_samples"]} |
"""
    create_markdown_artifact(key="lightning-lgbm-metrics", markdown=markdown)

    # Image artifacts
    if card:
        try:
            from prefect.artifacts import create_image_artifact

            fi = card.get("feature_importance", [])
            if fi:
                names = [f["feature"] for f in fi]
                scores = [f["importance"] for f in fi]
                fig = feature_importance_chart(names, scores)
                create_image_artifact(fig_to_png(fig), key="lightning-feature-importance")

            log.info("Lightning artifacts created")
        except Exception as e:
            log.warning("Chart artifact creation failed", error=str(e))


@flow(name="lightning-network-analysis")
async def train_lightning_pipeline():
    nodes = await fetch_topology()
    if not nodes:
        return
    G = build_lightning_graph(nodes)
    features = compute_features(G)
    save_features(features)
    model = train_model(features)
    metrics = evaluate_model(model, features)
    save_model(model)
    card = build_card(model, metrics, features)
    create_artifacts(metrics, card)


if __name__ == "__main__":
    import asyncio

    asyncio.run(train_lightning_pipeline())
