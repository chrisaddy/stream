"""Prefect flow for Lightning Network analysis."""

import httpx
import structlog
from prefect import flow, task

from stream.lightning.data import build_graph, get_top_nodes_connectivity
from stream.lightning.features import compute_all_features
from stream.lightning.model import serialize_lightning_model, train_capacity_model

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


@task(name="train-lightning-model")
def train_model(features):
    model = train_capacity_model(features)
    log.info("Lightning model trained")
    return model


@task(name="save-lightning-model")
def save_model(model, name: str = "lightning-lgbm"):
    try:
        from io import BytesIO
        from prefect_aws.s3 import S3Bucket
        s3 = S3Bucket.load("model-store")
        s3.upload_from_file_object(BytesIO(serialize_lightning_model(model)), f"models/{name}/latest.pkl")
    except Exception as e:
        log.warning("R2 upload failed, saving locally", error=str(e))
        import os
        os.makedirs(f"models/{name}", exist_ok=True)
        with open(f"models/{name}/latest.pkl", "wb") as f:
            f.write(serialize_lightning_model(model))


@flow(name="lightning-network-analysis")
async def train_lightning_pipeline():
    nodes = await fetch_topology()
    if not nodes:
        return
    G = build_lightning_graph(nodes)
    features = compute_features(G)
    model = train_model(features)
    save_model(model)


if __name__ == "__main__":
    import asyncio
    asyncio.run(train_lightning_pipeline())
