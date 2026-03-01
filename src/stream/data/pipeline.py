"""Hourly data collection pipeline.

Fetches mempool snapshots and lightning topology, appends to existing
data in R2, and saves back. No training — just data accumulation.

Training pipelines pull from the same R2 keys when they run.
"""

import json
import os
from datetime import datetime, timezone
from io import BytesIO

import httpx
import structlog
from prefect import flow, task

from stream.fees.collector import collect_snapshot
from stream.lightning.data import get_top_nodes_connectivity

log = structlog.get_logger()

# R2 keys — shared with training pipelines
FEE_SNAPSHOTS_KEY = "models/fee-lgbm/snapshots.json"
LIGHTNING_SNAPSHOTS_KEY = "data/lightning/snapshots.json"
MAX_FEE_SNAPSHOTS = 5000
MAX_LIGHTNING_SNAPSHOTS = 500


def _get_s3_client():
    """Get boto3 S3 client for R2."""
    import boto3
    from stream.config import settings

    if not settings.R2_ENDPOINT_URL:
        return None, None
    client = boto3.client(
        "s3",
        endpoint_url=settings.R2_ENDPOINT_URL,
        aws_access_key_id=settings.R2_ACCESS_KEY_ID,
        aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
    )
    return client, settings.R2_BUCKET_NAME


def _load_json_from_r2(key: str) -> list | None:
    """Load a JSON list from R2."""
    try:
        s3, bucket = _get_s3_client()
        if s3 is None:
            return None
        resp = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(resp["Body"].read())
    except Exception as e:
        log.debug("R2 load failed", key=key, error=str(e))
        return None


def _save_json_to_r2(data: list, key: str):
    """Save a JSON list to R2 with local fallback."""
    encoded = json.dumps(data, default=str).encode()
    try:
        s3, bucket = _get_s3_client()
        if s3 is None:
            raise ValueError("No R2 config")
        s3.put_object(Bucket=bucket, Key=key, Body=encoded)
        log.info("Saved to R2", key=key, count=len(data))
    except Exception as e:
        log.warning("R2 save failed, saving locally", key=key, error=str(e))
        os.makedirs(os.path.dirname(key), exist_ok=True)
        with open(key, "wb") as f:
            f.write(encoded)


# === Fee Snapshots ===


@task(name="collect-fee-snapshots")
async def collect_fee_snapshots(n_snapshots: int = 10):
    """Collect a batch of mempool snapshots.

    Default 10 snapshots with 2s spacing = ~20s per run.
    At once per hour, that's 240 snapshots/day.
    """
    import asyncio

    snapshots = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        for _ in range(n_snapshots):
            snapshot = await collect_snapshot(client)
            if snapshot:
                snapshots.append(snapshot)
            await asyncio.sleep(2)

    log.info("Collected fee snapshots", count=len(snapshots))
    return snapshots


@task(name="append-fee-snapshots")
def append_fee_snapshots(new_snapshots: list[dict]):
    """Load existing snapshots from R2, append new, save back."""
    existing = _load_json_from_r2(FEE_SNAPSHOTS_KEY) or []
    combined = existing + new_snapshots
    # Keep most recent, capped
    combined = combined[-MAX_FEE_SNAPSHOTS:]
    _save_json_to_r2(combined, FEE_SNAPSHOTS_KEY)
    log.info("Fee snapshots accumulated", existing=len(existing), new=len(new_snapshots), total=len(combined))
    return len(combined)


# === Lightning Topology ===


@task(name="collect-lightning-snapshot")
async def collect_lightning_snapshot():
    """Collect a snapshot of lightning network topology."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        nodes = await get_top_nodes_connectivity(client, limit=200)

    if not nodes:
        log.warning("No lightning nodes fetched")
        return None

    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "nodes": nodes,
    }
    log.info("Collected lightning snapshot", node_count=len(nodes))
    return snapshot


@task(name="append-lightning-snapshot")
def append_lightning_snapshot(snapshot: dict | None):
    """Append lightning topology snapshot to R2."""
    if snapshot is None:
        return 0

    existing = _load_json_from_r2(LIGHTNING_SNAPSHOTS_KEY) or []
    existing.append(snapshot)
    existing = existing[-MAX_LIGHTNING_SNAPSHOTS:]
    _save_json_to_r2(existing, LIGHTNING_SNAPSHOTS_KEY)
    log.info("Lightning snapshots accumulated", total=len(existing))
    return len(existing)


# === Combined Flow ===


@flow(name="hourly-data-collection")
async def collect_data_pipeline(fee_snapshots: int = 10):
    """Collect and persist data for all models.

    Designed to run every hour. Lightweight — no training, just data.
    ~30s per run (10 fee snapshots + 1 lightning snapshot).
    """
    fee_data = await collect_fee_snapshots(fee_snapshots)
    fee_count = append_fee_snapshots(fee_data)

    lightning_data = await collect_lightning_snapshot()
    lightning_count = append_lightning_snapshot(lightning_data)

    log.info(
        "Data collection complete",
        fee_total=fee_count,
        lightning_total=lightning_count,
    )


if __name__ == "__main__":
    import asyncio
    asyncio.run(collect_data_pipeline())
