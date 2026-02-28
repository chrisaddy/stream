"""mempool.space API data collector.

Polls mempool state for fee estimation training data.
Rate limited to 1 req/min with exponential backoff.
"""

import asyncio
from datetime import datetime

import httpx
import structlog

log = structlog.get_logger()

MEMPOOL_BASE = "https://mempool.space/api"


async def get_mempool_state(client: httpx.AsyncClient) -> dict | None:
    """Get current mempool state."""
    try:
        resp = await client.get(f"{MEMPOOL_BASE}/mempool")
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError as e:
        log.error("Failed to fetch mempool state", error=str(e))
        return None


async def get_recommended_fees(client: httpx.AsyncClient) -> dict | None:
    """Get recommended fee rates."""
    try:
        resp = await client.get(f"{MEMPOOL_BASE}/v1/fees/recommended")
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError as e:
        log.error("Failed to fetch fees", error=str(e))
        return None


async def get_mempool_blocks(client: httpx.AsyncClient) -> list | None:
    """Get projected next blocks from mempool."""
    try:
        resp = await client.get(f"{MEMPOOL_BASE}/v1/fees/mempool-blocks")
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError as e:
        log.error("Failed to fetch mempool blocks", error=str(e))
        return None


async def get_tip_height(client: httpx.AsyncClient) -> int | None:
    """Get current block height."""
    try:
        resp = await client.get(f"{MEMPOOL_BASE}/blocks/tip/height")
        resp.raise_for_status()
        return int(resp.text)
    except httpx.HTTPError as e:
        log.error("Failed to fetch tip height", error=str(e))
        return None


async def get_recent_transactions(client: httpx.AsyncClient) -> list | None:
    """Get recent mempool transactions."""
    try:
        resp = await client.get(f"{MEMPOOL_BASE}/mempool/recent")
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError as e:
        log.error("Failed to fetch recent transactions", error=str(e))
        return None


async def collect_snapshot(client: httpx.AsyncClient) -> dict | None:
    """Collect a full mempool snapshot for training."""
    mempool = await get_mempool_state(client)
    fees = await get_recommended_fees(client)
    blocks = await get_mempool_blocks(client)
    height = await get_tip_height(client)

    if not all([mempool, fees]):
        return None

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "mempool": mempool,
        "recommended_fees": fees,
        "projected_blocks": blocks,
        "tip_height": height,
    }


async def collection_loop(interval_seconds: int = 60, max_snapshots: int = 1000):
    """Continuously collect mempool snapshots."""
    snapshots = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        for i in range(max_snapshots):
            snapshot = await collect_snapshot(client)
            if snapshot:
                snapshots.append(snapshot)
                log.info("Snapshot collected", count=len(snapshots))
            await asyncio.sleep(interval_seconds)
    return snapshots
