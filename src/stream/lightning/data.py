"""Lightning Network data from mempool.space API."""

import httpx
import networkx as nx
import structlog

log = structlog.get_logger()

MEMPOOL_BASE = "https://mempool.space/api"


async def get_network_stats(client: httpx.AsyncClient) -> dict | None:
    try:
        resp = await client.get(f"{MEMPOOL_BASE}/v1/lightning/statistics/latest")
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError as e:
        log.error("Failed to fetch lightning stats", error=str(e))
        return None


async def get_top_nodes_connectivity(client: httpx.AsyncClient, limit: int = 100) -> list | None:
    try:
        resp = await client.get(f"{MEMPOOL_BASE}/v1/lightning/nodes/rankings/connectivity")
        resp.raise_for_status()
        return resp.json()[:limit]
    except httpx.HTTPError as e:
        log.error("Failed to fetch top nodes", error=str(e))
        return None


async def get_top_nodes_liquidity(client: httpx.AsyncClient, limit: int = 100) -> list | None:
    try:
        resp = await client.get(f"{MEMPOOL_BASE}/v1/lightning/nodes/rankings/liquidity")
        resp.raise_for_status()
        return resp.json()[:limit]
    except httpx.HTTPError as e:
        log.error("Failed to fetch liquidity nodes", error=str(e))
        return None


async def get_node_details(client: httpx.AsyncClient, pubkey: str) -> dict | None:
    try:
        resp = await client.get(f"{MEMPOOL_BASE}/v1/lightning/nodes/{pubkey}")
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError as e:
        log.error("Failed to fetch node details", error=str(e), pubkey=pubkey)
        return None


async def get_node_channels(client: httpx.AsyncClient, pubkey: str) -> list | None:
    try:
        resp = await client.get(f"{MEMPOOL_BASE}/v1/lightning/channels?public_key={pubkey}")
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError as e:
        log.error("Failed to fetch channels", error=str(e))
        return None


def build_graph(nodes: list[dict], channels: list[dict] | None = None) -> nx.Graph:
    """Build NetworkX graph from Lightning nodes and channels."""
    G = nx.Graph()

    for node in nodes:
        pubkey = node.get("publicKey", node.get("public_key", ""))
        G.add_node(pubkey, **{
            "alias": node.get("alias", ""),
            "capacity": node.get("capacity", 0),
            "channels": node.get("channels", node.get("active_channel_count", 0)),
            "first_seen": node.get("firstSeen", 0),
            "city": node.get("city", {}).get("en", "") if isinstance(node.get("city"), dict) else "",
            "country": node.get("country", {}).get("en", "") if isinstance(node.get("country"), dict) else "",
        })

    if channels:
        for ch in channels:
            n1 = ch.get("node1_public_key", "")
            n2 = ch.get("node2_public_key", "")
            if n1 and n2:
                G.add_edge(n1, n2, capacity=ch.get("capacity", 0))

    return G
