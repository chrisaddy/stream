"""Graph features for Lightning Network nodes."""

import networkx as nx
import numpy as np


def compute_node_features(G: nx.Graph, pubkey: str) -> dict:
    """Compute features for a single node."""
    if pubkey not in G:
        return {}

    node_data = G.nodes[pubkey]
    degree = G.degree(pubkey)

    # Neighbor capacities
    neighbor_caps = [G.nodes[n].get("capacity", 0) for n in G.neighbors(pubkey)]

    features = {
        "degree": degree,
        "capacity": node_data.get("capacity", 0),
        "channels": node_data.get("channels", 0),
        "avg_neighbor_capacity": np.mean(neighbor_caps) if neighbor_caps else 0,
        "max_neighbor_capacity": max(neighbor_caps) if neighbor_caps else 0,
        "capacity_per_channel": node_data.get("capacity", 0) / max(degree, 1),
    }

    # Graph centrality (only compute if graph is small enough)
    if G.number_of_nodes() <= 500:
        try:
            features["betweenness"] = nx.betweenness_centrality(G).get(pubkey, 0)
            features["closeness"] = nx.closeness_centrality(G).get(pubkey, 0)
        except Exception:
            features["betweenness"] = 0
            features["closeness"] = 0
    else:
        features["betweenness"] = 0
        features["closeness"] = 0

    # Edge capacities
    edge_caps = [G.edges[pubkey, n].get("capacity", 0) for n in G.neighbors(pubkey)]
    features["avg_channel_capacity"] = np.mean(edge_caps) if edge_caps else 0
    features["total_edge_capacity"] = sum(edge_caps)

    return features


def compute_all_features(G: nx.Graph) -> list[dict]:
    """Compute features for all nodes in the graph."""
    results = []
    for node in G.nodes:
        feat = compute_node_features(G, node)
        feat["pubkey"] = node
        feat["alias"] = G.nodes[node].get("alias", "")
        results.append(feat)
    return results


def compute_routing_score(features: dict) -> float:
    """Heuristic routing score: higher = better routing node.

    Combines connectivity, capacity, and centrality.
    """
    degree = features.get("degree", 0)
    capacity = features.get("capacity", 0)
    betweenness = features.get("betweenness", 0)
    avg_channel = features.get("avg_channel_capacity", 0)

    # Normalize and combine (simple weighted sum)
    score = (
        0.3 * min(degree / 100, 1.0) +
        0.3 * min(capacity / 10_000_000_000, 1.0) +  # 100 BTC in sats
        0.2 * min(betweenness * 100, 1.0) +
        0.2 * min(avg_channel / 1_000_000_000, 1.0)  # 10 BTC in sats
    )
    return round(score, 4)
