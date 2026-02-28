"""Elliptic Bitcoin dataset loading.

Uses PyTorch Geometric's built-in EllipticBitcoinDataset loader
and also provides a pandas-based loader for XGBoost.
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.datasets import EllipticBitcoinDataset


DATA_DIR = "./data/elliptic"


def load_pyg_dataset(root: str = DATA_DIR) -> EllipticBitcoinDataset:
    """Load Elliptic dataset via PyTorch Geometric."""
    dataset = EllipticBitcoinDataset(root=root)
    return dataset


def load_pandas_dataset(root: str = DATA_DIR) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load Elliptic dataset and return as pandas DataFrame.

    Returns:
        features: DataFrame with 166 columns (timestep + 93 local + 72 aggregated)
        labels: Series with 0=licit, 1=illicit, 2=unknown
        timesteps: Series with timestep for each transaction
    """
    dataset = load_pyg_dataset(root)
    data = dataset[0]

    features = pd.DataFrame(data.x.numpy())
    features.columns = [f"feature_{i}" for i in range(features.shape[1])]

    # Labels: 0=unknown, 1=licit, 2=illicit in original
    # Remap to: 0=licit, 1=illicit, 2=unknown
    raw_labels = data.y.numpy()
    label_map = {0: 2, 1: 0, 2: 1}  # unknown->2, licit->0, illicit->1
    labels = pd.Series([label_map.get(l, 2) for l in raw_labels])

    # Timestep is the first feature (feature_0)
    timesteps = features["feature_0"].astype(int)

    return features, labels, timesteps


def get_edge_index(root: str = DATA_DIR) -> torch.Tensor:
    """Get edge index for GCN."""
    dataset = load_pyg_dataset(root)
    return dataset[0].edge_index
