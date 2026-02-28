"""Temporal split and feature engineering for Elliptic dataset.

Key insight: random split causes data leakage because transactions are temporally
correlated. Temporal split mirrors real production: train on past, predict future.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def temporal_split(
    features: pd.DataFrame,
    labels: pd.Series,
    timesteps: pd.Series,
    split_timestep: int = 34,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data temporally: train on timesteps 1-34, test on 35-49.

    Filters out unknown labels (y=2).

    Returns:
        X_train, y_train, X_test, y_test as numpy arrays
    """
    known_mask = labels != 2

    train_mask = known_mask & (timesteps <= split_timestep)
    test_mask = known_mask & (timesteps > split_timestep)

    X_train = features[train_mask].values
    y_train = labels[train_mask].values
    X_test = features[test_mask].values
    y_test = labels[test_mask].values

    return X_train, y_train, X_test, y_test


def scale_features(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Standardize features. Fit on train only to avoid leakage."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def get_class_weight(y_train: np.ndarray) -> float:
    """Calculate scale_pos_weight for XGBoost (handles ~10:1 class imbalance)."""
    n_licit = (y_train == 0).sum()
    n_illicit = (y_train == 1).sum()
    return n_licit / max(n_illicit, 1)


def get_temporal_groups(
    timesteps: pd.Series,
    labels: pd.Series,
    split_timestep: int = 34,
) -> list[int]:
    """Get unique test timesteps for per-timestep evaluation."""
    known_mask = labels != 2
    test_mask = known_mask & (timesteps > split_timestep)
    return sorted(timesteps[test_mask].unique().tolist())
