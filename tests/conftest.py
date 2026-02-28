import pytest


@pytest.fixture
def sample_features():
    """166 features: 1 timestep + 93 local + 72 aggregated."""
    import numpy as np
    return np.random.randn(166).tolist()
