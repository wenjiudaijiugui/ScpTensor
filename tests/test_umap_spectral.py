

import numpy as np
import scipy.sparse
from scptensor.dim_reduction._umap import spectral_layout
import pytest

def test_spectral_layout_no_density_norm():
    n_samples = 100
    n_components = 2
    random_state = np.random.RandomState(42)
    
    # Create a random sparse graph (symmetric)
    adjacency = scipy.sparse.random(n_samples, n_samples, density=0.1, random_state=random_state)
    graph = (adjacency + adjacency.T) / 2
    graph = graph.tocoo()
    
    # Run spectral layout
    embedding = spectral_layout(
        graph=graph,
        dim=n_components,
        random_state=random_state
    )
    
    assert embedding.shape == (n_samples, n_components)
    assert not np.any(np.isnan(embedding)), "Embedding contains NaNs"
    assert not np.any(np.isinf(embedding)), "Embedding contains Infs"
    
    # Check scaling
    assert np.max(embedding) <= 10.0 + 1e-5 # Should be within range roughly
    assert np.min(embedding) >= -10.0 - 1e-5
    
    print("Spectral layout ran successfully without density renormalization.")

if __name__ == "__main__":
    test_spectral_layout_no_density_norm()
