
import numpy as np
import scipy.sparse
from scptensor.dim_reduction._umap import optimize_layout
import pytest

def test_optimize_layout_runs():
    n_samples = 100
    n_components = 2
    random_state = np.random.RandomState(42)
    
    # Create random embedding
    embedding = random_state.uniform(low=-10, high=10, size=(n_samples, n_components)).astype(np.float32)
    
    # Create random sparse graph (symmetric)
    # Create a random adjacency matrix
    adjacency = scipy.sparse.random(n_samples, n_samples, density=0.1, random_state=random_state)
    # Symmetrize
    graph = (adjacency + adjacency.T) / 2
    graph = graph.tocoo()
    
    # Parameters
    n_epochs = 10
    learning_rate = 1.0
    a = 1.0
    b = 1.0
    repulsion_strength = 1.0
    negative_sample_rate = 5
    
    # Run optimization
    new_embedding = optimize_layout(
        embedding=embedding,
        graph=graph,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        a=a,
        b=b,
        repulsion_strength=repulsion_strength,
        negative_sample_rate=negative_sample_rate,
        random_state=random_state
    )
    
    assert new_embedding.shape == (n_samples, n_components)
    assert not np.allclose(embedding, new_embedding), "Embedding should have changed"
    print("Optimization ran successfully.")

if __name__ == "__main__":
    test_optimize_layout_runs()
