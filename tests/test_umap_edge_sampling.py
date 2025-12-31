
import numpy as np
import scipy.sparse
from scptensor.dim_reduction._umap import optimize_layout
import pytest

def test_high_weight_edge_impact():
    # This test conceptually checks if high weight edges are processed correctly.
    # Since we cannot easily inspect internal loop counters without modifying the code to log,
    # we will ensure the code runs without error and produces a valid embedding.
    # The logic fix is structural (changing if to while), so functional verification 
    # that the code doesn't hang or crash is the primary automated check here.
    
    n_samples = 20
    n_components = 2
    random_state = np.random.RandomState(42)
    
    embedding = random_state.uniform(low=-10, high=10, size=(n_samples, n_components)).astype(np.float32)
    
    # Create a graph with some very high weight edges
    rows = [0, 1, 2]
    cols = [1, 2, 0]
    data = [10.0, 1.0, 0.5] # One very high weight edge
    graph = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(n_samples, n_samples))
    
    # Parameters
    n_epochs = 5
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
    assert not np.allclose(embedding, new_embedding)
    print("Optimization with high weights ran successfully.")

if __name__ == "__main__":
    test_high_weight_edge_impact()
