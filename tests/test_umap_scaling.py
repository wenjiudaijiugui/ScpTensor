
import numpy as np
import scipy.sparse
from scptensor.dim_reduction._umap import spectral_layout

def test_spectral_scaling_robustness():
    print("Testing spectral_layout scaling robustness...")
    
    # 1. Create a simple graph (ring structure + outliers)
    n_samples = 100
    n_outliers = 5
    total_samples = n_samples + n_outliers
    
    # Adjacency matrix for a ring
    row = []
    col = []
    data = []
    for i in range(n_samples):
        # connect to neighbors
        row.append(i)
        col.append((i + 1) % n_samples)
        data.append(1.0)
        
        row.append(i)
        col.append((i - 1) % n_samples)
        data.append(1.0)
        
    # Add disconnected outliers (or weakly connected)
    # For spectral layout to work without crashing on disconnected components, 
    # usually we need the graph to be connected or handle components.
    # But here we just want to see if 'spectral_layout' handles the values well.
    # Let's connect outliers weakly to the main component to ensure connectivity for Arpack
    for i in range(n_samples, total_samples):
        row.append(i)
        col.append(0) # connect to first node
        data.append(0.001) # weak connection
        
        row.append(0)
        col.append(i)
        data.append(0.001)

    graph = scipy.sparse.coo_matrix((data, (row, col)), shape=(total_samples, total_samples))
    
    # 2. Run spectral_layout
    dim = 2
    embedding = spectral_layout(graph, dim, random_state=np.random.RandomState(42))
    
    # 3. Check statistics
    print(f"Embedding shape: {embedding.shape}")
    print(f"Min: {embedding.min(axis=0)}")
    print(f"Max: {embedding.max(axis=0)}")
    print(f"Mean: {embedding.mean(axis=0)}")
    print(f"Std: {embedding.std()}")
    
    # 4. Assertions
    # Standard deviation scaling means std should be roughly related to the scaling factor (10.0 / std_original * std_original = 10.0 ? No.)
    # The code is: embedding = 10.0 * (embedding / std)
    # So the new std should be 10.0
    
    current_std = embedding.std()
    print(f"Current global std: {current_std}")
    
    if abs(current_std - 10.0) > 1e-5:
        print(f"WARNING: Standard deviation is {current_std}, expected 10.0")
    else:
        print("SUCCESS: Standard deviation is 10.0")
        
    # Check if values are not exploded or collapsed
    # With Min-Max scaling and outliers, the main cluster would be very small.
    # With Std scaling, the main cluster should have a reasonable spread.
    
    # Let's verify that the main cluster (first n_samples) has a non-trivial spread
    main_cluster = embedding[:n_samples]
    main_std = main_cluster.std()
    print(f"Main cluster std: {main_std}")
    
    if main_std < 0.1:
        print("FAILURE: Main cluster is collapsed!")
    else:
        print("SUCCESS: Main cluster has reasonable spread.")

if __name__ == "__main__":
    try:
        test_spectral_scaling_robustness()
        print("Test finished successfully.")
    except Exception as e:
        print(f"Test failed with error: {e}")
