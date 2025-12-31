
import numpy as np
import scipy.sparse
import polars as pl
from scptensor.dim_reduction._umap import umap
from scptensor.core.structures import ScpContainer, ScpMatrix, Assay

def test_umap():
    print("Creating mock data...")
    n_samples = 100
    n_features = 50
    X = np.random.rand(n_samples, n_features).astype(np.float32)
    
    # Create a mock ScpContainer
    # We need to check how ScpContainer is structured.
    # Based on imports: from scptensor.core.structures import ScpContainer, ScpMatrix, Assay
    
    # Mocking the structure since I don't want to import the whole heavy library if not needed
    # But I have the code available.
    
    layer = ScpMatrix(X, M=np.zeros_like(X, dtype=bool))
    
    # Assay needs 'var' which is feature metadata
    var_df = pl.DataFrame({
        "feature_id": [f"gene_{i}" for i in range(n_features)]
    })
    
    # Container needs 'obs' which is sample metadata
    obs_df = pl.DataFrame({
        "sample_id": [f"cell_{i}" for i in range(n_samples)]
    })
    
    assay = Assay(layers={'counts': layer}, var=var_df)
    container = ScpContainer(assays={'RNA': assay}, obs=obs_df)
    
    print("Running UMAP...")
    container = umap(
        container=container,
        assay_name='RNA',
        base_layer='counts',
        n_epochs=100, # Short run
        n_neighbors=10
    )
    
    print("UMAP finished.")
    
    # Check if 'umap' layer exists
    # Note: The current implementation creates a NEW assay for UMAP result
    # It is named f"{assay_name}_{new_layer_name}"
    
    expected_assay_name = "RNA_umap"
    
    if expected_assay_name in container.assays:
        print(f"Assay '{expected_assay_name}' created successfully.")
        umap_assay = container.assays[expected_assay_name]
        if 'umap' in umap_assay.layers:
             print("UMAP layer found in the new assay.")
             print("Shape:", umap_assay.layers['umap'].X.shape)
        else:
             print("UMAP layer NOT found in the new assay.")
    else:
        print(f"Assay '{expected_assay_name}' NOT found.")

if __name__ == "__main__":
    test_umap()
