import numpy as np
import polars as pl
from scptensor.core.structures import ScpContainer
from scptensor.viz.base import scatter

def embedding(
    container: ScpContainer, 
    basis: str = 'umap', 
    color: str = None, 
    layer: str = 'imputed',
    show_missing: bool = False
):
    """
    Plot 2D embedding (PCA/UMAP).
    
    Args:
        container: ScpContainer.
        basis: Assay name for coordinates (e.g. 'umap', 'pca').
        color: Column name in obs (metadata) or Feature name (expression).
        layer: Layer to use if color is a Feature.
        show_missing: If True, use explicit markers for missing values.
    """
    # 1. Get Coordinates
    if basis not in container.assays:
        raise ValueError(f"Basis assay '{basis}' not found.")
    
    basis_assay = container.assays[basis]
    # Assume coordinates are in 'X' layer
    if 'X' not in basis_assay.layers:
         # Try 'raw' or first available if X not found
         coord_layer = list(basis_assay.layers.keys())[0]
         coords = basis_assay.layers[coord_layer].X
    else:
         coords = basis_assay.layers['X'].X
         
    if coords.shape[1] < 2:
        raise ValueError(f"Basis '{basis}' has less than 2 dimensions.")
        
    X_plot = coords[:, :2]
    
    # 2. Determine Color and Mask
    c = None
    m = None
    title = f"{basis} colored by {color}" if color else basis
    
    if color:
        # Check if color is in obs (Metadata)
        if color in container.obs.columns:
            # Handle categorical strings -> factorize or map to colors
            # For simplicity, let matplotlib handle it if it's numeric, 
            # or we need to encode it if it's string.
            col_data = container.obs[color]
            
            if col_data.dtype == pl.String or col_data.dtype == pl.Categorical:
                # Map strings to integers/colors
                # A simple way is to use categorical codes
                # Or pass raw strings if scatter supports it (it usually doesn't nicely with arrays)
                # We factorize it
                c, uniques = col_data.to_numpy(), np.unique(col_data.to_numpy())
                # For matplotlib scatter 'c', we need numbers or RGBA. 
                # Let's map to numbers and let cmap handle it, or use seaborn externally?
                # But we are in primitives. 
                # Simple Hack: Factorize
                categories, c = np.unique(col_data.to_numpy(), return_inverse=True)
                # We might want a discrete cmap
            else:
                c = col_data.to_numpy()
                
            # Metadata implies valid everywhere usually, unless obs has NaNs
            m = np.zeros(container.n_samples, dtype=int)
            
        else:
            # Check if color is a Feature (Protein)
            # Search in all assays or default 'protein'
            # Assuming 'protein' assay for expression
            target_assay_name = 'protein' # Could be parameterized
            if target_assay_name in container.assays:
                target_assay = container.assays[target_assay_name]
                # Check if color is in var index/columns?
                # var structure: protein_id might be a column or we assume row order
                # We need to find the index of the feature named `color`
                
                # Assume 'protein_id' column exists or similar. 
                # Let's search in var columns for a match or specific ID column
                # For now, assume user passes an index or ID that matches 'protein_id'
                
                feature_idx = None
                
                # Try to find in 'protein_id' column
                if 'protein_id' in target_assay.var.columns:
                    ids = target_assay.var['protein_id'].to_list()
                    if color in ids:
                        feature_idx = ids.index(color)
                        
                if feature_idx is None:
                     raise ValueError(f"Color '{color}' not found in obs or {target_assay_name} features.")
                     
                # Get Expression and Mask
                matrix = target_assay.layers[layer]
                c = matrix.X[:, feature_idx]
                m = matrix.M[:, feature_idx]
            else:
                raise ValueError(f"Color '{color}' not found in obs and assay '{target_assay_name}' missing.")
                
    mask_style = 'explicit' if show_missing else 'subtle'
    
    return scatter(
        X=X_plot,
        c=c,
        m=m,
        mask_style=mask_style,
        title=title,
        xlabel=f"{basis}_1",
        ylabel=f"{basis}_2",
        cmap='viridis' if c is not None and np.issubdtype(c.dtype, np.number) else 'tab10'
    )
