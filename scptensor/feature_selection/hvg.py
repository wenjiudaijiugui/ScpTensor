from typing import Optional, Literal
import numpy as np
import polars as pl
from scptensor.core.structures import ScpContainer, Assay

def select_hvg(
    container: ScpContainer,
    assay_name: str = 'protein',
    layer: str = 'raw',
    n_top_features: int = 2000,
    method: Literal['cv', 'dispersion'] = 'cv',
    subset: bool = True
) -> ScpContainer:
    """
    Select Highly Variable Genes/Proteins (HVG).
    
    Args:
        container: The ScpContainer object.
        assay_name: Name of the assay.
        layer: Layer to use for calculation.
        n_top_features: Number of top features to select.
        method: Method to calculate variability ('cv' for Coefficient of Variation, 'dispersion').
        subset: If True, returns a container with only HVGs. If False, adds 'highly_variable' column to var.
        
    Returns:
        ScpContainer.
    """
    if assay_name not in container.assays:
        raise ValueError(f"Assay '{assay_name}' not found.")
        
    assay = container.assays[assay_name]
    if layer not in assay.layers:
        raise ValueError(f"Layer '{layer}' not found in assay '{assay_name}'.")
        
    X = assay.layers[layer].X
    
    mean = np.nanmean(X, axis=0)
    var = np.nanvar(X, axis=0)
    
    if method == 'cv':
        # Coefficient of Variation: std / mean
        # Add small epsilon to avoid division by zero
        score = np.sqrt(var) / (mean + 1e-9)
    else:
        # Dispersion: var / mean
        score = var / (mean + 1e-9)
        
    # Handle NaNs in score
    score = np.nan_to_num(score, nan=-np.inf)
    
    # Select top features
    # argpartition gets indices of k largest elements (not sorted)
    # We want top N, so we partition at len - N
    if n_top_features >= len(score):
        top_indices = np.arange(len(score))
    else:
        top_indices = np.argpartition(score, -n_top_features)[-n_top_features:]
    
    if subset:
        return container.filter_features(assay_name, top_indices)
    else:
        # Add annotation to var
        is_hvg = np.zeros(assay.n_features, dtype=bool)
        is_hvg[top_indices] = True
        
        new_var = assay.var.with_columns(
            pl.Series("highly_variable", is_hvg),
            pl.Series("variability_score", score)
        )
        
        # Create new assay with updated var
        new_assay = Assay(var=new_var, layers=assay.layers)
        
        new_assays = container.assays.copy()
        new_assays[assay_name] = new_assay
        
        new_container = ScpContainer(
            obs=container.obs,
            assays=new_assays,
            history=list(container.history)
        )
        
        new_container.log_operation(
            action="select_hvg",
            params={"n_top": n_top_features, "method": method, "subset": subset},
            description=f"Identified {n_top_features} highly variable features."
        )
        
        return new_container
