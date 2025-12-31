from typing import List, Optional, Tuple
import numpy as np
import polars as pl
from scptensor.core.structures import ScpContainer, ScpMatrix, Assay

def basic_qc(
    container: ScpContainer,
    assay_name: str = 'protein',
    min_features: int = 200,
    min_cells: int = 3,
    detection_threshold: float = 0.0,
    new_layer_name: Optional[str] = None
) -> ScpContainer:
    """
    Perform basic Quality Control (QC) on samples and features.
    
    Args:
        container: The ScpContainer object.
        assay_name: Name of the assay to perform QC on.
        min_features: Minimum number of features required for a cell to be kept.
        min_cells: Minimum number of cells a feature must be detected in.
        detection_threshold: Threshold for a value to be considered detected (e.g., > 0).
        new_layer_name: (Not used in filtering, but kept for consistency)
        
    Returns:
        A new ScpContainer with filtered samples and features.
    """
    if assay_name not in container.assays:
        raise ValueError(f"Assay '{assay_name}' not found.")
    
    assay = container.assays[assay_name]
    layer = assay.layers.get('raw') or list(assay.layers.values())[0] # Use 'raw' or first layer
    X = layer.X
    
    # 1. Sample QC: Filter cells with too few features
    n_features_per_cell = np.sum(X > detection_threshold, axis=1)
    keep_samples_mask = n_features_per_cell >= min_features
    keep_samples_indices = np.where(keep_samples_mask)[0]
    
    container_filtered_samples = container.filter_samples(keep_samples_indices)
    
    # Update X and layer after sample filtering for feature filtering
    # We need to access the updated assay in the new container
    assay_filtered = container_filtered_samples.assays[assay_name]
    layer_filtered = assay_filtered.layers.get('raw') or list(assay_filtered.layers.values())[0]
    X_filtered = layer_filtered.X
    
    # 2. Feature QC: Filter features detected in too few cells
    n_cells_per_feature = np.sum(X_filtered > detection_threshold, axis=0)
    keep_features_mask = n_cells_per_feature >= min_cells
    keep_features_indices = np.where(keep_features_mask)[0]
    
    container_final = container_filtered_samples.filter_features(assay_name, keep_features_indices)
    
    # Log QC stats
    n_samples_removed = container.n_samples - container_final.n_samples
    n_features_removed = assay.n_features - container_final.assays[assay_name].n_features
    
    container_final.log_operation(
        action="basic_qc",
        params={
            "assay": assay_name,
            "min_features": min_features,
            "min_cells": min_cells
        },
        description=f"Removed {n_samples_removed} samples and {n_features_removed} features."
    )
    
    return container_final
