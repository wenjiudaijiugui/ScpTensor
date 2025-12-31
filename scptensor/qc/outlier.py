from typing import List, Optional
import numpy as np
from sklearn.ensemble import IsolationForest
from scptensor.core.structures import ScpContainer

def detect_outliers(
    container: ScpContainer,
    assay_name: str = 'protein',
    layer: str = 'raw',
    contamination: float = 0.05,
    random_state: int = 42
) -> ScpContainer:
    """
    Detect outlier samples using Isolation Forest.
    
    Args:
        container: The ScpContainer object.
        assay_name: Name of the assay to use.
        layer: Layer to use for detection.
        contamination: The amount of contamination of the data set, i.e. the proportion of outliers in the data set.
        random_state: Random state for reproducibility.
        
    Returns:
        ScpContainer with an added column 'is_outlier' in obs.
    """
    if assay_name not in container.assays:
        raise ValueError(f"Assay '{assay_name}' not found.")
        
    assay = container.assays[assay_name]
    if layer not in assay.layers:
        raise ValueError(f"Layer '{layer}' not found in assay '{assay_name}'.")
        
    X = assay.layers[layer].X
    
    # Handle NaN values if any (simple imputation for detection)
    if np.isnan(X).any():
         # Simple mean imputation for IsolationForest
        col_mean = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X_imputed = X.copy()
        X_imputed[inds] = np.take(col_mean, inds[1])
        data_to_fit = X_imputed
    else:
        data_to_fit = X

    clf = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=-1)
    preds = clf.fit_predict(data_to_fit) # 1 for inlier, -1 for outlier
    
    is_outlier = preds == -1
    
    # Add to obs
    new_obs = container.obs.with_columns(
        pl.Series("is_outlier", is_outlier)
    )
    
    # Create new container
    new_container = ScpContainer(
        obs=new_obs,
        assays=container.assays,
        history=list(container.history)
    )
    
    new_container.log_operation(
        action="detect_outliers",
        params={"contamination": contamination, "method": "IsolationForest"},
        description=f"Detected {sum(is_outlier)} outliers."
    )
    
    return new_container
