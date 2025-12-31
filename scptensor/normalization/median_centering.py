from typing import Optional
import numpy as np
from scptensor.core.structures import ScpContainer, ScpMatrix

def median_centering(
    container: ScpContainer,
    assay_name: str = 'protein',
    base_layer_name: str = 'raw',
    new_layer_name: Optional[str] = 'median_centered'
) -> ScpContainer:
    """
    Subtract the median of each sample.
    """
    if assay_name not in container.assays:
        raise ValueError(f"Assay '{assay_name}' not found.")
    
    assay = container.assays[assay_name]
    if base_layer_name not in assay.layers:
         raise ValueError(f"Layer '{base_layer_name}' not found in assay '{assay_name}'.")

    X = assay.layers[base_layer_name].X.copy()
    
    medians = np.nanmedian(X, axis=1, keepdims=True)
    X_centered = X - medians
    
    new_matrix = ScpMatrix(X=X_centered, M=assay.layers[base_layer_name].M.copy())
    
    container.assays[assay_name].add_layer(new_layer_name, new_matrix)
    
    container.log_operation(
        action="normalization_median_centering",
        params={"assay": assay_name},
        description=f"Median centering on layer '{base_layer_name}' -> '{new_layer_name}'."
    )
    
    return container


