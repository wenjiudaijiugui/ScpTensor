from typing import Optional
import numpy as np
from scptensor.core.structures import ScpContainer, ScpMatrix


def median_scaling(
    container: ScpContainer,
    assay_name: str,
    base_layer_name: str,
    new_layer_name: str = "median_scaling"
) -> ScpContainer:
    if assay_name not in container.assays:
        raise ValueError(f"Assay '{assay_name}' not found.")
    assay = container.assays[assay_name]
    if base_layer_name not in assay.layers:
        raise ValueError(f"Layer '{base_layer_name}' not found in assay '{assay_name}'.")
    base_layer = assay.layers[base_layer_name]
    X = base_layer.X
    global_med = np.nanmedian(X)
    bias = np.nanmedian(X, axis=1, keepdims=True)-global_med
    X_centered = X-bias
    new_layer= ScpMatrix(X_centered, base_layer.M.copy())
    container.assays[assay_name].add_layer(new_layer_name, new_layer)
    container.log_operation(
        action="normalization_median_scaling",
        params={"assay": assay_name},
        description=f"Median scaling on layer '{base_layer_name}' -> '{new_layer_name}'."
    )
    return container