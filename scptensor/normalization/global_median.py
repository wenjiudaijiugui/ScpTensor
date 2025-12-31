from typing import Optional
import numpy as np
from scptensor.core.structures import ScpContainer, ScpMatrix

def global_median_normalization(
    container: ScpContainer,
    assay_name: str = 'protein',
    base_layer_name: str = 'raw',
    new_layer_name: str = "global_median_norm"
) -> ScpContainer:
    """
    Global median normalization to align all samples to the global median.

    Mathematical Formulation:
        bias = median(X, axis=1) - global_median(X)
        X_normalized = X - bias

    This method ensures all samples have the same median value (global median),
    which helps remove systematic technical variation while preserving biological
    differences between samples.

    Args:
        container: ScpContainer containing the data
        assay_name: Name of the assay to process
        base_layer_name: Name of the layer to normalize
        new_layer_name: Name for the new normalized layer

    Returns:
        ScpContainer with added normalized layer
    """
    if assay_name not in container.assays:
        raise ValueError(f"Assay '{assay_name}' not found.")

    assay = container.assays[assay_name]
    if base_layer_name not in assay.layers:
        raise ValueError(f"Layer '{base_layer_name}' not found in assay '{assay_name}'.")

    base_layer = assay.layers[base_layer_name]
    X = base_layer.X

    # Calculate global median and sample-wise biases
    global_median = np.nanmedian(X)
    sample_medians = np.nanmedian(X, axis=1, keepdims=True)
    bias = sample_medians - global_median
    X_normalized = X - bias

    new_layer = ScpMatrix(X_normalized, base_layer.M.copy())
    container.assays[assay_name].add_layer(new_layer_name, new_layer)

    container.log_operation(
        action="normalization_global_median",
        params={"assay": assay_name},
        description=f"Global median normalization on layer '{base_layer_name}' -> '{new_layer_name}'."
    )

    return container