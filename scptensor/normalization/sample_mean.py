from typing import Optional
import numpy as np
from scptensor.core.structures import ScpContainer, ScpMatrix

def sample_mean_normalization(
    container: ScpContainer,
    assay_name: str = 'protein',
    base_layer_name: str = 'raw',
    new_layer_name: str = "sample_mean_norm"
) -> ScpContainer:
    """
    Sample mean normalization to eliminate systematic biases from loading differences.

    Mathematical Formulation:
        X_normalized = X - mean(X, axis=1)

    This method centers each sample around its mean value, which helps remove
    technical variation in sample loading amounts. It's more sensitive to outliers
    compared to median normalization.

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

    X = assay.layers[base_layer_name].X.copy()

    # Calculate sample-wise means and center
    means = np.nanmean(X, axis=1, keepdims=True)
    X_normalized = X - means

    new_matrix = ScpMatrix(X=X_normalized, M=assay.layers[base_layer_name].M.copy())

    container.assays[assay_name].add_layer(new_layer_name, new_matrix)

    container.log_operation(
        action="normalization_sample_mean",
        params={"assay": assay_name},
        description=f"Sample mean normalization on layer '{base_layer_name}' -> '{new_layer_name}'."
    )

    return container