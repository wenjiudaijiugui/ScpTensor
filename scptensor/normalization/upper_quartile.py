from typing import Optional
import numpy as np
from scptensor.core.structures import ScpContainer, ScpMatrix

def upper_quartile_normalization(
    container: ScpContainer,
    assay_name: str = 'protein',
    base_layer_name: str = 'raw',
    new_layer_name: str = "upper_quartile_norm",
    percentile: float = 0.75
) -> ScpContainer:
    """
    Upper quartile normalization to align samples based on their 75th percentile values.

    This robust normalization method is less sensitive to outliers compared to mean normalization
    and more stable than median normalization for datasets with many zero values.

    Mathematical Formulation:
        UQ_i = percentile(X_i, percentile)
        UQ_global = percentile(X, percentile)
        scaling_factor_i = UQ_global / UQ_i
        X_normalized_i = X_i * scaling_factor_i

    Reference:
        Bullard, J. H., Purdom, E., Hansen, K. D., & Dudoit, S. (2010).
        Evaluation of statistical methods for normalization and differential expression
        in mRNA-Seq experiments. BMC Bioinformatics, 11, 94.

    Args:
        container: ScpContainer containing the data
        assay_name: Name of the assay to process
        base_layer_name: Name of the layer to normalize
        new_layer_name: Name for the new normalized layer
        percentile: Percentile to use (default 0.75 for upper quartile)

    Returns:
        ScpContainer with added upper quartile normalized layer
    """
    if assay_name not in container.assays:
        raise ValueError(f"Assay '{assay_name}' not found.")

    assay = container.assays[assay_name]
    if base_layer_name not in assay.layers:
        raise ValueError(f"Layer '{base_layer_name}' not found in assay '{assay_name}'.")

    X = assay.layers[base_layer_name].X.copy()

    # Calculate upper quartile for each sample
    sample_uqs = np.nanpercentile(X, percentile * 100, axis=1)

    # Calculate global upper quartile across all samples
    global_uq = np.nanpercentile(X, percentile * 100)

    # Calculate scaling factors
    scaling_factors = global_uq / sample_uqs

    # Handle cases where upper quartile is zero
    scaling_factors[~np.isfinite(scaling_factors)] = 1.0

    # Apply scaling factors
    X_normalized = X * scaling_factors[:, np.newaxis]

    new_matrix = ScpMatrix(X=X_normalized, M=assay.layers[base_layer_name].M.copy())
    container.assays[assay_name].add_layer(new_layer_name, new_matrix)

    container.log_operation(
        action="normalization_upper_quartile",
        params={
            "assay": assay_name,
            "percentile": percentile
        },
        description=f"Upper quartile normalization on layer '{base_layer_name}' -> '{new_layer_name}'."
    )

    return container