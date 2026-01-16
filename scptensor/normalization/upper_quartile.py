"""Upper quartile normalization for ScpTensor.

Reference:
    Bullard, J. H., Purdom, E., Hansen, K. D., & Dudoit, S. (2010).
    Evaluation of statistical methods for normalization and differential
    expression in mRNA-Seq experiments. BMC Bioinformatics, 11, 94.
"""

import numpy as np

from scptensor.core.exceptions import (
    AssayNotFoundError,
    LayerNotFoundError,
    ScpValueError,
)
from scptensor.core.structures import ScpContainer, ScpMatrix


def norm_quartile(
    container: ScpContainer,
    assay_name: str = "protein",
    source_layer: str = "raw",
    new_layer_name: str = "upper_quartile_norm",
    percentile: float = 0.75,
) -> ScpContainer:
    """
    Upper quartile normalization to align samples based on their 75th percentile values.

    This robust normalization method is less sensitive to outliers compared to
    mean normalization and more stable than median normalization for datasets
    with many zero values.

    Mathematical Formulation:
        UQ_i = percentile(X_i, percentile)
        UQ_global = percentile(X, percentile)
        scaling_factor_i = UQ_global / UQ_i
        X_normalized_i = X_i * scaling_factor_i

    Parameters
    ----------
    container : ScpContainer
        ScpContainer containing the data.
    assay_name : str, default="protein"
        Name of the assay to process.
    source_layer : str, default="raw"
        Name of the layer to normalize.
    new_layer_name : str, default="upper_quartile_norm"
        Name for the new normalized layer.
    percentile : float, default=0.75
        Percentile to use. Must be in (0, 1). Default is 0.75 for upper quartile.

    Returns
    -------
    ScpContainer
        ScpContainer with added upper quartile normalized layer.

    Raises
    ------
    ScpValueError
        If percentile parameter is invalid.
    AssayNotFoundError
        If the specified assay does not exist.
    LayerNotFoundError
        If the specified layer does not exist in the assay.

    Examples
    --------
    >>> import polars as pl
    >>> from scptensor.core.structures import ScpContainer, Assay, ScpMatrix
    >>> import numpy as np
    >>> container = ScpContainer(obs=pl.DataFrame({'_index': ['s1', 's2']}))
    >>> assay = Assay(var=pl.DataFrame({'_index': ['p1', 'p2', 'p3', 'p3']}))
    >>> assay.add_layer('raw', ScpMatrix(X=np.array([[1, 2, 3, 4], [5, 6, 7, 8]])))
    >>> container.add_assay('protein', assay)
    >>> result = norm_quartile(container)
    >>> 'upper_quartile_norm' in result.assays['protein'].layers
    True
    """
    # Validate parameters (guard clause pattern)
    if not (0 < percentile < 1):
        raise ScpValueError(
            f"Percentile must be in (0, 1), got {percentile}. "
            "Use 0.75 for upper quartile, 0.5 for median, or 0.25 for lower quartile.",
            parameter="percentile",
            value=percentile,
        )

    # Validate assay and layer existence
    if assay_name not in container.assays:
        available = ", ".join(f"'{k}'" for k in container.assays.keys())
        raise AssayNotFoundError(
            assay_name,
            hint=f"Available assays: {available}. Use container.list_assays() to see all assays.",
        )

    assay = container.assays[assay_name]
    if source_layer not in assay.layers:
        available = ", ".join(f"'{k}'" for k in assay.layers.keys())
        raise LayerNotFoundError(
            source_layer,
            assay_name,
            hint=f"Available layers in assay '{assay_name}': {available}. "
            f"Use assay.list_layers() to see all layers.",
        )

    input_layer = assay.layers[source_layer]
    X = input_layer.X.copy()

    # Calculate percentile for each sample (vectorized)
    sample_pcts = np.nanpercentile(X, percentile * 100, axis=1)

    # Calculate global percentile across all samples
    global_pct = np.nanpercentile(X, percentile * 100)

    # Compute scaling factors with zero-division safety
    scaling_factors = global_pct / sample_pcts
    scaling_factors[~np.isfinite(scaling_factors)] = 1.0

    # Apply scaling factors (broadcasting)
    X_normalized = X * scaling_factors[:, np.newaxis]

    # Create new layer
    new_matrix = ScpMatrix(
        X=X_normalized,
        M=input_layer.M.copy() if input_layer.M is not None else None,
    )
    assay.add_layer(new_layer_name, new_matrix)

    container.log_operation(
        action="normalization_upper_quartile",
        params={
            "assay": assay_name,
            "source_layer": source_layer,
            "new_layer_name": new_layer_name,
            "percentile": percentile,
        },
        description=f"Upper quartile normalization on layer '{source_layer}' -> '{new_layer_name}'.",
    )

    return container
