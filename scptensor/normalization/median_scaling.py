"""Median scaling normalization for ScpTensor."""

import numpy as np

from scptensor.core.exceptions import AssayNotFoundError, LayerNotFoundError
from scptensor.core.structures import ScpContainer, ScpMatrix


def median_scaling(
    container: ScpContainer,
    assay_name: str,
    base_layer_name: str,
    new_layer_name: str = "median_scaling",
) -> ScpContainer:
    """
    Apply median scaling normalization to align samples.

    This method aligns all samples to the global median by subtracting
    sample-specific median bias.

    Mathematical Formulation:
        bias = median(X, axis=1) - global_median(X)
        X_scaled = X - bias

    Parameters
    ----------
    container : ScpContainer
        ScpContainer containing the data.
    assay_name : str
        Name of the assay to process.
    base_layer_name : str
        Name of the layer to normalize.
    new_layer_name : str, default="median_scaling"
        Name for the new normalized layer.

    Returns
    -------
    ScpContainer
        ScpContainer with added normalized layer.

    Raises
    ------
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
    >>> assay = Assay(var=pl.DataFrame({'_index': ['p1', 'p2']}))
    >>> assay.add_layer('raw', ScpMatrix(X=np.array([[1, 2], [3, 4]])))
    >>> container.add_assay('protein', assay)
    >>> result = median_scaling(container, 'protein', 'raw')
    >>> 'median_scaling' in result.assays['protein'].layers
    True
    """
    # Validate assay and layer existence
    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]
    if base_layer_name not in assay.layers:
        raise LayerNotFoundError(base_layer_name, assay_name)

    input_layer = assay.layers[base_layer_name]
    X = input_layer.X

    # Compute and apply median scaling (vectorized)
    global_median = np.nanmedian(X)
    bias = np.nanmedian(X, axis=1, keepdims=True) - global_median
    X_scaled = X - bias

    # Create new layer
    new_matrix = ScpMatrix(
        X=X_scaled,
        M=input_layer.M.copy() if input_layer.M is not None else None,
    )
    assay.add_layer(new_layer_name, new_matrix)

    container.log_operation(
        action="normalization_median_scaling",
        params={"assay": assay_name},
        description=f"Median scaling on layer '{base_layer_name}' -> '{new_layer_name}'.",
    )

    return container
