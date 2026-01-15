"""Median scaling normalization for ScpTensor."""

import warnings

from typing import overload

import numpy as np

from scptensor.core.exceptions import (
    AssayNotFoundError,
    LayerNotFoundError,
)
from scptensor.core.structures import ScpContainer, ScpMatrix


@overload
def norm_median_scale(
    container: ScpContainer,
    assay_name: str,
    source_layer: str,
    new_layer_name: str = "median_scaling",
) -> ScpContainer: ...


def norm_median_scale(
    container: ScpContainer,
    assay_name: str,
    source_layer: str,
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
    source_layer : str
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
    >>> result = norm_median_scale(container, 'protein', 'raw')
    >>> 'median_scaling' in result.assays['protein'].layers
    True
    """
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
        params={
            "assay": assay_name,
            "source_layer": source_layer,
            "new_layer_name": new_layer_name,
        },
        description=f"Median scaling on layer '{source_layer}' -> '{new_layer_name}'.",
    )

    return container


# Backward compatibility alias
def median_scaling(*args, **kwargs):
    """Deprecated: Use norm_median_scale instead.

    This function will be removed in version 1.0.0.
    """
    warnings.warn(
        "'median_scaling' is deprecated, use 'norm_median_scale' instead. "
        "This will be removed in version 1.0.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    return norm_median_scale(*args, **kwargs)
