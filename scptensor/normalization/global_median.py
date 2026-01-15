"""Global median normalization for ScpTensor."""

import warnings

from typing import overload

import numpy as np

from scptensor.core.exceptions import (
    AssayNotFoundError,
    LayerNotFoundError,
)
from scptensor.core.structures import ScpContainer, ScpMatrix


@overload
def norm_global_median(
    container: ScpContainer,
    assay_name: str = "protein",
    source_layer: str = "raw",
    new_layer_name: str = "global_median_norm",
) -> ScpContainer: ...


def norm_global_median(
    container: ScpContainer,
    assay_name: str = "protein",
    source_layer: str = "raw",
    new_layer_name: str = "global_median_norm",
) -> ScpContainer:
    """
    Global median normalization to align all samples to the global median.

    This method ensures all samples have the same median value (global median),
    which helps remove systematic technical variation while preserving biological
    differences between samples.

    Mathematical Formulation:
        bias = median(X, axis=1) - global_median(X)
        X_normalized = X - bias

    Parameters
    ----------
    container : ScpContainer
        ScpContainer containing the data.
    assay_name : str, default="protein"
        Name of the assay to process.
    source_layer : str, default="raw"
        Name of the layer to normalize.
    new_layer_name : str, default="global_median_norm"
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
    >>> result = norm_global_median(container)
    >>> 'global_median_norm' in result.assays['protein'].layers
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

    # Compute global median and sample-wise biases (vectorized)
    global_median = np.nanmedian(X)
    sample_medians = np.nanmedian(X, axis=1, keepdims=True)
    X_normalized = X - (sample_medians - global_median)

    # Create new layer
    new_matrix = ScpMatrix(
        X=X_normalized,
        M=input_layer.M.copy() if input_layer.M is not None else None,
    )
    assay.add_layer(new_layer_name, new_matrix)

    container.log_operation(
        action="normalization_global_median",
        params={
            "assay": assay_name,
            "source_layer": source_layer,
            "new_layer_name": new_layer_name,
        },
        description=f"Global median normalization on layer '{source_layer}' -> '{new_layer_name}'.",
    )

    return container


# Backward compatibility alias
def global_median_normalization(container, assay_name="protein", base_layer_name=None,
                                 source_layer="raw", new_layer_name="global_median_norm"):
    """Deprecated: Use norm_global_median instead.

    This function will be removed in version 1.0.0.

    Notes
    -----
    For backward compatibility, accepts both 'base_layer_name' (deprecated) and
    'source_layer' (new parameter name). If 'source_layer' is default, 'base_layer_name' takes precedence.
    """
    warnings.warn(
        "'global_median_normalization' is deprecated, use 'norm_global_median' instead. "
        "This will be removed in version 1.0.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Handle parameter name change for backward compatibility
    if base_layer_name is not None and source_layer == "raw":
        source_layer = base_layer_name
    return norm_global_median(container, assay_name, source_layer, new_layer_name)
