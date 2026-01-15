"""Sample mean normalization for ScpTensor."""

import warnings
from typing import overload

import numpy as np

from scptensor.core.exceptions import (
    AssayNotFoundError,
    LayerNotFoundError,
)
from scptensor.core.structures import ScpContainer, ScpMatrix


@overload
def sample_mean_normalization(
    container: ScpContainer,
    assay_name: str = "protein",
    source_layer: str = "raw",
    new_layer_name: str = "sample_mean_norm",
    base_layer_name: str | None = None,
) -> ScpContainer: ...


def sample_mean_normalization(
    container: ScpContainer,
    assay_name: str = "protein",
    source_layer: str = "raw",
    new_layer_name: str = "sample_mean_norm",
    base_layer_name: str | None = None,
) -> ScpContainer:
    """
    Sample mean normalization to eliminate systematic biases from loading differences.

    This method centers each sample around its mean value, which helps remove
    technical variation in sample loading amounts. It's more sensitive to outliers
    compared to median normalization.

    Mathematical Formulation:
        X_normalized = X - mean(X, axis=1, keepdims=True)

    Parameters
    ----------
    container : ScpContainer
        ScpContainer containing the data.
    assay_name : str, default="protein"
        Name of the assay to process.
    source_layer : str, default="raw"
        Name of the layer to normalize.
    new_layer_name : str, default="sample_mean_norm"
        Name for the new normalized layer.
    base_layer_name : str, optional
        .. deprecated:: 0.2.0
            Use ``source_layer`` instead. Will be removed in version 1.0.0.

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
    >>> result = sample_mean_normalization(container)
    >>> 'sample_mean_norm' in result.assays['protein'].layers
    True
    """
    # Handle deprecated parameter name
    if base_layer_name is not None:
        warnings.warn(
            "'base_layer_name' is deprecated, use 'source_layer' instead. "
            "This will be removed in version 1.0.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        source_layer = base_layer_name

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

    # Calculate and subtract sample-wise means (vectorized)
    means = np.nanmean(X, axis=1, keepdims=True)
    X_normalized = X - means

    # Create new layer
    new_matrix = ScpMatrix(
        X=X_normalized,
        M=input_layer.M.copy() if input_layer.M is not None else None,
    )
    assay.add_layer(new_layer_name, new_matrix)

    container.log_operation(
        action="normalization_sample_mean",
        params={
            "assay": assay_name,
            "source_layer": source_layer,
            "new_layer_name": new_layer_name,
        },
        description=f"Sample mean normalization on layer '{source_layer}' -> '{new_layer_name}'.",
    )

    return container
