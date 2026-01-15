"""Median centering normalization for ScpTensor."""

import numpy as np

from scptensor.core.exceptions import AssayNotFoundError, LayerNotFoundError
from scptensor.core.structures import ScpContainer, ScpMatrix


def median_centering(
    container: ScpContainer,
    assay_name: str = "protein",
    base_layer_name: str = "raw",
    new_layer_name: str | None = "median_centered",
) -> ScpContainer:
    """
    Subtract the median of each sample.

    This method centers each sample around its median value, removing
    sample-specific bias while preserving relative differences between features.

    Mathematical Formulation:
        X_centered = X - median(X, axis=1, keepdims=True)

    Parameters
    ----------
    container : ScpContainer
        ScpContainer containing the data.
    assay_name : str, default="protein"
        Name of the assay to process.
    base_layer_name : str, default="raw"
        Name of the layer to normalize.
    new_layer_name : Optional[str], default="median_centered"
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
    >>> result = median_centering(container)
    >>> 'median_centered' in result.assays['protein'].layers
    True
    """
    # Validate assay and layer existence
    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]
    if base_layer_name not in assay.layers:
        raise LayerNotFoundError(base_layer_name, assay_name)

    input_layer = assay.layers[base_layer_name]
    X = input_layer.X.copy()

    # Compute and subtract sample-wise medians (vectorized)
    medians = np.nanmedian(X, axis=1, keepdims=True)
    X_centered = X - medians

    # Resolve layer name
    layer_name = new_layer_name or "median_centered"

    # Create new layer
    new_matrix = ScpMatrix(
        X=X_centered,
        M=input_layer.M.copy() if input_layer.M is not None else None,
    )
    assay.add_layer(layer_name, new_matrix)

    container.log_operation(
        action="normalization_median_centering",
        params={"assay": assay_name},
        description=f"Median centering on layer '{base_layer_name}' -> '{layer_name}'.",
    )

    return container
