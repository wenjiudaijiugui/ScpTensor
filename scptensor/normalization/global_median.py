"""Global median normalization for ScpTensor."""

import numpy as np

from scptensor.core.exceptions import AssayNotFoundError, LayerNotFoundError
from scptensor.core.structures import ScpContainer, ScpMatrix


def global_median_normalization(
    container: ScpContainer,
    assay_name: str = "protein",
    base_layer_name: str = "raw",
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
    base_layer_name : str, default="raw"
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
    >>> result = global_median_normalization(container)
    >>> 'global_median_norm' in result.assays['protein'].layers
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
        params={"assay": assay_name},
        description=f"Global median normalization on layer '{base_layer_name}' -> '{new_layer_name}'.",
    )

    return container
