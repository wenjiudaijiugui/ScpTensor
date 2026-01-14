"""Sample mean normalization for ScpTensor."""

import numpy as np

from scptensor.core.exceptions import AssayNotFoundError, LayerNotFoundError
from scptensor.core.structures import ScpContainer, ScpMatrix


def sample_mean_normalization(
    container: ScpContainer,
    assay_name: str = "protein",
    base_layer_name: str = "raw",
    new_layer_name: str = "sample_mean_norm",
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
    base_layer_name : str, default="raw"
        Name of the layer to normalize.
    new_layer_name : str, default="sample_mean_norm"
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
    >>> result = sample_mean_normalization(container)
    >>> 'sample_mean_norm' in result.assays['protein'].layers
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
        params={"assay": assay_name},
        description=f"Sample mean normalization on layer '{base_layer_name}' -> '{new_layer_name}'.",
    )

    return container
