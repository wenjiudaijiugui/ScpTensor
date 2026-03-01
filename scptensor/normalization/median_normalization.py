"""Median centering normalization for ScpTensor.

This module provides median-based normalization to remove sample-specific bias.

Mathematical Formulation:
    Centering Mode (add_global_median=False):

    .. math::

        X_{centered} = X - \\tilde{X}_{row}

    where :math:`\\tilde{X}_{row}` is the row-wise median.

    Scaling Mode (add_global_median=True):

    .. math::

        X_{normalized} = X - \\tilde{X}_{row} + \\tilde{X}_{global}

    where :math:`\\tilde{X}_{global}` is the global median.

Reference:
    Median normalization is robust to outliers and provides a stable
    centering method for proteomics data. It is preferred over mean
    normalization when the data contains extreme values or heavy tails.
"""

import numpy as np

from scptensor.core.structures import ScpContainer

from .base import (
    create_result_layer,
    get_layer_name,
    log_operation,
    validate_assay_and_layer,
)


def norm_median(
    container: ScpContainer,
    assay_name: str = "protein",
    source_layer: str = "raw",
    new_layer_name: str | None = "median_centered",
    add_global_median: bool = False,
) -> ScpContainer:
    """Subtract the median of each sample with optional global median restoration.

    This method provides two modes for median-based normalization:

    1. Centering Mode (add_global_median=False):
        Centers each sample around its median value, removing sample-specific
        bias while preserving relative differences between features.

        Mathematical Formulation:
            .. math::

                X_{centered} = X - \\tilde{X}_{row}

        Output has median ~0 for each sample.

    2. Scaling Mode (add_global_median=True):
        Centers each sample around its median, then adds back the global median
        across all samples. This aligns samples to a common reference point.

        Mathematical Formulation:
            .. math::

                X_{normalized} = X - \\tilde{X}_{row} + \\tilde{X}_{global}

        Output has median approximately equal to the global median.

    **Note:** Median normalization is more robust to outliers than mean
    normalization (norm_mean). It is recommended for proteomics data with
    extreme values or heavy-tailed distributions.

    Parameters
    ----------
    container : ScpContainer
        ScpContainer containing the data.
    assay_name : str, default="protein"
        Name of the assay to process.
    source_layer : str, default="raw"
        Name of the layer to normalize.
    new_layer_name : Optional[str], default="median_centered"
        Name for the new normalized layer. None uses default.
    add_global_median : bool, default=False
        If True, add global median after centering (scaling mode).
        If False, only center each sample (centering mode).

    Returns
    -------
    ScpContainer
        Container with added normalized layer.

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
    >>> result = norm_median(container)
    >>> 'median_centered' in result.assays['protein'].layers
    True
    >>> # Scaling mode: add global median back
    >>> result2 = norm_median(container, add_global_median=True)

    Notes
    -----
    - Median normalization is robust to outliers.
    - Recommended for proteomics data with skewed distributions.
    - The median is less affected by extreme values than the mean.
    """
    # Validate and get objects
    assay, input_layer = validate_assay_and_layer(container, assay_name, source_layer)

    # Apply median normalization
    X_centered = input_layer.X - np.nanmedian(input_layer.X, axis=1, keepdims=True)
    if add_global_median:
        X_centered = X_centered + np.nanmedian(input_layer.X)

    # Get layer name
    layer_name = get_layer_name(new_layer_name, "median_centered")

    # Create and add new layer
    new_matrix = create_result_layer(X_centered, input_layer)
    assay.add_layer(layer_name, new_matrix)

    # Log operation
    log_operation(
        container,
        action="normalization_median_centering",
        params={
            "assay": assay_name,
            "source_layer": source_layer,
            "new_layer_name": layer_name,
            "add_global_median": add_global_median,
        },
        description=f"Median centering on layer '{source_layer}' -> '{layer_name}' "
        f"({'scaling mode' if add_global_median else 'centering mode'}).",
    )

    return container
