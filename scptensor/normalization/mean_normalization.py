"""Sample mean normalization for ScpTensor.

This module provides mean-based normalization to remove sample-specific bias.

Mathematical Formulation:
    Centering Mode (add_global_mean=False):

    .. math::

        X_{centered} = X - \\bar{X}_{row}

    where :math:`\\bar{X}_{row}` is the row-wise mean.

    Scaling Mode (add_global_mean=True):

    .. math::

        X_{normalized} = X - \\bar{X}_{row} + \\bar{X}_{global}

    where :math:`\\bar{X}_{global}` is the global mean across all values.

Reference:
    Mean normalization is a simple centering technique to remove
    sample-specific technical bias while preserving relative differences
    between features within each sample.
"""

import numpy as np

from scptensor.core.structures import ScpContainer

from .base import (
    ensure_dense,
    finalize_normalization_layer,
    get_layer_name,
    validate_assay_and_layer,
)


def norm_mean(
    container: ScpContainer,
    assay_name: str = "protein",
    source_layer: str = "raw",
    new_layer_name: str | None = "sample_mean_norm",
    add_global_mean: bool = False,
) -> ScpContainer:
    """Subtract the mean of each sample with optional global mean restoration.

    This method provides two modes for mean-based normalization:

    1. Centering Mode (add_global_mean=False):
        Centers each sample around its mean value, removing sample-specific
        bias while preserving relative differences between features.

        Mathematical Formulation:
            .. math::

                X_{centered} = X - \\bar{X}_{row}

        Output has mean ~0 for each sample.

    2. Scaling Mode (add_global_mean=True):
        Centers each sample around its mean, then adds back the global mean
        across all samples. This aligns samples to a common reference point.

        Mathematical Formulation:
            .. math::

                X_{normalized} = X - \\bar{X}_{row} + \\bar{X}_{global}

        Output has mean approximately equal to the global mean.

    **Note:** Mean normalization is more sensitive to outliers than median
    normalization. Use norm_median if your data contains extreme outliers.

    Parameters
    ----------
    container : ScpContainer
        ScpContainer containing the data.
    assay_name : str, default="protein"
        Name of the assay to process.
    source_layer : str, default="raw"
        Name of the layer to normalize.
    new_layer_name : Optional[str], default="sample_mean_norm"
        Name for the new normalized layer. None uses default.
    add_global_mean : bool, default=False
        If True, add global mean after centering (scaling mode).
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
    >>> result = norm_mean(container)
    >>> 'sample_mean_norm' in result.assays['protein'].layers
    True
    >>> # Scaling mode: add global mean back
    >>> result2 = norm_mean(container, add_global_mean=True)

    Notes
    -----
    - Mean normalization assumes data is roughly symmetric.
    - For skewed data or data with outliers, use median normalization instead.
    - Centering mode is useful for removing technical bias between samples.
    """
    # Validate and get objects
    assay, input_layer = validate_assay_and_layer(container, assay_name, source_layer)
    x_dense = ensure_dense(input_layer.X)

    # Apply mean normalization
    X_centered = x_dense - np.nanmean(x_dense, axis=1, keepdims=True)
    if add_global_mean:
        X_centered = X_centered + np.nanmean(x_dense)

    # Get layer name
    layer_name = get_layer_name(new_layer_name, "sample_mean_norm")

    return finalize_normalization_layer(
        container,
        assay,
        input_layer,
        X=X_centered,
        new_layer_name=layer_name,
        action="normalization_sample_mean",
        params={
            "assay": assay_name,
            "source_layer": source_layer,
            "new_layer_name": layer_name,
            "add_global_mean": add_global_mean,
        },
        description=f"Sample mean normalization on layer '{source_layer}' -> '{layer_name}' "
        f"({'scaling mode' if add_global_mean else 'centering mode'}).",
    )
