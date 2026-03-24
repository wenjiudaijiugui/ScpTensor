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

from scptensor.core._layer_processing import (
    ensure_dense_matrix,
    resolve_result_layer_name,
    write_result_layer_and_log,
)
from scptensor.core._structure_container import ScpContainer
from scptensor.core.exceptions import ValidationError

from ._context import resolve_normalization_context


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
    ctx = resolve_normalization_context(container, assay_name, source_layer)
    assay = ctx.assay
    input_layer = ctx.layer
    x_dense = np.asarray(ensure_dense_matrix(input_layer.X), dtype=float)
    if np.any(np.isinf(x_dense)):
        raise ValidationError(
            "Median normalization does not accept Inf values. "
            "Use NaN for missing entries or clean infinite intensities first.",
            field="X",
        )
    row_medians = np.nanmedian(x_dense, axis=1, keepdims=True)
    global_median = float(np.nanmedian(x_dense)) if add_global_median else 0.0

    # Densify is part of the stable contract for this method, but keep the
    # dense working set to one output buffer instead of chaining temporaries.
    X_centered = np.array(x_dense, dtype=float, copy=True)
    np.subtract(X_centered, row_medians, out=X_centered)
    if add_global_median:
        np.add(X_centered, global_median, out=X_centered)

    layer_name = resolve_result_layer_name(new_layer_name, "median_centered")

    return write_result_layer_and_log(
        container,
        assay,
        source_layer=input_layer,
        layer_name=layer_name,
        x=X_centered,
        action="normalization_median_centering",
        params={
            "assay": ctx.resolved_assay_name,
            "source_layer": source_layer,
            "new_layer_name": layer_name,
            "add_global_median": add_global_median,
        },
        description=f"Median centering on layer '{source_layer}' -> '{layer_name}' "
        f"({'scaling mode' if add_global_median else 'centering mode'}).",
    )
