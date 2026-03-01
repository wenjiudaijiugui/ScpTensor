"""Logarithmic transformation for ScpTensor.

This module provides log transformation with configurable base and offset.
This is a preprocessing step, not a normalization method per se.

Mathematical Formulation:
    The log transformation is defined as:

    .. math::

        X_{log} = \\frac{\\log(X + c)}{\\log(b)}

    where:
    - :math:`X` is the input data matrix
    - :math:`b` is the logarithm base (default: 2.0)
    - :math:`c` is the offset to handle zeros (default: 1.0)

    For :math:`b = 2` (default), this computes :math:`\\log_2(X + c)`.
    For :math:`b = e`, this computes the natural log :math:`\\ln(X + c)`.

Reference:
    Log transformation is a standard preprocessing technique in
    proteomics to reduce skewness and make data more normally distributed.
"""

import warnings

import numpy as np

from scptensor.core.exceptions import ScpValueError
from scptensor.core.sparse_utils import (
    ensure_sparse_format,
    is_sparse_matrix,
    sparse_safe_log1p_with_scale,
)
from scptensor.core.structures import ScpContainer

from .base import (
    create_result_layer,
    validate_assay_and_layer,
)


def log_transform(
    container: ScpContainer,
    assay_name: str = "protein",
    source_layer: str = "raw",
    new_layer_name: str = "log",
    base: float = 2.0,
    offset: float = 1.0,
    use_jit: bool = True,
) -> ScpContainer:
    """Apply logarithmic transformation with configurable base and offset.

    This is a data preprocessing step that applies log transformation to reduce
    skewness and make the data more normally distributed. It is NOT a normalization
    method in the strict sense, but a preprocessing transformation.

    Mathematical Formulation:
        .. math::

            X_{log} = \\frac{\\log(X + c)}{\\log(b)}

        where :math:`b` is the base and :math:`c` is the offset.

    Supports both dense and sparse matrices. For sparse matrices, the log
    transformation is applied only to non-zero elements, preserving sparsity.

    Parameters
    ----------
    container : ScpContainer
        Input container with data to transform.
    assay_name : str, default="protein"
        Name of the assay to transform.
    source_layer : str, default="raw"
        Name of the layer to use as input.
    new_layer_name : str, default="log"
        Name for the new transformed layer.
    base : float, default=2.0
        Logarithm base (e.g., 2.0 for log2, np.e for natural log, 10.0 for log10).
    offset : float, default=1.0
        Offset to add before logging to handle zeros and negative values.
        Must be non-negative.
    use_jit : bool, default=True
        Whether to use JIT acceleration for large sparse matrices.

    Returns
    -------
    ScpContainer
        Container with added log-transformed layer.

    Raises
    ------
    ScpValueError
        If base or offset parameters are invalid.
    AssayNotFoundError
        If the specified assay does not exist.
    LayerNotFoundError
        If the specified layer does not exist in the assay.

    Examples
    --------
    >>> import polars as pl
    >>> from scptensor.core.structures import ScpContainer, Assay, ScpMatrix
    >>> container = ScpContainer(obs=pl.DataFrame({'_index': ['s1', 's2']}))
    >>> assay = Assay(var=pl.DataFrame({'_index': ['p1', 'p2']}))
    >>> assay.add_layer('raw', ScpMatrix(X=np.array([[1, 2], [3, 4]])))
    >>> container.add_assay('protein', assay)
    >>> result = log_transform(container, base=2.0)
    >>> 'log' in result.assays['protein'].layers
    True

    Notes
    -----
    - Log2 transformation is most common in proteomics (base=2.0).
    - For counts data, consider using CPM normalization before log transform.
    - Offset=1.0 is standard to handle zero values in intensity data.
    """
    # Validate parameters
    if base <= 0:
        raise ScpValueError(
            f"Log base must be positive, got {base}. "
            "Use base=2.0 for log2, base=10.0 for log10, or base=np.e for natural log.",
            parameter="base",
            value=base,
        )
    if offset < 0:
        raise ScpValueError(
            f"Offset must be non-negative, got {offset}. "
            "Offset is added before taking the log to handle zero values.",
            parameter="offset",
            value=offset,
        )

    # Validate and get assay/layer
    assay, input_layer = validate_assay_and_layer(container, assay_name, source_layer)

    X = input_layer.X

    # Apply log transformation: log_b(X + offset) = ln(X + offset) / ln(b)
    log_scale = np.log(base)

    # Check for negative values and clip to 0
    if is_sparse_matrix(X):
        # For sparse matrices, check only non-zero elements
        if np.any(X.data < 0):
            min_val = np.nanmin(X.data)
            warnings.warn(
                f"Input contains negative values (min={min_val:.4f}). "
                f"These will be clipped to 0 before log transform.",
                UserWarning,
                stacklevel=2,
            )
            X = X.copy()
            X.data = np.maximum(X.data, 0)
    else:
        # For dense matrices, check entire array
        if np.any(X < 0):
            min_val = np.nanmin(X)
            warnings.warn(
                f"Input contains negative values (min={min_val:.4f}). "
                f"These will be clipped to 0 before log transform.",
                UserWarning,
                stacklevel=2,
            )
            X = np.maximum(X, 0)

    if is_sparse_matrix(X):
        X_log = sparse_safe_log1p_with_scale(X, offset=offset, scale=log_scale, use_jit=use_jit)
        X_log = ensure_sparse_format(X_log, "csr")
    else:
        X_log = np.log(X + offset) / log_scale

    # Create new layer
    new_matrix = create_result_layer(X_log, input_layer)
    assay.add_layer(new_layer_name, new_matrix)

    # Log operation
    container.log_operation(
        action="log_transform",
        params={
            "assay": assay_name,
            "source_layer": source_layer,
            "new_layer_name": new_layer_name,
            "base": base,
            "offset": offset,
            "sparse_input": is_sparse_matrix(X),
            "use_jit": use_jit,
        },
        description=f"Log{base} transformation applied to {assay_name}/{source_layer}.",
    )

    return container
