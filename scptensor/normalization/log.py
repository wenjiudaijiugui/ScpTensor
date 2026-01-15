"""Log normalization module for ScpTensor.

Provides log transformation with configurable base and offset.
Optimized for both dense and sparse matrices.
"""

import warnings

from typing import overload

import numpy as np

from scptensor.core.exceptions import (
    AssayNotFoundError,
    LayerNotFoundError,
    ScpValueError,
)
from scptensor.core.sparse_utils import (
    ensure_sparse_format,
    is_sparse_matrix,
    sparse_safe_log1p_with_scale,
)
from scptensor.core.structures import ScpContainer, ScpMatrix


@overload
def norm_log(
    container: ScpContainer,
    assay_name: str = "protein",
    source_layer: str = "raw",
    new_layer_name: str = "log",
    base: float = 2.0,
    offset: float = 1.0,
    use_jit: bool = True,
) -> ScpContainer: ...


def norm_log(
    container: ScpContainer,
    assay_name: str = "protein",
    source_layer: str = "raw",
    new_layer_name: str = "log",
    base: float = 2.0,
    offset: float = 1.0,
    use_jit: bool = True,
) -> ScpContainer:
    """
    Apply log transformation with configurable base and offset.

    Supports both dense and sparse matrices. For sparse matrices, the log
    transformation is applied only to non-zero elements, preserving sparsity
    for efficiency.

    Mathematical Formulation:
        X_log = log_base(X + offset)

    Performance Characteristics:
    - Small/Medium matrices (<500K non-zero elements): NumPy vectorized operations
      are already optimal; JIT is not used.
    - Large matrices (>500K non-zero elements): JIT may provide marginal benefits
      for very large datasets. The threshold can be configured via the
      SCPTENSOR_JIT_THRESHOLD environment variable.

    Parameters
    ----------
    container : ScpContainer
        The ScpContainer object.
    assay_name : str, default="protein"
        Name of the assay to transform.
    source_layer : str, default="raw"
        Name of the layer to use as input.
    new_layer_name : str, default="log"
        Name of the new layer to create.
    base : float, default=2.0
        Log base (e.g., 2.0 for log2, np.e for natural log).
    offset : float, default=1.0
        Offset to add before logging to handle zeros.
    use_jit : bool, default=True
        Whether to use JIT acceleration for very large matrices.

    Returns
    -------
    ScpContainer
        ScpContainer with added log-normalized layer.

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
    >>> import numpy as np
    >>> container = ScpContainer(obs=pl.DataFrame({'_index': ['s1', 's2']}))
    >>> assay = Assay(var=pl.DataFrame({'_index': ['p1', 'p2']}))
    >>> assay.add_layer('raw', ScpMatrix(X=np.array([[1, 2], [3, 4]])))
    >>> container.add_assay('protein', assay)
    >>> result = norm_log(container, base=2.0)
    >>> 'log' in result.assays['protein'].layers
    True
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

    # Validate assay exists
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

    input_matrix = assay.layers[source_layer]
    X = input_matrix.X
    M = input_matrix.M

    # Perform log transformation: log_base(X + offset)
    # Using change of base formula: log_b(x) = ln(x) / ln(b)
    log_base = np.log(base)

    if is_sparse_matrix(X):
        # For sparse matrices, use combined log+scale operation
        # This reduces memory allocations and improves cache locality
        X_log = sparse_safe_log1p_with_scale(X, offset=offset, scale=log_base, use_jit=use_jit)
        # Ensure CSR format for efficiency
        X_log = ensure_sparse_format(X_log, "csr")
    else:
        # Dense matrix transformation
        X_log = np.log(X + offset) / log_base

    # Create new matrix
    new_matrix = ScpMatrix(X=X_log, M=M.copy() if M is not None else None)

    # Add new layer to assay
    assay.add_layer(new_layer_name, new_matrix)

    # Log operation
    container.log_operation(
        action="log_normalize",
        params={
            "assay": assay_name,
            "source_layer": source_layer,
            "new_layer_name": new_layer_name,
            "base": base,
            "offset": offset,
            "sparse_input": is_sparse_matrix(X),
            "use_jit": use_jit,
        },
        description=f"Log{base} normalization applied to {assay_name}/{source_layer}.",
    )

    return container


# Backward compatibility alias
def log_normalize(container, assay_name, base_layer=None, source_layer=None, new_layer_name="log",
                  base=2.0, offset=1.0, use_jit=True):
    """Deprecated: Use norm_log instead.

    This function will be removed in version 1.0.0.

    Notes
    -----
    For backward compatibility, accepts both 'base_layer' (deprecated) and
    'source_layer' (new parameter name). If both are provided, 'source_layer' takes precedence.
    """
    warnings.warn(
        "'log_normalize' is deprecated, use 'norm_log' instead. "
        "This will be removed in version 1.0.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Handle parameter name change for backward compatibility
    if source_layer is None:
        source_layer = base_layer
    return norm_log(container, assay_name, source_layer, new_layer_name, base, offset, use_jit)
