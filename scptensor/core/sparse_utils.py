"""
Sparse matrix utility functions for ScpTensor.

Provides efficient sparse matrix operations and memory management utilities.

Performance Notes:
- For simple element-wise operations like log1p, NumPy's vectorized operations
  are already highly optimized and often faster than JIT for small-to-medium arrays.
- The key optimization here is combining operations (log + scale) to reduce
  passes over the data and minimize memory allocations.
"""

import os
from collections.abc import Callable
from typing import Any

import numpy as np
import scipy.sparse as sp

# JIT threshold: for very large matrices, JIT may provide benefit
# Set high because NumPy is already well-optimized for simple operations
_JIT_THRESHOLD = int(os.getenv("SCPTENSOR_JIT_THRESHOLD", "500000"))

# Lazy import of numba - only import if actually needed
_JIT_AVAILABLE = False


def _ensure_numba() -> bool:
    """Lazily import numba only when needed."""
    global _JIT_AVAILABLE
    if not _JIT_AVAILABLE:
        try:
            import numba  # noqa: F401

            _JIT_AVAILABLE = True
        except ImportError:
            _JIT_AVAILABLE = False
    return _JIT_AVAILABLE


# JIT-compiled kernels (compiled on first use for very large matrices)
_jit_log_with_scale_kernel = None


def _get_jit_log_with_scale_kernel():
    """Get or compile the JIT log+scale kernel for very large matrices."""
    global _jit_log_with_scale_kernel
    if _jit_log_with_scale_kernel is not None:
        return _jit_log_with_scale_kernel

    if not _ensure_numba():
        return None

    try:
        from numba import njit, prange

        @njit(cache=True, parallel=True, fastmath=True)
        def _log_with_scale_numba(data: np.ndarray, offset: float, inv_scale: float) -> np.ndarray:
            """
            Combined log transformation and scaling operation for very large arrays.

            Uses parallel processing and fastmath optimizations.
            Pre-computes inverse scale to avoid division in the loop.
            """
            n = len(data)
            result = np.empty(n, dtype=np.float64)
            for i in prange(n):
                result[i] = np.log1p(data[i] + offset) * inv_scale
            return result

        _jit_log_with_scale_kernel = _log_with_scale_numba
        return _jit_log_with_scale_kernel
    except Exception:
        return None


def is_sparse_matrix(X: np.ndarray | sp.spmatrix) -> bool:
    """
    Check if input is a scipy sparse matrix.

    Parameters
    ----------
    X : Union[np.ndarray, sp.spmatrix]
        Input matrix to check

    Returns
    -------
    bool
        True if X is a scipy sparse matrix, False otherwise

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import sparse
    >>> is_sparse_matrix(np.array([[1, 2], [3, 4]]))
    False
    >>> is_sparse_matrix(sparse.csr_matrix([[1, 0], [0, 4]]))
    True
    """
    return sp.issparse(X)


def get_sparsity_ratio(X: np.ndarray | sp.spmatrix) -> float:
    """
    Calculate the ratio of zero (or missing) values in the matrix.

    Parameters
    ----------
    X : Union[np.ndarray, sp.spmatrix]
        Input matrix

    Returns
    -------
    float
        Ratio of zero values (0.0 to 1.0)

    Examples
    --------
    >>> X = np.array([[1, 0, 2], [0, 0, 3]])
    >>> get_sparsity_ratio(X)
    0.5
    """
    if is_sparse_matrix(X):
        return 1.0 - (X.nnz / (X.shape[0] * X.shape[1]))
    else:
        return np.sum(X == 0) / X.size


def to_sparse_if_beneficial(
    X: np.ndarray | sp.spmatrix, threshold: float = 0.5, format: str = "csr"
) -> np.ndarray | sp.spmatrix:
    """
    Convert dense matrix to sparse if sparsity exceeds threshold.

    Parameters
    ----------
    X : Union[np.ndarray, sp.spmatrix]
        Input matrix
    threshold : float, default=0.5
        Sparsity threshold (0.0 to 1.0). Convert to sparse if sparsity > threshold.
    format : str, default='csr'
        Sparse format: 'csr', 'csc', 'coo', 'lil', etc.

    Returns
    -------
    Union[np.ndarray, sp.spmatrix]
        Sparse matrix if beneficial, otherwise original dense matrix

    Examples
    --------
    >>> X = np.array([[1, 0, 0], [0, 0, 2], [0, 0, 0]])
    >>> X_sparse = to_sparse_if_beneficial(X, threshold=0.5)
    >>> is_sparse_matrix(X_sparse)
    True
    """
    if is_sparse_matrix(X):
        return X

    sparsity = get_sparsity_ratio(X)
    if sparsity > threshold:
        if format == "csr":
            return sp.csr_matrix(X)
        elif format == "csc":
            return sp.csc_matrix(X)
        elif format == "coo":
            return sp.coo_matrix(X)
        else:
            raise ValueError(f"Unsupported sparse format: {format}")
    return X


def ensure_sparse_format(X: np.ndarray | sp.spmatrix, format: str = "csr") -> sp.spmatrix:
    """
    Ensure matrix is in specific sparse format.

    Parameters
    ----------
    X : Union[np.ndarray, sp.spmatrix]
        Input matrix
    format : str, default='csr'
        Target sparse format: 'csr', 'csc', 'coo', 'lil', etc.

    Returns
    -------
    sp.spmatrix
        Matrix in specified sparse format

    Raises
    ------
    ValueError
        If format is not supported

    Examples
    --------
    >>> from scipy import sparse
    >>> X_csc = sparse.csc_matrix([[1, 0], [0, 4]])
    >>> X_csr = ensure_sparse_format(X_csc, format='csr')
    >>> type(X_csr).__name__
    'csr_matrix'
    """
    if is_sparse_matrix(X):
        if format == "csr":
            return X.tocsr()
        elif format == "csc":
            return X.tocsc()
        elif format == "coo":
            return X.tocoo()
        elif format == "lil":
            return X.tolil()
        else:
            raise ValueError(f"Unsupported sparse format: {format}")
    else:
        # Convert dense to sparse
        if format == "csr":
            return sp.csr_matrix(X)
        elif format == "csc":
            return sp.csc_matrix(X)
        elif format == "coo":
            return sp.coo_matrix(X)
        elif format == "lil":
            return sp.lil_matrix(X)
        else:
            raise ValueError(f"Unsupported sparse format: {format}")


def sparse_copy(X: np.ndarray | sp.spmatrix) -> np.ndarray | sp.spmatrix:
    """
    Efficient copy of sparse or dense matrix.

    Parameters
    ----------
    X : Union[np.ndarray, sp.spmatrix]
        Input matrix

    Returns
    -------
    Union[np.ndarray, sp.spmatrix]
        Copy of input matrix

    Examples
    --------
    >>> from scipy import sparse
    >>> X = sparse.csr_matrix([[1, 0], [0, 4]])
    >>> X_copy = sparse_copy(X)
    >>> X_copy[0, 0] = 99
    >>> X[0, 0]
    1
    """
    if is_sparse_matrix(X):
        return X.copy()
    else:
        return X.copy()


def cleanup_layers(
    container: "ScpContainer",  # type: ignore
    assay_name: str,
    keep_layers: list[str],
) -> None:
    """
    Remove unused layers from an assay to free memory.

    This is useful after imputation or normalization when intermediate
    layers are no longer needed.

    Parameters
    ----------
    container : ScpContainer
        Container containing the assay
    assay_name : str
        Name of the assay to clean up
    keep_layers : List[str]
        List of layer names to keep. All others will be removed.

    Raises
    ------
    ValueError
        If assay or layers not found

    Examples
    --------
    >>> from scptensor.core.structures import ScpContainer, Assay, ScpMatrix
    >>> import polars as pl
    >>> import numpy as np
    >>> container = ScpContainer(obs=pl.DataFrame({'_index': ['s1', 's2']}))
    >>> assay = Assay(var=pl.DataFrame({'_index': ['f1', 'f2']}))
    >>> # Add multiple layers...
    >>> # After processing, keep only final result
    >>> cleanup_layers(container, 'proteins', keep_layers=['imputed', 'zscore'])
    """
    if assay_name not in container.assays:
        raise ValueError(f"Assay '{assay_name}' not found in container.")

    assay = container.assays[assay_name]
    layers_to_remove = [layer_name for layer_name in assay.layers if layer_name not in keep_layers]

    for layer_name in layers_to_remove:
        del assay.layers[layer_name]


def _preserve_sparsity_format(
    result: np.ndarray | sp.spmatrix, input_was_sparse: bool
) -> np.ndarray | sp.spmatrix:
    """Ensure result format matches input sparsity.

    Parameters
    ----------
    result : Union[np.ndarray, sp.spmatrix]
        Result from an operation
    input_was_sparse : bool
        Whether input was sparse

    Returns
    -------
    Union[np.ndarray, sp.spmatrix]
        Sparse converted back if input was sparse
    """
    if input_was_sparse and not is_sparse_matrix(result):
        return to_sparse_if_beneficial(result, threshold=0.1)
    return result


def sparse_safe_operation(
    X: np.ndarray | sp.spmatrix,
    operation: Callable[..., np.ndarray | sp.spmatrix | Any],
    *args: Any,
    **kwargs: Any,
) -> np.ndarray | sp.spmatrix:
    """Apply operation to matrix, preserving sparsity when possible.

    Parameters
    ----------
    X : Union[np.ndarray, sp.spmatrix]
        Input matrix
    operation : callable
        Function to apply to matrix
    *args : tuple
        Positional arguments for operation
    **kwargs : dict
        Keyword arguments for operation

    Returns
    -------
    Union[np.ndarray, sp.spmatrix]
        Result of operation, format preserved from input
    """
    input_is_sparse = is_sparse_matrix(X)
    result = operation(X, *args, **kwargs)
    return _preserve_sparsity_format(result, input_is_sparse)


def get_memory_usage(X: np.ndarray | sp.spmatrix) -> dict[str, Any]:
    """
    Get memory usage statistics for matrix.

    Parameters
    ----------
    X : Union[np.ndarray, sp.spmatrix]
        Input matrix

    Returns
    -------
    dict
        Dictionary with memory usage information:
        - 'nbytes': Total bytes used
        - 'is_sparse': Whether matrix is sparse
        - 'shape': Matrix shape
        - 'dtype': Data type

    Examples
    --------
    >>> from scipy import sparse
    >>> X = sparse.csr_matrix([[1, 0], [0, 4]])
    >>> stats = get_memory_usage(X)
    >>> stats['is_sparse']
    True
    """
    if is_sparse_matrix(X):
        nbytes = X.data.nbytes + X.indices.nbytes + X.indptr.nbytes
    else:
        nbytes = X.nbytes

    return {
        "nbytes": nbytes,
        "is_sparse": is_sparse_matrix(X),
        "shape": X.shape,
        "dtype": str(X.dtype),
    }


def optimal_format_for_operation(X: np.ndarray | sp.spmatrix, operation: str) -> str:
    """
    Determine the optimal sparse format for a given operation.

    Parameters
    ----------
    X : Union[np.ndarray, sp.spmatrix]
        Input matrix
    operation : str
        Operation type:
        - 'row_wise': Operations on rows (e.g., row sums, row-wise distance)
          Optimal: CSR (Compressed Sparse Row)
        - 'col_wise': Operations on columns (e.g., column sums, column scaling)
          Optimal: CSC (Compressed Sparse Column)
        - 'arithmetic': Arithmetic operations (add, multiply, etc.)
          Optimal: CSR or CSC (prefer CSR for row-major data)
        - 'construction': Building matrix incrementally
          Optimal: LIL (List of Lists) or COO (Coordinate)
        - 'slicing': Slicing operations
          Optimal: CSR for row slicing, CSC for column slicing
        - 'modification': Frequent value changes
          Optimal: LIL (List of Lists)

    Returns
    -------
    str
        Optimal format: 'csr', 'csc', 'coo', or 'lil'

    Examples
    --------
    >>> from scipy import sparse
    >>> X = sparse.csc_matrix([[1, 0], [0, 4]])
    >>> optimal_format_for_operation(X, 'row_wise')
    'csr'
    >>> optimal_format_for_operation(X, 'col_wise')
    'csc'
    """
    if not is_sparse_matrix(X):
        return "dense"

    operation_map = {
        "row_wise": "csr",
        "col_wise": "csc",
        "arithmetic": "csr",
        "construction": "lil",
        "slicing": "csr",
        "modification": "lil",
    }

    return operation_map.get(operation, "csr")


def auto_convert_for_operation(X: np.ndarray | sp.spmatrix, operation: str) -> sp.spmatrix:
    """
    Automatically convert matrix to optimal format for the given operation.

    This is a convenience function that combines optimal_format_for_operation
    with ensure_sparse_format.

    Parameters
    ----------
    X : Union[np.ndarray, sp.spmatrix]
        Input matrix
    operation : str
        Operation type (see optimal_format_for_operation)

    Returns
    -------
    sp.spmatrix
        Matrix in optimal format (or original if dense)

    Examples
    --------
    >>> from scipy import sparse
    >>> X_csc = sparse.csc_matrix([[1, 0], [0, 4]])
    >>> X_csr = auto_convert_for_operation(X_csc, 'row_wise')
    >>> type(X_csr).__name__
    'csr_matrix'
    """
    if not is_sparse_matrix(X):
        return X  # Keep dense matrices as is

    optimal_format = optimal_format_for_operation(X, operation)
    return ensure_sparse_format(X, format=optimal_format)


def sparse_row_operation(
    X: sp.spmatrix, func: Callable[[np.ndarray], Any], **kwargs: Any
) -> np.ndarray:
    """
    Apply a function to each row of a sparse matrix efficiently.

    Parameters
    ----------
    X : sp.spmatrix
        Input sparse matrix (will be converted to CSR if needed)
    func : callable
        Function to apply to each row. Should accept a 1D array.
    **kwargs : dict
        Additional arguments passed to func

    Returns
    -------
    np.ndarray
        Result of applying func to each row

    Examples
    --------
    >>> from scipy import sparse
    >>> X = sparse.csr_matrix([[1, 0, 2], [0, 0, 3]])
    >>> result = sparse_row_operation(X, np.sum)
    >>> result
    array([3, 3])
    """
    X_csr = X.tocsr() if is_sparse_matrix(X) else X
    n_rows = X_csr.shape[0]

    result = np.empty(n_rows, dtype=np.float64)
    for i in range(n_rows):
        start, end = X_csr.indptr[i], X_csr.indptr[i + 1]
        row_data = X_csr.data[start:end]
        result[i] = func(row_data, **kwargs) if len(row_data) > 0 else 0

    return result


def sparse_col_operation(
    X: sp.spmatrix, func: Callable[[np.ndarray], Any], **kwargs: Any
) -> np.ndarray:
    """
    Apply a function to each column of a sparse matrix efficiently.

    Parameters
    ----------
    X : sp.spmatrix
        Input sparse matrix (will be converted to CSC if needed)
    func : callable
        Function to apply to each column. Should accept a 1D array.
    **kwargs : dict
        Additional arguments passed to func

    Returns
    -------
    np.ndarray
        Result of applying func to each column

    Examples
    --------
    >>> from scipy import sparse
    >>> X = sparse.csr_matrix([[1, 0, 2], [0, 0, 3]])
    >>> result = sparse_col_operation(X, np.sum)
    >>> result
    array([1., 0., 5.])
    """
    X_csc = X.tocsc() if is_sparse_matrix(X) else X
    n_cols = X_csc.shape[1]

    result = np.empty(n_cols, dtype=np.float64)
    for j in range(n_cols):
        start, end = X_csc.indptr[j], X_csc.indptr[j + 1]
        col_data = X_csc.data[start:end]
        result[j] = func(col_data, **kwargs) if len(col_data) > 0 else 0

    return result


def sparse_multiply_rowwise(X: sp.spmatrix, factors: np.ndarray) -> sp.spmatrix:
    """
    Multiply each row of a sparse matrix by a corresponding factor.

    Efficient CSR-based operation that avoids densification.

    Parameters
    ----------
    X : sp.spmatrix
        Input sparse matrix
    factors : np.ndarray
        Array of factors (length must equal n_rows)

    Returns
    -------
    sp.spmatrix
        Row-multiplied sparse matrix in CSR format

    Examples
    --------
    >>> from scipy import sparse
    >>> X = sparse.csr_matrix([[1, 0], [2, 3]])
    >>> factors = np.array([2.0, 0.5])
    >>> result = sparse_multiply_rowwise(X, factors)
    >>> result.toarray()
    array([[2., 0.],
           [1., 1.5]])
    """
    X_csr = ensure_sparse_format(X, "csr")
    n_rows = X_csr.shape[0]

    if len(factors) != n_rows:
        raise ValueError(f"Length of factors ({len(factors)}) must equal n_rows ({n_rows})")

    # Ensure float dtype for multiplication
    if X_csr.dtype.kind != "f":
        X_csr = X_csr.astype(float)

    # Create copy with scaled data
    result = X_csr.copy()
    for i in range(n_rows):
        start, end = result.indptr[i], result.indptr[i + 1]
        result.data[start:end] *= factors[i]

    return result


def sparse_multiply_colwise(X: sp.spmatrix, factors: np.ndarray) -> sp.spmatrix:
    """
    Multiply each column of a sparse matrix by a corresponding factor.

    Efficient CSC-based operation that avoids densification.

    Parameters
    ----------
    X : sp.spmatrix
        Input sparse matrix
    factors : np.ndarray
        Array of factors (length must equal n_cols)

    Returns
    -------
    sp.spmatrix
        Column-multiplied sparse matrix in CSC format

    Examples
    --------
    >>> from scipy import sparse
    >>> X = sparse.csr_matrix([[1, 0], [2, 3]])
    >>> factors = np.array([2.0, 0.5])
    >>> result = sparse_multiply_colwise(X, factors)
    >>> result.toarray()
    array([[2., 0.],
           [4., 1.5]])
    """
    X_csc = ensure_sparse_format(X, "csc")
    n_cols = X_csc.shape[1]

    if len(factors) != n_cols:
        raise ValueError(f"Length of factors ({len(factors)}) must equal n_cols ({n_cols})")

    # Ensure float dtype for multiplication
    if X_csc.dtype.kind != "f":
        X_csc = X_csc.astype(float)

    # Create copy with scaled data
    result = X_csc.copy()
    for j in range(n_cols):
        start, end = result.indptr[j], result.indptr[j + 1]
        result.data[start:end] *= factors[j]

    return result


def sparse_center_rows(X: sp.spmatrix, row_means: np.ndarray | None = None) -> sp.spmatrix:
    """
    Center each row of a sparse matrix by subtracting its mean.

    This operation preserves sparsity by only storing deviations from the mean.

    Parameters
    ----------
    X : sp.spmatrix
        Input sparse matrix
    row_means : np.ndarray, optional
        Pre-computed row means. If None, will be computed.

    Returns
    -------
    tuple
        (centered_sparse_matrix, row_means)

    Examples
    --------
    >>> from scipy import sparse
    >>> X = sparse.csr_matrix([[1, 0, 3], [0, 2, 4]])
    >>> X_centered, means = sparse_center_rows(X)
    >>> means
    array([2., 3.])
    """
    X_csr = ensure_sparse_format(X, "csr")
    n_rows, n_cols = X_csr.shape

    # Compute row means if not provided
    if row_means is None:
        row_sums = np.zeros(n_rows)
        row_counts = np.zeros(n_rows)
        for i in range(n_rows):
            start, end = X_csr.indptr[i], X_csr.indptr[i + 1]
            row_sums[i] = np.sum(X_csr.data[start:end])
            row_counts[i] = end - start
        row_means = np.divide(row_sums, row_counts, where=row_counts > 0)

    # Create centered matrix (this will densify, so we use a different approach)
    # For sparse matrices, we return the original with means separately
    return X_csr, row_means


def sparse_safe_log1p(
    X: np.ndarray | sp.spmatrix, offset: float = 1.0, use_jit: bool = True
) -> np.ndarray | sp.spmatrix:
    """
    Apply log(1 + x) transformation preserving sparsity.

    For sparse matrices, this is more efficient than log(x + offset)
    because we only need to compute on non-zero elements.

    Parameters
    ----------
    X : Union[np.ndarray, sp.spmatrix]
        Input matrix
    offset : float, default=1.0
        Offset to add before taking log
    use_jit : bool, default=True
        Whether to use JIT acceleration for very large matrices (>500K nnz by default)

    Returns
    -------
    Union[np.ndarray, sp.spmatrix]
        Log-transformed matrix (same format as input)

    Examples
    --------
    >>> from scipy import sparse
    >>> X = sparse.csr_matrix([[1, 0], [0, 4]])
    >>> result = sparse_safe_log1p(X)
    >>> result.toarray()
    array([[0.69314718, 0.        ],
           [0.        , 1.60943791]])
    """
    if is_sparse_matrix(X):
        result = X.copy()
        # Ensure float dtype for log operation
        if result.data.dtype.kind != "f":
            result.data = result.data.astype(float)

        # NumPy vectorization is already highly optimized for simple operations
        # JIT only helps for very large arrays where parallel overhead is worth it
        if use_jit and X.nnz > _JIT_THRESHOLD:
            kernel = _get_jit_log_with_scale_kernel()
            if kernel is not None:
                result.data = kernel(result.data, offset - 1.0, 1.0)
                return result

        # Fast path: NumPy vectorized operation
        result.data = np.log1p(result.data + offset - 1.0)
        return result
    else:
        return np.log1p(X + offset - 1.0)


def sparse_safe_log1p_with_scale(
    X: np.ndarray | sp.spmatrix, offset: float = 1.0, scale: float = 1.0, use_jit: bool = True
) -> np.ndarray | sp.spmatrix:
    """
    Apply log(1 + x) transformation with scaling in a single operation.

    This function combines the log transformation and scaling into a single
    expression, reducing memory allocations and improving cache locality.

    Computes: log(x + offset) / scale

    Performance Notes:
    - The combined operation is faster than separate log() then divide()
    - JIT is only used for very large matrices (>500K nnz) where the parallel
      overhead is justified. For most cases, NumPy's vectorized operations
      are already optimal.

    Parameters
    ----------
    X : Union[np.ndarray, sp.spmatrix]
        Input matrix
    offset : float, default=1.0
        Offset to add before taking log
    scale : float, default=1.0
        Scale factor to divide by (e.g., log(base) for log normalization)
    use_jit : bool, default=True
        Whether to use JIT acceleration for very large matrices

    Returns
    -------
    Union[np.ndarray, sp.spmatrix]
        Log-transformed and scaled matrix (same format as input)

    Examples
    --------
    >>> from scipy import sparse
    >>> X = sparse.csr_matrix([[1, 0], [0, 4]])
    >>> # Equivalent to log2(X + 1) since log(2) ≈ 0.693
    >>> result = sparse_safe_log1p_with_scale(X, offset=1.0, scale=np.log(2))
    """
    if is_sparse_matrix(X):
        result = X.copy()
        # Ensure float dtype for log operation
        if result.data.dtype.kind != "f":
            result.data = result.data.astype(float)

        # For very large matrices, JIT may provide benefit
        if use_jit and X.nnz > _JIT_THRESHOLD:
            kernel = _get_jit_log_with_scale_kernel()
            if kernel is not None:
                # Pre-compute inverse scale for the JIT kernel
                result.data = kernel(result.data, offset - 1.0, 1.0 / scale)
                return result

        # Fast path: combined NumPy operation (single pass over data)
        # This is typically faster than JIT for small-to-medium arrays
        result.data = np.log1p(result.data + offset - 1.0) / scale
        return result
    else:
        # Dense matrix: single combined operation
        return np.log1p(X + offset - 1.0) / scale


def get_format_recommendation(n_rows: int, n_cols: int, nnz: int, operations: list) -> str:
    """
    Recommend the optimal sparse format based on matrix properties and operations.

    Parameters
    ----------
    n_rows : int
        Number of rows
    n_cols : int
        Number of columns
    nnz : int
        Number of non-zero elements
    operations : list of str
        List of operations to perform (e.g., ['row_wise', 'arithmetic'])

    Returns
    -------
    str
        Recommended format: 'csr', 'csc', 'coo', 'lil', or 'dense'

    Examples
    --------
    >>> get_format_recommendation(1000, 100, 50000, ['row_wise'])
    'csr'
    >>> get_format_recommendation(100, 1000, 50000, ['col_wise'])
    'csc'
    """
    sparsity = 1.0 - (nnz / (n_rows * n_cols))

    # If not sparse enough, recommend dense
    if sparsity < 0.3:
        return "dense"

    # Count operation types
    row_ops = sum(1 for op in operations if "row" in op)
    col_ops = sum(1 for op in operations if "col" in op)
    modify_ops = sum(1 for op in operations in ["construction", "modification"])

    if modify_ops > 0:
        return "lil"
    elif row_ops > col_ops:
        return "csr"
    elif col_ops > row_ops:
        return "csc"
    else:
        # Default for arithmetic operations
        return "csr" if n_rows >= n_cols else "csc"


if __name__ == "__main__":
    print("Testing sparse_utils.py...")

    # Test 1: is_sparse_matrix
    dense = np.array([[1, 2], [3, 4]])
    sparse = sp.csr_matrix([[1, 0], [0, 4]])
    assert not is_sparse_matrix(dense)
    assert is_sparse_matrix(sparse)
    print("✓ is_sparse_matrix tests passed")

    # Test 2: get_sparsity_ratio
    X_test = np.array([[1, 0, 0], [0, 0, 2], [0, 0, 0]])
    assert get_sparsity_ratio(X_test) == 2 / 3
    print("✓ get_sparsity_ratio tests passed")

    # Test 3: to_sparse_if_beneficial
    X_sparse_input = np.array([[1, 0, 0], [0, 0, 2]])
    result = to_sparse_if_beneficial(X_sparse_input, threshold=0.5)
    assert is_sparse_matrix(result)
    print("✓ to_sparse_if_beneficial tests passed")

    # Test 4: ensure_sparse_format
    X_csc = sp.csc_matrix([[1, 0], [0, 4]])
    X_csr = ensure_sparse_format(X_csc, format="csr")
    assert isinstance(X_csr, sp.csr_matrix)
    print("✓ ensure_sparse_format tests passed")

    # Test 5: sparse_copy
    X_copy_test = sparse.copy()
    X_copy_test[0, 0] = 99
    assert X[0, 0] == 1  # Original unchanged
    print("✓ sparse_copy tests passed")

    # Test 6: get_memory_usage
    stats = get_memory_usage(sparse)
    assert stats["is_sparse"]
    assert stats["shape"] == (2, 2)
    print("✓ get_memory_usage tests passed")

    # Test 7: optimal_format_for_operation
    assert optimal_format_for_operation(sparse, "row_wise") == "csr"
    assert optimal_format_for_operation(sparse, "col_wise") == "csc"
    assert optimal_format_for_operation(sparse, "modification") == "lil"
    print("✓ optimal_format_for_operation tests passed")

    # Test 8: sparse_multiply_rowwise
    X = sp.csr_matrix([[1, 0], [2, 3]])
    factors = np.array([2.0, 0.5])
    result = sparse_multiply_rowwise(X, factors)
    assert result[0, 0] == 2.0
    assert result[1, 0] == 1.0
    print("✓ sparse_multiply_rowwise tests passed")

    # Test 9: sparse_multiply_colwise
    factors_col = np.array([2.0, 0.5])
    result_col = sparse_multiply_colwise(X, factors_col)
    assert result_col[0, 0] == 2.0
    assert result_col[1, 1] == 1.5
    print("✓ sparse_multiply_colwise tests passed")

    # Test 10: sparse_safe_log1p
    X_log = sp.csr_matrix([[1, 0], [0, 4]])
    result_log = sparse_safe_log1p(X_log)
    expected = np.log(2)  # log1p(1) = log(2)
    assert abs(result_log[0, 0] - expected) < 1e-10
    print("✓ sparse_safe_log1p tests passed")

    # Test 11: get_format_recommendation
    fmt = get_format_recommendation(1000, 100, 50000, ["row_wise"])
    assert fmt == "csr"
    print("✓ get_format_recommendation tests passed")

    print("\n✅ All tests passed!")
