"""ScpTensor core module.

Provides the hierarchical data structure (ScpContainer -> Assay -> ScpMatrix)
and core I/O operations for single-cell proteomics data analysis.
"""

# Exceptions
from .exceptions import (
    AssayNotFoundError,
    DimensionError,
    LayerNotFoundError,
    MaskCodeError,
    MissingDependencyError,
    ScpTensorError,
    ScpValueError,
    StructureError,
    ValidationError,
)

# JIT operations
from .jit_ops import (
    NUMBA_AVAILABLE,
    apply_mask_threshold,
    compute_euclidean_distance,
    count_mask_codes,
    fill_missing_with_value,
    find_missing_indices,
)

# Matrix operations
from .matrix_ops import MatrixOps

# I/O functions
from .io import (
    from_scanpy,
    load_csv,
    load_h5ad,
    load_npz,
    read_h5ad,
    save_csv,
    save_h5ad,
    save_npz,
    to_scanpy,
    write_h5ad,
)

# Reader
from .reader import reader

# Sparse utilities
from .sparse_utils import (
    auto_convert_for_operation,
    cleanup_layers,
    ensure_sparse_format,
    get_format_recommendation,
    get_memory_usage,
    get_sparsity_ratio,
    is_sparse_matrix,
    optimal_format_for_operation,
    sparse_center_rows,
    sparse_col_operation,
    sparse_copy,
    sparse_multiply_colwise,
    sparse_multiply_rowwise,
    sparse_row_operation,
    sparse_safe_log1p,
    to_sparse_if_beneficial,
)

# Core structures
from .structures import (
    AggregationLink,
    Assay,
    MaskCode,
    MatrixMetadata,
    ProvenanceLog,
    ScpContainer,
    ScpMatrix,
)

__all__ = [
    # Structures
    "ScpContainer",
    "Assay",
    "ScpMatrix",
    "ProvenanceLog",
    "MatrixMetadata",
    "MaskCode",
    "AggregationLink",
    # Operations
    "MatrixOps",
    "reader",
    # I/O
    "load_csv",
    "save_csv",
    "load_h5ad",
    "save_h5ad",
    "load_npz",
    "save_npz",
    "from_scanpy",
    "to_scanpy",
    "read_h5ad",
    "write_h5ad",
    # Sparse utilities
    "is_sparse_matrix",
    "get_sparsity_ratio",
    "to_sparse_if_beneficial",
    "ensure_sparse_format",
    "sparse_copy",
    "cleanup_layers",
    "get_memory_usage",
    "optimal_format_for_operation",
    "auto_convert_for_operation",
    "sparse_row_operation",
    "sparse_col_operation",
    "sparse_multiply_rowwise",
    "sparse_multiply_colwise",
    "sparse_center_rows",
    "sparse_safe_log1p",
    "get_format_recommendation",
    # JIT
    "NUMBA_AVAILABLE",
    "count_mask_codes",
    "find_missing_indices",
    "compute_euclidean_distance",
    "apply_mask_threshold",
    "fill_missing_with_value",
    # Exceptions
    "ScpTensorError",
    "StructureError",
    "ValidationError",
    "LayerNotFoundError",
    "AssayNotFoundError",
    "MissingDependencyError",
    "DimensionError",
    "ScpValueError",
    "MaskCodeError",
]
