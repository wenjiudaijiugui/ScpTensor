"""ScpTensor core module type definitions.

This module provides concrete type aliases to replace generic 'Any' types,
improving type safety, IDE support, and early error detection throughout
the ScpTensor codebase.

Type Categories
--------------
Matrix Types: Dense/sparse matrix representations
Serialization Types: JSON-serializable data structures
Function Types: Callback and operation signatures
ID/Index Types: Sample/feature identifiers and filters
Metadata Types: Metadata dictionary value types

Examples
--------
>>> import numpy as np
>>> from scptensor.core.types import DenseMatrix, RowFunction
>>>
>>> def mean_intensity(row: np.ndarray) -> float:
...     return float(np.mean(row))
>>>
>>> func: RowFunction = mean_intensity
>>> matrix: DenseMatrix = np.array([[1.0, 2.0], [3.0, 4.0]])
>>> result = func(matrix[0])
>>> print(result)  # 1.5
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import scipy.sparse as sp

if TYPE_CHECKING:
    pass

# =============================================================================
# Matrix Type Aliases
# =============================================================================

type DenseMatrix = np.ndarray
"""Dense NumPy matrix (2D array).

Used for dense data storage where most values are non-zero.
Typically float64 or int32 dtype.
"""

type SparseMatrix = sp.spmatrix
"""Any scipy sparse matrix format.

Supports CSR, CSC, COO, LIL, and other sparse formats.
Use for data with >50% missing values to save memory.
"""

type Matrix = DenseMatrix | SparseMatrix
"""Union of dense and sparse matrix types.

Allows functions to accept either format transparently.
Functions should use is_sparse_matrix() to check type.
"""

type MaskMatrix = np.ndarray | sp.spmatrix
"""Mask code matrix (dense int8 or sparse int8).

Contains integer codes from MaskCode enum:
- 0: VALID (detected value)
- 1: MBR (missing between runs)
- 2: LOD (below detection limit)
- 3: FILTERED (QC removed)
- 4: OUTLIER (statistical outlier)
- 5: IMPUTED (filled value)
- 6: UNCERTAIN (uncertain quality)
"""

# =============================================================================
# Serialization Type Aliases
# =============================================================================

type JsonArray = list["JsonValue"]
"""JSON array (list of JSON values).

Represents ordered sequences in JSON format.
Can contain nested mixed types.
"""

type JsonObject = dict[str, "JsonValue"]
"""JSON object (string-keyed dictionary).

Represents key-value mappings in JSON format.
All keys must be strings.
"""

type JsonValue = None | bool | int | float | str | JsonArray | JsonObject
"""JSON-serializable value (recursive definition).

Valid JSON types that can be serialized with json.dumps().
Excludes complex objects, datetime, NaN, infinity.
"""

type SerializableDict = dict[str, JsonValue]
"""JSON-serializable dictionary.

Used for parameters that must be serializable for
provenance tracking and I/O operations.
"""

type ProvenanceParams = dict[str, JsonValue]
"""Parameter dictionary for ProvenanceLog.

Stores operation parameters with JSON-serializable values.
Enables reproducibility and provenance tracking.
"""

type LayerMetadataDict = dict[str, JsonValue]
"""Metadata dictionary for ScpMatrix layers.

Stores layer-specific metadata with JSON-serializable values.
Used for serialization and provenance tracking.
"""

# =============================================================================
# Function Type Aliases
# =============================================================================

type RowFunction = Callable[[np.ndarray], float]
"""Function that accepts 1D array and returns scalar.

Used for row-wise operations like statistics aggregation.
Example: mean, median, sum, std operations.

Examples
--------
>>> import numpy as np
>>> from scptensor.core.types import RowFunction
>>>
>>> def row_mean(row: np.ndarray) -> float:
...     return float(np.mean(row))
>>>
>>> func: RowFunction = row_mean
"""

type MatrixOperation = Callable[[Matrix], Matrix | float | np.ndarray]
"""Function that operates on matrix and returns various types.

Generic matrix operation signature that can return:
- Matrix: transformation operations
- float: scalar aggregation (e.g., determinant)
- np.ndarray: vector results (e.g., eigenvalues)

Examples
--------
>>> import numpy as np
>>> from scptensor.core.types import MatrixOperation, Matrix
>>>
>>> def normalize(mat: Matrix) -> Matrix:
...     if isinstance(mat, np.ndarray):
...         return mat / np.max(mat)
...     return mat
>>>
>>> op: MatrixOperation = normalize
"""

# =============================================================================
# ID and Index Type Aliases
# =============================================================================

type SampleIDs = list[str] | np.ndarray | pl.Series
"""Sample identifier (multiple formats supported).

Accepts sample IDs in three common formats:
- list[str]: Python list of strings
- np.ndarray: NumPy array with object or string dtype
- pl.Series: Polars series with string dtype

Used throughout API for flexible sample specification.
"""

type FeatureIDs = list[str] | np.ndarray | pl.Series
"""Feature identifier (multiple formats supported).

Accepts feature IDs in three common formats:
- list[str]: Python list of strings
- np.ndarray: NumPy array with object or string dtype
- pl.Series: Polars series with string dtype

Used throughout API for flexible feature specification.
"""

type Indices = list[int] | np.ndarray
"""Positional index array.

Integer indices for positional selection.
Can be Python list or NumPy array of integers.
Used for index-based filtering and slicing.
"""

type BooleanMask = np.ndarray | pl.Series
"""Boolean mask for filtering.

True values indicate items to keep, False indicates discard.
Used for mask-based filtering of samples/features.
"""

# =============================================================================
# Metadata Type Aliases
# =============================================================================

type MetadataValue = None | bool | int | float | str | list | dict
"""Metadata value (basic or nested types).

Accepts common Python types for metadata storage:
- None: missing values
- bool: boolean flags
- int/float: numeric values
- str: text values
- list/dict: nested structures

Lists and dicts must contain JSON-serializable values.
"""

type MetadataDict = dict[str, MetadataValue]
"""Metadata dictionary mapping.

String keys to metadata values.
Used for obs and var DataFrames in ScpContainer/Assay.
"""

# =============================================================================
# Export all type aliases
# =============================================================================

__all__ = [
    # Matrix types
    "DenseMatrix",
    "SparseMatrix",
    "Matrix",
    "MaskMatrix",
    # Serialization types
    "JsonArray",
    "JsonObject",
    "JsonValue",
    "SerializableDict",
    "ProvenanceParams",
    "LayerMetadataDict",
    # Function types
    "RowFunction",
    "MatrixOperation",
    # ID and index types
    "SampleIDs",
    "FeatureIDs",
    "Indices",
    "BooleanMask",
    # Metadata types
    "MetadataValue",
    "MetadataDict",
]
