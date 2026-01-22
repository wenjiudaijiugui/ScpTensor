"""Base serialization utilities for ScpTensor I/O operations.

This module provides low-level serialization functions for converting
between ScpTensor data structures and serializable formats.
"""

from __future__ import annotations

import polars as pl
import scipy.sparse as sp

from scptensor.core.types import SerializableDict

__all__ = [
    "_sparse_to_dict",
    "_dict_to_sparse",
    "_serialize_dataframe",
    "_deserialize_dataframe",
]


def _sparse_to_dict(matrix: sp.spmatrix) -> SerializableDict:
    """Convert sparse matrix to serializable dictionary.

    Parameters
    ----------
    matrix : sp.spmatrix
        Sparse matrix (CSR or CSC format).

    Returns
    -------
    dict
        Dictionary with format, data, indices, indptr, and shape.
    """
    if sp.isspmatrix_csr(matrix):
        format_name = "csr"
    elif sp.isspmatrix_csc(matrix):
        format_name = "csc"
    else:
        # Convert to CSR for other formats
        if sp.issparse(matrix):
            matrix = matrix.tocsr()
        else:
            matrix = sp.csr_matrix(matrix)
        format_name = "csr"

    return {
        "format": format_name,
        "data": matrix.data,
        "indices": matrix.indices,
        "indptr": matrix.indptr,
        "shape": list(matrix.shape),  # Convert tuple to list for JSON serialization
    }


def _dict_to_sparse(d: SerializableDict) -> sp.spmatrix:
    """Convert dictionary back to sparse matrix.

    Parameters
    ----------
    d : dict
        Dictionary with sparse matrix data.

    Returns
    -------
    sp.spmatrix
        Reconstructed sparse matrix.
    """
    format_name = d["format"]
    data = d["data"]
    indices = d["indices"]
    indptr = d["indptr"]
    shape_value = d["shape"]
    # Type narrowing: shape can be a list or tuple
    if isinstance(shape_value, list):
        shape = tuple(shape_value)
    else:
        shape = shape_value  # Already a tuple

    if format_name == "csr":
        return sp.csr_matrix((data, indices, indptr), shape=shape)
    elif format_name == "csc":
        return sp.csc_matrix((data, indices, indptr), shape=shape)
    else:
        raise ValueError(f"Unknown sparse format: {format_name}")


def _serialize_dataframe(df: pl.DataFrame) -> SerializableDict:
    """Serialize Polars DataFrame to dictionary.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to serialize.

    Returns
    -------
    dict
        Dictionary with columns data and schema.
    """
    return {
        "columns": {name: df[name].to_numpy() for name in df.columns},  # type: ignore[misc]
        "dtypes": {name: str(dtype) for name, dtype in df.schema.items()},
    }


def _deserialize_dataframe(data: SerializableDict) -> pl.DataFrame:
    """Deserialize dictionary back to Polars DataFrame.

    Parameters
    ----------
    data : dict
        Dictionary with DataFrame data.

    Returns
    -------
    pl.DataFrame
        Reconstructed DataFrame.
    """
    return pl.DataFrame(data["columns"])
