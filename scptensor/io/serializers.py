"""Serialization utilities for ScpTensor data structures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import h5py
import numpy as np
import polars as pl
import scipy as sp

if TYPE_CHECKING:
    pass


def serialize_dataframe(df: pl.DataFrame, group: h5py.Group) -> None:
    """Serialize polars DataFrame to HDF5 group.

    Each column becomes a dataset. String columns use variable-length dtype.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to serialize
    group : h5py.Group
        Target HDF5 group
    """
    string_dtype = h5py.string_dtype(encoding="utf-8")

    for col_name in df.columns:
        col_data = df[col_name].to_numpy()

        # Determine dtype
        if col_data.dtype == object or col_data.dtype == str:
            # String column
            dtype = string_dtype
        elif np.issubdtype(col_data.dtype, np.integer):
            dtype = np.int64
        elif np.issubdtype(col_data.dtype, np.floating):
            dtype = np.float64
        elif col_data.dtype == bool:
            dtype = bool
        else:
            dtype = None  # Let h5py infer

        group.create_dataset(col_name, data=col_data, dtype=dtype)


def deserialize_dataframe(group: h5py.Group) -> pl.DataFrame:
    """Deserialize HDF5 group to polars DataFrame.

    Parameters
    ----------
    group : h5py.Group
        Source HDF5 group containing column datasets

    Returns
    -------
    pl.DataFrame
        Reconstructed DataFrame
    """
    data = {}
    for col_name in group:
        dataset = group[col_name]
        col_data = dataset[:]

        # Decode bytes to strings if necessary
        if len(col_data) > 0 and isinstance(col_data[0], bytes):
            col_data = np.array([x.decode("utf-8") for x in col_data])

        data[col_name] = col_data

    return pl.DataFrame(data)


def serialize_dense_matrix(
    X: np.ndarray,  # noqa: N803
    group: h5py.Group,
    name: str = "X",
    compression: str | None = "gzip",
    compression_level: int = 4,
) -> None:
    """Serialize dense numpy array to HDF5 dataset.

    Parameters
    ----------
    X : np.ndarray
        Dense array to serialize
    group : h5py.Group
        Target HDF5 group
    name : str, default "X"
        Dataset name
    compression : str | None, default "gzip"
        Compression algorithm
    compression_level : int, default 4
        Compression level (0-9)
    """
    group.create_dataset(
        name,
        data=X.astype(np.float32),
        compression=compression,
        compression_opts=compression_level if compression else None,
    )


def serialize_sparse_matrix(
    X: sp.spmatrix,  # noqa: N803
    group: h5py.Group,
    name: str = "X",
) -> None:
    """Serialize scipy sparse matrix to HDF5 group.

    Stores as three datasets: data, indices, indptr (CSR format).

    Parameters
    ----------
    X : sp.spmatrix
        Sparse matrix to serialize (will be converted to CSR)
    group : h5py.Group
        Target HDF5 group
    name : str, default "X"
        Base name for datasets
    """
    csr = X.tocsr()
    group.create_dataset(f"{name}_data", data=csr.data)
    group.create_dataset(f"{name}_indices", data=csr.indices)
    group.create_dataset(f"{name}_indptr", data=csr.indptr)
    group.attrs[f"{name}_format"] = "csr"
    group.attrs[f"{name}_shape"] = csr.shape
