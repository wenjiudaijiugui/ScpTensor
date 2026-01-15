"""Serialization utilities for ScpTensor data structures."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import h5py
import numpy as np
import polars as pl
import scipy as sp

if TYPE_CHECKING:
    from scptensor.core.structures import ProvenanceLog


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


def serialize_provenance(
    history: list[ProvenanceLog],
    group: h5py.Group,
) -> None:
    """Serialize operation history to HDF5 group.

    Parameters
    ----------
    history : list[ProvenanceLog]
        List of provenance logs
    group : h5py.Group
        Target HDF5 group
    """
    if not history:
        return

    actions = [log.action for log in history]
    timestamps = [log.timestamp for log in history]
    params_json = [json.dumps(log.params) for log in history]
    versions = [log.software_version or "" for log in history]
    descriptions = [log.description or "" for log in history]

    string_dtype = h5py.string_dtype(encoding="utf-8")

    group.create_dataset("action", data=actions, dtype=string_dtype)
    group.create_dataset("timestamp", data=timestamps, dtype=string_dtype)
    group.create_dataset("params", data=params_json, dtype=string_dtype)
    group.create_dataset("software_version", data=versions, dtype=string_dtype)
    group.create_dataset("description", data=descriptions, dtype=string_dtype)


def deserialize_provenance(group: h5py.Group) -> list[ProvenanceLog]:
    """Deserialize operation history from HDF5 group.

    Parameters
    ----------
    group : h5py.Group
        Source HDF5 group

    Returns
    -------
    list[ProvenanceLog]
        Reconstructed history
    """
    from scptensor.core.structures import ProvenanceLog

    if "action" not in group:
        return []

    n = len(group["action"])
    history = []

    for i in range(n):
        # Decode bytes to strings if necessary
        timestamp = group["timestamp"][i]
        if isinstance(timestamp, bytes):
            timestamp = timestamp.decode("utf-8")

        action = group["action"][i]
        if isinstance(action, bytes):
            action = action.decode("utf-8")

        params_bytes = group["params"][i]
        if isinstance(params_bytes, bytes):
            params_bytes = params_bytes.decode("utf-8")

        software_version = group["software_version"][i]
        if isinstance(software_version, bytes):
            software_version = software_version.decode("utf-8")

        description = group["description"][i]
        if isinstance(description, bytes):
            description = description.decode("utf-8")

        log = ProvenanceLog(
            timestamp=timestamp,
            action=action,
            params=json.loads(params_bytes),
            software_version=software_version or None,
            description=description or None,
        )
        history.append(log)

    return history
