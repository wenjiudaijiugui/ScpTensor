"""HDF5 format support for ScpTensor I/O.

This module provides import/export functionality for ScpContainer data
in HDF5 format, with support for compression and selective data export.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import polars as pl
import scipy as sp
from scipy import sparse

from scptensor import __version__ as scptensor_version
from scptensor.core.structures import Assay, ProvenanceLog, ScpContainer, ScpMatrix
from scptensor.io.exceptions import IOFormatError, IOWriteError

__all__ = [
    "save_hdf5",
    "load_hdf5",
]

FORMAT_VERSION = "1.0"
SUPPORTED_VERSIONS = {"1.0"}


# ============================================================================
# Serialization Functions
# ============================================================================


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


# ============================================================================
# Export Functions
# ============================================================================


def save_hdf5(
    container: ScpContainer,
    path: str | Path,
    *,
    compression: str | None = "gzip",
    compression_level: int = 4,
    save_obs: bool = True,
    save_assays: list[str] | None = None,
    save_layers: list[str] | None = None,
    save_history: bool = True,
    overwrite: bool = False,
) -> None:
    """
    Export ScpContainer to HDF5 format.

    Parameters
    ----------
    container : ScpContainer
        Container to export
    path : str | Path
        Output file path (.h5)
    compression : str | None, default "gzip"
        Compression algorithm (gzip/lzf/blosc/None)
    compression_level : int, default 4
        Compression level (0-9)
    save_obs : bool, default True
        Whether to save sample metadata
    save_assays : list[str] | None, default None
        Assays to save, None = all
    save_layers : list[str] | None, default None
        Layers to save per assay, None = all
    save_history : bool, default True
        Whether to save operation history
    overwrite : bool, default False
        Whether to overwrite existing file

    Raises
    ------
    IOWriteError
        If file exists and overwrite=False

    Returns
    -------
    None
    """
    path = Path(path)

    # Check file existence
    if path.exists() and not overwrite:
        raise IOWriteError("File already exists", path=str(path))

    # Determine which assays to save
    assays_to_save = save_assays if save_assays is not None else list(container.assays.keys())

    # Create file
    with h5py.File(path, "w") as f:
        # Write attributes
        f.attrs["format_version"] = FORMAT_VERSION
        f.attrs["scptensor_version"] = scptensor_version
        f.attrs["creation_time"] = datetime.now().isoformat()

        # Save obs
        if save_obs:
            obs_group = f.create_group("obs")
            serialize_dataframe(container.obs, obs_group)

        # Save assays
        assays_group = f.create_group("assays")
        for assay_name in assays_to_save:
            if assay_name not in container.assays:
                continue
            _save_assay(
                container.assays[assay_name],
                assays_group.create_group(assay_name),
                save_layers=save_layers,
                compression=compression,
                compression_level=compression_level,
            )

        # Save history
        if save_history and container.history:
            prov_group = f.create_group("provenance")
            history_group = prov_group.create_group("history")
            serialize_provenance(container.history, history_group)


def _save_assay(
    assay: Assay,
    group: h5py.Group,
    save_layers: list[str] | None = None,
    compression: str | None = "gzip",
    compression_level: int = 4,
) -> None:
    """Save a single assay to HDF5 group.

    Parameters
    ----------
    assay : Assay
        Assay object to save
    group : h5py.Group
        Target HDF5 group
    save_layers : list[str] | None
        Layers to save, None = all
    compression : str | None
        Compression algorithm
    compression_level : int
        Compression level
    """
    # Save var (feature metadata)
    var_group = group.create_group("var")
    serialize_dataframe(assay.var, var_group)

    # Save layers
    layers_to_save = save_layers if save_layers is not None else list(assay.layers.keys())
    layers_group = group.create_group("layers")

    for layer_name in layers_to_save:
        if layer_name not in assay.layers:
            continue

        layer_group = layers_group.create_group(layer_name)
        matrix = assay.layers[layer_name]

        # Save X
        if sparse.issparse(matrix.X):
            serialize_sparse_matrix(matrix.X, layer_group, "X")
        else:
            serialize_dense_matrix(matrix.X, layer_group, "X", compression, compression_level)

        # Save M if present
        if matrix.M is not None:
            if sparse.issparse(matrix.M):
                serialize_sparse_matrix(matrix.M, layer_group, "M")
            else:
                serialize_dense_matrix(matrix.M, layer_group, "M", compression=None)


# ============================================================================
# Import Functions
# ============================================================================


def _is_compatible_version(version: str) -> bool:
    """Check if format version is supported.

    Parameters
    ----------
    version : str
        Format version string

    Returns
    -------
    bool
        True if version is supported
    """
    return version in SUPPORTED_VERSIONS


def _validate_hdf5_structure(file: h5py.File) -> None:
    """Verify HDF5 file is valid ScpTensor format.

    Parameters
    ----------
    file : h5py.File
        Open HDF5 file to validate

    Raises
    ------
    IOFormatError
        If required groups are missing or version is incompatible
    """
    required_groups = ["obs", "assays"]
    for group in required_groups:
        if group not in file:
            raise IOFormatError(f"Missing required group: {group}")

    version = file.attrs.get("format_version", "0.0")
    if not _is_compatible_version(version):
        raise IOFormatError(f"Version {version} not supported")


def load_hdf5(path: str | Path) -> ScpContainer:
    """
    Load ScpContainer from HDF5 file.

    Parameters
    ----------
    path : str | Path
        Path to HDF5 file

    Returns
    -------
    ScpContainer
        Loaded container

    Raises
    ------
    IOFormatError
        If file format is invalid or incompatible
    """
    path = Path(path)

    with h5py.File(path, "r") as f:
        _validate_hdf5_structure(f)

        # Load obs
        obs = deserialize_dataframe(f["obs"])

        # Load assays
        assays = {}
        for assay_name in f["assays"]:
            assay_group = f["assays"][assay_name]
            assays[assay_name] = _load_assay(assay_group)

        # Load history
        history = []
        if "provenance" in f and "history" in f["provenance"]:
            history = deserialize_provenance(f["provenance"]["history"])

    return ScpContainer(obs=obs, assays=assays, history=history)


def _load_assay(group: h5py.Group) -> Assay:
    """Load assay from HDF5 group.

    Parameters
    ----------
    group : h5py.Group
        HDF5 group containing assay data

    Returns
    -------
    Assay
        Loaded assay
    """
    # Load var
    var = deserialize_dataframe(group["var"])

    # Load layers
    layers = {}
    if "layers" in group:
        for layer_name in group["layers"]:
            layer_group = group["layers"][layer_name]
            layers[layer_name] = _load_matrix(layer_group)

    return Assay(var=var, layers=layers)


def _load_matrix(group: h5py.Group) -> ScpMatrix:
    """Load ScpMatrix from HDF5 group.

    Parameters
    ----------
    group : h5py.Group
        HDF5 group containing matrix data

    Returns
    -------
    ScpMatrix
        Loaded matrix with X and optional M
    """
    # Load X
    X = _load_matrix_data(group, "X")  # noqa: N806

    # Load M (optional)
    M = None  # noqa: N806
    if "M" in group or "M_data" in group:
        M = _load_matrix_data(group, "M")  # noqa: N806

    return ScpMatrix(X=X, M=M)


def _load_matrix_data(group: h5py.Group, name: str) -> np.ndarray | sp.csr_matrix:
    """Load matrix data (dense or sparse) from HDF5.

    Parameters
    ----------
    group : h5py.Group
        HDF5 group containing matrix datasets
    name : str
        Base name for matrix datasets (X or M)

    Returns
    -------
    np.ndarray or sp.csr_matrix
        Loaded matrix data
    """
    # Check for sparse format
    if f"{name}_data" in group:
        # CSR sparse format
        data = group[f"{name}_data"][:]
        indices = group[f"{name}_indices"][:]
        indptr = group[f"{name}_indptr"][:]
        shape = tuple(group.attrs.get(f"{name}_shape", (len(indptr) - 1, len(indices))))
        return sparse.csr_matrix((data, indices, indptr), shape=shape)
    elif name in group:
        # Dense format
        return group[name][:]
    else:
        msg = f"Matrix data '{name}' not found in group"
        raise ValueError(msg)
