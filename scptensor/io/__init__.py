"""Data I/O module for ScpTensor.

Provides comprehensive export and import functionality for ScpContainer,
supporting multiple file formats for different use cases.

Supported Formats:
    HDF5: Primary format with complete data fidelity and compression
    NPZ: Native binary format for fast save/load
    CSV: Human-readable directory format for interoperability

Main Functions:
    save_hdf5, load_hdf5: HDF5 format (recommended for production)
    save_npz, load_npz: NPZ format (fast, native to ScpTensor)
    save_csv, load_csv: CSV directory format (human-readable)
    read_diann: DIA-NN mass spectrometry data importer

Exceptions:
    IOFormatError: File format corruption or version incompatibility
    IOPasswordError: HDF5 file password protection errors
    IOWriteError: Write failures (disk space, permissions, etc.)

Basic Usage (HDF5):
    >>> from scptensor import save_hdf5, load_hdf5, ScpContainer
    >>>
    >>> # Export container to HDF5
    >>> save_hdf5(container, "analysis_results.h5")
    >>>
    >>> # Import container from HDF5
    >>> loaded = load_hdf5("analysis_results.h5")

NPZ Format (Fast Native Format):
    >>> from scptensor.io import save_npz, load_npz
    >>>
    >>> # Save to NPZ (ScpTensor native binary format)
    >>> save_npz(container, "results.npz")
    >>>
    >>> # Load from NPZ
    >>> loaded = load_npz("results.npz")

CSV Format (Human-Readable):
    >>> from scptensor.io import save_csv, load_csv
    >>>
    >>> # Save to CSV directory structure
    >>> save_csv(container, "results_dir/", layer_name="X")
    >>>
    >>> # Load from CSV directory
    >>> loaded = load_csv("results_dir/", layer_name="X")

DIA-NN Mass Spectrometry Importer:
    >>> from scptensor.io import read_diann
    >>>
    >>> # Import DIA-NN output
    >>> container = read_diann("diann_output.csv")

Selective Export (HDF5):
    >>> # Export only specific assays and layers
    >>> save_hdf5(
    ...     container,
    ...     "selected_data.h5",
    ...     assays={"proteins": ["log", "imputed"]},
    ...     compression="gzip"
    ... )

Export with Custom Settings:
    >>> # Export with high compression
    >>> save_hdf5(
    ...     container,
    ...     "compressed_results.h5",
    ...     compression="gzip",
    ...     compression_opts=9,
    ...     save_history=True
    ... )

Error Handling:
    >>> from scptensor.io import IOFormatError, IOWriteError
    >>>
    >>> try:
    ...     save_hdf5(container, "/invalid/path/results.h5")
    ... except IOWriteError as e:
    ...     print(f"Write failed: {e}")
    ...
    >>> try:
    ...     load_hdf5("corrupted_file.h5")
    ... except IOFormatError as e:
    ...     print(f"Format error: {e}")

HDF5 Storage Format:
    The HDF5 file contains:
    - /version: Format version string
    - /obs: Sample metadata DataFrame (Polars)
    - /assays/{name}/var: Feature metadata DataFrame
    - /assays/{name}/layers/{layer}/X: Data matrix (dense or sparse CSR)
    - /assays/{name}/layers/{layer}/M: Mask matrix (if present)
    - /provenance/history: Operation history log (if save_history=True)

NPZ Storage Format:
    The NPZ format stores:
    - obs and var as serialized DataFrames
    - X and M matrices for each layer (sparse or dense)
    - Provenance history
    - Links between assays
    - Matrix metadata (confidence scores, detection limits, etc.)

CSV Storage Format:
    The CSV directory contains:
    - obs.csv: Sample metadata
    - assay_{name}_var.csv: Feature metadata per assay
    - assay_{name}_{layer}.csv: Data matrix per assay/layer
    - assay_{name}_{layer}_mask.csv: Mask matrix (if mask=True)
    - metadata.json: Container metadata and history

Version: 1.0
"""

from __future__ import annotations

# Base serialization utilities (internal)
from scptensor.io.base import (
    _deserialize_dataframe,
    _dict_to_sparse,
    _serialize_dataframe,
    _sparse_to_dict,
)

# CSV format support
from scptensor.io.csv import load_csv, read_diann, save_csv

# Exceptions
from scptensor.io.exceptions import IOFormatError, IOPasswordError, IOWriteError

# HDF5 format support
from scptensor.io.hdf5 import (
    deserialize_dataframe,
    load_hdf5,
    save_hdf5,
    serialize_dataframe,
    serialize_dense_matrix,
    serialize_sparse_matrix,
)

# NPZ format support
from scptensor.io.npz import load_npz, save_npz

__all__ = [
    # Exceptions
    "IOFormatError",
    "IOPasswordError",
    "IOWriteError",
    # Base utilities (internal)
    "_sparse_to_dict",
    "_dict_to_sparse",
    "_serialize_dataframe",
    "_deserialize_dataframe",
    # HDF5 format
    "save_hdf5",
    "load_hdf5",
    "serialize_dataframe",
    "deserialize_dataframe",
    "serialize_dense_matrix",
    "serialize_sparse_matrix",
    # NPZ format
    "save_npz",
    "load_npz",
    # CSV format
    "save_csv",
    "load_csv",
    "read_diann",
]
