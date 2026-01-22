"""Data I/O module for ScpTensor.

Provides comprehensive export and import functionality for ScpContainer,
supporting multiple file formats for different use cases.

Supported Formats:
    HDF5: Primary format with complete data fidelity and compression
    NPZ: Native binary format for fast save/load
    CSV: Human-readable directory format for interoperability
    Scanpy/AnnData: h5ad format for scanpy ecosystem integration
    DIA Formats: Importers for DIA-NN, Spectronaut, MaxQuant, FragPipe

Main Functions:
    save_hdf5, load_hdf5: HDF5 format (recommended for production)
    save_npz, load_npz: NPZ format (fast, native to ScpTensor)
    save_csv, load_csv: CSV directory format (human-readable)
    save_h5ad, load_h5ad: Scanpy/AnnData h5ad format
    from_scanpy, to_scanpy: Direct AnnData object conversion
    read_diann_csv, read_spectronaut_csv: DIA mass spectrometry importers

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

Scanpy/AnnData Integration:
    >>> from scptensor.io import save_h5ad, load_h5ad
    >>>
    >>> # Export to h5ad for scanpy
    >>> save_h5ad(container, "data.h5ad", assay_name="proteins")
    >>>
    >>> # Import from h5ad
    >>> loaded = load_h5ad("data.h5ad", assay_name="proteins")
    >>>
    >>> # Direct AnnData object conversion
    >>> from scptensor.io import from_scanpy, to_scanpy
    >>> import scanpy as sc
    >>>
    >>> # Convert AnnData to ScpContainer
    >>> adata = sc.read_h5ad("data.h5ad")
    >>> container = from_scanpy(adata, assay_name="proteins")
    >>>
    >>> # Convert ScpContainer to AnnData
    >>> adata = to_scanpy(container, assay_name="proteins")

DIA Mass Spectrometry Importers:
    >>> from scptensor.io import read_diann_csv
    >>>
    >>> # Import DIA-NN output
    >>> container = read_diann_csv("diann_output.csv")
    >>>
    >>> # Import Spectronaut output
    >>> from scptensor.io import read_spectronaut_csv
    >>> container = read_spectronaut_csv("spectronaut_output.csv")

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
from scptensor.io.csv import load_csv, save_csv

# Exceptions
from scptensor.io.exceptions import IOFormatError, IOPasswordError, IOWriteError

# DIA mass spectrometry format importers
from scptensor.io.formats import (
    read_diann_csv,
    read_diann_parquet,
    read_fragments_csv,
    read_maxquant_csv,
    read_spectronaut_csv,
)

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

# Scanpy/AnnData format support
from scptensor.io.scanpy import (
    from_scanpy,
    load_h5ad,
    read_h5ad,
    save_h5ad,
    to_scanpy,
    write_h5ad,
)

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
    # Scanpy/AnnData format
    "save_h5ad",
    "load_h5ad",
    "from_scanpy",
    "to_scanpy",
    "read_h5ad",
    "write_h5ad",
    # DIA format importers
    "read_diann_csv",
    "read_diann_parquet",
    "read_spectronaut_csv",
    "read_maxquant_csv",
    "read_fragments_csv",
]
