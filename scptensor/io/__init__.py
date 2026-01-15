"""Data I/O module for ScpTensor.

Provides export and import functionality for ScpContainer, supporting HDF5
format with complete data fidelity.

Main Functions:
    save_hdf5: Export ScpContainer to HDF5 format
    load_hdf5: Import ScpContainer from HDF5 file

Exceptions:
    IOFormatError: File format corruption or version incompatibility
    IOPasswordError: HDF5 file password protection errors
    IOWriteError: Write failures (disk space, permissions, etc.)

Basic Usage:
    >>> from scptensor import save_hdf5, load_hdf5, ScpContainer
    >>>
    >>> # Export container to HDF5
    >>> save_hdf5(container, "analysis_results.h5")
    >>>
    >>> # Import container from HDF5
    >>> loaded = load_hdf5("analysis_results.h5")
    >>>
    >>> # Or use convenience methods on ScpContainer
    >>> container.save("analysis_results.h5")
    >>> loaded = ScpContainer.load("analysis_results.h5")

Selective Export:
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

Version: 1.0
"""

from scptensor.io.exceptions import IOFormatError, IOPasswordError, IOWriteError
from scptensor.io.exporters import save_hdf5
from scptensor.io.importers import load_hdf5
from scptensor.io.serializers import (
    deserialize_dataframe,
    serialize_dataframe,
    serialize_dense_matrix,
    serialize_sparse_matrix,
)

__all__ = [
    "IOFormatError",
    "IOPasswordError",
    "IOWriteError",
    "save_hdf5",
    "load_hdf5",
    "deserialize_dataframe",
    "serialize_dataframe",
    "serialize_dense_matrix",
    "serialize_sparse_matrix",
]
