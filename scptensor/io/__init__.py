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

Example:
    >>> from scptensor import save_hdf5, load_hdf5, ScpContainer
    >>>
    >>> # Export container
    >>> save_hdf5(container, "analysis_results.h5")
    >>>
    >>> # Import container
    >>> loaded = load_hdf5("analysis_results.h5")
    >>>
    >>> # Or use convenience methods
    >>> container.save("analysis_results.h5")
    >>> loaded = ScpContainer.load("analysis_results.h5")

HDF5 Storage Format:
    The HDF5 file contains:
    - /obs: Sample metadata DataFrame
    - /assays/{name}/var: Feature metadata DataFrame
    - /assays/{name}/layers/{layer}/X: Data matrix (dense or sparse CSR)
    - /assays/{name}/layers/{layer}/M: Mask matrix (if present)
    - /provenance/history: Operation history log

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
