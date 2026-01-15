"""Data I/O module for ScpTensor.

Provides export and import functionality for ScpContainer,
supporting HDF5 and Parquet formats with complete data fidelity.
"""

from scptensor.io.exceptions import IOFormatError, IOPasswordError, IOWriteError
from scptensor.io.exporters import save_hdf5
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
    "deserialize_dataframe",
    "serialize_dataframe",
    "serialize_dense_matrix",
    "serialize_sparse_matrix",
]
