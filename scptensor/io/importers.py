"""HDF5 import functionality for ScpContainer."""

from __future__ import annotations

from pathlib import Path

import h5py
from scipy import sparse

from scptensor.core.structures import Assay, ScpContainer, ScpMatrix
from scptensor.io.exceptions import IOFormatError
from scptensor.io.serializers import deserialize_dataframe, deserialize_provenance

FORMAT_VERSION = "1.0"
SUPPORTED_VERSIONS = {"1.0"}


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


def _load_matrix_data(group: h5py.Group, name: str):
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
