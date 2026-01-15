"""HDF5 export functionality for ScpContainer."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import h5py

from scptensor.core.structures import ScpContainer
from scptensor.io.exceptions import IOWriteError
from scptensor.io.serializers import (
    serialize_dataframe,
    serialize_dense_matrix,
    serialize_provenance,
    serialize_sparse_matrix,
)

FORMAT_VERSION = "1.0"


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
        f.attrs["scptensor_version"] = "0.2.0"
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
    assay,
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
        from scipy import sparse

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
