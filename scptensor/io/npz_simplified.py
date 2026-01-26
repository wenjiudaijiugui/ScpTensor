"""NPZ (numpy archive) format I/O for ScpContainer.

Provides save and load functions for NPZ format, a ScpTensor-native
binary format that preserves all data including sparse matrices,
masks, and provenance history.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from scptensor.core.exceptions import ValidationError
from scptensor.core.structures import (
    AggregationLink,
    Assay,
    ProvenanceLog,
    ScpContainer,
    ScpMatrix,
)
from scptensor.core.types import SerializableDict
from scptensor.io.base import (
    _deserialize_dataframe,
    _dict_to_sparse,
    _serialize_dataframe,
    _sparse_to_dict,
)

__all__ = [
    "save_npz",
    "load_npz",
]

# Metadata keys for NPZ format
_NPZ_VERSION_KEY = "_scptensor_version"
_NPZ_METADATA_KEY = "_metadata"
_NPZ_OBS_KEY = "obs"
_NPZ_ASSAYS_PREFIX = "assay_"


def save_npz(
    container: ScpContainer,
    path: str | Path,
    *,
    compressed: bool = True,
) -> None:
    """Export ScpContainer to NPZ (numpy archive) format.

    This is a ScpTensor-native binary format that preserves all data
    including sparse matrices, masks, and provenance history.

    Parameters
    ----------
    container : ScpContainer
        Container to export.
    path : str | Path
        Path for output NPZ file.
    compressed : bool, optional
        Whether to use compressed NPZ format. Default is True.

    Raises
    ------
    IOError
        If unable to write file.

    Notes
    -----
    The NPZ format stores:
    - obs as serialized DataFrame
    - var for each assay
    - X and M matrices for each layer (sparse or dense)
    - Provenance history
    - Links between assays
    """
    path = Path(path)

    # Prepare data dictionary
    save_dict: SerializableDict = {
        _NPZ_VERSION_KEY: "0.1.0",
    }

    # Serialize obs
    save_dict[_NPZ_OBS_KEY] = _serialize_dataframe(container.obs)
    save_dict["sample_id_col"] = container.sample_id_col

    # Serialize assays
    for assay_name, assay in container.assays.items():
        prefix = f"{_NPZ_ASSAYS_PREFIX}{assay_name}_"

        # Serialize var
        save_dict[f"{prefix}var"] = _serialize_dataframe(assay.var)
        save_dict[f"{prefix}feature_id_col"] = assay.feature_id_col

        # Serialize layers
        layers_dict: dict[str, SerializableDict] = {}
        for layer_name, matrix in assay.layers.items():
            layer_info: SerializableDict = {
                "shape": list(matrix.X.shape),
            }

            # Handle X
            if sp.issparse(matrix.X):
                layer_info["X"] = _sparse_to_dict(matrix.X)
                layer_info["X_is_sparse"] = True
            else:
                layer_info["X"] = (
                    matrix.X.tolist() if isinstance(matrix.X, np.ndarray) else matrix.X
                )
                layer_info["X_is_sparse"] = False

            # Handle M
            if matrix.M is not None:
                if sp.issparse(matrix.M):
                    layer_info["M"] = _sparse_to_dict(matrix.M)
                    layer_info["M_is_sparse"] = True
                else:
                    layer_info["M"] = (
                        matrix.M.tolist() if isinstance(matrix.M, np.ndarray) else matrix.M
                    )
                    layer_info["M_is_sparse"] = False
            else:
                layer_info["M"] = None

            # Skip metadata serialization (optional field)

            layers_dict[layer_name] = layer_info

        save_dict[f"{prefix}layers"] = layers_dict  # type: ignore[assignment]

    # Serialize history
    history_list = [
        {
            "timestamp": log.timestamp,
            "action": log.action,
            "params": log.params,
            "software_version": log.software_version,
            "description": log.description,
        }
        for log in container.history
    ]
    save_dict["history"] = history_list  # type: ignore[assignment]

    # Serialize links
    if container.links:
        links_list = [
            {
                "source_assay": link.source_assay,
                "target_assay": link.target_assay,
                "linkage": _serialize_dataframe(link.linkage),
            }
            for link in container.links
        ]
        save_dict["links"] = links_list  # type: ignore[assignment]

    # Save as NPZ
    metadata_json = json.dumps(
        save_dict, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x
    )

    npz_dict: dict[str, Any] = {
        _NPZ_METADATA_KEY: np.array(metadata_json, dtype=object),
    }

    if compressed:
        np.savez_compressed(path, **npz_dict)
    else:
        np.savez(path, **npz_dict)


def load_npz(path: str | Path) -> ScpContainer:
    """Load ScpContainer from NPZ format.

    Reads a file previously saved by :func:`save_npz`.

    Parameters
    ----------
    path : str | Path
        Path to NPZ file.

    Returns
    -------
    ScpContainer
        Loaded container.

    Raises
    ------
    ValidationError
        If file is malformed or missing required data.
    IOError
        If unable to read file.
    """
    path = Path(path)

    # Load NPZ file
    npz = np.load(path, allow_pickle=True)

    # Get metadata JSON
    if _NPZ_METADATA_KEY not in npz:
        raise ValidationError("Invalid NPZ file: missing metadata")

    metadata_json = str(npz[_NPZ_METADATA_KEY].item())
    data: dict[str, Any] = json.loads(metadata_json)  # Complex nested structure

    # Check version
    data.get(_NPZ_VERSION_KEY, "0.0.0")

    # Deserialize obs
    obs_json = data[_NPZ_OBS_KEY]
    obs = _deserialize_dataframe(obs_json)
    sample_id_col = data.get("sample_id_col", "_index")

    # Deserialize history
    history = []
    for log_entry in data.get("history", []):
        history.append(
            ProvenanceLog(
                timestamp=log_entry["timestamp"],
                action=log_entry["action"],
                params=log_entry["params"],
                software_version=log_entry.get("software_version"),
                description=log_entry.get("description"),
            )
        )

    # Deserialize assays
    assays: dict[str, Assay] = {}
    for key, value in data.items():
        if key.startswith(_NPZ_ASSAYS_PREFIX) and key.endswith("_var"):
            # Extract assay name
            assay_name = key[len(_NPZ_ASSAYS_PREFIX) : -len("_var")]

            # Get var
            var = _deserialize_dataframe(value)
            feature_id_col = data.get(f"{_NPZ_ASSAYS_PREFIX}{assay_name}_feature_id_col", "_index")

            # Get layers
            layers_dict = data.get(f"{_NPZ_ASSAYS_PREFIX}{assay_name}_layers", {})
            layers: dict[str, ScpMatrix] = {}

            for layer_name, layer_info in layers_dict.items():
                # Handle X
                if layer_info.get("X_is_sparse", False):
                    x = _dict_to_sparse(layer_info["X"])
                else:
                    x = np.array(layer_info["X"])

                # Handle M
                m = None
                if layer_info.get("M") is not None:
                    if layer_info.get("M_is_sparse", False):
                        m = _dict_to_sparse(layer_info["M"])
                    else:
                        m = np.array(layer_info["M"])

                # Skip metadata deserialization
                metadata = None

                layers[layer_name] = ScpMatrix(X=x, M=m, metadata=metadata)

            assays[assay_name] = Assay(
                var=var,
                layers=layers,
                feature_id_col=feature_id_col,
            )

    # Deserialize links
    links = []
    for link_entry in data.get("links", []):
        links.append(
            AggregationLink(
                source_assay=link_entry["source_assay"],
                target_assay=link_entry["target_assay"],
                linkage=_deserialize_dataframe(link_entry["linkage"]),
            )
        )

    return ScpContainer(
        obs=obs,
        assays=assays,
        links=links,
        history=history,
        sample_id_col=sample_id_col,
    )
