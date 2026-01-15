"""Export and import functions for ScpContainer data.

This module provides functions to save and load ScpContainer objects to/from
various file formats including CSV, h5ad (AnnData), and NPZ (numpy archive).

All export functions preserve the complete data structure including obs, var,
layers, mask matrices, and provenance history.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import scipy.sparse as sp

from scptensor.core.exceptions import (
    AssayNotFoundError,
    MissingDependencyError,
    ScpTensorError,
    ValidationError,
)
from scptensor.core.structures import (
    Assay,
    AggregationLink,
    MaskCode,
    MatrixMetadata,
    ProvenanceLog,
    ScpContainer,
    ScpMatrix,
)

__all__ = [
    "save_csv",
    "load_csv",
    "save_h5ad",
    "load_h5ad",
    "save_npz",
    "load_npz",
    "from_scanpy",
    "to_scanpy",
    "read_h5ad",
    "write_h5ad",
]

# Scanpy/AnnData availability flag
try:
    import anndata as ad

    ANNDATA_AVAILABLE = True
except ImportError:
    ANNDATA_AVAILABLE = False
    ad = None

# Metadata keys for NPZ format
_NPZ_VERSION_KEY = "_scptensor_version"
_NPZ_METADATA_KEY = "_metadata"
_NPZ_OBS_KEY = "obs"
_NPZ_ASSAYS_PREFIX = "assay_"


def _sparse_to_dict(matrix: sp.spmatrix) -> dict[str, Any]:
    """Convert sparse matrix to serializable dictionary.

    Parameters
    ----------
    matrix : sp.spmatrix
        Sparse matrix (CSR or CSC format).

    Returns
    -------
    dict
        Dictionary with format, data, indices, indptr, and shape.
    """
    if sp.isspmatrix_csr(matrix):
        format_name = "csr"
    elif sp.isspmatrix_csc(matrix):
        format_name = "csc"
    else:
        # Convert to CSR for other formats
        matrix = matrix.tocsr()
        format_name = "csr"

    return {
        "format": format_name,
        "data": matrix.data,
        "indices": matrix.indices,
        "indptr": matrix.indptr,
        "shape": matrix.shape,
    }


def _dict_to_sparse(d: dict[str, Any]) -> sp.spmatrix:
    """Convert dictionary back to sparse matrix.

    Parameters
    ----------
    d : dict
        Dictionary with sparse matrix data.

    Returns
    -------
    sp.spmatrix
        Reconstructed sparse matrix.
    """
    format_name = d["format"]
    data = d["data"]
    indices = d["indices"]
    indptr = d["indptr"]
    shape = tuple(d["shape"])

    if format_name == "csr":
        return sp.csr_matrix((data, indices, indptr), shape=shape)
    elif format_name == "csc":
        return sp.csc_matrix((data, indices, indptr), shape=shape)
    else:
        raise ValueError(f"Unknown sparse format: {format_name}")


def _serialize_dataframe(df: pl.DataFrame) -> dict[str, Any]:
    """Serialize Polars DataFrame to dictionary.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to serialize.

    Returns
    -------
    dict
        Dictionary with columns data and schema.
    """
    return {
        "columns": {name: df[name].to_numpy() for name in df.columns},
        "dtypes": {name: str(dtype) for name, dtype in df.schema.items()},
    }


def _deserialize_dataframe(data: dict[str, Any]) -> pl.DataFrame:
    """Deserialize dictionary back to Polars DataFrame.

    Parameters
    ----------
    data : dict
        Dictionary with DataFrame data.

    Returns
    -------
    pl.DataFrame
        Reconstructed DataFrame.
    """
    return pl.DataFrame(data["columns"])


def save_csv(
    container: ScpContainer,
    path: str | Path,
    *,
    layer_name: str = "X",
    mask: bool = True,
) -> None:
    """Export ScpContainer to CSV files.

    Creates a directory with the following structure:
    ```
    path/
    ├── obs.csv              # Sample metadata
    ├── assay_<name>_var.csv # Feature metadata per assay
    ├── assay_<name>_<layer>.csv  # Data matrix per assay/layer
    ├── assay_<name>_<layer>_mask.csv  # Mask matrix (if mask=True)
    └── metadata.json        # Container metadata
    ```

    Parameters
    ----------
    container : ScpContainer
        Container to export.
    path : str | Path
        Directory path for output files. Will be created if it doesn't exist.
    layer_name : str, optional
        Name of the layer to export. Default is "X".
    mask : bool, optional
        Whether to export mask matrices. Default is True.

    Raises
    ------
    ValidationError
        If layer_name is not found in assays.
    IOError
        If unable to write files.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Save obs as CSV
    obs_path = path / "obs.csv"
    container.obs.write_csv(obs_path)

    # Save metadata
    metadata = {
        "version": "0.1.0",
        "n_samples": container.n_samples,
        "sample_id_col": container.sample_id_col,
        "assays": list(container.assays.keys()),
        "history": [
            {
                "timestamp": log.timestamp,
                "action": log.action,
                "params": log.params,
                "software_version": log.software_version,
                "description": log.description,
            }
            for log in container.history
        ],
    }
    metadata_path = path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Save assays
    for assay_name, assay in container.assays.items():
        # Save var
        var_path = path / f"assay_{assay_name}_var.csv"
        assay.var.write_csv(var_path)

        # Check if layer exists
        if layer_name not in assay.layers:
            raise ValidationError(
                f"Layer '{layer_name}' not found in assay '{assay_name}'"
            )

        matrix = assay.layers[layer_name]

        # Save data matrix
        data_path = path / f"assay_{assay_name}_{layer_name}.csv"

        # Convert sparse to dense for CSV
        if sp.issparse(matrix.X):
            X_dense = matrix.X.toarray()
        else:
            X_dense = matrix.X

        # Create DataFrame with sample and feature IDs
        feature_ids = assay.feature_ids.to_list()
        sample_ids = container.sample_ids.to_list()

        data_dict = {"_index": sample_ids}
        for i, fid in enumerate(feature_ids):
            data_dict[str(fid)] = X_dense[:, i]

        data_df = pl.DataFrame(data_dict)
        data_df.write_csv(data_path)

        # Save mask if present and requested
        if mask and matrix.M is not None:
            mask_path = path / f"assay_{assay_name}_{layer_name}_mask.csv"

            if sp.issparse(matrix.M):
                M_dense = matrix.M.toarray()
            else:
                M_dense = matrix.M

            mask_dict = {"_index": sample_ids}
            for i, fid in enumerate(feature_ids):
                mask_dict[str(fid)] = M_dense[:, i]

            mask_df = pl.DataFrame(mask_dict)
            mask_df.write_csv(mask_path)


def load_csv(
    path: str | Path,
    *,
    layer_name: str = "X",
    mask: bool = True,
) -> ScpContainer:
    """Load ScpContainer from CSV files.

    Reads data previously saved by :func:`save_csv`.

    Parameters
    ----------
    path : str | Path
        Directory path containing the CSV files.
    layer_name : str, optional
        Name of the layer to load. Default is "X".
    mask : bool, optional
        Whether to load mask matrices. Default is True.

    Returns
    -------
    ScpContainer
        Loaded container.

    Raises
    ------
    ValidationError
        If required files are missing.
    IOError
        If unable to read files.
    """
    path = Path(path)

    # Load metadata
    metadata_path = path / "metadata.json"
    if not metadata_path.exists():
        raise ValidationError(f"metadata.json not found in {path}")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Load obs
    obs_path = path / "obs.csv"
    if not obs_path.exists():
        raise ValidationError(f"obs.csv not found in {path}")

    obs = pl.read_csv(obs_path)
    sample_id_col = metadata.get("sample_id_col", "_index")

    # Reconstruct history
    history = []
    for log_entry in metadata.get("history", []):
        history.append(
            ProvenanceLog(
                timestamp=log_entry["timestamp"],
                action=log_entry["action"],
                params=log_entry["params"],
                software_version=log_entry.get("software_version"),
                description=log_entry.get("description"),
            )
        )

    # Load assays
    assays: dict[str, Assay] = {}
    for assay_name in metadata.get("assays", []):
        # Load var
        var_path = path / f"assay_{assay_name}_var.csv"
        if not var_path.exists():
            raise ValidationError(f"assay_{assay_name}_var.csv not found in {path}")

        var = pl.read_csv(var_path)

        # Load data matrix
        data_path = path / f"assay_{assay_name}_{layer_name}.csv"
        if not data_path.exists():
            raise ValidationError(f"assay_{assay_name}_{layer_name}.csv not found in {path}")

        data_df = pl.read_csv(data_path)

        # Extract feature IDs from var
        feature_id_col = "_index" if "_index" in var.columns else var.columns[0]
        feature_ids = var[feature_id_col].to_list()

        # Build data matrix
        # First column is sample IDs, rest are features
        sample_col = data_df.columns[0]
        X_data = data_df.select(data_df.columns[1:]).to_numpy().astype(np.float64)

        # Load mask if present
        M_data = None
        if mask:
            mask_path = path / f"assay_{assay_name}_{layer_name}_mask.csv"

            if mask_path.exists():
                mask_df = pl.read_csv(mask_path)
                M_data = mask_df.select(mask_df.columns[1:]).to_numpy().astype(np.int8)

        assays[assay_name] = Assay(
            var=var,
            layers={layer_name: ScpMatrix(X=X_data, M=M_data)},
            feature_id_col=feature_id_col,
        )

    return ScpContainer(
        obs=obs,
        assays=assays,
        history=history,
        sample_id_col=sample_id_col,
    )


def save_h5ad(
    container: ScpContainer,
    path: str | Path,
    *,
    assay_name: str | None = None,
    layer_name: str = "X",
) -> None:
    """Export ScpContainer to h5ad format (AnnData compatible).

    This function converts a ScpContainer to the AnnData format for
    interoperability with the scanpy ecosystem. The mask matrix is stored
    as a layer called "mask".

    For multi-assay containers, specify which assay to export. If None,
    the first assay is used.

    Parameters
    ----------
    container : ScpContainer
        Container to export.
    path : str | Path
        Path for output h5ad file.
    assay_name : str, optional
        Name of assay to export. If None, uses the first assay.
    layer_name : str, optional
        Name of the layer to use as the main data matrix. Default is "X".

    Raises
    ------
    MissingDependencyError
        If anndata is not installed.
    AssayNotFoundError
        If the specified assay doesn't exist.

    Notes
    -----
    Requires the optional 'anndata' package:
    ``pip install anndata``

    The conversion follows this mapping:
    - ScpContainer.obs -> AnnData.obs
    - ScpContainer.assays[name].var -> AnnData.var
    - ScpMatrix.X -> AnnData.X
    - ScpMatrix.M -> AnnData.layers["mask"]
    - Provenance history -> AnnData.uns["history"]
    """
    try:
        import anndata as ad
    except ImportError:
        raise MissingDependencyError("anndata")

    # Select assay
    if assay_name is None:
        if not container.assays:
            raise ValidationError("Container has no assays")
        assay_name = next(iter(container.assays.keys()))
    elif assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]

    # Check if layer exists
    if layer_name not in assay.layers:
        raise ValidationError(
            f"Layer '{layer_name}' not found in assay '{assay_name}'"
        )

    matrix = assay.layers[layer_name]

    # Convert obs to pandas (AnnData expects pandas)
    # AnnData reserves '_index', so rename to 'obs_names' if needed
    obs_pl = container.obs.clone()
    if container.sample_id_col == "_index":
        # Rename for AnnData compatibility, will restore on load
        obs_pl = obs_pl.rename({"_index": "_original_index"})
    obs_pd = obs_pl.to_pandas()

    # Convert var to pandas (similar handling for feature_id_col)
    var_pl = assay.var.clone()
    if assay.feature_id_col == "_index":
        var_pl = var_pl.rename({"_index": "_original_index"})
    var_pd = var_pl.to_pandas()

    # Create AnnData object
    adata = ad.AnnData(
        X=matrix.X if not sp.issparse(matrix.X) else matrix.X,
        obs=obs_pd,
        var=var_pd,
    )

    # Store mask as layer
    if matrix.M is not None:
        adata.layers["mask"] = matrix.M

    # Store provenance history as JSON string (AnnData can't handle nested dicts well)
    if container.history:
        history_dicts = [
            {
                "timestamp": log.timestamp,
                "action": log.action,
                "params": str(log.params),  # Convert to string for AnnData compatibility
                "software_version": log.software_version,
                "description": log.description,
            }
            for log in container.history
        ]
        import json as _json
        adata.uns["scptensor_history_json"] = _json.dumps(history_dicts)
        adata.uns["scptensor_assay_name"] = assay_name
        adata.uns["scptensor_layer_name"] = layer_name
        adata.uns["scptensor_sample_id_col"] = container.sample_id_col
        adata.uns["scptensor_feature_id_col"] = assay.feature_id_col

    # Save
    adata.write_h5ad(str(path))


def load_h5ad(path: str | Path, *, assay_name: str = "imported") -> ScpContainer:
    """Load ScpContainer from h5ad format.

    Reads an AnnData file and converts it to ScpContainer format.
    The mask layer is detected and restored if present.

    Parameters
    ----------
    path : str | Path
        Path to h5ad file.
    assay_name : str, optional
        Name to use for the imported assay. Default is "imported".

    Returns
    -------
    ScpContainer
        Loaded container.

    Raises
    ------
    MissingDependencyError
        If anndata is not installed.

    Notes
    -----
    Requires the optional 'anndata' package:
    ``pip install anndata``

    The conversion follows this mapping:
    - AnnData.obs -> ScpContainer.obs
    - AnnData.var -> Assay.var
    - AnnData.X -> ScpMatrix.X
    - AnnData.layers["mask"] -> ScpMatrix.M
    - AnnData.uns["history"] -> ProvenanceLog list
    """
    try:
        import anndata as ad
    except ImportError:
        raise MissingDependencyError("anndata")

    # Load AnnData
    adata = ad.read_h5ad(str(path))

    # Convert obs from pandas to polars
    obs = pl.from_pandas(adata.obs)

    # Restore original sample ID column name if saved
    sample_id_col = "_index"
    feature_id_col = "_index"
    if "scptensor_sample_id_col" in adata.uns:
        sample_id_col = adata.uns["scptensor_sample_id_col"]
    if "scptensor_feature_id_col" in adata.uns:
        feature_id_col = adata.uns["scptensor_feature_id_col"]

    # Restore _original_index back to _index if needed
    if sample_id_col == "_index" and "_original_index" in obs.columns:
        obs = obs.rename({"_original_index": "_index"})

    # Convert var from pandas to polars
    var = pl.from_pandas(adata.var)

    # Restore _original_index back to _index if needed
    if feature_id_col == "_index" and "_original_index" in var.columns:
        var = var.rename({"_original_index": "_index"})

    # Get mask from layers
    M = adata.layers.get("mask", None)

    # Create matrix
    matrix = ScpMatrix(X=adata.X, M=M)

    # Create assay
    assay = Assay(
        var=var,
        layers={"X": matrix},
        feature_id_col=feature_id_col,
    )

    # Restore history
    history = []
    if "scptensor_history_json" in adata.uns:
        import json as _json
        history_list = _json.loads(adata.uns["scptensor_history_json"])
        for log_entry in history_list:
            history.append(
                ProvenanceLog(
                    timestamp=log_entry["timestamp"],
                    action=log_entry["action"],
                    params={},  # Params were converted to string, use empty dict
                    software_version=log_entry.get("software_version"),
                    description=log_entry.get("description"),
                )
            )

    # Use original assay name if stored
    if "scptensor_assay_name" in adata.uns:
        assay_name = adata.uns["scptensor_assay_name"]

    return ScpContainer(
        obs=obs,
        assays={assay_name: assay},
        history=history,
        sample_id_col=sample_id_col,
    )


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
    save_dict: dict[str, Any] = {
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
        layers_dict: dict[str, Any] = {}
        for layer_name, matrix in assay.layers.items():
            layer_info: dict[str, Any] = {
                "shape": list(matrix.X.shape),
            }

            # Handle X
            if sp.issparse(matrix.X):
                layer_info["X"] = _sparse_to_dict(matrix.X)
                layer_info["X_is_sparse"] = True
            else:
                layer_info["X"] = matrix.X.tolist() if isinstance(matrix.X, np.ndarray) else matrix.X
                layer_info["X_is_sparse"] = False

            # Handle M
            if matrix.M is not None:
                if sp.issparse(matrix.M):
                    layer_info["M"] = _sparse_to_dict(matrix.M)
                    layer_info["M_is_sparse"] = True
                else:
                    layer_info["M"] = matrix.M.tolist() if isinstance(matrix.M, np.ndarray) else matrix.M
                    layer_info["M_is_sparse"] = False
            else:
                layer_info["M"] = None

            # Handle metadata
            if matrix.metadata is not None:
                metadata_dict: dict[str, Any] = {}
                if matrix.metadata.confidence_scores is not None:
                    if sp.issparse(matrix.metadata.confidence_scores):
                        metadata_dict["confidence_scores"] = _sparse_to_dict(
                            matrix.metadata.confidence_scores
                        )
                        metadata_dict["confidence_scores_is_sparse"] = True
                    else:
                        cs = matrix.metadata.confidence_scores
                        metadata_dict["confidence_scores"] = (
                            cs.tolist() if isinstance(cs, np.ndarray) else cs
                        )
                        metadata_dict["confidence_scores_is_sparse"] = False

                if matrix.metadata.detection_limits is not None:
                    if sp.issparse(matrix.metadata.detection_limits):
                        metadata_dict["detection_limits"] = _sparse_to_dict(
                            matrix.metadata.detection_limits
                        )
                        metadata_dict["detection_limits_is_sparse"] = True
                    else:
                        dl = matrix.metadata.detection_limits
                        metadata_dict["detection_limits"] = (
                            dl.tolist() if isinstance(dl, np.ndarray) else dl
                        )
                        metadata_dict["detection_limits_is_sparse"] = False

                if matrix.metadata.imputation_quality is not None:
                    if sp.issparse(matrix.metadata.imputation_quality):
                        metadata_dict["imputation_quality"] = _sparse_to_dict(
                            matrix.metadata.imputation_quality
                        )
                        metadata_dict["imputation_quality_is_sparse"] = True
                    else:
                        iq = matrix.metadata.imputation_quality
                        metadata_dict["imputation_quality"] = (
                            iq.tolist() if isinstance(iq, np.ndarray) else iq
                        )
                        metadata_dict["imputation_quality_is_sparse"] = False

                if matrix.metadata.outlier_scores is not None:
                    if sp.issparse(matrix.metadata.outlier_scores):
                        metadata_dict["outlier_scores"] = _sparse_to_dict(
                            matrix.metadata.outlier_scores
                        )
                        metadata_dict["outlier_scores_is_sparse"] = True
                    else:
                        os = matrix.metadata.outlier_scores
                        metadata_dict["outlier_scores"] = (
                            os.tolist() if isinstance(os, np.ndarray) else os
                        )
                        metadata_dict["outlier_scores_is_sparse"] = False

                if matrix.metadata.creation_info is not None:
                    metadata_dict["creation_info"] = matrix.metadata.creation_info

                layer_info["metadata"] = metadata_dict

            layers_dict[layer_name] = layer_info

        save_dict[f"{prefix}layers"] = layers_dict

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
    save_dict["history"] = history_list

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
        save_dict["links"] = links_list

    # Save as NPZ
    metadata_json = json.dumps(save_dict, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

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
    data: dict[str, Any] = json.loads(metadata_json)

    # Check version
    version = data.get(_NPZ_VERSION_KEY, "0.0.0")

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
            assay_name = key[len(_NPZ_ASSAYS_PREFIX): -len("_var")]

            # Get var
            var = _deserialize_dataframe(value)
            feature_id_col = data.get(
                f"{_NPZ_ASSAYS_PREFIX}{assay_name}_feature_id_col", "_index"
            )

            # Get layers
            layers_dict = data.get(f"{_NPZ_ASSAYS_PREFIX}{assay_name}_layers", {})
            layers: dict[str, ScpMatrix] = {}

            for layer_name, layer_info in layers_dict.items():
                # Handle X
                if layer_info.get("X_is_sparse", False):
                    X = _dict_to_sparse(layer_info["X"])
                else:
                    X = np.array(layer_info["X"])

                # Handle M
                M = None
                if layer_info.get("M") is not None:
                    if layer_info.get("M_is_sparse", False):
                        M = _dict_to_sparse(layer_info["M"])
                    else:
                        M = np.array(layer_info["M"])

                # Handle metadata
                metadata = None
                if "metadata" in layer_info:
                    meta_dict = layer_info["metadata"]
                    metadata = MatrixMetadata()

                    if "confidence_scores" in meta_dict:
                        cs = meta_dict["confidence_scores"]
                        if meta_dict.get("confidence_scores_is_sparse", False):
                            metadata.confidence_scores = _dict_to_sparse(cs)
                        else:
                            metadata.confidence_scores = np.array(cs)

                    if "detection_limits" in meta_dict:
                        dl = meta_dict["detection_limits"]
                        if meta_dict.get("detection_limits_is_sparse", False):
                            metadata.detection_limits = _dict_to_sparse(dl)
                        else:
                            metadata.detection_limits = np.array(dl)

                    if "imputation_quality" in meta_dict:
                        iq = meta_dict["imputation_quality"]
                        if meta_dict.get("imputation_quality_is_sparse", False):
                            metadata.imputation_quality = _dict_to_sparse(iq)
                        else:
                            metadata.imputation_quality = np.array(iq)

                    if "outlier_scores" in meta_dict:
                        os = meta_dict["outlier_scores"]
                        if meta_dict.get("outlier_scores_is_sparse", False):
                            metadata.outlier_scores = _dict_to_sparse(os)
                        else:
                            metadata.outlier_scores = np.array(os)

                    if "creation_info" in meta_dict:
                        metadata.creation_info = meta_dict["creation_info"]

                layers[layer_name] = ScpMatrix(X=X, M=M, metadata=metadata)

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


def _create_test_container() -> ScpContainer:
    """Create a test container for round-trip testing.

    Returns
    -------
    ScpContainer
        Container with sample data for testing.
    """
    np.random.seed(42)

    n_samples = 10
    n_proteins = 20

    # Create obs
    obs = pl.DataFrame(
        {
            "_index": [f"sample_{i}" for i in range(n_samples)],
            "batch": ["A" if i < 5 else "B" for i in range(n_samples)],
            "n_detected": np.random.randint(100, 200, n_samples),
        }
    )

    # Create protein data
    X = np.random.rand(n_samples, n_proteins) * 10

    # Create mask with some missing values
    M = np.zeros((n_samples, n_proteins), dtype=np.int8)
    M[:2, :5] = MaskCode.MBR  # Some MBR values
    M[3:5, 10:15] = MaskCode.LOD  # Some LOD values
    M[5:7, 15:18] = MaskCode.IMPUTED  # Some imputed values

    # Create var
    var = pl.DataFrame(
        {
            "_index": [f"protein_{i}" for i in range(n_proteins)],
            "n_detected": np.random.randint(5, n_samples, n_proteins),
            "mean_intensity": np.random.rand(n_proteins) * 10,
        }
    )

    # Create assay
    assay = Assay(
        var=var,
        layers={
            "X": ScpMatrix(X=X, M=M),
        },
    )

    # Create container
    container = ScpContainer(
        obs=obs,
        assays={"proteins": assay},
        sample_id_col="_index",
    )

    # Add some history
    container.log_operation(
        action="test_create",
        params={"n_samples": n_samples, "n_proteins": n_proteins},
        description="Created test container",
    )

    return container


def from_scanpy(
    adata: "ad.AnnData",
    assay_name: str = "proteins",
    layer_name: str = "X",
    copy: bool = True,
) -> ScpContainer:
    """Convert Scanpy AnnData to ScpTensor ScpContainer.

    This function provides bidirectional conversion between ScpTensor's
    ScpContainer and Scanpy's AnnData formats, enabling seamless
    interoperability with the scanpy ecosystem.

    Parameters
    ----------
    adata : AnnData
        Scanpy AnnData object to convert.
    assay_name : str, optional
        Name for the assay in the resulting ScpContainer.
        Default is "proteins".
    layer_name : str, optional
        Name for the data layer in the resulting assay.
        Default is "X".
    copy : bool, optional
        Whether to copy data arrays. Default is True.

    Returns
    -------
    ScpContainer
        ScpTensor container with converted data.

    Raises
    ------
    MissingDependencyError
        If anndata is not installed.
    ValidationError
        If input data has invalid structure.

    Examples
    --------
    >>> import scanpy as sc
    >>> from scptensor.core.io import from_scanpy
    >>> adata = sc.read_h5ad("data.h5ad")
    >>> container = from_scanpy(adata, assay_name="proteins")
    >>> print(container)
    """
    if not ANNDATA_AVAILABLE:
        raise MissingDependencyError("anndata")

    # Extract data from AnnData
    X = adata.X

    # Handle sparse vs dense matrices
    if sp.issparse(X):
        X = X.copy() if copy else X
    else:
        X = X.copy() if copy else X

    # Extract mask from layers if present
    M = adata.layers.get("mask", None)
    if M is not None and copy:
        if sp.issparse(M):
            M = M.copy()
        else:
            M = M.copy()

    # Convert obs from pandas to polars
    obs_pl = pl.from_pandas(adata.obs)

    # Handle index column - AnnData stores index separately
    sample_id_col = "_index"
    if "_original_index" in obs_pl.columns:
        # Restored from previous ScpTensor export
        obs_pl = obs_pl.rename({"_original_index": "_index"})
    elif "_index" not in obs_pl.columns:
        # Add index from obs_names
        obs_pl = obs_pl.with_columns(pl.Series("_index", adata.obs_names))

    # Convert var from pandas to polars
    var_pl = pl.from_pandas(adata.var)

    # Handle index column for var
    feature_id_col = "_index"
    if "_original_index" in var_pl.columns:
        # Restored from previous ScpTensor export
        var_pl = var_pl.rename({"_original_index": "_index"})
    elif "_index" not in var_pl.columns:
        # Add index from var_names
        var_pl = var_pl.with_columns(pl.Series("_index", adata.var_names))

    # Create ScpMatrix with metadata
    matrix = ScpMatrix(
        X=X,
        M=M,
        metadata=MatrixMetadata(
            creation_info={
                "source": "scanpy",
                "original_shape": list(X.shape),  # Convert tuple to list for AnnData compatibility
                "is_sparse": sp.issparse(X),
            }
        ),
    )

    # Create Assay
    assay = Assay(
        var=var_pl,
        layers={layer_name: matrix},
        feature_id_col=feature_id_col,
    )

    # Create ScpContainer
    container = ScpContainer(
        obs=obs_pl,
        assays={assay_name: assay},
        sample_id_col=sample_id_col,
    )

    # Restore provenance history if available
    if "scptensor_history_json" in adata.uns:
        import json as _json

        history_list = _json.loads(adata.uns["scptensor_history_json"])
        for log_entry in history_list:
            container.history.append(
                ProvenanceLog(
                    timestamp=log_entry["timestamp"],
                    action=log_entry["action"],
                    params={},  # Params were converted to string
                    software_version=log_entry.get("software_version"),
                    description=log_entry.get("description"),
                )
            )

    # Add provenance log for this conversion
    container.log_operation(
        action="from_scanpy",
        params={
            "assay_name": assay_name,
            "layer_name": layer_name,
            "original_shape": list(X.shape),  # Convert tuple to list for JSON serialization
            "was_sparse": sp.issparse(X),
        },
        description=f"Converted from Scanpy AnnData (shape: {X.shape})",
    )

    return container


def to_scanpy(
    container: ScpContainer,
    assay_name: str | None = None,
    layer_name: str = "X",
) -> "ad.AnnData":
    """Convert ScpTensor ScpContainer to Scanpy AnnData.

    This function converts a ScpContainer to AnnData format for
    interoperability with the scanpy ecosystem.

    Parameters
    ----------
    container : ScpContainer
        ScpTensor container to convert.
    assay_name : str, optional
        Name of the assay to convert. If None, uses the first assay.
    layer_name : str, optional
        Name of the layer to use as the main data matrix.
        Default is "X".

    Returns
    -------
    AnnData
        Scanpy AnnData object with converted data.

    Raises
    ------
    MissingDependencyError
        If anndata is not installed.
    AssayNotFoundError
        If the specified assay doesn't exist.
    LayerNotFoundError
        If the specified layer doesn't exist.

    Examples
    --------
    >>> from scptensor.core.io import to_scanpy
    >>> adata = to_scanpy(container, assay_name="proteins")
    >>> import scanpy as sc
    >>> sc.pl.umap(adata)
    """
    if not ANNDATA_AVAILABLE:
        raise MissingDependencyError("anndata")

    # Select assay
    if assay_name is None:
        if not container.assays:
            raise ValidationError("Container has no assays")
        assay_name = next(iter(container.assays.keys()))
    elif assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]

    # Check if layer exists
    if layer_name not in assay.layers:
        from scptensor.core.exceptions import LayerNotFoundError

        raise LayerNotFoundError(layer_name, assay_name)

    matrix = assay.layers[layer_name]

    # Convert obs to pandas (AnnData expects pandas)
    obs_pl = container.obs.clone()
    if container.sample_id_col == "_index":
        # Rename for AnnData compatibility
        obs_pl = obs_pl.rename({"_index": "_original_index"})
    obs_pd = obs_pl.to_pandas()

    # Convert var to pandas
    var_pl = assay.var.clone()
    if assay.feature_id_col == "_index":
        var_pl = var_pl.rename({"_index": "_original_index"})
    var_pd = var_pl.to_pandas()

    # Create AnnData object
    adata = ad.AnnData(
        X=matrix.X,
        obs=obs_pd,
        var=var_pd,
    )

    # Store mask as layer
    if matrix.M is not None:
        adata.layers["mask"] = matrix.M

    # Store provenance history
    if container.history:
        history_dicts = [
            {
                "timestamp": log.timestamp,
                "action": log.action,
                "params": str(log.params),
                "software_version": log.software_version,
                "description": log.description,
            }
            for log in container.history
        ]
        import json as _json

        adata.uns["scptensor_history_json"] = _json.dumps(history_dicts)
        adata.uns["scptensor_assay_name"] = assay_name
        adata.uns["scptensor_layer_name"] = layer_name
        adata.uns["scptensor_sample_id_col"] = container.sample_id_col
        adata.uns["scptensor_feature_id_col"] = assay.feature_id_col

    # Store matrix metadata if available
    if matrix.metadata is not None:
        metadata_dict = {}
        if matrix.metadata.creation_info is not None:
            metadata_dict["creation_info"] = matrix.metadata.creation_info
        if metadata_dict:
            adata.uns["scptensor_matrix_metadata"] = metadata_dict

    return adata


def read_h5ad(
    path: str | Path,
    assay_name: str = "imported",
    layer_name: str = "X",
    **kwargs,
) -> ScpContainer:
    """Read h5ad file and convert to ScpContainer.

    This is a convenience function that combines reading an h5ad file
    with conversion to ScpContainer format.

    Parameters
    ----------
    path : str | Path
        Path to h5ad file.
    assay_name : str, optional
        Name to use for the imported assay. Default is "imported".
    layer_name : str, optional
        Name for the data layer. Default is "X".
    **kwargs
        Additional arguments passed to anndata.read_h5ad.

    Returns
    -------
    ScpContainer
        ScpTensor container with data from the h5ad file.

    Raises
    ------
    MissingDependencyError
        If anndata is not installed.

    Examples
    --------
    >>> from scptensor.core.io import read_h5ad
    >>> container = read_h5ad("data.h5ad", assay_name="proteins")
    >>> print(container)
    """
    if not ANNDATA_AVAILABLE:
        raise MissingDependencyError("anndata")

    adata = ad.read_h5ad(str(path), **kwargs)
    return from_scanpy(adata, assay_name=assay_name, layer_name=layer_name)


def write_h5ad(
    container: ScpContainer,
    path: str | Path,
    *,
    assay_name: str | None = None,
    layer_name: str = "X",
    **kwargs,
) -> None:
    """Write ScpContainer to h5ad file.

    This is a convenience function that converts a ScpContainer to
    AnnData format and writes it to an h5ad file.

    Parameters
    ----------
    container : ScpContainer
        ScpTensor container to write.
    path : str | Path
        Output path for the h5ad file.
    assay_name : str, optional
        Name of the assay to export. If None, uses the first assay.
    layer_name : str, optional
        Name of the layer to export. Default is "X".
    **kwargs
        Additional arguments passed to AnnData.write_h5ad.

    Raises
    ------
    MissingDependencyError
        If anndata is not installed.
    AssayNotFoundError
        If the specified assay doesn't exist.

    Examples
    --------
    >>> from scptensor.core.io import write_h5ad
    >>> write_h5ad(container, "output.h5ad", assay_name="proteins")
    """
    if not ANNDATA_AVAILABLE:
        raise MissingDependencyError("anndata")

    adata = to_scanpy(container, assay_name=assay_name, layer_name=layer_name)
    adata.write_h5ad(str(path), **kwargs)


if __name__ == "__main__":
    import tempfile
    import sys

    print("Testing ScpTensor I/O functions...")
    print()

    # Create test container
    print("Creating test container...")
    container = _create_test_container()
    print(f"  Created container: {container}")
    print()

    # Test CSV save/load
    print("Testing CSV save/load...")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save
        save_csv(container, tmpdir)
        print(f"  Saved CSV to {tmpdir}")

        # Load
        loaded = load_csv(tmpdir)
        print(f"  Loaded CSV: {loaded}")

        # Verify
        assert loaded.n_samples == container.n_samples
        assert loaded.assays["proteins"].n_features == container.assays[
            "proteins"
        ].n_features
        assert len(loaded.history) == len(container.history)
        print("  CSV round-trip verified")
    print()

    # Test NPZ save/load
    print("Testing NPZ save/load...")
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = Path(tmpdir) / "test.npz"

        # Save
        save_npz(container, npz_path)
        print(f"  Saved NPZ to {npz_path}")

        # Load
        loaded = load_npz(npz_path)
        print(f"  Loaded NPZ: {loaded}")

        # Verify
        assert loaded.n_samples == container.n_samples
        assert loaded.assays["proteins"].n_features == container.assays[
            "proteins"
        ].n_features
        assert len(loaded.history) == len(container.history)
        print("  NPZ round-trip verified")
    print()

    # Test h5ad save/load (if available)
    print("Testing h5ad save/load...")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            h5ad_path = Path(tmpdir) / "test.h5ad"

            # Save
            save_h5ad(container, h5ad_path)
            print(f"  Saved h5ad to {h5ad_path}")

            # Load
            loaded = load_h5ad(h5ad_path)
            print(f"  Loaded h5ad: {loaded}")

            # Verify
            assert loaded.n_samples == container.n_samples
            # Get the first (and only) assay from loaded
            loaded_assay_name = list(loaded.assays.keys())[0]
            assert loaded.assays[loaded_assay_name].n_features == container.assays[
                "proteins"
            ].n_features
            print("  h5ad round-trip verified")
    except MissingDependencyError:
        print("  Skipped (anndata not installed)")
    print()

    # Test sparse matrices
    print("Testing with sparse matrices...")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create container with sparse data
        X_sparse = sp.random(10, 20, density=0.3, format="csr", random_state=42)
        M_sparse = sp.csr_matrix(
            (np.ones(20), (np.array([0, 1, 2, 3, 4] * 4), np.arange(20))),
            shape=(10, 20),
        )
        M_sparse = M_sparse.astype(np.int8)

        obs_sparse = pl.DataFrame(
            {
                "_index": [f"sample_{i}" for i in range(10)],
                "batch": ["A"] * 5 + ["B"] * 5,
            }
        )

        var_sparse = pl.DataFrame(
            {
                "_index": [f"protein_{i}" for i in range(20)],
                "n_detected": np.random.randint(1, 10, 20),
            }
        )

        container_sparse = ScpContainer(
            obs=obs_sparse,
            assays={
                "proteins": Assay(
                    var=var_sparse, layers={"X": ScpMatrix(X=X_sparse, M=M_sparse)}
                )
            },
        )

        # Save and load
        npz_path = Path(tmpdir) / "sparse.npz"
        save_npz(container_sparse, npz_path)
        loaded_sparse = load_npz(npz_path)

        # Verify sparsity preserved
        assert sp.issparse(loaded_sparse.assays["proteins"].layers["X"].X)
        assert sp.issparse(loaded_sparse.assays["proteins"].layers["X"].M)
        print("  Sparse matrix NPZ round-trip verified")
    print()

    # Test multi-assay container
    print("Testing multi-assay container...")
    with tempfile.TemporaryDirectory() as tmpdir:
        obs_multi = pl.DataFrame(
            {"_index": [f"s_{i}" for i in range(5)], "batch": ["A"] * 3 + ["B"] * 2}
        )

        assay1 = Assay(
            var=pl.DataFrame({"_index": ["p1", "p2", "p3"], "mean": [1.0, 2.0, 3.0]}),
            layers={"X": ScpMatrix(X=np.random.rand(5, 3))},
        )

        assay2 = Assay(
            var=pl.DataFrame({"_index": ["pep1", "pep2"], "mean": [0.5, 1.5]}),
            layers={"X": ScpMatrix(X=np.random.rand(5, 2))},
        )

        container_multi = ScpContainer(
            obs=obs_multi, assays={"proteins": assay1, "peptides": assay2}
        )

        npz_path = Path(tmpdir) / "multi.npz"
        save_npz(container_multi, npz_path)
        loaded_multi = load_npz(npz_path)

        assert "proteins" in loaded_multi.assays
        assert "peptides" in loaded_multi.assays
        assert loaded_multi.assays["proteins"].n_features == 3
        assert loaded_multi.assays["peptides"].n_features == 2
        print("  Multi-assay NPZ round-trip verified")
    print()

    # Test metadata preservation
    print("Testing metadata preservation...")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create container with metadata
        confidence = np.random.rand(10, 20)
        X_meta = np.random.rand(10, 20)

        matrix_meta = ScpMatrix(
            X=X_meta,
            metadata=MatrixMetadata(
                confidence_scores=confidence,
                creation_info={"method": "test", "timestamp": "2025-01-01"},
            ),
        )

        container_meta = ScpContainer(
            obs=pl.DataFrame({"_index": [f"s_{i}" for i in range(10)]}),
            assays={
                "test": Assay(
                    var=pl.DataFrame(
                        {"_index": [f"f_{i}" for i in range(20)], "mean": np.random.rand(20)}
                    ),
                    layers={"X": matrix_meta},
                )
            },
        )

        npz_path = Path(tmpdir) / "meta.npz"
        save_npz(container_meta, npz_path)
        loaded_meta = load_npz(npz_path)

        loaded_matrix = loaded_meta.assays["test"].layers["X"]
        assert loaded_matrix.metadata is not None
        assert loaded_matrix.metadata.confidence_scores is not None
        assert loaded_matrix.metadata.creation_info is not None
        assert loaded_matrix.metadata.creation_info["method"] == "test"
        print("  Metadata preservation verified")
    print()

    # Test data integrity
    print("Testing data integrity...")
    with tempfile.TemporaryDirectory() as tmpdir:
        test_X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        test_M = np.array([[0, 1, 2], [0, 0, 5]], dtype=np.int8)

        test_container = ScpContainer(
            obs=pl.DataFrame({"_index": ["s1", "s2"]}),
            assays={
                "test": Assay(
                    var=pl.DataFrame({"_index": ["f1", "f2", "f3"]}),
                    layers={"X": ScpMatrix(X=test_X, M=test_M)},
                )
            },
        )

        npz_path = Path(tmpdir) / "integrity.npz"
        save_npz(test_container, npz_path)
        loaded = load_npz(npz_path)

        loaded_X = loaded.assays["test"].layers["X"].X
        loaded_M = loaded.assays["test"].layers["X"].M

        assert np.allclose(loaded_X, test_X), "X data mismatch"
        assert np.array_equal(loaded_M, test_M), "M data mismatch"
        print("  Data integrity verified (X and M preserved exactly)")
    print()

    # Test from_scanpy and to_scanpy conversion
    print("Testing Scanpy/AnnData conversion...")
    if ANNDATA_AVAILABLE:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test AnnData object
            n_obs = 10
            n_vars = 20
            test_X = np.random.rand(n_obs, n_vars) * 10
            test_obs_pd = pl.DataFrame(
                {
                    "_index": [f"sample_{i}" for i in range(n_obs)],
                    "batch": ["A" if i < 5 else "B" for i in range(n_obs)],
                }
            ).to_pandas()
            test_var_pd = pl.DataFrame(
                {
                    "_index": [f"protein_{i}" for i in range(n_vars)],
                    "n_detected": np.random.randint(5, n_obs, n_vars),
                }
            ).to_pandas()

            # Create AnnData
            adata_orig = ad.AnnData(
                X=test_X,
                obs=test_obs_pd,
                var=test_var_pd,
            )

            # Add some mask layer
            test_M = np.zeros((n_obs, n_vars), dtype=np.int8)
            test_M[:2, :5] = 1  # Some MBR values
            adata_orig.layers["mask"] = test_M

            # Test from_scanpy
            print("  Testing from_scanpy...")
            container = from_scanpy(adata_orig, assay_name="test_proteins")
            assert container.n_samples == n_obs
            assert container.assays["test_proteins"].n_features == n_vars
            assert "_index" in container.obs.columns
            print("    from_scanpy conversion successful")

            # Test to_scanpy
            print("  Testing to_scanpy...")
            adata_converted = to_scanpy(container, assay_name="test_proteins")
            assert adata_converted.shape == (n_obs, n_vars)
            assert "mask" in adata_converted.layers
            assert np.array_equal(adata_converted.X, adata_orig.X)
            print("    to_scanpy conversion successful")

            # Test round-trip conversion
            print("  Testing round-trip conversion...")
            container_back = from_scanpy(adata_converted, assay_name="round_trip")
            assert container_back.n_samples == container.n_samples
            assert container_back.assays["round_trip"].n_features == container.assays[
                "test_proteins"
            ].n_features
            print("    Round-trip conversion successful")

            # Test read_h5ad and write_h5ad
            print("  Testing read_h5ad/write_h5ad...")
            h5ad_path = Path(tmpdir) / "test_scanpy.h5ad"
            write_h5ad(container, h5ad_path, assay_name="test_proteins")
            container_loaded = read_h5ad(h5ad_path, assay_name="loaded")
            assert container_loaded.n_samples == n_obs
            assert "loaded" in container_loaded.assays
            print("    read_h5ad/write_h5ad successful")

            # Test with sparse matrices
            print("  Testing with sparse matrices...")
            X_sparse = sp.random(n_obs, n_vars, density=0.3, format="csr", random_state=42)
            adata_sparse = ad.AnnData(
                X=X_sparse,
                obs=test_obs_pd,
                var=test_var_pd,
            )
            container_sparse = from_scanpy(adata_sparse, copy=True)
            assert sp.issparse(container_sparse.assays["proteins"].layers["X"].X)
            print("    Sparse matrix conversion successful")

            # Test to_scanpy with sparse
            adata_sparse_back = to_scanpy(container_sparse)
            assert sp.issparse(adata_sparse_back.X)
            print("    Sparse round-trip successful")

            print("  All Scanpy conversion tests passed")
    else:
        print("  Skipped (anndata not installed)")
    print()

    print("=" * 60)
    print("All I/O tests passed successfully!")
    print("=" * 60)
    sys.exit(0)
