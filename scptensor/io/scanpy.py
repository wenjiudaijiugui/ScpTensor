"""Scanpy/AnnData format I/O for ScpContainer.

Provides bidirectional conversion with Scanpy's AnnData format,
enabling interoperability with the scanpy ecosystem.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import scipy.sparse as sp

from scptensor.core.exceptions import (
    AssayNotFoundError,
    MissingDependencyError,
    ValidationError,
)
from scptensor.core.structures import (
    Assay,
    MatrixMetadata,
    ProvenanceLog,
    ScpContainer,
    ScpMatrix,
)

__all__ = [
    "save_h5ad",
    "load_h5ad",
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
        raise ValidationError(f"Layer '{layer_name}' not found in assay '{assay_name}'")

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
    if not ANNDATA_AVAILABLE:
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


def from_scanpy(
    adata: ad.AnnData,
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
) -> ad.AnnData:
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
