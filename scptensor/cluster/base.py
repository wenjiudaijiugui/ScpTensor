"""Base utilities for clustering modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import scipy.sparse as sp

from scptensor.core._layer_processing import ensure_dense_matrix, resolve_layer_context

if TYPE_CHECKING:
    from scptensor.core.structures import Assay, ScpContainer


def _validate_assay_layer(
    container: ScpContainer,
    assay_name: str,
    layer_name: str,
) -> tuple[Assay, np.ndarray | sp.spmatrix]:
    """Validate and get assay and layer data.

    Parameters
    ----------
    container : ScpContainer
        Input container.
    assay_name : str
        Name of assay.
    layer_name : str
        Name of layer.

    Returns
    -------
    tuple[Assay, np.ndarray | sp.spmatrix]
        Assay object and data matrix X.

    Raises
    ------
    AssayNotFoundError
        If assay not found.
    LayerNotFoundError
        If layer not found.

    """
    ctx = resolve_layer_context(container, assay_name, layer_name)
    return ctx.assay, ctx.layer.X


def _prepare_matrix(X: np.ndarray | sp.spmatrix) -> np.ndarray:
    """Convert matrix to dense numpy array.

    Parameters
    ----------
    X : np.ndarray | sp.spmatrix
        Input matrix (dense or sparse).

    Returns
    -------
    np.ndarray
        Dense numpy array.

    """
    return ensure_dense_matrix(X)


def _get_default_key(method: str, params: dict) -> str:
    """Generate default key name for clustering results.

    Parameters
    ----------
    method : str
        Method name (e.g., 'leiden', 'kmeans').
    params : dict
        Key parameters to include in key name.

    Returns
    -------
    str
        Generated key name (e.g., 'leiden_r1.0', 'kmeans_k5').

    """
    parts = [method]
    for key, value in params.items():
        if key == "n_clusters":
            short_key = "k"
        elif key == "resolution":
            short_key = "r"
        else:
            # Generic shortening fallback
            short_key = key[0] if len(key) > 3 else key
        parts.append(f"{short_key}{value}")
    return "_".join(parts)


def _add_labels_to_obs(
    container: ScpContainer,
    labels: np.ndarray,
    key: str,
) -> ScpContainer:
    """Add cluster labels to obs DataFrame.

    Parameters
    ----------
    container : ScpContainer
        Input container.
    labels : np.ndarray
        Cluster labels (length = n_samples).
    key : str
        Column name for labels.

    Returns
    -------
    ScpContainer
        New container with updated obs.

    """
    new_obs = container.obs.with_columns(pl.Series(name=key, values=labels.astype(str)))

    # Create new container with updated obs
    from scptensor.core._structure_container import ScpContainer as _ScpContainer

    return _ScpContainer(
        obs=new_obs,
        assays=container.assays,
        links=list(container.links),
        history=list(container.history),
        sample_id_col=container.sample_id_col,
    )


__all__ = [
    "_validate_assay_layer",
    "_prepare_matrix",
    "_get_default_key",
    "_add_labels_to_obs",
]
