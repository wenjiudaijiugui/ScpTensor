"""Base utilities for normalization modules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Any

import numpy as np
import polars as pl
import scipy.sparse as sp

from scptensor.core.exceptions import AssayNotFoundError, LayerNotFoundError
from scptensor.core.structures import ScpMatrix

if TYPE_CHECKING:
    from scptensor.core.structures import Assay, ScpContainer


def validate_assay_and_layer(
    container: "ScpContainer",
    assay_name: str,
    layer_name: str,
) -> tuple["Assay", np.ndarray | sp.spmatrix]:
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
    if assay_name not in container.assays:
        available = list(container.assays.keys())
        raise AssayNotFoundError(
            f"Assay '{assay_name}' not found.",
            assay_name=assay_name,
            available_assays=available,
        )

    assay = container.assays[assay_name]

    if layer_name not in assay.layers:
        available = list(assay.layers.keys())
        raise LayerNotFoundError(
            f"Layer '{layer_name}' not found in assay '{assay_name}'.",
            layer_name=layer_name,
            assay_name=assay_name,
            available_layers=available,
        )

    return assay, assay.layers[layer_name].X


def create_result_layer(
    X: np.ndarray | sp.spmatrix,
    source_layer: str,
    mask: np.ndarray | sp.spmatrix | None = None,
) -> ScpMatrix:
    """Create result ScpMatrix from transformed data.

    Parameters
    ----------
    X : np.ndarray | sp.spmatrix
        Transformed data matrix.
    source_layer : str
        Name of source layer (for provenance).
    mask : np.ndarray | sp.spmatrix | None, optional
        Mask matrix to preserve.

    Returns
    -------
    ScpMatrix
        New ScpMatrix with data and optional mask.
    """
    return ScpMatrix(X=X, M=mask)


def ensure_dense(X: np.ndarray | sp.spmatrix) -> np.ndarray:
    """Convert sparse matrix to dense if needed.

    Parameters
    ----------
    X : np.ndarray | sp.spmatrix
        Input matrix.

    Returns
    -------
    np.ndarray
        Dense numpy array.
    """
    if sp.issparse(X):
        return X.toarray()
    return np.asarray(X)


def get_layer_name(
    source_layer: str,
    suffix: str,
) -> str:
    """Generate new layer name.

    Parameters
    ----------
    source_layer : str
        Source layer name.
    suffix : str
        Suffix to append.

    Returns
    -------
    str
        New layer name.
    """
    return f"{source_layer}_{suffix}" if source_layer != "X" else suffix


def log_operation(
    container: "ScpContainer",
    action: str,
    params: dict[str, Any],
    description: str,
) -> "ScpContainer":
    """Log operation to container history.

    Parameters
    ----------
    container : ScpContainer
        Container to update.
    action : str
        Action name.
    params : dict
        Parameters.
    description : str
        Description.

    Returns
    -------
    ScpContainer
        Updated container.
    """
    container.log_operation(action=action, params=params, description=description)
    return container


def apply_normalization(
    container: "ScpContainer",
    assay_name: str,
    source_layer: str,
    new_layer_name: str,
    transform_func: Callable[[np.ndarray], np.ndarray],
    operation_name: str,
) -> "ScpContainer":
    """Apply normalization transformation to a layer.

    Parameters
    ----------
    container : ScpContainer
        Input container.
    assay_name : str
        Assay name.
    source_layer : str
        Source layer name.
    new_layer_name : str
        New layer name.
    transform_func : Callable
        Transformation function.
    operation_name : str
        Operation name for logging.

    Returns
    -------
    ScpContainer
        Container with new layer.
    """
    assay, X = validate_assay_and_layer(container, assay_name, source_layer)
    X_dense = ensure_dense(X)
    X_transformed = transform_func(X_dense)

    new_layer = create_result_layer(X_transformed, source_layer, assay.layers[source_layer].M)
    assay.layers[new_layer_name] = new_layer

    return log_operation(
        container,
        action=operation_name,
        params={"assay": assay_name, "source": source_layer, "target": new_layer_name},
        description=f"{operation_name} on {assay_name}/{source_layer}",
    )


__all__ = [
    "validate_assay_and_layer",
    "create_result_layer",
    "ensure_dense",
    "get_layer_name",
    "log_operation",
    "apply_normalization",
]
