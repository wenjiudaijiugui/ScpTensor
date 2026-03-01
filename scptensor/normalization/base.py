"""Base utilities for normalization modules."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp

from scptensor.core.exceptions import AssayNotFoundError, LayerNotFoundError
from scptensor.core.structures import ScpMatrix

if TYPE_CHECKING:
    from scptensor.core.structures import Assay, ScpContainer


def validate_assay_and_layer(
    container: ScpContainer,
    assay_name: str,
    layer_name: str,
) -> tuple[Assay, ScpMatrix]:
    """Validate and get assay and layer.

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
    tuple[Assay, ScpMatrix]
        Assay object and ScpMatrix layer.

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
            assay_name=assay_name,
            available_assays=available,
        )

    assay = container.assays[assay_name]

    if layer_name not in assay.layers:
        available = list(assay.layers.keys())
        raise LayerNotFoundError(
            layer_name=layer_name,
            assay_name=assay_name,
            available_layers=available,
        )

    return assay, assay.layers[layer_name]


def create_result_layer(
    X: np.ndarray | sp.spmatrix,
    source_layer: str | ScpMatrix = "",
    mask: np.ndarray | sp.spmatrix | None = None,
) -> ScpMatrix:
    """Create result ScpMatrix from transformed data.

    Parameters
    ----------
    X : np.ndarray | sp.spmatrix
        Transformed data matrix.
    source_layer : str | ScpMatrix
        Name of source layer (for provenance) or ScpMatrix to copy mask from.
    mask : np.ndarray | sp.spmatrix | None, optional
        Mask matrix to preserve.

    Returns
    -------
    ScpMatrix
        New ScpMatrix with data and optional mask.
    """
    # If source_layer is a ScpMatrix, extract its mask
    if isinstance(source_layer, ScpMatrix):
        mask = source_layer.M
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
        return X.toarray()  # type: ignore[union-attr]
    return np.asarray(X)


def get_layer_name(
    new_layer_name: str | None,
    default_suffix: str,
) -> str:
    """Generate new layer name.

    Parameters
    ----------
    new_layer_name : str | None
        Desired layer name, or None to use default.
    default_suffix : str
        Default suffix to use if new_layer_name is None.

    Returns
    -------
    str
        Layer name.
    """
    if new_layer_name is None:
        return default_suffix
    return new_layer_name


def log_operation(
    container: ScpContainer,
    action: str,
    params: dict[str, Any],
    description: str,
) -> ScpContainer:
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
    container: ScpContainer,
    assay_name: str,
    source_layer: str,
    new_layer_name: str,
    transform_func: Callable[[np.ndarray], np.ndarray],
    operation_name: str,
) -> ScpContainer:
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
    assay, layer = validate_assay_and_layer(container, assay_name, source_layer)
    X_dense = ensure_dense(layer.X)
    X_transformed = transform_func(X_dense)

    new_layer = create_result_layer(X_transformed, layer)
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
