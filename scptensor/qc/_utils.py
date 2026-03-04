"""Shared utilities for QC modules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from scptensor.core.exceptions import AssayNotFoundError, LayerNotFoundError, ScpValueError

if TYPE_CHECKING:
    from scptensor.core.structures import Assay, ScpContainer


def validate_assay(
    container: ScpContainer,
    assay_name: str,
) -> Assay:
    """Validate assay exists and return it.

    Parameters
    ----------
    container : ScpContainer
        Input container.
    assay_name : str
        Assay name.

    Returns
    -------
    Assay
        The assay object.

    Raises
    ------
    AssayNotFoundError
        If assay not found.
    """
    if assay_name not in container.assays:
        available = list(container.assays.keys())
        raise AssayNotFoundError(
            assay_name=assay_name,
            available_assays=available,
        )
    return container.assays[assay_name]


def validate_layer(
    assay: Assay,
    layer_name: str,
    assay_name: str | None = None,
) -> np.ndarray:
    """Validate layer exists and return X matrix.

    Parameters
    ----------
    assay : Assay
        Assay object.
    layer_name : str
        Layer name.

    Returns
    -------
    np.ndarray
        Data matrix X.

    Raises
    ------
    LayerNotFoundError
        If layer not found.
    """
    if layer_name not in assay.layers:
        available = list(assay.layers.keys())
        raise LayerNotFoundError(
            layer_name=layer_name,
            assay_name=assay_name or "<unknown>",
            available_layers=available,
        )
    X = assay.layers[layer_name].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X)


def validate_threshold(
    value: float,
    name: str,
    min_val: float | None = None,
    max_val: float | None = None,
) -> None:
    """Validate threshold parameter.

    Parameters
    ----------
    value : float
        Value to validate.
    name : str
        Parameter name.
    min_val : float | None, optional
        Minimum allowed value.
    max_val : float | None, optional
        Maximum allowed value.

    Raises
    ------
    ScpValueError
        If value is invalid.
    """
    if min_val is not None and value < min_val:
        raise ScpValueError(
            f"{name} must be >= {min_val}, got {value}.",
            parameter=name,
            value=value,
        )
    if max_val is not None and value > max_val:
        raise ScpValueError(
            f"{name} must be <= {max_val}, got {value}.",
            parameter=name,
            value=value,
        )


def validate_column_exists(
    df: pl.DataFrame,
    column: str,
    context: str = "obs",
) -> None:
    """Validate column exists in DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to check.
    column : str
        Column name.
    context : str, default="obs"
        Context for error message.

    Raises
    ------
    ScpValueError
        If column not found.
    """
    if column not in df.columns:
        raise ScpValueError(
            f"Column '{column}' not found in {context}. Available columns: {df.columns}",
            parameter=column,
        )


def compute_detection_stats(
    X: np.ndarray,
    axis: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute detection statistics.

    Parameters
    ----------
    X : np.ndarray
        Data matrix.
    axis : int, default=0
        Axis for computation (0=per feature, 1=per sample).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple of (n_detected, detection_rate, means).
    """
    n_total = X.shape[axis]
    n_detected = np.sum(~np.isnan(X), axis=axis)
    detection_rate = n_detected / n_total

    # Compute means ignoring NaN
    means = np.nanmean(X, axis=axis)

    return n_detected, detection_rate, means


def log_filtering_operation(
    container: ScpContainer,
    action: str,
    params: dict[str, Any],
    description: str,
) -> ScpContainer:
    """Log filtering operation to container history.

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


__all__ = [
    "validate_assay",
    "validate_layer",
    "validate_threshold",
    "validate_column_exists",
    "compute_detection_stats",
    "log_filtering_operation",
]
