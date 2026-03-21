"""Shared utilities for QC modules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import polars as pl
import scipy.sparse as sp

from scptensor.core.assay_alias import resolve_assay_name
from scptensor.core.exceptions import AssayNotFoundError, LayerNotFoundError, ScpValueError
from scptensor.core.filtering import FilterCriteria
from scptensor.core.structures import MaskCode

if TYPE_CHECKING:
    from scptensor.core.structures import Assay, ScpContainer, ScpMatrix


_DETECTED_MASK_CODES = (MaskCode.VALID.value,)


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
    resolved_assay_name = resolve_assay_name(container, assay_name)

    if resolved_assay_name not in container.assays:
        available = list(container.assays.keys())
        raise AssayNotFoundError(
            assay_name=assay_name,
            available_assays=available,
        )
    return container.assays[resolved_assay_name]


def resolve_assay(
    container: ScpContainer,
    assay_name: str,
) -> tuple[str, Assay]:
    """Resolve assay aliases and return ``(resolved_name, assay)``."""
    resolved_assay_name = resolve_assay_name(container, assay_name)
    if resolved_assay_name not in container.assays:
        available = list(container.assays.keys())
        raise AssayNotFoundError(
            assay_name=assay_name,
            available_assays=available,
        )
    return resolved_assay_name, container.assays[resolved_assay_name]


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
    if sp.issparse(X):
        sparse_x = cast(sp.spmatrix, X)
        return np.asarray(sparse_x.toarray())
    return np.asarray(X)


def resolve_layer_name(
    assay: Assay,
    *,
    assay_name: str,
    layer_name: str | None = None,
    preferred_layer: str | None = None,
    fallback_to_first: bool = False,
) -> str:
    """Resolve a QC layer name while preserving entrypoint-specific defaults."""
    if layer_name is not None:
        validate_layer(assay, layer_name, assay_name=assay_name)
        return layer_name

    if preferred_layer is not None and preferred_layer in assay.layers:
        return preferred_layer

    if fallback_to_first:
        return next(iter(assay.layers.keys()))

    if preferred_layer is not None:
        validate_layer(assay, preferred_layer, assay_name=assay_name)
        return preferred_layer

    raise ScpValueError(
        "layer_name must be provided when no default layer rule is configured.",
        parameter="layer_name",
        value=layer_name,
    )


def resolve_layer(
    assay: Assay,
    *,
    assay_name: str,
    layer_name: str | None = None,
    preferred_layer: str | None = None,
    fallback_to_first: bool = False,
) -> tuple[str, ScpMatrix]:
    """Resolve and return a QC layer."""
    resolved_layer_name = resolve_layer_name(
        assay,
        assay_name=assay_name,
        layer_name=layer_name,
        preferred_layer=preferred_layer,
        fallback_to_first=fallback_to_first,
    )
    return resolved_layer_name, assay.layers[resolved_layer_name]


def _to_dense_float64(X: np.ndarray | sp.spmatrix) -> np.ndarray:  # noqa: N803
    """Convert dense/sparse matrices to dense float64 arrays."""
    if sp.issparse(X):
        sparse_x = cast(sp.spmatrix, X)
        return sparse_x.toarray().astype(np.float64, copy=False)
    return np.asarray(X, dtype=np.float64)


def _to_dense_int8(M: np.ndarray | sp.spmatrix) -> np.ndarray:  # noqa: N803
    """Convert dense/sparse mask matrices to dense int8 arrays."""
    if sp.issparse(M):
        sparse_m = cast(sp.spmatrix, M)
        return sparse_m.toarray().astype(np.int8, copy=False)
    return np.asarray(M, dtype=np.int8)


def get_detection_mask(
    X: np.ndarray | sp.spmatrix,  # noqa: N803
    M: np.ndarray | sp.spmatrix | None = None,  # noqa: N803
    *,
    detected_codes: tuple[int, ...] = _DETECTED_MASK_CODES,
) -> np.ndarray:
    """Return a boolean mask of detected observations.

    Detection semantics:
    - When ``M`` is provided, mask codes are authoritative.
    - For sparse matrices without ``M``, structural zeros are treated as undetected.
    - For dense matrices without ``M``, finite values are treated as detected.
    """
    if M is None:
        if sp.issparse(X):
            sparse_x = cast(sp.spmatrix, X)
            return sparse_x.toarray() != 0
        return np.isfinite(np.asarray(X))

    x_dense = _to_dense_float64(X)
    m_dense = _to_dense_int8(M)
    return np.isin(m_dense, detected_codes) & np.isfinite(x_dense)


def count_detected(
    X: np.ndarray | sp.spmatrix,  # noqa: N803
    M: np.ndarray | sp.spmatrix | None = None,  # noqa: N803
    *,
    axis: int = 0,
    detected_codes: tuple[int, ...] = _DETECTED_MASK_CODES,
) -> np.ndarray:
    """Count detected values along an axis."""
    if M is None and sp.issparse(X):
        sparse_x = cast(sp.spmatrix, X)
        return np.asarray(sparse_x.getnnz(axis=axis))
    detected_mask = get_detection_mask(X, M, detected_codes=detected_codes)
    return np.sum(detected_mask, axis=axis)


def compute_sample_total_intensity(X: np.ndarray | sp.spmatrix) -> np.ndarray:  # noqa: N803
    """Compute per-sample total intensity from the numeric matrix."""
    if sp.issparse(X):
        sparse_x = cast(sp.spmatrix, X)
        return np.asarray(sparse_x.sum(axis=1)).ravel()
    return np.nansum(np.asarray(X), axis=1)


def compute_sample_qc_vectors(layer: ScpMatrix) -> tuple[np.ndarray, np.ndarray]:
    """Compute sample-level detected feature counts and total intensities."""
    n_features = count_detected(layer.X, layer.M, axis=1)
    total_intensity = compute_sample_total_intensity(layer.X)
    return n_features, total_intensity


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
    X: np.ndarray | sp.spmatrix,
    M: np.ndarray | sp.spmatrix | None = None,  # noqa: N803
    axis: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute detection statistics.

    Parameters
    ----------
    X : np.ndarray | sp.spmatrix
        Data matrix.
    M : np.ndarray | sp.spmatrix | None, optional
        Mask matrix defining observation state. When provided, ``MaskCode.VALID``
        entries are treated as detected observations.
    axis : int, default=0
        Axis for computation (0=per feature, 1=per sample).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple of (n_detected, detection_rate, means).
    """
    x_dense = _to_dense_float64(X)
    detected_mask = get_detection_mask(X, M)
    n_total = x_dense.shape[axis]
    n_detected = np.sum(detected_mask, axis=axis)
    detection_rate = n_detected / n_total

    masked_values = np.where(detected_mask, x_dense, 0.0)
    sums = np.sum(masked_values, axis=axis)
    means = np.divide(
        sums,
        n_detected,
        out=np.full_like(sums, np.nan, dtype=np.float64),
        where=n_detected > 0,
    )

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


def filter_samples_with_provenance(
    container: ScpContainer,
    keep_indices: np.ndarray,
    *,
    action: str,
    params: dict[str, Any],
    description: str,
) -> ScpContainer:
    """Filter samples and append the entrypoint-specific provenance record."""
    criteria = FilterCriteria.by_indices(keep_indices)
    new_container = container.filter_samples(criteria)
    return log_filtering_operation(new_container, action, params, description)


def filter_features_with_provenance(
    container: ScpContainer,
    assay_name: str,
    keep_indices: np.ndarray,
    *,
    action: str,
    params: dict[str, Any],
    description: str,
) -> ScpContainer:
    """Filter features and append the entrypoint-specific provenance record."""
    criteria = FilterCriteria.by_indices(keep_indices)
    new_container = container.filter_features(assay_name, criteria)
    return log_filtering_operation(new_container, action, params, description)


__all__ = [
    "compute_sample_qc_vectors",
    "compute_sample_total_intensity",
    "resolve_assay",
    "resolve_layer",
    "resolve_layer_name",
    "validate_assay",
    "validate_layer",
    "validate_threshold",
    "validate_column_exists",
    "count_detected",
    "get_detection_mask",
    "compute_detection_stats",
    "filter_features_with_provenance",
    "filter_samples_with_provenance",
    "log_filtering_operation",
]
