"""Shared utilities for feature selection modules.

This module provides common validation and data processing functions
to avoid code duplication across feature selection methods.
"""

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from scipy import sparse

if TYPE_CHECKING:
    from scptensor.core.structures import Assay, ScpContainer

# Constants
_EPS = np.finfo(float).eps


def _validate_assay_layer(
    container: "ScpContainer",
    assay_name: str,
    layer: str,
) -> "Assay":
    """Validate assay and layer existence.

    Parameters
    ----------
    container : ScpContainer
        Input container.
    assay_name : str
        Name of the assay.
    layer : str
        Name of the layer.

    Returns
    -------
    Assay
        The validated assay.

    Raises
    ------
    ValueError
        If assay or layer not found.
    """
    if assay_name not in container.assays:
        raise ValueError(f"Assay '{assay_name}' not found in container.")
    assay = container.assays[assay_name]
    if layer not in assay.layers:
        raise ValueError(f"Layer '{layer}' not found in assay '{assay_name}'.")
    return assay


def _compute_mean_var(
    X: np.ndarray | sparse.spmatrix,
    axis: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and variance, sparse-aware.

    Parameters
    ----------
    X : ndarray or spmatrix
        Input data.
    axis : int, default=0
        Axis along which to compute.

    Returns
    -------
    mean : ndarray
        Mean values.
    var : ndarray
        Variance values.
    """
    if sparse.issparse(X):
        # Use sparse-aware mean to avoid dense conversion
        mean = np.asarray(X.mean(axis=axis)).ravel()
        # For variance: E[X^2] - E[X]^2
        # Use X * X element-wise multiplication
        X_squared = X * X
        mean_sq = np.asarray(X_squared.mean(axis=axis)).ravel()
        var = mean_sq - mean**2
        var = np.maximum(var, 0)  # Ensure non-negative
    else:
        mean = np.nanmean(X, axis=axis)
        var = np.nanvar(X, axis=axis)
    return mean, var


def _subset_or_annotate(
    container: "ScpContainer",
    assay_name: str,
    assay: "Assay",
    top_indices: np.ndarray,
    subset: bool,
    action: str,
    score: np.ndarray | None = None,
    score_col: str | None = None,
    bool_col: str = "highly_variable",
    params: dict | None = None,
) -> "ScpContainer":
    """Either subset features or annotate var.

    Parameters
    ----------
    container : ScpContainer
        Input container.
    assay_name : str
        Name of the assay.
    assay : Assay
        The assay object.
    top_indices : ndarray
        Indices of selected features.
    subset : bool
        If True, subset features; else annotate.
    action : str
        Action name for logging.
    score : ndarray, optional
        Score array to add to var.
    score_col : str, optional
        Column name for scores.
    bool_col : str, default="highly_variable"
        Column name for boolean selection.
    params : dict, optional
        Parameters for logging.

    Returns
    -------
    ScpContainer
        Modified container.
    """
    params = params or {}
    n_selected = len(top_indices)

    if subset:
        new_container = container.filter_features(assay_name, feature_indices=top_indices)
        description = f"Selected {n_selected}/{assay.n_features} features."
    else:
        # Add annotation to var
        is_selected = np.zeros(assay.n_features, dtype=bool)
        is_selected[top_indices] = True

        columns = [pl.Series(bool_col, is_selected)]
        if score is not None and score_col:
            columns.append(pl.Series(score_col, score))

        new_var = assay.var.with_columns(columns)
        new_assay = assay.__class__(var=new_var, layers=assay.layers)

        new_assays = dict(container.assays)
        new_assays[assay_name] = new_assay

        new_container = container.__class__(
            obs=container.obs, assays=new_assays, history=list(container.history)
        )
        description = f"Identified {n_selected}/{assay.n_features} features."

    new_container.log_operation(
        action=action,
        params=params,
        description=description,
    )
    return new_container


def _compute_dropout_stats(
    X: np.ndarray,
    M: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute detection counts and dropout rate per feature.

    Parameters
    ----------
    X : ndarray
        Data matrix (n_samples, n_features).
    M : ndarray or None
        Mask matrix. Values with mask != 0 are invalid.

    Returns
    -------
    n_detected : ndarray
        Number of detections per feature.
    dropout_rate : ndarray
        Dropout rate per feature.
    """
    n_samples = X.shape[0]

    # Detected: non-zero AND (mask is valid OR no mask)
    detected_mask = X != 0
    detected_mask &= ~np.isnan(X)
    if M is not None:
        detected_mask &= M == 0

    n_detected = np.sum(detected_mask, axis=0)
    dropout_rate = 1 - (n_detected / n_samples)
    return n_detected, dropout_rate
