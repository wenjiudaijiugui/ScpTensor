"""Base utilities for integration modules.

Provides method registry and validation utilities for batch effect correction.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import scipy.sparse as sp

from scptensor.core.exceptions import AssayNotFoundError, LayerNotFoundError, ScpValueError

if TYPE_CHECKING:
    from scptensor.core.structures import Assay, ScpContainer, ScpMatrix

# Type alias for integrate method
IntegrateMethod = Callable[..., "ScpContainer"]

# Registry for integration methods
_INTEGRATE_METHODS: dict[str, IntegrateMethod] = {}


def register_integrate_method(name: str) -> Callable[[IntegrateMethod], IntegrateMethod]:
    """Decorator to register an integration method.

    Parameters
    ----------
    name : str
        Name to register the method under.

    Returns
    -------
    Callable
        Decorator function.
    """

    def decorator(func: IntegrateMethod) -> IntegrateMethod:
        _INTEGRATE_METHODS[name] = func
        return func

    return decorator


def get_integrate_method(name: str) -> IntegrateMethod:
    """Get a registered integration method by name.

    Parameters
    ----------
    name : str
        Method name.

    Returns
    -------
    IntegrateMethod
        The registered method.

    Raises
    ------
    ScpValueError
        If method not found.
    """
    if name not in _INTEGRATE_METHODS:
        available = list(_INTEGRATE_METHODS.keys())
        raise ScpValueError(
            f"Integration method '{name}' not found. Available methods: {available}",
            parameter="method",
            value=name,
        )
    return _INTEGRATE_METHODS[name]


def list_integrate_methods() -> list[str]:
    """List all registered integration methods.

    Returns
    -------
    list[str]
        List of method names.
    """
    return list(_INTEGRATE_METHODS.keys())


def integrate(
    container: ScpContainer,
    method: str = "combat",
    **kwargs,
) -> ScpContainer:
    """Unified interface for batch effect correction.

    Parameters
    ----------
    container : ScpContainer
        Input container.
    method : str, default="combat"
        Integration method name.
    **kwargs
        Additional arguments passed to the method.

    Returns
    -------
    ScpContainer
        Container with corrected data.
    """
    func = get_integrate_method(method)
    return func(container, **kwargs)


def validate_layer_params(
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
        Assay name.
    layer_name : str
        Layer name.

    Returns
    -------
    tuple[Assay, ScpMatrix]
        Assay and layer objects.

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


def validate_batch_integration_params(
    container: ScpContainer,
    batch_key: str,
    assay_name: str,
    min_batches: int = 2,
    min_samples_per_batch: int = 2,
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray, dict]:
    """Validate batch integration parameters.

    Parameters
    ----------
    container : ScpContainer
        Input container.
    batch_key : str
        Column name in obs containing batch labels.
    assay_name : str
        Assay name (for error messages).
    min_batches : int, default=2
        Minimum number of batches required.
    min_samples_per_batch : int, default=2
        Minimum samples per batch.

    Returns
    -------
    tuple[pl.DataFrame, np.ndarray, np.ndarray, dict]
        obs DataFrame, batch labels, unique batches, batch counts.

    Raises
    ------
    ScpValueError
        If batch_key not found or insufficient batches/samples.
    """
    obs_df = container.obs

    if batch_key not in obs_df.columns:
        raise ScpValueError(
            f"Batch key '{batch_key}' not found in obs. Available columns: {obs_df.columns}",
            parameter="batch_key",
            value=batch_key,
        )

    batches = obs_df[batch_key].to_numpy()
    unique_batches, batch_counts = np.unique(batches, return_counts=True)

    if len(unique_batches) < min_batches:
        raise ScpValueError(
            f"Need at least {min_batches} batches, got {len(unique_batches)}.",
            parameter="batch_key",
            value=batch_key,
        )

    if min(batch_counts) < min_samples_per_batch:
        raise ScpValueError(
            f"Each batch needs at least {min_samples_per_batch} samples.",
            parameter="batch_key",
            value=batch_key,
        )

    batch_counts_dict = dict(zip(unique_batches, batch_counts, strict=False))

    return obs_df, batches, unique_batches, batch_counts_dict


def prepare_integration_data(
    X: np.ndarray | sp.spmatrix,
) -> np.ndarray:
    """Prepare data for integration by converting to dense and handling NaN.

    Parameters
    ----------
    X : np.ndarray | sp.spmatrix
        Input matrix.

    Returns
    -------
    np.ndarray
        Dense array with NaN values filled with 0.
    """
    if sp.issparse(X):
        X = X.toarray()  # type: ignore[union-attr]
    else:
        X = np.asarray(X)

    # Fill NaN with 0 for integration
    if np.any(np.isnan(X)):
        X = np.nan_to_num(X, nan=0.0)

    return X


def preserve_sparsity(
    X: np.ndarray,
    was_sparse: bool,
    threshold: float = 0.3,
) -> np.ndarray | sp.spmatrix:
    """Preserve sparsity in result if input was sparse.

    Parameters
    ----------
    X : np.ndarray
        Input dense array.
    was_sparse : bool
        Whether the original input was sparse.
    threshold : float, default=0.3
        Sparsity threshold for converting to sparse.

    Returns
    -------
    np.ndarray | sp.spmatrix
        Sparse matrix if was_sparse and sparsity > threshold, else dense.
    """
    if not was_sparse:
        return X

    sparsity = np.mean(X == 0)
    if sparsity > threshold:
        return sp.csr_matrix(X)

    return X


__all__ = [
    "IntegrateMethod",
    "register_integrate_method",
    "get_integrate_method",
    "list_integrate_methods",
    "integrate",
    "validate_layer_params",
    "validate_batch_integration_params",
    "prepare_integration_data",
    "preserve_sparsity",
]
