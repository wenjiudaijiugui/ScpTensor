"""Base utilities for integration modules.

Provides method registry and validation utilities for batch effect correction.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import polars as pl
import scipy.sparse as sp

from scptensor.core.assay_alias import resolve_assay_name
from scptensor.core.exceptions import AssayNotFoundError, LayerNotFoundError, ScpValueError
from scptensor.core.structures import ScpMatrix

if TYPE_CHECKING:
    from scptensor.core.structures import Assay, ScpContainer

# Type alias for integrate method
IntegrateMethod = Callable[..., "ScpContainer"]
IntegrationLevel = Literal["matrix", "embedding"]
_PCA_LIKE_LAYER_NAMES = {"pc", "pcs", "pca"}
_PROTEIN_ASSAY_ALIASES = {"protein", "proteins"}


@dataclass(frozen=True, slots=True)
class IntegrationMethodInfo:
    """Metadata describing an integration method contract."""

    name: str
    function_name: str
    integration_level: IntegrationLevel
    recommended_for_de: bool


# Registry for integration methods
_INTEGRATE_METHODS: dict[str, IntegrateMethod] = {}
_INTEGRATE_METHOD_INFO: dict[str, IntegrationMethodInfo] = {}


def register_integrate_method(
    name: str,
    *,
    integration_level: IntegrationLevel = "matrix",
    recommended_for_de: bool = True,
) -> Callable[[IntegrateMethod], IntegrateMethod]:
    """Decorator to register an integration method.

    Parameters
    ----------
    name : str
        Name to register the method under.
    integration_level : {"matrix", "embedding"}, default="matrix"
        Declares whether method output is intended as matrix-level corrected
        quantification or embedding-level integrated representation.
    recommended_for_de : bool, default=True
        Whether method output is generally suitable for differential analysis
        on the corrected matrix.

    Returns
    -------
    Callable
        Decorator function.
    """

    def decorator(func: IntegrateMethod) -> IntegrateMethod:
        _INTEGRATE_METHODS[name] = func
        _INTEGRATE_METHOD_INFO[name] = IntegrationMethodInfo(
            name=name,
            function_name=func.__name__,
            integration_level=integration_level,
            recommended_for_de=recommended_for_de,
        )
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


def get_integrate_method_info(name: str) -> IntegrationMethodInfo:
    """Get metadata of a registered integration method by name."""
    if name not in _INTEGRATE_METHOD_INFO:
        available = list(_INTEGRATE_METHOD_INFO.keys())
        raise ScpValueError(
            f"Integration method '{name}' metadata not found. Available methods: {available}",
            parameter="method",
            value=name,
        )
    return _INTEGRATE_METHOD_INFO[name]


def list_integrate_methods() -> list[str]:
    """List all registered integration methods.

    Returns
    -------
    list[str]
        List of method names.
    """
    return list(_INTEGRATE_METHODS.keys())


def list_integrate_method_info() -> dict[str, IntegrationMethodInfo]:
    """List metadata for all registered integration methods."""
    return dict(_INTEGRATE_METHOD_INFO)


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
    resolved_assay_name = resolve_assay_name(container, assay_name)

    if resolved_assay_name not in container.assays:
        available = list(container.assays.keys())
        raise AssayNotFoundError(
            assay_name=assay_name,
            available_assays=available,
        )

    assay = container.assays[resolved_assay_name]

    if layer_name not in assay.layers:
        available = list(assay.layers.keys())
        raise LayerNotFoundError(
            layer_name=layer_name,
            assay_name=resolved_assay_name,
            available_layers=available,
        )

    return assay, assay.layers[layer_name]


def clone_layer_matrix(layer: ScpMatrix) -> ScpMatrix:
    """Clone X and M for baseline/no-op integration outputs."""
    if sp.issparse(layer.X):
        x_copy: np.ndarray | sp.spmatrix = layer.X.copy()
    else:
        x_copy = np.array(layer.X, copy=True)

    m_copy = layer.M.copy() if layer.M is not None else None
    return ScpMatrix(X=x_copy, M=m_copy)


def add_integrated_layer(
    assay: Assay,
    layer_name: str,
    X: np.ndarray | sp.spmatrix,
    source_layer: ScpMatrix,
) -> ScpMatrix:
    """Write an integration result while preserving copy semantics for M."""
    result = ScpMatrix(
        X=X,
        M=source_layer.M.copy() if source_layer.M is not None else None,
    )
    assay.add_layer(layer_name, result)
    return result


def validate_embedding_input(
    container: ScpContainer,
    assay_name: str,
    layer_name: str,
    *,
    method_name: str,
) -> tuple[Assay, ScpMatrix]:
    """Validate that an integration method receives embedding-like input.

    Accepted inputs are:
    - an embedding assay using layer ``"X"`` (for example assay ``"pca"``)
    - a PCA-like layer on an assay (for example layer ``"pca"``)
    """
    resolved_assay_name = resolve_assay_name(container, assay_name)
    assay, layer = validate_layer_params(container, assay_name, layer_name)

    normalized_assay = resolved_assay_name.strip().lower()
    normalized_layer = layer_name.strip().lower()

    is_embedding_assay = normalized_layer == "x" and normalized_assay not in _PROTEIN_ASSAY_ALIASES
    is_pca_like_layer = (
        normalized_layer in _PCA_LIKE_LAYER_NAMES
        or normalized_layer.startswith("pca_")
        or normalized_layer.endswith("_pca")
    )

    if not (is_embedding_assay or is_pca_like_layer):
        raise ScpValueError(
            f"{method_name} requires a low-dimensional embedding input, "
            f"but received assay '{resolved_assay_name}' layer '{layer_name}'. "
            "Pass an embedding assay with layer 'X' (for example assay='pca', "
            "base_layer='X') or a PCA-like layer on the protein assay "
            "(for example base_layer='pca'). Raw protein matrices are not valid "
            "inputs for embedding-level integration methods.",
            parameter="base_layer",
            value=layer_name,
        )

    return assay, layer


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
            f"Each batch needs at least {min_samples_per_batch} samples per batch.",
            parameter="batch_key",
            value=batch_key,
        )

    batch_counts_dict = dict(zip(unique_batches, batch_counts, strict=False))

    return obs_df, batches, unique_batches, batch_counts_dict


def prepare_integration_data(
    X: np.ndarray | sp.spmatrix,
    *,
    allow_missing: bool = False,
    context: str = "integration",
) -> np.ndarray:
    """Prepare data for integration by converting to dense and validating NaN.

    Parameters
    ----------
    X : np.ndarray | sp.spmatrix
        Input matrix.

    Returns
    -------
    np.ndarray
        Dense array for downstream integration.
    """
    X = to_dense_array(X, copy=not sp.issparse(X))

    if np.isnan(X).any() and not allow_missing:
        raise ScpValueError(
            f"{context} requires a complete matrix (no NaN values). "
            "Please impute or filter missing values before batch integration.",
            parameter="X",
        )

    return X


def to_dense_array(
    X: np.ndarray | sp.spmatrix,
    *,
    copy: bool = False,
) -> np.ndarray:
    """Convert sparse/dense input to a dense ndarray."""
    if sp.issparse(X):
        return X.toarray()  # type: ignore[union-attr]
    if copy:
        return np.array(X, copy=True)
    return np.asarray(X)


def prepare_integration_input(
    layer: ScpMatrix,
    *,
    allow_missing: bool = False,
    context: str = "integration",
) -> tuple[np.ndarray, bool]:
    """Return dense input data and the original sparse flag."""
    input_was_sparse = sp.issparse(layer.X)
    X_dense = prepare_integration_data(
        layer.X,
        allow_missing=allow_missing,
        context=context,
    )
    return X_dense, input_was_sparse


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


def log_integration_operation(
    container: ScpContainer,
    *,
    action: str,
    method_name: str,
    params: dict[str, Any],
    description: str,
) -> ScpContainer:
    """Append integration provenance with registered method metadata."""
    method_info = get_integrate_method_info(method_name)
    container.log_operation(
        action=action,
        params={
            **params,
            "integration_level": method_info.integration_level,
            "recommended_for_de": method_info.recommended_for_de,
        },
        description=description,
    )
    return container


__all__ = [
    "IntegrateMethod",
    "IntegrationLevel",
    "IntegrationMethodInfo",
    "register_integrate_method",
    "get_integrate_method",
    "get_integrate_method_info",
    "list_integrate_methods",
    "list_integrate_method_info",
    "integrate",
    "validate_layer_params",
    "clone_layer_matrix",
    "add_integrated_layer",
    "validate_embedding_input",
    "validate_batch_integration_params",
    "prepare_integration_data",
    "to_dense_array",
    "prepare_integration_input",
    "preserve_sparsity",
    "log_integration_operation",
]
