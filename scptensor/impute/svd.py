"""SVD-based matrix completion imputation methods.

This module provides:
- Iterative low-rank SVD imputation (no extra dependency).
- SoftImpute wrapper (optional dependency: fancyimpute).
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
from sklearn.decomposition import TruncatedSVD

from scptensor.core._structure_container import ScpContainer
from scptensor.core.exceptions import MissingDependencyError, ScpValueError
from scptensor.impute._utils import (
    add_imputed_layer,
    log_imputation_operation,
    preserve_observed_values,
    to_dense_float_copy,
)
from scptensor.impute.base import ImputeMethod, register_impute_method, validate_layer_context


def _initial_fill_with_column_means(data: np.ndarray) -> np.ndarray:
    """Initialize NaN positions with column means (global fallback when needed)."""
    x = np.asarray(data, dtype=np.float64).copy()
    missing_mask = np.isnan(x)
    if not np.any(missing_mask):
        return x

    col_means = np.nanmean(x, axis=0)
    global_mean = float(np.nanmean(x)) if np.any(np.isfinite(x)) else 0.0
    col_means = np.where(np.isfinite(col_means), col_means, global_mean)
    fill_cols = np.where(missing_mask)[1]
    x[missing_mask] = col_means[fill_cols]
    return x


def iterative_svd_impute(
    data: np.ndarray,
    n_components: int = 5,
    max_iter: int = 100,
    tol: float = 1e-5,
    random_state: int | None = None,
    return_n_iter: bool = False,
) -> np.ndarray | tuple[np.ndarray, int]:
    """Iteratively impute missing values with low-rank SVD reconstruction."""
    x = np.asarray(data, dtype=np.float64).copy()
    missing_mask = np.isnan(x)

    if not np.any(missing_mask):
        return (x, 0) if return_n_iter else x
    if np.all(missing_mask):
        zeros = np.zeros_like(x)
        return (zeros, 0) if return_n_iter else zeros

    n_samples, n_features = x.shape
    min_dim = min(n_samples, n_features)
    if min_dim <= 1:
        x_filled = _initial_fill_with_column_means(x)
        return (x_filled, 0) if return_n_iter else x_filled

    effective_n_components = min(max(1, n_components), min_dim - 1)
    x_filled = _initial_fill_with_column_means(x)

    n_iterations = 0
    for iteration in range(max_iter):
        n_iterations = iteration + 1
        x_prev_missing = x_filled[missing_mask].copy()

        svd = TruncatedSVD(n_components=effective_n_components, random_state=random_state)
        z = svd.fit_transform(x_filled)
        x_recon = z @ svd.components_
        x_filled[missing_mask] = x_recon[missing_mask]

        delta = np.max(np.abs(x_filled[missing_mask] - x_prev_missing))
        scale = np.mean(np.abs(x_filled[missing_mask])) + 1e-12
        if delta / scale < tol:
            break

    # Guard against rare numerical instabilities.
    if np.any(~np.isfinite(x_filled[missing_mask])):
        fallback = _initial_fill_with_column_means(x)
        x_filled[missing_mask] = fallback[missing_mask]

    if return_n_iter:
        return x_filled, n_iterations
    return x_filled


def softimpute_impute(
    data: np.ndarray,
    rank: int | None = None,
    shrinkage_value: float | None = None,
    max_iter: int = 100,
    convergence_threshold: float = 1e-5,
    random_state: int | None = None,
) -> np.ndarray:
    """SoftImpute wrapper (requires fancyimpute)."""
    x = np.asarray(data, dtype=np.float64).copy()
    missing_mask = np.isnan(x)
    if not np.any(missing_mask):
        return x
    if np.all(missing_mask):
        return np.zeros_like(x)

    try:
        from fancyimpute import SoftImpute
    except ImportError as exc:
        raise MissingDependencyError("fancyimpute") from exc

    init_params = inspect.signature(SoftImpute.__init__).parameters
    kwargs: dict[str, Any] = {}

    if shrinkage_value is not None and "shrinkage_value" in init_params:
        kwargs["shrinkage_value"] = shrinkage_value
    if rank is not None:
        if "max_rank" in init_params:
            kwargs["max_rank"] = rank
        elif "rank" in init_params:
            kwargs["rank"] = rank
    if "max_iters" in init_params:
        kwargs["max_iters"] = max_iter
    elif "max_iter" in init_params:
        kwargs["max_iter"] = max_iter
    if "convergence_threshold" in init_params:
        kwargs["convergence_threshold"] = convergence_threshold
    if random_state is not None and "random_state" in init_params:
        kwargs["random_state"] = random_state
    if "verbose" in init_params:
        kwargs["verbose"] = False

    model = SoftImpute(**kwargs)
    try:
        return np.asarray(model.fit_transform(x), dtype=np.float64)
    except TypeError as exc:
        message = str(exc)
        if "force_all_finite" in message or "ensure_all_finite" in message:
            raise ScpValueError(
                "SoftImpute dependency interface mismatch detected between "
                "fancyimpute and scikit-learn "
                "(check_array force_all_finite/ensure_all_finite). "
                "Install a compatible dependency pair. "
                "ScpTensor no longer applies runtime monkey patches for this path.",
                parameter="softimpute",
            ) from exc
        raise


def validate_iterative_svd(data: np.ndarray) -> bool:
    """Validate data for iterative SVD imputation."""
    return data.size > 0 and min(data.shape) > 1


def validate_softimpute(data: np.ndarray) -> bool:
    """Validate data for SoftImpute."""
    return data.size > 0 and min(data.shape) > 1


def impute_iterative_svd(
    container: ScpContainer,
    assay_name: str,
    source_layer: str,
    new_layer_name: str = "imputed_iterative_svd",
    n_components: int = 5,
    max_iter: int = 100,
    tol: float = 1e-5,
    random_state: int | None = None,
) -> ScpContainer:
    """Impute missing values using iterative low-rank SVD reconstruction."""
    if n_components <= 0:
        raise ScpValueError(
            f"n_components must be positive, got {n_components}.",
            parameter="n_components",
            value=n_components,
        )
    if max_iter <= 0:
        raise ScpValueError(
            f"max_iter must be positive, got {max_iter}.",
            parameter="max_iter",
            value=max_iter,
        )
    if tol <= 0:
        raise ScpValueError(
            f"tol must be positive, got {tol}.",
            parameter="tol",
            value=tol,
        )

    ctx = validate_layer_context(container, assay_name, source_layer)
    assay = ctx.assay
    input_matrix = ctx.layer
    x_original = to_dense_float_copy(input_matrix.X)
    missing_mask = np.isnan(x_original)

    if not np.any(missing_mask):
        add_imputed_layer(assay, new_layer_name, x_original.copy(), input_matrix, missing_mask)
        return log_imputation_operation(
            container,
            action="impute_iterative_svd",
            params={
                "assay": ctx.resolved_assay_name,
                "source_layer": source_layer,
                "new_layer_name": new_layer_name,
                "n_components": n_components,
                "max_iter": max_iter,
                "tol": tol,
            },
            description=(
                "Iterative SVD imputation "
                f"on assay '{ctx.resolved_assay_name}': no missing values found."
            ),
        )

    x_imputed, n_iterations = iterative_svd_impute(
        x_original,
        n_components=n_components,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        return_n_iter=True,
    )
    preserve_observed_values(x_imputed, x_original, missing_mask)

    add_imputed_layer(assay, new_layer_name, x_imputed, input_matrix, missing_mask)
    return log_imputation_operation(
        container,
        action="impute_iterative_svd",
        params={
            "assay": ctx.resolved_assay_name,
            "source_layer": source_layer,
            "new_layer_name": new_layer_name,
            "n_components": n_components,
            "max_iter": max_iter,
            "tol": tol,
            "n_iterations": n_iterations,
        },
        description=(
            "Iterative SVD imputation "
            f"(n_components={n_components}, iterations={n_iterations}) "
            f"on assay '{ctx.resolved_assay_name}'."
        ),
    )


def impute_softimpute(
    container: ScpContainer,
    assay_name: str,
    source_layer: str,
    new_layer_name: str = "imputed_softimpute",
    rank: int | None = None,
    shrinkage_value: float | None = None,
    max_iter: int = 100,
    convergence_threshold: float = 1e-5,
    random_state: int | None = None,
) -> ScpContainer:
    """Impute missing values using SoftImpute (optional fancyimpute dependency)."""
    if rank is not None and rank <= 0:
        raise ScpValueError(
            f"rank must be positive when provided, got {rank}.",
            parameter="rank",
            value=rank,
        )
    if shrinkage_value is not None and shrinkage_value < 0:
        raise ScpValueError(
            f"shrinkage_value must be non-negative, got {shrinkage_value}.",
            parameter="shrinkage_value",
            value=shrinkage_value,
        )
    if max_iter <= 0:
        raise ScpValueError(
            f"max_iter must be positive, got {max_iter}.",
            parameter="max_iter",
            value=max_iter,
        )
    if convergence_threshold <= 0:
        raise ScpValueError(
            f"convergence_threshold must be positive, got {convergence_threshold}.",
            parameter="convergence_threshold",
            value=convergence_threshold,
        )

    ctx = validate_layer_context(container, assay_name, source_layer)
    assay = ctx.assay
    input_matrix = ctx.layer
    x_original = to_dense_float_copy(input_matrix.X)
    missing_mask = np.isnan(x_original)

    if not np.any(missing_mask):
        add_imputed_layer(assay, new_layer_name, x_original.copy(), input_matrix, missing_mask)
        return log_imputation_operation(
            container,
            action="impute_softimpute",
            params={
                "assay": ctx.resolved_assay_name,
                "source_layer": source_layer,
                "new_layer_name": new_layer_name,
                "rank": rank,
                "shrinkage_value": shrinkage_value,
                "max_iter": max_iter,
                "convergence_threshold": convergence_threshold,
            },
            description=(
                f"SoftImpute on assay '{ctx.resolved_assay_name}': no missing values found."
            ),
        )

    try:
        x_imputed = softimpute_impute(
            x_original,
            rank=rank,
            shrinkage_value=shrinkage_value,
            max_iter=max_iter,
            convergence_threshold=convergence_threshold,
            random_state=random_state,
        )
    except MissingDependencyError:
        raise
    except ScpValueError:
        raise
    except Exception as exc:
        raise ScpValueError(
            f"SoftImpute failed: {exc}",
            parameter="softimpute",
        ) from exc

    preserve_observed_values(x_imputed, x_original, missing_mask)
    add_imputed_layer(assay, new_layer_name, x_imputed, input_matrix, missing_mask)
    return log_imputation_operation(
        container,
        action="impute_softimpute",
        params={
            "assay": ctx.resolved_assay_name,
            "source_layer": source_layer,
            "new_layer_name": new_layer_name,
            "rank": rank,
            "shrinkage_value": shrinkage_value,
            "max_iter": max_iter,
            "convergence_threshold": convergence_threshold,
        },
        description=f"SoftImpute imputation on assay '{ctx.resolved_assay_name}'.",
    )


register_impute_method(
    ImputeMethod(
        name="iterative_svd",
        supports_sparse=False,
        validate=validate_iterative_svd,
        apply=impute_iterative_svd,
    ),
)

register_impute_method(
    ImputeMethod(
        name="softimpute",
        supports_sparse=False,
        validate=validate_softimpute,
        apply=impute_softimpute,
    ),
)
