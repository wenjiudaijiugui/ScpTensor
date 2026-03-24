"""Baseline imputation methods for DIA single-cell proteomics."""

from __future__ import annotations

import numpy as np

from scptensor.core._structure_container import ScpContainer
from scptensor.core.exceptions import ScpValueError
from scptensor.impute._utils import (
    add_imputed_layer,
    clone_layer_matrix,
    log_imputation_operation,
    preserve_observed_values,
    to_dense_float_copy,
)
from scptensor.impute.base import ImputeMethod, register_impute_method, validate_layer_context


def impute_none_core(data: np.ndarray) -> np.ndarray:
    """Passthrough imputation core (no-op)."""
    return np.array(data, copy=True)


def impute_zero_core(data: np.ndarray) -> np.ndarray:
    """Fill missing values with zero."""
    x = np.array(data, copy=True)
    x[np.isnan(x)] = 0.0
    return x


def _row_stat_impute_core(data: np.ndarray, stat: str) -> np.ndarray:
    x = np.array(data, copy=True)
    missing_mask = np.isnan(x)
    if not np.any(missing_mask):
        return x

    finite_vals = x[np.isfinite(x)]
    if finite_vals.size == 0:
        return np.zeros_like(x)

    if stat == "mean":
        global_stat = float(np.mean(finite_vals))
        row_stat = np.nanmean(x, axis=1)
    elif stat == "median":
        global_stat = float(np.median(finite_vals))
        row_stat = np.nanmedian(x, axis=1)
    else:
        raise ValueError(f"Unsupported stat '{stat}'.")

    row_stat = np.where(np.isfinite(row_stat), row_stat, global_stat)
    for row_idx in range(x.shape[0]):
        row_missing = missing_mask[row_idx]
        if np.any(row_missing):
            x[row_idx, row_missing] = row_stat[row_idx]
    return x


def impute_row_mean_core(data: np.ndarray) -> np.ndarray:
    """Fill missing values with row mean (sample-wise mean)."""
    return _row_stat_impute_core(data, stat="mean")


def impute_row_median_core(data: np.ndarray) -> np.ndarray:
    """Fill missing values with row median (sample-wise median)."""
    return _row_stat_impute_core(data, stat="median")


def impute_half_row_min_core(data: np.ndarray, fraction: float = 0.5) -> np.ndarray:
    """Fill missing values with fraction * row minimum."""
    if fraction <= 0:
        raise ScpValueError(
            f"fraction must be positive, got {fraction}.",
            parameter="fraction",
            value=fraction,
        )

    x = np.array(data, copy=True)
    missing_mask = np.isnan(x)
    if not np.any(missing_mask):
        return x

    finite_vals = x[np.isfinite(x)]
    if finite_vals.size == 0:
        return np.zeros_like(x)

    global_min = float(np.min(finite_vals))
    row_min = np.nanmin(x, axis=1)
    row_min = np.where(np.isfinite(row_min), row_min, global_min)
    fill_values = row_min * fraction

    for row_idx in range(x.shape[0]):
        row_missing = missing_mask[row_idx]
        if np.any(row_missing):
            x[row_idx, row_missing] = fill_values[row_idx]
    return x


def impute_none(
    container: ScpContainer,
    assay_name: str,
    source_layer: str,
    new_layer_name: str = "imputed_none",
) -> ScpContainer:
    """Add a passthrough layer without filling missing values."""
    ctx = validate_layer_context(container, assay_name, source_layer)
    assay = ctx.assay
    matrix = ctx.layer

    assay.add_layer(
        new_layer_name,
        clone_layer_matrix(
            matrix,
            source_assay_name=ctx.resolved_assay_name,
            source_layer_name=source_layer,
            action="impute_none",
            output_layer_name=new_layer_name,
        ),
    )
    return log_imputation_operation(
        container,
        action="impute_none",
        params={
            "assay": ctx.resolved_assay_name,
            "source_layer": source_layer,
            "new_layer_name": new_layer_name,
        },
        description=f"Passthrough imputation (no-op) on assay '{ctx.resolved_assay_name}'.",
    )


def impute_zero(
    container: ScpContainer,
    assay_name: str,
    source_layer: str,
    new_layer_name: str = "imputed_zero",
) -> ScpContainer:
    """Fill missing values with zeros."""
    ctx = validate_layer_context(container, assay_name, source_layer)
    assay = ctx.assay
    input_matrix = ctx.layer

    x_dense = to_dense_float_copy(input_matrix.X)
    missing_mask = np.isnan(x_dense)
    x_imputed = preserve_observed_values(impute_zero_core(x_dense), x_dense, missing_mask)

    add_imputed_layer(
        assay,
        new_layer_name,
        x_imputed,
        input_matrix,
        missing_mask,
        source_assay_name=ctx.resolved_assay_name,
        source_layer_name=source_layer,
        action="impute_zero",
    )
    return log_imputation_operation(
        container,
        action="impute_zero",
        params={
            "assay": ctx.resolved_assay_name,
            "source_layer": source_layer,
            "new_layer_name": new_layer_name,
        },
        description=f"Zero imputation on assay '{ctx.resolved_assay_name}'.",
    )


def impute_row_mean(
    container: ScpContainer,
    assay_name: str,
    source_layer: str,
    new_layer_name: str = "imputed_row_mean",
) -> ScpContainer:
    """Fill missing values with sample-wise row means."""
    ctx = validate_layer_context(container, assay_name, source_layer)
    assay = ctx.assay
    input_matrix = ctx.layer

    x_dense = to_dense_float_copy(input_matrix.X)
    missing_mask = np.isnan(x_dense)
    x_imputed = preserve_observed_values(impute_row_mean_core(x_dense), x_dense, missing_mask)

    add_imputed_layer(
        assay,
        new_layer_name,
        x_imputed,
        input_matrix,
        missing_mask,
        source_assay_name=ctx.resolved_assay_name,
        source_layer_name=source_layer,
        action="impute_row_mean",
    )
    return log_imputation_operation(
        container,
        action="impute_row_mean",
        params={
            "assay": ctx.resolved_assay_name,
            "source_layer": source_layer,
            "new_layer_name": new_layer_name,
        },
        description=f"Row-mean imputation on assay '{ctx.resolved_assay_name}'.",
    )


def impute_row_median(
    container: ScpContainer,
    assay_name: str,
    source_layer: str,
    new_layer_name: str = "imputed_row_median",
) -> ScpContainer:
    """Fill missing values with sample-wise row medians."""
    ctx = validate_layer_context(container, assay_name, source_layer)
    assay = ctx.assay
    input_matrix = ctx.layer

    x_dense = to_dense_float_copy(input_matrix.X)
    missing_mask = np.isnan(x_dense)
    x_imputed = preserve_observed_values(impute_row_median_core(x_dense), x_dense, missing_mask)

    add_imputed_layer(
        assay,
        new_layer_name,
        x_imputed,
        input_matrix,
        missing_mask,
        source_assay_name=ctx.resolved_assay_name,
        source_layer_name=source_layer,
        action="impute_row_median",
    )
    return log_imputation_operation(
        container,
        action="impute_row_median",
        params={
            "assay": ctx.resolved_assay_name,
            "source_layer": source_layer,
            "new_layer_name": new_layer_name,
        },
        description=f"Row-median imputation on assay '{ctx.resolved_assay_name}'.",
    )


def impute_half_row_min(
    container: ScpContainer,
    assay_name: str,
    source_layer: str,
    new_layer_name: str = "imputed_half_row_min",
    fraction: float = 0.5,
) -> ScpContainer:
    """Fill missing values with ``fraction * row_min``."""
    if fraction <= 0:
        raise ScpValueError(
            f"fraction must be positive, got {fraction}.",
            parameter="fraction",
            value=fraction,
        )

    ctx = validate_layer_context(container, assay_name, source_layer)
    assay = ctx.assay
    input_matrix = ctx.layer

    x_dense = to_dense_float_copy(input_matrix.X)
    missing_mask = np.isnan(x_dense)
    x_imputed = preserve_observed_values(
        impute_half_row_min_core(x_dense, fraction=fraction),
        x_dense,
        missing_mask,
    )

    add_imputed_layer(
        assay,
        new_layer_name,
        x_imputed,
        input_matrix,
        missing_mask,
        source_assay_name=ctx.resolved_assay_name,
        source_layer_name=source_layer,
        action="impute_half_row_min",
    )
    return log_imputation_operation(
        container,
        action="impute_half_row_min",
        params={
            "assay": ctx.resolved_assay_name,
            "source_layer": source_layer,
            "new_layer_name": new_layer_name,
            "fraction": fraction,
        },
        description=(
            f"Half-row-min imputation (fraction={fraction}) on assay '{ctx.resolved_assay_name}'."
        ),
    )


register_impute_method(
    ImputeMethod(
        name="none",
        supports_sparse=True,
        validate=lambda data: data.size > 0,
        apply=impute_none,
    ),
)
register_impute_method(
    ImputeMethod(
        name="zero",
        supports_sparse=False,
        validate=lambda data: data.size > 0,
        apply=impute_zero,
    ),
)
register_impute_method(
    ImputeMethod(
        name="row_mean",
        supports_sparse=False,
        validate=lambda data: data.size > 0,
        apply=impute_row_mean,
    ),
)
register_impute_method(
    ImputeMethod(
        name="row_median",
        supports_sparse=False,
        validate=lambda data: data.size > 0,
        apply=impute_row_median,
    ),
)
register_impute_method(
    ImputeMethod(
        name="half_row_min",
        supports_sparse=False,
        validate=lambda data: data.size > 0,
        apply=impute_half_row_min,
    ),
)
