"""Baseline imputation methods for DIA single-cell proteomics."""

from __future__ import annotations

import numpy as np

from scptensor.core.exceptions import AssayNotFoundError, LayerNotFoundError, ScpValueError
from scptensor.core.structures import ScpContainer, ScpMatrix
from scptensor.impute._utils import (
    add_imputed_layer,
    clone_layer_matrix,
    log_imputation_operation,
    preserve_observed_values,
    to_dense_float_copy,
)
from scptensor.impute.base import ImputeMethod, register_impute_method


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


def _validate_and_get_matrix(
    container: ScpContainer,
    assay_name: str,
    source_layer: str,
) -> ScpMatrix:
    if assay_name not in container.assays:
        available = ", ".join(f"'{k}'" for k in container.assays)
        raise AssayNotFoundError(assay_name, hint=f"Available assays: {available}.")

    assay = container.assays[assay_name]
    if source_layer not in assay.layers:
        available = ", ".join(f"'{k}'" for k in assay.layers)
        raise LayerNotFoundError(source_layer, assay_name, hint=f"Available layers: {available}.")

    return assay.layers[source_layer]


def impute_none(
    container: ScpContainer,
    assay_name: str,
    source_layer: str,
    new_layer_name: str = "imputed_none",
) -> ScpContainer:
    """Add a passthrough layer without filling missing values."""
    matrix = _validate_and_get_matrix(container, assay_name, source_layer)
    assay = container.assays[assay_name]

    assay.add_layer(new_layer_name, clone_layer_matrix(matrix))
    return log_imputation_operation(
        container,
        action="impute_none",
        params={
            "assay": assay_name,
            "source_layer": source_layer,
            "new_layer_name": new_layer_name,
        },
        description=f"Passthrough imputation (no-op) on assay '{assay_name}'.",
    )


def impute_zero(
    container: ScpContainer,
    assay_name: str,
    source_layer: str,
    new_layer_name: str = "imputed_zero",
) -> ScpContainer:
    """Fill missing values with zeros."""
    input_matrix = _validate_and_get_matrix(container, assay_name, source_layer)
    assay = container.assays[assay_name]

    x_dense = to_dense_float_copy(input_matrix.X)
    missing_mask = np.isnan(x_dense)
    x_imputed = preserve_observed_values(impute_zero_core(x_dense), x_dense, missing_mask)

    add_imputed_layer(assay, new_layer_name, x_imputed, input_matrix, missing_mask)
    return log_imputation_operation(
        container,
        action="impute_zero",
        params={
            "assay": assay_name,
            "source_layer": source_layer,
            "new_layer_name": new_layer_name,
        },
        description=f"Zero imputation on assay '{assay_name}'.",
    )


def impute_row_mean(
    container: ScpContainer,
    assay_name: str,
    source_layer: str,
    new_layer_name: str = "imputed_row_mean",
) -> ScpContainer:
    """Fill missing values with sample-wise row means."""
    input_matrix = _validate_and_get_matrix(container, assay_name, source_layer)
    assay = container.assays[assay_name]

    x_dense = to_dense_float_copy(input_matrix.X)
    missing_mask = np.isnan(x_dense)
    x_imputed = preserve_observed_values(impute_row_mean_core(x_dense), x_dense, missing_mask)

    add_imputed_layer(assay, new_layer_name, x_imputed, input_matrix, missing_mask)
    return log_imputation_operation(
        container,
        action="impute_row_mean",
        params={
            "assay": assay_name,
            "source_layer": source_layer,
            "new_layer_name": new_layer_name,
        },
        description=f"Row-mean imputation on assay '{assay_name}'.",
    )


def impute_row_median(
    container: ScpContainer,
    assay_name: str,
    source_layer: str,
    new_layer_name: str = "imputed_row_median",
) -> ScpContainer:
    """Fill missing values with sample-wise row medians."""
    input_matrix = _validate_and_get_matrix(container, assay_name, source_layer)
    assay = container.assays[assay_name]

    x_dense = to_dense_float_copy(input_matrix.X)
    missing_mask = np.isnan(x_dense)
    x_imputed = preserve_observed_values(impute_row_median_core(x_dense), x_dense, missing_mask)

    add_imputed_layer(assay, new_layer_name, x_imputed, input_matrix, missing_mask)
    return log_imputation_operation(
        container,
        action="impute_row_median",
        params={
            "assay": assay_name,
            "source_layer": source_layer,
            "new_layer_name": new_layer_name,
        },
        description=f"Row-median imputation on assay '{assay_name}'.",
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

    input_matrix = _validate_and_get_matrix(container, assay_name, source_layer)
    assay = container.assays[assay_name]

    x_dense = to_dense_float_copy(input_matrix.X)
    missing_mask = np.isnan(x_dense)
    x_imputed = preserve_observed_values(
        impute_half_row_min_core(x_dense, fraction=fraction),
        x_dense,
        missing_mask,
    )

    add_imputed_layer(assay, new_layer_name, x_imputed, input_matrix, missing_mask)
    return log_imputation_operation(
        container,
        action="impute_half_row_min",
        params={
            "assay": assay_name,
            "source_layer": source_layer,
            "new_layer_name": new_layer_name,
            "fraction": fraction,
        },
        description=f"Half-row-min imputation (fraction={fraction}) on assay '{assay_name}'.",
    )


register_impute_method(
    ImputeMethod(
        name="none", supports_sparse=True, validate=lambda data: data.size > 0, apply=impute_none
    )
)
register_impute_method(
    ImputeMethod(
        name="zero", supports_sparse=False, validate=lambda data: data.size > 0, apply=impute_zero
    )
)
register_impute_method(
    ImputeMethod(
        name="row_mean",
        supports_sparse=False,
        validate=lambda data: data.size > 0,
        apply=impute_row_mean,
    )
)
register_impute_method(
    ImputeMethod(
        name="row_median",
        supports_sparse=False,
        validate=lambda data: data.size > 0,
        apply=impute_row_median,
    )
)
register_impute_method(
    ImputeMethod(
        name="half_row_min",
        supports_sparse=False,
        validate=lambda data: data.size > 0,
        apply=impute_half_row_min,
    )
)
