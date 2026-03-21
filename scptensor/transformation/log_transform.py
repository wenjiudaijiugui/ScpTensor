"""Logarithmic transformation for ScpTensor."""

from __future__ import annotations

import re
import warnings

import numpy as np
import scipy.sparse as sp

from scptensor.core._layer_processing import (
    add_result_layer,
    clone_matrix_data,
    log_container_operation,
    resolve_layer_context,
)
from scptensor.core.assay_alias import resolve_assay_name
from scptensor.core.exceptions import ScpValueError
from scptensor.core.sparse_utils import (
    ensure_sparse_format,
    is_sparse_matrix,
    sparse_safe_log1p_with_scale,
)
from scptensor.core.structures import ScpContainer

_LAYER_LOG_PATTERN = re.compile(r"(^|[_\-])(log|log2|log10|ln)([_\-]|$)")


def _layer_name_suggests_logged(layer_name: str) -> bool:
    """Return True if layer naming strongly indicates log scale."""
    return bool(_LAYER_LOG_PATTERN.search(layer_name.lower()))


def _history_suggests_logged(
    container: ScpContainer,
    assay_name: str,
    layer_name: str,
) -> bool:
    """Return True if provenance indicates layer came from log transformation."""
    resolved_assay_name = resolve_assay_name(container, assay_name)
    for record in reversed(container.history):
        if record.action not in {"log_transform", "log_transform_skipped"}:
            continue
        params = record.params
        record_assay = params.get("assay")
        if not isinstance(record_assay, str):
            continue
        if resolve_assay_name(container, record_assay) != resolved_assay_name:
            continue
        if params.get("new_layer_name") == layer_name:
            return True
    return False


def _finite_value_sample(
    x: np.ndarray | sp.spmatrix,
    max_values: int = 200_000,
) -> np.ndarray:
    """Extract finite values from dense/sparse matrix with deterministic downsampling."""
    if is_sparse_matrix(x):
        values = np.asarray(x.data)  # type: ignore[union-attr]
    else:
        values = np.asarray(x).ravel()

    finite = values[np.isfinite(values)]
    if finite.size <= max_values:
        return finite

    step = int(np.ceil(finite.size / max_values))
    return finite[::step]


def _data_suggests_logged(values: np.ndarray) -> tuple[bool, str]:
    """Heuristic detection for log-transformed proteomics intensity values."""
    if values.size < 50:
        return False, "insufficient finite values for robust detection"

    q01, q50, q95, q99 = np.percentile(values, [1, 50, 95, 99])
    value_range = q99 - q01
    frac_negative = float(np.mean(values < 0))
    frac_le_100 = float(np.mean(values <= 100))
    frac_gt_1000 = float(np.mean(values > 1000))

    if frac_negative > 0.05 and q99 <= 80:
        reason = (
            f"5%+ values are negative ({frac_negative:.2%}) with p99={q99:.2f}, "
            "consistent with log-scaled/centered data"
        )
        return True, reason

    if q99 <= 80 and q95 <= 60 and value_range <= 40 and frac_le_100 >= 0.99:
        reason = (
            f"compressed dynamic range (p01={q01:.2f}, p50={q50:.2f}, p99={q99:.2f}, "
            f"range={value_range:.2f}) is consistent with log scale"
        )
        return True, reason

    if q99 <= 40 and q50 <= 20 and frac_gt_1000 < 0.001:
        reason = (
            f"low-intensity bounded profile (p50={q50:.2f}, p99={q99:.2f}) "
            "is consistent with log scale"
        )
        return True, reason

    return False, "distribution does not match log-scale heuristics"


def _detect_already_logged(
    container: ScpContainer,
    assay_name: str,
    source_layer: str,
    x: np.ndarray | sp.spmatrix,
    *,
    detect_logged_by_distribution: bool = False,
) -> tuple[bool, str]:
    """Detect whether source data appears already log-transformed."""
    if _layer_name_suggests_logged(source_layer):
        return True, f"layer name '{source_layer}' suggests log scale"

    if _history_suggests_logged(container, assay_name, source_layer):
        return True, f"provenance shows '{source_layer}' was created by log_transform"

    if not detect_logged_by_distribution:
        return False, "no explicit log provenance found from layer naming or history"

    values = _finite_value_sample(x)
    return _data_suggests_logged(values)


def log_transform(
    container: ScpContainer,
    assay_name: str = "protein",
    source_layer: str = "raw",
    new_layer_name: str = "log",
    base: float = 2.0,
    offset: float = 1.0,
    use_jit: bool = True,
    detect_logged: bool = True,
    skip_if_logged: bool = True,
    detect_logged_by_distribution: bool = False,
) -> ScpContainer:
    """Apply log transformation with configurable base and offset.

    Parameters
    ----------
    container : ScpContainer
        Input container with data to transform.
    assay_name : str, default="protein"
        Name of assay to transform.
    source_layer : str, default="raw"
        Name of source layer.
    new_layer_name : str, default="log"
        Name of destination layer.
    base : float, default=2.0
        Logarithm base.
    offset : float, default=1.0
        Non-negative pseudo-count added before log.
    use_jit : bool, default=True
        Whether to use JIT accelerated sparse path when available.
    detect_logged : bool, default=True
        If True, detect whether source data appears already log-transformed
        using layer naming and provenance history.
    skip_if_logged : bool, default=True
        Behavior when data appears already log-transformed:
        - True: warn and skip re-transformation (pass through source values)
        - False: warn but still apply log transformation
    detect_logged_by_distribution : bool, default=False
        If True, also apply a value-distribution heuristic to infer whether
        source data may already be log-transformed. Disabled by default
        because low-range linear intensity matrices can be misclassified.

    Returns
    -------
    ScpContainer
        Container with transformed layer added.
    """
    if base <= 0:
        raise ScpValueError(
            f"Log base must be positive, got {base}. "
            "Use base=2.0 for log2, base=10.0 for log10, or base=np.e for natural log.",
            parameter="base",
            value=base,
        )
    if offset < 0:
        raise ScpValueError(
            f"Offset must be non-negative, got {offset}. "
            "Offset is added before taking the log to handle zero values.",
            parameter="offset",
            value=offset,
        )

    ctx = resolve_layer_context(container, assay_name, source_layer)
    assay = ctx.assay
    input_layer = ctx.layer
    resolved_assay_name = ctx.resolved_assay_name

    x = input_layer.X
    log_scale = np.log(base)

    already_logged = False
    detection_reason = "logged-detection disabled"
    if detect_logged:
        already_logged, detection_reason = _detect_already_logged(
            container=container,
            assay_name=resolved_assay_name,
            source_layer=source_layer,
            x=x,
            detect_logged_by_distribution=detect_logged_by_distribution,
        )

    if already_logged:
        action_text = (
            "Skipping second log transform and passing data through unchanged."
            if skip_if_logged
            else "Applying log transform anyway because skip_if_logged=False."
        )
        warnings.warn(
            f"Source layer '{source_layer}' appears already log-transformed "
            f"({detection_reason}). {action_text}",
            UserWarning,
            stacklevel=2,
        )

        if skip_if_logged:
            if new_layer_name != source_layer:
                passthrough = clone_matrix_data(input_layer.X)
                add_result_layer(assay, new_layer_name, passthrough, input_layer)
            log_container_operation(
                container,
                action="log_transform_skipped",
                params={
                    "assay": resolved_assay_name,
                    "source_layer": source_layer,
                    "new_layer_name": new_layer_name,
                    "base": base,
                    "offset": offset,
                    "detect_logged": detect_logged,
                    "detect_logged_by_distribution": detect_logged_by_distribution,
                    "skip_if_logged": skip_if_logged,
                    "reason": detection_reason,
                },
                description=(
                    f"Skipped log transform on {resolved_assay_name}/{source_layer} because "
                    "data appears already log-transformed."
                ),
            )
            return container

    if is_sparse_matrix(x):
        if np.any(x.data < 0):  # type: ignore[union-attr,operator]
            min_val = np.nanmin(x.data)  # type: ignore[union-attr]
            warnings.warn(
                f"Input contains negative values (min={min_val:.4f}). "
                "These will be clipped to 0 before log transform.",
                UserWarning,
                stacklevel=2,
            )
            x = x.copy()
            x.data = np.maximum(x.data, 0)  # type: ignore[misc]
    else:
        if np.any(x < 0):
            min_val = np.nanmin(x)
            warnings.warn(
                f"Input contains negative values (min={min_val:.4f}). "
                "These will be clipped to 0 before log transform.",
                UserWarning,
                stacklevel=2,
            )
            x = np.maximum(x, 0)

    if is_sparse_matrix(x):
        x_log = sparse_safe_log1p_with_scale(x, offset=offset, scale=log_scale, use_jit=use_jit)
        x_log = ensure_sparse_format(x_log, "csr")
    else:
        x_log = np.log(x + offset) / log_scale

    add_result_layer(assay, new_layer_name, x_log, input_layer)

    log_container_operation(
        container,
        action="log_transform",
        params={
            "assay": resolved_assay_name,
            "source_layer": source_layer,
            "new_layer_name": new_layer_name,
            "base": base,
            "offset": offset,
            "sparse_input": is_sparse_matrix(x),
            "use_jit": use_jit,
            "detect_logged": detect_logged,
            "detect_logged_by_distribution": detect_logged_by_distribution,
            "skip_if_logged": skip_if_logged,
            "already_logged_detected": already_logged,
            "logged_detection_reason": detection_reason,
        },
        description=(f"Log{base} transformation applied to {resolved_assay_name}/{source_layer}."),
    )

    return container


__all__ = ["log_transform"]
