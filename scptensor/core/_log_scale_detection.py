"""Shared internal helpers for explicit log-scale provenance detection."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

from scptensor.core.assay_alias import resolve_assay_name

if TYPE_CHECKING:
    from scptensor.core.structures import ScpContainer


_LAYER_LOG_PATTERN = re.compile(r"(^|[_\-])(log|log2|log10|ln)([_\-]|$)")


def layer_name_suggests_logged(layer_name: str) -> bool:
    """Return True if layer naming strongly indicates log scale."""
    return bool(_LAYER_LOG_PATTERN.search(layer_name.lower()))


def history_suggests_logged(
    container: ScpContainer,
    assay_name: str,
    layer_name: str,
) -> bool:
    """Return True if provenance indicates layer came from log transformation."""
    resolved_assay_name = resolve_assay_name(container, assay_name)
    assay_resolution_cache: dict[str, str] = {assay_name: resolved_assay_name}
    for record in reversed(container.history):
        if record.action not in {"log_transform", "log_transform_skipped"}:
            continue
        params = record.params
        record_assay = params.get("assay")
        if not isinstance(record_assay, str):
            continue
        resolved_record_assay = assay_resolution_cache.get(record_assay)
        if resolved_record_assay is None:
            resolved_record_assay = resolve_assay_name(container, record_assay)
            assay_resolution_cache[record_assay] = resolved_record_assay
        if resolved_record_assay != resolved_assay_name:
            continue
        if params.get("new_layer_name") == layer_name:
            return True
    return False


def finite_value_sample(
    x: np.ndarray | sp.spmatrix,
    max_values: int = 200_000,
) -> np.ndarray:
    """Extract finite values from dense/sparse matrix with deterministic downsampling."""
    if sp.issparse(x):
        values = np.asarray(x.data)
    else:
        values = np.asarray(x).ravel()

    finite = values[np.isfinite(values)]
    if finite.size <= max_values:
        return finite

    step = int(np.ceil(finite.size / max_values))
    return finite[::step]


def data_suggests_logged(values: np.ndarray) -> tuple[bool, str]:
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


def detect_logged_source_layer(
    container: ScpContainer,
    assay_name: str,
    source_layer: str,
    x: np.ndarray | sp.spmatrix,
    *,
    detect_logged_by_distribution: bool = False,
) -> tuple[bool, str]:
    """Detect whether source data appears already log-transformed."""
    if layer_name_suggests_logged(source_layer):
        return True, f"layer name '{source_layer}' suggests log scale"

    if history_suggests_logged(container, assay_name, source_layer):
        return True, f"provenance shows '{source_layer}' was created by log_transform"

    if not detect_logged_by_distribution:
        return False, "no explicit log provenance found from layer naming or history"

    values = finite_value_sample(x)
    return data_suggests_logged(values)


__all__ = [
    "data_suggests_logged",
    "detect_logged_source_layer",
    "finite_value_sample",
    "history_suggests_logged",
    "layer_name_suggests_logged",
]
