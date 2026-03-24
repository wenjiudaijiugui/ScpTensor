"""Internal normalization-specific context helpers."""

from __future__ import annotations

import warnings

from scptensor.core._layer_processing import LayerContext, resolve_layer_context
from scptensor.core._structure_container import ScpContainer


def resolve_normalization_context(
    container: ScpContainer,
    assay_name: str,
    layer_name: str,
) -> LayerContext:
    """Resolve assay aliases and emit normalization-specific input warnings."""
    ctx = resolve_layer_context(container, assay_name, layer_name)
    _warn_if_vendor_normalized_input(container, ctx.resolved_assay_name, layer_name)
    return ctx


def _warn_if_vendor_normalized_input(
    container: ScpContainer,
    assay_name: str,
    layer_name: str,
) -> None:
    """Warn before re-normalizing vendor-normalized raw inputs."""
    if layer_name != "raw":
        return

    for log in reversed(container.history):
        if log.action != "load_quant_table":
            continue

        params = log.params
        if params.get("assay_name") != assay_name or params.get("layer_name") != layer_name:
            continue
        if not params.get("input_quantity_is_vendor_normalized", False):
            return

        quantity_desc = params.get("resolved_quantity_column") or "vendor-normalized quantity"
        warnings.warn(
            "Source layer 'raw' originates from vendor-normalized intensities "
            f"({quantity_desc}). Compare against `norm_none` or load an "
            "unnormalized vendor column when available.",
            UserWarning,
            stacklevel=3,
        )
        return
