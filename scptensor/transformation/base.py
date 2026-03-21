"""Shared helpers for transformation modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

from scptensor.core._layer_processing import create_result_layer, resolve_layer_context

if TYPE_CHECKING:
    from scptensor.core.structures import Assay, ScpContainer, ScpMatrix


def validate_assay_and_layer(
    container: ScpContainer,
    assay_name: str,
    layer_name: str,
) -> tuple[Assay, ScpMatrix]:
    """Return validated assay and layer objects."""
    ctx = resolve_layer_context(container, assay_name, layer_name)
    return ctx.assay, ctx.layer


__all__ = [
    "create_result_layer",
    "validate_assay_and_layer",
]
