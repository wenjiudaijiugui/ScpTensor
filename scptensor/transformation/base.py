"""Shared helpers for transformation modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

from scptensor.core.assay_alias import resolve_assay_name
from scptensor.core.exceptions import AssayNotFoundError, LayerNotFoundError
from scptensor.core.structures import ScpMatrix

if TYPE_CHECKING:
    from scptensor.core.structures import Assay, ScpContainer


def validate_assay_and_layer(
    container: ScpContainer,
    assay_name: str,
    layer_name: str,
) -> tuple[Assay, ScpMatrix]:
    """Return validated assay and layer objects."""
    resolved_assay_name = resolve_assay_name(container, assay_name)

    if resolved_assay_name not in container.assays:
        available_assays = list(container.assays.keys())
        raise AssayNotFoundError(
            assay_name=assay_name,
            available_assays=available_assays,
        )

    assay = container.assays[resolved_assay_name]

    if layer_name not in assay.layers:
        available_layers = list(assay.layers.keys())
        raise LayerNotFoundError(
            layer_name=layer_name,
            assay_name=resolved_assay_name,
            available_layers=available_layers,
        )

    return assay, assay.layers[layer_name]


def create_result_layer(
    x: np.ndarray | sp.spmatrix,
    source_layer: ScpMatrix,
) -> ScpMatrix:
    """Create a new result layer while preserving mask provenance."""
    return ScpMatrix(X=x, M=source_layer.M)


__all__ = [
    "create_result_layer",
    "validate_assay_and_layer",
]
