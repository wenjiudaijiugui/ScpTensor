"""Internal shared helpers for layer-level preprocessing stages."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp

from scptensor.core._structure_matrix import MatrixMetadata, ScpMatrix
from scptensor.core.assay_alias import resolve_assay_name
from scptensor.core.exceptions import AssayNotFoundError, LayerNotFoundError

if TYPE_CHECKING:
    from scptensor.core.structures import Assay, ScpContainer


@dataclass(frozen=True, slots=True)
class LayerContext:
    """Resolved assay/layer objects for a preprocessing operation."""

    resolved_assay_name: str
    assay: Assay
    layer_name: str
    layer: ScpMatrix


def resolve_layer_context(
    container: ScpContainer,
    assay_name: str,
    layer_name: str,
) -> LayerContext:
    """Resolve assay aliases and return the validated layer context."""
    resolved_assay_name = resolve_assay_name(container, assay_name)

    if resolved_assay_name not in container.assays:
        raise AssayNotFoundError(
            assay_name=assay_name,
            available_assays=list(container.assays.keys()),
        )

    assay = container.assays[resolved_assay_name]
    if layer_name not in assay.layers:
        raise LayerNotFoundError(
            layer_name=layer_name,
            assay_name=resolved_assay_name,
            available_layers=list(assay.layers.keys()),
        )

    return LayerContext(
        resolved_assay_name=resolved_assay_name,
        assay=assay,
        layer_name=layer_name,
        layer=assay.layers[layer_name],
    )


def resolve_assay_and_layer(
    container: ScpContainer,
    assay_name: str,
    layer_name: str,
) -> tuple[Assay, ScpMatrix]:
    """Resolve assay aliases and return the validated assay/layer pair."""
    ctx = resolve_layer_context(container, assay_name, layer_name)
    return ctx.assay, ctx.layer


def create_result_layer(
    x: np.ndarray | sp.spmatrix,
    source_layer: ScpMatrix,
    *,
    source_assay_name: str | None = None,
    source_layer_name: str | None = None,
    action: str | None = None,
    output_layer_name: str | None = None,
) -> ScpMatrix:
    """Create a derived layer while preserving current mask semantics."""
    metadata = copy.deepcopy(source_layer.metadata) if source_layer.metadata is not None else None
    if any(
        value is not None
        for value in (source_assay_name, source_layer_name, action, output_layer_name)
    ):
        if metadata is None:
            metadata = MatrixMetadata()
        creation_info = dict(metadata.creation_info or {})
        if source_assay_name is not None:
            creation_info["source_assay"] = source_assay_name
        if source_layer_name is not None:
            creation_info["source_layer"] = source_layer_name
        if action is not None:
            creation_info["action"] = action
        if output_layer_name is not None:
            creation_info["output_layer"] = output_layer_name
        metadata.creation_info = creation_info
    return ScpMatrix(X=x, M=source_layer.M, metadata=metadata)


def clone_matrix_data(x: np.ndarray | sp.spmatrix) -> np.ndarray | sp.spmatrix:
    """Clone dense or sparse matrix data for passthrough writes."""
    if sp.issparse(x):
        return x.copy()  # type: ignore[union-attr]
    return np.array(x, copy=True)


def ensure_dense_matrix(x: np.ndarray | sp.spmatrix) -> np.ndarray:
    """Return a dense ndarray for dense or sparse matrix input."""
    if sp.issparse(x):
        return x.toarray()  # type: ignore[union-attr]
    return np.asarray(x)


def resolve_result_layer_name(
    layer_name: str | None,
    default_layer_name: str,
) -> str:
    """Return the explicit output layer name or fall back to the stage default."""
    if layer_name is None:
        return default_layer_name
    return layer_name


def add_result_layer(
    assay: Assay,
    layer_name: str,
    x: np.ndarray | sp.spmatrix,
    source_layer: ScpMatrix,
    *,
    source_assay_name: str | None = None,
    source_layer_name: str | None = None,
    action: str | None = None,
) -> ScpMatrix:
    """Write a derived layer into an assay and return it."""
    result = create_result_layer(
        x,
        source_layer,
        source_assay_name=source_assay_name,
        source_layer_name=source_layer_name,
        action=action,
        output_layer_name=layer_name,
    )
    assay.add_layer(layer_name, result)
    return result


def write_result_layer_and_log(
    container: ScpContainer,
    assay: Assay,
    *,
    layer_name: str,
    x: np.ndarray | sp.spmatrix,
    source_layer: ScpMatrix,
    action: str,
    params: dict[str, Any],
    description: str,
) -> ScpContainer:
    """Write a derived layer and append the matching provenance record."""
    lineage_assay = params.get("assay")
    if not isinstance(lineage_assay, str):
        lineage_assay = params.get("source_assay")
    lineage_source_layer = params.get("source_layer")
    if not isinstance(lineage_source_layer, str):
        lineage_source_layer = params.get("base_layer")

    add_result_layer(
        assay,
        layer_name,
        x,
        source_layer,
        source_assay_name=lineage_assay if isinstance(lineage_assay, str) else None,
        source_layer_name=lineage_source_layer if isinstance(lineage_source_layer, str) else None,
        action=action,
    )
    return log_container_operation(
        container,
        action=action,
        params=params,
        description=description,
    )


def log_container_operation(
    container: ScpContainer,
    action: str,
    params: dict[str, Any],
    description: str,
) -> ScpContainer:
    """Append a provenance record and return the same container."""
    container.log_operation(action=action, params=params, description=description)
    return container
