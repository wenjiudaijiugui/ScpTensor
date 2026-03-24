"""Shared state-aware metrics and layer-lineage helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from scptensor.core.assay_alias import resolve_assay_name
from scptensor.core.exceptions import AssayNotFoundError, LayerNotFoundError
from scptensor.core.matrix_ops import MatrixOps
from scptensor.core.structures import MaskCode, ScpMatrix

if TYPE_CHECKING:
    from scptensor.core.structures import ProvenanceLog, ScpContainer

_COUNT_FIELDS: dict[MaskCode, str] = {
    MaskCode.VALID: "valid_count",
    MaskCode.MBR: "mbr_count",
    MaskCode.LOD: "lod_count",
    MaskCode.FILTERED: "filtered_count",
    MaskCode.OUTLIER: "outlier_count",
    MaskCode.IMPUTED: "imputed_count",
    MaskCode.UNCERTAIN: "uncertain_count",
}

_RATE_FIELDS: dict[MaskCode, str] = {
    MaskCode.VALID: "valid_rate",
    MaskCode.MBR: "mbr_rate",
    MaskCode.LOD: "lod_rate",
    MaskCode.FILTERED: "filtered_rate",
    MaskCode.OUTLIER: "outlier_rate",
    MaskCode.IMPUTED: "imputed_rate",
    MaskCode.UNCERTAIN: "uncertain_rate",
}


@dataclass(frozen=True, slots=True)
class LayerLineageStep:
    """One provenance edge for a derived assay/layer."""

    assay_name: str
    layer_name: str
    action: str
    source_assay: str
    source_layer: str


def compute_state_counts(matrix: ScpMatrix) -> dict[str, int]:
    """Return canonical per-state counts for a matrix."""
    stats = MatrixOps.get_mask_statistics(matrix)
    return {_COUNT_FIELDS[code]: int(stats[code.name]["count"]) for code in MaskCode}


def compute_state_rates(matrix: ScpMatrix) -> dict[str, float]:
    """Return canonical per-state rates for a matrix."""
    counts = compute_state_counts(matrix)
    total = matrix.X.shape[0] * matrix.X.shape[1]
    if total <= 0:
        return dict.fromkeys(_RATE_FIELDS.values(), 0.0)
    return {_RATE_FIELDS[code]: float(counts[_COUNT_FIELDS[code]] / total) for code in MaskCode}


def compute_direct_observation_rate(matrix: ScpMatrix) -> float:
    """Return the direct observation rate (`VALID`)."""
    return compute_state_rates(matrix)["valid_rate"]


def compute_supported_observation_rate(matrix: ScpMatrix) -> float:
    """Return the supported observation rate (`VALID + MBR`)."""
    rates = compute_state_rates(matrix)
    return float(rates["valid_rate"] + rates["mbr_rate"])


def compute_uncertainty_burden(matrix: ScpMatrix) -> float:
    """Return the combined uncertainty burden used by benchmark/report layers."""
    rates = compute_state_rates(matrix)
    return float(
        rates["filtered_rate"]
        + rates["outlier_rate"]
        + rates["imputed_rate"]
        + rates["uncertain_rate"]
    )


def compute_state_summary(matrix: ScpMatrix) -> dict[str, int | float]:
    """Return counts, rates, and derived state-aware summary metrics."""
    counts = compute_state_counts(matrix)
    rates = compute_state_rates(matrix)
    return {
        **counts,
        **rates,
        "direct_observation_rate": compute_direct_observation_rate(matrix),
        "supported_observation_rate": compute_supported_observation_rate(matrix),
        "uncertainty_burden": compute_uncertainty_burden(matrix),
    }


_STATE_RATE_AND_DERIVED_KEYS = (
    "valid_rate",
    "mbr_rate",
    "lod_rate",
    "filtered_rate",
    "outlier_rate",
    "imputed_rate",
    "uncertain_rate",
    "direct_observation_rate",
    "supported_observation_rate",
    "uncertainty_burden",
)


def _select_state_rate_metrics(summary: dict[str, int | float]) -> dict[str, float]:
    return {key: float(summary[key]) for key in _STATE_RATE_AND_DERIVED_KEYS}


def _extract_source_assay_name(params: Mapping[str, object]) -> str | None:
    assay = params.get("source_assay")
    if isinstance(assay, str):
        return assay
    assay = params.get("assay")
    if isinstance(assay, str):
        return assay
    return None


def _extract_source_layer_name(params: Mapping[str, object]) -> str | None:
    layer = params.get("source_layer")
    if isinstance(layer, str):
        return layer
    layer = params.get("base_layer")
    if isinstance(layer, str):
        return layer
    return None


def _extract_output_assay_name(params: Mapping[str, object]) -> str | None:
    assay = params.get("target_assay")
    if isinstance(assay, str):
        return assay
    assay = params.get("assay")
    if isinstance(assay, str):
        return assay
    return None


def _extract_output_layer_name(record: ProvenanceLog) -> str | None:
    params = record.params
    layer = params.get("new_layer_name")
    if isinstance(layer, str):
        return layer
    if record.action == "aggregate_to_protein":
        source_layer = params.get("source_layer")
        if isinstance(source_layer, str):
            return source_layer
    return None


def _resolve_if_present(container: ScpContainer, assay_name: str) -> str:
    resolved = resolve_assay_name(container, assay_name)
    if resolved in container.assays:
        return resolved
    return assay_name


def _lineage_step_from_metadata(
    container: ScpContainer,
    assay_name: str,
    layer_name: str,
    matrix: ScpMatrix,
) -> LayerLineageStep | None:
    metadata = matrix.metadata
    if metadata is None or metadata.creation_info is None:
        return None

    creation_info = metadata.creation_info
    source_assay = creation_info.get("source_assay")
    source_layer = creation_info.get("source_layer")
    action = creation_info.get("action")
    if not (
        isinstance(source_assay, str) and isinstance(source_layer, str) and isinstance(action, str)
    ):
        return None

    return LayerLineageStep(
        assay_name=assay_name,
        layer_name=layer_name,
        action=action,
        source_assay=_resolve_if_present(container, source_assay),
        source_layer=source_layer,
    )


def _lineage_step_from_history(
    container: ScpContainer,
    assay_name: str,
    layer_name: str,
) -> LayerLineageStep | None:
    assay_resolution_cache: dict[str, str] = {assay_name: assay_name}

    for record in reversed(container.history):
        output_layer = _extract_output_layer_name(record)
        if output_layer != layer_name:
            continue

        output_assay = _extract_output_assay_name(record.params)
        if not isinstance(output_assay, str):
            continue

        resolved_output_assay = assay_resolution_cache.get(output_assay)
        if resolved_output_assay is None:
            resolved_output_assay = _resolve_if_present(container, output_assay)
            assay_resolution_cache[output_assay] = resolved_output_assay
        if resolved_output_assay != assay_name:
            continue

        source_assay = _extract_source_assay_name(record.params)
        source_layer = _extract_source_layer_name(record.params)
        if not (isinstance(source_assay, str) and isinstance(source_layer, str)):
            continue

        return LayerLineageStep(
            assay_name=assay_name,
            layer_name=layer_name,
            action=record.action,
            source_assay=_resolve_if_present(container, source_assay),
            source_layer=source_layer,
        )

    return None


def resolve_layer_lineage(
    container: ScpContainer,
    assay_name: str,
    layer_name: str,
    *,
    max_depth: int = 20,
) -> list[LayerLineageStep]:
    """Resolve derived-layer lineage via metadata first, then history fallback."""
    resolved_assay_name = resolve_assay_name(container, assay_name)
    if resolved_assay_name not in container.assays:
        raise AssayNotFoundError(assay_name, available_assays=container.assays.keys())

    assay = container.assays[resolved_assay_name]
    if layer_name not in assay.layers:
        raise LayerNotFoundError(
            layer_name=layer_name,
            assay_name=resolved_assay_name,
            available_layers=assay.layers.keys(),
        )

    lineage: list[LayerLineageStep] = []
    seen = {(resolved_assay_name, layer_name)}
    current_assay = resolved_assay_name
    current_layer = layer_name

    for _ in range(max_depth):
        matrix = container.assays[current_assay].layers[current_layer]
        step = _lineage_step_from_metadata(container, current_assay, current_layer, matrix)
        if step is None:
            step = _lineage_step_from_history(container, current_assay, current_layer)
        if step is None:
            break

        lineage.append(step)
        next_assay = _resolve_if_present(container, step.source_assay)
        next_key = (next_assay, step.source_layer)
        if next_key in seen:
            break
        if next_assay not in container.assays:
            break
        if step.source_layer not in container.assays[next_assay].layers:
            break

        seen.add(next_key)
        current_assay = next_assay
        current_layer = step.source_layer

    return lineage


def resolve_origin_layer(
    container: ScpContainer,
    assay_name: str,
    layer_name: str,
    *,
    max_depth: int = 20,
) -> tuple[str, str]:
    """Resolve the earliest reachable origin layer for a derived layer."""
    resolved_assay_name = resolve_assay_name(container, assay_name)
    lineage = resolve_layer_lineage(
        container,
        assay_name=resolved_assay_name,
        layer_name=layer_name,
        max_depth=max_depth,
    )
    if not lineage:
        return resolved_assay_name, layer_name

    last = lineage[-1]
    return _resolve_if_present(container, last.source_assay), last.source_layer


def resolve_source_layer(
    container: ScpContainer,
    assay_name: str,
    layer_name: str,
    *,
    max_depth: int = 20,
) -> tuple[str, str]:
    """Resolve the immediate source layer for a derived layer when present."""
    resolved_assay_name = resolve_assay_name(container, assay_name)
    lineage = resolve_layer_lineage(
        container,
        assay_name=resolved_assay_name,
        layer_name=layer_name,
        max_depth=max_depth,
    )
    if not lineage:
        return resolved_assay_name, layer_name

    first = lineage[0]
    return _resolve_if_present(container, first.source_assay), first.source_layer


def compute_layer_state_metrics(
    container: ScpContainer,
    assay_name: str,
    layer_name: str,
) -> dict[str, float]:
    """Return the canonical state-rate vector for one assay/layer."""
    resolved_assay_name = resolve_assay_name(container, assay_name)
    if resolved_assay_name not in container.assays:
        raise AssayNotFoundError(assay_name, available_assays=container.assays.keys())

    assay = container.assays[resolved_assay_name]
    if layer_name not in assay.layers:
        raise LayerNotFoundError(
            layer_name=layer_name,
            assay_name=resolved_assay_name,
            available_layers=assay.layers.keys(),
        )
    matrix = assay.layers[layer_name]
    return _select_state_rate_metrics(compute_state_summary(matrix))


def compute_state_transition_metrics(
    container: ScpContainer,
    assay_name: str,
    layer_name: str,
    *,
    reference: Literal["source", "origin"] = "source",
    max_depth: int = 20,
) -> dict[str, float]:
    """Compare a layer's state metrics against its source or origin layer."""
    current = compute_layer_state_metrics(container, assay_name, layer_name)

    if reference == "source":
        ref_assay_name, ref_layer_name = resolve_source_layer(
            container,
            assay_name,
            layer_name,
            max_depth=max_depth,
        )
    else:
        ref_assay_name, ref_layer_name = resolve_origin_layer(
            container,
            assay_name,
            layer_name,
            max_depth=max_depth,
        )

    reference_metrics = compute_layer_state_metrics(container, ref_assay_name, ref_layer_name)
    out = dict(current)
    out.update({f"reference_{key}": value for key, value in reference_metrics.items()})
    out.update({f"delta_{key}": float(current[key] - reference_metrics[key]) for key in current})
    return out


__all__ = [
    "LayerLineageStep",
    "compute_layer_state_metrics",
    "compute_direct_observation_rate",
    "compute_state_counts",
    "compute_state_rates",
    "compute_state_summary",
    "compute_state_transition_metrics",
    "compute_supported_observation_rate",
    "compute_uncertainty_burden",
    "resolve_layer_lineage",
    "resolve_origin_layer",
    "resolve_source_layer",
]
