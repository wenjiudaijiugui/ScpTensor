"""Centralized heuristic policy constants for AutoSelect.

These values are tuning policy for evaluator scoring and budget limits, not
algorithmic invariants. Keep them centralized so changes are explicit,
documented, and reviewed as policy changes rather than incidental literal edits.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DimReductionHeuristics:
    """Heuristic targets and budgets for dimensionality-reduction evaluation."""

    nonlinear_embedding_components: int = 2
    variance_explained_target: float = 0.8
    reconstruction_sample_limit: int = 1000
    reconstruction_pair_limit: int = 1000
    local_structure_neighbors: int = 15
    clustering_sample_limit: int = 1000
    clustering_candidate_counts: tuple[int, ...] = (3, 5, 7, 10)


@dataclass(frozen=True)
class ImputationHoldoutHeuristics:
    """Deterministic holdout policy for imputation AutoSelect scoring."""

    fraction: float = 0.05
    min_entries: int = 128
    max_entries: int = 50_000
    random_state: int = 42


@dataclass(frozen=True)
class DynamicRangeHeuristics:
    """Bell-curve policy for dynamic-range scoring."""

    target_orders_of_magnitude: float = 6.0
    spread_orders_of_magnitude: float = 3.0


DIM_REDUCTION_HEURISTICS = DimReductionHeuristics()
IMPUTATION_HOLDOUT_HEURISTICS = ImputationHoldoutHeuristics()
DYNAMIC_RANGE_HEURISTICS = DynamicRangeHeuristics()


__all__ = [
    "DIM_REDUCTION_HEURISTICS",
    "DYNAMIC_RANGE_HEURISTICS",
    "IMPUTATION_HOLDOUT_HEURISTICS",
    "DimReductionHeuristics",
    "DynamicRangeHeuristics",
    "ImputationHoldoutHeuristics",
]
