"""Aggregation methods for feature-level consolidation."""

from .protein import AggMethod, BasicAggMethod, aggregate_to_protein, resolve_protein_mapping_column

__all__ = [
    "AggMethod",
    "BasicAggMethod",
    "aggregate_to_protein",
    "resolve_protein_mapping_column",
]
