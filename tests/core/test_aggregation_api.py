"""Tests for stable aggregation namespace exports."""

from __future__ import annotations

import inspect

import scptensor as scp
import scptensor.aggregation as stable_aggregation
from scptensor.aggregation import (
    AggMethod,
    BasicAggMethod,
    aggregate_to_protein,
    resolve_protein_mapping_column,
)
from scptensor.aggregation.protein import AggMethod as AggMethodCore
from scptensor.aggregation.protein import BasicAggMethod as BasicAggMethodCore
from scptensor.aggregation.protein import aggregate_to_protein as aggregate_to_protein_core
from scptensor.aggregation.protein import (
    resolve_protein_mapping_column as resolve_protein_mapping_column_core,
)
from scptensor.io import aggregate_to_protein as io_aggregate_to_protein


def test_stable_aggregation_namespace_all_freezes_package_surface() -> None:
    assert stable_aggregation.__all__ == [
        "AggMethod",
        "BasicAggMethod",
        "aggregate_to_protein",
        "resolve_protein_mapping_column",
    ]


def test_stable_aggregation_namespace_reexports_current_implementations() -> None:
    assert AggMethod == AggMethodCore
    assert BasicAggMethod == BasicAggMethodCore
    assert aggregate_to_protein is aggregate_to_protein_core
    assert resolve_protein_mapping_column is resolve_protein_mapping_column_core


def test_top_level_package_only_reexports_aggregate_function() -> None:
    assert scp.aggregate_to_protein is aggregate_to_protein_core
    assert "aggregate_to_protein" in scp.__all__

    for name in (
        "AggMethod",
        "BasicAggMethod",
        "resolve_protein_mapping_column",
    ):
        assert name not in scp.__all__


def test_io_aggregate_wrapper_remains_distinct_and_omits_unmapped_label() -> None:
    assert io_aggregate_to_protein is not aggregate_to_protein_core

    aggregation_params = inspect.signature(aggregate_to_protein_core).parameters
    io_params = inspect.signature(io_aggregate_to_protein).parameters

    assert "unmapped_label" in aggregation_params
    assert "unmapped_label" not in io_params
