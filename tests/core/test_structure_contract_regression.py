"""Regression tests for PR-1 core structure refactor invariants."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from scptensor.core import (
    AggregationLink,
    Assay,
    AssayNotFoundError,
    FilterCriteria,
    ScpContainer,
    ScpMatrix,
)


def _build_multi_assay_container() -> ScpContainer:
    obs = pl.DataFrame(
        {
            "_index": ["S1", "S2", "S3"],
            "batch": ["b1", "b1", "b2"],
        },
    )
    protein_var = pl.DataFrame({"_index": ["P1", "P2", "P3"]})
    peptide_var = pl.DataFrame({"_index": ["pep1", "pep2", "pep3", "pep4"]})

    proteins = Assay(
        var=protein_var,
        layers={"raw": ScpMatrix(X=np.arange(9, dtype=float).reshape(3, 3))},
    )
    peptides = Assay(
        var=peptide_var,
        layers={"raw": ScpMatrix(X=np.arange(12, dtype=float).reshape(3, 4))},
    )
    link = AggregationLink(
        source_assay="peptides",
        target_assay="proteins",
        linkage=pl.DataFrame(
            {
                "source_id": ["pep1", "pep2", "pep3", "pep4"],
                "target_id": ["P1", "P1", "P2", "P3"],
            },
        ),
    )
    return ScpContainer(
        obs=obs,
        assays={"proteins": proteins, "peptides": peptides},
        links=[link],
    )


def test_container_requires_explicit_assay_for_feature_shape_queries() -> None:
    """Multi-assay containers require explicit assay selection for feature dimensions."""
    container = _build_multi_assay_container()

    assert not hasattr(container, "shape")
    assert not hasattr(container, "n_features")
    assert container.assay_shape("proteins") == (3, 3)
    assert container.assay_shape("peptides") == (3, 4)
    assert container.assay_shape("protein") == (3, 3)
    assert container.assay_shape("peptide") == (3, 4)


def test_container_removed_generic_persistence_stubs() -> None:
    """Core container no longer exposes unsupported generic persistence methods."""
    container = _build_multi_assay_container()

    assert not hasattr(container, "save")
    assert not hasattr(type(container), "load")


def test_assay_shape_rejects_unknown_assays() -> None:
    """Explicit assay-shape queries should fail fast for missing assays."""
    container = _build_multi_assay_container()

    with pytest.raises(AssayNotFoundError, match="Assay 'metabolites' not found"):
        container.assay_shape("metabolites")


def test_filter_samples_preserves_links_and_appends_history() -> None:
    """Filtering samples returns a new container and keeps assay-link structure intact."""
    container = _build_multi_assay_container()

    filtered = container.filter_samples(FilterCriteria.by_ids(["S1", "S3"]))

    assert filtered is not container
    assert filtered.obs["_index"].to_list() == ["S1", "S3"]
    assert filtered.assays["proteins"].layers["raw"].X.shape == (2, 3)
    assert filtered.assays["peptides"].layers["raw"].X.shape == (2, 4)
    assert len(filtered.links) == 1
    assert filtered.links[0].source_assay == "peptides"
    assert filtered.history[-1].action == "filter_samples"
    assert filtered.history[-1].params["n_samples_kept"] == 2


def test_filter_features_only_subsets_requested_assay_and_appends_history() -> None:
    """Feature filtering must stay assay-scoped in multi-assay containers."""
    container = _build_multi_assay_container()

    filtered = container.filter_features("proteins", FilterCriteria.by_ids(["P1", "P3"]))

    assert filtered.assays["proteins"].feature_ids.to_list() == ["P1", "P3"]
    assert filtered.assays["proteins"].layers["raw"].X.shape == (3, 2)
    assert filtered.assays["peptides"].feature_ids.to_list() == ["pep1", "pep2", "pep3", "pep4"]
    assert filtered.assays["peptides"].layers["raw"].X.shape == (3, 4)
    assert filtered.links[0].linkage["target_id"].to_list() == ["P1", "P1", "P3"]
    assert filtered.history[-1].action == "filter_features"
    assert filtered.history[-1].params["assay_name"] == "proteins"


def test_public_import_path_remains_valid_without_module_identity_spoofing() -> None:
    """Stable public imports remain valid without rewriting implementation module identity."""
    from scptensor.core.structures import Assay as PublicAssay
    from scptensor.core.structures import ScpContainer as PublicScpContainer
    from scptensor.core.structures import ScpMatrix as PublicScpMatrix

    container = _build_multi_assay_container()

    assert isinstance(container, PublicScpContainer)
    assert isinstance(container.assays["proteins"], PublicAssay)
    assert isinstance(container.assays["proteins"].layers["raw"], PublicScpMatrix)
    assert type(container).__module__ != "scptensor.core.structures"
    assert type(container.assays["proteins"]).__module__ != "scptensor.core.structures"
    assert (
        type(container.assays["proteins"].layers["raw"]).__module__ != "scptensor.core.structures"
    )
