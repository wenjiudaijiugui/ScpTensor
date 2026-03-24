"""Tests for FilterCriteria-driven sample and feature filtering."""

import numpy as np
import polars as pl

from scptensor.core.filtering import FilterCriteria
from scptensor.core.structures import Assay, ScpContainer, ScpMatrix


def _build_container() -> ScpContainer:
    obs = pl.DataFrame(
        {
            "sid": ["s1", "s2", "s3"],
            "qc_pass": [True, False, True],
        },
    )
    var = pl.DataFrame(
        {
            "protein_id": ["p1", "p2", "p3"],
            "mean_intensity": [10.0, 20.0, 30.0],
        },
    )
    X = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
    )

    assay = Assay(var=var, layers={"X": ScpMatrix(X=X)}, feature_id_col="protein_id")
    return ScpContainer(obs=obs, assays={"proteins": assay}, sample_id_col="sid")


def test_filter_samples_by_expression_criteria():
    """Filter samples using FilterCriteria.by_expression."""
    container = _build_container()
    criteria = FilterCriteria.by_expression(pl.col("qc_pass"))

    filtered = container.filter_samples(criteria)

    assert filtered.obs["sid"].to_list() == ["s1", "s3"]
    assert filtered.assays["proteins"].layers["X"].X.shape == (2, 3)


def test_filter_features_by_ids_with_custom_feature_id_col():
    """Feature ID lookup should respect assay.feature_id_col."""
    container = _build_container()
    criteria = FilterCriteria.by_ids(["p2", "p3"])

    filtered = container.filter_features("proteins", criteria)

    assert filtered.assays["proteins"].feature_ids.to_list() == ["p2", "p3"]
    assert filtered.assays["proteins"].layers["X"].X.shape == (3, 2)
