"""Tests for TRQN normalization."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp

from scptensor.core.exceptions import ScpValueError, ValidationError
from scptensor.core.structures import Assay, ScpContainer, ScpMatrix
from scptensor.normalization import norm_quantile
from scptensor.normalization.trqn_normalization import _rank_invariance_frequency, norm_trqn


def _make_container() -> ScpContainer:
    obs = pl.DataFrame({"_index": ["s1", "s2", "s3", "s4"]})
    var = pl.DataFrame({"_index": ["p1", "p2", "p3", "p4", "p5"]})
    x = np.array(
        [
            [100.0, 1.0, 2.0, 3.0, 4.0],
            [200.0, 10.0, 20.0, 30.0, 40.0],
            [300.0, 2.0, 4.0, 6.0, 8.0],
            [400.0, 5.0, 7.0, 9.0, 11.0],
        ]
    )
    m = np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 0, 3, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.int8,
    )
    assay = Assay(var=var, layers={"raw": ScpMatrix(X=x, M=m)})
    return ScpContainer(obs=obs, assays={"protein": assay})


def test_trqn_basic() -> None:
    container = _make_container()
    norm_trqn(container, assay_name="protein", source_layer="raw")
    assert "trqn_norm" in container.assays["protein"].layers
    assert container.assays["protein"].layers["trqn_norm"].X.shape == (4, 5)
    assert container.history[-1].action == "normalization_trqn"


def test_trqn_preserves_mask() -> None:
    container = _make_container()
    source_m = container.assays["protein"].layers["raw"].M.copy()
    norm_trqn(container, assay_name="protein", source_layer="raw", new_layer_name="trqn")
    out_m = container.assays["protein"].layers["trqn"].M
    assert np.array_equal(source_m, out_m)


def test_trqn_sparse_input_still_returns_dense_output() -> None:
    container = _make_container()
    container.assays["protein"].layers["raw"].X = sp.csr_matrix(
        container.assays["protein"].layers["raw"].X
    )

    norm_trqn(container, assay_name="protein", source_layer="raw", new_layer_name="trqn")

    assert not sp.issparse(container.assays["protein"].layers["trqn"].X)


def test_trqn_feature_indices_empty_matches_quantile() -> None:
    container_a = _make_container()
    container_b = _make_container()

    norm_quantile(container_a, assay_name="protein", source_layer="raw", new_layer_name="qn")
    norm_trqn(
        container_b,
        assay_name="protein",
        source_layer="raw",
        new_layer_name="trqn",
        feature_indices=[],
    )

    x_qn = container_a.assays["protein"].layers["qn"].X
    x_trqn = container_b.assays["protein"].layers["trqn"].X
    assert np.allclose(x_qn, x_trqn, equal_nan=True)


def test_trqn_rank_invariant_feature_variation_higher_than_quantile() -> None:
    container_qn = _make_container()
    container_trqn = _make_container()

    norm_quantile(container_qn, assay_name="protein", source_layer="raw", new_layer_name="qn")
    norm_trqn(container_trqn, assay_name="protein", source_layer="raw", new_layer_name="trqn")

    # p1 is rank-invariant top feature in all samples
    std_qn = np.nanstd(container_qn.assays["protein"].layers["qn"].X[:, 0])
    std_trqn = np.nanstd(container_trqn.assays["protein"].layers["trqn"].X[:, 0])
    assert std_trqn >= std_qn


def test_trqn_invalid_low_thr_raises() -> None:
    container = _make_container()
    with pytest.raises(ScpValueError, match="low_thr must be in"):
        norm_trqn(container, assay_name="protein", source_layer="raw", low_thr=0.0)


def test_trqn_invalid_balance_stat_raises() -> None:
    container = _make_container()
    with pytest.raises(ScpValueError, match="balance_stat must be"):
        norm_trqn(container, assay_name="protein", source_layer="raw", balance_stat="mode")


def test_trqn_invalid_feature_indices_raises() -> None:
    container = _make_container()
    with pytest.raises(ScpValueError, match="out-of-range indices"):
        norm_trqn(
            container,
            assay_name="protein",
            source_layer="raw",
            feature_indices=[0, 99],
        )


def test_rank_invariance_frequency_uses_assigned_positive_ranks_only() -> None:
    feature_sample = np.array(
        [
            [10.0, 11.0, 12.0, np.nan],
            [1.0, 2.0, 3.0, 4.0],
            [7.0, 8.0, 9.0, 10.0],
        ]
    )
    qn_feature_sample = np.array(
        [
            [5.0, 5.0, 4.0, np.nan],
            [4.0, 3.0, 5.0, 2.0],
            [1.0, 2.0, 1.0, 1.0],
        ]
    )

    frequencies = _rank_invariance_frequency(
        feature_sample,
        qn_feature_sample=qn_feature_sample,
    )

    expected = np.array([2.0 / 3.0, 0.5, 0.75])
    assert np.allclose(frequencies, expected)


def test_trqn_rejects_inf_input() -> None:
    container = _make_container()
    container.assays["protein"].layers["raw"].X[0, 0] = np.inf

    with pytest.raises(ValidationError, match="does not accept Inf values"):
        norm_trqn(container, assay_name="protein", source_layer="raw")
