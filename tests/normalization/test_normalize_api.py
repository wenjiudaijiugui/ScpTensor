"""Tests for normalization unified API."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from scptensor.core.exceptions import ScpValueError
from scptensor.core.structures import Assay, ScpContainer, ScpMatrix
from scptensor.normalization import norm_none, normalize


def _make_container() -> ScpContainer:
    obs = pl.DataFrame({"_index": ["s1", "s2", "s3"]})
    var = pl.DataFrame({"_index": ["p1", "p2", "p3", "p4"]})
    x = np.array(
        [
            [10.0, 11.0, 13.0, 15.0],
            [5.0, 6.0, 7.0, 9.0],
            [20.0, 30.0, 10.0, 25.0],
        ]
    )
    m = np.array(
        [
            [0, 0, 1, 0],
            [0, 2, 0, 0],
            [0, 0, 0, 3],
        ],
        dtype=np.int8,
    )
    assay = Assay(var=var, layers={"raw": ScpMatrix(X=x, M=m)})
    return ScpContainer(obs=obs, assays={"protein": assay})


def test_norm_none_creates_passthrough_layer() -> None:
    container = _make_container()
    source_x = container.assays["protein"].layers["raw"].X.copy()
    source_m = container.assays["protein"].layers["raw"].M.copy()

    norm_none(container, assay_name="protein", source_layer="raw", new_layer_name="passthrough")

    passthrough = container.assays["protein"].layers["passthrough"]
    assert np.array_equal(source_x, passthrough.X)
    assert np.array_equal(source_m, passthrough.M)
    assert container.history[-1].action == "normalization_none"


def test_normalize_dispatch_median_default_layer() -> None:
    container = _make_container()
    normalize(container, method="median", assay_name="protein", source_layer="raw")
    assert "median_centered" in container.assays["protein"].layers


def test_normalize_dispatch_quantile_alias() -> None:
    container = _make_container()
    normalize(container, method="norm_quantile", assay_name="protein", source_layer="raw")
    assert "quantile_norm" in container.assays["protein"].layers


def test_normalize_dispatch_mean() -> None:
    container = _make_container()
    normalize(container, method="mean", assay_name="protein", source_layer="raw")
    assert "sample_mean_norm" in container.assays["protein"].layers


def test_normalize_dispatch_trqn() -> None:
    container = _make_container()
    normalize(container, method="trqn", assay_name="protein", source_layer="raw")
    assert "trqn_norm" in container.assays["protein"].layers


def test_normalize_invalid_method() -> None:
    container = _make_container()
    with pytest.raises(ScpValueError, match="Unsupported normalization method"):
        normalize(container, method="unsupported", assay_name="protein", source_layer="raw")
