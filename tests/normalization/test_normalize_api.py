"""Tests for normalization unified API."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from scptensor.core.exceptions import ScpValueError
from scptensor.core.structures import Assay, ScpContainer, ScpMatrix
from scptensor.normalization import norm_mean, norm_none, normalize


def _make_container() -> ScpContainer:
    obs = pl.DataFrame({"_index": ["s1", "s2", "s3"]})
    var = pl.DataFrame({"_index": ["p1", "p2", "p3", "p4"]})
    x = np.array(
        [
            [10.0, 11.0, 13.0, 15.0],
            [5.0, 6.0, 7.0, 9.0],
            [20.0, 30.0, 10.0, 25.0],
        ],
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


def test_norm_none_preserves_mask_reference() -> None:
    container = _make_container()
    source_m = container.assays["protein"].layers["raw"].M

    norm_none(container, assay_name="protein", source_layer="raw", new_layer_name="passthrough")

    assert container.assays["protein"].layers["passthrough"].M is source_m


def test_norm_none_same_name_only_logs_without_replacing_source_layer() -> None:
    container = _make_container()
    raw_layer = container.assays["protein"].layers["raw"]

    norm_none(container, assay_name="protein", source_layer="raw", new_layer_name="raw")

    assert list(container.assays["protein"].layers.keys()) == ["raw"]
    assert container.assays["protein"].layers["raw"] is raw_layer
    assert container.history[-1].action == "normalization_none"
    assert container.history[-1].params["new_layer_name"] == "raw"


def test_norm_mean_same_name_overwrites_source_layer_entry() -> None:
    container = _make_container()
    raw_layer = container.assays["protein"].layers["raw"]
    raw_x_before = raw_layer.X.copy()

    norm_mean(container, assay_name="protein", source_layer="raw", new_layer_name="raw")

    raw_after = container.assays["protein"].layers["raw"]
    assert raw_after is not raw_layer
    assert not np.allclose(raw_after.X, raw_x_before)
    assert container.history[-1].action == "normalization_sample_mean"


def test_norm_mean_output_shares_mask_reference_under_current_helper_contract() -> None:
    container = _make_container()
    source_m = container.assays["protein"].layers["raw"].M

    norm_mean(container, assay_name="protein", source_layer="raw", new_layer_name="mean")

    assert container.assays["protein"].layers["mean"].M is source_m


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


def test_normalize_history_uses_resolved_assay_name_for_aliases() -> None:
    container = _make_container()

    normalize(container, method="median", assay_name="proteins", source_layer="raw")

    assert "median_centered" in container.assays["protein"].layers
    assert container.history[-1].params["assay"] == "protein"


def test_norm_none_history_uses_resolved_container_assay_key() -> None:
    container = _make_container()
    container.assays = {"proteins": container.assays.pop("protein")}

    norm_none(container, assay_name="protein", source_layer="raw", new_layer_name="passthrough")

    assert "passthrough" in container.assays["proteins"].layers
    assert container.history[-1].params["assay"] == "proteins"


def test_vendor_normalized_raw_input_warns_before_normalization() -> None:
    container = _make_container()
    container.assays = {"proteins": container.assays.pop("protein")}
    container.log_operation(
        action="load_quant_table",
        params={
            "assay_name": "proteins",
            "layer_name": "raw",
            "input_quantity_is_vendor_normalized": True,
            "resolved_quantity_column": "PG.Quantity",
        },
        description="load vendor-normalized quantity",
    )

    with pytest.warns(UserWarning, match="vendor-normalized intensities"):
        norm_mean(container, assay_name="protein", source_layer="raw", new_layer_name="mean")
