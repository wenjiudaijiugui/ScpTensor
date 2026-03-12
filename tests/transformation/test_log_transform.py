"""Tests for transformation.log_transform."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import scptensor.normalization as normalization
from scptensor.core.structures import Assay, ScpContainer, ScpMatrix
from scptensor.transformation import log_transform


def _make_container() -> ScpContainer:
    obs = pl.DataFrame({"_index": ["s1", "s2"]})
    var = pl.DataFrame({"_index": ["p1", "p2", "p3"]})
    x = np.array([[0.0, 2.0, 4.0], [1.0, 3.0, 7.0]])
    assay = Assay(var=var, layers={"raw": ScpMatrix(X=x)})
    return ScpContainer(obs=obs, assays={"protein": assay})


def _make_low_range_linear_container(n_samples: int = 16, n_features: int = 8) -> ScpContainer:
    rng = np.random.default_rng(7)
    obs = pl.DataFrame({"_index": [f"s{i}" for i in range(n_samples)]})
    var = pl.DataFrame({"_index": [f"p{j}" for j in range(n_features)]})
    x = rng.uniform(0.0, 12.0, size=(n_samples, n_features))
    assay = Assay(var=var, layers={"raw": ScpMatrix(X=x)})
    return ScpContainer(obs=obs, assays={"protein": assay})


def _make_logged_container(
    layer_name: str = "log2", n_samples: int = 16, n_features: int = 8
) -> ScpContainer:
    rng = np.random.default_rng(42)
    obs = pl.DataFrame({"_index": [f"s{i}" for i in range(n_samples)]})
    var = pl.DataFrame({"_index": [f"p{j}" for j in range(n_features)]})

    raw_like = rng.lognormal(mean=12.0, sigma=1.2, size=(n_samples, n_features))
    x_logged = np.log2(raw_like + 1.0)

    assay = Assay(var=var, layers={layer_name: ScpMatrix(X=x_logged)})
    return ScpContainer(obs=obs, assays={"protein": assay})


def test_log_transform_in_transformation_module() -> None:
    container = _make_container()
    result = log_transform(
        container, assay_name="protein", source_layer="raw", new_layer_name="log2"
    )

    assert "log2" in result.assays["protein"].layers
    transformed = result.assays["protein"].layers["log2"].X
    expected = np.log2(np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 8.0]]))
    assert np.allclose(transformed, expected)


def test_log_transform_is_not_in_normalization_public_api() -> None:
    assert "log_transform" not in normalization.__all__
    assert not hasattr(normalization, "log_transform")


def test_log_transform_warns_and_skips_when_source_layer_looks_logged() -> None:
    container = _make_logged_container(layer_name="log2")
    source = container.assays["protein"].layers["log2"].X.copy()

    with pytest.warns(UserWarning, match="already log-transformed"):
        log_transform(
            container,
            assay_name="protein",
            source_layer="log2",
            new_layer_name="log_checked",
        )

    result = container.assays["protein"].layers["log_checked"].X
    assert np.allclose(result, source)
    assert container.history[-1].action == "log_transform_skipped"


def test_log_transform_warns_and_skips_by_distribution_detection() -> None:
    container = _make_logged_container(layer_name="raw")
    source = container.assays["protein"].layers["raw"].X.copy()

    with pytest.warns(UserWarning, match="already log-transformed"):
        log_transform(
            container,
            assay_name="protein",
            source_layer="raw",
            new_layer_name="log_checked",
            detect_logged_by_distribution=True,
        )

    result = container.assays["protein"].layers["log_checked"].X
    assert np.allclose(result, source)
    assert container.history[-1].action == "log_transform_skipped"


def test_log_transform_can_force_apply_when_already_logged() -> None:
    container = _make_logged_container(layer_name="raw")
    source = container.assays["protein"].layers["raw"].X.copy()

    with pytest.warns(UserWarning, match="already log-transformed"):
        log_transform(
            container,
            assay_name="protein",
            source_layer="raw",
            new_layer_name="log2_again",
            skip_if_logged=False,
            detect_logged_by_distribution=True,
        )

    result = container.assays["protein"].layers["log2_again"].X
    assert np.allclose(result, np.log2(source + 1.0))
    assert container.history[-1].action == "log_transform"


def test_log_transform_can_disable_logged_detection() -> None:
    container = _make_logged_container(layer_name="log2")
    source = container.assays["protein"].layers["log2"].X.copy()

    log_transform(
        container,
        assay_name="protein",
        source_layer="log2",
        new_layer_name="log2_again",
        detect_logged=False,
    )

    result = container.assays["protein"].layers["log2_again"].X
    assert np.allclose(result, np.log2(source + 1.0))


def test_log_transform_does_not_skip_low_range_linear_data_by_default() -> None:
    container = _make_low_range_linear_container()
    source = container.assays["protein"].layers["raw"].X.copy()

    log_transform(
        container,
        assay_name="protein",
        source_layer="raw",
        new_layer_name="log2",
    )

    result = container.assays["protein"].layers["log2"].X
    assert np.allclose(result, np.log2(source + 1.0))
    assert container.history[-1].action == "log_transform"


def test_log_transform_history_uses_resolved_assay_name_for_aliases() -> None:
    container = _make_container()

    log_transform(
        container,
        assay_name="proteins",
        source_layer="raw",
        new_layer_name="scaled",
    )

    assert container.history[-1].params["assay"] == "protein"

    with pytest.warns(UserWarning, match="already log-transformed"):
        log_transform(
            container,
            assay_name="protein",
            source_layer="scaled",
            new_layer_name="scaled_again",
        )

    assert container.history[-1].action == "log_transform_skipped"
    assert container.history[-1].params["assay"] == "protein"
