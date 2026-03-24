"""Tests for scptensor.standardization.zscore."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from scptensor.core.exceptions import ScpValueError, ValidationError
from scptensor.core.structures import Assay, ScpContainer, ScpMatrix
from scptensor.standardization import zscore


def _make_container(
    x: np.ndarray,
    layer_name: str = "imputed",
    *,
    mask: np.ndarray | None = None,
    assay_name: str = "protein",
) -> ScpContainer:
    obs = pl.DataFrame({"_index": [f"s{i}" for i in range(x.shape[0])]})
    var = pl.DataFrame({"_index": [f"p{j}" for j in range(x.shape[1])]})
    assay = Assay(var=var, layers={layer_name: ScpMatrix(X=x, M=mask)})
    return ScpContainer(obs=obs, assays={assay_name: assay})


def test_zscore_feature_wise_standardization() -> None:
    x = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
        ],
    )
    container = _make_container(x)

    result = zscore(container, assay_name="protein", source_layer="imputed", new_layer_name="z")
    z = result.assays["protein"].layers["z"].X

    assert np.allclose(np.mean(z, axis=0), 0.0, atol=1e-12)
    assert np.allclose(np.std(z, axis=0, ddof=1), 1.0, atol=1e-12)


def test_zscore_rejects_nan_input() -> None:
    x = np.array([[1.0, np.nan], [2.0, 3.0]])
    container = _make_container(x)

    with pytest.raises(ValidationError, match="requires a complete matrix"):
        zscore(container)


def test_zscore_rejects_inf_input() -> None:
    x = np.array([[1.0, np.inf], [2.0, 3.0]])
    container = _make_container(x)

    with pytest.raises(ValidationError, match="does not accept Inf values"):
        zscore(container)


def test_zscore_rejects_ddof_too_large_for_axis() -> None:
    x = np.array([[1.0, 2.0]])
    container = _make_container(x)

    with pytest.raises(ValidationError, match="requires at least 2 values along axis 0"):
        zscore(container, axis=0, ddof=1)


def test_zscore_rejects_invalid_axis() -> None:
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    container = _make_container(x)

    with pytest.raises(ScpValueError, match="Axis must be 0 or 1"):
        zscore(container, axis=2)


def test_zscore_exported_from_standardization_namespace() -> None:
    assert callable(zscore)


def test_zscore_runs_on_complete_raw_layer_without_logged_gate() -> None:
    x = np.array(
        [
            [10.0, 100.0],
            [20.0, 120.0],
            [40.0, 140.0],
        ],
    )
    container = _make_container(x, layer_name="raw")

    result = zscore(container, assay_name="protein", source_layer="raw", new_layer_name="z_raw")
    z = result.assays["protein"].layers["z_raw"].X

    assert np.allclose(np.mean(z, axis=0), 0.0, atol=1e-12)
    assert np.allclose(np.std(z, axis=0, ddof=1), 1.0, atol=1e-12)
    assert result.history[-1].action == "standardization_zscore"
    assert result.history[-1].params["source_layer"] == "raw"


def test_zscore_resolves_protein_aliases_and_uses_canonical_default_name() -> None:
    x = np.array([[1.0, 3.0], [2.0, 4.0], [3.0, 5.0]])
    container = _make_container(x, assay_name="proteins")

    result = zscore(container)

    assert "zscore" in result.assays["proteins"].layers
    assert result.history[-1].params["assay"] == "proteins"


def test_zscore_only_checks_x_for_completeness_and_copies_mask() -> None:
    x = np.array([[1.0, 5.0], [2.0, 6.0], [3.0, 7.0]])
    mask = np.array([[0, 1], [2, 0], [0, 0]], dtype=np.int8)
    container = _make_container(x, mask=mask)

    result = zscore(container, assay_name="protein", source_layer="imputed", new_layer_name="z")
    source_layer = result.assays["protein"].layers["imputed"]
    z_layer = result.assays["protein"].layers["z"]

    assert z_layer.M is not None
    assert np.array_equal(z_layer.M, mask)
    assert z_layer.M is not source_layer.M


def test_zscore_same_name_target_overwrites_source_layer_entry() -> None:
    x = np.array([[1.0, 5.0], [2.0, 6.0], [3.0, 7.0]])
    mask = np.array([[0, 1], [2, 0], [0, 0]], dtype=np.int8)
    container = _make_container(x, mask=mask)
    source_layer = container.assays["protein"].layers["imputed"]

    result = zscore(
        container,
        assay_name="protein",
        source_layer="imputed",
        new_layer_name="imputed",
    )
    overwritten = result.assays["protein"].layers["imputed"]

    assert overwritten is not source_layer
    assert overwritten.M is not None
    assert np.array_equal(overwritten.M, mask)
    assert overwritten.M is not source_layer.M
    assert np.allclose(np.mean(overwritten.X, axis=0), 0.0, atol=1e-12)
    assert np.allclose(np.std(overwritten.X, axis=0, ddof=1), 1.0, atol=1e-12)
    assert result.history[-1].params["new_layer_name"] == "imputed"


def test_zscore_logs_stable_provenance_fields() -> None:
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    container = _make_container(x)

    result = zscore(
        container,
        assay_name="protein",
        source_layer="imputed",
        new_layer_name="z_custom",
        axis=1,
        ddof=0,
    )
    log = result.history[-1]

    assert log.action == "standardization_zscore"
    assert log.params == {
        "assay": "protein",
        "source_layer": "imputed",
        "new_layer_name": "z_custom",
        "axis": 1,
        "ddof": 0,
    }
