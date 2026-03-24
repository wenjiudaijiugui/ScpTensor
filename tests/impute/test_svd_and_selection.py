"""Tests for SVD-family imputation methods and mechanism-aware dispatch."""

from __future__ import annotations

import sys
import types
from importlib.util import find_spec

import numpy as np
import polars as pl
import pytest

from scptensor.core import Assay, MaskCode, MissingDependencyError, ScpContainer, ScpMatrix
from scptensor.core.exceptions import ScpValueError
from scptensor.impute import (
    impute,
    impute_iterative_svd,
    impute_softimpute,
)
from scptensor.impute.base import infer_missing_mechanism, list_impute_methods


@pytest.fixture
def svd_container() -> tuple[ScpContainer, np.ndarray, np.ndarray]:
    """Create deterministic dense matrix with missing values."""
    rng = np.random.default_rng(123)
    n_samples, n_features = 24, 12
    x_true = rng.normal(loc=8.0, scale=1.2, size=(n_samples, n_features))
    missing_mask = rng.random((n_samples, n_features)) < 0.22

    # Avoid all-missing row/col in this fixture.
    for i in range(n_samples):
        if np.all(missing_mask[i]):
            missing_mask[i, 0] = False
    for j in range(n_features):
        if np.all(missing_mask[:, j]):
            missing_mask[0, j] = False

    x_missing = x_true.copy()
    x_missing[missing_mask] = np.nan

    m = np.full((n_samples, n_features), MaskCode.VALID, dtype=np.int8)
    m[missing_mask] = MaskCode.LOD

    assay = Assay(var=pl.DataFrame({"_index": [f"p{j}" for j in range(n_features)]}))
    assay.add_layer("raw", ScpMatrix(X=x_missing, M=m))
    container = ScpContainer(
        obs=pl.DataFrame({"_index": [f"s{i}" for i in range(n_samples)]}),
        assays={"protein": assay},
    )
    return container, x_missing, missing_mask


def test_list_impute_methods_contains_svd_family() -> None:
    methods = set(list_impute_methods())
    assert {"iterative_svd", "softimpute"}.issubset(methods)


def test_iterative_svd_fills_missing_and_preserves_observed(
    svd_container: tuple[ScpContainer, np.ndarray, np.ndarray],
) -> None:
    container, x_missing, missing_mask = svd_container

    result = impute_iterative_svd(
        container,
        assay_name="protein",
        source_layer="raw",
        n_components=4,
        max_iter=30,
        tol=1e-6,
    )
    x_out = result.assays["protein"].layers["imputed_iterative_svd"].X
    m_out = result.assays["protein"].layers["imputed_iterative_svd"].M

    assert not np.any(np.isnan(x_out))
    np.testing.assert_allclose(x_out[~missing_mask], x_missing[~missing_mask], equal_nan=True)
    assert m_out is not None
    assert np.all(m_out[missing_mask] == MaskCode.IMPUTED)


def test_iterative_svd_available_via_unified_dispatch(
    svd_container: tuple[ScpContainer, np.ndarray, np.ndarray],
) -> None:
    container, _, _ = svd_container
    result = impute(
        container,
        method="iterative_svd",
        assay_name="protein",
        source_layer="raw",
        new_layer_name="via_dispatch_svd",
        n_components=3,
        max_iter=25,
    )
    assert "via_dispatch_svd" in result.assays["protein"].layers


def test_impute_auto_with_explicit_mechanism_selects_recommended_method(
    svd_container: tuple[ScpContainer, np.ndarray, np.ndarray],
) -> None:
    container, _, _ = svd_container
    result = impute(
        container,
        method="auto",
        missing_mechanism="mnar",
        assay_name="protein",
        source_layer="raw",
    )
    assert "imputed_qrilc" in result.assays["protein"].layers


def test_impute_auto_logs_selection_before_method_action(
    svd_container: tuple[ScpContainer, np.ndarray, np.ndarray],
) -> None:
    container, _, _ = svd_container
    initial_len = len(container.history)

    result = impute(
        container,
        method="auto",
        missing_mechanism="mnar",
        assay_name="protein",
        source_layer="raw",
        random_state=11,
    )

    assert len(result.history) == initial_len + 2
    assert result.history[-2].action == "impute_method_selection"
    assert result.history[-1].action == "impute_qrilc"
    assert result.history[-2].params["selected_method"] == "qrilc"
    assert result.history[-2].params["requested_method"] == "auto"
    assert result.history[-2].params["missing_mechanism"] == "mnar"


def test_impute_auto_fails_closed_when_recommended_method_unavailable(
    svd_container: tuple[ScpContainer, np.ndarray, np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scptensor.impute.base as impute_base

    container, _, _ = svd_container
    initial_len = len(container.history)
    monkeypatch.setattr(impute_base, "recommend_impute_method", lambda mechanism: "nonexistent")

    with pytest.raises(
        ScpValueError,
        match="recommended method 'nonexistent'.*unavailable.*Available methods",
    ):
        impute(
            container,
            method="auto",
            missing_mechanism="mnar",
            assay_name="protein",
            source_layer="raw",
        )

    assert len(container.history) == initial_len


def test_impute_explicit_mechanism_mismatch_logs_warning_before_method_action(
    svd_container: tuple[ScpContainer, np.ndarray, np.ndarray],
) -> None:
    container, _, _ = svd_container
    initial_len = len(container.history)

    result = impute(
        container,
        method="knn",
        missing_mechanism="mnar",
        assay_name="protein",
        source_layer="raw",
        k=3,
    )

    assert len(result.history) == initial_len + 2
    assert result.history[-2].action == "impute_mechanism_warning"
    assert result.history[-1].action == "impute_knn"
    assert result.history[-2].params["requested_method"] == "knn"
    assert result.history[-2].params["missing_mechanism"] == "mnar"
    assert result.history[-2].params["recommended_method"] == "qrilc"


def test_impute_auto_requires_assay_and_source_layer(
    svd_container: tuple[ScpContainer, np.ndarray, np.ndarray],
) -> None:
    container, _, _ = svd_container
    with pytest.raises(ScpValueError, match="assay_name"):
        impute(container, method="auto")


def test_invalid_missing_mechanism_raises(
    svd_container: tuple[ScpContainer, np.ndarray, np.ndarray],
) -> None:
    container, _, _ = svd_container
    with pytest.raises(ScpValueError, match="missing_mechanism"):
        impute(
            container,
            method="knn",
            missing_mechanism="invalid",
            assay_name="protein",
            source_layer="raw",
        )


def test_infer_missing_mechanism_detects_mnar_pattern() -> None:
    """Construct strong intensity-dependent missingness (left-censoring-like)."""
    rng = np.random.default_rng(77)
    n_samples, n_features = 30, 12
    feature_means = np.linspace(3.0, 12.0, n_features)

    x = np.vstack(
        [rng.normal(loc=feature_means[j], scale=0.3, size=n_samples) for j in range(n_features)],
    ).T

    missing_mask = np.zeros_like(x, dtype=bool)
    for j in range(n_features):
        miss_rate = 0.82 - 0.06 * j  # low-intensity features have more missingness
        miss_rate = float(np.clip(miss_rate, 0.1, 0.9))
        missing_mask[:, j] = rng.random(n_samples) < miss_rate
        if np.all(missing_mask[:, j]):
            missing_mask[0, j] = False

    x[missing_mask] = np.nan
    assay = Assay(var=pl.DataFrame({"_index": [f"p{j}" for j in range(n_features)]}))
    assay.add_layer("raw", ScpMatrix(X=x, M=None))
    container = ScpContainer(
        obs=pl.DataFrame({"_index": [f"s{i}" for i in range(n_samples)]}),
        assays={"protein": assay},
    )

    mechanism, _ = infer_missing_mechanism(container, assay_name="protein", source_layer="raw")
    assert mechanism == "mnar"


def test_softimpute_dependency_or_functionality(
    svd_container: tuple[ScpContainer, np.ndarray, np.ndarray],
) -> None:
    container, x_missing, missing_mask = svd_container

    if find_spec("fancyimpute") is None:
        with pytest.raises(MissingDependencyError, match="fancyimpute"):
            impute_softimpute(container, assay_name="protein", source_layer="raw")
        return

    result = impute_softimpute(
        container,
        assay_name="protein",
        source_layer="raw",
        rank=3,
        max_iter=30,
        convergence_threshold=1e-5,
        random_state=11,
    )
    x_out = result.assays["protein"].layers["imputed_softimpute"].X
    assert not np.any(np.isnan(x_out))
    np.testing.assert_allclose(x_out[~missing_mask], x_missing[~missing_mask], equal_nan=True)


def test_softimpute_does_not_patch_sklearn_check_array(
    svd_container: tuple[ScpContainer, np.ndarray, np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sklearn.utils

    class _FakeSoftImpute:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def fit_transform(self, x: np.ndarray) -> np.ndarray:
            out = np.array(x, copy=True)
            out[np.isnan(out)] = 0.0
            return out

    fake_module = types.ModuleType("fancyimpute")
    fake_module.SoftImpute = _FakeSoftImpute  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fancyimpute", fake_module)

    original_check_array = sklearn.utils.check_array
    container, _, _ = svd_container
    impute_softimpute(container, assay_name="protein", source_layer="raw")

    assert sklearn.utils.check_array is original_check_array


def test_softimpute_incompatible_sklearn_interface_raises_actionable_error(
    svd_container: tuple[ScpContainer, np.ndarray, np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BrokenSoftImpute:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def fit_transform(self, x: np.ndarray) -> np.ndarray:
            del x
            raise TypeError("check_array() got an unexpected keyword argument 'force_all_finite'")

    fake_module = types.ModuleType("fancyimpute")
    fake_module.SoftImpute = _BrokenSoftImpute  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fancyimpute", fake_module)

    container, _, _ = svd_container
    with pytest.raises(ScpValueError, match="dependency interface mismatch"):
        impute_softimpute(container, assay_name="protein", source_layer="raw")
