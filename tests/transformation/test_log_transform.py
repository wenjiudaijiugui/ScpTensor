"""Tests for transformation.log_transform."""

from __future__ import annotations

import importlib

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp

from scptensor import normalization
from scptensor.core.exceptions import ScpValueError, ValidationError
from scptensor.core.structures import Assay, ScpContainer, ScpMatrix
from scptensor.transformation import log_transform


def _make_container() -> ScpContainer:
    obs = pl.DataFrame({"_index": ["s1", "s2"]})
    var = pl.DataFrame({"_index": ["p1", "p2", "p3"]})
    x = np.array([[0.0, 2.0, 4.0], [1.0, 3.0, 7.0]])
    assay = Assay(var=var, layers={"raw": ScpMatrix(X=x)})
    return ScpContainer(obs=obs, assays={"protein": assay})


def _make_masked_container() -> ScpContainer:
    obs = pl.DataFrame({"_index": ["s1", "s2"]})
    var = pl.DataFrame({"_index": ["p1", "p2", "p3"]})
    x = np.array([[0.0, 2.0, 4.0], [1.0, 3.0, 7.0]])
    m = np.array([[0, 1, 0], [0, 0, 2]], dtype=np.int8)
    assay = Assay(var=var, layers={"raw": ScpMatrix(X=x, M=m)})
    return ScpContainer(obs=obs, assays={"protein": assay})


def _make_low_range_linear_container(n_samples: int = 16, n_features: int = 8) -> ScpContainer:
    rng = np.random.default_rng(7)
    obs = pl.DataFrame({"_index": [f"s{i}" for i in range(n_samples)]})
    var = pl.DataFrame({"_index": [f"p{j}" for j in range(n_features)]})
    x = rng.uniform(0.0, 12.0, size=(n_samples, n_features))
    assay = Assay(var=var, layers={"raw": ScpMatrix(X=x)})
    return ScpContainer(obs=obs, assays={"protein": assay})


def _make_logged_container(
    layer_name: str = "log2",
    n_samples: int = 16,
    n_features: int = 8,
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
        container,
        assay_name="protein",
        source_layer="raw",
        new_layer_name="log2",
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


def test_log_transform_preserves_mask_reference_on_new_layer() -> None:
    container = _make_masked_container()
    source_m = container.assays["protein"].layers["raw"].M

    log_transform(
        container,
        assay_name="protein",
        source_layer="raw",
        new_layer_name="log2",
    )

    assert container.assays["protein"].layers["log2"].M is source_m


def test_log_transform_same_name_overwrites_source_layer_entry() -> None:
    container = _make_container()
    raw_layer = container.assays["protein"].layers["raw"]
    raw_x_before = raw_layer.X.copy()

    log_transform(
        container,
        assay_name="protein",
        source_layer="raw",
        new_layer_name="raw",
    )

    raw_after = container.assays["protein"].layers["raw"]
    assert raw_after is not raw_layer
    assert np.allclose(raw_after.X, np.log2(raw_x_before + 1.0))
    assert container.history[-1].action == "log_transform"


def test_log_transform_skip_same_name_does_not_replace_source_layer() -> None:
    container = _make_logged_container(layer_name="log2")
    source_layer = container.assays["protein"].layers["log2"]
    source_x = source_layer.X.copy()

    with pytest.warns(UserWarning, match="already log-transformed"):
        log_transform(
            container,
            assay_name="protein",
            source_layer="log2",
            new_layer_name="log2",
        )

    assert list(container.assays["protein"].layers.keys()) == ["log2"]
    assert container.assays["protein"].layers["log2"] is source_layer
    assert np.allclose(container.assays["protein"].layers["log2"].X, source_x)
    assert container.history[-1].action == "log_transform_skipped"


def test_log_transform_custom_base_offset_dense_path_preserves_source_and_history() -> None:
    obs = pl.DataFrame({"_index": ["s1", "s2"]})
    var = pl.DataFrame({"_index": ["p1", "p2"]})
    x = np.array([[-2.0, 3.0], [0.0, 8.0]])
    assay = Assay(var=var, layers={"raw": ScpMatrix(X=x.copy())})
    container = ScpContainer(obs=obs, assays={"protein": assay})

    with pytest.warns(UserWarning, match="negative values"):
        log_transform(
            container,
            assay_name="protein",
            source_layer="raw",
            new_layer_name="log10",
            base=10.0,
            offset=2.0,
            use_jit=False,
            detect_logged=False,
        )

    result = container.assays["protein"].layers["log10"].X
    expected = np.log10(np.maximum(x, 0.0) + 2.0)
    assert np.allclose(result, expected)
    assert np.allclose(container.assays["protein"].layers["raw"].X, x)

    history = container.history[-1]
    assert history.action == "log_transform"
    assert history.params["base"] == 10.0
    assert history.params["offset"] == 2.0
    assert history.params["use_jit"] is False
    assert history.params["sparse_input"] is False
    assert history.params["detect_logged"] is False
    assert history.params["logged_detection_reason"] == "logged-detection disabled"


def test_log_transform_rejects_base_equal_to_one() -> None:
    container = _make_container()

    with pytest.raises(ScpValueError, match="not equal to 1"):
        log_transform(
            container,
            assay_name="protein",
            source_layer="raw",
            new_layer_name="log_bad",
            base=1.0,
            detect_logged=False,
        )


def test_log_transform_rejects_offset_zero_when_zero_values_present() -> None:
    container = _make_container()

    with pytest.raises(ValidationError, match="requires strictly positive input values"):
        log_transform(
            container,
            assay_name="protein",
            source_layer="raw",
            new_layer_name="log_bad",
            offset=0.0,
            detect_logged=False,
        )


def test_log_transform_sparse_path_propagates_runtime_params_and_forces_csr() -> None:
    module = importlib.import_module("scptensor.transformation.log_transform")
    obs = pl.DataFrame({"_index": ["s1", "s2"]})
    var = pl.DataFrame({"_index": ["p1", "p2", "p3"]})
    x = sp.csc_matrix(np.array([[0.0, 1.0, 0.0], [2.0, 0.0, 3.0]]))
    assay = Assay(var=var, layers={"raw": ScpMatrix(X=x)})
    container = ScpContainer(obs=obs, assays={"protein": assay})
    calls: dict[str, float | bool] = {}

    def fake_sparse_safe_log1p_with_scale(
        matrix: np.ndarray | sp.spmatrix,
        offset: float = 1.0,
        scale: float = 1.0,
        use_jit: bool = True,
    ) -> np.ndarray | sp.spmatrix:
        calls["offset"] = offset
        calls["scale"] = scale
        calls["use_jit"] = use_jit
        assert sp.isspmatrix(matrix)
        return sp.csc_matrix(np.full(matrix.shape, 7.0))

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(module, "sparse_safe_log1p_with_scale", fake_sparse_safe_log1p_with_scale)
    try:
        module.log_transform(
            container,
            assay_name="protein",
            source_layer="raw",
            new_layer_name="log",
            base=np.e,
            offset=1.0,
            use_jit=False,
            detect_logged=False,
        )
    finally:
        monkeypatch.undo()

    result = container.assays["protein"].layers["log"].X
    assert sp.isspmatrix_csr(result)
    assert np.allclose(result.toarray(), np.full((2, 3), 7.0))
    assert calls == {"offset": 1.0, "scale": 1.0, "use_jit": False}
    assert container.history[-1].params["sparse_input"] is True
    assert container.history[-1].params["use_jit"] is False


def test_log_transform_sparse_negative_path_uses_single_copy_helper_when_offset_preserves_zeros() -> (
    None
):
    module = importlib.import_module("scptensor.transformation.log_transform")
    obs = pl.DataFrame({"_index": ["s1", "s2"]})
    var = pl.DataFrame({"_index": ["p1", "p2", "p3"]})
    x = sp.csc_matrix(np.array([[0.0, -2.0, 0.0], [3.0, 0.0, 4.0]]))
    x_before = x.copy()
    assay = Assay(var=var, layers={"raw": ScpMatrix(X=x)})
    container = ScpContainer(obs=obs, assays={"protein": assay})
    calls: dict[str, float | bool] = {}

    def fake_copy_and_transform_sparse_log(
        matrix: sp.spmatrix,
        *,
        offset: float,
        scale: float,
        use_jit: bool,
        clip_negative: bool = False,
    ) -> sp.spmatrix:
        calls["offset"] = offset
        calls["scale"] = scale
        calls["use_jit"] = use_jit
        calls["clip_negative"] = clip_negative
        assert sp.isspmatrix(matrix)
        return sp.csc_matrix(np.full(matrix.shape, 11.0))

    def fail_sparse_safe_log1p_with_scale(*args: object, **kwargs: object) -> object:
        raise AssertionError(
            "negative sparse path should not re-enter sparse_safe_log1p_with_scale",
        )

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        module,
        "_copy_and_transform_sparse_log",
        fake_copy_and_transform_sparse_log,
    )
    monkeypatch.setattr(module, "sparse_safe_log1p_with_scale", fail_sparse_safe_log1p_with_scale)
    try:
        with pytest.warns(UserWarning, match="negative values"):
            module.log_transform(
                container,
                assay_name="protein",
                source_layer="raw",
                new_layer_name="log",
                base=np.e,
                offset=1.0,
                use_jit=False,
                detect_logged=False,
            )
    finally:
        monkeypatch.undo()

    result = container.assays["protein"].layers["log"].X
    assert sp.isspmatrix_csr(result)
    assert np.allclose(result.toarray(), np.full((2, 3), 11.0))
    assert calls == {
        "offset": 1.0,
        "scale": 1.0,
        "use_jit": False,
        "clip_negative": True,
    }
    np.testing.assert_allclose(
        container.assays["protein"].layers["raw"].X.toarray(),
        x_before.toarray(),
    )


def test_log_transform_sparse_nonunit_offset_densifies_and_matches_dense_formula() -> None:
    obs = pl.DataFrame({"_index": ["s1", "s2"]})
    var = pl.DataFrame({"_index": ["p1", "p2", "p3"]})
    x_dense = np.array([[0.0, 1.0, 0.0], [2.0, 0.0, 3.0]])
    assay = Assay(var=var, layers={"raw": ScpMatrix(X=sp.csr_matrix(x_dense))})
    container = ScpContainer(obs=obs, assays={"protein": assay})

    log_transform(
        container,
        assay_name="protein",
        source_layer="raw",
        new_layer_name="log",
        base=np.e,
        offset=2.5,
        use_jit=False,
        detect_logged=False,
    )

    result = container.assays["protein"].layers["log"].X
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, np.log(x_dense + 2.5))
