"""Checklist-oriented correctness tests for existing imputation methods."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl
import pytest

from scptensor.core import Assay, MaskCode, ScpContainer, ScpMatrix
from scptensor.impute import (
    impute_bpca,
    impute_knn,
    impute_lls,
    impute_mf,
    impute_minprob,
    impute_qrilc,
)


def _build_container(seed: int = 7) -> tuple[ScpContainer, np.ndarray, np.ndarray]:
    """Create deterministic test container with partial missingness."""
    rng = np.random.default_rng(seed)
    n_samples, n_features = 12, 9
    x = rng.normal(loc=6.0, scale=1.5, size=(n_samples, n_features))

    missing_mask = rng.random((n_samples, n_features)) < 0.2
    # Ensure every row/column has at least one observed value.
    for i in range(n_samples):
        if np.all(missing_mask[i, :]):
            missing_mask[i, 0] = False
    for j in range(n_features):
        if np.all(missing_mask[:, j]):
            missing_mask[0, j] = False

    x_missing = x.copy()
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


MethodSpec = tuple[str, Callable[..., ScpContainer], dict[str, object]]

METHOD_SPECS: list[MethodSpec] = [
    ("knn", impute_knn, {"k": 3}),
    ("lls", impute_lls, {"k": 3, "max_iter": 20, "tol": 1e-6}),
    ("bpca", impute_bpca, {"n_components": 3, "max_iter": 40, "random_state": 7}),
    ("missforest", impute_mf, {"max_iter": 2, "n_estimators": 20, "n_jobs": 1, "random_state": 7}),
    ("qrilc", impute_qrilc, {"q": 0.1, "random_state": 7}),
    ("minprob", impute_minprob, {"sigma": 2.0, "random_state": 7}),
]


@pytest.mark.parametrize(("method_name", "method_fn", "kwargs"), METHOD_SPECS)
def test_checklist_shape_fill_mask_and_source_integrity(
    method_name: str,
    method_fn: Callable[..., ScpContainer],
    kwargs: dict[str, object],
) -> None:
    container, x_before, missing_mask = _build_container(seed=11)
    source_snapshot = container.assays["protein"].layers["raw"].X.copy()
    source_mask_snapshot = container.assays["protein"].layers["raw"].M.copy()

    out_layer = f"imputed_{method_name}"
    result = method_fn(
        container,
        assay_name="protein",
        source_layer="raw",
        new_layer_name=out_layer,
        **kwargs,
    )
    x_out = result.assays["protein"].layers[out_layer].X
    m_out = result.assays["protein"].layers[out_layer].M

    # Shape preserved
    assert x_out.shape == x_before.shape

    # Source layer integrity
    np.testing.assert_allclose(
        container.assays["protein"].layers["raw"].X, source_snapshot, equal_nan=True
    )
    assert np.array_equal(container.assays["protein"].layers["raw"].M, source_mask_snapshot)

    # Non-missing values unchanged
    np.testing.assert_allclose(x_out[~missing_mask], x_before[~missing_mask], equal_nan=True)

    # Missing values filled
    assert not np.any(np.isnan(x_out[missing_mask]))

    # Mask update
    assert m_out is not None
    assert np.all(m_out[missing_mask] == MaskCode.IMPUTED)
    assert np.array_equal(m_out[~missing_mask], source_mask_snapshot[~missing_mask])


@pytest.mark.parametrize(("method_name", "method_fn", "kwargs"), METHOD_SPECS)
def test_checklist_reproducibility(
    method_name: str,
    method_fn: Callable[..., ScpContainer],
    kwargs: dict[str, object],
) -> None:
    c1, _, _ = _build_container(seed=19)
    c2, _, _ = _build_container(seed=19)
    out_layer = f"imputed_{method_name}"

    r1 = method_fn(c1, assay_name="protein", source_layer="raw", new_layer_name=out_layer, **kwargs)
    r2 = method_fn(c2, assay_name="protein", source_layer="raw", new_layer_name=out_layer, **kwargs)

    x1 = r1.assays["protein"].layers[out_layer].X
    x2 = r2.assays["protein"].layers[out_layer].X
    np.testing.assert_allclose(x1, x2, equal_nan=True)
