"""Tests for baseline imputation methods (none/zero/row_*)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from scptensor.core import Assay, MaskCode, ScpContainer, ScpMatrix
from scptensor.impute import (
    impute,
    impute_half_row_min,
    impute_none,
    impute_row_mean,
    impute_row_median,
    impute_zero,
)
from scptensor.impute.base import list_impute_methods


@pytest.fixture
def baseline_container() -> tuple[ScpContainer, np.ndarray, np.ndarray]:
    """Create a small dense container with deterministic missingness."""
    rng = np.random.default_rng(42)
    n_samples, n_features = 8, 6
    x_true = rng.normal(loc=10.0, scale=2.0, size=(n_samples, n_features))

    missing_mask = rng.random((n_samples, n_features)) < 0.25
    # Avoid all-missing rows/cols for deterministic row-stat checks.
    for row in range(n_samples):
        if np.all(missing_mask[row, :]):
            missing_mask[row, 0] = False
    for col in range(n_features):
        if np.all(missing_mask[:, col]):
            missing_mask[0, col] = False

    x_missing = x_true.copy()
    x_missing[missing_mask] = np.nan

    m = np.full((n_samples, n_features), MaskCode.VALID, dtype=np.int8)
    m[missing_mask] = MaskCode.LOD

    assay = Assay(var=pl.DataFrame({"_index": [f"p{i}" for i in range(n_features)]}))
    assay.add_layer("raw", ScpMatrix(X=x_missing, M=m))
    container = ScpContainer(
        obs=pl.DataFrame({"_index": [f"s{i}" for i in range(n_samples)]}),
        assays={"protein": assay},
    )
    return container, x_missing, missing_mask


def test_list_impute_methods_contains_baselines() -> None:
    methods = set(list_impute_methods())
    assert {"none", "zero", "row_mean", "row_median", "half_row_min"}.issubset(methods)


def test_impute_none_passthrough_keeps_nan_and_mask(
    baseline_container: tuple[ScpContainer, np.ndarray, np.ndarray],
) -> None:
    container, x_missing, missing_mask = baseline_container
    original_mask = container.assays["protein"].layers["raw"].M.copy()

    result = impute_none(container, assay_name="protein", source_layer="raw")
    x_out = result.assays["protein"].layers["imputed_none"].X
    m_out = result.assays["protein"].layers["imputed_none"].M

    np.testing.assert_allclose(x_out, x_missing, equal_nan=True)
    assert np.array_equal(np.isnan(x_out), missing_mask)
    assert np.array_equal(m_out, original_mask)


@pytest.mark.parametrize(
    ("method_fn", "layer_name"),
    [
        (impute_zero, "imputed_zero"),
        (impute_row_mean, "imputed_row_mean"),
        (impute_row_median, "imputed_row_median"),
        (impute_half_row_min, "imputed_half_row_min"),
    ],
)
def test_baseline_methods_fill_missing_and_preserve_observed(
    baseline_container: tuple[ScpContainer, np.ndarray, np.ndarray],
    method_fn,
    layer_name: str,
) -> None:
    container, x_missing, missing_mask = baseline_container
    result = method_fn(container, assay_name="protein", source_layer="raw")

    x_out = result.assays["protein"].layers[layer_name].X
    m_out = result.assays["protein"].layers[layer_name].M

    assert x_out.shape == x_missing.shape
    assert not np.any(np.isnan(x_out))
    np.testing.assert_allclose(x_out[~missing_mask], x_missing[~missing_mask], equal_nan=True)
    assert m_out is not None
    assert np.all(m_out[missing_mask] == MaskCode.IMPUTED)


def test_half_row_min_expected_fill_values(
    baseline_container: tuple[ScpContainer, np.ndarray, np.ndarray],
) -> None:
    container, x_missing, missing_mask = baseline_container
    fraction = 0.5
    result = impute_half_row_min(
        container,
        assay_name="protein",
        source_layer="raw",
        fraction=fraction,
    )
    x_out = result.assays["protein"].layers["imputed_half_row_min"].X

    for row_idx in range(x_missing.shape[0]):
        row = x_missing[row_idx, :]
        valid = np.isfinite(row)
        row_min = float(np.min(row[valid]))
        expected_fill = row_min * fraction
        row_missing = missing_mask[row_idx, :]
        if np.any(row_missing):
            np.testing.assert_allclose(x_out[row_idx, row_missing], expected_fill)


def test_unified_impute_dispatch_for_new_methods(
    baseline_container: tuple[ScpContainer, np.ndarray, np.ndarray],
) -> None:
    container, _, _ = baseline_container
    result = impute(
        container,
        method="row_mean",
        assay_name="protein",
        source_layer="raw",
        new_layer_name="via_dispatch",
    )
    assert "via_dispatch" in result.assays["protein"].layers
