"""Regression tests for JIT/fallback public-surface alignment."""

import numpy as np

from scptensor.core.jit_ops import (
    count_above_threshold,
    impute_missing_with_col_means_jit,
    kmeans_core_numba,
    kmeans_plusplus_init_numba,
    mean_axis_no_nan,
    mean_no_nan,
    var_axis_no_nan,
    var_no_nan,
    vectorized_mannwhitney_row,
)


def test_nan_statistics_helpers_return_zero_for_all_nan_inputs() -> None:
    """JIT and fallback code paths should share the zero-for-empty convention."""
    arr = np.array([np.nan, np.nan], dtype=np.float64)

    assert mean_no_nan(arr) == 0.0
    assert var_no_nan(arr) == 0.0


def test_nan_axis_statistics_helpers_return_zero_for_all_nan_axes() -> None:
    """Axis reducers should keep the JIT convention on all-NaN rows and columns."""
    X = np.array(
        [
            [np.nan, 1.0, np.nan],
            [np.nan, np.nan, np.nan],
            [2.0, np.nan, np.nan],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(mean_axis_no_nan(X, axis=0), np.array([2.0, 1.0, 0.0]))
    np.testing.assert_allclose(mean_axis_no_nan(X, axis=1), np.array([1.0, 0.0, 2.0]))
    np.testing.assert_allclose(var_axis_no_nan(X, axis=0), np.array([0.0, 0.0, 0.0]))
    np.testing.assert_allclose(var_axis_no_nan(X, axis=1), np.array([0.0, 0.0, 0.0]))


def test_vectorized_mannwhitney_row_is_available_from_public_surface() -> None:
    """The helper exported in __all__ should exist regardless of Numba availability."""
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array([3.0, 4.0, 5.0], dtype=np.float64)

    statistic, p_value = vectorized_mannwhitney_row(x, y)

    assert np.isfinite(statistic)
    assert 0.0 <= p_value <= 1.0


def test_count_above_threshold_keeps_axis_contract() -> None:
    """Axis semantics must stay aligned between JIT and fallback implementations."""
    X = np.array(
        [
            [0.1, 2.0, 3.0],
            [4.0, 0.2, 5.0],
            [6.0, 7.0, 0.3],
        ],
        dtype=np.float64,
    )

    np.testing.assert_array_equal(
        count_above_threshold(X, threshold=1.0, axis=0), np.array([2, 2, 2])
    )
    np.testing.assert_array_equal(
        count_above_threshold(X, threshold=1.0, axis=1), np.array([2, 2, 2])
    )


def test_impute_missing_with_col_means_jit_uses_zero_for_all_nan_columns() -> None:
    """Public mean-fill helper should keep the documented all-NaN-column fallback."""
    X = np.array(
        [
            [1.0, np.nan, np.nan],
            [3.0, 5.0, np.nan],
            [np.nan, 7.0, np.nan],
        ],
        dtype=np.float64,
    )

    impute_missing_with_col_means_jit(X)

    expected = np.array(
        [
            [1.0, 6.0, 0.0],
            [3.0, 5.0, 0.0],
            [2.0, 7.0, 0.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(X, expected)


def test_kmeans_numba_helpers_keep_public_shape_and_assignment_contract() -> None:
    """Public k-means helpers should remain usable regardless of backend."""
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.2],
            [10.0, 10.0],
            [10.2, 10.1],
        ],
        dtype=np.float64,
    )

    init = kmeans_plusplus_init_numba(X, k=2, seed=0)
    centers, labels, inertia = kmeans_core_numba(
        X,
        k=2,
        max_iter=50,
        tol=1e-6,
        init_centers=init,
        rng_seed=0,
    )

    assert centers.shape == (2, 2)
    assert labels.shape == (4,)
    assert np.isfinite(inertia)
    assert len(np.unique(labels)) == 2
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]
