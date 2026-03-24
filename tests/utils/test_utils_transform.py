"""Tests for DIA-focused utilities in scptensor.utils.transform."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from scptensor.utils.transform import quantile_normalize, robust_scale


@pytest.fixture
def sample_data() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(loc=10.0, scale=3.0, size=(20, 10))


class TestQuantileNormalize:
    """Tests for quantile_normalize function."""

    def test_quantile_normalize_shape(self, sample_data: np.ndarray) -> None:
        result = quantile_normalize(sample_data, axis=0)
        assert result.shape == sample_data.shape

    def test_quantile_normalize_axis_0(self, sample_data: np.ndarray) -> None:
        result = quantile_normalize(sample_data, axis=0)
        sorted_result = np.sort(result, axis=0)
        for col in range(1, sorted_result.shape[1]):
            np.testing.assert_allclose(sorted_result[:, 0], sorted_result[:, col])

    def test_quantile_normalize_axis_1(self, sample_data: np.ndarray) -> None:
        result = quantile_normalize(sample_data, axis=1)
        sorted_result = np.sort(result, axis=1)
        for row in range(1, sorted_result.shape[0]):
            np.testing.assert_allclose(sorted_result[0, :], sorted_result[row, :])

    def test_quantile_normalize_invalid_axis(self, sample_data: np.ndarray) -> None:
        with pytest.raises(ValueError, match="axis must be 0 or 1"):
            quantile_normalize(sample_data, axis=2)

    def test_quantile_normalize_sparse_matrix(self, sample_data: np.ndarray) -> None:
        result = quantile_normalize(sp.csr_matrix(sample_data), axis=0)
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_data.shape

    def test_quantile_normalize_preserves_average_rank_ties(self) -> None:
        x = np.array([[1.0, 2.0, 3.0], [1.0, 4.0, 6.0], [3.0, 5.0, 9.0]])

        result = quantile_normalize(x, axis=0)

        expected = np.array(
            [
                [17.0 / 6.0, 2.0, 2.0],
                [17.0 / 6.0, 11.0 / 3.0, 11.0 / 3.0],
                [17.0 / 3.0, 17.0 / 3.0, 17.0 / 3.0],
            ],
        )
        np.testing.assert_allclose(result, expected)

    def test_quantile_normalize_copy_false_keeps_fractional_values_for_int_input(self) -> None:
        x = np.array([[1, 2, 3], [10, 20, 30]], dtype=np.int64)

        result = quantile_normalize(x, axis=1, copy=False)

        np.testing.assert_allclose(result, np.array([[5.5, 11.0, 16.5], [5.5, 11.0, 16.5]]))


class TestRobustScale:
    """Tests for robust_scale function."""

    def test_robust_scale_shape(self, sample_data: np.ndarray) -> None:
        result = robust_scale(sample_data, axis=0)
        assert result.shape == sample_data.shape

    def test_robust_scale_centering(self, sample_data: np.ndarray) -> None:
        result = robust_scale(sample_data, axis=0, with_scaling=False)
        medians = np.median(result, axis=0)
        assert np.allclose(medians, 0.0, atol=1e-10)

    def test_robust_scale_scaling(self, sample_data: np.ndarray) -> None:
        result = robust_scale(sample_data, axis=0, with_centering=False)
        q75 = np.percentile(result, 75, axis=0)
        q25 = np.percentile(result, 25, axis=0)
        iqr = q75 - q25
        assert np.allclose(iqr, 1.0, atol=1e-8)

    def test_robust_scale_sparse_matrix(self, sample_data: np.ndarray) -> None:
        result = robust_scale(sp.csr_matrix(sample_data), axis=0)
        assert result.shape == sample_data.shape

    def test_robust_scale_handles_zero_iqr(self) -> None:
        x = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        result = robust_scale(x, axis=0)
        assert np.all(np.isfinite(result))

    def test_robust_scale_rejects_invalid_axis(self, sample_data: np.ndarray) -> None:
        with pytest.raises(ValueError, match="axis must be 0 or 1"):
            robust_scale(sample_data, axis=2)
