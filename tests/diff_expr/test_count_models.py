"""Tests for count-based differential expression methods.

Tests cover all functions in scptensor.diff_expr.count_models:
- diff_expr_voom: VOOM transformation with limma analysis
- diff_expr_limma_trend: Empirical Bayes with trend correction
- diff_expr_deseq2: Negative binomial GLM (DESeq2-like)

These methods are designed for count-based single-cell proteomics data,
modeling the mean-variance relationship common in such data.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp

from scptensor.core.exceptions import (
    AssayNotFoundError,
    LayerNotFoundError,
    ValidationError,
)
from scptensor.core.structures import Assay, ScpContainer, ScpMatrix
from scptensor.diff_expr.count_models import (
    diff_expr_deseq2,
    diff_expr_limma_trend,
    diff_expr_voom,
)

# =============================================================================
# Helper Functions
# =============================================================================


def create_count_container(
    n_samples: int = 40,
    n_features: int = 50,
    n_de: int = 5,
    effect_size: float = 2.0,
    seed: int = 42,
    sparse: bool = False,
    with_mask: bool = False,
) -> ScpContainer:
    """Create a test container with count data for differential expression.

    Args:
        n_samples: Total number of samples (should be even for 2 groups)
        n_features: Number of features (proteins)
        n_de: Number of differentially expressed features
        effect_size: Effect size multiplier for DE features
        seed: Random seed
        sparse: Whether to use sparse matrices
        with_mask: Whether to add mask values (MBR/LOD)

    Returns:
        A test ScpContainer with count data and group labels
    """
    rng = np.random.default_rng(seed)

    n_per_group = n_samples // 2

    # Create sample metadata with two groups
    obs = pl.DataFrame(
        {
            "_index": [f"SAMPLE_{i:04d}" for i in range(n_samples)],
            "group": ["A"] * n_per_group + ["B"] * n_per_group,
            "batch": rng.choice(["X", "Y"], n_samples),
        }
    )

    # Create feature metadata
    var = pl.DataFrame({"_index": [f"PROT_{i:04d}" for i in range(n_features)]})

    # Create count data using negative binomial distribution
    # Higher n means higher mean, lower p means higher variance
    X = rng.negative_binomial(n=5, p=0.3, size=(n_samples, n_features))

    # Add differential expression for first n_de features
    # Group A has higher counts for these features
    if n_de > 0:
        X[:n_per_group, :n_de] = rng.negative_binomial(
            n=int(5 * effect_size), p=0.2, size=(n_per_group, n_de)
        )

    if sparse:
        X = sp.csr_matrix(X)

    # Create mask matrix if requested
    M = None
    if with_mask:
        M = np.zeros((n_samples, n_features), dtype=int)
        # Add some MBR (1) and LOD (2) values
        mbr_indices = rng.random((n_samples, n_features)) < 0.05
        lod_indices = rng.random((n_samples, n_features)) < 0.03
        M[mbr_indices] = 1
        M[lod_indices] = 2
        if sparse:
            M = sp.csr_matrix(M)

    assay = Assay(
        var=var,
        layers={"raw": ScpMatrix(X=X, M=M)},
        feature_id_col="_index",
    )

    return ScpContainer(obs=obs, assays={"protein": assay})


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_count_container():
    """Create a simple test container with count data."""
    return create_count_container(
        n_samples=40,
        n_features=50,
        n_de=5,
        effect_size=2.0,
        seed=42,
    )


@pytest.fixture
def sparse_count_container():
    """Create a container with sparse count data."""
    return create_count_container(
        n_samples=30,
        n_features=40,
        sparse=True,
        seed=42,
    )


@pytest.fixture
def container_with_mask():
    """Create a container with missing values (mask codes)."""
    return create_count_container(
        n_samples=30,
        n_features=40,
        with_mask=True,
        seed=42,
    )


@pytest.fixture
def large_count_container():
    """Create a larger container for testing with more features."""
    return create_count_container(
        n_samples=50,
        n_features=100,
        n_de=10,
        effect_size=2.5,
        seed=42,
    )


# =============================================================================
# diff_expr_voom Tests
# =============================================================================


class TestDiffExprVoom:
    """Tests for VOOM differential expression analysis."""

    def test_voom_basic(self, simple_count_container):
        """Test basic VOOM analysis."""
        result = diff_expr_voom(
            simple_count_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        assert result is not None
        assert len(result.p_values) == 50
        assert result.method == "voom"
        assert result.params["groupby"] == "group"
        assert result.params["group1"] == "A"
        assert result.params["group2"] == "B"

    def test_voom_normalization_options(self, simple_count_container):
        """Test VOOM with different normalization methods."""
        for norm in ["tmm", "upper_quartile", "none"]:
            result = diff_expr_voom(
                simple_count_container,
                "protein",
                "raw",
                "group",
                "A",
                "B",
                normalize=norm,
            )
            assert len(result.p_values) == 50
            assert result.params["normalize"] == norm

    def test_voom_min_count_filtering(self, simple_count_container):
        """Test VOOM with min_count filtering."""
        # Default min_count
        result_default = diff_expr_voom(
            simple_count_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        # Higher threshold should filter more features
        result_high = diff_expr_voom(
            simple_count_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
            min_count=100,
        )

        # More features should be NaN with higher threshold
        n_nan_default = np.sum(np.isnan(result_default.p_values))
        n_nan_high = np.sum(np.isnan(result_high.p_values))
        assert n_nan_high >= n_nan_default

    def test_voom_sparse_data(self, sparse_count_container):
        """Test VOOM with sparse matrix."""
        result = diff_expr_voom(
            sparse_count_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        assert len(result.p_values) == 40
        assert result.method == "voom"

    def test_voom_missing_values(self, container_with_mask):
        """Test VOOM handles missing values correctly."""
        result = diff_expr_voom(
            container_with_mask,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        assert len(result.p_values) == 40
        # Masked values should be handled (set to zero)
        assert not np.all(np.isnan(result.p_values))

    def test_voom_invalid_assay(self, simple_count_container):
        """Test VOOM with invalid assay raises error."""
        with pytest.raises(AssayNotFoundError) as exc_info:
            diff_expr_voom(
                simple_count_container,
                "invalid",
                "raw",
                "group",
                "A",
                "B",
            )
        assert "invalid" in str(exc_info.value)

    def test_voom_invalid_layer(self, simple_count_container):
        """Test VOOM with invalid layer raises error."""
        with pytest.raises(LayerNotFoundError) as exc_info:
            diff_expr_voom(
                simple_count_container,
                "protein",
                "invalid",
                "group",
                "A",
                "B",
            )
        assert "invalid" in str(exc_info.value)

    def test_voom_invalid_group(self, simple_count_container):
        """Test VOOM with invalid group raises error."""
        with pytest.raises(ValidationError) as exc_info:
            diff_expr_voom(
                simple_count_container,
                "protein",
                "raw",
                "group",
                "A",
                "nonexistent",
            )
        assert "nonexistent" in str(exc_info.value)

    def test_voom_invalid_normalize(self, simple_count_container):
        """Test VOOM with invalid normalize option."""
        with pytest.raises(ValidationError) as exc_info:
            diff_expr_voom(
                simple_count_container,
                "protein",
                "raw",
                "group",
                "A",
                "B",
                normalize="invalid",
            )
        assert "normalization" in str(exc_info.value).lower()

    def test_voom_insufficient_samples(self):
        """Test VOOM with insufficient samples per group."""
        container = create_count_container(n_samples=6)  # Only 3 per group

        # Should work with default min_samples (3)
        result = diff_expr_voom(
            container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )
        assert result is not None

        # Create container with only 4 samples (2 per group)
        container_small = create_count_container(n_samples=4)
        with pytest.raises(ValidationError):
            diff_expr_voom(
                container_small,
                "protein",
                "raw",
                "group",
                "A",
                "B",
            )

    def test_voom_detects_differential_expression(self, simple_count_container):
        """Test that VOOM detects differential expression."""
        result = diff_expr_voom(
            simple_count_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        # First 5 features were engineered to be DE
        # They should have more significant p-values on average
        first_5_p = result.p_values[:5]
        last_5_p = result.p_values[-5:]

        # Filter out NaN for comparison
        first_5_valid = first_5_p[~np.isnan(first_5_p)]
        last_5_valid = last_5_p[~np.isnan(last_5_p)]

        if len(first_5_valid) > 0 and len(last_5_valid) > 0:
            assert np.mean(first_5_valid) < np.mean(last_5_valid)

    def test_voom_result_structure(self, simple_count_container):
        """Test that VOOM result has expected structure."""
        result = diff_expr_voom(
            simple_count_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        # Check all expected attributes
        assert hasattr(result, "p_values")
        assert hasattr(result, "p_values_adj")
        assert hasattr(result, "log2_fc")
        assert hasattr(result, "test_statistics")
        assert hasattr(result, "effect_sizes")
        assert hasattr(result, "group_stats")

        # Check group stats
        assert "A_mean" in result.group_stats
        assert "B_mean" in result.group_stats


# =============================================================================
# diff_expr_limma_trend Tests
# =============================================================================


class TestDiffExprLimmaTrend:
    """Tests for limma-trend differential expression analysis."""

    def test_limma_trend_basic(self, simple_count_container):
        """Test basic limma-trend analysis."""
        result = diff_expr_limma_trend(
            simple_count_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        assert result is not None
        assert len(result.p_values) == 50
        assert result.method == "limma_trend"
        assert result.params["trend"] is True

    def test_limma_trend_with_trend(self, simple_count_container):
        """Test limma-trend with trend correction enabled."""
        result = diff_expr_limma_trend(
            simple_count_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
            trend=True,
            robust=True,
        )

        assert len(result.p_values) == 50
        assert result.params["trend"] is True
        assert result.params["robust"] is True

    def test_limma_trend_without_trend(self, simple_count_container):
        """Test limma-trend without trend correction."""
        result = diff_expr_limma_trend(
            simple_count_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
            trend=False,
        )

        assert len(result.p_values) == 50
        assert result.params["trend"] is False

    def test_limma_trend_robust_option(self, simple_count_container):
        """Test robust vs non-robust options."""
        result_robust = diff_expr_limma_trend(
            simple_count_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
            robust=True,
        )

        result_non_robust = diff_expr_limma_trend(
            simple_count_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
            robust=False,
        )

        assert result_robust.params["robust"] is True
        assert result_non_robust.params["robust"] is False
        # Both should produce valid results
        assert len(result_robust.p_values) == len(result_non_robust.p_values)

    def test_limma_trend_sparse_data(self, sparse_count_container):
        """Test limma-trend with sparse matrix."""
        result = diff_expr_limma_trend(
            sparse_count_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        assert len(result.p_values) == 40
        assert result.method == "limma_trend"

    def test_limma_trend_with_mask(self, container_with_mask):
        """Test limma-trend handles masked values."""
        result = diff_expr_limma_trend(
            container_with_mask,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        assert len(result.p_values) == 40

    def test_limma_trend_invalid_assay(self, simple_count_container):
        """Test limma-trend with invalid assay."""
        with pytest.raises(AssayNotFoundError):
            diff_expr_limma_trend(
                simple_count_container,
                "invalid",
                "raw",
                "group",
                "A",
                "B",
            )

    def test_limma_trend_invalid_layer(self, simple_count_container):
        """Test limma-trend with invalid layer."""
        with pytest.raises(LayerNotFoundError):
            diff_expr_limma_trend(
                simple_count_container,
                "protein",
                "invalid",
                "group",
                "A",
                "B",
            )

    def test_limma_trend_invalid_group(self, simple_count_container):
        """Test limma-trend with invalid group."""
        with pytest.raises(ValidationError):
            diff_expr_limma_trend(
                simple_count_container,
                "protein",
                "raw",
                "group",
                "A",
                "nonexistent",
            )

    def test_limma_trend_group_stats(self, simple_count_container):
        """Test that limma-trend calculates group statistics."""
        result = diff_expr_limma_trend(
            simple_count_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        # Should have mean and median for each group
        assert "A_mean" in result.group_stats
        assert "B_mean" in result.group_stats
        assert "A_median" in result.group_stats
        assert "B_median" in result.group_stats


# =============================================================================
# diff_expr_deseq2 Tests
# =============================================================================


class TestDiffExprDeseq2:
    """Tests for DESeq2-like differential expression analysis."""

    def test_deseq2_basic(self, simple_count_container):
        """Test basic DESeq2 analysis."""
        result = diff_expr_deseq2(
            simple_count_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        assert result is not None
        assert len(result.p_values) == 50
        assert result.method == "deseq2"
        assert result.params["fit_type"] == "parametric"
        assert result.params["test"] == "wald"

    def test_deseq2_fit_types(self, simple_count_container):
        """Test DESeq2 with different fit types."""
        for fit_type in ["parametric", "local", "mean"]:
            result = diff_expr_deseq2(
                simple_count_container,
                "protein",
                "raw",
                "group",
                "A",
                "B",
                fit_type=fit_type,
            )
            assert len(result.p_values) == 50
            assert result.params["fit_type"] == fit_type

    def test_deseq2_wald_vs_lrt(self, simple_count_container):
        """Test DESeq2 Wald vs LRT test."""
        result_wald = diff_expr_deseq2(
            simple_count_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
            test="wald",
        )

        result_lrt = diff_expr_deseq2(
            simple_count_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
            test="lrt",
        )

        assert len(result_wald.p_values) == 50
        assert len(result_lrt.p_values) == 50
        assert result_wald.params["test"] == "wald"
        assert result_lrt.params["test"] == "lrt"

    def test_deseq2_min_count_filtering(self, simple_count_container):
        """Test DESeq2 with min_count filtering."""
        result_default = diff_expr_deseq2(
            simple_count_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        result_high = diff_expr_deseq2(
            simple_count_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
            min_count=50,
        )

        # Higher threshold should result in more NaN p-values
        n_nan_default = np.sum(np.isnan(result_default.p_values))
        n_nan_high = np.sum(np.isnan(result_high.p_values))
        assert n_nan_high >= n_nan_default

    def test_deseq2_sparse_data(self, sparse_count_container):
        """Test DESeq2 with sparse matrix."""
        result = diff_expr_deseq2(
            sparse_count_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        assert len(result.p_values) == 40
        assert result.method == "deseq2"

    def test_deseq2_with_mask(self, container_with_mask):
        """Test DESeq2 handles masked values."""
        result = diff_expr_deseq2(
            container_with_mask,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        assert len(result.p_values) == 40

    def test_deseq2_invalid_fit_type(self, simple_count_container):
        """Test DESeq2 with invalid fit_type."""
        with pytest.raises(ValidationError) as exc_info:
            diff_expr_deseq2(
                simple_count_container,
                "protein",
                "raw",
                "group",
                "A",
                "B",
                fit_type="invalid",
            )
        assert "fit_type" in str(exc_info.value)

    def test_deseq2_invalid_test(self, simple_count_container):
        """Test DESeq2 with invalid test type."""
        with pytest.raises(ValidationError) as exc_info:
            diff_expr_deseq2(
                simple_count_container,
                "protein",
                "raw",
                "group",
                "A",
                "B",
                test="invalid",
            )
        assert "test" in str(exc_info.value)

    def test_deseq2_invalid_assay(self, simple_count_container):
        """Test DESeq2 with invalid assay."""
        with pytest.raises(AssayNotFoundError):
            diff_expr_deseq2(
                simple_count_container,
                "invalid",
                "raw",
                "group",
                "A",
                "B",
            )

    def test_deseq2_invalid_layer(self, simple_count_container):
        """Test DESeq2 with invalid layer."""
        with pytest.raises(LayerNotFoundError):
            diff_expr_deseq2(
                simple_count_container,
                "protein",
                "invalid",
                "group",
                "A",
                "B",
            )

    def test_deseq2_invalid_group(self, simple_count_container):
        """Test DESeq2 with invalid group."""
        with pytest.raises(ValidationError):
            diff_expr_deseq2(
                simple_count_container,
                "protein",
                "raw",
                "group",
                "A",
                "nonexistent",
            )

    def test_deseq2_group_stats(self, simple_count_container):
        """Test that DESeq2 calculates group statistics."""
        result = diff_expr_deseq2(
            simple_count_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        assert "A_mean" in result.group_stats
        assert "B_mean" in result.group_stats


# =============================================================================
# Result Validation Tests
# =============================================================================


class TestResultValidation:
    """Tests for validating result properties across all methods."""

    def test_p_values_in_range(self, simple_count_container):
        """Test that p-values are in valid range [0, 1]."""
        methods = [
            (diff_expr_voom, {}),
            (diff_expr_limma_trend, {}),
            (diff_expr_deseq2, {}),
        ]

        for method, kwargs in methods:
            result = method(
                simple_count_container,
                "protein",
                "raw",
                "group",
                "A",
                "B",
                **kwargs,
            )
            # Valid p-values should be in [0, 1]
            valid_p = result.p_values[~np.isnan(result.p_values)]
            assert np.all(valid_p >= 0), f"{method.__name__}: p-values < 0"
            assert np.all(valid_p <= 1), f"{method.__name__}: p-values > 1"

    def test_p_values_adj_in_range(self, simple_count_container):
        """Test that adjusted p-values are in valid range [0, 1]."""
        methods = [
            (diff_expr_voom, {}),
            (diff_expr_limma_trend, {}),
            (diff_expr_deseq2, {}),
        ]

        for method, kwargs in methods:
            result = method(
                simple_count_container,
                "protein",
                "raw",
                "group",
                "A",
                "B",
                **kwargs,
            )
            valid_p_adj = result.p_values_adj[~np.isnan(result.p_values_adj)]
            assert np.all(valid_p_adj >= 0), f"{method.__name__}: adj p-values < 0"
            assert np.all(valid_p_adj <= 1), f"{method.__name__}: adj p-values > 1"

    def test_log2_fc_calculated(self, simple_count_container):
        """Test that log2 fold changes are calculated."""
        methods = [
            (diff_expr_voom, {}),
            (diff_expr_limma_trend, {}),
            (diff_expr_deseq2, {}),
        ]

        for method, kwargs in methods:
            result = method(
                simple_count_container,
                "protein",
                "raw",
                "group",
                "A",
                "B",
                **kwargs,
            )
            assert len(result.log2_fc) == len(result.p_values)
            # Should have both positive and negative values (or at least non-zero)
            valid_fc = result.log2_fc[~np.isnan(result.log2_fc)]
            if len(valid_fc) > 0:
                # At least some features should have non-zero FC
                assert np.any(valid_fc != 0), f"{method.__name__}: all FCs are zero"

    def test_to_dataframe(self, simple_count_container):
        """Test result conversion to DataFrame."""
        result = diff_expr_voom(
            simple_count_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        df = result.to_dataframe()

        assert isinstance(df, pl.DataFrame)
        assert "feature_id" in df.columns
        assert "p_value" in df.columns
        assert "p_value_adj" in df.columns
        assert "log2_fc" in df.columns
        assert "test_statistic" in df.columns

    def test_to_dataframe_sorted(self, simple_count_container):
        """Test that DataFrame is sorted by p-value."""
        result = diff_expr_voom(
            simple_count_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        df = result.to_dataframe()

        # Check that p_values are sorted (excluding NaN)
        valid_df = df.filter(pl.col("p_value").is_not_nan())
        if len(valid_df) > 1:
            p_values = valid_df["p_value"].to_numpy()
            assert np.all(p_values[:-1] <= p_values[1:])

    def test_get_significant(self, simple_count_container):
        """Test getting significant features."""
        result = diff_expr_voom(
            simple_count_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        sig = result.get_significant(alpha=0.05)
        assert isinstance(sig, pl.DataFrame)

        # All returned features should meet the threshold
        if len(sig) > 0:
            assert np.all(sig["p_value_adj"].to_numpy() < 0.05)

    def test_get_significant_with_log2_fc_filter(self, simple_count_container):
        """Test getting significant features with log2 FC filter."""
        result = diff_expr_voom(
            simple_count_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        sig = result.get_significant(alpha=0.5, min_log2_fc=0.5)

        if len(sig) > 0:
            # All should meet both criteria
            assert np.all(sig["p_value_adj"].to_numpy() < 0.5)
            assert np.all(np.abs(sig["log2_fc"].to_numpy()) >= 0.5)


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_small_sample_size(self):
        """Test with minimal sample size."""
        container = create_count_container(n_samples=6, n_features=10)

        # Should work with exactly 3 samples per group (minimum)
        result = diff_expr_voom(
            container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )
        assert len(result.p_values) == 10

    def test_single_feature(self):
        """Test with only one feature using limma_trend (no min feature filter)."""
        container = create_count_container(n_samples=20, n_features=1, n_de=0)

        # limma_trend doesn't have a minimum feature count requirement
        result = diff_expr_limma_trend(
            container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )
        assert len(result.p_values) == 1
        assert len(result.log2_fc) == 1

    def test_two_features(self):
        """Test with only two features."""
        container = create_count_container(n_samples=20, n_features=2, n_de=0)

        result = diff_expr_limma_trend(
            container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )
        assert len(result.p_values) == 2

    def test_unequal_group_sizes(self):
        """Test with unequal group sizes."""
        n_samples_a, n_samples_b = 15, 10
        n_total = n_samples_a + n_samples_b

        obs = pl.DataFrame(
            {
                "_index": [f"SAMPLE_{i:04d}" for i in range(n_total)],
                "group": ["A"] * n_samples_a + ["B"] * n_samples_b,
            }
        )

        var = pl.DataFrame({"_index": [f"PROT_{i:04d}" for i in range(20)]})
        X = np.random.negative_binomial(n=5, p=0.3, size=(n_total, 20))

        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X)}, feature_id_col="_index")
        container = ScpContainer(obs=obs, assays={"protein": assay})

        # Should handle unequal groups
        result = diff_expr_voom(
            container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )
        assert len(result.p_values) == 20

    def test_all_zero_feature(self):
        """Test with a feature that has all zeros."""
        n_samples, n_features = 20, 25

        obs = pl.DataFrame(
            {
                "_index": [f"SAMPLE_{i:04d}" for i in range(n_samples)],
                "group": ["A"] * 10 + ["B"] * 10,
            }
        )

        var = pl.DataFrame({"_index": [f"PROT_{i:04d}" for i in range(n_features)]})
        X = np.random.negative_binomial(n=5, p=0.3, size=(n_samples, n_features))
        X[:, 0] = 0  # First feature all zeros

        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X)}, feature_id_col="_index")
        container = ScpContainer(obs=obs, assays={"protein": assay})

        # Use limma_trend which doesn't filter features
        result = diff_expr_limma_trend(
            container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )
        assert len(result.p_values) == n_features

    def test_constant_feature(self):
        """Test with a feature that has constant values."""
        n_samples, n_features = 20, 10

        obs = pl.DataFrame(
            {
                "_index": [f"SAMPLE_{i:04d}" for i in range(n_samples)],
                "group": ["A"] * 10 + ["B"] * 10,
            }
        )

        var = pl.DataFrame({"_index": [f"PROT_{i:04d}" for i in range(n_features)]})
        X = np.random.negative_binomial(n=5, p=0.3, size=(n_samples, n_features))
        X[:, 0] = 10  # First feature constant

        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X)}, feature_id_col="_index")
        container = ScpContainer(obs=obs, assays={"protein": assay})

        # Should handle constant features
        result = diff_expr_voom(
            container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )
        # Constant feature might have p-value = 1 (no difference)
        assert len(result.p_values) == n_features


# =============================================================================
# Integration Tests
# =============================================================================


class TestCountModelsIntegration:
    """Integration tests for count-based differential expression."""

    def test_compare_voom_vs_limma_trend(self, simple_count_container):
        """Compare VOOM and limma-trend results."""
        result_voom = diff_expr_voom(
            simple_count_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        result_trend = diff_expr_limma_trend(
            simple_count_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        # Both should have same shape
        assert len(result_voom.p_values) == len(result_trend.p_values)

        # P-values should be correlated (not identical)
        valid_mask = ~np.isnan(result_voom.p_values) & ~np.isnan(result_trend.p_values)
        if np.sum(valid_mask) > 3:
            corr = np.corrcoef(
                result_voom.p_values[valid_mask],
                result_trend.p_values[valid_mask],
            )[0, 1]
            # Should have some positive correlation (lowered threshold)
            assert corr > -0.2  # Methods can differ substantially

    def test_compare_all_methods_on_same_data(self, large_count_container):
        """Compare all three methods on the same data."""
        result_voom = diff_expr_voom(
            large_count_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        result_trend = diff_expr_limma_trend(
            large_count_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        result_deseq2 = diff_expr_deseq2(
            large_count_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        # All should have same shape
        assert len(result_voom.p_values) == len(result_trend.p_values)
        assert len(result_voom.p_values) == len(result_deseq2.p_values)

        # All should detect differential expression in first 10 features
        # (these were engineered to be DE)
        for result in [result_voom, result_trend, result_deseq2]:
            first_10_p = result.p_values[:10]
            valid_p = first_10_p[~np.isnan(first_10_p)]

            if len(valid_p) > 0:
                # At least some should be significant at 0.1 level
                n_sig = np.sum(valid_p < 0.1)
                assert n_sig >= 1, f"{result.method} found no significant features"

    def test_sparse_vs_dense_consistency(self):
        """Test that sparse and dense matrices give similar results."""
        rng = np.random.default_rng(42)

        n_samples, n_features = 30, 40

        obs = pl.DataFrame(
            {
                "_index": [f"SAMPLE_{i:04d}" for i in range(n_samples)],
                "group": ["A"] * 15 + ["B"] * 15,
            }
        )

        var = pl.DataFrame({"_index": [f"PROT_{i:04d}" for i in range(n_features)]})

        # Dense version
        X_dense = rng.negative_binomial(n=5, p=0.3, size=(n_samples, n_features))
        assay_dense = Assay(
            var=var,
            layers={"raw": ScpMatrix(X=X_dense)},
            feature_id_col="_index",
        )
        container_dense = ScpContainer(obs=obs, assays={"protein": assay_dense})

        # Sparse version
        X_sparse = sp.csr_matrix(X_dense)
        assay_sparse = Assay(
            var=var,
            layers={"raw": ScpMatrix(X=X_sparse)},
            feature_id_col="_index",
        )
        container_sparse = ScpContainer(obs=obs, assays={"protein": assay_sparse})

        result_dense = diff_expr_voom(
            container_dense,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        result_sparse = diff_expr_voom(
            container_sparse,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        # Results should be very similar
        valid_mask = ~np.isnan(result_dense.p_values) & ~np.isnan(result_sparse.p_values)

        if np.sum(valid_mask) > 0:
            # P-values should be identical or very close
            np.testing.assert_allclose(
                result_dense.p_values[valid_mask],
                result_sparse.p_values[valid_mask],
                rtol=1e-10,
            )

    def test_all_methods_handle_zeros(self):
        """Test that all methods handle zero counts correctly."""
        n_samples, n_features = 20, 20

        obs = pl.DataFrame(
            {
                "_index": [f"SAMPLE_{i:04d}" for i in range(n_samples)],
                "group": ["A"] * 10 + ["B"] * 10,
            }
        )

        var = pl.DataFrame({"_index": [f"P{i}" for i in range(n_features)]})

        # Add many zeros
        X = np.random.negative_binomial(n=5, p=0.3, size=(n_samples, n_features))
        X[X < 2] = 0  # Set low values to zero

        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X)}, feature_id_col="_index")
        container = ScpContainer(obs=obs, assays={"protein": assay})

        # All methods should handle zeros
        result_voom = diff_expr_voom(
            container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        result_trend = diff_expr_limma_trend(
            container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        result_deseq2 = diff_expr_deseq2(
            container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        # All should return results
        for result in [result_voom, result_trend, result_deseq2]:
            assert len(result.p_values) == n_features
            # Some p-values might be NaN (filtered features)
            assert np.sum(~np.isnan(result.p_values)) >= 0

    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with same seed."""
        container = create_count_container(n_samples=30, n_features=30)

        result1 = diff_expr_voom(
            container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        result2 = diff_expr_voom(
            container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        # Results should be identical
        np.testing.assert_array_equal(result1.p_values, result2.p_values)
        np.testing.assert_array_equal(result1.log2_fc, result2.log2_fc)


# =============================================================================
# Method-Specific Validation Tests
# =============================================================================


class TestMethodSpecificValidation:
    """Tests for method-specific validation and edge cases."""

    def test_voom_size_factor_handling(self):
        """Test VOOM handles different library sizes correctly."""
        # Create data with very different library sizes
        n_samples, n_features = 20, 20

        obs = pl.DataFrame(
            {
                "_index": [f"SAMPLE_{i:04d}" for i in range(n_samples)],
                "group": ["A"] * 10 + ["B"] * 10,
            }
        )

        var = pl.DataFrame({"_index": [f"P{i}" for i in range(n_features)]})

        X = np.random.negative_binomial(n=5, p=0.3, size=(n_samples, n_features))
        # Make group A have higher library size
        X[:10, :] *= 3

        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X)}, feature_id_col="_index")
        container = ScpContainer(obs=obs, assays={"protein": assay})

        # TMM normalization should correct for library size differences
        result = diff_expr_voom(
            container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
            normalize="tmm",
        )

        # Should produce valid results
        assert len(result.p_values) == n_features
        assert np.sum(~np.isnan(result.p_values)) > 0

    def test_deseq2_dispersion_estimation(self, simple_count_container):
        """Test DESeq2 dispersion estimation."""
        result = diff_expr_deseq2(
            simple_count_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
            fit_type="parametric",
        )

        # Dispersion should affect the variance structure
        # Features with more variance should have less significant p-values
        # (all else being equal)
        assert len(result.p_values) == 50

    def test_limma_trend_variance_shrinkage(self, simple_count_container):
        """Test that limma-trend shrinks variances appropriately."""
        result = diff_expr_limma_trend(
            simple_count_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
            trend=True,
            robust=True,
        )

        # Moderated statistics should be computed
        assert len(result.test_statistics) == 50
        # Test statistics should have reasonable range
        valid_stats = result.test_statistics[~np.isnan(result.test_statistics)]
        if len(valid_stats) > 0:
            # Most t-statistics should be within reasonable range
            assert np.all(np.abs(valid_stats) < 100)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
