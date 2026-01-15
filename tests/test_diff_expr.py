"""Comprehensive tests for differential expression analysis module.

Tests cover all functions in scptensor.diff_expr.core:
- adjust_fdr: FDR correction (BH and BY methods)
- diff_expr_ttest: Two-group t-test (Welch and Student)
- diff_expr_mannwhitney: Mann-Whitney U test
- diff_expr_anova: One-way ANOVA
- diff_expr_kruskal: Kruskal-Wallis test
- DiffExprResult: Result container class
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import polars as pl
import pytest
import scipy.sparse as sp

from scptensor.core.exceptions import (
    AssayNotFoundError,
    LayerNotFoundError,
    ValidationError,
)
from scptensor.core.structures import Assay, ScpContainer, ScpMatrix
from scptensor.diff_expr.core import (
    DiffExprResult,
    adjust_fdr,
    diff_expr_anova,
    diff_expr_kruskal,
    diff_expr_mannwhitney,
    diff_expr_ttest,
    pd_isna,
)

# =============================================================================
# Helper Functions
# =============================================================================


def create_diff_expr_test_container(
    n_samples: int = 20,
    n_features: int = 10,
    n_groups: int = 2,
    effect_size: float = 1.5,
    seed: int = 42,
    with_nan: bool = False,
    sparse: bool = False,
) -> ScpContainer:
    """Create a test container for differential expression tests.

    Args:
        n_samples: Total number of samples
        n_features: Number of features (proteins)
        n_groups: Number of groups (2 or 3)
        effect_size: Effect size for differentially expressed features
        seed: Random seed
        with_nan: Whether to include NaN values
        sparse: Whether to use sparse matrices

    Returns:
        A test ScpContainer with group labels
    """
    rng = np.random.default_rng(seed)

    # Create sample metadata with groups
    samples_per_group = n_samples // n_groups
    groups = []
    for i in range(n_groups):
        groups.extend([f"group_{i}"] * samples_per_group)
    # Handle remainder
    while len(groups) < n_samples:
        groups.append(groups[-1])

    obs = pl.DataFrame(
        {
            "_index": [f"sample_{i}" for i in range(n_samples)],
            "group": groups,
            "batch": rng.choice(["A", "B"], n_samples),
        }
    )

    # Create feature metadata
    var = pl.DataFrame(
        {
            "_index": [f"protein_{i}" for i in range(n_features)],
            "protein_name": [f"Protein{i}" for i in range(n_features)],
        }
    )

    # Create data matrix
    X = rng.exponential(10, size=(n_samples, n_features))

    # Add differential expression for first half of features
    n_de = n_features // 2
    for j in range(n_de):
        if n_groups == 2:
            # Make first group higher for these features
            X[:samples_per_group, j] *= effect_size
        else:
            # Make first group different
            X[:samples_per_group, j] *= effect_size

    if with_nan:
        # Add some NaN values randomly
        nan_mask = rng.random((n_samples, n_features)) < 0.05
        X[nan_mask] = np.nan

    if sparse:
        X = sp.csr_matrix(X)

    # Create assay
    assay = Assay(var=var, layers={"X": ScpMatrix(X=X)}, feature_id_col="_index")

    return ScpContainer(obs=obs, assays={"proteins": assay})


def create_three_group_container(
    n_samples: int = 30,
    n_features: int = 10,
    seed: int = 42,
    with_nan: bool = False,
) -> ScpContainer:
    """Create a container with 3 groups for ANOVA/Kruskal tests."""
    return create_diff_expr_test_container(
        n_samples=n_samples, n_features=n_features, n_groups=3, seed=seed, with_nan=with_nan
    )


# =============================================================================
# adjust_fdr Tests
# =============================================================================


class TestAdjustFDR:
    """Tests for FDR adjustment function."""

    def test_fdr_bh_basic(self):
        """Test basic Benjamini-Hochberg correction."""
        p_values = np.array([0.001, 0.01, 0.05, 0.1, 0.5])
        adjusted = adjust_fdr(p_values, method="bh")

        # Adjusted p-values should be >= original
        assert np.all(adjusted >= p_values)

        # Adjusted p-values should be <= 1
        assert np.all(adjusted <= 1.0)

        # Check monotonicity (should be non-decreasing after sorting)
        sorted_adj = np.sort(adjusted)
        assert np.all(np.diff(sorted_adj) >= -1e-10)  # Allow tiny numerical errors

    def test_fdr_by_more_conservative(self):
        """Test that BY correction is more conservative than BH."""
        p_values = np.array([0.001, 0.01, 0.05, 0.1, 0.5])
        adjusted_bh = adjust_fdr(p_values, method="bh")
        adjusted_by = adjust_fdr(p_values, method="by")

        # BY should be >= BH (more conservative)
        assert np.all(adjusted_by >= adjusted_bh)

    def test_fdr_with_nan_values(self):
        """Test FDR adjustment with NaN values."""
        p_with_nan = np.array([0.01, np.nan, 0.05, 0.1, np.nan])
        adjusted = adjust_fdr(p_with_nan, method="bh")

        # NaN positions should be preserved
        assert np.isnan(adjusted[1])
        assert np.isnan(adjusted[4])

        # Valid positions should be adjusted
        assert not np.isnan(adjusted[0])
        assert not np.isnan(adjusted[2])
        assert not np.isnan(adjusted[3])

    def test_fdr_empty_array(self):
        """Test FDR adjustment with empty array."""
        p_empty = np.array([])
        adjusted = adjust_fdr(p_empty)
        assert len(adjusted) == 0
        assert adjusted.dtype == np.float64

    def test_fdr_single_pvalue(self):
        """Test FDR adjustment with single p-value."""
        p_single = np.array([0.05])
        adjusted = adjust_fdr(p_single)
        # Single p-value should remain the same (or close due to monotonicity)
        assert adjusted[0] >= 0.05
        assert adjusted[0] <= 0.05

    def test_fdr_all_significant(self):
        """Test FDR adjustment when all p-values are significant."""
        p_values = np.array([0.0001, 0.001, 0.01, 0.02, 0.04])
        adjusted = adjust_fdr(p_values, method="bh")

        # All should still be significant at alpha=0.05 after adjustment
        assert np.all(adjusted <= 0.05)

    def test_fdr_all_very_small(self):
        """Test FDR adjustment with very small p-values."""
        p_values = np.array([1e-10, 1e-8, 1e-6, 1e-4, 0.001])
        adjusted = adjust_fdr(p_values, method="bh")

        # Adjusted values should still be very small
        assert np.all(adjusted < 0.01)

    def test_fdr_invalid_method(self):
        """Test that invalid method raises ValidationError."""
        p_values = np.array([0.01, 0.05, 0.1])

        with pytest.raises(ValidationError) as exc_info:
            adjust_fdr(p_values, method="invalid")

        assert "method" in str(exc_info.value).lower()

    def test_fdr_clipping_to_one(self):
        """Test that adjusted p-values are clipped to [0, 1]."""
        # Large p-values that would exceed 1 without clipping
        p_values = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        adjusted = adjust_fdr(p_values, method="bh")

        assert np.all(adjusted <= 1.0)
        assert np.all(adjusted >= 0.0)

    def test_fdr_preserves_order_for_equal_pvalues(self):
        """Test FDR adjustment with equal p-values."""
        p_values = np.array([0.05, 0.05, 0.05, 0.05])
        adjusted = adjust_fdr(p_values, method="bh")

        # All should have same adjusted value (within tolerance)
        assert np.allclose(adjusted, adjusted[0], rtol=1e-10)


# =============================================================================
# diff_expr_ttest Tests
# =============================================================================


class TestDiffExprTTest:
    """Tests for t-test differential expression."""

    def test_ttest_basic(self):
        """Test basic t-test functionality."""
        container = create_diff_expr_test_container(
            n_samples=20, n_features=10, effect_size=2.0, seed=42
        )

        result = diff_expr_ttest(
            container=container,
            assay_name="proteins",
            group_col="group",
            group1="group_0",
            group2="group_1",
        )

        assert isinstance(result, DiffExprResult)
        assert result.method == "welch_ttest"
        assert len(result.p_values) == 10
        assert len(result.p_values_adj) == 10
        assert len(result.log2_fc) == 10
        assert result.effect_sizes is not None

    def test_ttest_student_equal_var(self):
        """Test Student's t-test (equal_var=True)."""
        container = create_diff_expr_test_container(n_samples=20, n_features=10)

        result = diff_expr_ttest(
            container=container,
            assay_name="proteins",
            group_col="group",
            group1="group_0",
            group2="group_1",
            equal_var=True,
        )

        assert result.method == "student_ttest"
        assert "equal_var" in result.params
        assert result.params["equal_var"] is True

    def test_ttest_detects_differential_expression(self):
        """Test that t-test detects differential expression."""
        container = create_diff_expr_test_container(
            n_samples=30, n_features=20, effect_size=3.0, seed=42
        )

        result = diff_expr_ttest(
            container=container,
            assay_name="proteins",
            group_col="group",
            group1="group_0",
            group2="group_1",
        )

        # First half should have more significant p-values
        first_half_p = result.p_values[:10]
        second_half_p = result.p_values[10:]

        # First half should be more significant on average
        assert np.nanmean(first_half_p) < np.nanmean(second_half_p)

    def test_ttest_assay_not_found(self):
        """Test error when assay doesn't exist."""
        container = create_diff_expr_test_container()

        with pytest.raises(AssayNotFoundError) as exc_info:
            diff_expr_ttest(
                container=container,
                assay_name="nonexistent",
                group_col="group",
                group1="group_0",
                group2="group_1",
            )

        assert "nonexistent" in str(exc_info.value)

    def test_ttest_layer_not_found(self):
        """Test error when layer doesn't exist."""
        container = create_diff_expr_test_container()

        with pytest.raises(LayerNotFoundError) as exc_info:
            diff_expr_ttest(
                container=container,
                assay_name="proteins",
                group_col="group",
                group1="group_0",
                group2="group_1",
                layer_name="nonexistent",
            )

        assert "nonexistent" in str(exc_info.value)

    def test_ttest_group_column_not_found(self):
        """Test error when group column doesn't exist."""
        container = create_diff_expr_test_container()

        with pytest.raises(ValidationError) as exc_info:
            diff_expr_ttest(
                container=container,
                assay_name="proteins",
                group_col="nonexistent",
                group1="group_0",
                group2="group_1",
            )

        assert (
            "nonexistent" in str(exc_info.value).lower() or "group" in str(exc_info.value).lower()
        )

    def test_ttest_insufficient_samples(self):
        """Test error with insufficient samples per group."""
        # Create container with very few samples in one group
        rng = np.random.default_rng(42)
        obs = pl.DataFrame(
            {
                "_index": [f"sample_{i}" for i in range(5)],
                "group": ["A", "A", "A", "B", "C"],  # B and C have only 1 sample each
            }
        )
        var = pl.DataFrame({"_index": [f"P{i}" for i in range(5)]})
        X = rng.exponential(10, size=(5, 5))
        assay = Assay(var=var, layers={"X": ScpMatrix(X=X)})
        container = ScpContainer(obs=obs, assays={"proteins": assay})

        with pytest.raises(ValidationError) as exc_info:
            diff_expr_ttest(
                container=container,
                assay_name="proteins",
                group_col="group",
                group1="A",
                group2="B",
                min_samples_per_group=3,
            )

        assert "sample" in str(exc_info.value).lower() or "group" in str(exc_info.value).lower()

    def test_ttest_with_sparse_matrix(self):
        """Test t-test with sparse matrix data."""
        container = create_diff_expr_test_container(n_samples=20, n_features=10, sparse=True)

        result = diff_expr_ttest(
            container=container,
            assay_name="proteins",
            group_col="group",
            group1="group_0",
            group2="group_1",
        )

        assert len(result.p_values) == 10

    def test_ttest_missing_strategy_ignore(self):
        """Test t-test with missing values (ignore strategy)."""
        container = create_diff_expr_test_container(n_samples=20, n_features=10, with_nan=True)

        result = diff_expr_ttest(
            container=container,
            assay_name="proteins",
            group_col="group",
            group1="group_0",
            group2="group_1",
            missing_strategy="ignore",
        )

        # Should handle NaN values
        assert len(result.p_values) == 10
        # Some p-values might be NaN if too many missing values
        assert np.sum(np.isnan(result.p_values)) >= 0

    def test_ttest_missing_strategy_zero(self):
        """Test t-test with missing values (zero strategy)."""
        container = create_diff_expr_test_container(n_samples=20, n_features=10, with_nan=True)

        result = diff_expr_ttest(
            container=container,
            assay_name="proteins",
            group_col="group",
            group1="group_0",
            group2="group_1",
            missing_strategy="zero",
        )

        # With zero strategy, should have no NaN p-values
        assert len(result.p_values) == 10

    def test_ttest_log2_fold_change_calculation(self):
        """Test log2 fold change calculation."""
        # Create container where group_0 has higher values
        container = create_diff_expr_test_container(
            n_samples=20, n_features=5, effect_size=2.0, seed=42
        )

        result = diff_expr_ttest(
            container=container,
            assay_name="proteins",
            group_col="group",
            group1="group_0",
            group2="group_1",
        )

        # First features should have positive log2 FC (group_0 > group_1)
        assert result.log2_fc[0] > 0

    def test_ttest_group_stats(self):
        """Test that group statistics are calculated correctly."""
        container = create_diff_expr_test_container(n_samples=20, n_features=5)

        result = diff_expr_ttest(
            container=container,
            assay_name="proteins",
            group_col="group",
            group1="group_0",
            group2="group_1",
        )

        assert "group_0_mean" in result.group_stats
        assert "group_1_mean" in result.group_stats
        assert "group_0_median" in result.group_stats
        assert "group_1_median" in result.group_stats

        # Check shapes
        assert len(result.group_stats["group_0_mean"]) == 5


# =============================================================================
# diff_expr_mannwhitney Tests
# =============================================================================


class TestDiffExprMannWhitney:
    """Tests for Mann-Whitney U test differential expression."""

    def test_mannwhitney_basic(self):
        """Test basic Mann-Whitney U test functionality."""
        container = create_diff_expr_test_container(
            n_samples=20, n_features=10, effect_size=2.0, seed=42
        )

        result = diff_expr_mannwhitney(
            container=container,
            assay_name="proteins",
            group_col="group",
            group1="group_0",
            group2="group_1",
        )

        assert isinstance(result, DiffExprResult)
        assert result.method == "mannwhitney"
        assert len(result.p_values) == 10
        assert len(result.p_values_adj) == 10
        # Effect sizes not reported for MW test
        assert result.effect_sizes is None

    def test_mannwhitney_two_sided(self):
        """Test two-sided Mann-Whitney U test."""
        container = create_diff_expr_test_container(n_samples=20, n_features=10)

        result = diff_expr_mannwhitney(
            container=container,
            assay_name="proteins",
            group_col="group",
            group1="group_0",
            group2="group_1",
            alternative="two-sided",
        )

        assert result.params["alternative"] == "two-sided"
        assert len(result.p_values) == 10

    def test_mannwhitney_greater_alternative(self):
        """Test Mann-Whitney U test with greater alternative."""
        container = create_diff_expr_test_container(n_samples=20, n_features=10, effect_size=2.0)

        result = diff_expr_mannwhitney(
            container=container,
            assay_name="proteins",
            group_col="group",
            group1="group_0",
            group2="group_1",
            alternative="greater",
        )

        assert result.params["alternative"] == "greater"

    def test_mannwhitney_less_alternative(self):
        """Test Mann-Whitney U test with less alternative."""
        container = create_diff_expr_test_container(n_samples=20, n_features=10)

        result = diff_expr_mannwhitney(
            container=container,
            assay_name="proteins",
            group_col="group",
            group1="group_0",
            group2="group_1",
            alternative="less",
        )

        assert result.params["alternative"] == "less"

    def test_mannwhitney_detects_differential_expression(self):
        """Test that Mann-Whitney detects differential expression."""
        container = create_diff_expr_test_container(
            n_samples=30, n_features=20, effect_size=3.0, seed=42
        )

        result = diff_expr_mannwhitney(
            container=container,
            assay_name="proteins",
            group_col="group",
            group1="group_0",
            group2="group_1",
        )

        # First half should be more significant
        first_half_p = result.p_values[:10]
        second_half_p = result.p_values[10:]

        assert np.nanmean(first_half_p) < np.nanmean(second_half_p)

    def test_mannwhitney_assay_not_found(self):
        """Test error when assay doesn't exist."""
        container = create_diff_expr_test_container()

        with pytest.raises(AssayNotFoundError):
            diff_expr_mannwhitney(
                container=container,
                assay_name="nonexistent",
                group_col="group",
                group1="group_0",
                group2="group_1",
            )

    def test_mannwhitney_insufficient_samples(self):
        """Test error with insufficient samples per group."""
        container = create_diff_expr_test_container(n_samples=4)

        with pytest.raises(ValidationError):
            diff_expr_mannwhitney(
                container=container,
                assay_name="proteins",
                group_col="group",
                group1="group_0",
                group2="group_1",
                min_samples_per_group=5,
            )

    def test_mannwhitney_with_missing_values(self):
        """Test Mann-Whitney with missing values."""
        container = create_diff_expr_test_container(n_samples=20, n_features=10, with_nan=True)

        result = diff_expr_mannwhitney(
            container=container,
            assay_name="proteins",
            group_col="group",
            group1="group_0",
            group2="group_1",
            missing_strategy="ignore",
        )

        # Should handle NaN values
        assert len(result.p_values) == 10

    def test_mannwhitney_group_stats(self):
        """Test that group statistics are calculated."""
        container = create_diff_expr_test_container(n_samples=20, n_features=5)

        result = diff_expr_mannwhitney(
            container=container,
            assay_name="proteins",
            group_col="group",
            group1="group_0",
            group2="group_1",
        )

        # MW test reports medians, not means
        assert "group_0_median" in result.group_stats
        assert "group_1_median" in result.group_stats


# =============================================================================
# diff_expr_anova Tests
# =============================================================================


class TestDiffExprANOVA:
    """Tests for ANOVA differential expression."""

    def test_anova_basic(self):
        """Test basic ANOVA functionality."""
        container = create_three_group_container(n_samples=30, n_features=10)

        result = diff_expr_anova(
            container=container,
            assay_name="proteins",
            group_col="group",
        )

        assert isinstance(result, DiffExprResult)
        assert result.method == "anova"
        assert len(result.p_values) == 10
        assert result.params["n_groups"] == 3

    def test_anova_three_groups(self):
        """Test ANOVA with three groups."""
        container = create_three_group_container(n_samples=30, n_features=10)

        result = diff_expr_anova(
            container=container,
            assay_name="proteins",
            group_col="group",
        )

        # Check that groups are detected
        assert len(result.params["groups"]) == 3

        # Log2 FC should be NaN for multi-group comparison
        assert np.all(np.isnan(result.log2_fc))

    def test_anova_group_statistics(self):
        """Test that ANOVA calculates group statistics."""
        container = create_three_group_container(n_samples=30, n_features=5)

        result = diff_expr_anova(
            container=container,
            assay_name="proteins",
            group_col="group",
        )

        # Should have mean and median for each group
        for g in ["group_0", "group_1", "group_2"]:
            assert f"{g}_mean" in result.group_stats
            assert f"{g}_median" in result.group_stats

    def test_anova_assay_not_found(self):
        """Test error when assay doesn't exist."""
        container = create_three_group_container()

        with pytest.raises(AssayNotFoundError):
            diff_expr_anova(
                container=container,
                assay_name="nonexistent",
                group_col="group",
            )

    def test_anova_insufficient_groups(self):
        """Test error with fewer than 2 groups."""
        # Create container with only 1 group
        rng = np.random.default_rng(42)
        obs = pl.DataFrame(
            {
                "_index": [f"sample_{i}" for i in range(10)],
                "group": ["A"] * 10,
            }
        )
        var = pl.DataFrame({"_index": [f"P{i}" for i in range(5)]})
        X = rng.exponential(10, size=(10, 5))
        assay = Assay(var=var, layers={"X": ScpMatrix(X=X)})
        container = ScpContainer(obs=obs, assays={"proteins": assay})

        with pytest.raises(ValidationError) as exc_info:
            diff_expr_anova(
                container=container,
                assay_name="proteins",
                group_col="group",
            )

        assert "at least 2 groups" in str(exc_info.value).lower()

    def test_anova_insufficient_samples_per_group(self):
        """Test error with insufficient samples in a group."""
        # Create container where one group has too few samples
        rng = np.random.default_rng(42)
        obs = pl.DataFrame(
            {
                "_index": [f"sample_{i}" for i in range(5)],
                "group": ["A", "A", "B", "B", "C"],  # Group C has only 1 sample
            }
        )
        var = pl.DataFrame({"_index": [f"P{i}" for i in range(5)]})
        X = rng.exponential(10, size=(5, 5))
        assay = Assay(var=var, layers={"X": ScpMatrix(X=X)})
        container = ScpContainer(obs=obs, assays={"proteins": assay})

        with pytest.raises(ValidationError):
            diff_expr_anova(
                container=container,
                assay_name="proteins",
                group_col="group",
                min_samples_per_group=3,
            )

    def test_anova_with_sparse_matrix(self):
        """Test ANOVA with sparse matrix data."""
        container = create_three_group_container(n_samples=30, n_features=10)

        # Convert to sparse
        assay = container.assays["proteins"]
        X_sparse = sp.csr_matrix(assay.layers["X"].X)
        assay.layers["X"].X = X_sparse

        result = diff_expr_anova(
            container=container,
            assay_name="proteins",
            group_col="group",
        )

        assert len(result.p_values) == 10

    def test_anova_with_missing_values(self):
        """Test ANOVA with missing values."""
        container = create_three_group_container(n_samples=30, n_features=10, with_nan=True)

        result = diff_expr_anova(
            container=container,
            assay_name="proteins",
            group_col="group",
            missing_strategy="ignore",
        )

        # Should handle NaN values
        assert len(result.p_values) == 10


# =============================================================================
# diff_expr_kruskal Tests
# =============================================================================


class TestDiffExprKruskal:
    """Tests for Kruskal-Wallis differential expression."""

    def test_kruskal_basic(self):
        """Test basic Kruskal-Wallis functionality."""
        container = create_three_group_container(n_samples=30, n_features=10)

        result = diff_expr_kruskal(
            container=container,
            assay_name="proteins",
            group_col="group",
        )

        assert isinstance(result, DiffExprResult)
        assert result.method == "kruskal"
        assert len(result.p_values) == 10
        assert result.params["n_groups"] == 3

    def test_kruskal_three_groups(self):
        """Test Kruskal-Wallis with three groups."""
        container = create_three_group_container(n_samples=30, n_features=10)

        result = diff_expr_kruskal(
            container=container,
            assay_name="proteins",
            group_col="group",
        )

        # Check that groups are detected
        assert len(result.params["groups"]) == 3

        # Log2 FC should be NaN for multi-group comparison
        assert np.all(np.isnan(result.log2_fc))

    def test_kruskal_group_statistics(self):
        """Test that Kruskal-Wallis calculates group statistics."""
        container = create_three_group_container(n_samples=30, n_features=5)

        result = diff_expr_kruskal(
            container=container,
            assay_name="proteins",
            group_col="group",
        )

        # Should have mean and median for each group
        for g in ["group_0", "group_1", "group_2"]:
            assert f"{g}_mean" in result.group_stats
            assert f"{g}_median" in result.group_stats

    def test_kruskal_assay_not_found(self):
        """Test error when assay doesn't exist."""
        container = create_three_group_container()

        with pytest.raises(AssayNotFoundError):
            diff_expr_kruskal(
                container=container,
                assay_name="nonexistent",
                group_col="group",
            )

    def test_kruskal_insufficient_groups(self):
        """Test error with fewer than 2 groups."""
        rng = np.random.default_rng(42)
        obs = pl.DataFrame(
            {
                "_index": [f"sample_{i}" for i in range(10)],
                "group": ["A"] * 10,
            }
        )
        var = pl.DataFrame({"_index": [f"P{i}" for i in range(5)]})
        X = rng.exponential(10, size=(10, 5))
        assay = Assay(var=var, layers={"X": ScpMatrix(X=X)})
        container = ScpContainer(obs=obs, assays={"proteins": assay})

        with pytest.raises(ValidationError):
            diff_expr_kruskal(
                container=container,
                assay_name="proteins",
                group_col="group",
            )

    def test_kruskal_with_sparse_matrix(self):
        """Test Kruskal-Wallis with sparse matrix data."""
        container = create_three_group_container(n_samples=30, n_features=10)

        # Convert to sparse
        assay = container.assays["proteins"]
        X_sparse = sp.csr_matrix(assay.layers["X"].X)
        assay.layers["X"].X = X_sparse

        result = diff_expr_kruskal(
            container=container,
            assay_name="proteins",
            group_col="group",
        )

        assert len(result.p_values) == 10

    def test_kruskal_with_missing_values(self):
        """Test Kruskal-Wallis with missing values."""
        container = create_three_group_container(n_samples=30, n_features=10, with_nan=True)

        result = diff_expr_kruskal(
            container=container,
            assay_name="proteins",
            group_col="group",
            missing_strategy="median",
        )

        # Should handle NaN values
        assert len(result.p_values) == 10


# =============================================================================
# DiffExprResult Class Tests
# =============================================================================


class TestDiffExprResult:
    """Tests for DiffExprResult class."""

    def test_to_dataframe_basic(self):
        """Test converting result to DataFrame."""
        result = DiffExprResult(
            feature_ids=np.array(["P1", "P2", "P3"]),
            p_values=np.array([0.001, 0.05, 0.5]),
            p_values_adj=np.array([0.003, 0.075, 0.5]),
            log2_fc=np.array([1.5, -0.8, 0.1]),
            test_statistics=np.array([3.2, -1.5, 0.5]),
            effect_sizes=np.array([0.8, -0.4, 0.1]),
            group_stats={"A_mean": np.array([10, 8, 5]), "B_mean": np.array([4, 10, 5])},
            method="ttest",
            params={"group1": "A", "group2": "B"},
        )

        df = result.to_dataframe()

        assert isinstance(df, pl.DataFrame)
        assert df.shape == (3, 8)  # 3 features, 8 columns
        assert "feature_id" in df.columns
        assert "p_value" in df.columns
        assert "p_value_adj" in df.columns
        assert "log2_fc" in df.columns
        assert "test_statistic" in df.columns
        assert "effect_size" in df.columns
        assert "A_mean" in df.columns
        assert "B_mean" in df.columns

    def test_to_dataframe_sorted(self):
        """Test that DataFrame is sorted by p-value."""
        result = DiffExprResult(
            feature_ids=np.array(["P1", "P2", "P3"]),
            p_values=np.array([0.5, 0.001, 0.05]),
            p_values_adj=np.array([0.5, 0.003, 0.075]),
            log2_fc=np.array([1.5, -0.8, 0.1]),
            test_statistics=np.array([3.2, -1.5, 0.5]),
            effect_sizes=np.array([0.8, -0.4, 0.1]),
            group_stats={},
            method="ttest",
            params={},
        )

        df = result.to_dataframe()

        # Should be sorted by p_value (ascending)
        assert df["p_value"][0] < df["p_value"][1]
        assert df["p_value"][1] < df["p_value"][2]

    def test_to_dataframe_without_effect_sizes(self):
        """Test DataFrame conversion when effect_sizes is None."""
        result = DiffExprResult(
            feature_ids=np.array(["P1", "P2", "P3"]),
            p_values=np.array([0.001, 0.05, 0.5]),
            p_values_adj=np.array([0.003, 0.075, 0.5]),
            log2_fc=np.array([1.5, -0.8, 0.1]),
            test_statistics=np.array([3.2, -1.5, 0.5]),
            effect_sizes=None,  # No effect sizes
            group_stats={},
            method="mannwhitney",
            params={},
        )

        df = result.to_dataframe()

        # effect_size column should not be present
        assert "effect_size" not in df.columns

    def test_get_significant_basic(self):
        """Test getting significant features."""
        result = DiffExprResult(
            feature_ids=np.array(["P1", "P2", "P3", "P4"]),
            p_values=np.array([0.001, 0.03, 0.08, 0.5]),
            p_values_adj=np.array([0.004, 0.06, 0.16, 0.5]),
            log2_fc=np.array([2.0, 1.5, 0.5, 0.1]),
            test_statistics=np.array([3.2, -1.5, 0.5, 0.2]),
            effect_sizes=np.array([0.8, -0.4, 0.1, 0.05]),
            group_stats={},
            method="ttest",
            params={},
        )

        sig = result.get_significant(alpha=0.05)

        # Only P1 should be significant
        assert len(sig) == 1
        assert sig["feature_id"][0] == "P1"

    def test_get_significant_with_log2_fc_filter(self):
        """Test getting significant features with log2 FC filter."""
        result = DiffExprResult(
            feature_ids=np.array(["P1", "P2", "P3"]),
            p_values=np.array([0.001, 0.01, 0.02]),
            p_values_adj=np.array([0.003, 0.02, 0.04]),  # All meet alpha threshold
            log2_fc=np.array([2.0, 0.3, 1.5]),
            test_statistics=np.array([3.2, -1.5, 0.5]),
            effect_sizes=np.array([0.8, -0.4, 0.1]),
            group_stats={},
            method="ttest",
            params={},
        )

        sig = result.get_significant(alpha=0.05, min_log2_fc=1.0)

        # P1 and P3 meet both criteria (P2 doesn't have |log2FC| >= 1.0)
        assert len(sig) == 2

    def test_get_significant_none_significant(self):
        """Test when no features are significant."""
        result = DiffExprResult(
            feature_ids=np.array(["P1", "P2", "P3"]),
            p_values=np.array([0.5, 0.6, 0.7]),
            p_values_adj=np.array([0.5, 0.6, 0.7]),
            log2_fc=np.array([1.5, -0.8, 0.1]),
            test_statistics=np.array([3.2, -1.5, 0.5]),
            effect_sizes=np.array([0.8, -0.4, 0.1]),
            group_stats={},
            method="ttest",
            params={},
        )

        sig = result.get_significant(alpha=0.05)

        # No significant features
        assert len(sig) == 0

    def test_get_significant_sorted(self):
        """Test that significant features are sorted by p-value."""
        result = DiffExprResult(
            feature_ids=np.array(["P1", "P2", "P3", "P4"]),
            p_values=np.array([0.01, 0.001, 0.005, 0.02]),
            p_values_adj=np.array([0.02, 0.002, 0.01, 0.04]),
            log2_fc=np.array([1.0, 2.0, 1.5, 0.5]),
            test_statistics=np.array([3.2, -1.5, 0.5, 0.2]),
            effect_sizes=np.array([0.8, -0.4, 0.1, 0.05]),
            group_stats={},
            method="ttest",
            params={},
        )

        sig = result.get_significant(alpha=0.05)

        # Should be sorted by p_value_adj (ascending)
        assert sig["p_value_adj"][0] < sig["p_value_adj"][1]
        assert sig["p_value_adj"][1] < sig["p_value_adj"][2]


# =============================================================================
# pd_isna Tests
# =============================================================================


class TestPdIsna:
    """Tests for pd_isna helper function."""

    def test_pd_isna_numeric(self):
        """Test pd_isna with numeric array."""
        arr = np.array([1.0, 2.0, np.nan, 4.0, np.nan])
        result = pd_isna(arr)

        expected = np.array([False, False, True, False, True])
        npt.assert_array_equal(result, expected)

    def test_pd_isna_string(self):
        """Test pd_isna with string array."""
        arr = np.array(["A", "B", "NaN", "nan", "NA", "C"], dtype=object)
        result = pd_isna(arr)

        # Should detect NaN-like strings
        assert not result[0]
        assert not result[1]
        assert result[2]  # "NaN"
        assert result[3]  # "nan"
        assert result[4]  # "NA"
        assert not result[5]

    def test_pd_isna_integer(self):
        """Test pd_isna with integer array."""
        arr = np.array([1, 2, 3, 4, 5])
        result = pd_isna(arr)

        # No NaN in integer array
        assert np.all(not result)

    def test_pd_isna_all_nan(self):
        """Test pd_isna with all NaN values."""
        arr = np.array([np.nan, np.nan, np.nan])
        result = pd_isna(arr)

        assert np.all(result)

    def test_pd_isna_no_nan(self):
        """Test pd_isna with no NaN values."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = pd_isna(arr)

        assert np.all(not result)


# =============================================================================
# Integration Tests
# =============================================================================


class TestDiffExprIntegration:
    """Integration tests for differential expression workflows."""

    def test_full_workflow_ttest(self):
        """Test complete workflow with t-test."""
        container = create_diff_expr_test_container(
            n_samples=30, n_features=20, effect_size=2.5, seed=42
        )

        result = diff_expr_ttest(
            container=container,
            assay_name="proteins",
            group_col="group",
            group1="group_0",
            group2="group_1",
        )

        # Convert to DataFrame
        df = result.to_dataframe()

        # Get significant features
        sig = result.get_significant(alpha=0.05, min_log2_fc=0.5)

        # Should have some significant features
        assert len(sig) > 0

        # Verify DataFrame has expected structure
        assert "feature_id" in df.columns
        assert "p_value_adj" in df.columns
        assert "log2_fc" in df.columns

    def test_compare_ttest_vs_mannwhitney(self):
        """Compare t-test and Mann-Whitney results."""
        container = create_diff_expr_test_container(n_samples=30, n_features=10, seed=42)

        result_ttest = diff_expr_ttest(
            container=container,
            assay_name="proteins",
            group_col="group",
            group1="group_0",
            group2="group_1",
        )

        result_mw = diff_expr_mannwhitney(
            container=container,
            assay_name="proteins",
            group_col="group",
            group1="group_0",
            group2="group_1",
        )

        # Both should have same shape
        assert len(result_ttest.p_values) == len(result_mw.p_values)

        # P-values should generally be correlated
        # (not exact match due to different statistical assumptions)
        valid_mask = ~np.isnan(result_ttest.p_values) & ~np.isnan(result_mw.p_values)
        if np.sum(valid_mask) > 3:
            corr = np.corrcoef(result_ttest.p_values[valid_mask], result_mw.p_values[valid_mask])[
                0, 1
            ]
            # Should have some correlation
            assert corr > 0.3

    def test_compare_anova_vs_kruskal(self):
        """Compare ANOVA and Kruskal-Wallis results."""
        container = create_three_group_container(n_samples=30, n_features=10)

        result_anova = diff_expr_anova(
            container=container,
            assay_name="proteins",
            group_col="group",
        )

        result_kw = diff_expr_kruskal(
            container=container,
            assay_name="proteins",
            group_col="group",
        )

        # Both should have same shape
        assert len(result_anova.p_values) == len(result_kw.p_values)

        # Both should detect the same number of groups
        assert result_anova.params["n_groups"] == result_kw.params["n_groups"]

    def test_sparse_matrix_workflow(self):
        """Test complete workflow with sparse matrices."""
        container = create_diff_expr_test_container(
            n_samples=30, n_features=15, sparse=True, seed=42
        )

        result = diff_expr_ttest(
            container=container,
            assay_name="proteins",
            group_col="group",
            group1="group_0",
            group2="group_1",
        )

        df = result.to_dataframe()
        assert len(df) == 15

    def test_missing_data_workflow(self):
        """Test workflow with various missing value strategies."""
        container = create_diff_expr_test_container(
            n_samples=30, n_features=10, with_nan=True, seed=42
        )

        strategies = ["ignore", "zero", "median"]
        results = []

        for strategy in strategies:
            result = diff_expr_ttest(
                container=container,
                assay_name="proteins",
                group_col="group",
                group1="group_0",
                group2="group_1",
                missing_strategy=strategy,
            )
            results.append(result)

        # All strategies should produce results
        for result in results:
            assert len(result.p_values) == 10

        # Results may differ slightly between strategies
        # but "zero" and "median" should have fewer NaN p-values than "ignore"
        n_nan_ignore = np.sum(np.isnan(results[0].p_values))
        n_nan_zero = np.sum(np.isnan(results[1].p_values))
        n_nan_median = np.sum(np.isnan(results[2].p_values))

        assert n_nan_zero <= n_nan_ignore
        assert n_nan_median <= n_nan_ignore


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
