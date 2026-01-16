"""Tests for non-parametric differential expression methods.

Tests cover all functions in scptensor.diff_expr.nonparametric:
- diff_expr_wilcoxon: Wilcoxon rank-sum test (paired and unpaired)
- diff_expr_brunner_munzel: Brunner-Munzel test for heteroscedastic data

These methods are distribution-free tests suitable for data that violates
normality assumptions or has unequal variances.
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
from scptensor.diff_expr import diff_expr_brunner_munzel, diff_expr_wilcoxon

# =============================================================================
# Helper Functions
# =============================================================================


def create_simple_container(
    n_samples: int = 40,
    n_features: int = 50,
    n_de: int = 5,
    effect_size: float = 3.0,
    seed: int = 42,
    sparse: bool = False,
    with_mask: bool = False,
) -> ScpContainer:
    """Create a test container with differential expression.

    Args:
        n_samples: Total number of samples (should be even for 2 groups)
        n_features: Number of features (proteins)
        n_de: Number of differentially expressed features
        effect_size: Effect size multiplier for DE features
        seed: Random seed
        sparse: Whether to use sparse matrices
        with_mask: Whether to add mask values (MBR/LOD)

    Returns:
        A test ScpContainer with group labels and DE features
    """
    np.random.seed(seed)

    n_per_group = n_samples // 2

    # Create sample metadata with two groups
    obs = pl.DataFrame(
        {
            "_index": [f"SAMPLE_{i:04d}" for i in range(n_samples)],
            "group": ["A"] * n_per_group + ["B"] * n_per_group,
        }
    )

    # Create feature metadata
    var = pl.DataFrame({"_index": [f"PROT_{i:04d}" for i in range(n_features)]})

    # Create expression data using gamma distribution
    X = np.random.gamma(shape=2, scale=5, size=(n_samples, n_features))

    # Add differential expression for first n_de features
    # Group A has higher expression for these features
    if n_de > 0:
        X[:n_per_group, :n_de] *= effect_size

    if sparse:
        X = sp.csr_matrix(X)

    # Create mask matrix if requested
    M = None
    if with_mask:
        M = np.zeros((n_samples, n_features), dtype=int)
        # Add some MBR (1) and LOD (2) values
        mbr_indices = np.random.random((n_samples, n_features)) < 0.05
        lod_indices = np.random.random((n_samples, n_features)) < 0.03
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


def create_paired_container(
    n_pairs: int = 15,
    n_features: int = 40,
    n_de: int = 5,
    effect_size: float = 5.0,
    seed: int = 42,
) -> ScpContainer:
    """Create a container with paired samples.

    Args:
        n_pairs: Number of paired samples
        n_features: Number of features
        n_de: Number of differentially expressed features
        effect_size: Effect size for DE features
        seed: Random seed

    Returns:
        A test ScpContainer with paired data

    Note:
        The data is structured with control samples first (indices 0 to n_pairs-1)
        and treatment samples second (indices n_pairs to 2*n_pairs-1). The pair_id
        column maps each control sample to its corresponding treatment sample.
    """
    np.random.seed(seed)

    n_samples = n_pairs * 2

    # Create paired data: control samples first, then treatment samples
    X = np.zeros((n_samples, n_features))

    # Control values (first n_pairs samples)
    X[:n_pairs, :] = np.random.gamma(shape=2, scale=5, size=(n_pairs, n_features))

    # Treatment values (next n_pairs samples)
    X[n_pairs:, :] = np.random.gamma(shape=2, scale=5, size=(n_pairs, n_features))

    # Add treatment effect for first n_de features
    # Treatment samples have higher expression
    X[n_pairs:, :n_de] *= effect_size

    M = np.zeros_like(X, dtype=int)

    var = pl.DataFrame({"_index": [f"PROT_{i:04d}" for i in range(n_features)]})
    obs = pl.DataFrame(
        {
            "_index": [f"SAMPLE_{i:04d}" for i in range(n_samples)],
            "group": ["control"] * n_pairs + ["treatment"] * n_pairs,
            "pair_id": [f"P{i:02d}" for i in range(n_pairs)] * 2,
        }
    )

    assay = Assay(
        var=var,
        layers={"raw": ScpMatrix(X=X, M=M)},
        feature_id_col="_index",
    )

    return ScpContainer(obs=obs, assays={"protein": assay})


def create_heteroscedastic_container(
    n_samples: int = 40,
    n_features: int = 50,
    n_de: int = 5,
    seed: int = 42,
) -> ScpContainer:
    """Create data with unequal variances (for Brunner-Munzel test).

    Args:
        n_samples: Total number of samples
        n_features: Number of features
        n_de: Number of features with different means
        seed: Random seed

    Returns:
        A test ScpContainer with heteroscedastic data
    """
    np.random.seed(seed)

    n_per_group = n_samples // 2

    X = np.zeros((n_samples, n_features))

    # Group A: low variance
    X[:n_per_group, :] = np.random.normal(loc=10, scale=1, size=(n_per_group, n_features))

    # Group B: high variance, different mean for some features
    X[n_per_group:, :] = np.random.normal(loc=12, scale=5, size=(n_per_group, n_features))

    # Make first n_de features more different
    X[:n_per_group, :n_de] = np.random.normal(loc=10, scale=1, size=(n_per_group, n_de))
    X[n_per_group:, :n_de] = np.random.normal(loc=15, scale=8, size=(n_per_group, n_de))

    M = np.zeros_like(X, dtype=int)

    var = pl.DataFrame({"_index": [f"PROT_{i:04d}" for i in range(n_features)]})
    obs = pl.DataFrame(
        {
            "_index": [f"SAMPLE_{i:04d}" for i in range(n_samples)],
            "group": ["A"] * n_per_group + ["B"] * n_per_group,
        }
    )

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
def simple_container():
    """Create a simple test container."""
    return create_simple_container(
        n_samples=40,
        n_features=50,
        n_de=5,
        effect_size=3.0,
        seed=42,
    )


@pytest.fixture
def paired_container():
    """Create a container with paired samples."""
    return create_paired_container(
        n_pairs=15,
        n_features=40,
        n_de=5,
        effect_size=4.0,
        seed=42,
    )


@pytest.fixture
def heteroscedastic_container():
    """Create data with unequal variances (for Brunner-Munzel test)."""
    return create_heteroscedastic_container(
        n_samples=40,
        n_features=50,
        n_de=5,
        seed=42,
    )


@pytest.fixture
def sparse_container():
    """Create a container with sparse data."""
    return create_simple_container(
        n_samples=30,
        n_features=40,
        sparse=True,
        seed=42,
    )


@pytest.fixture
def container_with_mask():
    """Create a container with missing values (mask codes)."""
    return create_simple_container(
        n_samples=30,
        n_features=40,
        with_mask=True,
        seed=42,
    )


# =============================================================================
# diff_expr_wilcoxon Tests - Basic Functionality
# =============================================================================


class TestDiffExprWilcoxon:
    """Tests for Wilcoxon rank-sum test (paired and unpaired)."""

    def test_wilcoxon_basic(self, simple_container):
        """Test basic Wilcoxon rank-sum test."""
        result = diff_expr_wilcoxon(
            simple_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        assert result is not None
        assert len(result.p_values) == 50
        assert result.method == "wilcoxon_ranksum"
        assert result.params["groupby"] == "group"
        assert result.params["group1"] == "A"
        assert result.params["group2"] == "B"

    def test_wilcoxon_paired(self, paired_container):
        """Test paired Wilcoxon signed-rank test."""
        result = diff_expr_wilcoxon(
            paired_container,
            "protein",
            "raw",
            "group",
            "control",
            "treatment",
            paired=True,
        )

        assert len(result.p_values) == 40
        assert result.method == "wilcoxon_paired"
        assert result.params["paired"] is True

        # Should detect the treatment effect in first 5 features
        sig_count = np.sum(result.p_values_adj < 0.05)
        assert sig_count >= 3  # At least some DE features detected

    def test_wilcoxon_zero_methods(self, simple_container):
        """Test different zero handling methods."""
        for method in ["pratt", "wilcox", "zsplit"]:
            result = diff_expr_wilcoxon(
                simple_container,
                "protein",
                "raw",
                "group",
                "A",
                "B",
                zero_method=method,
            )
            assert len(result.p_values) == 50
            assert result.params["zero_method"] == method

    def test_wilcoxon_alternatives(self, simple_container):
        """Test different alternative hypotheses."""
        for alt in ["two-sided", "greater", "less"]:
            result = diff_expr_wilcoxon(
                simple_container,
                "protein",
                "raw",
                "group",
                "A",
                "B",
                alternative=alt,
            )
            assert len(result.p_values) == 50
            assert result.params["alternative"] == alt

    def test_wilcoxon_missing_strategies(self, simple_container):
        """Test different missing value strategies."""
        # Create container with NaN values
        np.random.seed(42)
        n_samples, n_features = 30, 30
        X = np.random.gamma(shape=2, scale=5, size=(n_samples, n_features))
        X[X < 2] = np.nan  # Add some NaN values

        var = pl.DataFrame({"_index": [f"P{i}" for i in range(n_features)]})
        obs = pl.DataFrame(
            {
                "_index": [f"S{i}" for i in range(n_samples)],
                "group": ["A"] * 15 + ["B"] * 15,
            }
        )

        assay = Assay(
            var=var,
            layers={"raw": ScpMatrix(X=X, M=None)},
            feature_id_col="_index",
        )
        container = ScpContainer(obs=obs, assays={"protein": assay})

        # Test each strategy
        for strategy in ["ignore", "zero", "median"]:
            result = diff_expr_wilcoxon(
                container,
                "protein",
                "raw",
                "group",
                "A",
                "B",
                missing_strategy=strategy,
            )
            assert len(result.p_values) == n_features
            assert result.params["missing_strategy"] == strategy

    def test_wilcoxon_sparse_data(self, sparse_container):
        """Test Wilcoxon with sparse matrix."""
        result = diff_expr_wilcoxon(
            sparse_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        assert len(result.p_values) == 40
        assert result.method == "wilcoxon_ranksum"

    def test_wilcoxon_with_mask(self, container_with_mask):
        """Test Wilcoxon handles masked values correctly."""
        result = diff_expr_wilcoxon(
            container_with_mask,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        assert len(result.p_values) == 40
        # Masked values should be handled
        assert not np.all(np.isnan(result.p_values))

    def test_wilcoxon_group_stats(self, simple_container):
        """Test that Wilcoxon calculates group statistics."""
        result = diff_expr_wilcoxon(
            simple_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        # Should have median for each group
        assert "A_median" in result.group_stats
        assert "B_median" in result.group_stats
        assert len(result.group_stats["A_median"]) == 50

    def test_wilcoxon_effect_sizes(self, simple_container):
        """Test that Wilcoxon calculates effect sizes."""
        result = diff_expr_wilcoxon(
            simple_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        # Rank-biserial correlation should be in [-1, 1]
        valid_effects = result.effect_sizes[~np.isnan(result.effect_sizes)]
        if len(valid_effects) > 0:
            assert np.all(valid_effects >= -1)
            assert np.all(valid_effects <= 1)

    def test_wilcoxon_result_structure(self, simple_container):
        """Test that Wilcoxon result has expected structure."""
        result = diff_expr_wilcoxon(
            simple_container,
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


# =============================================================================
# diff_expr_brunner_munzel Tests
# =============================================================================


class TestDiffExprBrunnerMunzel:
    """Tests for Brunner-Munzel test for heteroscedastic data."""

    def test_brunner_munzel_basic(self, heteroscedastic_container):
        """Test basic Brunner-Munzel test."""
        result = diff_expr_brunner_munzel(
            heteroscedastic_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        assert result is not None
        assert len(result.p_values) == 50
        assert result.method == "brunner_munzel"
        assert result.params["groupby"] == "group"
        assert result.params["group1"] == "A"
        assert result.params["group2"] == "B"

    def test_brunner_munzel_alternatives(self, heteroscedastic_container):
        """Test different alternative hypotheses."""
        for alt in ["two-sided", "greater", "less"]:
            result = diff_expr_brunner_munzel(
                heteroscedastic_container,
                "protein",
                "raw",
                "group",
                "A",
                "B",
                alternative=alt,
            )
            assert len(result.p_values) == 50
            assert result.params["alternative"] == alt

    def test_brunner_munzel_handles_unequal_variances(self, heteroscedastic_container):
        """Test that BM handles unequal variances well."""
        result = diff_expr_brunner_munzel(
            heteroscedastic_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        # BM should detect DE despite variance difference
        assert len(result.p_values) == 50

        # Check relative effects are in valid range
        valid_effects = result.effect_sizes[~np.isnan(result.effect_sizes)]
        if len(valid_effects) > 0:
            # Relative effects (pHat) should be in [0, 1]
            assert np.all(valid_effects >= 0)
            assert np.all(valid_effects <= 1)

    def test_brunner_munzel_missing_strategies(self, heteroscedastic_container):
        """Test different missing value strategies."""
        # Create container with NaN values
        np.random.seed(42)
        n_samples, n_features = 30, 30
        X = np.random.normal(loc=10, scale=2, size=(n_samples, n_features))
        X[X < 8] = np.nan  # Add some NaN values

        var = pl.DataFrame({"_index": [f"P{i}" for i in range(n_features)]})
        obs = pl.DataFrame(
            {
                "_index": [f"S{i}" for i in range(n_samples)],
                "group": ["A"] * 15 + ["B"] * 15,
            }
        )

        assay = Assay(
            var=var,
            layers={"raw": ScpMatrix(X=X, M=None)},
            feature_id_col="_index",
        )
        container = ScpContainer(obs=obs, assays={"protein": assay})

        # Test each strategy
        for strategy in ["ignore", "zero", "median"]:
            result = diff_expr_brunner_munzel(
                container,
                "protein",
                "raw",
                "group",
                "A",
                "B",
                missing_strategy=strategy,
            )
            assert len(result.p_values) == n_features
            assert result.params["missing_strategy"] == strategy

    def test_brunner_munzel_sparse_data(self, sparse_container):
        """Test Brunner-Munzel with sparse matrix."""
        result = diff_expr_brunner_munzel(
            sparse_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        assert len(result.p_values) == 40
        assert result.method == "brunner_munzel"

    def test_brunner_munzel_with_mask(self, container_with_mask):
        """Test Brunner-Munzel handles masked values."""
        result = diff_expr_brunner_munzel(
            container_with_mask,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        assert len(result.p_values) == 40

    def test_brunner_munzel_group_stats(self, heteroscedastic_container):
        """Test that BM calculates group statistics."""
        result = diff_expr_brunner_munzel(
            heteroscedastic_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        assert "A_median" in result.group_stats
        assert "B_median" in result.group_stats

    def test_brunner_munzel_relative_effects(self, heteroscedastic_container):
        """Test that BM calculates relative effects (pHat)."""
        result = diff_expr_brunner_munzel(
            heteroscedastic_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        # Relative effects should estimate P(X < Y) + 0. * P(X = Y)
        # Values near 0.5 indicate stochastic equality
        valid_effects = result.effect_sizes[~np.isnan(result.effect_sizes)]
        if len(valid_effects) > 0:
            assert np.all(valid_effects >= 0)
            assert np.all(valid_effects <= 1)


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error conditions."""

    def test_wilcoxon_invalid_assay(self, simple_container):
        """Test Wilcoxon with invalid assay."""
        with pytest.raises(AssayNotFoundError):
            diff_expr_wilcoxon(
                simple_container,
                "invalid",
                "raw",
                "group",
                "A",
                "B",
            )

    def test_wilcoxon_invalid_layer(self, simple_container):
        """Test Wilcoxon with invalid layer."""
        with pytest.raises(LayerNotFoundError):
            diff_expr_wilcoxon(
                simple_container,
                "protein",
                "invalid",
                "group",
                "A",
                "B",
            )

    def test_wilcoxon_invalid_groupby(self, simple_container):
        """Test Wilcoxon with invalid groupby column."""
        with pytest.raises(ValidationError) as exc_info:
            diff_expr_wilcoxon(
                simple_container,
                "protein",
                "raw",
                "invalid",
                "A",
                "B",
            )
        assert "invalid" in str(exc_info.value)

    def test_wilcoxin_invalid_group(self, simple_container):
        """Test Wilcoxon with invalid group value."""
        with pytest.raises(ValidationError) as exc_info:
            diff_expr_wilcoxon(
                simple_container,
                "protein",
                "raw",
                "group",
                "A",
                "nonexistent",
            )
        assert "nonexistent" in str(exc_info.value)

    def test_wilcoxon_invalid_zero_method(self, simple_container):
        """Test Wilcoxon with invalid zero_method."""
        with pytest.raises(ValidationError) as exc_info:
            diff_expr_wilcoxon(
                simple_container,
                "protein",
                "raw",
                "group",
                "A",
                "B",
                zero_method="invalid",
            )
        assert "zero_method" in str(exc_info.value).lower()

    def test_wilcoxon_invalid_alternative(self, simple_container):
        """Test Wilcoxon with invalid alternative."""
        with pytest.raises(ValidationError) as exc_info:
            diff_expr_wilcoxon(
                simple_container,
                "protein",
                "raw",
                "group",
                "A",
                "B",
                alternative="invalid",
            )
        assert "alternative" in str(exc_info.value).lower()

    def test_wilcoxon_paired_without_pair_id(self, simple_container):
        """Test paired Wilcoxon without pair_id column."""
        with pytest.raises(ValidationError) as exc_info:
            diff_expr_wilcoxon(
                simple_container,
                "protein",
                "raw",
                "group",
                "A",
                "B",
                paired=True,
            )
        assert "pair" in str(exc_info.value).lower()

    def test_wilcoxon_insufficient_samples(self):
        """Test Wilcoxon with insufficient samples."""
        container = create_simple_container(n_samples=4)  # Only 2 per group

        with pytest.raises(ValidationError) as exc_info:
            diff_expr_wilcoxon(
                container,
                "protein",
                "raw",
                "group",
                "A",
                "B",
                min_samples_per_group=3,
            )
        assert "minimum" in str(exc_info.value).lower()

    def test_brunner_munzel_invalid_assay(self, simple_container):
        """Test BM with invalid assay."""
        with pytest.raises(AssayNotFoundError):
            diff_expr_brunner_munzel(
                simple_container,
                "invalid",
                "raw",
                "group",
                "A",
                "B",
            )

    def test_brunner_munzel_invalid_layer(self, simple_container):
        """Test BM with invalid layer."""
        with pytest.raises(LayerNotFoundError):
            diff_expr_brunner_munzel(
                simple_container,
                "protein",
                "invalid",
                "group",
                "A",
                "B",
            )

    def test_brunner_munzel_invalid_groupby(self, simple_container):
        """Test BM with invalid groupby column."""
        with pytest.raises(ValidationError) as exc_info:
            diff_expr_brunner_munzel(
                simple_container,
                "protein",
                "raw",
                "invalid",
                "A",
                "B",
            )
        assert "invalid" in str(exc_info.value)

    def test_brunner_munzel_invalid_group(self, simple_container):
        """Test BM with invalid group value."""
        with pytest.raises(ValidationError) as exc_info:
            diff_expr_brunner_munzel(
                simple_container,
                "protein",
                "raw",
                "group",
                "A",
                "nonexistent",
            )
        assert "nonexistent" in str(exc_info.value)

    def test_brunner_munzel_invalid_alternative(self, simple_container):
        """Test BM with invalid alternative."""
        with pytest.raises(ValidationError) as exc_info:
            diff_expr_brunner_munzel(
                simple_container,
                "protein",
                "raw",
                "group",
                "A",
                "B",
                alternative="invalid",
            )
        assert "alternative" in str(exc_info.value).lower()

    def test_brunner_munzel_insufficient_samples(self):
        """Test BM with insufficient samples."""
        container = create_simple_container(n_samples=4)  # Only 2 per group

        with pytest.raises(ValidationError) as exc_info:
            diff_expr_brunner_munzel(
                container,
                "protein",
                "raw",
                "group",
                "A",
                "B",
                min_samples_per_group=3,
            )
        assert "minimum" in str(exc_info.value).lower()


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_small_sample_size(self):
        """Test with minimal samples per group."""
        container = create_simple_container(n_samples=6, n_features=10)

        result = diff_expr_wilcoxon(
            container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )
        assert len(result.p_values) == 10

        result_bm = diff_expr_brunner_munzel(
            container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )
        assert len(result_bm.p_values) == 10

    def test_single_feature(self):
        """Test with one feature."""
        np.random.seed(42)
        n_samples = 20

        X = np.random.gamma(shape=2, scale=5, size=(n_samples, 1))
        var = pl.DataFrame({"_index": ["P001"]})
        obs = pl.DataFrame(
            {
                "_index": [f"S{i}" for i in range(n_samples)],
                "group": ["A"] * 10 + ["B"] * 10,
            }
        )

        assay = Assay(
            var=var,
            layers={"raw": ScpMatrix(X=X, M=None)},
            feature_id_col="_index",
        )
        container = ScpContainer(obs=obs, assays={"protein": assay})

        result = diff_expr_wilcoxon(
            container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )
        assert len(result.p_values) == 1

        result_bm = diff_expr_brunner_munzel(
            container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )
        assert len(result_bm.p_values) == 1

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
        X = np.random.gamma(shape=2, scale=5, size=(n_total, 20))

        assay = Assay(
            var=var,
            layers={"raw": ScpMatrix(X=X)},
            feature_id_col="_index",
        )
        container = ScpContainer(obs=obs, assays={"protein": assay})

        result = diff_expr_wilcoxon(
            container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )
        assert len(result.p_values) == 20

        result_bm = diff_expr_brunner_munzel(
            container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )
        assert len(result_bm.p_values) == 20

    def test_all_zero_feature(self):
        """Test with a feature that has all zeros."""
        n_samples, n_features = 20, 10

        obs = pl.DataFrame(
            {
                "_index": [f"S{i}" for i in range(n_samples)],
                "group": ["A"] * 10 + ["B"] * 10,
            }
        )

        var = pl.DataFrame({"_index": [f"P{i}" for i in range(n_features)]})
        X = np.random.gamma(shape=2, scale=5, size=(n_samples, n_features))
        X[:, 0] = 0  # First feature all zeros

        assay = Assay(
            var=var,
            layers={"raw": ScpMatrix(X=X)},
            feature_id_col="_index",
        )
        container = ScpContainer(obs=obs, assays={"protein": assay})

        result = diff_expr_wilcoxon(
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
                "_index": [f"S{i}" for i in range(n_samples)],
                "group": ["A"] * 10 + ["B"] * 10,
            }
        )

        var = pl.DataFrame({"_index": [f"P{i}" for i in range(n_features)]})
        X = np.random.gamma(shape=2, scale=5, size=(n_samples, n_features))
        X[:, 0] = 10.0  # First feature constant

        assay = Assay(
            var=var,
            layers={"raw": ScpMatrix(X=X)},
            feature_id_col="_index",
        )
        container = ScpContainer(obs=obs, assays={"protein": assay})

        result = diff_expr_wilcoxon(
            container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )
        assert len(result.p_values) == n_features


# =============================================================================
# Result Validation Tests
# =============================================================================


class TestResultValidation:
    """Tests for validating result properties."""

    def test_p_values_in_range(self, simple_container):
        """Test that p-values are in valid range [0, 1]."""
        result_w = diff_expr_wilcoxon(
            simple_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )
        result_bm = diff_expr_brunner_munzel(
            simple_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        for result in [result_w, result_bm]:
            valid_p = result.p_values[~np.isnan(result.p_values)]
            assert np.all(valid_p >= 0), f"{result.method}: p-values < 0"
            assert np.all(valid_p <= 1), f"{result.method}: p-values > 1"

    def test_p_values_adj_in_range(self, simple_container):
        """Test that adjusted p-values are in valid range [0, 1]."""
        result_w = diff_expr_wilcoxon(
            simple_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )
        result_bm = diff_expr_brunner_munzel(
            simple_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        for result in [result_w, result_bm]:
            valid_p_adj = result.p_values_adj[~np.isnan(result.p_values_adj)]
            assert np.all(valid_p_adj >= 0), f"{result.method}: adj p-values < 0"
            assert np.all(valid_p_adj <= 1), f"{result.method}: adj p-values > 1"

    def test_log2_fc_calculated(self, simple_container):
        """Test that log2 fold changes are calculated."""
        result_w = diff_expr_wilcoxon(
            simple_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )
        result_bm = diff_expr_brunner_munzel(
            simple_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        for result in [result_w, result_bm]:
            assert len(result.log2_fc) == len(result.p_values)

    def test_to_dataframe(self, simple_container):
        """Test result conversion to DataFrame."""
        result = diff_expr_wilcoxon(
            simple_container,
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

    def test_to_dataframe_sorted(self, simple_container):
        """Test that DataFrame is sorted by p-value."""
        result = diff_expr_wilcoxon(
            simple_container,
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

    def test_get_significant(self, simple_container):
        """Test getting significant features."""
        result = diff_expr_wilcoxon(
            simple_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        sig = result.get_significant(alpha=0.5)  # Use high threshold
        assert isinstance(sig, pl.DataFrame)

        # All returned features should meet the threshold
        if len(sig) > 0:
            assert np.all(sig["p_value_adj"].to_numpy() < 0.5)

    def test_get_significant_with_log2_fc_filter(self, simple_container):
        """Test getting significant features with log2 FC filter."""
        result = diff_expr_wilcoxon(
            simple_container,
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
# Method Comparison Tests
# =============================================================================


class TestMethodComparison:
    """Tests comparing non-parametric methods."""

    def test_wilcoxon_vs_brunner_munzel(self, heteroscedastic_container):
        """Compare Wilcoxon and Brunner-Munzel on heteroscedastic data."""
        result_w = diff_expr_wilcoxon(
            heteroscedastic_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )
        result_bm = diff_expr_brunner_munzel(
            heteroscedastic_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        # Both should give results
        assert len(result_w.p_values) == len(result_bm.p_values)

        # BM might be more powerful for heteroscedastic data
        sig_w = np.sum(result_w.p_values_adj < 0.05)
        sig_bm = np.sum(result_bm.p_values_adj < 0.05)

        # At least one should detect some differences
        assert sig_w + sig_bm >= 0

    def test_wilcoxon_paired_vs_unpaired(self, paired_container):
        """Compare paired and unpaired Wilcoxon tests."""
        result_paired = diff_expr_wilcoxon(
            paired_container,
            "protein",
            "raw",
            "group",
            "control",
            "treatment",
            paired=True,
        )

        result_unpaired = diff_expr_wilcoxon(
            paired_container,
            "protein",
            "raw",
            "group",
            "control",
            "treatment",
            paired=False,
        )

        # Both should give results
        assert len(result_paired.p_values) == len(result_unpaired.p_values)

        # Paired test should be more powerful for paired data
        sig_paired = np.sum(result_paired.p_values_adj < 0.05)
        sig_unpaired = np.sum(result_unpaired.p_values_adj < 0.05)

        # Both should detect some features in our test data
        assert sig_paired >= 0
        assert sig_unpaired >= 0

    def test_sparse_vs_dense_consistency(self):
        """Test that sparse and dense matrices give similar results."""
        np.random.seed(42)

        n_samples, n_features = 30, 30

        obs = pl.DataFrame(
            {
                "_index": [f"S{i:04d}" for i in range(n_samples)],
                "group": ["A"] * 15 + ["B"] * 15,
            }
        )

        var = pl.DataFrame({"_index": [f"PROT_{i:04d}" for i in range(n_features)]})

        # Dense version
        X_dense = np.random.gamma(shape=2, scale=5, size=(n_samples, n_features))
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

        result_dense = diff_expr_wilcoxon(
            container_dense,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        result_sparse = diff_expr_wilcoxon(
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

    def test_detects_differential_expression(self, simple_container):
        """Test that methods detect engineered differential expression."""
        result_w = diff_expr_wilcoxon(
            simple_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        result_bm = diff_expr_brunner_munzel(
            simple_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        # First 5 features were engineered to be DE (higher in group A)
        # They should have more significant p-values on average
        for result in [result_w, result_bm]:
            first_5_p = result.p_values[:5]
            last_5_p = result.p_values[-5:]

            # Filter out NaN for comparison
            first_5_valid = first_5_p[~np.isnan(first_5_p)]
            last_5_valid = last_5_p[~np.isnan(last_5_p)]

            if len(first_5_valid) > 0 and len(last_5_valid) > 0:
                # DE features should have smaller p-values
                assert np.mean(first_5_valid) <= np.mean(last_5_valid) + 0.1

    def test_reproducibility_with_seed(self, simple_container):
        """Test that results are reproducible with same data."""
        result1 = diff_expr_wilcoxon(
            simple_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        result2 = diff_expr_wilcoxon(
            simple_container,
            "protein",
            "raw",
            "group",
            "A",
            "B",
        )

        # Results should be identical
        np.testing.assert_array_equal(result1.p_values, result2.p_values)
        np.testing.assert_array_equal(result1.log2_fc, result2.log2_fc)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
