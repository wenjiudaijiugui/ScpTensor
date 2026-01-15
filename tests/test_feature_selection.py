"""
Tests for feature_selection module.

This module tests all feature selection methods:
- dropout.py: select_by_dropout, get_dropout_stats
- hvg.py: select_hvg
- vst.py: select_by_vst, select_by_dispersion
- model.py: select_by_model_importance, select_by_pca_loadings
"""

import numpy as np
import polars as pl
import pytest
from scipy import sparse

from scptensor.core import Assay, ScpContainer, ScpMatrix
from scptensor.feature_selection import (
    get_dropout_stats,
    select_by_dispersion,
    select_by_dropout,
    select_by_model_importance,
    select_by_pca_loadings,
    select_by_vst,
    select_hvg,
)

# =============================================================================
# Fixtures for feature selection tests
# =============================================================================


@pytest.fixture
def fs_container():
    """Create a container suitable for feature selection testing.

    This container has:
    - 50 samples
    - 100 features
    - Varying dropout rates and variance patterns
    """
    np.random.seed(42)
    n_samples = 50
    n_features = 100

    # Create data with varying patterns
    X = np.random.gamma(shape=1, scale=5, size=(n_samples, n_features))

    # Introduce varying dropout rates
    # First 20 features: high dropout (>50%)
    X[:30, :20] = 0
    # Features 20-40: medium dropout (~30%)
    X[:15, 20:40] = 0
    # Features 40-60: low dropout (~10%)
    X[:5, 40:60] = 0
    # Last 40 features: very low dropout (<5%)

    # Add some NaN values
    X[5:10, 5:10] = np.nan

    # Make some features more variable
    for i in range(60, 80):
        X[:, i] = np.random.gamma(shape=0.5 + (i - 60) * 0.1, scale=10, size=n_samples)

    # Add low variance features
    for i in range(80, 100):
        X[:, i] = np.random.normal(loc=5, scale=0.5, size=n_samples)

    var = pl.DataFrame(
        {
            "_index": [f"feature_{i}" for i in range(n_features)],
            "protein_name": [f"Protein{i}" for i in range(n_features)],
        }
    )
    obs = pl.DataFrame(
        {
            "_index": [f"sample_{i}" for i in range(n_samples)],
            "batch": ["batch1"] * 25 + ["batch2"] * 25,
        }
    )

    assay = Assay(var=var, layers={"raw": ScpMatrix(X=X, M=None)}, feature_id_col="_index")

    return ScpContainer(obs=obs, assays={"protein": assay})


@pytest.fixture
def fs_container_with_mask():
    """Create a container with mask matrix for testing."""
    np.random.seed(42)
    n_samples = 30
    n_features = 50

    X = np.random.rand(n_samples, n_features) * 10

    # Create mask with various codes
    M = np.zeros((n_samples, n_features), dtype=np.int8)
    # Some MBR (missing between runs)
    M[0:5, 0:10] = 1
    # Some LOD (below detection limit)
    M[5:10, 10:15] = 2
    # Some FILTERED
    M[10:12, 15:17] = 3

    var = pl.DataFrame({"_index": [f"feature_{i}" for i in range(n_features)]})
    obs = pl.DataFrame({"_index": [f"sample_{i}" for i in range(n_samples)]})

    assay = Assay(var=var, layers={"raw": ScpMatrix(X=X, M=M)}, feature_id_col="_index")

    return ScpContainer(obs=obs, assays={"protein": assay})


@pytest.fixture
def fs_sparse_container():
    """Create a container with sparse matrix for testing."""
    np.random.seed(42)
    n_samples = 40
    n_features = 60

    # Create sparse-like data
    X_dense = np.random.gamma(shape=1, scale=5, size=(n_samples, n_features))
    X_dense[X_dense < 3] = 0  # Introduce sparsity

    X_sparse = sparse.csr_matrix(X_dense)

    var = pl.DataFrame({"_index": [f"feature_{i}" for i in range(n_features)]})
    obs = pl.DataFrame({"_index": [f"sample_{i}" for i in range(n_samples)]})

    assay = Assay(var=var, layers={"raw": ScpMatrix(X=X_sparse, M=None)}, feature_id_col="_index")

    return ScpContainer(obs=obs, assays={"protein": assay})


@pytest.fixture
def fs_small_container():
    """Create a small container for edge case testing."""
    np.random.seed(42)
    n_samples = 10
    n_features = 20

    X = np.random.rand(n_samples, n_features) * 5

    var = pl.DataFrame({"_index": [f"feature_{i}" for i in range(n_features)]})
    obs = pl.DataFrame({"_index": [f"sample_{i}" for i in range(n_samples)]})

    assay = Assay(var=var, layers={"raw": ScpMatrix(X=X, M=None)}, feature_id_col="_index")

    return ScpContainer(obs=obs, assays={"protein": assay})


@pytest.fixture
def fs_multi_layer_container():
    """Create a container with multiple layers."""
    np.random.seed(42)
    n_samples = 30
    n_features = 50

    X_raw = np.random.gamma(shape=1, scale=5, size=(n_samples, n_features))
    X_log = np.log1p(X_raw)
    X_norm = (X_log - X_log.mean()) / (X_log.std() + 1e-8)

    var = pl.DataFrame({"_index": [f"feature_{i}" for i in range(n_features)]})
    obs = pl.DataFrame({"_index": [f"sample_{i}" for i in range(n_samples)]})

    assay = Assay(
        var=var,
        layers={
            "raw": ScpMatrix(X=X_raw, M=None),
            "log": ScpMatrix(X=X_log, M=None),
            "normalized": ScpMatrix(X=X_norm, M=None),
        },
        feature_id_col="_index",
    )

    return ScpContainer(obs=obs, assays={"protein": assay})


# =============================================================================
# Tests for dropout.py
# =============================================================================


class TestSelectByDropout:
    """Tests for select_by_dropout function."""

    def test_basic_dropout_filtering(self, fs_container):
        """Test basic dropout filtering with subset=True."""
        result = select_by_dropout(
            fs_container, assay_name="protein", max_dropout_rate=0.4, subset=True
        )

        # Should filter out high-dropout features
        assert result.assays["protein"].n_features < fs_container.assays["protein"].n_features
        # Check history was logged
        assert len(result.history) > len(fs_container.history)

    def test_dropout_annotation_mode(self, fs_container):
        """Test dropout filtering with subset=False (annotation mode)."""
        result = select_by_dropout(
            fs_container, assay_name="protein", max_dropout_rate=0.3, subset=False
        )

        # Should have same number of features
        assert result.assays["protein"].n_features == fs_container.assays["protein"].n_features

        # Check annotation columns were added
        var = result.assays["protein"].var
        assert "pass_dropout_filter" in var.columns
        assert "dropout_rate" in var.columns
        assert "n_detected" in var.columns

        # Check that some features pass
        n_pass = var["pass_dropout_filter"].sum()
        assert n_pass > 0
        assert n_pass < fs_container.assays["protein"].n_features

    def test_dropout_with_min_detected(self, fs_container):
        """Test dropout filtering with min_detected constraint."""
        result = select_by_dropout(fs_container, assay_name="protein", min_detected=25, subset=True)

        # Should filter out features detected in fewer samples
        assert result.assays["protein"].n_features < fs_container.assays["protein"].n_features

    def test_dropout_with_mask(self, fs_container_with_mask):
        """Test dropout filtering respects mask matrix."""
        result = select_by_dropout(
            fs_container_with_mask, assay_name="protein", max_dropout_rate=0.5, subset=True
        )

        # Mask values should be considered as missing
        assert result.assays["protein"].n_features > 0

    def test_dropout_with_sparse_matrix(self, fs_sparse_container):
        """Test dropout filtering works with sparse matrices."""
        result = select_by_dropout(
            fs_sparse_container, assay_name="protein", max_dropout_rate=0.6, subset=True
        )

        # Should handle sparse matrices
        assert result.assays["protein"].n_features > 0

    def test_dropout_invalid_assay(self, fs_container):
        """Test ValueError for non-existent assay."""
        with pytest.raises(ValueError, match="Assay '.*' not found"):
            select_by_dropout(fs_container, assay_name="nonexistent")

    def test_dropout_invalid_layer(self, fs_container):
        """Test ValueError for non-existent layer."""
        with pytest.raises(ValueError, match="Layer '.*' not found"):
            select_by_dropout(fs_container, layer="nonexistent")

    def test_dropout_invalid_max_rate(self, fs_container):
        """Test ValueError for invalid max_dropout_rate."""
        with pytest.raises(ValueError, match="max_dropout_rate must be in \\[0, 1\\]"):
            select_by_dropout(fs_container, max_dropout_rate=1.5)

        with pytest.raises(ValueError, match="max_dropout_rate must be in \\[0, 1\\]"):
            select_by_dropout(fs_container, max_dropout_rate=-0.1)

    def test_dropout_no_features_pass(self, fs_small_container):
        """Test ValueError when no features pass filter."""
        with pytest.raises(ValueError, match="No features pass the dropout filter"):
            select_by_dropout(
                fs_small_container, max_dropout_rate=0.001, min_detected=100, subset=True
            )

    def test_dropout_relaxed_filter(self, fs_container):
        """Test with very relaxed filter criteria."""
        result = select_by_dropout(fs_container, max_dropout_rate=0.99, min_detected=1, subset=True)

        # Most features should pass
        assert result.assays["protein"].n_features > fs_container.assays["protein"].n_features * 0.5

    def test_dropout_multi_layer(self, fs_multi_layer_container):
        """Test dropout filtering on different layers."""
        # Test on log layer
        result_log = select_by_dropout(
            fs_multi_layer_container, layer="log", max_dropout_rate=0.5, subset=True
        )
        assert "log" in result_log.assays["protein"].layers


class TestGetDropoutStats:
    """Tests for get_dropout_stats function."""

    def test_basic_stats(self, fs_container):
        """Test basic dropout statistics calculation."""
        stats = get_dropout_stats(fs_container, assay_name="protein")

        assert isinstance(stats, pl.DataFrame)
        assert len(stats) == fs_container.assays["protein"].n_features
        assert "dropout_rate" in stats.columns
        assert "n_detected" in stats.columns
        assert "n_missing" in stats.columns
        assert "mean_intensity" in stats.columns

    def test_stats_values(self, fs_container):
        """Test that stats values are within expected ranges."""
        stats = get_dropout_stats(fs_container, assay_name="protein")

        # Dropout rate should be in [0, 1]
        assert (
            stats["dropout_rate"]
            .filter((stats["dropout_rate"] < 0) | (stats["dropout_rate"] > 1))
            .is_empty()
        )

        # n_detected + n_missing should equal n_samples
        n_samples = fs_container.n_samples
        for row in stats.iter_rows(named=True):
            assert row["n_detected"] + row["n_missing"] == n_samples

    def test_stats_with_mask(self, fs_container_with_mask):
        """Test stats calculation respects mask matrix."""
        stats = get_dropout_stats(fs_container_with_mask, assay_name="protein")

        assert len(stats) == fs_container_with_mask.assays["protein"].n_features
        # Features with mask should have higher dropout
        assert stats["dropout_rate"].max() > 0

    def test_stats_with_sparse(self, fs_sparse_container):
        """Test stats calculation with sparse matrix."""
        stats = get_dropout_stats(fs_sparse_container, assay_name="protein")

        assert len(stats) == fs_sparse_container.assays["protein"].n_features

    def test_stats_invalid_assay(self, fs_container):
        """Test ValueError for non-existent assay."""
        with pytest.raises(ValueError, match="Assay '.*' not found"):
            get_dropout_stats(fs_container, assay_name="nonexistent")

    def test_stats_invalid_layer(self, fs_container):
        """Test ValueError for non-existent layer."""
        with pytest.raises(ValueError, match="Layer '.*' not found"):
            get_dropout_stats(fs_container, layer="nonexistent")


# =============================================================================
# Tests for hvg.py
# =============================================================================


class TestSelectHVG:
    """Tests for select_hvg function."""

    def test_basic_hvg_selection_cv(self, fs_container):
        """Test basic HVG selection with CV method."""
        result = select_hvg(
            fs_container, assay_name="protein", n_top_features=30, method="cv", subset=True
        )

        assert result.assays["protein"].n_features == 30
        assert len(result.history) > len(fs_container.history)

    def test_basic_hvg_selection_dispersion(self, fs_container):
        """Test HVG selection with dispersion method."""
        result = select_hvg(
            fs_container, assay_name="protein", n_top_features=25, method="dispersion", subset=True
        )

        assert result.assays["protein"].n_features == 25

    def test_hvg_annotation_mode(self, fs_container):
        """Test HVG annotation mode."""
        result = select_hvg(
            fs_container, assay_name="protein", n_top_features=30, method="cv", subset=False
        )

        # Same number of features
        assert result.assays["protein"].n_features == fs_container.assays["protein"].n_features

        # Check annotation columns
        var = result.assays["protein"].var
        assert "highly_variable" in var.columns
        assert "variability_score" in var.columns

        n_hvg = var["highly_variable"].sum()
        assert n_hvg == 30

    def test_hvg_n_top_greater_than_features(self, fs_small_container):
        """Test when n_top_features > n_features."""
        n_features = fs_small_container.assays["protein"].n_features
        result = select_hvg(fs_small_container, n_top_features=n_features + 100, subset=True)

        # Should return all features
        assert result.assays["protein"].n_features == n_features

    def test_hvg_invalid_assay(self, fs_container):
        """Test ValueError for non-existent assay."""
        with pytest.raises(ValueError, match="Assay '.*' not found"):
            select_hvg(fs_container, assay_name="nonexistent")

    def test_hvg_invalid_layer(self, fs_container):
        """Test ValueError for non-existent layer."""
        with pytest.raises(ValueError, match="Layer '.*' not found"):
            select_hvg(fs_container, layer="nonexistent")

    def test_hvg_with_sparse(self, fs_sparse_container):
        """Test HVG with sparse matrix."""
        result = select_hvg(fs_sparse_container, n_top_features=20, subset=True)

        assert result.assays["protein"].n_features == 20

    def test_hvg_multi_layer(self, fs_multi_layer_container):
        """Test HVG on different layers."""
        result = select_hvg(fs_multi_layer_container, layer="log", n_top_features=25, subset=True)

        assert result.assays["protein"].n_features == 25


# =============================================================================
# Tests for vst.py
# =============================================================================


class TestSelectByVST:
    """Tests for select_by_vst function."""

    def test_basic_vst_selection(self, fs_container):
        """Test basic VST selection."""
        result = select_by_vst(fs_container, assay_name="protein", n_top_features=30, subset=True)

        assert result.assays["protein"].n_features == 30
        assert len(result.history) > len(fs_container.history)

    def test_vst_annotation_mode(self, fs_container):
        """Test VST annotation mode."""
        result = select_by_vst(fs_container, assay_name="protein", n_top_features=30, subset=False)

        # Same number of features
        assert result.assays["protein"].n_features == fs_container.assays["protein"].n_features

        # Check annotation columns
        var = result.assays["protein"].var
        assert "highly_variable" in var.columns
        assert "vst_score" in var.columns

        n_hvg = var["highly_variable"].sum()
        assert n_hvg == 30

    def test_vst_with_min_mean(self, fs_container):
        """Test VST with min_mean filtering."""
        result = select_by_vst(
            fs_container, assay_name="protein", n_top_features=20, min_mean=5.0, subset=True
        )

        # Should select features above mean threshold
        assert result.assays["protein"].n_features == 20

    def test_vst_custom_bins(self, fs_container):
        """Test VST with custom number of bins."""
        result = select_by_vst(
            fs_container, assay_name="protein", n_top_features=25, n_bins=10, subset=True
        )

        assert result.assays["protein"].n_features == 25

    def test_vst_invalid_span(self, fs_container):
        """Test ValueError for invalid span parameter."""
        with pytest.raises(ValueError, match="span must be in \\(0, 1\\]"):
            select_by_vst(fs_container, span=1.5)

        with pytest.raises(ValueError, match="span must be in \\(0, 1\\]"):
            select_by_vst(fs_container, span=0)

    def test_vst_invalid_assay(self, fs_container):
        """Test ValueError for non-existent assay."""
        with pytest.raises(ValueError, match="Assay '.*' not found"):
            select_by_vst(fs_container, assay_name="nonexistent")

    def test_vst_invalid_layer(self, fs_container):
        """Test ValueError for non-existent layer."""
        with pytest.raises(ValueError, match="Layer '.*' not found"):
            select_by_vst(fs_container, layer="nonexistent")

    def test_vst_with_sparse(self, fs_sparse_container):
        """Test VST with sparse matrix."""
        result = select_by_vst(fs_sparse_container, n_top_features=20, subset=True)

        assert result.assays["protein"].n_features == 20

    def test_vst_n_top_greater_than_features(self, fs_small_container):
        """Test when n_top_features > n_features."""
        n_features = fs_small_container.assays["protein"].n_features
        result = select_by_vst(fs_small_container, n_top_features=n_features + 100, subset=True)

        assert result.assays["protein"].n_features == n_features


class TestSelectByDispersion:
    """Tests for select_by_dispersion function."""

    def test_basic_dispersion_selection(self, fs_container):
        """Test basic dispersion-based selection."""
        result = select_by_dispersion(
            fs_container, assay_name="protein", n_top_features=30, subset=True
        )

        assert result.assays["protein"].n_features == 30
        assert len(result.history) > len(fs_container.history)

    def test_dispersion_annotation_mode(self, fs_container):
        """Test dispersion annotation mode."""
        result = select_by_dispersion(
            fs_container, assay_name="protein", n_top_features=25, subset=False
        )

        # Same number of features
        assert result.assays["protein"].n_features == fs_container.assays["protein"].n_features

        # Check annotation columns
        var = result.assays["protein"].var
        assert "highly_variable" in var.columns
        assert "dispersion_score" in var.columns

        n_hvg = var["highly_variable"].sum()
        assert n_hvg == 25

    def test_dispersion_custom_bins(self, fs_container):
        """Test dispersion with custom number of bins."""
        result = select_by_dispersion(
            fs_container, assay_name="protein", n_top_features=20, n_bins=10, subset=True
        )

        assert result.assays["protein"].n_features == 20

    def test_dispersion_invalid_assay(self, fs_container):
        """Test ValueError for non-existent assay."""
        with pytest.raises(ValueError, match="Assay '.*' not found"):
            select_by_dispersion(fs_container, assay_name="nonexistent")

    def test_dispersion_invalid_layer(self, fs_container):
        """Test ValueError for non-existent layer."""
        with pytest.raises(ValueError, match="Layer '.*' not found"):
            select_by_dispersion(fs_container, layer="nonexistent")

    def test_dispersion_with_sparse(self, fs_sparse_container):
        """Test dispersion with sparse matrix."""
        result = select_by_dispersion(fs_sparse_container, n_top_features=20, subset=True)

        assert result.assays["protein"].n_features == 20

    def test_dispersion_n_top_greater_than_features(self, fs_small_container):
        """Test when n_top_features > n_features."""
        n_features = fs_small_container.assays["protein"].n_features
        result = select_by_dispersion(
            fs_small_container, n_top_features=n_features + 100, subset=True
        )

        assert result.assays["protein"].n_features == n_features


# =============================================================================
# Tests for model.py
# =============================================================================


class TestSelectByModelImportance:
    """Tests for select_by_model_importance function."""

    def test_variance_threshold_method(self, fs_container):
        """Test model-based selection with variance threshold."""
        result = select_by_model_importance(
            fs_container,
            assay_name="protein",
            method="variance_threshold",
            variance_threshold=1.0,
            n_top_features=30,
            subset=True,
        )

        assert result.assays["protein"].n_features <= 30
        assert len(result.history) > len(fs_container.history)

    def test_variance_threshold_annotation_mode(self, fs_container):
        """Test variance threshold in annotation mode."""
        result = select_by_model_importance(
            fs_container,
            assay_name="protein",
            method="variance_threshold",
            n_top_features=25,
            subset=False,
        )

        # Same number of features
        assert result.assays["protein"].n_features == fs_container.assays["protein"].n_features

        # Check annotation columns
        var = result.assays["protein"].var
        assert "selected_by_model" in var.columns
        assert "model_importance" in var.columns

        n_selected = var["selected_by_model"].sum()
        assert n_selected == 25

    def test_random_forest_method(self, fs_container):
        """Test random forest importance method."""
        try:
            result = select_by_model_importance(
                fs_container,
                assay_name="protein",
                method="random_forest",
                n_top_features=30,
                n_estimators=20,
                subset=True,
            )

            assert result.assays["protein"].n_features == 30
        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_mutual_info_method(self, fs_container):
        """Test mutual info method."""
        result = select_by_model_importance(
            fs_container, assay_name="protein", method="mutual_info", n_top_features=30, subset=True
        )

        # Should fall back to variance-based selection
        assert result.assays["protein"].n_features == 30

    def test_model_invalid_method(self, fs_container):
        """Test ValueError for invalid method."""
        with pytest.raises(ValueError, match="method must be"):
            select_by_model_importance(fs_container, method="invalid_method")

    def test_model_invalid_assay(self, fs_container):
        """Test ValueError for non-existent assay."""
        with pytest.raises(ValueError, match="Assay '.*' not found"):
            select_by_model_importance(fs_container, assay_name="nonexistent")

    def test_model_invalid_layer(self, fs_container):
        """Test ValueError for non-existent layer."""
        with pytest.raises(ValueError, match="Layer '.*' not found"):
            select_by_model_importance(fs_container, layer="nonexistent")

    def test_model_with_sparse(self, fs_sparse_container):
        """Test model-based selection with sparse matrix."""
        result = select_by_model_importance(
            fs_sparse_container,
            assay_name="protein",
            method="variance_threshold",
            n_top_features=20,
            subset=True,
        )

        assert result.assays["protein"].n_features == 20

    def test_model_n_top_greater_than_features(self, fs_small_container):
        """Test when n_top_features > n_features."""
        n_features = fs_small_container.assays["protein"].n_features
        result = select_by_model_importance(
            fs_small_container,
            method="variance_threshold",
            n_top_features=n_features + 100,
            subset=True,
        )

        assert result.assays["protein"].n_features == n_features

    def test_model_max_depth_parameter(self, fs_container):
        """Test max_depth parameter for random forest."""
        try:
            result = select_by_model_importance(
                fs_container,
                assay_name="protein",
                method="random_forest",
                n_top_features=25,
                max_depth=5,
                n_estimators=20,
                subset=True,
            )

            assert result.assays["protein"].n_features == 25
        except ImportError:
            pytest.skip("scikit-learn not available")


class TestSelectByPCALoadings:
    """Tests for select_by_pca_loadings function."""

    def test_basic_pca_selection(self, fs_container):
        """Test basic PCA loading selection."""
        try:
            result = select_by_pca_loadings(
                fs_container, assay_name="protein", n_top_features=30, n_components=10, subset=True
            )

            assert result.assays["protein"].n_features == 30
            assert len(result.history) > len(fs_container.history)
        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_pca_annotation_mode(self, fs_container):
        """Test PCA in annotation mode."""
        try:
            result = select_by_pca_loadings(
                fs_container, assay_name="protein", n_top_features=25, n_components=10, subset=False
            )

            # Same number of features
            assert result.assays["protein"].n_features == fs_container.assays["protein"].n_features

            # Check annotation columns
            var = result.assays["protein"].var
            assert "selected_by_pca" in var.columns
            assert "pca_importance" in var.columns

            n_selected = var["selected_by_pca"].sum()
            assert n_selected == 25
        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_pca_invalid_assay(self, fs_container):
        """Test ValueError for non-existent assay."""
        with pytest.raises(ValueError, match="Assay '.*' not found"):
            select_by_pca_loadings(fs_container, assay_name="nonexistent")

    def test_pca_invalid_layer(self, fs_container):
        """Test ValueError for non-existent layer."""
        with pytest.raises(ValueError, match="Layer '.*' not found"):
            select_by_pca_loadings(fs_container, layer="nonexistent")

    def test_pca_with_sparse(self, fs_sparse_container):
        """Test PCA with sparse matrix."""
        try:
            result = select_by_pca_loadings(
                fs_sparse_container,
                assay_name="protein",
                n_top_features=20,
                n_components=5,
                subset=True,
            )

            assert result.assays["protein"].n_features == 20
        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_pca_n_top_greater_than_features(self, fs_small_container):
        """Test when n_top_features > n_features."""
        try:
            n_features = fs_small_container.assays["protein"].n_features
            result = select_by_pca_loadings(
                fs_small_container, n_top_features=n_features + 100, subset=True
            )

            assert result.assays["protein"].n_features == n_features
        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_pca_custom_components(self, fs_container):
        """Test PCA with custom number of components."""
        try:
            result = select_by_pca_loadings(
                fs_container, assay_name="protein", n_top_features=20, n_components=15, subset=True
            )

            assert result.assays["protein"].n_features == 20
        except ImportError:
            pytest.skip("scikit-learn not available")


# =============================================================================
# Integration tests
# =============================================================================


class TestFeatureSelectionIntegration:
    """Integration tests for feature selection methods."""

    def test_combined_dropout_and_hvg(self, fs_container):
        """Test combining dropout filter with HVG selection."""
        # First filter by dropout
        filtered = select_by_dropout(fs_container, max_dropout_rate=0.5, subset=True)

        # Then select HVG from filtered
        result = select_hvg(filtered, n_top_features=20, subset=True)

        assert result.assays["protein"].n_features == 20

    def test_vst_and_dispersion_consistency(self, fs_container):
        """Test that VST and dispersion give reasonable results."""
        result_vst = select_by_vst(fs_container, n_top_features=30, subset=False)

        result_disp = select_by_dispersion(fs_container, n_top_features=30, subset=False)

        # Both should select 30 features
        assert result_vst.assays["protein"].var["highly_variable"].sum() == 30
        assert result_disp.assays["protein"].var["highly_variable"].sum() == 30

    def test_pipeline_multi_assay(self, fs_multi_layer_container):
        """Test feature selection pipeline with multiple layers."""
        # Select features on raw layer
        result = select_by_dropout(
            fs_multi_layer_container, layer="raw", max_dropout_rate=0.5, subset=True
        )

        # Other layers should still be accessible
        assert "log" in result.assays["protein"].layers
        assert "normalized" in result.assays["protein"].layers
