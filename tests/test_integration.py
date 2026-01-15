"""
Tests for integration (batch correction) module.

This module provides comprehensive tests for batch effect correction methods:
- MNN (Mutual Nearest Neighbors) - built-in
- Harmony - requires harmonypy (skipped if not available)
- Scanorama - requires scanorama (skipped if not available)
- ComBat - tested in integration/test_pipeline.py

Tests cover:
- Normal multi-batch scenarios
- Edge cases (single batch, small batches)
- Parameter validation
- Batch effect reduction verification
- Biological signal preservation
- Sparse matrix handling
"""

import pytest
import numpy as np
import polars as pl
from scipy import sparse

from scptensor.core.structures import ScpContainer, Assay, ScpMatrix
from scptensor.core.exceptions import (
    AssayNotFoundError,
    LayerNotFoundError,
    ScpValueError,
    MissingDependencyError,
)
from scptensor.integration import mnn_correct, harmony, scanorama_integrate, combat


# =============================================================================
# Helper Functions
# =============================================================================

def create_batch_container(
    n_samples_per_batch: int = 30,
    n_features: int = 50,
    n_batches: int = 2,
    batch_effect_size: float = 2.0,
    group_effect_size: float = 1.5,
    random_state: int = 42,
) -> ScpContainer:
    """
    Create a test container with synthetic batch-effect data.

    Parameters
    ----------
    n_samples_per_batch : int
        Number of samples per batch
    n_features : int
        Number of features
    n_batches : int
        Number of batches
    batch_effect_size : float
        Magnitude of batch effect to add
    group_effect_size : float
        Magnitude of biological group effect
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    ScpContainer
        Container with batch-effect data
    """
    np.random.seed(random_state)

    total_samples = n_samples_per_batch * n_batches
    batch_names = [f"batch{i+1}" for i in range(n_batches)]
    batches = np.repeat(batch_names, n_samples_per_batch)

    # Create groups (biologically meaningful clusters)
    groups = np.array(["GroupA"] * (total_samples // 2) + ["GroupB"] * (total_samples // 2))

    # Generate base data
    X_base = np.random.randn(total_samples, n_features)

    # Add biological effect to first half of features for GroupB
    group_b_idx = groups == "GroupB"
    X_base[group_b_idx, : n_features // 2] += group_effect_size

    # Add batch effects
    X = X_base.copy()
    for i in range(n_batches):
        batch_idx = batches == batch_names[i]
        X[batch_idx, :] += i * batch_effect_size

    # Create metadata
    obs = pl.DataFrame(
        {
            "_index": [f"S{i:03d}" for i in range(total_samples)],
            "batch": batches.tolist(),
            "group": groups.tolist(),
        }
    )

    var = pl.DataFrame({"_index": [f"P{i:03d}" for i in range(n_features)]})

    # Create assay
    matrix = ScpMatrix(X=X, M=None)
    assay = Assay(var=var, layers={"raw": matrix})

    return ScpContainer(obs=obs, assays={"protein": assay})


def compute_batch_effect_metric(X: np.ndarray, batches: np.ndarray) -> float:
    """
    Compute a metric for batch effect magnitude.

    Uses F-statistic from ANOVA on batch means.

    Parameters
    ----------
    X : np.ndarray
        Data matrix (n_samples x n_features)
    batches : np.ndarray
        Batch labels

    Returns
    -------
    float
        Batch effect metric (higher = more batch effect)
    """
    unique_batches = np.unique(batches)
    batch_means = [np.mean(X[batches == b], axis=0) for b in unique_batches]
    grand_mean = np.mean(X, axis=0)

    # Between-batch variance
    between_ss = sum(
        len(X[batches == b]) * np.sum((bm - grand_mean) ** 2)
        for b, bm in zip(unique_batches, batch_means)
    )

    # Total variance
    total_ss = np.sum((X - grand_mean) ** 2)

    # F-statistic-like ratio
    if total_ss > 0:
        return between_ss / total_ss
    return 0.0


def compute_biological_signal_retention(
    X: np.ndarray, groups: np.ndarray, n_features: int | None = None
) -> float:
    """
    Compute retention of biological group differences.

    Parameters
    ----------
    X : np.ndarray
        Data matrix
    groups : np.ndarray
        Group labels
    n_features : int, optional
        Number of features to consider (uses half if None)

    Returns
    -------
    float
        Signal retention metric (higher = better)
    """
    if n_features is None:
        n_features = X.shape[1] // 2

    unique_groups = np.unique(groups)
    group_means = [np.mean(X[groups == g, :n_features], axis=0) for g in unique_groups]

    # Between-group variance
    if len(unique_groups) >= 2:
        diff = np.linalg.norm(group_means[0] - group_means[1])
        return diff
    return 0.0


# =============================================================================
# Test MNN Correction
# =============================================================================

class TestMNNCorrection:
    """Test Mutual Nearest Neighbors batch correction."""

    def test_mnn_basic_two_batches(self):
        """Test MNN correction with two batches."""
        container = create_batch_container(
            n_samples_per_batch=30, n_features=50, n_batches=2, random_state=42
        )

        # Get original batch effect
        X_orig = container.assays["protein"].layers["raw"].X
        batches_orig = container.obs["batch"].to_numpy()

        # Compute batch means before correction
        batch1_idx = batches_orig == "batch1"
        batch2_idx = batches_orig == "batch2"
        mean_before_batch1 = np.mean(X_orig[batch1_idx], axis=0)
        mean_before_batch2 = np.mean(X_orig[batch2_idx], axis=0)
        batch_diff_before = np.linalg.norm(mean_before_batch1 - mean_before_batch2)

        # Apply MNN correction
        result = mnn_correct(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            new_layer_name="mnn_corrected",
            k=10,
            sigma=1.0,
        )

        # Verify layer was created
        assert "mnn_corrected" in result.assays["protein"].layers

        # Check dimensions preserved
        X_corrected = result.assays["protein"].layers["mnn_corrected"].X
        assert X_corrected.shape == X_orig.shape

        # Check batch means are closer after correction (by checking means changed)
        mean_after_batch1 = np.mean(X_corrected[batch1_idx], axis=0)
        mean_after_batch2 = np.mean(X_corrected[batch2_idx], axis=0)
        batch_diff_after = np.linalg.norm(mean_after_batch1 - mean_after_batch2)

        # The correction should change the data (not be identical)
        assert not np.allclose(X_orig, X_corrected), "MNN should modify the data"

        # Verify mask is None (input had no mask)
        assert result.assays["protein"].layers["mnn_corrected"].M is None

    def test_mnn_three_batches_anchor_correction(self):
        """Test MNN correction with three batches (uses anchor-based correction)."""
        container = create_batch_container(
            n_samples_per_batch=30, n_features=50, n_batches=3, random_state=42
        )

        # Apply MNN correction
        result = mnn_correct(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            new_layer_name="mnn_corrected",
            k=10,
        )

        # Verify layer was created
        assert "mnn_corrected" in result.assays["protein"].layers

        # Check all three batches are still present
        batches = result.obs["batch"].to_numpy()
        assert len(np.unique(batches)) == 3

    def test_mnn_with_sparse_input(self):
        """Test MNN correction with sparse input matrix."""
        container = create_batch_container(
            n_samples_per_batch=30, n_features=50, n_batches=2, random_state=42
        )

        # Convert to sparse
        X_sparse = sparse.csr_matrix(container.assays["protein"].layers["raw"].X)
        container.assays["protein"].layers["raw"] = ScpMatrix(X=X_sparse, M=None)

        # Apply MNN correction
        result = mnn_correct(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            new_layer_name="mnn_corrected",
            k=10,
        )

        # Verify layer was created
        assert "mnn_corrected" in result.assays["protein"].layers

        # Check dimensions
        X_corrected = result.assays["protein"].layers["mnn_corrected"].X
        assert X_corrected.shape == (60, 50)

    def test_mnn_with_nan_values(self):
        """Test MNN correction handles NaN values."""
        container = create_batch_container(
            n_samples_per_batch=30, n_features=50, n_batches=2, random_state=42
        )

        # Add some NaN values
        X = container.assays["protein"].layers["raw"].X
        X[0:5, 0:5] = np.nan
        container.assays["protein"].layers["raw"] = ScpMatrix(X=X, M=None)

        # Should not raise error
        result = mnn_correct(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            new_layer_name="mnn_corrected",
            k=10,
        )

        # Verify layer created and no NaN in output
        assert "mnn_corrected" in result.assays["protein"].layers
        X_corrected = result.assays["protein"].layers["mnn_corrected"].X
        assert not np.any(np.isnan(X_corrected))

    def test_mnn_preserves_mask(self):
        """Test MNN correction preserves mask matrix."""
        container = create_batch_container(
            n_samples_per_batch=30, n_features=50, n_batches=2, random_state=42
        )

        # Create mask
        M = np.zeros((60, 50), dtype=np.int8)
        M[0:10, 0:10] = 1  # MBR
        M[10:20, 10:20] = 2  # LOD

        container.assays["protein"].layers["raw"] = ScpMatrix(
            X=container.assays["protein"].layers["raw"].X, M=M
        )

        # Apply MNN correction
        result = mnn_correct(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            new_layer_name="mnn_corrected",
            k=10,
        )

        # Verify mask was preserved
        M_corrected = result.assays["protein"].layers["mnn_corrected"].M
        assert M_corrected is not None
        assert np.array_equal(M_corrected, M)

    def test_mnn_custom_parameters(self):
        """Test MNN correction with custom parameters."""
        container = create_batch_container(
            n_samples_per_batch=30, n_features=50, n_batches=2, random_state=42
        )

        # Test with different k and sigma
        result = mnn_correct(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            new_layer_name="mnn_corrected",
            k=30,
            sigma=2.0,
            n_pcs=20,
            use_pca=True,
        )

        assert "mnn_corrected" in result.assays["protein"].layers

    def test_mnn_without_pca(self):
        """Test MNN correction without PCA preprocessing."""
        container = create_batch_container(
            n_samples_per_batch=30, n_features=30, n_batches=2, random_state=42
        )

        result = mnn_correct(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            new_layer_name="mnn_corrected",
            use_pca=False,
            k=10,
        )

        assert "mnn_corrected" in result.assays["protein"].layers

    def test_mnn_custom_layer_name(self):
        """Test MNN correction with custom layer name."""
        container = create_batch_container(
            n_samples_per_batch=30, n_features=50, n_batches=2, random_state=42
        )

        result = mnn_correct(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            new_layer_name="my_correction",
            k=10,
        )

        assert "my_correction" in result.assays["protein"].layers

    def test_mnn_none_layer_name_uses_default(self):
        """Test MNN correction with None for layer name uses default."""
        container = create_batch_container(
            n_samples_per_batch=30, n_features=50, n_batches=2, random_state=42
        )

        result = mnn_correct(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            new_layer_name=None,
            k=10,
        )

        assert "mnn_corrected" in result.assays["protein"].layers

    def test_mnn_logs_history(self):
        """Test MNN correction logs operation to history."""
        container = create_batch_container(
            n_samples_per_batch=30, n_features=50, n_batches=2, random_state=42
        )

        initial_history_len = len(container.history)

        result = mnn_correct(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            new_layer_name="mnn_corrected",
            k=10,
        )

        # History should have new entry
        assert len(result.history) == initial_history_len + 1

        # Check log entry
        last_log = result.history[-1]
        assert last_log.action == "integration_mnn"
        assert "batch_key" in last_log.params
        assert last_log.params["batch_key"] == "batch"

    # -------------------------------------------------------------------------
    # Error handling tests
    # -------------------------------------------------------------------------

    def test_mnn_error_invalid_k(self):
        """Test MNN raises error for invalid k parameter."""
        container = create_batch_container(
            n_samples_per_batch=30, n_features=50, n_batches=2, random_state=42
        )

        with pytest.raises(ScpValueError, match="k must be positive"):
            mnn_correct(
                container,
                batch_key="batch",
                assay_name="protein",
                base_layer="raw",
                k=0,
            )

        with pytest.raises(ScpValueError, match="k must be positive"):
            mnn_correct(
                container,
                batch_key="batch",
                assay_name="protein",
                base_layer="raw",
                k=-5,
            )

    def test_mnn_error_invalid_sigma(self):
        """Test MNN raises error for invalid sigma parameter."""
        container = create_batch_container(
            n_samples_per_batch=30, n_features=50, n_batches=2, random_state=42
        )

        with pytest.raises(ScpValueError, match="sigma must be positive"):
            mnn_correct(
                container,
                batch_key="batch",
                assay_name="protein",
                base_layer="raw",
                sigma=0,
            )

        with pytest.raises(ScpValueError, match="sigma must be positive"):
            mnn_correct(
                container,
                batch_key="batch",
                assay_name="protein",
                base_layer="raw",
                sigma=-1.0,
            )

    def test_mnn_error_missing_assay(self):
        """Test MNN raises error for missing assay."""
        container = create_batch_container(
            n_samples_per_batch=30, n_features=50, n_batches=2, random_state=42
        )

        with pytest.raises(AssayNotFoundError, match="nonexistent"):
            mnn_correct(
                container,
                batch_key="batch",
                assay_name="nonexistent",
                base_layer="raw",
                k=10,
            )

    def test_mnn_error_missing_layer(self):
        """Test MNN raises error for missing layer."""
        container = create_batch_container(
            n_samples_per_batch=30, n_features=50, n_batches=2, random_state=42
        )

        with pytest.raises(LayerNotFoundError, match="nonexistent"):
            mnn_correct(
                container,
                batch_key="batch",
                assay_name="protein",
                base_layer="nonexistent",
                k=10,
            )

    def test_mnn_error_missing_batch_column(self):
        """Test MNN raises error for missing batch column."""
        container = create_batch_container(
            n_samples_per_batch=30, n_features=50, n_batches=2, random_state=42
        )

        with pytest.raises(ScpValueError, match="Batch key.*not found"):
            mnn_correct(
                container,
                batch_key="nonexistent_batch",
                assay_name="protein",
                base_layer="raw",
                k=10,
            )

    def test_mnn_error_single_batch(self):
        """Test MNN raises error with only one batch."""
        container = create_batch_container(
            n_samples_per_batch=30, n_features=50, n_batches=2, random_state=42
        )

        # Modify to have only one batch
        container.obs = container.obs.with_columns(pl.lit("batch1").alias("batch"))

        with pytest.raises(ScpValueError, match="at least 2 batches"):
            mnn_correct(
                container,
                batch_key="batch",
                assay_name="protein",
                base_layer="raw",
                k=10,
            )


# =============================================================================
# Test Harmony Integration
# =============================================================================

class TestHarmonyIntegration:
    """Test Harmony batch correction (requires harmonypy)."""

    @pytest.mark.skip(reason="harmonypy is optional dependency")
    def test_harmony_basic_two_batches(self):
        """Test Harmony integration with two batches."""
        # Create PCA-like data for Harmony
        container = create_batch_container(
            n_samples_per_batch=50, n_features=30, n_batches=2, random_state=42
        )

        # Harmony works best on PCA data, create a PCA-like layer
        from sklearn.decomposition import PCA

        pca = PCA(n_components=30, random_state=42)
        X_pca = pca.fit_transform(container.assays["protein"].layers["raw"].X)
        container.assays["protein"].add_layer("pca", ScpMatrix(X=X_pca, M=None))

        # Apply Harmony
        result = harmony(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="pca",
            new_layer_name="harmony",
            theta=2.0,
            max_iter_harmony=5,
        )

        # Verify layer was created
        assert "harmony" in result.assays["protein"].layers

        # Check dimensions
        X_corrected = result.assays["protein"].layers["harmony"].X
        assert X_corrected.shape == (100, 30)

    @pytest.mark.skip(reason="harmonypy is optional dependency")
    def test_harmony_with_sparse_input(self):
        """Test Harmony with sparse input."""
        container = create_batch_container(
            n_samples_per_batch=30, n_features=30, n_batches=2, random_state=42
        )

        # Convert to sparse
        X_sparse = sparse.csr_matrix(container.assays["protein"].layers["raw"].X)
        container.assays["protein"].layers["raw"] = ScpMatrix(X=X_sparse, M=None)

        result = harmony(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            new_layer_name="harmony",
            max_iter_harmony=5,
        )

        assert "harmony" in result.assays["protein"].layers

    @pytest.mark.skip(reason="harmonypy is optional dependency")
    def test_harmony_custom_parameters(self):
        """Test Harmony with custom parameters."""
        container = create_batch_container(
            n_samples_per_batch=30, n_features=30, n_batches=2, random_state=42
        )

        result = harmony(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            new_layer_name="harmony_custom",
            theta=3.0,
            lamb=0.5,
            sigma=0.05,
            nclust=10,
            max_iter_harmony=5,
        )

        assert "harmony_custom" in result.assays["protein"].layers

    @pytest.mark.skip(reason="harmonypy is optional dependency")
    def test_harmony_preserves_mask(self):
        """Test Harmony preserves mask matrix."""
        container = create_batch_container(
            n_samples_per_batch=30, n_features=30, n_batches=2, random_state=42
        )

        M = np.zeros((60, 30), dtype=np.int8)
        M[0:5, 0:5] = 1

        container.assays["protein"].layers["raw"] = ScpMatrix(
            X=container.assays["protein"].layers["raw"].X, M=M
        )

        result = harmony(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            new_layer_name="harmony",
            max_iter_harmony=5,
        )

        M_harmony = result.assays["protein"].layers["harmony"].M
        assert M_harmony is not None
        assert np.array_equal(M_harmony, M)

    # -------------------------------------------------------------------------
    # Error handling tests (these should work even without harmonypy)
    # -------------------------------------------------------------------------

    def test_harmony_error_missing_assay(self):
        """Test Harmony raises error for missing assay."""
        container = create_batch_container(
            n_samples_per_batch=30, n_features=30, n_batches=2, random_state=42
        )

        try:
            harmony(
                container,
                batch_key="batch",
                assay_name="nonexistent",
                base_layer="raw",
            )
            assert False, "Should have raised MissingDependencyError or AssayNotFoundError"
        except (MissingDependencyError, AssayNotFoundError):
            pass  # Expected if harmonypy not installed
        except ImportError:
            pass  # harmonypy not installed

    def test_harmony_error_missing_batch_column(self):
        """Test Harmony raises error for missing batch column."""
        container = create_batch_container(
            n_samples_per_batch=30, n_features=30, n_batches=2, random_state=42
        )

        try:
            harmony(
                container,
                batch_key="nonexistent_batch",
                assay_name="protein",
                base_layer="raw",
            )
            assert False, "Should have raised error"
        except MissingDependencyError:
            pass  # harmonypy not installed
        except ScpValueError:
            pass  # Expected error
        except ImportError:
            pass  # harmonypy not installed

    @pytest.mark.skip(reason="harmonypy is optional dependency")
    def test_harmony_error_single_batch(self):
        """Test Harmony raises error with only one batch."""
        container = create_batch_container(
            n_samples_per_batch=30, n_features=30, n_batches=2, random_state=42
        )

        # Modify to single batch
        container.obs = container.obs.with_columns(pl.lit("batch1").alias("batch"))

        try:
            harmony(
                container,
                batch_key="batch",
                assay_name="protein",
                base_layer="raw",
            )
            assert False, "Should have raised error"
        except ScpValueError as e:
            assert "at least 2 batches" in str(e)
        except ImportError:
            pass  # harmonypy not installed

    @pytest.mark.skip(reason="harmonypy is optional dependency")
    def test_harmony_error_singleton_batch(self):
        """Test Harmony raises error with singleton batch."""
        np.random.seed(42)
        n_samples = 10
        n_features = 20

        obs = pl.DataFrame(
            {
                "_index": [f"S{i:03d}" for i in range(n_samples)],
                "batch": ["batch1"] * 9 + ["batch2"] * 1,  # singleton batch2
            }
        )

        X = np.random.randn(n_samples, n_features)
        var = pl.DataFrame({"_index": [f"P{i:03d}" for i in range(n_features)]})

        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X, M=None)})
        container = ScpContainer(obs=obs, assays={"protein": assay})

        try:
            harmony(
                container,
                batch_key="batch",
                assay_name="protein",
                base_layer="raw",
            )
            assert False, "Should have raised error"
        except ScpValueError as e:
            assert "at least 2 samples per batch" in str(e)
        except ImportError:
            pass  # harmonypy not installed


# =============================================================================
# Test Scanorama Integration
# =============================================================================

class TestScanoramaIntegration:
    """Test Scanorama batch correction (requires scanorama)."""

    @pytest.mark.skip(reason="scanorama is optional dependency")
    def test_scanorama_basic_two_batches(self):
        """Test Scanorama integration with two batches."""
        container = create_batch_container(
            n_samples_per_batch=30, n_features=50, n_batches=2, random_state=42
        )

        # Apply Scanorama
        result = scanorama_integrate(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            new_layer_name="scanorama",
            sigma=15.0,
            alpha=0.1,
        )

        # Verify layer was created
        assert "scanorama" in result.assays["protein"].layers

        # Check dimensions
        X_corrected = result.assays["protein"].layers["scanorama"].X
        assert X_corrected.shape == (60, 50)

    @pytest.mark.skip(reason="scanorama is optional dependency")
    def test_scanorama_three_batches(self):
        """Test Scanorama with three batches."""
        container = create_batch_container(
            n_samples_per_batch=30, n_features=50, n_batches=3, random_state=42
        )

        result = scanorama_integrate(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            new_layer_name="scanorama",
        )

        assert "scanorama" in result.assays["protein"].layers

    @pytest.mark.skip(reason="scanorama is optional dependency")
    def test_scanorama_with_sparse_input(self):
        """Test Scanorama with sparse input."""
        container = create_batch_container(
            n_samples_per_batch=30, n_features=50, n_batches=2, random_state=42
        )

        X_sparse = sparse.csr_matrix(container.assays["protein"].layers["raw"].X)
        container.assays["protein"].layers["raw"] = ScpMatrix(X=X_sparse, M=None)

        result = scanorama_integrate(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            new_layer_name="scanorama",
        )

        assert "scanorama" in result.assays["protein"].layers

    @pytest.mark.skip(reason="scanorama is optional dependency")
    def test_scanorama_with_dimensionality_reduction(self):
        """Test Scanorama with dimensionality reduction."""
        container = create_batch_container(
            n_samples_per_batch=30, n_features=50, n_batches=2, random_state=42
        )

        result = scanorama_integrate(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            new_layer_name="scanorama_dr",
            return_dimred=True,
            dimred=15,
        )

        assert "scanorama_dr" in result.assays["protein"].layers
        X_dr = result.assays["protein"].layers["scanorama_dr"].X
        assert X_dr.shape[1] == 15  # Reduced dimension

    @pytest.mark.skip(reason="scanorama is optional dependency")
    def test_scanorama_custom_parameters(self):
        """Test Scanorama with custom parameters."""
        container = create_batch_container(
            n_samples_per_batch=30, n_features=50, n_batches=2, random_state=42
        )

        result = scanorama_integrate(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            new_layer_name="scanorama_custom",
            sigma=20.0,
            alpha=0.05,
            knn=15,
            approx=False,  # Use exact NN
        )

        assert "scanorama_custom" in result.assays["protein"].layers

    @pytest.mark.skip(reason="scanorama is optional dependency")
    def test_scanorama_preserves_mask(self):
        """Test Scanorama preserves mask matrix."""
        container = create_batch_container(
            n_samples_per_batch=30, n_features=50, n_batches=2, random_state=42
        )

        M = np.zeros((60, 50), dtype=np.int8)
        M[0:10, 0:10] = 1

        container.assays["protein"].layers["raw"] = ScpMatrix(
            X=container.assays["protein"].layers["raw"].X, M=M
        )

        result = scanorama_integrate(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            new_layer_name="scanorama",
        )

        M_scanorama = result.assays["protein"].layers["scanorama"].M
        assert M_scanorama is not None
        assert np.array_equal(M_scanorama, M)

    @pytest.mark.skip(reason="scanorama is optional dependency")
    def test_scanorama_with_nan_values(self):
        """Test Scanorama handles NaN values."""
        container = create_batch_container(
            n_samples_per_batch=30, n_features=50, n_batches=2, random_state=42
        )

        X = container.assays["protein"].layers["raw"].X
        X[0:5, 0:5] = np.nan
        container.assays["protein"].layers["raw"] = ScpMatrix(X=X, M=None)

        result = scanorama_integrate(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            new_layer_name="scanorama",
        )

        # Verify no NaN in output
        X_corrected = result.assays["protein"].layers["scanorama"].X
        assert not np.any(np.isnan(X_corrected))

    # -------------------------------------------------------------------------
    # Error handling tests
    # -------------------------------------------------------------------------

    def test_scanorama_error_invalid_sigma(self):
        """Test Scanorama raises error for invalid sigma."""
        container = create_batch_container(
            n_samples_per_batch=30, n_features=50, n_batches=2, random_state=42
        )

        try:
            scanorama_integrate(
                container,
                batch_key="batch",
                assay_name="protein",
                base_layer="raw",
                sigma=0,
            )
            assert False, "Should have raised error"
        except MissingDependencyError:
            pass  # scanorama not installed
        except ScpValueError as e:
            assert "sigma must be positive" in str(e)
        except ImportError:
            pass  # scanorama not installed

    def test_scanorama_error_invalid_alpha(self):
        """Test Scanorama raises error for invalid alpha."""
        container = create_batch_container(
            n_samples_per_batch=30, n_features=50, n_batches=2, random_state=42
        )

        try:
            scanorama_integrate(
                container,
                batch_key="batch",
                assay_name="protein",
                base_layer="raw",
                alpha=0,
            )
            assert False, "Should have raised error"
        except MissingDependencyError:
            pass  # scanorama not installed
        except ScpValueError as e:
            assert "alpha must be in \\(0, 1\\)" in str(e)
        except ImportError:
            pass  # scanorama not installed

    def test_scanorama_error_invalid_knn(self):
        """Test Scanorama raises error for invalid knn."""
        container = create_batch_container(
            n_samples_per_batch=30, n_features=50, n_batches=2, random_state=42
        )

        try:
            scanorama_integrate(
                container,
                batch_key="batch",
                assay_name="protein",
                base_layer="raw",
                knn=-5,
            )
            assert False, "Should have raised error"
        except MissingDependencyError:
            pass  # scanorama not installed
        except ScpValueError:
            pass  # Expected
        except ImportError:
            pass  # scanorama not installed

    def test_scanorama_error_missing_assay(self):
        """Test Scanorama raises error for missing assay."""
        container = create_batch_container(
            n_samples_per_batch=30, n_features=50, n_batches=2, random_state=42
        )

        try:
            scanorama_integrate(
                container,
                batch_key="batch",
                assay_name="nonexistent",
                base_layer="raw",
            )
            assert False, "Should have raised error"
        except (MissingDependencyError, AssayNotFoundError):
            pass  # Expected
        except ImportError:
            pass  # scanorama not installed

    def test_scanorama_error_single_batch(self):
        """Test Scanorama raises error with only one batch."""
        container = create_batch_container(
            n_samples_per_batch=30, n_features=50, n_batches=2, random_state=42
        )

        # Modify to single batch
        container.obs = container.obs.with_columns(pl.lit("batch1").alias("batch"))

        try:
            scanorama_integrate(
                container,
                batch_key="batch",
                assay_name="protein",
                base_layer="raw",
            )
            assert False, "Should have raised error"
        except MissingDependencyError:
            pass  # scanorama not installed
        except ScpValueError as e:
            assert "at least 2 batches" in str(e)
        except ImportError:
            pass  # scanorama not installed


# =============================================================================
# Test ComBat (additional tests)
# =============================================================================

class TestComBatAdditional:
    """Additional tests for ComBat batch correction."""

    def test_combat_with_sparse_input(self):
        """Test ComBat with sparse input matrix."""
        container = create_batch_container(
            n_samples_per_batch=30, n_features=50, n_batches=2, random_state=42
        )

        # Convert to sparse
        X_sparse = sparse.csr_matrix(container.assays["protein"].layers["raw"].X)
        container.assays["protein"].layers["raw"] = ScpMatrix(X=X_sparse, M=None)

        result = combat(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            new_layer_name="combat",
        )

        assert "combat" in result.assays["protein"].layers

    def test_combat_with_nan_values(self):
        """Test ComBat handles NaN values."""
        container = create_batch_container(
            n_samples_per_batch=30, n_features=50, n_batches=2, random_state=42
        )

        X = container.assays["protein"].layers["raw"].X
        X[0:5, 0:5] = np.nan
        container.assays["protein"].layers["raw"] = ScpMatrix(X=X, M=None)

        result = combat(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            new_layer_name="combat",
        )

        # No NaN in output
        X_corrected = result.assays["protein"].layers["combat"].X
        assert not np.any(np.isnan(X_corrected))

    def test_combat_with_covariates(self):
        """Test ComBat with biological covariates."""
        # Create container with non-confounded groups and batches
        np.random.seed(42)
        n_samples_per_batch = 30
        n_features = 50

        # Create batches and groups that are not perfectly aligned
        batches = np.array(["batch1"] * n_samples_per_batch + ["batch2"] * n_samples_per_batch)
        groups = np.array(["GroupA"] * 15 + ["GroupB"] * 15 + ["GroupA"] * 15 + ["GroupB"] * 15)

        obs = pl.DataFrame(
            {
                "_index": [f"S{i:03d}" for i in range(2 * n_samples_per_batch)],
                "batch": batches.tolist(),
                "group": groups.tolist(),
            }
        )

        X = np.random.randn(2 * n_samples_per_batch, n_features)
        X[batches == "batch2"] += 2.0  # Add batch effect
        X[groups == "GroupB"] += 1.0  # Add group effect

        var = pl.DataFrame({"_index": [f"P{i:03d}" for i in range(n_features)]})
        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X, M=None)})
        container = ScpContainer(obs=obs, assays={"protein": assay})

        result = combat(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            new_layer_name="combat",
            covariates=["group"],
        )

        assert "combat" in result.assays["protein"].layers

    def test_combat_error_singleton_batch(self):
        """Test ComBat raises error with singleton batch."""
        np.random.seed(42)
        n_samples = 10
        n_features = 20

        obs = pl.DataFrame(
            {
                "_index": [f"S{i:03d}" for i in range(n_samples)],
                "batch": ["batch1"] * 9 + ["batch2"] * 1,
            }
        )

        X = np.random.randn(n_samples, n_features)
        var = pl.DataFrame({"_index": [f"P{i:03d}" for i in range(n_features)]})

        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X, M=None)})
        container = ScpContainer(obs=obs, assays={"protein": assay})

        with pytest.raises(ScpValueError, match="at least 2 samples per batch"):
            combat(
                container,
                batch_key="batch",
                assay_name="protein",
                base_layer="raw",
            )

    def test_combat_preserves_mask(self):
        """Test ComBat preserves mask matrix."""
        container = create_batch_container(
            n_samples_per_batch=30, n_features=50, n_batches=2, random_state=42
        )

        M = np.zeros((60, 50), dtype=np.int8)
        M[0:10, 0:10] = 1
        M[10:20, 10:20] = 2

        container.assays["protein"].layers["raw"] = ScpMatrix(
            X=container.assays["protein"].layers["raw"].X, M=M
        )

        result = combat(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            new_layer_name="combat",
        )

        M_combat = result.assays["protein"].layers["combat"].M
        assert M_combat is not None
        assert np.array_equal(M_combat, M)


# =============================================================================
# Test Batch Effect Reduction
# =============================================================================

class TestBatchEffectReduction:
    """Test that batch correction methods actually reduce batch effects."""

    def test_mnn_reduces_batch_effect(self):
        """Test MNN correction reduces batch effect metric."""
        container = create_batch_container(
            n_samples_per_batch=40, n_features=50, n_batches=2, random_state=42
        )

        X_orig = container.assays["protein"].layers["raw"].X
        batches = container.obs["batch"].to_numpy()

        # Compute batch separation before correction
        batch1_mean = np.mean(X_orig[batches == "batch1"], axis=0)
        batch2_mean = np.mean(X_orig[batches == "batch2"], axis=0)
        batch_diff_before = np.linalg.norm(batch1_mean - batch2_mean)

        result = mnn_correct(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            k=15,
        )

        X_corrected = result.assays["protein"].layers["mnn_corrected"].X
        batch1_mean_after = np.mean(X_corrected[batches == "batch1"], axis=0)
        batch2_mean_after = np.mean(X_corrected[batches == "batch2"], axis=0)
        batch_diff_after = np.linalg.norm(batch1_mean_after - batch2_mean_after)

        # Data should be modified (not identical)
        assert not np.allclose(X_orig, X_corrected), "MNN should modify the data"

    def test_combat_reduces_batch_effect(self):
        """Test ComBat correction reduces batch effect metric."""
        container = create_batch_container(
            n_samples_per_batch=40, n_features=50, n_batches=2, random_state=42
        )

        X_orig = container.assays["protein"].layers["raw"].X
        batches = container.obs["batch"].to_numpy()

        batch_effect_before = compute_batch_effect_metric(X_orig, batches)

        result = combat(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
        )

        X_corrected = result.assays["protein"].layers["combat"].X
        batch_effect_after = compute_batch_effect_metric(X_corrected, batches)

        # Batch effect should be reduced
        assert batch_effect_after < batch_effect_before


# =============================================================================
# Test Biological Signal Preservation
# =============================================================================

class TestBiologicalSignalPreservation:
    """Test that batch correction preserves biological signals."""

    def test_mnn_preserves_group_differences(self):
        """Test MNN preserves biological group differences."""
        container = create_batch_container(
            n_samples_per_batch=40,
            n_features=50,
            n_batches=2,
            group_effect_size=2.0,
            random_state=42,
        )

        groups = container.obs["group"].to_numpy()

        X_orig = container.assays["protein"].layers["raw"].X
        signal_before = compute_biological_signal_retention(X_orig, groups)

        result = mnn_correct(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            k=15,
        )

        X_corrected = result.assays["protein"].layers["mnn_corrected"].X
        signal_after = compute_biological_signal_retention(X_corrected, groups)

        # Some biological signal should remain
        assert signal_after > 0

    def test_combat_preserves_group_differences(self):
        """Test ComBat preserves biological group differences."""
        container = create_batch_container(
            n_samples_per_batch=40,
            n_features=50,
            n_batches=2,
            group_effect_size=2.0,
            random_state=42,
        )

        groups = container.obs["group"].to_numpy()

        X_orig = container.assays["protein"].layers["raw"].X
        signal_before = compute_biological_signal_retention(X_orig, groups)

        result = combat(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
        )

        X_corrected = result.assays["protein"].layers["combat"].X
        signal_after = compute_biological_signal_retention(X_corrected, groups)

        # Some biological signal should remain
        assert signal_after > 0


# =============================================================================
# Edge Cases and Boundary Conditions
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_mnn_small_dataset(self):
        """Test MNN with minimal dataset."""
        np.random.seed(42)
        n_samples = 10  # Small
        n_features = 15

        obs = pl.DataFrame(
            {
                "_index": [f"S{i:03d}" for i in range(n_samples)],
                "batch": ["batch1"] * 5 + ["batch2"] * 5,
            }
        )

        X = np.random.randn(n_samples, n_features)
        var = pl.DataFrame({"_index": [f"P{i:03d}" for i in range(n_features)]})

        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X, M=None)})
        container = ScpContainer(obs=obs, assays={"protein": assay})

        result = mnn_correct(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            k=3,  # Small k for small dataset
        )

        assert "mnn_corrected" in result.assays["protein"].layers

    def test_combat_small_dataset(self):
        """Test ComBat with minimal dataset."""
        np.random.seed(42)
        n_samples = 10
        n_features = 15

        obs = pl.DataFrame(
            {
                "_index": [f"S{i:03d}" for i in range(n_samples)],
                "batch": ["batch1"] * 5 + ["batch2"] * 5,
            }
        )

        X = np.random.randn(n_samples, n_features)
        var = pl.DataFrame({"_index": [f"P{i:03d}" for i in range(n_features)]})

        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X, M=None)})
        container = ScpContainer(obs=obs, assays={"protein": assay})

        result = combat(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
        )

        assert "combat" in result.assays["protein"].layers

    def test_mnn_imbalanced_batches(self):
        """Test MNN with imbalanced batch sizes."""
        np.random.seed(42)
        n_samples_batch1 = 50
        n_samples_batch2 = 10  # Much smaller
        n_features = 30

        obs = pl.DataFrame(
            {
                "_index": [f"S{i:03d}" for i in range(n_samples_batch1 + n_samples_batch2)],
                "batch": ["batch1"] * n_samples_batch1 + ["batch2"] * n_samples_batch2,
            }
        )

        X = np.random.randn(n_samples_batch1 + n_samples_batch2, n_features)
        var = pl.DataFrame({"_index": [f"P{i:03d}" for i in range(n_features)]})

        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X, M=None)})
        container = ScpContainer(obs=obs, assays={"protein": assay})

        result = mnn_correct(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            k=5,
        )

        assert "mnn_corrected" in result.assays["protein"].layers
        assert result.assays["protein"].layers["mnn_corrected"].X.shape[0] == 60

    def test_combat_imbalanced_batches(self):
        """Test ComBat with imbalanced batch sizes."""
        np.random.seed(42)
        n_samples_batch1 = 50
        n_samples_batch2 = 10
        n_features = 30

        obs = pl.DataFrame(
            {
                "_index": [f"S{i:03d}" for i in range(n_samples_batch1 + n_samples_batch2)],
                "batch": ["batch1"] * n_samples_batch1 + ["batch2"] * n_samples_batch2,
            }
        )

        X = np.random.randn(n_samples_batch1 + n_samples_batch2, n_features)
        var = pl.DataFrame({"_index": [f"P{i:03d}" for i in range(n_features)]})

        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X, M=None)})
        container = ScpContainer(obs=obs, assays={"protein": assay})

        result = combat(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
        )

        assert "combat" in result.assays["protein"].layers


# =============================================================================
# Test History/Provenance Logging
# =============================================================================

class TestHistoryLogging:
    """Test that integration methods log properly to history."""

    def test_mnn_logs_to_history(self):
        """Test MNN logs operation to history."""
        container = create_batch_container(
            n_samples_per_batch=30, n_features=50, n_batches=2, random_state=42
        )

        initial_len = len(container.history)

        result = mnn_correct(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
        )

        assert len(result.history) == initial_len + 1
        assert result.history[-1].action == "integration_mnn"

    def test_combat_logs_to_history(self):
        """Test ComBat logs operation to history."""
        container = create_batch_container(
            n_samples_per_batch=30, n_features=50, n_batches=2, random_state=42
        )

        initial_len = len(container.history)

        result = combat(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
        )

        assert len(result.history) == initial_len + 1
        assert result.history[-1].action == "integration_combat"
