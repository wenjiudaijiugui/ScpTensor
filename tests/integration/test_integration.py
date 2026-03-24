"""Tests for integration (batch correction) module.

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

import importlib.util
import sys
import types

import numpy as np
import polars as pl
import pytest
from scipy import sparse

from scptensor.core.exceptions import (
    AssayNotFoundError,
    LayerNotFoundError,
    MissingDependencyError,
    ScpValueError,
    ValidationError,
)
from scptensor.core.structures import Assay, ScpContainer, ScpMatrix
from scptensor.integration import (
    integrate_combat as combat,
)
from scptensor.integration import (
    integrate_harmony as harmony,
)
from scptensor.integration import (
    integrate_limma as limma_correct,
)
from scptensor.integration import (
    integrate_mnn as mnn_correct,
)
from scptensor.integration import (
    integrate_none,
)
from scptensor.integration import (
    integrate_scanorama as scanorama_integrate,
)
from scptensor.integration.base import get_integrate_method_info
from scptensor.integration.combat import _solve_eb
from scptensor.integration.mnn import (
    _adjust_shift_variance,
    _compute_smoothed_correction,
    _subtract_biological_components,
)

HARMONYPY_AVAILABLE = importlib.util.find_spec("harmonypy") is not None
SCANORAMA_AVAILABLE = importlib.util.find_spec("scanorama") is not None

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
    """Create a test container with synthetic batch-effect data.

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
    batch_names = [f"batch{i + 1}" for i in range(n_batches)]
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
        },
    )

    var = pl.DataFrame({"_index": [f"P{i:03d}" for i in range(n_features)]})

    # Create assay
    matrix = ScpMatrix(X=X, M=None)
    assay = Assay(var=var, layers={"raw": matrix})

    return ScpContainer(obs=obs, assays={"protein": assay})


def add_pca_assay(
    container: ScpContainer,
    n_components: int = 10,
) -> np.ndarray:
    """Add a PCA embedding assay that Harmony can consume via assay='pca', layer='X'."""
    from sklearn.decomposition import PCA

    X = container.assays["protein"].layers["raw"].X
    if sparse.issparse(X):
        X = X.toarray()

    n_components = min(n_components, X.shape[0], X.shape[1])
    X_pca = PCA(n_components=n_components, random_state=42).fit_transform(X)
    M_pca = np.zeros(X_pca.shape, dtype=np.int8)

    container.assays["pca"] = Assay(
        var=pl.DataFrame({"pc_name": [f"PC{i + 1}" for i in range(X_pca.shape[1])]}),
        layers={"X": ScpMatrix(X=X_pca.copy(), M=M_pca.copy())},
        feature_id_col="pc_name",
    )
    return X_pca


def add_pca_layer_to_protein_assay(container: ScpContainer) -> np.ndarray:
    """Add a same-width PCA-like layer so protein/pca stays assay-compatible."""
    from sklearn.decomposition import PCA

    X = container.assays["protein"].layers["raw"].X
    if sparse.issparse(X):
        X = X.toarray()

    n_components = X.shape[1]
    X_pca = PCA(n_components=n_components, random_state=42).fit_transform(X)
    M_pca = np.zeros(X_pca.shape, dtype=np.int8)
    container.assays["protein"].add_layer("pca", ScpMatrix(X=X_pca, M=M_pca))
    return X_pca


def add_pca_inputs(
    container: ScpContainer,
    n_components: int = 10,
) -> np.ndarray:
    """Add both `pca/X` and `protein/pca` Harmony-compatible inputs."""
    add_pca_assay(container, n_components=n_components)
    return add_pca_layer_to_protein_assay(container)


def compute_batch_effect_metric(X: np.ndarray, batches: np.ndarray) -> float:
    """Compute a metric for batch effect magnitude.

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
        for b, bm in zip(unique_batches, batch_means, strict=False)
    )

    # Total variance
    total_ss = np.sum((X - grand_mean) ** 2)

    # F-statistic-like ratio
    if total_ss > 0:
        return between_ss / total_ss
    return 0.0


def compute_biological_signal_retention(
    X: np.ndarray,
    groups: np.ndarray,
    n_features: int | None = None,
) -> float:
    """Compute retention of biological group differences.

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

    def test_mnn_subtracts_biological_components_from_correction(self):
        """svd_dim helper should remove correction components parallel to biological spans."""
        correction = np.array([[1.0, 1.0], [2.0, -3.0]], dtype=np.float64)
        span_x = np.array([[1.0], [0.0]], dtype=np.float64)
        span_y = np.array([[0.0], [1.0]], dtype=np.float64)

        adjusted = _subtract_biological_components(correction, span_x, span_y)
        np.testing.assert_allclose(adjusted, np.zeros_like(correction))

    def test_mnn_variance_adjustment_matches_reference_formula(self):
        """Variance adjustment should match the batchelor test reference implementation."""
        rng = np.random.default_rng(100032)
        reference = rng.normal(scale=0.1, size=(40, 25))
        target = rng.normal(scale=0.1, size=(100, 25))
        correction = rng.random(size=target.shape)

        def ref_adjust_shift_variance(
            data1: np.ndarray,
            data2: np.ndarray,
            cell_vect: np.ndarray,
            sigma: float,
        ) -> np.ndarray:
            scaling = np.empty(cell_vect.shape[0], dtype=np.float64)
            for cell in range(cell_vect.shape[0]):
                cur_cor_vect = cell_vect[cell]
                l2norm = np.sqrt(np.sum(cur_cor_vect**2))
                cur_cor_vect = cur_cor_vect / l2norm
                coords2 = data2 @ cur_cor_vect
                coords1 = data1 @ cur_cor_vect

                dist2 = data2[cell] - data2
                dist2 = dist2 - np.outer(dist2 @ cur_cor_vect, cur_cor_vect)
                dist2 = np.sum(dist2**2, axis=1)
                weight2 = np.exp(-dist2 / sigma)

                dist1 = data2[cell] - data1
                dist1 = dist1 - np.outer(dist1 @ cur_cor_vect, cur_cor_vect)
                dist1 = np.sum(dist1**2, axis=1)
                weight1 = np.exp(-dist1 / sigma)

                rank2 = np.empty(coords2.shape[0], dtype=np.int64)
                rank2[np.argsort(coords2, kind="mergesort")] = np.arange(1, coords2.shape[0] + 1)
                prob2 = np.sum(weight2[rank2 <= rank2[cell]]) / np.sum(weight2)

                ord1 = np.argsort(coords1, kind="mergesort")
                ecdf1 = np.cumsum(weight1[ord1]) / np.sum(weight1)
                quantile_idx = int(np.searchsorted(ecdf1, prob2, side="left"))
                quantile_idx = min(quantile_idx, len(ord1) - 1)
                quan1 = coords1[ord1[quantile_idx]]
                quan2 = coords2[cell]
                scaling[cell] = max((quan1 - quan2) / l2norm, 1.0)

            return cell_vect * scaling[:, None]

        expected = ref_adjust_shift_variance(reference, target, correction, sigma=1.0)
        observed = _adjust_shift_variance(
            reference_data=reference,
            target_data=target,
            correction=correction,
            sigma=1.0,
        )
        np.testing.assert_allclose(observed, expected)

    def test_mnn_smoothed_correction_updates_all_target_cells(self):
        """Full MNN should smooth batch vectors onto all target cells, not only paired ones."""
        reference = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        target = np.array([[10.0, 0.0], [10.0, 1.0], [10.0, 2.0]], dtype=np.float64)

        correction = _compute_smoothed_correction(
            reference_data=reference,
            target_data=target,
            kernel_space=target,
            mnn_pairs=[(0, 0), (1, 1)],
            sigma=1.0,
        )

        assert correction.shape == target.shape
        assert np.allclose(correction[:, 0], -10.0)
        assert not np.allclose(correction[2], np.zeros(2))

    def test_mnn_basic_two_batches(self):
        """Test MNN correction with two batches."""
        container = create_batch_container(
            n_samples_per_batch=30,
            n_features=50,
            n_batches=2,
            random_state=42,
        )

        # Get original batch effect
        X_orig = container.assays["protein"].layers["raw"].X
        batches_orig = container.obs["batch"].to_numpy()

        # Compute batch means before correction
        batch1_idx = batches_orig == "batch1"
        batch2_idx = batches_orig == "batch2"
        mean_before_batch1 = np.mean(X_orig[batch1_idx], axis=0)
        mean_before_batch2 = np.mean(X_orig[batch2_idx], axis=0)
        np.linalg.norm(mean_before_batch1 - mean_before_batch2)

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
        np.linalg.norm(mean_after_batch1 - mean_after_batch2)

        # The correction should change the data (not be identical)
        assert not np.allclose(X_orig, X_corrected), "MNN should modify the data"

        # Verify mask is None (input had no mask)
        assert result.assays["protein"].layers["mnn_corrected"].M is None

    def test_mnn_three_batches_anchor_correction(self):
        """Test multi-batch MNN uses progressive merging by default."""
        container = create_batch_container(
            n_samples_per_batch=30,
            n_features=50,
            n_batches=3,
            random_state=42,
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

    def test_mnn_validates_merge_order(self):
        """Explicit merge order must list each observed batch exactly once."""
        container = create_batch_container(
            n_samples_per_batch=20,
            n_features=20,
            n_batches=3,
            random_state=42,
        )

        with pytest.raises(ScpValueError, match="merge_order must contain each batch exactly once"):
            mnn_correct(
                container,
                batch_key="batch",
                assay_name="protein",
                base_layer="raw",
                merge_order=["batch1", "batch2"],
            )

    def test_mnn_rejects_legacy_pairwise_mode_for_multibatch(self):
        """Regression: multi-batch full MNN no longer supports pairwise-only mode."""
        container = create_batch_container(
            n_samples_per_batch=20,
            n_features=20,
            n_batches=3,
            random_state=42,
        )

        with pytest.raises(ScpValueError, match="Full MNN now uses progressive merging"):
            mnn_correct(
                container,
                batch_key="batch",
                assay_name="protein",
                base_layer="raw",
                use_anchor_correction=False,
            )

    def test_mnn_with_sparse_input(self):
        """Test MNN correction with sparse input matrix."""
        container = create_batch_container(
            n_samples_per_batch=30,
            n_features=50,
            n_batches=2,
            random_state=42,
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
        """Test MNN correction rejects NaN values without explicit imputation."""
        container = create_batch_container(
            n_samples_per_batch=30,
            n_features=50,
            n_batches=2,
            random_state=42,
        )

        # Add some NaN values
        X = container.assays["protein"].layers["raw"].X
        X[0:5, 0:5] = np.nan
        container.assays["protein"].layers["raw"] = ScpMatrix(X=X, M=None)

        with pytest.raises(ScpValueError, match="requires a complete matrix"):
            mnn_correct(
                container,
                batch_key="batch",
                assay_name="protein",
                base_layer="raw",
                new_layer_name="mnn_corrected",
                k=10,
            )

    def test_mnn_with_inf_values(self):
        """Full MNN should reject Inf values under the complete-finite input contract."""
        container = create_batch_container(
            n_samples_per_batch=30,
            n_features=50,
            n_batches=2,
            random_state=42,
        )

        X = container.assays["protein"].layers["raw"].X
        X[0, 0] = np.inf
        container.assays["protein"].layers["raw"] = ScpMatrix(X=X, M=None)

        with pytest.raises(ScpValueError, match="only finite values"):
            mnn_correct(
                container,
                batch_key="batch",
                assay_name="protein",
                base_layer="raw",
                new_layer_name="mnn_corrected",
                k=10,
            )

    def test_mnn_preserves_mask(self):
        """Test MNN correction preserves mask matrix."""
        container = create_batch_container(
            n_samples_per_batch=30,
            n_features=50,
            n_batches=2,
            random_state=42,
        )

        # Create mask
        M = np.zeros((60, 50), dtype=np.int8)
        M[0:10, 0:10] = 1  # MBR
        M[10:20, 10:20] = 2  # LOD

        container.assays["protein"].layers["raw"] = ScpMatrix(
            X=container.assays["protein"].layers["raw"].X,
            M=M,
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
            n_samples_per_batch=30,
            n_features=50,
            n_batches=2,
            random_state=42,
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
            svd_dim=2,
            var_adj=False,
        )

        assert "mnn_corrected" in result.assays["protein"].layers

    def test_mnn_without_pca(self):
        """Test MNN correction without PCA preprocessing."""
        container = create_batch_container(
            n_samples_per_batch=30,
            n_features=30,
            n_batches=2,
            random_state=42,
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
            n_samples_per_batch=30,
            n_features=50,
            n_batches=2,
            random_state=42,
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
            n_samples_per_batch=30,
            n_features=50,
            n_batches=2,
            random_state=42,
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
            n_samples_per_batch=30,
            n_features=50,
            n_batches=2,
            random_state=42,
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
        assert last_log.params["cos_norm_in"] is True
        assert last_log.params["cos_norm_out"] is True
        assert last_log.params["svd_dim"] == 0
        assert last_log.params["var_adj"] is True

    # -------------------------------------------------------------------------
    # Error handling tests
    # -------------------------------------------------------------------------

    def test_mnn_error_invalid_k(self):
        """Test MNN raises error for invalid k parameter."""
        container = create_batch_container(
            n_samples_per_batch=30,
            n_features=50,
            n_batches=2,
            random_state=42,
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
            n_samples_per_batch=30,
            n_features=50,
            n_batches=2,
            random_state=42,
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

    def test_mnn_error_invalid_svd_dim(self):
        """Test MNN validates svd_dim values."""
        container = create_batch_container(
            n_samples_per_batch=30,
            n_features=50,
            n_batches=2,
            random_state=42,
        )

        with pytest.raises(ScpValueError, match="svd_dim must be >= 0"):
            mnn_correct(
                container,
                batch_key="batch",
                assay_name="protein",
                base_layer="raw",
                svd_dim=-1,
            )

    def test_mnn_error_missing_assay(self):
        """Test MNN raises error for missing assay."""
        container = create_batch_container(
            n_samples_per_batch=30,
            n_features=50,
            n_batches=2,
            random_state=42,
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
            n_samples_per_batch=30,
            n_features=50,
            n_batches=2,
            random_state=42,
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
            n_samples_per_batch=30,
            n_features=50,
            n_batches=2,
            random_state=42,
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
            n_samples_per_batch=30,
            n_features=50,
            n_batches=2,
            random_state=42,
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

    @pytest.mark.skipif(not HARMONYPY_AVAILABLE, reason="harmonypy is optional dependency")
    def test_harmony_basic_two_batches(self):
        """Test Harmony integration with two batches."""
        container = create_batch_container(
            n_samples_per_batch=50,
            n_features=30,
            n_batches=2,
            random_state=42,
        )
        add_pca_assay(container, n_components=20)

        # Apply Harmony
        result = harmony(
            container,
            batch_key="batch",
            assay_name="pca",
            base_layer="X",
            new_layer_name="harmony",
            theta=2.0,
            max_iter_harmony=5,
        )

        # Verify layer was created on the embedding assay
        assert "harmony" in result.assays["pca"].layers

        # Check dimensions
        X_corrected = result.assays["pca"].layers["harmony"].X
        assert X_corrected.shape == (100, 20)

    @pytest.mark.skipif(not HARMONYPY_AVAILABLE, reason="harmonypy is optional dependency")
    def test_harmony_with_sparse_input(self):
        """Test Harmony accepts a PCA-like layer on the protein assay."""
        container = create_batch_container(
            n_samples_per_batch=30,
            n_features=30,
            n_batches=2,
            random_state=42,
        )

        # Convert to sparse
        X_sparse = sparse.csr_matrix(container.assays["protein"].layers["raw"].X)
        container.assays["protein"].layers["raw"] = ScpMatrix(X=X_sparse, M=None)
        add_pca_layer_to_protein_assay(container)

        result = harmony(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="pca",
            new_layer_name="harmony",
            max_iter_harmony=5,
        )

        assert "harmony" in result.assays["protein"].layers

    @pytest.mark.skipif(not HARMONYPY_AVAILABLE, reason="harmonypy is optional dependency")
    def test_harmony_custom_parameters(self):
        """Test Harmony with custom parameters."""
        container = create_batch_container(
            n_samples_per_batch=30,
            n_features=30,
            n_batches=2,
            random_state=42,
        )
        add_pca_assay(container, n_components=15)

        result = harmony(
            container,
            batch_key="batch",
            assay_name="pca",
            base_layer="X",
            new_layer_name="harmony_custom",
            theta=3.0,
            lamb=0.5,
            sigma=0.05,
            nclust=10,
            max_iter_harmony=5,
        )

        assert "harmony_custom" in result.assays["pca"].layers

    @pytest.mark.skipif(not HARMONYPY_AVAILABLE, reason="harmonypy is optional dependency")
    def test_harmony_preserves_mask(self):
        """Test Harmony preserves mask matrix."""
        container = create_batch_container(
            n_samples_per_batch=30,
            n_features=30,
            n_batches=2,
            random_state=42,
        )

        X_pca = add_pca_assay(container, n_components=10)
        M = np.zeros(X_pca.shape, dtype=np.int8)
        M[0:5, 0:5] = 1

        container.assays["pca"].layers["X"] = ScpMatrix(X=X_pca, M=M)

        result = harmony(
            container,
            batch_key="batch",
            assay_name="pca",
            base_layer="X",
            new_layer_name="harmony",
            max_iter_harmony=5,
        )

        M_harmony = result.assays["pca"].layers["harmony"].M
        assert M_harmony is not None
        assert np.array_equal(M_harmony, M)

    @pytest.mark.skipif(not HARMONYPY_AVAILABLE, reason="harmonypy is optional dependency")
    def test_harmony_rejects_raw_protein_layer(self):
        """Harmony should reject raw protein matrices and require embeddings."""
        container = create_batch_container(
            n_samples_per_batch=30,
            n_features=30,
            n_batches=2,
            random_state=42,
        )

        with pytest.raises(ScpValueError, match="requires a low-dimensional embedding input"):
            harmony(
                container,
                batch_key="batch",
                assay_name="protein",
                base_layer="raw",
                new_layer_name="harmony",
                max_iter_harmony=5,
            )

    # -------------------------------------------------------------------------
    # Error handling tests (these should work even without harmonypy)
    # -------------------------------------------------------------------------

    def test_harmony_error_missing_assay(self):
        """Test Harmony raises error for missing assay."""
        container = create_batch_container(
            n_samples_per_batch=30,
            n_features=30,
            n_batches=2,
            random_state=42,
        )

        if not HARMONYPY_AVAILABLE:
            pytest.skip("harmonypy is optional dependency")

        with pytest.raises(AssayNotFoundError, match="nonexistent"):
            harmony(
                container,
                batch_key="batch",
                assay_name="nonexistent",
                base_layer="raw",
            )

    def test_harmony_error_missing_batch_column(self):
        """Test Harmony raises error for missing batch column."""
        container = create_batch_container(
            n_samples_per_batch=30,
            n_features=30,
            n_batches=2,
            random_state=42,
        )
        add_pca_assay(container, n_components=10)

        if not HARMONYPY_AVAILABLE:
            pytest.skip("harmonypy is optional dependency")

        with pytest.raises(ScpValueError):
            harmony(
                container,
                batch_key="nonexistent_batch",
                assay_name="pca",
                base_layer="X",
            )

    def test_harmony_error_single_batch(self):
        """Test Harmony raises error with only one batch."""
        container = create_batch_container(
            n_samples_per_batch=30,
            n_features=30,
            n_batches=2,
            random_state=42,
        )
        add_pca_assay(container, n_components=10)

        # Modify to single batch
        container.obs = container.obs.with_columns(pl.lit("batch1").alias("batch"))

        if not HARMONYPY_AVAILABLE:
            pytest.skip("harmonypy is optional dependency")

        with pytest.raises(ScpValueError, match="at least 2 batches"):
            harmony(
                container,
                batch_key="batch",
                assay_name="pca",
                base_layer="X",
            )

    def test_harmony_error_singleton_batch(self):
        """Test Harmony raises error with singleton batch."""
        np.random.seed(42)
        n_samples = 10
        n_features = 20

        obs = pl.DataFrame(
            {
                "_index": [f"S{i:03d}" for i in range(n_samples)],
                "batch": ["batch1"] * 9 + ["batch2"] * 1,  # singleton batch2
            },
        )

        X = np.random.randn(n_samples, n_features)
        var = pl.DataFrame({"_index": [f"P{i:03d}" for i in range(n_features)]})

        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X, M=None)})
        container = ScpContainer(obs=obs, assays={"protein": assay})
        add_pca_assay(container, n_components=6)

        if not HARMONYPY_AVAILABLE:
            pytest.skip("harmonypy is optional dependency")

        with pytest.raises(ScpValueError, match="at least 2 samples per batch"):
            harmony(
                container,
                batch_key="batch",
                assay_name="pca",
                base_layer="X",
            )


# =============================================================================
# Test Scanorama Integration
# =============================================================================


class TestScanoramaIntegration:
    """Test Scanorama batch correction (requires scanorama)."""

    def test_scanorama_wrapper_passes_genes_and_aligns_output_with_fake_module(
        self,
        monkeypatch,
    ):
        """Wrapper should realign features and restore original sample order.

        This must happen after per-batch correction.
        """
        obs = pl.DataFrame(
            {
                "_index": ["s1", "s2", "s3", "s4"],
                "batch": ["A", "B", "A", "B"],
            },
        )
        X = np.array(
            [
                [1.0, 2.0, 3.0],
                [10.0, 20.0, 30.0],
                [4.0, 5.0, 6.0],
                [40.0, 50.0, 60.0],
            ],
            dtype=np.float64,
        )
        var = pl.DataFrame({"_index": ["g1", "g2", "g3"]})
        container = ScpContainer(
            obs=obs,
            assays={"protein": Assay(var=var, layers={"raw": ScpMatrix(X=X, M=None)})},
        )

        captured: dict[str, object] = {}
        fake_scanorama = types.ModuleType("scanorama")

        def fake_correct(datasets_full, genes_list, **kwargs):
            captured["datasets_full"] = datasets_full
            captured["genes_list"] = genes_list
            captured["kwargs"] = kwargs
            corrected = [dataset[:, ::-1] + 100.0 for dataset in datasets_full]
            return corrected, list(reversed(genes_list[0]))

        fake_scanorama.correct = fake_correct  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "scanorama", fake_scanorama)

        result = scanorama_integrate(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            new_layer_name="scanorama",
        )

        assert captured["genes_list"] == [["g1", "g2", "g3"], ["g1", "g2", "g3"]]
        assert captured["kwargs"]["return_dense"] is True
        assert captured["kwargs"]["ds_names"] == ["A", "B"]
        assert "dimred" not in captured["kwargs"]
        np.testing.assert_allclose(result.assays["protein"].layers["scanorama"].X, X + 100.0)

    def test_scanorama_explicit_dimred_is_forwarded_with_fake_module(self, monkeypatch):
        """Explicit dimred should be forwarded, while None should use Scanorama's own default."""
        container = create_batch_container(
            n_samples_per_batch=5,
            n_features=3,
            n_batches=2,
            random_state=42,
        )

        captured: dict[str, object] = {}
        fake_scanorama = types.ModuleType("scanorama")

        def fake_correct(datasets_full, genes_list, **kwargs):
            captured["kwargs"] = kwargs
            return [dataset.copy() for dataset in datasets_full], genes_list[0]

        fake_scanorama.correct = fake_correct  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "scanorama", fake_scanorama)

        scanorama_integrate(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            new_layer_name="scanorama",
            dimred=17,
        )

        assert captured["kwargs"]["dimred"] == 17

    def test_scanorama_rejects_feature_mismatch_with_fake_module(self, monkeypatch):
        """Wrapper should fail if Scanorama returns a different feature set than the assay."""
        container = create_batch_container(
            n_samples_per_batch=5,
            n_features=3,
            n_batches=2,
            random_state=42,
        )
        fake_scanorama = types.ModuleType("scanorama")

        def fake_correct(datasets_full, genes_list, **kwargs):
            corrected = [dataset[:, :2] for dataset in datasets_full]
            return corrected, genes_list[0][:2]

        fake_scanorama.correct = fake_correct  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "scanorama", fake_scanorama)

        with pytest.raises(ScpValueError, match="corrected feature set that does not match"):
            scanorama_integrate(
                container,
                batch_key="batch",
                assay_name="protein",
                base_layer="raw",
            )

    def test_scanorama_rejects_return_dimred_before_external_call(
        self,
        monkeypatch,
    ):
        """Wrapper should fail before calling Scanorama.

        This should happen when low-dimensional output cannot be stored.
        """
        container = create_batch_container(
            n_samples_per_batch=5,
            n_features=3,
            n_batches=2,
            random_state=42,
        )
        fake_scanorama = types.ModuleType("scanorama")

        def fake_correct(*args, **kwargs):
            raise AssertionError("scanorama.correct should not be called")

        fake_scanorama.correct = fake_correct  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "scanorama", fake_scanorama)

        with pytest.raises(
            ScpValueError,
            match="cannot store low-dimensional Scanorama embeddings",
        ):
            scanorama_integrate(
                container,
                batch_key="batch",
                assay_name="protein",
                base_layer="raw",
                return_dimred=True,
                dimred=2,
            )

    def test_scanorama_rejects_inf_values_with_fake_module(self, monkeypatch):
        """Scanorama should reject Inf under the complete-finite matrix contract."""
        container = create_batch_container(
            n_samples_per_batch=5,
            n_features=3,
            n_batches=2,
            random_state=42,
        )
        x = container.assays["protein"].layers["raw"].X.copy()
        x[0, 0] = np.inf
        container.assays["protein"].layers["raw"] = ScpMatrix(X=x, M=None)

        fake_scanorama = types.ModuleType("scanorama")

        def fake_correct(*args, **kwargs):
            raise AssertionError("scanorama.correct should not be called")

        fake_scanorama.correct = fake_correct  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "scanorama", fake_scanorama)

        with pytest.raises(ScpValueError, match="only finite values"):
            scanorama_integrate(
                container,
                batch_key="batch",
                assay_name="protein",
                base_layer="raw",
            )

    @pytest.mark.skipif(not SCANORAMA_AVAILABLE, reason="scanorama is optional dependency")
    def test_scanorama_basic_two_batches(self):
        """Test Scanorama integration with two batches."""
        container = create_batch_container(
            n_samples_per_batch=30,
            n_features=50,
            n_batches=2,
            random_state=42,
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

    @pytest.mark.skipif(not SCANORAMA_AVAILABLE, reason="scanorama is optional dependency")
    def test_scanorama_three_batches(self):
        """Test Scanorama with three batches."""
        container = create_batch_container(
            n_samples_per_batch=30,
            n_features=50,
            n_batches=3,
            random_state=42,
        )

        result = scanorama_integrate(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            new_layer_name="scanorama",
        )

        assert "scanorama" in result.assays["protein"].layers

    @pytest.mark.skipif(not SCANORAMA_AVAILABLE, reason="scanorama is optional dependency")
    def test_scanorama_with_sparse_input(self):
        """Test Scanorama with sparse input."""
        container = create_batch_container(
            n_samples_per_batch=30,
            n_features=50,
            n_batches=2,
            random_state=42,
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

    @pytest.mark.skipif(not SCANORAMA_AVAILABLE, reason="scanorama is optional dependency")
    def test_scanorama_with_dimensionality_reduction(self):
        """Current wrapper should reject low-dimensional Scanorama output on assay layers."""
        container = create_batch_container(
            n_samples_per_batch=30,
            n_features=50,
            n_batches=2,
            random_state=42,
        )

        with pytest.raises(
            ScpValueError,
            match="cannot store low-dimensional Scanorama embeddings",
        ):
            scanorama_integrate(
                container,
                batch_key="batch",
                assay_name="protein",
                base_layer="raw",
                new_layer_name="scanorama_dr",
                return_dimred=True,
                dimred=15,
            )

    @pytest.mark.skipif(not SCANORAMA_AVAILABLE, reason="scanorama is optional dependency")
    def test_scanorama_custom_parameters(self):
        """Test Scanorama with custom parameters."""
        container = create_batch_container(
            n_samples_per_batch=30,
            n_features=50,
            n_batches=2,
            random_state=42,
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

    @pytest.mark.skipif(not SCANORAMA_AVAILABLE, reason="scanorama is optional dependency")
    def test_scanorama_preserves_mask(self):
        """Test Scanorama preserves mask matrix."""
        container = create_batch_container(
            n_samples_per_batch=30,
            n_features=50,
            n_batches=2,
            random_state=42,
        )

        M = np.zeros((60, 50), dtype=np.int8)
        M[0:10, 0:10] = 1

        container.assays["protein"].layers["raw"] = ScpMatrix(
            X=container.assays["protein"].layers["raw"].X,
            M=M,
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

    @pytest.mark.skipif(not SCANORAMA_AVAILABLE, reason="scanorama is optional dependency")
    def test_scanorama_with_nan_values(self):
        """Test Scanorama rejects NaN values without implicit imputation."""
        container = create_batch_container(
            n_samples_per_batch=30,
            n_features=50,
            n_batches=2,
            random_state=42,
        )

        X = container.assays["protein"].layers["raw"].X
        X[0:5, 0:5] = np.nan
        container.assays["protein"].layers["raw"] = ScpMatrix(X=X, M=None)

        with pytest.raises(ScpValueError, match="requires a complete matrix"):
            scanorama_integrate(
                container,
                batch_key="batch",
                assay_name="protein",
                base_layer="raw",
                new_layer_name="scanorama",
            )

    # -------------------------------------------------------------------------
    # Error handling tests
    # -------------------------------------------------------------------------

    def test_scanorama_error_invalid_sigma(self):
        """Test Scanorama raises error for invalid sigma."""
        container = create_batch_container(
            n_samples_per_batch=30,
            n_features=50,
            n_batches=2,
            random_state=42,
        )

        if not SCANORAMA_AVAILABLE:
            with pytest.raises(MissingDependencyError):
                scanorama_integrate(
                    container,
                    batch_key="batch",
                    assay_name="protein",
                    base_layer="raw",
                    sigma=0,
                )
            return

        with pytest.raises(ScpValueError, match="sigma must be positive"):
            scanorama_integrate(
                container,
                batch_key="batch",
                assay_name="protein",
                base_layer="raw",
                sigma=0,
            )

    def test_scanorama_error_invalid_alpha(self):
        """Test Scanorama raises error for invalid alpha."""
        container = create_batch_container(
            n_samples_per_batch=30,
            n_features=50,
            n_batches=2,
            random_state=42,
        )

        if not SCANORAMA_AVAILABLE:
            with pytest.raises(MissingDependencyError):
                scanorama_integrate(
                    container,
                    batch_key="batch",
                    assay_name="protein",
                    base_layer="raw",
                    alpha=0,
                )
            return

        with pytest.raises(ScpValueError, match=r"alpha must be in \(0, 1\)"):
            scanorama_integrate(
                container,
                batch_key="batch",
                assay_name="protein",
                base_layer="raw",
                alpha=0,
            )

    def test_scanorama_error_invalid_knn(self):
        """Test Scanorama raises error for invalid knn."""
        container = create_batch_container(
            n_samples_per_batch=30,
            n_features=50,
            n_batches=2,
            random_state=42,
        )

        if not SCANORAMA_AVAILABLE:
            with pytest.raises(MissingDependencyError):
                scanorama_integrate(
                    container,
                    batch_key="batch",
                    assay_name="protein",
                    base_layer="raw",
                    knn=-5,
                )
            return

        with pytest.raises(ScpValueError):
            scanorama_integrate(
                container,
                batch_key="batch",
                assay_name="protein",
                base_layer="raw",
                knn=-5,
            )

    def test_scanorama_error_missing_assay(self):
        """Test Scanorama raises error for missing assay."""
        container = create_batch_container(
            n_samples_per_batch=30,
            n_features=50,
            n_batches=2,
            random_state=42,
        )

        if not SCANORAMA_AVAILABLE:
            with pytest.raises(MissingDependencyError):
                scanorama_integrate(
                    container,
                    batch_key="batch",
                    assay_name="nonexistent",
                    base_layer="raw",
                )
            return

        with pytest.raises(AssayNotFoundError, match="nonexistent"):
            scanorama_integrate(
                container,
                batch_key="batch",
                assay_name="nonexistent",
                base_layer="raw",
            )

    def test_scanorama_error_single_batch(self):
        """Test Scanorama raises error with only one batch."""
        container = create_batch_container(
            n_samples_per_batch=30,
            n_features=50,
            n_batches=2,
            random_state=42,
        )

        # Modify to single batch
        container.obs = container.obs.with_columns(pl.lit("batch1").alias("batch"))

        if not SCANORAMA_AVAILABLE:
            with pytest.raises(MissingDependencyError):
                scanorama_integrate(
                    container,
                    batch_key="batch",
                    assay_name="protein",
                    base_layer="raw",
                )
            return

        with pytest.raises(ScpValueError, match="at least 2 batches"):
            scanorama_integrate(
                container,
                batch_key="batch",
                assay_name="protein",
                base_layer="raw",
            )


# =============================================================================
# Test ComBat (additional tests)
# =============================================================================


class TestComBatAdditional:
    """Additional tests for ComBat batch correction."""

    def test_combat_with_sparse_input(self):
        """Test ComBat with sparse input matrix."""
        container = create_batch_container(
            n_samples_per_batch=30,
            n_features=50,
            n_batches=2,
            random_state=42,
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
        """Test ComBat rejects NaN values instead of silently imputing."""
        container = create_batch_container(
            n_samples_per_batch=30,
            n_features=50,
            n_batches=2,
            random_state=42,
        )

        X = container.assays["protein"].layers["raw"].X
        X[0:5, 0:5] = np.nan
        container.assays["protein"].layers["raw"] = ScpMatrix(X=X, M=None)

        with pytest.raises(
            ValidationError,
            match="requires a complete finite matrix.*prefer integrate_limma\\(\\)",
        ):
            combat(
                container,
                batch_key="batch",
                assay_name="protein",
                base_layer="raw",
                new_layer_name="combat",
            )

    def test_combat_with_inf_values(self):
        """Test ComBat rejects Inf values under the complete-finite matrix contract."""
        container = create_batch_container(
            n_samples_per_batch=30,
            n_features=50,
            n_batches=2,
            random_state=42,
        )

        X = container.assays["protein"].layers["raw"].X
        X[0, 0] = np.inf
        container.assays["protein"].layers["raw"] = ScpMatrix(X=X, M=None)

        with pytest.raises(
            ValidationError,
            match="requires a complete finite matrix.*prefer integrate_limma\\(\\)",
        ):
            combat(
                container,
                batch_key="batch",
                assay_name="protein",
                base_layer="raw",
                new_layer_name="combat",
            )

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
            },
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

    def test_combat_with_covariates_matches_reference_stand_mean(self):
        """Reference: grand mean must exclude the average covariate effect."""
        base = np.array([0.0, 1.0, 10.0, 11.0], dtype=np.float64)[:, None]
        X = np.hstack([base, base + 100.0, base + 200.0])
        obs = pl.DataFrame(
            {
                "_index": ["s1", "s2", "s3", "s4"],
                "batch": ["A", "A", "B", "B"],
                "group": ["g1", "g2", "g1", "g2"],
            },
        )
        var = pl.DataFrame({"_index": ["p1", "p2", "p3"]})
        container = ScpContainer(
            obs=obs,
            assays={"protein": Assay(var=var, layers={"raw": ScpMatrix(X=X, M=None)})},
        )

        result = combat(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            new_layer_name="combat",
            covariates=["group"],
        )

        expected = np.array(
            [
                [5.0, 105.0, 205.0],
                [6.0, 106.0, 206.0],
                [5.0, 105.0, 205.0],
                [6.0, 106.0, 206.0],
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(result.assays["protein"].layers["combat"].X, expected)

    def test_combat_error_singleton_batch(self):
        """Test ComBat raises error with singleton batch."""
        np.random.seed(42)
        n_samples = 10
        n_features = 20

        obs = pl.DataFrame(
            {
                "_index": [f"S{i:03d}" for i in range(n_samples)],
                "batch": ["batch1"] * 9 + ["batch2"] * 1,
            },
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
            n_samples_per_batch=30,
            n_features=50,
            n_batches=2,
            random_state=42,
        )

        M = np.zeros((60, 50), dtype=np.int8)
        M[0:10, 0:10] = 1
        M[10:20, 10:20] = 2

        container.assays["protein"].layers["raw"] = ScpMatrix(
            X=container.assays["protein"].layers["raw"].X,
            M=M,
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
        assert M_combat is not M

    def test_combat_nonparametric_mode(self):
        """Test ComBat nonparametric EB mode."""
        container = create_batch_container(
            n_samples_per_batch=30,
            n_features=40,
            n_batches=2,
            random_state=42,
        )

        result = combat(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            new_layer_name="combat_np",
            eb_mode="nonparametric",
        )

        assert "combat_np" in result.assays["protein"].layers
        x = result.assays["protein"].layers["combat_np"].X
        assert x.shape == (60, 40)
        assert np.isfinite(x).all()
        assert result.history[-1].params["eb_mode"] == "nonparametric"

    def test_combat_invalid_eb_mode_raises_error(self):
        """Test ComBat validates EB mode values."""
        container = create_batch_container(
            n_samples_per_batch=20,
            n_features=30,
            n_batches=2,
            random_state=42,
        )

        with pytest.raises(ScpValueError, match="eb_mode must be one of"):
            combat(
                container,
                batch_key="batch",
                assay_name="protein",
                base_layer="raw",
                eb_mode="invalid_mode",  # type: ignore[arg-type]
            )

    def test_combat_missing_covariate_column_raises_error(self):
        """ComBat should report missing covariate columns with actionable message."""
        container = create_batch_container(
            n_samples_per_batch=20,
            n_features=30,
            n_batches=2,
            random_state=42,
        )

        with pytest.raises(ScpValueError, match="covariates contain missing columns"):
            combat(
                container,
                batch_key="batch",
                assay_name="protein",
                base_layer="raw",
                covariates=["missing_group"],
            )

    def test_combat_rejects_nonfinite_covariates(self):
        """ComBat should fail clearly when covariates contain NaN/Inf values."""
        container = create_batch_container(
            n_samples_per_batch=20,
            n_features=12,
            n_batches=2,
            random_state=42,
        )
        container.obs = container.obs.with_columns(
            pl.Series("score", [0.0] * 10 + [1.0] * 10 + [float("nan")] * 20),
        )

        with pytest.raises(ScpValueError, match="design matrix containing missing or non-finite"):
            combat(
                container,
                batch_key="batch",
                assay_name="protein",
                base_layer="raw",
                covariates=["score"],
            )

    def test_combat_solve_eb_matches_reference_postvar_denominator(self):
        """ComBat d* update should use denominator a + n/2 - 1 (reference implementation)."""
        g_hat = np.array([[0.5], [-0.2]])
        d_hat = np.array([[1.2], [0.8]])
        g_bar = 0.1
        t2 = 0.6
        a = 3.0
        b = 2.0
        n = 6

        g_new, d_new = _solve_eb(
            g_hat=g_hat,
            d_hat=d_hat,
            g_bar=g_bar,
            t2=t2,
            a=a,
            b=b,
            n=n,
            conv=np.inf,  # force one-step update for formula-level validation
        )

        expected_g = (n * t2 * g_hat + d_hat * g_bar) / (n * t2 + d_hat)
        expected_sum2 = (n - 1) * d_hat + n * (g_hat - expected_g) ** 2
        expected_d = (b + 0.5 * expected_sum2) / (a + n / 2 - 1)

        assert np.allclose(g_new, expected_g)
        assert np.allclose(d_new, expected_d)

    def test_combat_single_feature_falls_back_without_nan(self):
        """With one feature, EB priors are not estimable; correction should still stay finite."""
        X = np.array([[0.0], [1.0], [10.0], [11.0]], dtype=np.float64)
        obs = pl.DataFrame(
            {
                "_index": ["s1", "s2", "s3", "s4"],
                "batch": ["A", "A", "B", "B"],
                "group": ["g1", "g2", "g1", "g2"],
            },
        )
        var = pl.DataFrame({"_index": ["p1"]})
        container = ScpContainer(
            obs=obs,
            assays={"protein": Assay(var=var, layers={"raw": ScpMatrix(X=X, M=None)})},
        )

        result = combat(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            new_layer_name="combat",
            covariates=["group"],
        )
        np.testing.assert_allclose(
            result.assays["protein"].layers["combat"].X.ravel(),
            np.array([5.0, 6.0, 5.0, 6.0], dtype=np.float64),
        )


class TestLimmaIntegration:
    """Tests for limma-style matrix-level batch correction."""

    def test_limma_default_matches_remove_batch_effect_grand_mean(self):
        """Default limma behavior should remove batch effects to the centered mean, not batch1."""
        base = np.array([0.0, 1.0, 10.0, 11.0], dtype=np.float64)[:, None]
        X = np.hstack([base, base + 100.0, base + 200.0])
        obs = pl.DataFrame(
            {
                "_index": ["s1", "s2", "s3", "s4"],
                "batch": ["A", "A", "B", "B"],
            },
        )
        var = pl.DataFrame({"_index": ["p1", "p2", "p3"]})
        container = ScpContainer(
            obs=obs,
            assays={"protein": Assay(var=var, layers={"raw": ScpMatrix(X=X, M=None)})},
        )

        result = limma_correct(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            new_layer_name="limma",
        )
        expected = np.array(
            [
                [5.0, 105.0, 205.0],
                [6.0, 106.0, 206.0],
                [5.0, 105.0, 205.0],
                [6.0, 106.0, 206.0],
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(result.assays["protein"].layers["limma"].X, expected, atol=1e-12)

    def test_limma_reference_batch_is_scptensor_extension(self):
        """reference_batch should shift corrected values so the chosen batch remains unchanged."""
        base = np.array([0.0, 1.0, 10.0, 11.0], dtype=np.float64)[:, None]
        X = np.hstack([base, base + 100.0, base + 200.0])
        obs = pl.DataFrame(
            {
                "_index": ["s1", "s2", "s3", "s4"],
                "batch": ["A", "A", "B", "B"],
            },
        )
        var = pl.DataFrame({"_index": ["p1", "p2", "p3"]})
        container = ScpContainer(
            obs=obs,
            assays={"protein": Assay(var=var, layers={"raw": ScpMatrix(X=X, M=None)})},
        )

        result = limma_correct(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            new_layer_name="limma",
            reference_batch="A",
        )
        expected = np.array(
            [
                [0.0, 100.0, 200.0],
                [1.0, 101.0, 201.0],
                [0.0, 100.0, 200.0],
                [1.0, 101.0, 201.0],
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(result.assays["protein"].layers["limma"].X, expected, atol=1e-12)

    def test_limma_basic_two_batches(self):
        """Test limma correction adds output layer with preserved shape."""
        container = create_batch_container(
            n_samples_per_batch=30,
            n_features=50,
            n_batches=2,
            random_state=42,
        )

        result = limma_correct(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            new_layer_name="limma",
        )

        assert "limma" in result.assays["protein"].layers
        x = result.assays["protein"].layers["limma"].X
        assert x.shape == (60, 50)

    def test_limma_with_covariates(self):
        """Test limma correction with biological covariates."""
        np.random.seed(42)
        n_samples_per_batch = 30
        n_features = 40
        batches = np.array(["batch1"] * n_samples_per_batch + ["batch2"] * n_samples_per_batch)
        groups = np.array(["GroupA"] * 15 + ["GroupB"] * 15 + ["GroupA"] * 15 + ["GroupB"] * 15)

        obs = pl.DataFrame(
            {
                "_index": [f"S{i:03d}" for i in range(2 * n_samples_per_batch)],
                "batch": batches.tolist(),
                "group": groups.tolist(),
            },
        )

        X = np.random.randn(2 * n_samples_per_batch, n_features)
        X[batches == "batch2"] += 1.5
        X[groups == "GroupB", : n_features // 2] += 0.8

        var = pl.DataFrame({"_index": [f"P{i:03d}" for i in range(n_features)]})
        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X, M=None)})
        container = ScpContainer(obs=obs, assays={"protein": assay})

        result = limma_correct(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            new_layer_name="limma",
            covariates=["group"],
        )

        assert "limma" in result.assays["protein"].layers
        assert np.isfinite(result.assays["protein"].layers["limma"].X).all()

    def test_limma_confounded_design_raises_error(self):
        """Test limma raises clear error when batch and covariates are confounded."""
        container = create_batch_container(
            n_samples_per_batch=30,
            n_features=40,
            n_batches=2,
            random_state=42,
        )
        # In this fixture, group is fully aligned with batch by construction.
        with pytest.raises(ValueError, match="rank deficient"):
            limma_correct(
                container,
                batch_key="batch",
                assay_name="protein",
                base_layer="raw",
                covariates=["group"],
            )

    def test_limma_preserves_mask(self):
        """Test limma preserves mask matrix."""
        container = create_batch_container(
            n_samples_per_batch=30,
            n_features=25,
            n_batches=2,
            random_state=42,
        )
        M = np.zeros((60, 25), dtype=np.int8)
        M[0:10, 0:10] = 1
        container.assays["protein"].layers["raw"] = ScpMatrix(
            X=container.assays["protein"].layers["raw"].X,
            M=M,
        )

        result = limma_correct(container, batch_key="batch", assay_name="protein", base_layer="raw")
        m_out = result.assays["protein"].layers["limma"].M
        assert m_out is not None
        assert np.array_equal(m_out, M)

    def test_limma_reduces_batch_effect(self):
        """Test limma correction reduces ANOVA-style batch metric."""
        container = create_batch_container(
            n_samples_per_batch=40,
            n_features=50,
            n_batches=2,
            random_state=42,
        )
        X_orig = container.assays["protein"].layers["raw"].X
        batches = container.obs["batch"].to_numpy()
        before = compute_batch_effect_metric(X_orig, batches)

        result = limma_correct(container, batch_key="batch", assay_name="protein", base_layer="raw")
        after = compute_batch_effect_metric(result.assays["protein"].layers["limma"].X, batches)
        assert after < before

    def test_limma_logs_history(self):
        """Test limma logs method metadata in provenance."""
        container = create_batch_container(
            n_samples_per_batch=20,
            n_features=20,
            n_batches=2,
            random_state=42,
        )

        result = limma_correct(container, batch_key="batch", assay_name="protein", base_layer="raw")
        log_entry = result.history[-1]
        assert log_entry.action == "integration_limma"
        assert log_entry.params["integration_level"] == "matrix"
        assert log_entry.params["recommended_for_de"] is True

    def test_limma_copies_mask_not_reference(self):
        """Regression: integration output M should be copied, not aliased."""
        container = create_batch_container(
            n_samples_per_batch=20,
            n_features=12,
            n_batches=2,
            random_state=42,
        )
        raw_layer = container.assays["protein"].layers["raw"]
        raw_layer.M = np.zeros(raw_layer.X.shape, dtype=np.int8)

        result = limma_correct(container, batch_key="batch", assay_name="protein", base_layer="raw")

        m_out = result.assays["protein"].layers["limma"].M
        assert m_out is not None
        assert np.array_equal(m_out, raw_layer.M)
        assert m_out is not raw_layer.M

    def test_limma_preserves_missing_values_without_imputation(self):
        """Test limma fits on observed samples and keeps missing positions as NaN."""
        container = create_batch_container(
            n_samples_per_batch=20,
            n_features=12,
            n_batches=2,
            random_state=42,
        )
        x = container.assays["protein"].layers["raw"].X.copy()
        x[0:5, 0] = np.nan
        x[20:24, 1] = np.nan
        container.assays["protein"].layers["raw"] = ScpMatrix(X=x, M=None)

        result = limma_correct(container, batch_key="batch", assay_name="protein", base_layer="raw")
        x_out = result.assays["protein"].layers["limma"].X

        assert np.isnan(x_out[0:5, 0]).all()
        assert np.isnan(x_out[20:24, 1]).all()

    def test_limma_rejects_inf_values(self):
        """Limma-style batch correction should allow NaN but reject Inf inputs."""
        container = create_batch_container(
            n_samples_per_batch=20,
            n_features=12,
            n_batches=2,
            random_state=42,
        )
        x = container.assays["protein"].layers["raw"].X.copy()
        x[0, 0] = np.inf
        container.assays["protein"].layers["raw"] = ScpMatrix(X=x, M=None)

        with pytest.raises(ScpValueError, match="supports NaN missing values but not Inf"):
            limma_correct(container, batch_key="batch", assay_name="protein", base_layer="raw")

    def test_limma_rejects_nonfinite_covariates(self):
        """Covariates must be fully observed and finite for the design matrix."""
        container = create_batch_container(
            n_samples_per_batch=20,
            n_features=12,
            n_batches=2,
            random_state=42,
        )
        container.obs = container.obs.with_columns(
            pl.Series("score", [0.0] * 10 + [1.0] * 10 + [float("nan")] * 20),
        )

        with pytest.raises(ScpValueError, match="design matrix containing missing or non-finite"):
            limma_correct(
                container,
                batch_key="batch",
                assay_name="protein",
                base_layer="raw",
                covariates=["score"],
            )

    def test_mnn_rejects_nan_values_without_explicit_imputation(self):
        """Embedding-level integration should require complete input."""
        container = create_batch_container(
            n_samples_per_batch=20,
            n_features=10,
            n_batches=2,
            random_state=42,
        )
        x = container.assays["protein"].layers["raw"].X.copy()
        x[0, 0] = np.nan
        container.assays["protein"].layers["raw"] = ScpMatrix(X=x, M=None)

        with pytest.raises(ScpValueError, match="requires a complete matrix"):
            mnn_correct(container, batch_key="batch", assay_name="protein", base_layer="raw")


# =============================================================================
# Test Batch Effect Reduction
# =============================================================================


class TestBatchEffectReduction:
    """Test that batch correction methods actually reduce batch effects."""

    def test_mnn_reduces_batch_effect(self):
        """Test MNN correction reduces batch effect metric."""
        container = create_batch_container(
            n_samples_per_batch=40,
            n_features=50,
            n_batches=2,
            random_state=42,
        )

        X_orig = container.assays["protein"].layers["raw"].X
        batches = container.obs["batch"].to_numpy()

        batch_effect_before = compute_batch_effect_metric(X_orig, batches)

        result = mnn_correct(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            k=15,
        )

        X_corrected = result.assays["protein"].layers["mnn_corrected"].X
        batch_effect_after = compute_batch_effect_metric(X_corrected, batches)

        assert batch_effect_after < batch_effect_before

    def test_combat_reduces_batch_effect(self):
        """Test ComBat correction reduces batch effect metric."""
        container = create_batch_container(
            n_samples_per_batch=40,
            n_features=50,
            n_batches=2,
            random_state=42,
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
        compute_biological_signal_retention(X_orig, groups)

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
        compute_biological_signal_retention(X_orig, groups)

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
            },
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
            },
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
            },
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
            },
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
            n_samples_per_batch=30,
            n_features=50,
            n_batches=2,
            random_state=42,
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

    def test_mnn_history_uses_resolved_assay_name_for_aliases(self):
        """MNN provenance should record the resolved assay key, not the caller alias."""
        container = create_batch_container(
            n_samples_per_batch=10,
            n_features=12,
            n_batches=2,
            random_state=42,
        )
        container.assays["proteins"] = container.assays.pop("protein")

        result = mnn_correct(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            k=3,
            sigma=1.0,
        )

        assert result.history[-1].params["assay"] == "proteins"

    def test_combat_logs_to_history(self):
        """Test ComBat logs operation to history."""
        container = create_batch_container(
            n_samples_per_batch=30,
            n_features=50,
            n_batches=2,
            random_state=42,
        )

        initial_len = len(container.history)

        result = combat(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
        )

        assert len(result.history) == initial_len + 1
        log_entry = result.history[-1]
        assert log_entry.action == "integration_combat"
        assert log_entry.params["integration_level"] == "matrix"
        assert log_entry.params["recommended_for_de"] is False


class TestIntegrationBaselineAndMetadata:
    """Tests for explicit no-correction baseline and method metadata."""

    def test_integrate_none_creates_copied_layer_and_logs(self):
        """No-op integration should copy values into a new layer and log metadata."""
        container = create_batch_container(
            n_samples_per_batch=20,
            n_features=12,
            n_batches=2,
            random_state=42,
        )
        x_before = container.assays["protein"].layers["raw"].X
        raw_layer = container.assays["protein"].layers["raw"]
        raw_layer.M = np.zeros(raw_layer.X.shape, dtype=np.int8)

        result = integrate_none(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            new_layer_name="none",
        )

        assert "none" in result.assays["protein"].layers
        x_after = result.assays["protein"].layers["none"].X
        assert np.array_equal(x_before, x_after)
        assert x_before is not x_after
        m_after = result.assays["protein"].layers["none"].M
        assert m_after is not None
        assert np.array_equal(m_after, raw_layer.M)
        assert m_after is not raw_layer.M

        log_entry = result.history[-1]
        assert log_entry.action == "integration_none"
        assert log_entry.params["integration_level"] == "matrix"
        assert log_entry.params["recommended_for_de"] is True

    def test_integrate_none_history_uses_resolved_assay_name_for_aliases(self):
        """No-op integration should log the resolved assay key used by the container."""
        container = create_batch_container(
            n_samples_per_batch=20,
            n_features=12,
            n_batches=2,
            random_state=42,
        )
        container.assays["proteins"] = container.assays.pop("protein")

        result = integrate_none(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            new_layer_name="none",
        )

        assert "none" in result.assays["proteins"].layers
        assert result.history[-1].params["assay"] == "proteins"


def test_harmony_runs_with_stub_dependency_and_logs_metadata(monkeypatch):
    """Harmony optional path should stay testable without a real harmonypy install."""
    container = create_batch_container(
        n_samples_per_batch=8,
        n_features=10,
        n_batches=2,
        random_state=42,
    )
    add_pca_layer_to_protein_assay(container)

    fake_module = types.ModuleType("harmonypy")

    def fake_run_harmony(X, meta_data, batch_key, **kwargs):
        assert batch_key == "batch"
        assert meta_data.shape[0] == X.shape[0]
        sigma = np.asarray(kwargs["sigma"], dtype=np.float64)
        assert kwargs["nclust"] == 1
        assert sigma.shape == (1,)
        assert sigma[0] == pytest.approx(0.1)
        return types.SimpleNamespace(Z_corr=(X + 1.0))

    fake_module.run_harmony = fake_run_harmony
    monkeypatch.setitem(sys.modules, "harmonypy", fake_module)

    result = harmony(
        container,
        batch_key="batch",
        assay_name="protein",
        base_layer="pca",
        new_layer_name="harmony_stub",
        theta=3.0,
        lamb=0.5,
        max_iter_harmony=2,
        max_iter_cluster=4,
    )

    input_x = container.assays["protein"].layers["pca"].X
    output_x = result.assays["protein"].layers["harmony_stub"].X
    assert np.allclose(output_x, input_x + 1.0)
    assert result.history[-1].action == "integration_harmony"
    assert result.history[-1].params["integration_level"] == "embedding"
    assert result.history[-1].params["recommended_for_de"] is False
    assert result.history[-1].params["theta"] == 3.0
    assert result.history[-1].params["lamb"] == 0.5

    def test_integrate_none_preserves_missing_values_verbatim(self):
        """No-op integration should copy incomplete matrices without filling NaN."""
        container = create_batch_container(
            n_samples_per_batch=20,
            n_features=12,
            n_batches=2,
            random_state=42,
        )
        x = container.assays["protein"].layers["raw"].X.copy()
        x[0:4, 0] = np.nan
        x[10:12, 3] = np.nan
        container.assays["protein"].layers["raw"] = ScpMatrix(X=x, M=None)

        result = integrate_none(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            new_layer_name="none",
        )

        x_out = result.assays["protein"].layers["none"].X
        assert np.array_equal(np.isnan(x_out), np.isnan(x))
        observed = ~np.isnan(x)
        np.testing.assert_allclose(x_out[observed], x[observed])

    def test_combat_keeps_source_layer_unchanged_when_target_name_differs(self):
        """Matrix-level correction should not mutate the source layer when writing elsewhere."""
        container = create_batch_container(
            n_samples_per_batch=20,
            n_features=12,
            n_batches=2,
            random_state=42,
        )
        raw_layer = container.assays["protein"].layers["raw"]
        raw_x_before = raw_layer.X.copy()
        raw_layer.M = np.zeros(raw_layer.X.shape, dtype=np.int8)
        raw_m_before = raw_layer.M.copy()

        result = combat(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            new_layer_name="combat",
        )

        raw_after = result.assays["protein"].layers["raw"]
        np.testing.assert_allclose(raw_after.X, raw_x_before)
        assert np.array_equal(raw_after.M, raw_m_before)
        assert "combat" in result.assays["protein"].layers

    def test_combat_overwrites_base_layer_when_target_name_matches_source(self):
        """Current layer-collision semantics also apply when target name equals base layer."""
        container = create_batch_container(
            n_samples_per_batch=20,
            n_features=12,
            n_batches=2,
            random_state=42,
        )
        raw_layer_before = container.assays["protein"].layers["raw"]
        raw_x_before = raw_layer_before.X.copy()

        result = combat(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            new_layer_name="raw",
        )

        raw_layer_after = result.assays["protein"].layers["raw"]
        assert raw_layer_after is not raw_layer_before
        assert not np.allclose(raw_layer_after.X, raw_x_before)

    def test_integrate_none_missing_batch_key_raises_error(self):
        """No-op integration still validates batch_key for API consistency."""
        container = create_batch_container(
            n_samples_per_batch=20,
            n_features=12,
            n_batches=2,
            random_state=42,
        )

        with pytest.raises(ScpValueError, match="Batch key.*not found"):
            integrate_none(
                container,
                batch_key="missing_batch",
                assay_name="protein",
                base_layer="raw",
            )

    def test_assay_alias_allows_proteins_for_default_protein_methods(self):
        """Methods with default assay='protein' should work on assay named 'proteins'."""
        container = create_batch_container(
            n_samples_per_batch=20,
            n_features=12,
            n_batches=2,
            random_state=42,
        )
        protein_assay = container.assays.pop("protein")
        container.assays["proteins"] = protein_assay

        result = combat(
            container,
            batch_key="batch",
            base_layer="raw",
            new_layer_name="combat",
        )

        assert "combat" in result.assays["proteins"].layers

    def test_registered_method_metadata_exposed(self):
        """Integration methods should expose level/de suitability metadata."""
        none_info = get_integrate_method_info("none")
        combat_info = get_integrate_method_info("combat")
        limma_info = get_integrate_method_info("limma")
        mnn_info = get_integrate_method_info("mnn")
        harmony_info = get_integrate_method_info("harmony")
        scanorama_info = get_integrate_method_info("scanorama")

        assert none_info.integration_level == "matrix"
        assert none_info.recommended_for_de is True
        assert combat_info.integration_level == "matrix"
        assert combat_info.recommended_for_de is False
        assert limma_info.integration_level == "matrix"
        assert limma_info.recommended_for_de is True

        assert mnn_info.integration_level == "embedding"
        assert mnn_info.recommended_for_de is False
        assert harmony_info.integration_level == "embedding"
        assert harmony_info.recommended_for_de is False
        assert scanorama_info.integration_level == "embedding"
        assert scanorama_info.recommended_for_de is False
