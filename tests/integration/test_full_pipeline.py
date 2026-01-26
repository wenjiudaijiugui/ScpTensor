"""
Comprehensive end-to-end integration tests for ScpTensor.

This module provides extensive integration tests covering complete analysis workflows
including data loading, quality control, normalization, imputation, batch correction,
dimensionality reduction, and clustering. Tests verify:

1. Complete workflow execution from raw data to final results
2. Layer creation and immutability (functional pattern)
3. Mask code propagation through pipeline steps
4. Provenance logging (history tracking)
5. Sparse and dense matrix handling
6. Multi-assay workflows
7. Error handling and edge cases
8. Multiple imputation/batch correction combinations

Test Structure:
- TestBasicPipeline: Core workflow validation
- TestSparsePipeline: Sparse matrix workflows
- TestMaskPropagation: Mask code tracking
- TestProvenance: History logging verification
- TestCombinations: Different method combinations
- TestMultiAssay: Multiple assay handling
- TestEdgeCases: Boundary conditions
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy import sparse as sp

from scptensor.cluster import cluster_kmeans as run_kmeans
from scptensor.core.exceptions import (
    AssayNotFoundError,
    LayerNotFoundError,
    ScpValueError,
)
from scptensor.core.structures import (
    Assay,
    MaskCode,
    ScpContainer,
    ScpMatrix,
)
from scptensor.dim_reduction import reduce_pca as pca
from scptensor.dim_reduction import reduce_umap as umap
from scptensor.impute import impute_bpca as bpca
from scptensor.impute import impute_knn as knn
from scptensor.integration import integrate_combat as combat
from scptensor.integration import integrate_mnn as mnn_correct
from scptensor.normalization import log_transform as log_normalize
from scptensor.normalization import norm_median as norm_median_center

# =============================================================================
# Helper Functions
# =============================================================================


def create_test_container(
    n_samples: int = 30,
    n_features: int = 50,
    missing_rate: float = 0.3,
    sparse: bool = False,
    random_state: int = 42,
    include_lod: bool = True,
) -> ScpContainer:
    """
    Create a test container with synthetic data.

    Parameters
    ----------
    n_samples : int
        Number of samples
    n_features : int
        Number of features
    missing_rate : float
        Proportion of missing values (MBR type)
    sparse : bool
        Whether to use sparse matrix storage
    random_state : int
        Random seed
    include_lod : bool
        Whether to include LOD (limit of detection) missing values

    Returns
    -------
    ScpContainer
        Container with synthetic test data
    """
    rng = np.random.default_rng(random_state)

    # Create metadata with batches and groups
    groups = np.array(["GroupA"] * (n_samples // 2) + ["GroupB"] * (n_samples - n_samples // 2))
    batches = np.array(["Batch1"] * (n_samples // 2) + ["Batch2"] * (n_samples - n_samples // 2))

    obs = pl.DataFrame(
        {
            "_index": [f"S{i:03d}" for i in range(n_samples)],
            "batch": batches.tolist(),
            "group": groups.tolist(),
        }
    )

    # Create expression data
    X_true = rng.lognormal(mean=2, sigma=0.5, size=(n_samples, n_features))

    # Add group effect to first half of features
    X_true[groups == "GroupB", : n_features // 2] *= 2.0

    # Add batch effect
    X_true[batches == "Batch2", :] *= 1.5

    # Create observed data with missing values
    X_observed = X_true.copy()
    M = np.zeros((n_samples, n_features), dtype=np.int8)

    # Add LOD missing values (low abundance) - only if include_lod is True
    if include_lod:
        threshold = np.percentile(X_true, 20)
        lod_mask = X_true < threshold
        X_observed[lod_mask] = 0
        M[lod_mask] = MaskCode.LOD

    # Add random MBR missing values
    valid_mask = M == 0
    n_missing = int(n_samples * n_features * missing_rate)
    valid_indices = np.argwhere(valid_mask)

    if len(valid_indices) > n_missing:
        missing_idx = rng.choice(len(valid_indices), size=n_missing, replace=False)
        missing_indices = valid_indices[missing_idx]
        X_observed[missing_indices[:, 0], missing_indices[:, 1]] = 0
        M[missing_indices[:, 0], missing_indices[:, 1]] = MaskCode.MBR

    # Create feature metadata
    var = pl.DataFrame(
        {
            "_index": [f"P{i:03d}" for i in range(n_features)],
        }
    )

    # Create matrix with appropriate storage
    if sparse:
        X_final = sp.csr_matrix(X_observed)
    else:
        X_final = X_observed

    matrix = ScpMatrix(X=X_final, M=M)
    assay = Assay(var=var, layers={"raw": matrix})

    return ScpContainer(obs=obs, assays={"protein": assay})


def count_mask_codes(M: np.ndarray | None) -> dict[int, int]:
    """
    Count occurrences of each mask code.

    Parameters
    ----------
    M : np.ndarray or None
        Mask matrix

    Returns
    -------
    dict[int, int]
        Dictionary mapping mask codes to counts
    """
    if M is None:
        return {0: 0}
    unique, counts = np.unique(M, return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist(), strict=False))


# =============================================================================
# Test Basic Pipeline
# =============================================================================


class TestBasicPipeline:
    """Test the basic analysis pipeline workflow."""

    def test_full_pipeline_execution(self, small_synthetic_container):
        """
        Test complete pipeline from raw data to clustering results.

        Validates:
        - All pipeline steps execute without errors
        - Layers are created at each step
        - Final results are valid (cluster assignments)
        - History is logged properly
        """
        container = small_synthetic_container

        # Step 1: Log normalization
        log_normalize(container, assay_name="protein", source_layer="raw", new_layer_name="log")
        assert "log" in container.assays["protein"].layers

        # Step 2: KNN imputation
        knn(container, assay_name="protein", source_layer="log", new_layer_name="imputed", k=5)
        assert "imputed" in container.assays["protein"].layers

        # Step 3: ComBat batch correction
        combat(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="imputed",
            new_layer_name="corrected",
        )
        assert "corrected" in container.assays["protein"].layers

        # Step 4: PCA dimensionality reduction
        container = pca(
            container,
            assay_name="protein",
            base_layer_name="corrected",
            new_assay_name="pca",
            n_components=5,
        )
        assert "pca" in container.assays
        assert "scores" in container.assays["pca"].layers

        # Step 5: K-means clustering
        container = run_kmeans(
            container,
            assay_name="pca",
            base_layer="scores",
            n_clusters=2,
            key_added="kmeans_cluster",
        )
        assert "kmeans_cluster" in container.obs.columns

        # Verify cluster assignments
        clusters = container.obs["kmeans_cluster"].to_numpy()
        assert len(clusters) == container.n_samples
        assert len(np.unique(clusters)) == 2

    def test_pipeline_preserves_all_layers(self, small_synthetic_container):
        """
        Test that pipeline preserves all intermediate layers.

        Validates that the functional pattern is followed:
        - Original 'raw' layer is unchanged
        - All new layers are created independently
        - No in-place modifications occur
        """
        container = small_synthetic_container

        # Store original raw data
        raw_X = container.assays["protein"].layers["raw"].X.copy()
        raw_M = container.assays["protein"].layers["raw"].M.copy()

        # Run pipeline
        log_normalize(container, assay_name="protein", source_layer="raw", new_layer_name="log")
        knn(container, assay_name="protein", source_layer="log", new_layer_name="imputed", k=5)
        combat(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="imputed",
            new_layer_name="corrected",
        )

        # Verify all layers exist
        layers = container.assays["protein"].layers
        assert "raw" in layers
        assert "log" in layers
        assert "imputed" in layers
        assert "corrected" in layers

        # Verify original layer unchanged
        assert np.array_equal(layers["raw"].X, raw_X)
        assert np.array_equal(layers["raw"].M, raw_M)

    def test_pipeline_dimensions_consistency(self, small_synthetic_container):
        """
        Test that all layers maintain consistent dimensions.

        Validates:
        - All data layers have same shape
        - Sample count is preserved
        - Feature count is preserved
        - PCA creates proper reduced dimensions
        """
        container = small_synthetic_container

        original_n_samples = container.n_samples
        original_n_features = container.assays["protein"].n_features

        # Run pipeline
        log_normalize(container, assay_name="protein", source_layer="raw", new_layer_name="log")
        knn(container, assay_name="protein", source_layer="log", new_layer_name="imputed", k=5)
        combat(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="imputed",
            new_layer_name="corrected",
        )

        # Check all protein assay layers have same dimensions
        for name, layer in container.assays["protein"].layers.items():
            assert layer.X.shape == (original_n_samples, original_n_features), (
                f"Layer {name} has incorrect shape"
            )

        # Run PCA and verify reduced dimensions
        n_components = 5
        container = pca(
            container,
            assay_name="protein",
            base_layer_name="corrected",
            new_assay_name="pca",
            n_components=n_components,
        )

        pca_scores = container.assays["pca"].layers["scores"]
        assert pca_scores.X.shape == (original_n_samples, n_components)


# =============================================================================
# Test Sparse Matrix Pipeline
# =============================================================================


class TestSparsePipeline:
    """Test pipeline with sparse matrix storage."""

    def test_sparse_full_pipeline(self):
        """
        Test complete pipeline with sparse input data.

        Validates that sparse matrices are handled correctly
        throughout the entire analysis pipeline.
        """
        container = create_test_container(n_samples=30, n_features=50, sparse=True)

        # Verify input is sparse
        raw_layer = container.assays["protein"].layers["raw"]
        assert sp.issparse(raw_layer.X), "Input should be sparse"

        # Run full pipeline
        log_normalize(container, assay_name="protein", source_layer="raw", new_layer_name="log")
        assert "log" in container.assays["protein"].layers

        # Convert sparse to dense for KNN imputation (KNN doesn't support sparse)
        log_layer = container.assays["protein"].layers["log"]
        if sp.issparse(log_layer.X):
            X_dense = log_layer.X.toarray()
            container.assays["protein"].layers["log"] = ScpMatrix(X=X_dense, M=log_layer.M)

        # Create NaN values where mask is non-zero
        log_layer = container.assays["protein"].layers["log"]
        X_with_nan = log_layer.X.copy()
        missing_mask = log_layer.M != 0
        X_with_nan[missing_mask] = np.nan
        container.assays["protein"].layers["log"] = ScpMatrix(X=X_with_nan, M=log_layer.M)

        knn(container, assay_name="protein", source_layer="log", new_layer_name="imputed", k=5)
        assert "imputed" in container.assays["protein"].layers

        combat(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="imputed",
            new_layer_name="corrected",
        )
        assert "corrected" in container.assays["protein"].layers

        container = pca(
            container,
            assay_name="protein",
            base_layer_name="corrected",
            new_assay_name="pca",
            n_components=3,
        )
        assert "pca" in container.assays

        container = run_kmeans(
            container, assay_name="pca", base_layer="scores", n_clusters=2, key_added="cluster"
        )
        assert "cluster" in container.obs.columns

    def test_sparse_to_dense_conversion(self):
        """
        Test behavior when operations require dense conversion.

        Some operations may convert sparse to dense for computation.
        This test verifies the behavior is correct.
        """
        container = create_test_container(n_samples=20, n_features=30, sparse=True)

        log_normalize(container, assay_name="protein", source_layer="raw", new_layer_name="log")

        # Log layer may be dense or sparse depending on implementation
        log_layer = container.assays["protein"].layers["log"]
        assert log_layer.X.shape == (20, 30)

        # KNN requires dense - convert manually as workaround
        if sp.issparse(log_layer.X):
            X_dense = log_layer.X.toarray()
            container.assays["protein"].layers["log"] = ScpMatrix(X=X_dense, M=log_layer.M)

        # Verify conversion worked
        assert not sp.issparse(container.assays["protein"].layers["log"].X)

    def test_sparse_mask_preservation(self):
        """
        Test that mask codes are preserved with sparse matrices.

        Validates:
        - Mask is preserved when converting to/from sparse
        - Missing value codes remain intact
        """
        container = create_test_container(n_samples=20, n_features=30, sparse=True)

        original_M = container.assays["protein"].layers["raw"].M.copy()

        log_normalize(container, assay_name="protein", source_layer="raw", new_layer_name="log")

        log_M = container.assays["protein"].layers["log"].M
        assert np.array_equal(original_M, log_M), "Mask should be preserved after normalization"


# =============================================================================
# Test Mask Code Propagation
# =============================================================================


class TestMaskPropagation:
    """Test mask code propagation through pipeline steps."""

    def test_mask_preserved_through_normalization(self, small_synthetic_container):
        """
        Test that normalization preserves mask codes exactly.

        Normalization should NOT modify mask codes.
        """
        container = small_synthetic_container

        original_M = container.assays["protein"].layers["raw"].M.copy()
        original_counts = count_mask_codes(original_M)

        log_normalize(container, assay_name="protein", source_layer="raw", new_layer_name="log")

        log_M = container.assays["protein"].layers["log"].M
        log_counts = count_mask_codes(log_M)

        assert np.array_equal(original_M, log_M), "Mask should be identical after normalization"
        assert original_counts == log_counts, (
            "Mask code counts should be identical after normalization"
        )

    def test_imputation_sets_imputed_code(self, small_synthetic_container):
        """
        Test that imputation sets IMPUTED mask code for filled values.

        After KNN imputation:
        - Previously missing values should have IMPUTED code
        - Previously valid values should remain VALID
        """
        container = small_synthetic_container

        # First normalize
        log_normalize(container, assay_name="protein", source_layer="raw", new_layer_name="log")

        # Create NaN values where mask is non-zero
        log_layer = container.assays["protein"].layers["log"]
        X_with_nan = log_layer.X.copy()
        missing_mask = log_layer.M != 0
        X_with_nan[missing_mask] = np.nan
        container.assays["protein"].layers["log"] = ScpMatrix(X=X_with_nan, M=log_layer.M)

        # Track which values were missing
        M_before = container.assays["protein"].layers["log"].M.copy()
        was_missing = M_before != 0
        was_valid = M_before == 0

        # Impute
        knn(container, assay_name="protein", source_layer="log", new_layer_name="imputed", k=5)

        M_imputed = container.assays["protein"].layers["imputed"].M

        # All previously missing should now be IMPUTED
        assert np.all(M_imputed[was_missing] == MaskCode.IMPUTED), (
            "Previously missing values should have IMPUTED code"
        )

        # All previously valid should remain VALID
        assert np.all(M_imputed[was_valid] == MaskCode.VALID), (
            "Previously valid values should remain VALID"
        )

    def test_batch_correction_preserves_mask(self, small_synthetic_container):
        """
        Test that batch correction preserves mask codes.

        ComBat should not modify mask codes.
        """
        container = small_synthetic_container

        # Preprocess
        log_normalize(container, assay_name="protein", source_layer="raw", new_layer_name="log")
        knn(container, assay_name="protein", source_layer="log", new_layer_name="imputed", k=5)

        M_before = container.assays["protein"].layers["imputed"].M.copy()

        combat(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="imputed",
            new_layer_name="corrected",
        )

        M_after = container.assays["protein"].layers["corrected"].M

        assert np.array_equal(M_before, M_after), "ComBat should preserve mask codes exactly"

    def test_mask_code_valid_range_through_pipeline(self, small_synthetic_container):
        """
        Test that only valid mask codes appear throughout pipeline.

        Valid codes: 0 (VALID), 1 (MBR), 2 (LOD), 3 (FILTERED), 5 (IMPUTED)
        """
        container = small_synthetic_container

        # Run full pipeline
        log_normalize(container, assay_name="protein", source_layer="raw", new_layer_name="log")
        knn(container, assay_name="protein", source_layer="log", new_layer_name="imputed", k=5)
        combat(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="imputed",
            new_layer_name="corrected",
        )

        valid_codes = {0, 1, 2, 3, 5}

        for layer_name, layer in container.assays["protein"].layers.items():
            unique_codes = set(np.unique(layer.M))
            assert unique_codes.issubset(valid_codes), (
                f"Layer '{layer_name}' has invalid mask codes: {unique_codes - valid_codes}"
            )


# =============================================================================
# Test Provenance Logging
# =============================================================================


class TestProvenance:
    """Test provenance logging through pipeline."""

    def test_normalization_logs_history(self, small_synthetic_container):
        """Test that normalization creates a history entry."""
        container = small_synthetic_container
        initial_len = len(container.history)

        log_normalize(container, assay_name="protein", source_layer="raw", new_layer_name="log")

        assert len(container.history) == initial_len + 1, "History should have one new entry"

        last_log = container.history[-1]
        assert last_log.action == "log_transform"
        assert "assay" in last_log.params

    def test_imputation_logs_history(self, small_synthetic_container):
        """Test that imputation creates a history entry."""
        container = small_synthetic_container

        log_normalize(container, assay_name="protein", source_layer="raw", new_layer_name="log")

        # Create NaN values
        log_layer = container.assays["protein"].layers["log"]
        X_with_nan = log_layer.X.copy()
        missing_mask = log_layer.M != 0
        X_with_nan[missing_mask] = np.nan
        container.assays["protein"].layers["log"] = ScpMatrix(X=X_with_nan, M=log_layer.M)

        history_len = len(container.history)

        knn(container, assay_name="protein", source_layer="log", new_layer_name="imputed", k=5)

        assert len(container.history) == history_len + 1

        last_log = container.history[-1]
        assert last_log.action == "impute_knn"
        assert "k" in last_log.params

    def test_batch_correction_logs_history(self, small_synthetic_container):
        """Test that batch correction creates a history entry."""
        container = small_synthetic_container

        log_normalize(container, assay_name="protein", source_layer="raw", new_layer_name="log")
        knn(container, assay_name="protein", source_layer="log", new_layer_name="imputed", k=5)

        history_len = len(container.history)

        combat(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="imputed",
            new_layer_name="corrected",
        )

        assert len(container.history) == history_len + 1

        last_log = container.history[-1]
        assert last_log.action == "integration_combat"
        assert "batch_key" in last_log.params

    def test_complete_history_sequence(self, small_synthetic_container):
        """Test that complete pipeline has correct history sequence."""
        container = small_synthetic_container

        # Create NaN values for imputation
        log_normalize(container, assay_name="protein", source_layer="raw", new_layer_name="log")
        log_layer = container.assays["protein"].layers["log"]
        X_with_nan = log_layer.X.copy()
        missing_mask = log_layer.M != 0
        X_with_nan[missing_mask] = np.nan
        container.assays["protein"].layers["log"] = ScpMatrix(X=X_with_nan, M=log_layer.M)

        # Run pipeline
        knn(container, assay_name="protein", source_layer="log", new_layer_name="imputed", k=5)
        combat(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="imputed",
            new_layer_name="corrected",
        )
        container = pca(
            container,
            assay_name="protein",
            base_layer_name="corrected",
            new_assay_name="pca",
            n_components=2,
        )
        container = run_kmeans(
            container, assay_name="pca", base_layer="scores", n_clusters=2, key_added="cluster"
        )

        # Verify all actions were logged
        actions = [log.action for log in container.history]
        expected_actions = [
            "log_transform",
            "impute_knn",
            "integration_combat",
            "reduce_pca",
            "cluster_kmeans",
        ]

        for expected in expected_actions:
            assert expected in actions, f"Expected action '{expected}' not found in history"

    def test_history_params_complete(self, small_synthetic_container):
        """Test that history entries have complete parameters."""
        container = small_synthetic_container

        log_normalize(
            container,
            assay_name="protein",
            source_layer="raw",
            new_layer_name="log",
            base=2.0,
            offset=1.0,
        )

        last_log = container.history[-1]

        # Check essential params are logged
        assert "assay" in last_log.params
        assert "source_layer" in last_log.params  # norm_log uses source_layer param name
        assert "base" in last_log.params
        assert last_log.params["base"] == 2.0


# =============================================================================
# Test Different Method Combinations
# =============================================================================


class TestMethodCombinations:
    """Test different combinations of normalization, imputation, and integration."""

    def test_bpca_imputation_pipeline(self, small_synthetic_container):
        """Test pipeline with BPCA imputation instead of KNN."""
        container = small_synthetic_container

        # Normalize
        log_normalize(container, assay_name="protein", source_layer="raw", new_layer_name="log")

        # Create NaN values for BPCA
        log_layer = container.assays["protein"].layers["log"]
        X_with_nan = log_layer.X.copy()
        missing_mask = log_layer.M != 0
        X_with_nan[missing_mask] = np.nan
        container.assays["protein"].layers["log"] = ScpMatrix(X=X_with_nan, M=log_layer.M)

        # BPCA imputation
        bpca(
            container,
            assay_name="protein",
            source_layer="log",
            new_layer_name="imputed_bpca",
            n_components=5,
            random_state=42,
        )
        assert "imputed_bpca" in container.assays["protein"].layers

        # Verify IMPUTED codes
        imputed_M = container.assays["protein"].layers["imputed_bpca"].M
        assert np.all(imputed_M[missing_mask] == MaskCode.IMPUTED)

        # Continue pipeline
        combat(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="imputed_bpca",
            new_layer_name="corrected",
        )
        assert "corrected" in container.assays["protein"].layers

    @pytest.mark.skip(reason="SVD imputation not implemented - impute_svd function does not exist")
    def test_svd_imputation_pipeline(self, small_synthetic_container):
        """Test pipeline with SVD imputation."""
        container = small_synthetic_container

        log_normalize(container, assay_name="protein", source_layer="raw", new_layer_name="log")

        # Create NaN values
        log_layer = container.assays["protein"].layers["log"]
        X_with_nan = log_layer.X.copy()
        missing_mask = log_layer.M != 0
        X_with_nan[missing_mask] = np.nan
        container.assays["protein"].layers["log"] = ScpMatrix(X=X_with_nan, M=log_layer.M)

        # SVD imputation - NOT IMPLEMENTED
        # svd_impute(
        #     container,
        #     assay_name="protein",
        #     source_layer="log",
        #     new_layer_name="imputed_svd",
        #     n_components=5,
        # )
        # assert "imputed_svd" in container.assays["protein"].layers

    def test_mnn_batch_correction_pipeline(self, small_synthetic_container):
        """Test pipeline with MNN batch correction instead of ComBat."""
        container = small_synthetic_container

        log_normalize(container, assay_name="protein", source_layer="raw", new_layer_name="log")

        knn(container, assay_name="protein", source_layer="log", new_layer_name="imputed", k=5)

        # MNN correction
        result = mnn_correct(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="imputed",
            new_layer_name="mnn_corrected",
            k=10,
        )
        assert "mnn_corrected" in result.assays["protein"].layers

        # Verify mask preserved
        M_original = container.assays["protein"].layers["imputed"].M
        M_corrected = result.assays["protein"].layers["mnn_corrected"].M
        assert np.array_equal(M_original, M_corrected)

    def test_global_median_normalization_pipeline(self, small_synthetic_container):
        """Test pipeline with global median normalization (scaling mode)."""
        container = small_synthetic_container

        # Global median normalization using scaling mode
        norm_median_center(
            container,
            assay_name="protein",
            source_layer="raw",
            new_layer_name="norm",
            add_global_median=True,
        )
        assert "norm" in container.assays["protein"].layers

        # Continue pipeline
        knn(container, assay_name="protein", source_layer="norm", new_layer_name="imputed", k=5)
        combat(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="imputed",
            new_layer_name="corrected",
        )
        assert "corrected" in container.assays["protein"].layers


# =============================================================================
# Test Multi-Assay Workflows
# =============================================================================


class TestMultiAssayWorkflows:
    """Test workflows with multiple assays."""

    def test_multi_assay_container(self):
        """Test creating container with multiple assays."""
        n_samples = 20
        n_proteins = 30
        n_peptides = 50

        obs = pl.DataFrame(
            {
                "_index": [f"S{i:03d}" for i in range(n_samples)],
                "batch": (["Batch1"] * 10 + ["Batch2"] * 10),
            }
        )

        # Protein assay
        X_prot = np.random.lognormal(2, 0.5, (n_samples, n_proteins))
        M_prot = np.zeros((n_samples, n_proteins), dtype=np.int8)
        var_prot = pl.DataFrame({"_index": [f"P{i:03d}" for i in range(n_proteins)]})

        protein_assay = Assay(var=var_prot, layers={"raw": ScpMatrix(X=X_prot, M=M_prot)})

        # Peptide assay
        X_pep = np.random.lognormal(2, 0.5, (n_samples, n_peptides))
        M_pep = np.zeros((n_samples, n_peptides), dtype=np.int8)
        var_pep = pl.DataFrame({"_index": [f"PEP{i:03d}" for i in range(n_peptides)]})

        peptide_assay = Assay(var=var_pep, layers={"raw": ScpMatrix(X=X_pep, M=M_pep)})

        container = ScpContainer(
            obs=obs, assays={"proteins": protein_assay, "peptides": peptide_assay}
        )

        # Verify structure
        assert len(container.assays) == 2
        assert "proteins" in container.assays
        assert "peptides" in container.assays

    def test_process_multiple_assays(self):
        """Test processing multiple assays independently."""
        n_samples = 20

        obs = pl.DataFrame(
            {
                "_index": [f"S{i:03d}" for i in range(n_samples)],
                "batch": (["Batch1"] * 10 + ["Batch2"] * 10),
            }
        )

        # Create two assays
        assays = {}
        for assay_name, n_features in [("proteins", 30), ("peptides", 50)]:
            X = np.random.lognormal(2, 0.5, (n_samples, n_features))
            M = np.zeros((n_samples, n_features), dtype=np.int8)
            var = pl.DataFrame({"_index": [f"{assay_name[:-1]}{i:03d}" for i in range(n_features)]})

            # Add batch effect
            X[10:, :] *= 1.5

            assays[assay_name] = Assay(var=var, layers={"raw": ScpMatrix(X=X, M=M)})

        container = ScpContainer(obs=obs, assays=assays)

        # Process both assays
        for assay_name in ["proteins", "peptides"]:
            log_normalize(
                container, assay_name=assay_name, source_layer="raw", new_layer_name="log"
            )

            knn(container, assay_name=assay_name, source_layer="log", new_layer_name="imputed", k=3)

            combat(
                container,
                batch_key="batch",
                assay_name=assay_name,
                base_layer="imputed",
                new_layer_name="corrected",
            )

        # Verify both assays were processed
        for assay_name in ["proteins", "peptides"]:
            assert "log" in container.assays[assay_name].layers
            assert "imputed" in container.assays[assay_name].layers
            assert "corrected" in container.assays[assay_name].layers


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_pipeline_with_no_missing_values(self):
        """Test pipeline when there are no missing values."""
        container = create_test_container(
            n_samples=20, n_features=30, missing_rate=0.0, include_lod=False
        )

        # Verify no missing values
        assert np.all(container.assays["protein"].layers["raw"].M == 0)

        # Run pipeline - should work without issues
        log_normalize(container, assay_name="protein", source_layer="raw", new_layer_name="log")
        knn(container, assay_name="protein", source_layer="log", new_layer_name="imputed", k=5)
        combat(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="imputed",
            new_layer_name="corrected",
        )

        assert "corrected" in container.assays["protein"].layers

    def test_pipeline_with_high_missing_rate(self):
        """Test pipeline with very high missing rate (60%)."""
        container = create_test_container(n_samples=30, n_features=50, missing_rate=0.6)

        # Verify high missing rate
        M = container.assays["protein"].layers["raw"].M
        missing_rate = np.sum(M != 0) / M.size
        assert missing_rate > 0.5

        # Run pipeline - should handle high missing rate
        log_normalize(container, assay_name="protein", source_layer="raw", new_layer_name="log")
        knn(container, assay_name="protein", source_layer="log", new_layer_name="imputed", k=5)
        combat(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="imputed",
            new_layer_name="corrected",
        )

        assert "corrected" in container.assays["protein"].layers

    def test_pipeline_with_imbalanced_batches(self):
        """Test pipeline with imbalanced batch sizes."""
        n_batch1 = 40
        n_batch2 = 10
        n_features = 30

        obs = pl.DataFrame(
            {
                "_index": [f"S{i:03d}" for i in range(n_batch1 + n_batch2)],
                "batch": (["Batch1"] * n_batch1 + ["Batch2"] * n_batch2),
            }
        )

        X = np.random.lognormal(2, 0.5, (n_batch1 + n_batch2, n_features))
        X[n_batch1:, :] *= 2.0  # Add batch effect
        M = np.zeros((n_batch1 + n_batch2, n_features), dtype=np.int8)

        var = pl.DataFrame({"_index": [f"P{i:03d}" for i in range(n_features)]})

        container = ScpContainer(
            obs=obs, assays={"proteins": Assay(var=var, layers={"raw": ScpMatrix(X=X, M=M)})}
        )

        # Run pipeline
        log_normalize(container, assay_name="proteins", source_layer="raw", new_layer_name="log")
        combat(
            container,
            batch_key="batch",
            assay_name="proteins",
            base_layer="log",
            new_layer_name="corrected",
        )

        assert "corrected" in container.assays["proteins"].layers

    def test_pipeline_with_single_feature_type(self):
        """Test pipeline where all missing values are of one type (LOD)."""
        container = create_test_container(n_samples=20, n_features=30, missing_rate=0.3)

        # Set all missing to LOD
        M = container.assays["protein"].layers["raw"].M
        M[M != 0] = MaskCode.LOD

        # Run pipeline
        log_normalize(container, assay_name="protein", source_layer="raw", new_layer_name="log")

        # Create NaN values where mask is non-zero
        log_layer = container.assays["protein"].layers["log"]
        X_with_nan = log_layer.X.copy()
        missing_mask = log_layer.M != 0
        X_with_nan[missing_mask] = np.nan
        container.assays["protein"].layers["log"] = ScpMatrix(X=X_with_nan, M=log_layer.M)

        knn(container, assay_name="protein", source_layer="log", new_layer_name="imputed", k=5)

        # Verify all originally LOD values are now IMPUTED
        was_lod = M == MaskCode.LOD
        M_imputed = container.assays["protein"].layers["imputed"].M
        assert np.all(M_imputed[was_lod] == MaskCode.IMPUTED)


# =============================================================================
# Test UMAP Integration
# =============================================================================


class TestUMAPIntegration:
    """Test UMAP dimensionality reduction integration."""

    @pytest.mark.slow
    def test_umap_in_pipeline(self, small_synthetic_container):
        """Test UMAP as part of analysis pipeline."""
        container = small_synthetic_container

        # Preprocess
        log_normalize(container, assay_name="protein", source_layer="raw", new_layer_name="log")
        knn(container, assay_name="protein", source_layer="log", new_layer_name="imputed", k=5)
        combat(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="imputed",
            new_layer_name="corrected",
        )

        # First run PCA to reduce dimensions for UMAP
        container = pca(
            container,
            assay_name="protein",
            base_layer_name="corrected",
            new_assay_name="pca",
            n_components=10,
        )

        # Run UMAP on PCA scores
        container = umap(
            container,
            assay_name="pca",
            base_layer="scores",
            new_assay_name="umap",
            n_components=2,
            n_neighbors=10,
            min_dist=0.1,
            random_state=42,
        )

        # Verify UMAP assay was created
        assert "umap" in container.assays

        umap_embedding = container.assays["umap"].layers["embedding"]
        assert umap_embedding.X.shape == (container.n_samples, 2)

        # Test clustering on UMAP
        container = run_kmeans(
            container,
            assay_name="umap",
            base_layer="embedding",
            n_clusters=2,
            key_added="umap_cluster",
        )
        assert "umap_cluster" in container.obs.columns


# =============================================================================
# Test Error Handling
# =============================================================================


class TestPipelineErrorHandling:
    """Test error handling in pipeline workflows."""

    def test_error_missing_assay(self, small_synthetic_container):
        """Test error when assay doesn't exist."""
        container = small_synthetic_container

        with pytest.raises(AssayNotFoundError):
            log_normalize(
                container, assay_name="nonexistent", source_layer="raw", new_layer_name="log"
            )

    def test_error_missing_layer(self, small_synthetic_container):
        """Test error when layer doesn't exist."""
        container = small_synthetic_container

        with pytest.raises(LayerNotFoundError):
            log_normalize(
                container, assay_name="protein", source_layer="nonexistent", new_layer_name="log"
            )

    def test_error_missing_batch_column(self, small_synthetic_container):
        """Test error when batch column doesn't exist."""
        container = small_synthetic_container

        log_normalize(container, assay_name="protein", source_layer="raw", new_layer_name="log")

        with pytest.raises(ScpValueError, match="Batch key"):
            combat(
                container,
                batch_key="nonexistent_batch",
                assay_name="protein",
                base_layer="log",
                new_layer_name="corrected",
            )

    def test_error_invalid_n_clusters(self, small_synthetic_container):
        """Test error with invalid cluster count."""
        container = small_synthetic_container

        log_normalize(container, assay_name="protein", source_layer="raw", new_layer_name="log")
        container = pca(
            container,
            assay_name="protein",
            base_layer_name="log",
            new_assay_name="pca",
            n_components=2,
        )

        # n_clusters > n_samples should raise ValueError (from sklearn)
        with pytest.raises(ValueError):
            run_kmeans(
                container,
                assay_name="pca",
                base_layer="scores",
                n_clusters=100,  # More than samples
                key_added="cluster",
            )


# =============================================================================
# Test Data Integrity
# =============================================================================


class TestDataIntegrity:
    """Test data integrity throughout pipeline."""

    def test_no_nan_in_final_layers(self, small_synthetic_container):
        """Test that final pipeline layers have no NaN values."""
        container = small_synthetic_container

        # Run pipeline with KNN (should fill all NaNs)
        log_normalize(container, assay_name="protein", source_layer="raw", new_layer_name="log")
        knn(container, assay_name="protein", source_layer="log", new_layer_name="imputed", k=5)
        combat(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="imputed",
            new_layer_name="corrected",
        )

        # KNN imputed and ComBat corrected should have no NaN
        for layer_name in ["imputed", "corrected"]:
            X = container.assays["protein"].layers[layer_name].X
            if sp.issparse(X):
                assert not np.any(np.isnan(X.data)), (
                    f"Layer '{layer_name}' should not have NaN values"
                )
            else:
                assert not np.any(np.isnan(X)), f"Layer '{layer_name}' should not have NaN values"

    def test_metadata_preserved(self, small_synthetic_container):
        """Test that sample metadata is preserved."""
        container = small_synthetic_container

        original_obs = container.obs.clone()
        original_sample_ids = container.sample_ids.clone()

        # Run pipeline
        log_normalize(container, assay_name="protein", source_layer="raw", new_layer_name="log")
        knn(container, assay_name="protein", source_layer="log", new_layer_name="imputed", k=5)
        container = pca(
            container,
            assay_name="protein",
            base_layer_name="imputed",
            new_assay_name="pca",
            n_components=2,
        )

        # Check sample IDs preserved
        assert np.array_equal(container.sample_ids, original_sample_ids)

        # Check metadata columns
        assert "batch" in container.obs.columns
        assert "group" in container.obs.columns
        assert np.array_equal(container.obs["batch"], original_obs["batch"])

    def test_feature_metadata_preserved(self, small_synthetic_container):
        """Test that feature metadata is preserved."""
        container = small_synthetic_container

        original_var = container.assays["protein"].var.clone()

        # Run pipeline
        log_normalize(container, assay_name="protein", source_layer="raw", new_layer_name="log")
        knn(container, assay_name="protein", source_layer="log", new_layer_name="imputed", k=5)
        combat(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="imputed",
            new_layer_name="corrected",
        )

        # Feature metadata should be unchanged
        assert np.array_equal(container.assays["protein"].feature_ids, original_var["_index"])
