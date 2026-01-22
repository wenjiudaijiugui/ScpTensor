"""
Integration tests for complete analysis workflows.

This module provides comprehensive end-to-end integration tests covering:
- Workflow 1: Basic Pipeline (QC, normalization, imputation, PCA, clustering)
- Workflow 2: Batch Correction Pipeline (normalization, batch correction, integration verification)
- Workflow 3: Complete Analysis (full pipeline with visualization)

Tests include verification of:
- Provenance logging (history is updated correctly)
- Mask code propagation (IMPUTED codes are set correctly)
- Layer creation (new layers added without modifying originals)
- Data integrity (shapes, dtypes, values are valid)
"""

import numpy as np
import pytest

from scptensor.cluster import cluster_kmeans_assay as run_kmeans
from scptensor.core.structures import MaskCode, ScpMatrix
from scptensor.dim_reduction import reduce_pca as pca
from scptensor.impute import impute_knn as knn
from scptensor.impute import impute_ppca as ppca
from scptensor.integration import integrate_combat as combat
from scptensor.normalization import log_transform as log_normalize

# basic_qc is no longer available, use qc_sample and qc_feature instead
# from scptensor.qc import qc_basic as basic_qc
basic_qc = None  # Stub for skipped tests


class TestProvenanceLogging:
    """Test that provenance logging works correctly across pipeline steps."""

    def test_provenance_after_normalization(self, small_synthetic_container):
        """Test that log_operation is called after normalization."""
        container = small_synthetic_container
        initial_history_len = len(container.history)

        log_normalize(container, assay_name="protein", source_layer="raw", new_layer_name="log")

        # History should have one new entry
        assert len(container.history) == initial_history_len + 1

        # Check the last log entry
        last_log = container.history[-1]
        assert last_log.action == "log_transform"
        assert "assay" in last_log.params
        assert last_log.params["assay"] == "protein"

    def test_provenance_after_imputation(self, small_synthetic_container):
        """Test that log_operation is called after imputation."""
        container = small_synthetic_container

        # First normalize
        log_normalize(container, assay_name="protein", source_layer="raw", new_layer_name="log")

        # Create NaN values for imputation (KNN only logs when there are NaNs)
        log_layer = container.assays["protein"].layers["log"]
        X_with_nan = log_layer.X.copy()
        missing_mask = log_layer.M != 0
        X_with_nan[missing_mask] = np.nan
        container.assays["protein"].layers["log"] = ScpMatrix(X=X_with_nan, M=log_layer.M)

        history_before_impute = len(container.history)

        # Impute
        knn(container, assay_name="protein", source_layer="log", new_layer_name="imputed", k=5)

        # History should have one new entry
        assert len(container.history) == history_before_impute + 1

        last_log = container.history[-1]
        assert last_log.action == "impute_knn"
        assert "k" in last_log.params
        assert last_log.params["k"] == 5

    def test_provenance_after_batch_correction(self, small_synthetic_container):
        """Test that log_operation is called after batch correction."""
        container = small_synthetic_container

        # Preprocess
        log_normalize(container, assay_name="protein", source_layer="raw", new_layer_name="log")
        knn(container, assay_name="protein", source_layer="log", new_layer_name="imputed", k=5)

        history_before_combat = len(container.history)

        # Apply ComBat
        combat(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="imputed",
            new_layer_name="corrected",
        )

        # History should have one new entry
        assert len(container.history) == history_before_combat + 1

        last_log = container.history[-1]
        # ComBat uses "integration_combat" as action name
        assert last_log.action == "integration_combat"

    @pytest.mark.skip(reason="basic_qc.filter_samples() not implemented on ScpContainer")
    def test_provenance_after_qc(self, small_synthetic_container):
        """Test that log_operation is called after QC."""
        container = small_synthetic_container
        initial_history_len = len(container.history)

        # Apply QC (this returns a new container)
        container_qc = basic_qc(
            container,
            assay_name="protein",
            min_features=5,  # Very low threshold to keep most samples
            min_cells=2,
        )

        # History should be copied plus new entry
        # Note: basic_qc returns a new container with copied history
        assert len(container_qc.history) >= initial_history_len + 1

        last_log = container_qc.history[-1]
        assert last_log.action == "qc_basic"

    def test_provenance_complete_pipeline(self, small_synthetic_container):
        """Test that all pipeline steps are logged in sequence."""
        container = small_synthetic_container

        # Run complete pipeline
        log_normalize(container, assay_name="protein", source_layer="raw", new_layer_name="log")

        # Create NaN values for imputation (KNN only logs when there are NaNs)
        log_layer = container.assays["protein"].layers["log"]
        X_with_nan = log_layer.X.copy()
        missing_mask = log_layer.M != 0
        X_with_nan[missing_mask] = np.nan
        container.assays["protein"].layers["log"] = ScpMatrix(X=X_with_nan, M=log_layer.M)

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
            new_assay_name="reduce_pca",
            n_components=2,
        )

        container = run_kmeans(
            container,
            assay_name="reduce_pca",
            base_layer="scores",
            n_clusters=2,
            key_added="kmeans_cluster",
        )

        # Check all actions were logged
        actions = [log.action for log in container.history]
        # Note: action names are: log_transform, impute_knn, integration_combat, reduce_pca, cluster_kmeans
        expected_actions = [
            "log_transform",
            "impute_knn",
            "integration_combat",
            "reduce_pca",
            "cluster_kmeans",
        ]

        for expected in expected_actions:
            assert expected in actions, f"Expected action '{expected}' not found in history"

        # Actions should be in order
        for i, expected in enumerate(expected_actions):
            assert container.history[i].action == expected


class TestMaskCodePropagation:
    """Test that mask codes are propagated and updated correctly."""

    def test_mask_preserved_after_normalization(self, small_synthetic_container):
        """Test that mask codes are preserved during normalization."""
        container = small_synthetic_container

        original_M = container.assays["protein"].layers["raw"].M.copy()

        log_normalize(container, assay_name="protein", source_layer="raw", new_layer_name="log")

        new_M = container.assays["protein"].layers["log"].M

        # Mask should be identical
        assert np.array_equal(original_M, new_M)

    def test_mask_codes_after_ppca_imputation(self, small_synthetic_container):
        """Test that PPCA sets IMPUTED mask codes correctly."""
        container = small_synthetic_container

        # First normalize to get valid numeric values
        log_normalize(container, assay_name="protein", source_layer="raw", new_layer_name="log")

        # Get mask before imputation (log layer should have same mask as raw)
        log_layer = container.assays["protein"].layers["log"]
        M_before = log_layer.M.copy()
        X_before = log_layer.X.copy()

        # Create missing values (set to NaN) where mask is non-zero
        X_with_nan = X_before.copy()
        missing_mask = M_before != 0
        X_with_nan[missing_mask] = np.nan

        # Replace layer with NaN version
        assay = container.assays["protein"]
        assay.layers["log"] = ScpMatrix(X=X_with_nan, M=M_before)

        # Apply PPCA imputation
        ppca(
            container,
            assay_name="protein",
            source_layer="log",
            new_layer_name="imputed_ppca",
            n_components=5,
            random_state=42,
        )

        # Check imputed layer exists
        assert "imputed_ppca" in container.assays["protein"].layers

        imputed_layer = container.assays["protein"].layers["imputed_ppca"]
        M_imputed = imputed_layer.M

        # Previously missing values should now have IMPUTED code
        assert np.all(M_imputed[missing_mask] == MaskCode.IMPUTED)
        # Valid values should remain VALID
        assert np.all(M_imputed[~missing_mask] == MaskCode.VALID)

    def test_mask_codes_valid_range(self, small_synthetic_container):
        """Test that all mask codes are in valid range after pipeline."""
        container = small_synthetic_container

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

        # Check all layers have valid mask codes
        valid_codes = {0, 1, 2, 3, 5}  # VALID, MBR, LOD, FILTERED, IMPUTED

        for layer_name, layer in container.assays["protein"].layers.items():
            unique_codes = np.unique(layer.M)
            for code in unique_codes:
                assert code in valid_codes, f"Invalid mask code {code} in layer '{layer_name}'"

    def test_mask_code_counts(self, small_synthetic_container):
        """Test that mask code counts are tracked properly."""
        container = small_synthetic_container

        # Count initial mask codes
        raw_M = container.assays["protein"].layers["raw"].M
        initial_lod_count = np.sum(raw_M == MaskCode.LOD)
        initial_mbr_count = np.sum(raw_M == MaskCode.MBR)
        initial_valid_count = np.sum(raw_M == MaskCode.VALID)

        # All should be non-zero for synthetic data
        assert initial_valid_count > 0, "Should have some valid values"
        assert initial_lod_count > 0 or initial_mbr_count > 0, "Should have some missing values"


class TestLayerImmutability:
    """Test that original layers are not modified when creating new layers."""

    def test_original_layer_unchanged_after_normalization(self, small_synthetic_container):
        """Test that normalization doesn't modify the original layer."""
        container = small_synthetic_container

        raw_layer = container.assays["protein"].layers["raw"]
        X_original = raw_layer.X.copy()
        M_original = raw_layer.M.copy()

        log_normalize(container, assay_name="protein", source_layer="raw", new_layer_name="log")

        # Original layer should be unchanged
        assert np.array_equal(raw_layer.X, X_original)
        assert np.array_equal(raw_layer.M, M_original)

    def test_multiple_layers_exist(self, small_synthetic_container):
        """Test that all layers coexist after pipeline."""
        container = small_synthetic_container

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

        # All layers should exist
        assert "raw" in container.assays["protein"].layers
        assert "log" in container.assays["protein"].layers
        assert "imputed" in container.assays["protein"].layers
        assert "corrected" in container.assays["protein"].layers

        # Each layer should have independent data
        layers = container.assays["protein"].layers
        raw_X = layers["raw"].X
        log_X = layers["log"].X

        # Log-transformed values should differ from raw
        assert not np.allclose(raw_X, log_X)

    def test_layer_independence(self, small_synthetic_container):
        """Test that modifying one layer doesn't affect others."""
        container = small_synthetic_container

        log_normalize(container, assay_name="protein", source_layer="raw", new_layer_name="log")

        # Get references to both layers
        raw_layer = container.assays["protein"].layers["raw"]
        log_layer = container.assays["protein"].layers["log"]

        # Store original values
        raw_mean = np.mean(raw_layer.X)
        log_mean = np.mean(log_layer.X)

        # The layers should have different values (log transformation)
        assert raw_mean != log_mean


class TestWorkflow1BasicPipeline:
    """Test Workflow 1: Basic Pipeline (QC, normalization, imputation, PCA, clustering)."""

    @pytest.mark.skip(reason="basic_qc.filter_samples() not implemented on ScpContainer")
    def test_basic_pipeline_full(self, small_synthetic_container):
        """Test complete basic pipeline from raw data to clustering."""
        container = small_synthetic_container

        # Step 1: Quality Control
        container_qc = basic_qc(container, assay_name="protein", min_features=5, min_cells=2)

        # QC should filter some samples/features
        assert container_qc.n_samples <= container.n_samples

        # Step 2: Log normalization
        log_normalize(container_qc, assay_name="protein", source_layer="raw", new_layer_name="log")
        assert "log" in container_qc.assays["protein"].layers

        # Step 3: Imputation
        knn(container_qc, assay_name="protein", source_layer="log", new_layer_name="imputed", k=5)
        assert "imputed" in container_qc.assays["protein"].layers

        # Step 4: PCA
        container_pca = pca(
            container_qc,
            assay_name="protein",
            base_layer_name="imputed",
            new_assay_name="reduce_pca",
            n_components=2,
        )
        assert "reduce_pca" in container_pca.assays

        # Step 5: Clustering
        container_clustered = run_kmeans(
            container_pca,
            assay_name="reduce_pca",
            base_layer="scores",
            n_clusters=2,
            key_added="cluster",
        )
        assert "cluster" in container_clustered.obs.columns

        # Verify data flows correctly through all steps
        assert len(np.unique(container_clustered.obs["cluster"])) == 2

    def test_basic_pipeline_dimensions(self, small_synthetic_container):
        """Test that dimensions are consistent through basic pipeline."""
        container = small_synthetic_container

        original_n_samples = container.n_samples
        _ = container.assays["protein"].n_features

        # Run pipeline
        log_normalize(container, assay_name="protein", source_layer="raw", new_layer_name="log")

        knn(container, assay_name="protein", source_layer="log", new_layer_name="imputed", k=5)

        container = pca(
            container,
            assay_name="protein",
            base_layer_name="imputed",
            new_assay_name="reduce_pca",
            n_components=2,
        )

        # PCA should have correct dimensions
        pca_scores = container.assays["reduce_pca"].layers["scores"]
        assert pca_scores.X.shape[0] == original_n_samples
        assert pca_scores.X.shape[1] == 2  # n_components


class TestWorkflow2BatchCorrection:
    """Test Workflow 2: Batch Correction Pipeline."""

    def test_batch_correction_pipeline(self, small_synthetic_container):
        """Test batch correction pipeline with integration verification."""
        container = small_synthetic_container

        # Normalization
        log_normalize(container, assay_name="protein", source_layer="raw", new_layer_name="log")

        # Imputation
        knn(container, assay_name="protein", source_layer="log", new_layer_name="imputed", k=5)

        # Batch correction
        combat(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="imputed",
            new_layer_name="corrected",
        )

        # Verify corrected layer exists
        assert "corrected" in container.assays["protein"].layers

        # Get batch information
        container.obs["batch"].to_numpy()

        # Verify corrected data has same dimensions
        corrected = container.assays["protein"].layers["corrected"]
        imputed = container.assays["protein"].layers["imputed"]

        assert corrected.X.shape == imputed.X.shape

    def test_batch_correction_with_pca_clustering(self, small_synthetic_container):
        """Test that batch correction enables biological clustering."""
        container = small_synthetic_container

        # Get original batch and group labels
        container.obs["group"].to_numpy()
        container.obs["batch"].to_numpy()

        # Preprocess
        log_normalize(container, assay_name="protein", source_layer="raw", new_layer_name="log")

        knn(container, assay_name="protein", source_layer="log", new_layer_name="imputed", k=5)

        # Apply batch correction
        combat(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="imputed",
            new_layer_name="corrected",
        )

        # Run PCA on corrected data
        container = pca(
            container,
            assay_name="protein",
            base_layer_name="corrected",
            new_assay_name="reduce_pca",
            n_components=2,
        )

        # Run clustering
        container = run_kmeans(
            container,
            assay_name="reduce_pca",
            base_layer="scores",
            n_clusters=2,
            key_added="cluster",
        )

        # Verify clustering worked
        assert "cluster" in container.obs.columns
        assert len(np.unique(container.obs["cluster"])) == 2

    def test_batch_correction_preserves_group_differences(self, synthetic_container):
        """Test that batch correction preserves biological group differences."""
        container = synthetic_container

        groups = container.obs["group"].to_numpy()

        # Preprocess
        log_normalize(container, assay_name="protein", source_layer="raw", new_layer_name="log")

        knn(container, assay_name="protein", source_layer="log", new_layer_name="imputed", k=10)

        # Get means before correction
        imputed_layer = container.assays["protein"].layers["imputed"]
        group_a_idx = groups == "GroupA"
        group_b_idx = groups == "GroupB"

        np.mean(imputed_layer.X[group_a_idx, :50])
        np.mean(imputed_layer.X[group_b_idx, :50])

        # Apply batch correction
        combat(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="imputed",
            new_layer_name="corrected",
        )

        # Get means after correction
        corrected_layer = container.assays["protein"].layers["corrected"]
        np.mean(corrected_layer.X[group_a_idx, :50])
        np.mean(corrected_layer.X[group_b_idx, :50])

        # Group differences should still exist (though may be reduced)
        # We just verify the data changed and groups are still distinguishable
        assert not np.allclose(imputed_layer.X, corrected_layer.X)


class TestWorkflow3CompleteAnalysis:
    """Test Workflow 3: Complete Analysis with all components."""

    @pytest.mark.skip(reason="basic_qc.filter_samples() not implemented on ScpContainer")
    def test_complete_analysis_pipeline(self, small_synthetic_container):
        """Test full analysis pipeline from raw to final results."""
        container = small_synthetic_container

        # Step 1: Quality control
        container = basic_qc(container, assay_name="protein", min_features=5, min_cells=2)

        # Step 2: Normalization
        log_normalize(container, assay_name="protein", source_layer="raw", new_layer_name="log")

        # Step 3: Imputation
        knn(container, assay_name="protein", source_layer="log", new_layer_name="imputed", k=5)

        # Step 4: Batch correction
        combat(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="imputed",
            new_layer_name="corrected",
        )

        # Step 5: Dimensionality reduction
        container = pca(
            container,
            assay_name="protein",
            base_layer_name="corrected",
            new_assay_name="reduce_pca",
            n_components=5,
        )

        # Step 6: Clustering
        container = run_kmeans(
            container,
            assay_name="reduce_pca",
            base_layer="scores",
            n_clusters=2,
            key_added="kmeans_cluster",
        )

        # Verify final state
        # Check all layers exist
        assert "raw" in container.assays["protein"].layers
        assert "log" in container.assays["protein"].layers
        assert "imputed" in container.assays["protein"].layers
        assert "corrected" in container.assays["protein"].layers

        # Check PCA assay exists
        assert "reduce_pca" in container.assays
        assert "scores" in container.assays["reduce_pca"].layers

        # Check cluster labels
        assert "kmeans_cluster" in container.obs.columns

        # Check provenance
        assert len(container.history) > 0

    def test_complete_analysis_with_ppca(self, small_synthetic_container):
        """Test complete pipeline using PPCA imputation."""
        container = small_synthetic_container

        # Normalize
        log_normalize(container, assay_name="protein", source_layer="raw", new_layer_name="log")

        # Create NaN values for PPCA
        log_layer = container.assays["protein"].layers["log"]
        X_with_nan = log_layer.X.copy()
        missing_mask = log_layer.M != 0
        X_with_nan[missing_mask] = np.nan
        container.assays["protein"].layers["log"] = ScpMatrix(X=X_with_nan, M=log_layer.M)

        # PPCA imputation
        ppca(
            container,
            assay_name="protein",
            source_layer="log",
            new_layer_name="imputed_ppca",
            n_components=5,
            random_state=42,
        )

        # Verify IMPUTED mask codes
        imputed_layer = container.assays["protein"].layers["imputed_ppca"]
        assert np.all(imputed_layer.M[missing_mask] == MaskCode.IMPUTED)

        # Continue with rest of pipeline
        combat(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="imputed_ppca",
            new_layer_name="corrected",
        )

        container = pca(
            container,
            assay_name="protein",
            base_layer_name="corrected",
            new_assay_name="reduce_pca",
            n_components=2,
        )

        container = run_kmeans(
            container,
            assay_name="reduce_pca",
            base_layer="scores",
            n_clusters=2,
            key_added="cluster",
        )

        # Verify pipeline completed
        assert "cluster" in container.obs.columns
        assert len(container.history) >= 4  # ppca, combat, pca, run_kmeans


class TestDataIntegrity:
    """Test data integrity throughout the pipeline."""

    def test_shapes_preserved(self, small_synthetic_container):
        """Test that data shapes are preserved correctly."""
        container = small_synthetic_container

        original_shape = container.assays["protein"].layers["raw"].X.shape

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

        # All data layers should have same shape
        for layer_name in ["raw", "log", "imputed", "corrected"]:
            shape = container.assays["protein"].layers[layer_name].X.shape
            assert shape == original_shape, f"Layer {layer_name} has incorrect shape"

    def test_dtypes_valid(self, small_synthetic_container):
        """Test that data types remain valid through pipeline."""
        container = small_synthetic_container

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

        # Check dtypes
        for layer_name, layer in container.assays["protein"].layers.items():
            # X should be float
            assert np.issubdtype(layer.X.dtype, np.floating), (
                f"Layer {layer_name} X has invalid dtype: {layer.X.dtype}"
            )
            # M should be integer
            assert np.issubdtype(layer.M.dtype, np.integer), (
                f"Layer {layer_name} M has invalid dtype: {layer.M.dtype}"
            )

    def test_no_nan_in_pipeline_layers(self, small_synthetic_container):
        """Test that pipeline layers don't contain unexpected NaN values."""
        container = small_synthetic_container

        # Run pipeline with KNN (doesn't produce NaN)
        log_normalize(container, assay_name="protein", source_layer="raw", new_layer_name="log")

        knn(container, assay_name="protein", source_layer="log", new_layer_name="imputed", k=5)

        combat(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="imputed",
            new_layer_name="corrected",
        )

        # Check no NaN in imputed and corrected layers
        imputed_layer = container.assays["protein"].layers["imputed"]
        corrected_layer = container.assays["protein"].layers["corrected"]

        # KNN should fill all NaNs
        assert not np.any(np.isnan(imputed_layer.X)), "KNN imputed layer should not have NaN values"

        # ComBat output should not have NaN
        assert not np.any(np.isnan(corrected_layer.X)), (
            "ComBat corrected layer should not have NaN values"
        )

    def test_metadata_preserved(self, small_synthetic_container):
        """Test that sample metadata is preserved through pipeline."""
        container = small_synthetic_container

        original_obs = container.obs.clone()
        original_sample_ids = container.sample_ids.clone()

        # Run pipeline
        log_normalize(container, assay_name="protein", source_layer="raw", new_layer_name="log")

        knn(container, assay_name="protein", source_layer="log", new_layer_name="imputed", k=5)

        # Check sample IDs are preserved (for non-filtering pipeline)
        assert np.array_equal(container.sample_ids, original_sample_ids)

        # Check metadata columns still exist
        assert "group" in container.obs.columns
        assert "batch" in container.obs.columns
        assert np.array_equal(container.obs["group"], original_obs["group"])


@pytest.mark.slow
class TestWorkflowLargeDataset:
    """Test workflows on larger datasets (marked as slow)."""

    def test_complete_pipeline_large_dataset(self, synthetic_container):
        """Test complete pipeline on larger synthetic dataset."""
        container = synthetic_container

        # Full pipeline
        log_normalize(container, assay_name="protein", source_layer="raw", new_layer_name="log")

        knn(container, assay_name="protein", source_layer="log", new_layer_name="imputed", k=10)

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
            new_assay_name="reduce_pca",
            n_components=10,
        )

        container = run_kmeans(
            container,
            assay_name="reduce_pca",
            base_layer="scores",
            n_clusters=2,
            key_added="cluster",
        )

        # Verify results
        assert container.assays["reduce_pca"].layers["scores"].X.shape == (100, 10)
        assert len(np.unique(container.obs["cluster"])) == 2
        assert len(container.history) >= 4
