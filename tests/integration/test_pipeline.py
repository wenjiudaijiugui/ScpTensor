"""
Integration tests for complete analysis pipeline.

This module tests the end-to-end workflow including:
1. Data loading/validation
2. Quality control
3. Normalization
4. Imputation
5. Batch correction
6. Dimensionality reduction
7. Clustering
"""

import numpy as np
import pytest

from scptensor.cluster import run_kmeans
from scptensor.core.exceptions import ScpTensorError
from scptensor.dim_reduction import pca
from scptensor.impute import knn
from scptensor.integration import combat
from scptensor.normalization import log_normalize


class TestPipelineBasic:
    """Test basic pipeline functionality with synthetic data."""

    def test_pipeline_synthetic_data(self, small_synthetic_container):
        """Test complete pipeline on synthetic data."""
        container = small_synthetic_container

        # Verify initial state
        assert container.n_samples == 20
        assert "protein" in container.assays
        assert "raw" in container.assays["protein"].layers

        # Step 1: Log normalization
        log_normalize(container, assay_name="protein", base_layer="raw", new_layer_name="log")

        # Verify log layer was created
        assert "log" in container.assays["protein"].layers
        log_layer = container.assays["protein"].layers["log"]
        assert log_layer.X.shape == (20, 50)

        # Verify normalization worked (values should be log-transformed)
        raw_mean = np.mean(container.assays["protein"].layers["raw"].X)
        log_mean = np.mean(log_layer.X[log_layer.M == 0])
        assert log_mean < raw_mean  # Log should reduce scale

        # Step 2: KNN imputation
        knn(container, assay_name="protein", base_layer="log", new_layer_name="imputed", k=5)

        # Verify imputed layer was created
        assert "imputed" in container.assays["protein"].layers
        imputed_layer = container.assays["protein"].layers["imputed"]
        assert imputed_layer.X.shape == (20, 50)

        # Step 3: Batch correction (ComBat)
        combat(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="imputed",
            new_layer_name="corrected",
        )

        # Verify corrected layer was created
        assert "corrected" in container.assays["protein"].layers
        corrected_layer = container.assays["protein"].layers["corrected"]
        assert corrected_layer.X.shape == (20, 50)

        # Step 4: PCA
        container = pca(
            container,
            assay_name="protein",
            base_layer_name="corrected",
            new_assay_name="pca",
            n_components=2,
        )

        # Verify PCA assay was created
        assert "pca" in container.assays
        assert "scores" in container.assays["pca"].layers
        pca_scores = container.assays["pca"].layers["scores"]
        assert pca_scores.X.shape == (20, 2)

        # Step 5: Clustering
        container = run_kmeans(
            container,
            assay_name="pca",
            base_layer="scores",
            n_clusters=2,
            key_added="kmeans_cluster",
        )

        # Verify cluster labels were added
        assert "kmeans_cluster" in container.obs.columns
        clusters = container.obs["kmeans_cluster"].to_list()
        assert len(clusters) == 20
        assert len(np.unique(clusters)) == 2

    def test_pipeline_without_batch_correction(self, small_synthetic_container):
        """Test pipeline without batch correction step."""
        container = small_synthetic_container

        # Normalization
        log_normalize(container, assay_name="protein", base_layer="raw", new_layer_name="log")
        assert "log" in container.assays["protein"].layers

        # Imputation
        knn(container, assay_name="protein", base_layer="log", new_layer_name="imputed", k=5)
        assert "imputed" in container.assays["protein"].layers

        # Skip batch correction, go directly to PCA
        container = pca(
            container,
            assay_name="protein",
            base_layer_name="imputed",
            new_assay_name="pca",
            n_components=2,
        )
        assert "pca" in container.assays

        # Clustering
        container = run_kmeans(
            container, assay_name="pca", base_layer="scores", n_clusters=2, key_added="cluster"
        )
        assert "cluster" in container.obs.columns


class TestPipelineNormalization:
    """Test normalization step in pipeline."""

    def test_log_normalize_in_pipeline(self, small_synthetic_container):
        """Test log normalization creates proper layer."""
        container = small_synthetic_container

        log_normalize(
            container,
            assay_name="protein",
            base_layer="raw",
            new_layer_name="log",
            base=2.0,
            offset=1.0,
        )

        # Verify layer exists
        assert "log" in container.assays["protein"].layers

        # Verify mask was preserved
        raw_M = container.assays["protein"].layers["raw"].M
        log_M = container.assays["protein"].layers["log"].M
        assert np.array_equal(raw_M, log_M)

    def test_normalization_preserves_dimensions(self, small_synthetic_container):
        """Test that normalization preserves data dimensions."""
        container = small_synthetic_container

        original_shape = container.assays["protein"].layers["raw"].X.shape

        log_normalize(container, assay_name="protein", base_layer="raw", new_layer_name="log")

        new_shape = container.assays["protein"].layers["log"].X.shape
        assert original_shape == new_shape


class TestPipelineImputation:
    """Test imputation step in pipeline."""

    def test_knn_imputation_in_pipeline(self, small_synthetic_container):
        """Test KNN imputation in pipeline context."""
        container = small_synthetic_container

        # First normalize
        log_normalize(container, assay_name="protein", base_layer="raw", new_layer_name="log")

        # Count missing values before imputation
        log_layer = container.assays["protein"].layers["log"]
        missing_before = np.sum(log_layer.M != 0)

        # Impute
        knn(container, assay_name="protein", base_layer="log", new_layer_name="imputed", k=5)

        # Verify imputed layer exists
        assert "imputed" in container.assays["protein"].layers

        # Verify missing values were filled
        imputed_layer = container.assays["protein"].layers["imputed"]
        missing_after = np.sum(imputed_layer.M != 0)

        # Imputation should reduce missing values
        assert missing_after <= missing_before

    def test_imputation_preserves_dimensions(self, small_synthetic_container):
        """Test that imputation preserves data dimensions."""
        container = small_synthetic_container

        log_normalize(container, assay_name="protein", base_layer="raw", new_layer_name="log")

        original_shape = container.assays["protein"].layers["log"].X.shape

        knn(container, assay_name="protein", base_layer="log", new_layer_name="imputed", k=5)

        new_shape = container.assays["protein"].layers["imputed"].X.shape
        assert original_shape == new_shape


class TestPipelineBatchCorrection:
    """Test batch correction step in pipeline."""

    def test_combat_in_pipeline(self, small_synthetic_container):
        """Test ComBat batch correction in pipeline."""
        container = small_synthetic_container

        # Preprocess
        log_normalize(container, assay_name="protein", base_layer="raw", new_layer_name="log")

        knn(container, assay_name="protein", base_layer="log", new_layer_name="imputed", k=5)

        # Apply ComBat
        combat(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="imputed",
            new_layer_name="corrected",
        )

        # Verify corrected layer
        assert "corrected" in container.assays["protein"].layers
        corrected = container.assays["protein"].layers["corrected"]
        assert corrected.X.shape == (20, 50)

    def test_combat_preserves_dimensions(self, small_synthetic_container):
        """Test that ComBat preserves data dimensions."""
        container = small_synthetic_container

        log_normalize(container, assay_name="protein", base_layer="raw", new_layer_name="log")

        knn(container, assay_name="protein", base_layer="log", new_layer_name="imputed", k=3)

        original_shape = container.assays["protein"].layers["imputed"].X.shape

        combat(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="imputed",
            new_layer_name="corrected",
        )

        new_shape = container.assays["protein"].layers["corrected"].X.shape
        assert original_shape == new_shape


class TestPipelineDimensionalityReduction:
    """Test dimensionality reduction in pipeline."""

    def test_pca_in_pipeline(self, small_synthetic_container):
        """Test PCA in pipeline context."""
        container = small_synthetic_container

        # Preprocess
        log_normalize(container, assay_name="protein", base_layer="raw", new_layer_name="log")

        knn(container, assay_name="protein", base_layer="log", new_layer_name="imputed", k=5)

        combat(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="imputed",
            new_layer_name="corrected",
        )

        # Apply PCA
        container = pca(
            container,
            assay_name="protein",
            base_layer_name="corrected",
            new_assay_name="pca",
            n_components=5,
        )

        # Verify PCA results
        assert "pca" in container.assays
        assert "scores" in container.assays["pca"].layers

        pca_scores = container.assays["pca"].layers["scores"]
        assert pca_scores.X.shape == (20, 5)  # n_samples x n_components

    def test_pca_component_count(self, small_synthetic_container):
        """Test PCA with different component counts."""
        container = small_synthetic_container

        log_normalize(container, assay_name="protein", base_layer="raw", new_layer_name="log")

        knn(container, assay_name="protein", base_layer="log", new_layer_name="imputed", k=5)

        # Test with 2 components
        container = pca(
            container,
            assay_name="protein",
            base_layer_name="imputed",
            new_assay_name="pca2",
            n_components=2,
        )
        assert container.assays["pca2"].layers["scores"].X.shape[1] == 2

        # Test with 10 components
        container = pca(
            container,
            assay_name="protein",
            base_layer_name="imputed",
            new_assay_name="pca10",
            n_components=10,
        )
        assert container.assays["pca10"].layers["scores"].X.shape[1] == 10


class TestPipelineClustering:
    """Test clustering step in pipeline."""

    def test_kmeans_in_pipeline(self, small_synthetic_container):
        """Test K-means clustering in pipeline."""
        container = small_synthetic_container

        # Preprocess
        log_normalize(container, assay_name="protein", base_layer="raw", new_layer_name="log")

        knn(container, assay_name="protein", base_layer="log", new_layer_name="imputed", k=5)

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

        # Apply K-means
        container = run_kmeans(
            container,
            assay_name="pca",
            base_layer="scores",
            n_clusters=3,
            key_added="kmeans_cluster",
        )

        # Verify results
        assert "kmeans_cluster" in container.obs.columns
        clusters = container.obs["kmeans_cluster"].to_numpy()
        assert len(clusters) == 20
        assert len(np.unique(clusters)) == 3

    def test_kmeans_different_cluster_counts(self, small_synthetic_container):
        """Test K-means with different cluster counts."""
        container = small_synthetic_container

        # Preprocess
        log_normalize(container, assay_name="protein", base_layer="raw", new_layer_name="log")

        knn(container, assay_name="protein", base_layer="log", new_layer_name="imputed", k=5)

        container = pca(
            container,
            assay_name="protein",
            base_layer_name="imputed",
            new_assay_name="pca",
            n_components=2,
        )

        # Test with 2 clusters
        container = run_kmeans(
            container, assay_name="pca", base_layer="scores", n_clusters=2, key_added="cluster2"
        )
        assert len(np.unique(container.obs["cluster2"])) == 2

        # Test with 4 clusters - use different assay name to avoid conflict
        container = run_kmeans(
            container,
            assay_name="pca",
            base_layer="scores",
            n_clusters=4,
            key_added="cluster4",
            new_assay_name="kmeans4",
        )
        assert len(np.unique(container.obs["cluster4"])) == 4


class TestPipelineErrorHandling:
    """Test error handling in pipeline."""

    def test_pipeline_missing_assay(self, small_synthetic_container):
        """Test that pipeline fails gracefully with missing assay."""
        container = small_synthetic_container

        with pytest.raises(ScpTensorError, match="Assay 'nonexistent' not found"):
            log_normalize(
                container, assay_name="nonexistent", base_layer="raw", new_layer_name="log"
            )

    def test_pipeline_missing_layer(self, small_synthetic_container):
        """Test that pipeline fails gracefully with missing layer."""
        container = small_synthetic_container

        with pytest.raises(ScpTensorError, match="Layer 'nonexistent' not found"):
            log_normalize(
                container, assay_name="protein", base_layer="nonexistent", new_layer_name="log"
            )

    def test_pipeline_missing_batch_column(self, small_synthetic_container):
        """Test that ComBat fails gracefully without batch column."""
        container = small_synthetic_container

        log_normalize(container, assay_name="protein", base_layer="raw", new_layer_name="log")

        knn(container, assay_name="protein", base_layer="log", new_layer_name="imputed", k=5)

        # Try to use non-existent batch column
        with pytest.raises(ScpTensorError, match="Batch key 'nonexistent_batch' not found"):
            combat(
                container,
                batch_key="nonexistent_batch",
                assay_name="protein",
                base_layer="imputed",
                new_layer_name="corrected",
            )


@pytest.mark.slow
class TestPipelineLargeDataset:
    """Test pipeline on larger synthetic dataset (marked as slow)."""

    def test_pipeline_large_synthetic(self, synthetic_container):
        """Test complete pipeline on larger synthetic dataset."""
        container = synthetic_container

        # Verify initial state
        assert container.n_samples == 100
        assert "protein" in container.assays

        # Full pipeline
        log_normalize(container, assay_name="protein", base_layer="raw", new_layer_name="log")
        assert "log" in container.assays["protein"].layers

        knn(container, assay_name="protein", base_layer="log", new_layer_name="imputed", k=10)
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
            n_components=10,
        )
        assert "pca" in container.assays
        assert container.assays["pca"].layers["scores"].X.shape == (100, 10)

        container = run_kmeans(
            container,
            assay_name="pca",
            base_layer="scores",
            n_clusters=2,
            key_added="kmeans_cluster",
        )
        assert "kmeans_cluster" in container.obs.columns
        assert len(np.unique(container.obs["kmeans_cluster"])) == 2
