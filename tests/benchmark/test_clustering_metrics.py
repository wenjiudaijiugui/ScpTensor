"""Tests for clustering metrics evaluator."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from scptensor.benchmark.evaluators.clustering_metrics import (
    ClusteringEvaluator,
    compare_pca_variance_explained,
    compare_umap_embedding_quality,
    compute_clustering_ari,
    compute_clustering_nmi,
    compute_clustering_silhouette,
)


class TestClusteringMetrics:
    """Test clustering metric functions."""

    def test_compute_clustering_ari(self):
        """Test ARI computation."""
        # Perfect match
        labels_true = np.array([0, 0, 1, 1, 2, 2])
        labels_pred = np.array([0, 0, 1, 1, 2, 2])
        ari = compute_clustering_ari(labels_true, labels_pred)
        assert ari == pytest.approx(1.0)

        # Random match
        labels_pred = np.array([0, 1, 2, 0, 1, 2])
        ari = compute_clustering_ari(labels_true, labels_pred)
        assert ari < 1.0

    def test_compute_clustering_nmi(self):
        """Test NMI computation."""
        labels_true = np.array([0, 0, 1, 1, 2, 2])
        labels_pred = np.array([0, 0, 1, 1, 2, 2])
        nmi = compute_clustering_nmi(labels_true, labels_pred)
        assert nmi == pytest.approx(1.0)

    def test_compute_clustering_silhouette(self):
        """Test silhouette score computation."""
        # Well-separated clusters
        np.random.seed(42)
        X = np.vstack(
            [
                np.random.randn(50, 5) + 2,  # Cluster 0
                np.random.randn(50, 5) - 2,  # Cluster 1
            ]
        )
        labels = np.array([0] * 50 + [1] * 50)

        score = compute_clustering_silhouette(X, labels)
        assert 0 < score <= 1.0

    def test_compute_clustering_silhouette_sparse(self):
        """Test silhouette score with sparse matrix."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        X_sparse = csr_matrix(X)
        labels = np.random.randint(0, 3, 100)

        score = compute_clustering_silhouette(X_sparse, labels)
        assert -1 <= score <= 1

    def test_compare_pca_variance_explained(self):
        """Test PCA variance comparison."""
        var_scpt = np.array([0.3, 0.2, 0.15, 0.1, 0.05])
        var_scanpy = np.array([0.31, 0.19, 0.16, 0.09, 0.06])

        metrics = compare_pca_variance_explained(var_scpt, var_scanpy, n_components=5)

        assert "pearson_correlation" in metrics
        assert "spearman_correlation" in metrics
        assert "mse" in metrics
        assert "mae" in metrics
        assert metrics["pearson_correlation"] > 0.9  # Should be highly correlated

    def test_compare_umap_embedding_quality(self):
        """Test UMAP embedding comparison."""
        np.random.seed(42)
        embed_scpt = np.random.randn(100, 2)
        # Similar embedding with small noise
        embed_scanpy = embed_scpt + np.random.randn(100, 2) * 0.1

        metrics = compare_umap_embedding_quality(embed_scpt, embed_scanpy)

        assert "procrustes_distance" in metrics
        assert "pearson_correlation" in metrics
        assert "local_structure_preservation" in metrics

    def test_compare_umap_with_labels(self):
        """Test UMAP comparison with labels."""
        np.random.seed(42)
        embed_scpt = np.random.randn(100, 2)
        embed_scanpy = embed_scpt + np.random.randn(100, 2) * 0.1
        labels = np.random.randint(0, 3, 100)

        metrics = compare_umap_embedding_quality(embed_scpt, embed_scanpy, labels=labels)

        assert "label_consistency_ari" in metrics
        assert "label_consistency_nmi" in metrics


class TestClusteringEvaluator:
    """Test ClusteringEvaluator class."""

    def test_init(self):
        """Test evaluator initialization."""
        evaluator = ClusteringEvaluator()
        assert "ari" in evaluator.supported_metrics
        assert "nmi" in evaluator.supported_metrics
        assert "silhouette" in evaluator.supported_metrics

    def test_evaluate_clustering_basic(self):
        """Test basic clustering evaluation."""
        evaluator = ClusteringEvaluator()
        labels_true = np.array([0, 0, 1, 1, 2, 2])
        labels_pred = np.array([0, 0, 1, 2, 2, 2])

        metrics = evaluator.evaluate_clustering(labels_true, labels_pred)

        assert "ari" in metrics
        assert "nmi" in metrics

    def test_evaluate_clustering_with_silhouette(self):
        """Test clustering evaluation with silhouette."""
        evaluator = ClusteringEvaluator()
        np.random.seed(42)
        X = np.random.randn(100, 10)
        labels_true = np.random.randint(0, 3, 100)
        labels_pred = np.random.randint(0, 3, 100)

        metrics = evaluator.evaluate_clustering(labels_true, labels_pred, X=X)

        assert "silhouette" in metrics
        assert -1 <= metrics["silhouette"] <= 1

    def test_evaluate_clustering_selective_metrics(self):
        """Test evaluation with specific metrics."""
        evaluator = ClusteringEvaluator()
        labels_true = np.array([0, 0, 1, 1])
        labels_pred = np.array([0, 0, 1, 1])

        metrics = evaluator.evaluate_clustering(labels_true, labels_pred, metrics=["ari"])

        assert "ari" in metrics
        assert "nmi" not in metrics

    def test_evaluate_pca_variance(self):
        """Test PCA variance evaluation."""
        evaluator = ClusteringEvaluator()
        var_scpt = np.array([0.3, 0.2, 0.15])
        var_scanpy = np.array([0.31, 0.19, 0.16])

        metrics = evaluator.evaluate_pca_variance(var_scpt, var_scanpy)

        assert "pearson_correlation" in metrics
        assert metrics["pearson_correlation"] > 0.9

    def test_evaluate_umap_quality(self):
        """Test UMAP quality evaluation."""
        evaluator = ClusteringEvaluator()
        np.random.seed(42)
        embed_scpt = np.random.randn(100, 2)
        embed_scanpy = embed_scpt + np.random.randn(100, 2) * 0.1

        metrics = evaluator.evaluate_umap_quality(embed_scpt, embed_scanpy)

        assert "procrustes_distance" in metrics
        assert "pearson_correlation" in metrics


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_labels(self):
        """Test with empty label arrays."""
        labels_true = np.array([])
        labels_pred = np.array([])

        ari = compute_clustering_ari(labels_true, labels_pred)
        assert np.isnan(ari)

        nmi = compute_clustering_nmi(labels_true, labels_pred)
        assert np.isnan(nmi)

    def test_shape_mismatch(self):
        """Test with mismatched shapes."""
        labels_true = np.array([0, 0, 1, 1])
        labels_pred = np.array([0, 1, 2])

        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_clustering_ari(labels_true, labels_pred)

    def test_single_cluster_silhouette(self):
        """Test silhouette with single cluster."""
        X = np.random.randn(10, 5)
        labels = np.zeros(10)

        with pytest.raises(ValueError, match="at least 2 clusters"):
            compute_clustering_silhouette(X, labels)

    def test_invalid_average_method(self):
        """Test invalid average method for NMI."""
        labels_true = np.array([0, 0, 1, 1])
        labels_pred = np.array([0, 1, 0, 1])

        with pytest.raises(ValueError, match="Invalid average_method"):
            compute_clustering_nmi(labels_true, labels_pred, average_method="invalid")

    def test_invalid_metric_request(self):
        """Test requesting unsupported metric."""
        evaluator = ClusteringEvaluator()
        labels_true = np.array([0, 0, 1, 1])
        labels_pred = np.array([0, 1, 0, 1])

        with pytest.raises(ValueError, match="Unsupported metric"):
            evaluator.evaluate_clustering(labels_true, labels_pred, metrics=["invalid_metric"])
