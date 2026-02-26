"""Test that all viz conftest fixtures work correctly."""

import numpy as np


def test_sample_container(sample_container):
    """Test basic sample_container fixture."""
    assert sample_container is not None
    assert sample_container.n_samples == 50
    assert "batch" in sample_container.obs.columns
    assert "condition" in sample_container.obs.columns
    # np.repeat creates [A, A, ..., B, B, ...], not alternating
    assert sample_container.obs["batch"].to_list()[:2] == ["A", "A"]
    assert sample_container.obs["condition"].to_list()[:2] == ["ctrl", "ctrl"]
    assert "proteins" in sample_container.assays
    assert sample_container.assays["proteins"].n_features == 20


def test_container_with_norm(container_with_norm):
    """Test container_with_norm fixture has both raw and normalized layers."""
    assert container_with_norm is not None
    assert container_with_norm.n_samples == 50
    assay = container_with_norm.assays["proteins"]
    assert "raw" in assay.layers
    assert "normalized" in assay.layers

    # Verify normalization was applied (median-centered)
    X_raw = assay.layers["raw"].X
    X_norm = assay.layers["normalized"].X
    assert not np.allclose(X_raw, X_norm)


def test_container_with_batches(container_with_batches):
    """Test container_with_batches fixture has 3 batches."""
    assert container_with_batches is not None
    assert container_with_batches.n_samples == 60
    assert "batch" in container_with_batches.obs.columns
    batches = container_with_batches.obs["batch"].unique().to_list()
    assert len(batches) == 3
    assert set(batches) == {"batch1", "batch2", "batch3"}

    # Verify batch effects exist (different means)
    assay = container_with_batches.assays["proteins"]
    X = assay.layers["raw"].X
    mean_batch1 = X[0:20].mean()
    mean_batch2 = X[20:40].mean()
    mean_batch3 = X[40:60].mean()
    assert mean_batch1 < mean_batch2  # batch1 has +5 offset
    assert mean_batch2 > mean_batch3  # batch2 has +10 offset


def test_container_with_clusters(container_with_clusters):
    """Test container_with_clusters fixture has clustering results."""
    assert container_with_clusters is not None
    assert container_with_clusters.n_samples == 100
    assert "pca" in container_with_clusters.assays

    # Check that cluster labels were added to obs (column name is kmeans_k3)
    assert "kmeans_k3" in container_with_clusters.obs.columns

    # Verify we have 3 clusters
    clusters = container_with_clusters.obs["kmeans_k3"].unique().to_list()
    assert len(clusters) == 3
    assert set(clusters) == {"0", "1", "2"} or set(clusters) == {0, 1, 2}
