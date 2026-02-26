"""Shared pytest fixtures for viz module tests.

This module provides reusable fixtures for testing visualization components.
Fixtures are organized by common test scenarios: basic containers, normalized data,
batch effects, and clustering results.
"""

import numpy as np
import polars as pl
import pytest

from scptensor.cluster import cluster_kmeans
from scptensor.core import Assay, ScpContainer, ScpMatrix


@pytest.fixture
def sample_container() -> ScpContainer:
    """Create test container with batch and condition information (50 samples x 20 features).

    Returns
    -------
    ScpContainer
        Container with batch and condition metadata for visualization testing.
        Contains 50 samples and 20 features with random expression data.
    """
    np.random.seed(42)

    obs = pl.DataFrame(
        {
            "_index": [f"S{i}" for i in range(50)],
            "batch": np.repeat(["A", "B"], 25),
            "condition": np.repeat(["ctrl", "treat"], 25),
        }
    )

    var = pl.DataFrame({"_index": [f"P{i}" for i in range(20)]})

    X = np.random.rand(50, 20) * 10  # Random expression data
    assay = Assay(var=var, layers={"raw": ScpMatrix(X=X)})
    container = ScpContainer(obs=obs, assays={"proteins": assay})

    return container


@pytest.fixture
def container_with_norm() -> ScpContainer:
    """Container with pre- and post-normalization layers.

    Returns
    -------
    ScpContainer
        Container with 'raw' and 'normalized' layers to test normalization
        visualization. Contains 50 samples and 20 features.
    """
    np.random.seed(42)

    obs = pl.DataFrame({"_index": [f"S{i}" for i in range(50)], "batch": np.repeat(["A", "B"], 25)})

    var = pl.DataFrame({"_index": [f"P{i}" for i in range(20)]})

    # Pre-normalization
    X_raw = np.random.rand(50, 20) * 10 + 5

    # Post-normalization (median-centered)
    X_norm = X_raw - np.median(X_raw, axis=1, keepdims=True)

    assay = Assay(var=var, layers={"raw": ScpMatrix(X=X_raw), "normalized": ScpMatrix(X=X_norm)})
    container = ScpContainer(obs=obs, assays={"proteins": assay})

    return container


@pytest.fixture
def container_with_batches() -> ScpContainer:
    """Container with 3 batch structure (60 samples).

    Returns
    -------
    ScpContainer
        Container with simulated batch effects for testing batch correction
        visualization. Different batches have different mean offsets.
        Contains 60 samples and 15 features.
    """
    np.random.seed(42)

    obs = pl.DataFrame(
        {
            "_index": [f"S{i}" for i in range(60)],
            "batch": np.repeat(["batch1", "batch2", "batch3"], 20),
            "condition": np.repeat(["ctrl", "treat"], 30),
        }
    )

    var = pl.DataFrame({"_index": [f"P{i}" for i in range(15)]})

    # Simulate batch effects (different batches have different mean offsets)
    X = np.random.rand(60, 15) * 5
    X[0:20] += 5  # batch1
    X[20:40] += 10  # batch2

    assay = Assay(var=var, layers={"raw": ScpMatrix(X=X)})
    container = ScpContainer(obs=obs, assays={"proteins": assay})

    return container


@pytest.fixture
def container_with_clusters() -> ScpContainer:
    """Container with clustering results.

    Returns
    -------
    ScpContainer
        Container with K-means clustering results stored in obs.
        Contains 100 samples, 10 features, and 3 clusters.
    """
    np.random.seed(42)

    obs = pl.DataFrame({"_index": [f"S{i}" for i in range(100)]})

    var = pl.DataFrame({"_index": [f"PC{i}" for i in range(10)]})

    X = np.random.rand(100, 10) * 10
    assay = Assay(var=var, layers={"X": ScpMatrix(X=X)})
    container = ScpContainer(obs=obs, assays={"pca": assay})

    # Run clustering
    container = cluster_kmeans(container, n_clusters=3, storage="obs")

    return container
