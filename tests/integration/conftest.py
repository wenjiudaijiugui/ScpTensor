"""
Pytest fixtures for integration tests.

This module provides reusable fixtures for generating synthetic test data
and setting up test containers.
"""

import numpy as np
import polars as pl
import pytest

from scptensor.core.structures import Assay, ScpContainer, ScpMatrix


@pytest.fixture
def synthetic_container():
    """
    Generate a synthetic ScpContainer for testing.

    Creates a small test dataset with known structure:
    - 100 samples x 500 features
    - 50% missing values (30% random, 20% systematic/LOD)
    - 2 groups (GroupA, GroupB)
    - 2 batches (Batch1, Batch2)

    Returns
    -------
    ScpContainer
        Synthetic test container with 'raw' layer.
    """
    np.random.seed(42)

    n_samples = 100
    n_features = 500

    # 1. Metadata
    groups = np.array(["GroupA"] * 50 + ["GroupB"] * 50)
    batches = np.random.choice(["Batch1", "Batch2"], size=n_samples)

    obs = pl.DataFrame(
        {
            "sample_id": [f"S{i + 1:03d}" for i in range(n_samples)],
            "group": groups,
            "batch": batches,
            "_index": [f"S{i + 1:03d}" for i in range(n_samples)],
        }
    )

    # 2. Expression Data
    X_true = np.random.lognormal(mean=2, sigma=0.5, size=(n_samples, n_features))

    # Add group effect (shift for Group B in first 50 features)
    X_true[groups == "GroupB", :50] *= 2.0

    # Add batch effect (shift for Batch2)
    X_true[batches == "Batch2", :] *= 1.2

    # 3. Introduce missing values
    X_observed = X_true.copy()
    M = np.zeros((n_samples, n_features), dtype=int)

    # Systematic missing (LOD) - 20%
    threshold = np.percentile(X_true, 20)
    lod_mask = X_true < threshold
    X_observed[lod_mask] = 0
    M[lod_mask] = 2  # LOD

    # Random missing - 30% of remaining valid
    valid_mask = M == 0
    np.sum(valid_mask)
    n_random_missing = int(n_samples * n_features * 0.3)

    valid_indices = np.argwhere(valid_mask)
    random_indices_idx = np.random.choice(len(valid_indices), size=n_random_missing, replace=False)
    random_indices = valid_indices[random_indices_idx]

    X_observed[random_indices[:, 0], random_indices[:, 1]] = 0
    M[random_indices[:, 0], random_indices[:, 1]] = 1  # MBR

    # 4. Create container
    var = pl.DataFrame(
        {
            "protein_id": [f"P{i + 1:04d}" for i in range(n_features)],
            "_index": [f"P{i + 1:04d}" for i in range(n_features)],
        }
    )

    matrix = ScpMatrix(X=X_observed, M=M)
    assay = Assay(var=var, layers={"raw": matrix}, feature_id_col="protein_id")

    container = ScpContainer(assays={"protein": assay}, obs=obs, sample_id_col="sample_id")

    return container


@pytest.fixture
def small_synthetic_container():
    """
    Generate a very small synthetic container for fast tests.

    Creates a minimal test dataset:
    - 20 samples x 50 features
    - 40% missing values
    - 2 groups, 2 batches

    Returns
    -------
    ScpContainer
        Small synthetic test container.
    """
    np.random.seed(42)

    n_samples = 20
    n_features = 50

    groups = np.array(["GroupA"] * 10 + ["GroupB"] * 10)
    batches = np.random.choice(["Batch1", "Batch2"], size=n_samples)

    obs = pl.DataFrame(
        {
            "sample_id": [f"S{i + 1:03d}" for i in range(n_samples)],
            "group": groups,
            "batch": batches,
            "_index": [f"S{i + 1:03d}" for i in range(n_samples)],
        }
    )

    X_true = np.random.lognormal(mean=2, sigma=0.5, size=(n_samples, n_features))
    X_true[groups == "GroupB", :25] *= 2.0
    X_true[batches == "Batch2", :] *= 1.2

    X_observed = X_true.copy()
    M = np.zeros((n_samples, n_features), dtype=int)

    # 40% missing
    threshold = np.percentile(X_true, 25)
    lod_mask = X_true < threshold
    X_observed[lod_mask] = 0
    M[lod_mask] = 2

    valid_mask = M == 0
    np.sum(valid_mask)
    n_random_missing = int(n_samples * n_features * 0.15)

    valid_indices = np.argwhere(valid_mask)
    if len(valid_indices) > n_random_missing:
        random_indices_idx = np.random.choice(
            len(valid_indices), size=n_random_missing, replace=False
        )
        random_indices = valid_indices[random_indices_idx]
        X_observed[random_indices[:, 0], random_indices[:, 1]] = 0
        M[random_indices[:, 0], random_indices[:, 1]] = 1

    var = pl.DataFrame(
        {
            "protein_id": [f"P{i + 1:04d}" for i in range(n_features)],
            "_index": [f"P{i + 1:04d}" for i in range(n_features)],
        }
    )

    matrix = ScpMatrix(X=X_observed, M=M)
    assay = Assay(var=var, layers={"raw": matrix}, feature_id_col="protein_id")

    container = ScpContainer(assays={"protein": assay}, obs=obs, sample_id_col="sample_id")

    return container


@pytest.fixture
def temp_output_dir(tmp_path):
    """
    Create a temporary directory for test outputs.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest's built-in temporary directory fixture.

    Returns
    -------
    pathlib.Path
        Path to temporary output directory.
    """
    output_dir = tmp_path / "test_outputs"
    output_dir.mkdir(exist_ok=True)
    return output_dir
