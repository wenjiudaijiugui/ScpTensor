"""Synthetic data generation for comparison study.

This module generates synthetic single-cell proteomics datasets with
controlled properties for testing pipelines.
"""

import pickle
from pathlib import Path
from typing import Any

import numpy as np


def generate_synthetic_dataset(
    n_samples: int,
    n_features: int,
    n_batches: int = 1,
    sparsity: float = 0.6,
    batch_effect_size: float = 1.0,
    n_cell_types: int = 5,
    random_seed: int = 42,
) -> Any:
    """
    Generate a synthetic single-cell proteomics dataset.

    The dataset simulates:
    - Multiple cell types with distinct protein expression patterns
    - Batch effects (if n_batches > 1)
    - Missing values (controlled by sparsity)
    - Technical noise

    Parameters
    ----------
    n_samples : int
        Number of cells to generate
    n_features : int
        Number of proteins (features)
    n_batches : int
        Number of batches (1 = single batch, no batch effects)
    sparsity : float
        Fraction of missing values (0-1)
    batch_effect_size : float
        Strength of batch effects (0 = no batch effects)
    n_cell_types : int
        Number of distinct cell types
    random_seed : int
        Random seed for reproducibility

    Returns
    -------
    ScpContainer
        Synthetic dataset with batch information

    Examples
    --------
    >>> container = generate_synthetic_dataset(
    ...     n_samples=1000,
    ...     n_features=500,
    ...     n_batches=3,
    ...     sparsity=0.6
    ... )
    >>> print(container.n_samples)
    1000
    """
    import polars as pl

    from scptensor.core.structures import Assay, ScpContainer, ScpMatrix

    rng = np.random.default_rng(random_seed)

    # Validate parameters
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if n_features <= 0:
        raise ValueError(f"n_features must be positive, got {n_features}")
    if n_batches <= 0:
        raise ValueError(f"n_batches must be positive, got {n_batches}")
    if not (0 <= sparsity <= 1):
        raise ValueError(f"sparsity must be in [0, 1], got {sparsity}")
    if n_cell_types <= 0:
        raise ValueError(f"n_cell_types must be positive, got {n_cell_types}")
    if n_cell_types > n_samples:
        raise ValueError(f"n_cell_types ({n_cell_types}) cannot exceed n_samples ({n_samples})")

    # 1. Generate cell type labels
    cell_type_labels = rng.integers(0, n_cell_types, size=n_samples)

    # 2. Generate cell type-specific protein expression patterns
    # Each cell type has a subset of highly expressed proteins
    markers_per_type = n_features // (n_cell_types * 2)

    base_expression = np.zeros((n_cell_types, n_features))
    for ct in range(n_cell_types):
        # Select marker proteins for this cell type
        marker_start = ct * markers_per_type
        marker_end = min(marker_start + markers_per_type, n_features)

        # High expression for markers
        base_expression[ct, marker_start:marker_end] = rng.uniform(
            low=8.0, high=10.0, size=marker_end - marker_start
        )

        # Basal expression for other proteins
        non_marker_indices = [i for i in range(n_features) if not (marker_start <= i < marker_end)]
        base_expression[ct, non_marker_indices] = rng.exponential(
            scale=2.0, size=len(non_marker_indices)
        )

    # 3. Generate data matrix
    X = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        ct = cell_type_labels[i]
        X[i, :] = base_expression[ct, :] + rng.normal(loc=0.0, scale=0.5, size=n_features)

    # 4. Ensure non-negative values
    X = np.maximum(X, 0)

    # 5. Create batch labels
    if n_batches > 1:
        batch_labels = rng.integers(0, n_batches, size=n_samples)

        # Add batch effects
        if batch_effect_size > 0:
            X = _add_batch_effects(X, batch_labels, effect_size=batch_effect_size, rng=rng)
    else:
        batch_labels = np.zeros(n_samples, dtype=int)

    # 6. Add missing values (mask)
    M = np.zeros_like(X, dtype=np.int8)

    if sparsity > 0:
        missing_mask = rng.random(X.shape) < sparsity
        M[missing_mask] = 1  # MBR (missing between runs)
        X[missing_mask] = 0

    # 7. Create metadata
    obs_data = pl.DataFrame(
        {
            "_index": [f"cell_{i}" for i in range(n_samples)],
            "batch": batch_labels,
            "cell_type": cell_type_labels,
            "n_features": np.sum(M == 0, axis=1),
        }
    )

    # 8. Create feature metadata
    var_data = pl.DataFrame(
        {
            "_index": [f"protein_{i}" for i in range(n_features)],
            "n_cells": [n_samples] * n_features,
            "missing_rate": [np.sum(M[:, i] > 0) / n_samples for i in range(n_features)],
        }
    )

    # 9. Create container
    container = ScpContainer(
        obs=obs_data, assays={"proteins": Assay(var=var_data, layers={"raw": ScpMatrix(X=X, M=M)})}
    )

    return container


def _add_batch_effects(
    X: np.ndarray, batch_labels: np.ndarray, effect_size: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Add batch effects to data matrix.

    Parameters
    ----------
    X : np.ndarray
        Original data matrix
    batch_labels : np.ndarray
        Batch labels for each sample
    effect_size : float
        Strength of batch effects
    rng : np.random.Generator
        Random number generator

    Returns
    -------
    np.ndarray
        Data with batch effects
    """
    X_batched = X.copy()
    unique_batches = np.unique(batch_labels)

    for batch_id in unique_batches:
        batch_mask = batch_labels == batch_id

        # Location effect (shift)
        batch_shift = rng.normal(loc=0.0, scale=effect_size, size=X.shape[1])

        # Scale effect
        batch_scale = rng.normal(loc=1.0, scale=0.1 * effect_size, size=X.shape[1])

        X_batched[batch_mask] = X_batched[batch_mask] * batch_scale + batch_shift

    return X_batched


def generate_small_dataset(random_seed: int = 42) -> Any:
    """
    Generate small single-batch dataset for baseline testing.

    Returns
    -------
    ScpContainer
        Small dataset (~1K cells × 1K proteins, 1 batch)

    Examples
    --------
    >>> container = generate_small_dataset()
    >>> print(container.n_samples)
    1000
    """
    return generate_synthetic_dataset(
        n_samples=1000,
        n_features=1000,
        n_batches=1,
        sparsity=0.6,
        batch_effect_size=0.0,
        n_cell_types=5,
        random_seed=random_seed,
    )


def generate_medium_dataset(random_seed: int = 42) -> Any:
    """
    Generate medium multi-batch dataset for batch correction testing.

    Returns
    -------
    ScpContainer
        Medium dataset (~5K cells × 1.5K proteins, 5 batches)

    Examples
    --------
    >>> container = generate_medium_dataset()
    >>> print(container.n_samples)
    5000
    """
    return generate_synthetic_dataset(
        n_samples=5000,
        n_features=1500,
        n_batches=5,
        sparsity=0.7,
        batch_effect_size=1.5,
        n_cell_types=8,
        random_seed=random_seed,
    )


def generate_large_dataset(random_seed: int = 42) -> Any:
    """
    Generate large multi-batch dataset for scalability testing.

    Returns
    -------
    ScpContainer
        Large dataset (~20K cells × 2K proteins, 10 batches)

    Examples
    --------
    >>> container = generate_large_dataset()
    >>> print(container.n_samples)
    20000
    """
    return generate_synthetic_dataset(
        n_samples=20000,
        n_features=2000,
        n_batches=10,
        sparsity=0.75,
        batch_effect_size=2.0,
        n_cell_types=12,
        random_seed=random_seed,
    )


def load_all_datasets(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Load all three datasets for comparison study.

    Parameters
    ----------
    config : dict, optional
        Configuration dict with dataset parameters.
        Example: {"small": {"random_seed": 42}}

    Returns
    -------
    dict
        Dictionary with dataset names as keys and containers as values

    Examples
    --------
    >>> datasets = load_all_datasets()
    >>> print(list(datasets.keys()))
    ['small', 'medium', 'large']
    """
    if config is None:
        config = {}

    datasets = {}

    # Small dataset
    small_config = config.get("small", {})
    datasets["small"] = generate_small_dataset(random_seed=small_config.get("random_seed", 42))

    # Medium dataset
    medium_config = config.get("medium", {})
    datasets["medium"] = generate_medium_dataset(random_seed=medium_config.get("random_seed", 42))

    # Large dataset
    large_config = config.get("large", {})
    datasets["large"] = generate_large_dataset(random_seed=large_config.get("random_seed", 42))

    return datasets


def cache_datasets(datasets: dict[str, Any], cache_dir: str = "outputs/data_cache") -> None:
    """
    Cache generated datasets to disk.

    Parameters
    ----------
    datasets : dict
        Dictionary of dataset name → container
    cache_dir : str
        Directory to save cached datasets

    Examples
    --------
    >>> datasets = load_all_datasets()
    >>> cache_datasets(datasets, "outputs/data_cache")
    Cached dataset 'small' to outputs/data_cache/small.pkl
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    for name, container in datasets.items():
        file_path = cache_path / f"{name}.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(container, f)

        print(f"Cached dataset '{name}' to {file_path}")


def load_cached_datasets(cache_dir: str = "outputs/data_cache") -> dict[str, Any]:
    """
    Load cached datasets from disk.

    Parameters
    ----------
    cache_dir : str
        Directory containing cached datasets

    Returns
    -------
    dict
        Dictionary of dataset name → container

    Raises
    ------
    FileNotFoundError
        If cache directory does not exist

    Examples
    --------
    >>> datasets = load_cached_datasets("outputs/data_cache")
    >>> print(list(datasets.keys()))
    ['small', 'medium', 'large']
    """
    cache_path = Path(cache_dir)

    if not cache_path.exists():
        raise FileNotFoundError(f"Cache directory not found: {cache_dir}")

    datasets = {}
    for file_path in cache_path.glob("*.pkl"):
        name = file_path.stem
        with open(file_path, "rb") as f:
            datasets[name] = pickle.load(f)

        print(f"Loaded cached dataset '{name}' from {file_path}")

    return datasets
