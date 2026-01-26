"""Streamlined synthetic data generation for comparison study."""

import numpy as np
import polars as pl

from scptensor.core.structures import Assay, ScpContainer, ScpMatrix


def generate_synthetic_data(
    n_samples: int,
    n_features: int,
    n_batches: int = 2,
    missing_rate: float = 0.3,
    batch_effect: float = 0.5,
    n_cell_types: int = 5,
    random_seed: int = 42,
) -> ScpContainer:
    """Generate synthetic single-cell proteomics data."""
    np.random.seed(random_seed)
    rng = np.random.default_rng(random_seed)

    # Generate cell type labels
    cell_type_labels = rng.integers(0, n_cell_types, size=n_samples)

    # Generate cell type markers
    markers_per_type = n_features // (n_cell_types * 2)
    base_expression = np.zeros((n_cell_types, n_features))

    for ct in range(n_cell_types):
        marker_start = ct * markers_per_type
        marker_end = min(marker_start + markers_per_type, n_features)

        # High expression for markers
        base_expression[ct, marker_start:marker_end] = rng.uniform(
            low=8.0, high=10.0, size=marker_end - marker_start
        )

        # Basal expression for others
        non_marker_indices = [i for i in range(n_features) if not (marker_start <= i < marker_end)]
        base_expression[ct, non_marker_indices] = rng.exponential(
            scale=2.0, size=len(non_marker_indices)
        )

    # Generate data matrix
    x = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        ct = cell_type_labels[i]
        x[i, :] = base_expression[ct, :] + rng.normal(loc=0.0, scale=0.5, size=n_features)

    x = np.maximum(x, 0)

    # Add batch effects
    if n_batches > 1:
        batch_labels = rng.integers(0, n_batches, size=n_samples)
        if batch_effect > 0:
            x = add_batch_effects(x, batch_labels, batch_effect, rng)
    else:
        batch_labels = np.zeros(n_samples, dtype=int)

    # Add missing values
    x, m = add_missing_values(x, missing_rate, rng)

    # Create metadata
    obs = pl.DataFrame(
        {
            "_index": [f"cell_{i}" for i in range(n_samples)],
            "batch": batch_labels,
            "cell_type": cell_type_labels,
            "n_features": np.sum(m == 0, axis=1),
        }
    )

    var = pl.DataFrame(
        {
            "_index": [f"protein_{i}" for i in range(n_features)],
            "n_cells": [n_samples] * n_features,
            "missing_rate": [np.sum(m[:, i] > 0) / n_samples for i in range(n_features)],
        }
    )

    container = ScpContainer(
        obs=obs, assays={"proteins": Assay(var=var, layers={"raw": ScpMatrix(X=x, M=m)})}
    )

    return container


def add_missing_values(x, missing_rate, rng):
    """Add missing values to data matrix."""
    m = np.zeros_like(x, dtype=np.int8)

    if missing_rate > 0:
        missing_mask = rng.random(x.shape) < missing_rate
        m[missing_mask] = 1
        x[missing_mask] = 0

    return x, m


def add_batch_effects(x, batch_labels, effect_size, rng):
    """Add batch effects to data matrix."""
    x_batched = x.copy()
    unique_batches = np.unique(batch_labels)

    for batch_id in unique_batches:
        batch_mask = batch_labels == batch_id
        batch_shift = rng.normal(loc=0.0, scale=effect_size, size=x.shape[1])
        batch_scale = rng.normal(loc=1.0, scale=0.1 * effect_size, size=x.shape[1])
        x_batched[batch_mask] = x_batched[batch_mask] * batch_scale + batch_shift

    return x_batched


def generate_cell_type_markers(n_features, n_cell_types, rng):
    """Generate cell type marker protein patterns."""
    markers_per_type = n_features // (n_cell_types * 2)
    base_expression = np.zeros((n_cell_types, n_features))

    for ct in range(n_cell_types):
        marker_start = ct * markers_per_type
        marker_end = min(marker_start + markers_per_type, n_features)
        base_expression[ct, marker_start:marker_end] = rng.uniform(
            low=8.0, high=10.0, size=marker_end - marker_start
        )
        non_marker_indices = [i for i in range(n_features) if not (marker_start <= i < marker_end)]
        base_expression[ct, non_marker_indices] = rng.exponential(
            scale=2.0, size=len(non_marker_indices)
        )

    return base_expression


def generate_small_dataset(seed=42):
    """Generate small dataset."""
    return generate_synthetic_data(
        n_samples=1000,
        n_features=1000,
        n_batches=1,
        missing_rate=0.6,
        batch_effect=0.0,
        n_cell_types=5,
        random_seed=seed,
    )


def generate_medium_dataset(seed=42):
    """Generate medium dataset."""
    return generate_synthetic_data(
        n_samples=5000,
        n_features=1500,
        n_batches=5,
        missing_rate=0.7,
        batch_effect=1.5,
        n_cell_types=8,
        random_seed=seed,
    )


def generate_large_dataset(seed=42):
    """Generate large dataset."""
    return generate_synthetic_data(
        n_samples=20000,
        n_features=2000,
        n_batches=10,
        missing_rate=0.75,
        batch_effect=2.0,
        n_cell_types=12,
        random_seed=seed,
    )
