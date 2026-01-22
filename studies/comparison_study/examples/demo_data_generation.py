"""Demonstration of data loading and synthetic data generation."""

import sys

sys.path.insert(0, "/home/shenshang/projects/ScpTensor")

import numpy as np

from docs.comparison_study.data import (
    load_all_datasets,
)
from docs.comparison_study.data.load_datasets import (
    add_batch_effects,
    create_batch_labels,
)
from docs.comparison_study.data.prepare_synthetic import (
    generate_synthetic_dataset,
)


def main():
    """Run demonstrations."""
    print("=" * 70)
    print("Data Preparation Module Demo")
    print("=" * 70)

    # Demo 1: Batch labels
    print("\nDemo 1: Batch Labels")
    labels = create_batch_labels(n_samples=100, n_batches=4)
    print("Created labels for 100 samples in 4 batches")
    print(f"Batch distribution: {[np.sum(labels == i) for i in range(4)]}")

    # Demo 2: Batch effects
    print("\nDemo 2: Batch Effects")
    np.random.seed(42)
    X = np.random.rand(100, 5)
    batch_labels = np.array([0] * 50 + [1] * 50)
    X_batched = add_batch_effects(X, batch_labels, effect_size=1.0)
    print("Added batch effects to data")
    print(f"Shape preserved: {X.shape} -> {X_batched.shape}")

    # Demo 3: Synthetic dataset
    print("\nDemo 3: Synthetic Dataset")
    container = generate_synthetic_dataset(
        n_samples=500,
        n_features=200,
        n_batches=3,
        sparsity=0.6,
        batch_effect_size=1.5,
        n_cell_types=5,
        random_seed=42,
    )
    print("Generated dataset:")
    print(f"  Samples: {container.n_samples}")
    print(f"  Features: {container.n_features}")
    print(f"  Batches: {len(container.obs['batch'].unique())}")
    print(f"  Cell types: {len(container.obs['cell_type'].unique())}")

    # Demo 4: Load all datasets
    print("\nDemo 4: Load All Datasets")
    datasets = load_all_datasets()
    print(f"Loaded {len(datasets)} preset datasets")
    for name, container in datasets.items():
        print(f"  {name}: {container.n_samples} samples × {container.n_features} features")

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
