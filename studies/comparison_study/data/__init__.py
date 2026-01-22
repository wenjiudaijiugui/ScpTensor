"""Data loading and preparation module for comparison study.

This module provides utilities for loading and generating single-cell
proteomics datasets for pipeline comparison testing.
"""

from .load_datasets import (
    add_batch_effects,
    create_batch_labels,
    load_dataset,
    load_from_csv,
    load_from_h5ad,
    load_from_pickle,
)
from .prepare_synthetic import (
    cache_datasets,
    generate_large_dataset,
    generate_medium_dataset,
    generate_small_dataset,
    generate_synthetic_dataset,
    load_all_datasets,
    load_cached_datasets,
)

__all__ = [
    # load_datasets
    "load_dataset",
    "load_from_pickle",
    "load_from_csv",
    "load_from_h5ad",
    "create_batch_labels",
    "add_batch_effects",
    # prepare_synthetic
    "generate_synthetic_dataset",
    "generate_small_dataset",
    "generate_medium_dataset",
    "generate_large_dataset",
    "load_all_datasets",
    "cache_datasets",
    "load_cached_datasets",
]
