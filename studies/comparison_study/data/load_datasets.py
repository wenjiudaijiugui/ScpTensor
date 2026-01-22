"""Data loading module for comparison study.

This module provides functions to load real and synthetic single-cell
proteomics datasets for pipeline comparison.
"""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


def load_dataset(
    dataset_path: str, assay_name: str = "proteins", batch_column: str | None = "batch"
) -> Any:
    """
    Load a dataset from file.

    Parameters
    ----------
    dataset_path : str
        Path to dataset file (supports .pkl, .csv, .h5ad)
    assay_name : str
        Name of the assay to load
    batch_column : str, optional
        Name of the batch column in obs

    Returns
    -------
    ScpContainer
        Loaded data container

    Examples
    --------
    >>> container = load_dataset("data/small_dataset.pkl")
    >>> print(container.n_samples, container.n_features)
    """
    path = Path(dataset_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    if path.suffix == ".pkl":
        return load_from_pickle(dataset_path)
    elif path.suffix == ".csv":
        return load_from_csv(dataset_path, assay_name, batch_column)
    elif path.suffix in [".h5ad", ".h5"]:
        return load_from_h5ad(dataset_path, assay_name, batch_column)
    else:
        raise ValueError(
            f"Unsupported file format: {path.suffix}. Supported formats: .pkl, .csv, .h5ad, .h5"
        )


def load_from_pickle(pickle_path: str) -> Any:
    """
    Load ScpContainer from pickle file.

    Parameters
    ----------
    pickle_path : str
        Path to pickle file

    Returns
    -------
    ScpContainer
        Loaded container

    Examples
    --------
    >>> container = load_from_pickle("data/container.pkl")
    >>> print(container.n_samples)
    """
    with open(pickle_path, "rb") as f:
        return pickle.load(f)


def load_from_csv(
    csv_path: str, assay_name: str = "proteins", batch_column: str | None = "batch"
) -> Any:
    """
    Load dataset from CSV file.

    Expected format:
    - Rows: cells
    - Columns: features (proteins) + metadata columns
    - Metadata columns start with prefix 'meta_'

    Parameters
    ----------
    csv_path : str
        Path to CSV file
    assay_name : str
        Name for the assay
    batch_column : str, optional
        Name of batch column

    Returns
    -------
    ScpContainer
        Loaded container

    Examples
    --------
    >>> container = load_from_csv("data/data.csv")
    >>> print(container.obs.columns)
    """
    from scptensor.core.structures import Assay, ScpContainer, ScpMatrix

    # Read CSV
    df = pl.read_csv(csv_path)

    # Separate metadata and data columns
    meta_cols = [col for col in df.columns if col.startswith("meta_")]
    data_cols = [col for col in df.columns if not col.startswith("meta_")]

    if not data_cols:
        raise ValueError(
            "No data columns found in CSV. Ensure non-metadata columns don't start with 'meta_'"
        )

    # Extract metadata
    if meta_cols:
        obs_data = df.select(meta_cols).rename({col: col.replace("meta_", "") for col in meta_cols})
    else:
        # Create minimal obs if no metadata
        obs_data = pl.DataFrame({"_index": [f"cell_{i}" for i in range(df.height)]})

    # Extract data matrix
    X = df.select(data_cols).to_numpy()

    # Create mask (all valid initially)
    M = np.zeros_like(X, dtype=np.int8)

    # Create ScpContainer
    var_data = pl.DataFrame({"_index": data_cols, "n_cells": [X.shape[0]] * len(data_cols)})

    container = ScpContainer(
        obs=obs_data, assays={assay_name: Assay(var=var_data, layers={"raw": ScpMatrix(X=X, M=M)})}
    )

    return container


def load_from_h5ad(
    h5ad_path: str, assay_name: str = "proteins", batch_column: str | None = "batch"
) -> Any:
    """
    Load dataset from AnnData h5ad file.

    Parameters
    ----------
    h5ad_path : str
        Path to h5ad file
    assay_name : str
        Name for the assay
    batch_column : str, optional
        Name of batch column in obs

    Returns
    -------
    ScpContainer
        Converted ScpContainer

    Examples
    --------
    >>> container = load_from_h5ad("data/adata.h5ad")
    >>> print(container.n_samples)
    """
    try:
        import anndata as ad
    except ImportError:
        raise ImportError(
            "anndata is required to load h5ad files. Install with: pip install anndata"
        )

    from scptensor.core.structures import Assay, ScpContainer, ScpMatrix

    # Read AnnData object
    adata = ad.read_h5ad(h5ad_path)

    # Extract metadata
    obs_df = pl.DataFrame(adata.obs)

    # Extract data matrix
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()

    # Ensure numpy array
    if not isinstance(X, np.ndarray):
        X = np.array(X)

    # Create mask
    M = np.zeros_like(X, dtype=np.int8)

    # Extract feature metadata
    var_df = pl.DataFrame(
        {"_index": adata.var_names.tolist(), "n_cells": [X.shape[0]] * X.shape[1]}
    )

    # Create container
    container = ScpContainer(
        obs=obs_df, assays={assay_name: Assay(var=var_df, layers={"raw": ScpMatrix(X=X, M=M)})}
    )

    return container


def create_batch_labels(
    n_samples: int, n_batches: int, batch_sizes: list | None = None
) -> np.ndarray:
    """
    Create batch labels for synthetic data.

    Parameters
    ----------
    n_samples : int
        Total number of samples
    n_batches : int
        Number of batches
    batch_sizes : list, optional
        Exact sizes for each batch. If None, splits evenly

    Returns
    -------
    np.ndarray
        Batch labels (0 to n_batches-1)

    Examples
    --------
    >>> labels = create_batch_labels(100, 3)
    >>> print(labels.shape)
    (100,)
    """
    if n_batches <= 0:
        raise ValueError(f"n_batches must be positive, got {n_batches}")

    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    if batch_sizes is None:
        # Even split
        batch_size = n_samples // n_batches
        batch_sizes = [batch_size] * (n_batches - 1) + [n_samples - batch_size * (n_batches - 1)]

    if len(batch_sizes) != n_batches:
        raise ValueError(f"batch_sizes length {len(batch_sizes)} != n_batches {n_batches}")

    if sum(batch_sizes) != n_samples:
        raise ValueError(f"Sum of batch_sizes {sum(batch_sizes)} != n_samples {n_samples}")

    labels = []
    for batch_id, size in enumerate(batch_sizes):
        labels.extend([batch_id] * size)

    return np.array(labels)


def add_batch_effects(
    X: np.ndarray, batch_labels: np.ndarray, effect_size: float = 1.0
) -> np.ndarray:
    """
    Add synthetic batch effects to data.

    Parameters
    ----------
    X : np.ndarray
        Original data matrix (n_samples × n_features)
    batch_labels : np.ndarray
        Batch labels for each sample
    effect_size : float
        Strength of batch effect (standard deviation)

    Returns
    -------
    np.ndarray
        Data with added batch effects

    Examples
    --------
    >>> X = np.random.rand(100, 50)
    >>> batch_labels = np.array([0] * 50 + [1] * 50)
    >>> X_batched = add_batch_effects(X, batch_labels, effect_size=1.0)
    """
    if X.shape[0] != len(batch_labels):
        raise ValueError(f"X samples {X.shape[0]} != batch_labels length {len(batch_labels)}")

    if effect_size <= 0:
        return X.copy()

    X_batched = X.copy()
    unique_batches = np.unique(batch_labels)

    for batch_id in unique_batches:
        batch_mask = batch_labels == batch_id

        # Add batch-specific shift (location effect)
        batch_shift = np.random.normal(loc=0.0, scale=effect_size, size=X.shape[1])

        # Add batch-specific scaling (scale effect)
        batch_scale = np.random.normal(loc=1.0, scale=0.1 * effect_size, size=X.shape[1])

        X_batched[batch_mask] = X_batched[batch_mask] * batch_scale + batch_shift

    return X_batched
