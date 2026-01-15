"""Nonlinear integration methods for single-cell proteomics data.

Reference
---------
Korsunsky I, et al. Fast, sensitive and accurate integration of
single-cell data with Harmony. Nature Methods (2019).
"""

import numpy as np
import polars as pl
import scipy.sparse as sp

from scptensor.core.exceptions import AssayNotFoundError, LayerNotFoundError, ScpValueError
from scptensor.core.sparse_utils import is_sparse_matrix
from scptensor.core.structures import Assay, ScpContainer, ScpMatrix
from scptensor.core.utils import requires_dependency


@requires_dependency("harmonypy", "pip install harmonypy")
def harmony(
    container: ScpContainer,
    batch_key: str,
    assay_name: str = "protein",
    base_layer: str = "pca",
    new_layer_name: str | None = "harmony",
    theta: float | None = None,
    lamb: float | None = None,
    sigma: float = 0.1,
    nclust: int | None = None,
    max_iter_harmony: int = 10,
    max_iter_cluster: int = 20,
    epsilon_cluster: float = 1e-5,
    epsilon_harmony: float = 1e-4,
) -> ScpContainer:
    """Harmony integration for batch effect correction.

    Harmony uses an iterative clustering and correction approach to remove
    batch effects while preserving biological variation. It works best on
    PCA-reduced data.

    Parameters
    ----------
    container : ScpContainer
        Input container with multiple batches
    batch_key : str
        Column name in obs containing batch labels
    assay_name : str, optional
        Name of the assay to process (default: 'protein')
    base_layer : str, optional
        Name of the layer to use as input (default: 'pca')
        Harmony typically works best on PCA-reduced data
    new_layer_name : str, optional
        Name for the new layer with corrected data (default: 'harmony')
    theta : float, optional
        Clustering penalty parameter (default: 2.0)
        Higher values = more diversity encouraged in clusters
    lamb : float, optional
        Ridge regularization penalty (default: 1.0)
        Higher values = more shrinkage towards global centroid
    sigma : float, optional
        Bandwidth parameter for clustering kernel (default: 0.1)
    nclust : int, optional
        Number of clusters (default: None, auto-detect)
    max_iter_harmony : int, optional
        Maximum iterations for Harmony (default: 10)
    max_iter_cluster : int, optional
        Maximum iterations for clustering (default: 20)
    epsilon_cluster : float, optional
        Convergence threshold for clustering (default: 1e-5)
    epsilon_harmony : float, optional
        Convergence threshold for Harmony (default: 1e-4)

    Returns
    -------
    ScpContainer
        Container with batch-corrected layer

    Raises
    ------
    AssayNotFoundError
        If the specified assay does not exist
    LayerNotFoundError
        If the specified layer does not exist in the assay
    ScpValueError
        If batch_key not found in obs or parameters are invalid
    MissingDependencyError
        If harmonypy package is not installed

    Notes
    -----
    Harmony algorithm:
        1. Assigns cells to clusters using current cell embeddings
        2. Computes cluster-specific batch effects
        3. Corrects cell embeddings to maximize diversity
        4. Iterates until convergence

    Harmony is particularly effective for:
        - Large datasets with many batches
        - Complex batch structures
        - When biological variation should be preserved

    Examples
    --------
    >>> from scptensor.integration import harmony
    >>> container = harmony(container, batch_key='batch')
    >>> container = harmony(container, batch_key='batch', theta=3, nclust=15)
    """
    import harmonypy as hm

    # Validate assay and layer
    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]
    if base_layer not in assay.layers:
        raise LayerNotFoundError(base_layer, assay_name)

    obs_df = container.obs
    if batch_key not in obs_df.columns:
        raise ScpValueError(
            f"Batch key '{batch_key}' not found in obs. Available columns: {list(obs_df.columns)}",
            parameter="batch_key",
            value=batch_key,
        )

    # Get and prepare data
    X = assay.layers[base_layer].X
    input_was_sparse = is_sparse_matrix(X)

    # Convert to dense and impute NaNs
    X_dense = _prepare_harmony_data(X)

    # Validate batches
    batches = obs_df[batch_key].to_numpy()
    unique_batches = np.unique(batches)

    if len(unique_batches) < 2:
        raise ScpValueError(
            "Harmony requires at least 2 batches.",
            parameter="batch_key",
            value=batch_key,
        )

    # Validate minimum sample requirement
    batch_counts = {b: np.sum(batches == b) for b in unique_batches}
    if any(c < 2 for c in batch_counts.values()):
        raise ScpValueError(
            f"Harmony requires at least 2 samples per batch. Batch counts: {batch_counts}",
            parameter="batch_key",
            value=batch_key,
        )

    # Prepare metadata for Harmony
    meta_data = obs_df.to_pandas()

    # Set default values for Harmony parameters
    harmony_params = {
        "theta": theta if theta is not None else 2.0,
        "lamb": lamb if lamb is not None else 1.0,
        "sigma": sigma,
        "nclust": nclust,
        "max_iter_harmony": max_iter_harmony,
        "max_iter_cluster": max_iter_cluster,
        "epsilon_cluster": epsilon_cluster,
        "epsilon_harmony": epsilon_harmony,
    }

    # Run Harmony
    ho = hm.run_harmony(X_dense, meta_data, batch_key, **harmony_params)

    # Harmony returns (n_pcs, n_samples), transpose to (n_samples, n_pcs)
    res = ho.Z_corr.T

    # Preserve sparsity if appropriate
    if input_was_sparse:
        sparsity_ratio = 1.0 - (np.count_nonzero(res) / res.size)
        if sparsity_ratio > 0.5:
            res = sp.csr_matrix(res)

    # Create new layer
    M_input = assay.layers[base_layer].M
    new_matrix = ScpMatrix(
        X=res,
        M=M_input.copy() if M_input is not None else None,
    )
    container.assays[assay_name].add_layer(new_layer_name or "harmony", new_matrix)

    # Log operation
    container.log_operation(
        action="integration_harmony",
        params={
            "batch_key": batch_key,
            "theta": harmony_params["theta"],
            "lamb": harmony_params["lamb"],
            "sigma": harmony_params["sigma"],
            "nclust": harmony_params["nclust"],
            "n_batches": len(unique_batches),
        },
        description=f"Harmony integration (theta={harmony_params['theta']}, "
        f"lamb={harmony_params['lamb']}) on layer '{base_layer}'.",
    )

    return container


def _prepare_harmony_data(X: np.ndarray | sp.spmatrix) -> np.ndarray:
    """Convert sparse to dense and impute NaN values."""
    if is_sparse_matrix(X):
        X_dense = X.toarray() if hasattr(X, "toarray") else np.array(X.todense())
    else:
        X_dense = np.asarray(X)

    if np.isnan(X_dense).any():
        col_mean = np.nan_to_num(np.nanmean(X_dense, axis=0), nan=0.0)
        nan_idx = np.where(np.isnan(X_dense))
        X_dense[nan_idx] = np.take(col_mean, nan_idx[1])

    return X_dense


if __name__ == "__main__":
    # Test: Basic functionality
    print("Testing Harmony integration...")

    np.random.seed(42)
    n_samples_per_batch = 50
    n_features = 30

    # Generate PCA-like data with batch effects
    X_batch1 = np.random.randn(n_samples_per_batch, n_features) * 0.5
    X_batch2 = np.random.randn(n_samples_per_batch, n_features) * 0.5 + 2.0
    X = np.vstack([X_batch1, X_batch2])

    # Create container
    var = pl.DataFrame({"_index": [f"pc_{i}" for i in range(n_features)]})
    obs = pl.DataFrame(
        {
            "_index": [f"cell_{i}" for i in range(2 * n_samples_per_batch)],
            "batch": ["batch1"] * n_samples_per_batch + ["batch2"] * n_samples_per_batch,
        }
    )

    assay = Assay(var=var)
    assay.add_layer("pca", ScpMatrix(X=X, M=None))
    container = ScpContainer(obs=obs, assays={"protein": assay})

    try:
        result = harmony(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="pca",
            theta=2.0,
            max_iter_harmony=5,
        )

        assert "harmony" in result.assays["protein"].layers
        X_corrected = result.assays["protein"].layers["harmony"].X

        mean1 = np.mean(X_corrected[:n_samples_per_batch], axis=0)
        mean2 = np.mean(X_corrected[n_samples_per_batch:], axis=0)
        batch_diff = np.linalg.norm(mean1 - mean2)
        original_diff = np.linalg.norm(np.mean(X_batch1, axis=0) - np.mean(X_batch2, axis=0))

        print(f"  Original batch difference: {original_diff:.3f}")
        print(f"  Corrected batch difference: {batch_diff:.3f}")
        print(f"  Reduction ratio: {batch_diff / original_diff:.3f}")
        print(f"  Shape: {X_corrected.shape}")
        print("  All tests passed")
    except ImportError:
        print("  harmonypy not installed, skipping test")
        print("  Install with: pip install harmonypy")
