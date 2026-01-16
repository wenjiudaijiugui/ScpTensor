"""Mutual Nearest Neighbors (MNN) correction for single-cell proteomics data.

Reference
---------
Haghverdi L, et al. Batch effects in single-cell RNA-sequencing data
are corrected by matching mutual nearest neighbors. Nature Biotechnology (2018).
"""

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from scptensor.core.exceptions import AssayNotFoundError, LayerNotFoundError, ScpValueError
from scptensor.core.sparse_utils import is_sparse_matrix
from scptensor.core.structures import Assay, ScpContainer, ScpMatrix


def integrate_mnn(
    container: ScpContainer,
    batch_key: str,
    assay_name: str = "protein",
    base_layer: str = "raw",
    new_layer_name: str | None = "mnn_corrected",
    k: int = 20,
    sigma: float = 1.0,
    n_pcs: int | None = None,
    use_pca: bool = True,
    use_anchor_correction: bool = True,
) -> ScpContainer:
    """Correct batch effects using Mutual Nearest Neighbors (MNN) method.

    MNN identifies pairs of cells from different batches that are mutual
    nearest neighbors in the high-dimensional space. These pairs are used
    to estimate and correct batch-specific biases.

    Parameters
    ----------
    container : ScpContainer
        Input container with multiple batches
    batch_key : str
        Column name in obs containing batch labels
    assay_name : str, optional
        Name of the assay to use (default: 'protein')
    base_layer : str, optional
        Layer to use as input (default: 'raw')
    new_layer_name : str, optional
        Name for the corrected layer (default: 'mnn_corrected')
    k : int, optional
        Number of nearest neighbors to find (default: 20)
    sigma : float, optional
        Sigma parameter for Gaussian kernel (default: 1.0)
    n_pcs : int, optional
        Number of principal components to use for MNN search (default: 50)
    use_pca : bool, optional
        Whether to use PCA for MNN search (default: True)
    use_anchor_correction : bool, optional
        Whether to use anchor-based correction (default: True)
        When True, uses the first batch as anchor and corrects others

    Returns
    -------
    ScpContainer
        Container with batch-corrected data

    Raises
    ------
    AssayNotFoundError
        If the specified assay does not exist
    LayerNotFoundError
        If the specified layer does not exist in the assay
    ScpValueError
        If batch_key not found in obs, k is invalid, or fewer than 2 batches

    Notes
    -----
    The MNN algorithm:
        1. Computes PCA (optional) for dimensionality reduction
        2. For each pair of batches, finds mutual nearest neighbors
        3. Computes cell-specific correction vectors using MNN pairs
        4. Applies correction with Gaussian weighting based on distance

    The correction is applied as:
        X_corrected = X - sum(w_i * correction_vector_i)

    where w_i are Gaussian weights based on distance to MNN pairs.

    Examples
    --------
    >>> container = integrate_mnn(container, batch_key='batch')
    >>> # Use more neighbors and different sigma
    >>> container = integrate_mnn(container, batch_key='batch', k=30, sigma=1.5)
    """
    # Validate parameters
    if k <= 0:
        raise ScpValueError(f"k must be positive, got {k}.", parameter="k", value=k)
    if sigma <= 0:
        raise ScpValueError(f"sigma must be positive, got {sigma}.", parameter="sigma", value=sigma)

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
    X = assay.layers[base_layer].X.copy()
    M = assay.layers[base_layer].M
    input_was_sparse = is_sparse_matrix(X)

    # Convert to dense and impute NaNs
    X = _prepare_mnn_data(X)

    # Get batch information
    batches = obs_df[batch_key].to_numpy()
    unique_batches = np.unique(batches)

    if len(unique_batches) < 2:
        raise ScpValueError(
            "MNN correction requires at least 2 batches.",
            parameter="batch_key",
            value=batch_key,
        )

    # Compute PCA for efficient neighbor search
    X_pca = _compute_pca_for_mnn(X, n_pcs) if use_pca else X

    # Apply correction
    X_corrected = _apply_mnn_correction(
        X, X_pca, batches, unique_batches, k, sigma, use_anchor_correction
    )

    # Preserve sparsity if appropriate
    if input_was_sparse:
        sparsity_ratio = 1.0 - (np.count_nonzero(X_corrected) / X_corrected.size)
        if sparsity_ratio > 0.5:
            X_corrected = sp.csr_matrix(X_corrected)

    # Create new layer
    new_matrix = ScpMatrix(
        X=X_corrected,
        M=M.copy() if M is not None else None,
    )
    assay.add_layer(new_layer_name or "mnn_corrected", new_matrix)

    # Log operation
    container.log_operation(
        action="integration_mnn",
        params={
            "batch_key": batch_key,
            "assay": assay_name,
            "k": k,
            "sigma": sigma,
            "use_pca": use_pca,
            "n_batches": len(unique_batches),
        },
        description=f"MNN correction (k={k}, sigma={sigma}) on assay '{assay_name}'.",
    )

    return container


def _prepare_mnn_data(X: np.ndarray | sp.spmatrix) -> np.ndarray:
    """Convert sparse to dense and impute NaN values."""
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X)

    if np.isnan(X).any():
        col_mean = np.nan_to_num(np.nanmean(X, axis=0), nan=0.0)
        nan_idx = np.where(np.isnan(X))
        X[nan_idx] = np.take(col_mean, nan_idx[1])

    return X


def _compute_pca_for_mnn(X: np.ndarray, n_pcs: int | None) -> np.ndarray:
    """Compute PCA for MNN neighbor search."""
    n_pcs = n_pcs or min(50, X.shape[1], X.shape[0] - 1)
    pca = PCA(n_components=n_pcs, random_state=42)
    return pca.fit_transform(X)


def _apply_mnn_correction(
    X: np.ndarray,
    X_pca: np.ndarray,
    batches: np.ndarray,
    unique_batches: np.ndarray,
    k: int,
    sigma: float,
    use_anchor_correction: bool,
) -> np.ndarray:
    """Apply MNN correction using anchor-based or pairwise approach."""
    X_corrected = X.copy()

    if use_anchor_correction and len(unique_batches) > 2:
        return _anchor_based_correction(X, X_pca, batches, unique_batches, k, sigma)

    # Pairwise correction for two batches
    for batch1 in unique_batches:
        idx1 = np.where(batches == batch1)[0]
        if len(idx1) == 0:
            continue

        corrections = _collect_corrections_from_other_batches(
            X, X_pca, batches, batch1, unique_batches, idx1, k, sigma
        )

        # Apply averaged corrections
        for cell_idx in range(len(idx1)):
            global_idx = idx1[cell_idx]
            cell_corrections = [c.get(global_idx, np.zeros(X.shape[1])) for c in corrections]
            if cell_corrections:
                X_corrected[global_idx] -= np.mean(cell_corrections, axis=0)

    return X_corrected


def _collect_corrections_from_other_batches(
    X: np.ndarray,
    X_pca: np.ndarray,
    batches: np.ndarray,
    batch1: any,
    unique_batches: np.ndarray,
    idx1: np.ndarray,
    k: int,
    sigma: float,
) -> list[dict[int, np.ndarray]]:
    """Collect correction vectors from all other batches."""
    corrections = []

    for batch2 in unique_batches:
        if batch1 == batch2:
            continue

        idx2 = np.where(batches == batch2)[0]
        if len(idx2) == 0:
            continue

        mnn_pairs = _find_mnn_pairs(X_pca[idx1], X_pca[idx2], k)
        if len(mnn_pairs) == 0:
            continue

        mnn_pairs_global = [(idx1[p[0]], idx2[p[1]]) for p in mnn_pairs]
        pairs_correction = _compute_correction_vectors(X, X_pca, mnn_pairs_global, sigma)
        corrections.append(pairs_correction)

    return corrections


def _anchor_based_correction(
    X: np.ndarray,
    X_pca: np.ndarray,
    batches: np.ndarray,
    unique_batches: np.ndarray,
    k: int = 20,
    sigma: float = 1.0,
) -> np.ndarray:
    """Apply anchor-based MNN correction for multiple batches.

    Uses the first batch as an anchor and corrects all other batches
    to align with it. This is more stable for multi-batch integration.

    Parameters
    ----------
    X : np.ndarray
        Original data (n_samples x n_features)
    X_pca : np.ndarray
        PCA-reduced data for MNN search
    batches : np.ndarray
        Batch labels for all samples
    unique_batches : np.ndarray
        Unique batch values
    k : int
        Number of nearest neighbors
    sigma : float
        Gaussian kernel sigma

    Returns
    -------
    np.ndarray
        Corrected data matrix
    """
    X_corrected = X.copy()
    anchor_batch = unique_batches[0]
    anchor_idx = np.where(batches == anchor_batch)[0]

    for batch in unique_batches[1:]:
        batch_idx = np.where(batches == batch)[0]
        if len(batch_idx) == 0:
            continue

        mnn_pairs = _find_mnn_pairs(X_pca[anchor_idx], X_pca[batch_idx], k)
        if len(mnn_pairs) == 0:
            continue

        mnn_pairs_global = [(anchor_idx[p[0]], batch_idx[p[1]]) for p in mnn_pairs]
        batch_corrections = _compute_batch_correction(
            X, X_pca, mnn_pairs_global, anchor_idx, batch_idx, sigma
        )

        for i, local_idx in enumerate(batch_idx):
            if i in batch_corrections:
                X_corrected[local_idx] -= batch_corrections[i]

    return X_corrected


def _find_mnn_pairs(
    X1: np.ndarray,
    X2: np.ndarray,
    k: int = 20,
) -> list[tuple[int, int]]:
    """Find mutual nearest neighbors between two batches.

    Parameters
    ----------
    X1 : np.ndarray
        Data from batch 1 (n1_samples x n_features)
    X2 : np.ndarray
        Data from batch 2 (n2_samples x n_features)
    k : int
        Number of nearest neighbors

    Returns
    -------
    list of tuple
        List of (index_in_X1, index_in_X2) MNN pairs
    """
    # Find nearest neighbors in X2 for each cell in X1
    nn1 = NearestNeighbors(n_neighbors=min(k, len(X2)), algorithm="auto")
    nn1.fit(X2)
    indices1 = nn1.kneighbors(X1, return_distance=False)

    # Find nearest neighbors in X1 for each cell in X2
    nn2 = NearestNeighbors(n_neighbors=min(k, len(X1)), algorithm="auto")
    nn2.fit(X1)
    indices2 = nn2.kneighbors(X2, return_distance=False)

    # Build adjacency sets for efficient mutual lookup
    mnn_pairs = []
    for i, neighbors in enumerate(indices1):
        for j in neighbors:
            if i in indices2[j]:
                mnn_pairs.append((i, j))

    return mnn_pairs


def _compute_correction_vectors(
    X: np.ndarray,
    X_pca: np.ndarray,
    mnn_pairs: Sequence[tuple[int, int]],
    sigma: float = 1.0,
) -> dict[int, np.ndarray]:
    """Compute correction vectors for cells using MNN pairs.

    Parameters
    ----------
    X : np.ndarray
        Original data (high-dimensional)
    X_pca : np.ndarray
        PCA-reduced data
    mnn_pairs : Sequence of tuple
        List of MNN pairs (idx_batch1, idx_batch2)
    sigma : float
        Gaussian kernel sigma

    Returns
    -------
    dict
        Dictionary mapping cell index to correction vector
    """
    cell_pairs = defaultdict(list)
    for i, j in mnn_pairs:
        cell_pairs[i].append((i, j))

    correction_dict = {}
    for cell_idx, pairs in cell_pairs.items():
        correction, total_weight = _compute_weighted_correction(X, X_pca, pairs, sigma)
        correction_dict[cell_idx] = correction / total_weight if total_weight > 0 else correction

    return correction_dict


def _compute_batch_correction(
    X: np.ndarray,
    X_pca: np.ndarray,
    mnn_pairs: Sequence[tuple[int, int]],
    anchor_idx: np.ndarray,
    batch_idx: np.ndarray,
    sigma: float = 1.0,
) -> dict[int, np.ndarray]:
    """Compute correction vectors for a batch relative to anchor.

    Parameters
    ----------
    X : np.ndarray
        Original data
    X_pca : np.ndarray
        PCA-reduced data
    mnn_pairs : Sequence of tuple
        MNN pairs (anchor_global_idx, batch_global_idx)
    anchor_idx : np.ndarray
        Indices of anchor batch
    batch_idx : np.ndarray
        Indices of batch to correct
    sigma : float
        Gaussian kernel sigma

    Returns
    -------
    dict
        Maps local batch index to correction vector
    """
    global_to_local_batch = {global_idx: i for i, global_idx in enumerate(batch_idx)}

    cell_pairs = defaultdict(list)
    for anchor_global, batch_global in mnn_pairs:
        batch_local = global_to_local_batch[batch_global]
        cell_pairs[batch_local].append((anchor_global, batch_global))

    correction_dict = {}
    for batch_local, pairs in cell_pairs.items():
        correction, total_weight = _compute_weighted_correction(X, X_pca, pairs, sigma)
        correction_dict[batch_local] = correction / total_weight if total_weight > 0 else correction

    return correction_dict


def _compute_weighted_correction(
    X: np.ndarray,
    X_pca: np.ndarray,
    pairs: Sequence[tuple[int, int]],
    sigma: float,
) -> tuple[np.ndarray, float]:
    """Compute weighted correction vector from MNN pairs.

    Parameters
    ----------
    X : np.ndarray
        Original data
    X_pca : np.ndarray
        PCA-reduced data for distance computation
    pairs : Sequence of tuple
        MNN pairs (idx1, idx2)
    sigma : float
        Gaussian kernel sigma

    Returns
    -------
    correction : np.ndarray
        Weighted correction vector
    total_weight : float
        Sum of all weights (for normalization)
    """
    correction = np.zeros(X.shape[1])
    total_weight = 0.0

    for i, j in pairs:
        diff = X[j] - X[i]
        pca_dist = np.linalg.norm(X_pca[i] - X_pca[j])
        weight = np.exp(-(pca_dist**2) / (2 * sigma**2))

        correction += weight * diff
        total_weight += weight

    return correction, total_weight


__all__ = ["integrate_mnn"]


if __name__ == "__main__":
    # Test: Basic functionality
    print("Testing MNN correction...")

    np.random.seed(42)
    n_samples_per_batch = 50
    n_features = 30

    # Generate base data
    X_base = np.random.randn(n_samples_per_batch, n_features)

    # Create two batches with different shifts
    X_batch1 = X_base + np.random.randn(n_samples_per_batch, n_features) * 0.1
    X_batch2 = X_base + np.random.randn(n_samples_per_batch, n_features) * 0.1 + 2.0

    X = np.vstack([X_batch1, X_batch2])

    # Create container
    var = pl.DataFrame({"_index": [f"prot_{i}" for i in range(n_features)]})
    obs = pl.DataFrame(
        {
            "_index": [f"cell_{i}" for i in range(2 * n_samples_per_batch)],
            "batch": ["batch1"] * n_samples_per_batch + ["batch2"] * n_samples_per_batch,
        }
    )

    assay = Assay(var=var)
    assay.add_layer("raw", ScpMatrix(X=X, M=None))
    container = ScpContainer(obs=obs, assays={"protein": assay})

    # Test MNN correction
    result = integrate_mnn(
        container,
        batch_key="batch",
        assay_name="protein",
        base_layer="raw",
        k=10,
        sigma=1.0,
        n_pcs=15,
    )

    assert "mnn_corrected" in result.assays["protein"].layers
    X_corrected = result.assays["protein"].layers["mnn_corrected"].X

    # Check batch effect reduction
    mean1 = np.mean(X_corrected[:n_samples_per_batch], axis=0)
    mean2 = np.mean(X_corrected[n_samples_per_batch:], axis=0)
    batch_diff = np.linalg.norm(mean1 - mean2)
    original_diff = np.linalg.norm(np.mean(X_batch1, axis=0) - np.mean(X_batch2, axis=0))

    print(f"  Original batch difference: {original_diff:.3f}")
    print(f"  Corrected batch difference: {batch_diff:.3f}")
    print(f"  Reduction ratio: {batch_diff / original_diff:.3f}")
    print(f"  Shape: {X_corrected.shape}")

    # Test with sparse input
    print("  Testing with sparse input...")
    X_sparse = sp.csr_matrix(X)
    assay2 = Assay(var=var)
    assay2.add_layer("raw", ScpMatrix(X=X_sparse, M=None))
    container2 = ScpContainer(obs=obs, assays={"protein": assay2})

    result2 = integrate_mnn(
        container2,
        batch_key="batch",
        assay_name="protein",
        base_layer="raw",
        k=10,
        sigma=1.0,
        n_pcs=15,
    )

    assert "mnn_corrected" in result2.assays["protein"].layers
    print("  Sparse input test passed")

    # Test with three batches (anchor-based correction)
    print("  Testing with three batches...")
    X_batch3 = X_base + np.random.randn(n_samples_per_batch, n_features) * 0.1 + 1.0
    X3 = np.vstack([X_batch1, X_batch2, X_batch3])

    obs3 = pl.DataFrame(
        {
            "_index": [f"cell_{i}" for i in range(3 * n_samples_per_batch)],
            "batch": (
                ["batch1"] * n_samples_per_batch
                + ["batch2"] * n_samples_per_batch
                + ["batch3"] * n_samples_per_batch
            ),
        }
    )

    assay3 = Assay(var=var)
    assay3.add_layer("raw", ScpMatrix(X=X3, M=None))
    container3 = ScpContainer(obs=obs3, assays={"protein": assay3})

    result3 = integrate_mnn(
        container3,
        batch_key="batch",
        assay_name="protein",
        base_layer="raw",
        k=10,
        sigma=1.0,
    )

    assert "mnn_corrected" in result3.assays["protein"].layers
    print("  Three batch test passed")
    print("  All tests passed")
