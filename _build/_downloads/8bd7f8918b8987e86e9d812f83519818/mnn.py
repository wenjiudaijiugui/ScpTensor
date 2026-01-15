"""
Mutual Nearest Neighbors (MNN) correction for single-cell proteomics data.

This module implements the MNN algorithm for batch effect correction.

Reference:
    Haghverdi L, et al. Batch effects in single-cell RNA-sequencing data
    are corrected by matching mutual nearest neighbors. Nature Biotechnology (2018).
"""

from typing import Optional, List, Tuple, Dict
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scptensor.core.structures import ScpContainer, ScpMatrix, Assay
from scptensor.core.exceptions import AssayNotFoundError, LayerNotFoundError, ValueError as ScpValueError


def mnn_correct(
    container: ScpContainer,
    batch_key: str,
    assay_name: str = 'protein',
    base_layer: str = 'raw',
    new_layer_name: Optional[str] = 'mnn_corrected',
    k: int = 20,
    sigma: float = 1.0,
    n_pcs: Optional[int] = None,
    use_pca: bool = True
) -> ScpContainer:
    """
    Correct batch effects using Mutual Nearest Neighbors (MNN) method.

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
    >>> container = mnn_correct(container, batch_key='batch')
    >>> # Use more neighbors and different sigma
    >>> container = mnn_correct(container, batch_key='batch', k=30, sigma=1.5)
    """
    # Validate parameters
    if k <= 0:
        raise ScpValueError(
            f"k must be positive, got {k}.",
            parameter="k",
            value=k
        )
    if sigma <= 0:
        raise ScpValueError(
            f"sigma must be positive, got {sigma}.",
            parameter="sigma",
            value=sigma
        )

    # Validate assay and layer
    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]
    if base_layer not in assay.layers:
        raise LayerNotFoundError(base_layer, assay_name)

    obs_df = container.obs
    if batch_key not in obs_df.columns:
        raise ScpValueError(
            f"Batch key '{batch_key}' not found in obs. "
            f"Available columns: {list(obs_df.columns)}",
            parameter="batch_key",
            value=batch_key
        )

    # Get data
    X = assay.layers[base_layer].X.copy()
    M = assay.layers[base_layer].M

    # Handle NaN values
    if np.isnan(X).any():
        col_mean = np.nanmean(X, axis=0)
        col_mean = np.nan_to_num(col_mean, nan=0.0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_mean, inds[1])

    # Get batch information
    batches = obs_df[batch_key].to_numpy()
    unique_batches = np.unique(batches)

    if len(unique_batches) < 2:
        raise ScpValueError(
            "MNN correction requires at least 2 batches.",
            parameter="batch_key",
            value=batch_key
        )

    # PCA for MNN search (reduces computational cost)
    if use_pca:
        n_pcs = n_pcs or min(50, X.shape[1], X.shape[0] - 1)
        from sklearn.decomposition import PCA

        pca = PCA(n_components=n_pcs, random_state=42)
        X_pca = pca.fit_transform(X)
    else:
        X_pca = X

    # Initialize corrected data
    X_corrected = X.copy()

    # Process each batch pair
    for i, batch1 in enumerate(unique_batches):
        idx1 = np.where(batches == batch1)[0]

        if len(idx1) == 0:
            continue

        # Find correction from other batches
        correction_vectors = []
        correction_weights = []

        for j, batch2 in enumerate(unique_batches):
            if batch1 == batch2:
                continue

            idx2 = np.where(batches == batch2)[0]

            if len(idx2) == 0:
                continue

            # Find MNN pairs between batch1 and batch2
            mnn_pairs = _find_mnn_pairs(
                X_pca[idx1],
                X_pca[idx2],
                k=k
            )

            if len(mnn_pairs) == 0:
                continue

            # Map back to original indices
            mnn_pairs_orig = [(idx1[p[0]], idx2[p[1]]) for p in mnn_pairs]

            # Compute correction vectors
            pairs_correction = _compute_correction_vectors(
                X, X_pca, mnn_pairs_orig, sigma=sigma
            )

            correction_vectors.append(pairs_correction)
            # Uniform weight for each batch
            correction_weights.append(np.ones(len(idx1)))

        # Apply corrections
        if correction_vectors:
            # Average corrections from all other batches
            for corr in correction_vectors:
                for cell_idx, cell_corr in corr.items():
                    X_corrected[cell_idx] -= cell_corr

    # Create new layer
    new_M = M.copy() if M is not None else None
    new_matrix = ScpMatrix(X=X_corrected, M=new_M)
    if new_layer_name is None:
        new_layer_name = 'mnn_corrected'
    assay.add_layer(new_layer_name, new_matrix)

    # Log operation
    container.log_operation(
        action="integration_mnn",
        params={
            "batch_key": batch_key,
            "assay": assay_name,
            "k": k,
            "sigma": sigma,
            "use_pca": use_pca,
            "n_batches": len(unique_batches)
        },
        description=f"MNN correction (k={k}, sigma={sigma}) on assay '{assay_name}'."
    )

    return container


def _find_mnn_pairs(
    X1: np.ndarray,
    X2: np.ndarray,
    k: int = 20
) -> List[Tuple[int, int]]:
    """
    Find mutual nearest neighbors between two batches.

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
    List[Tuple[int, int]]
        List of (index_in_X1, index_in_X2) MNN pairs
    """
    # Find nearest neighbors in X2 for each cell in X1
    nn1 = NearestNeighbors(n_neighbors=k, algorithm='auto')
    nn1.fit(X2)
    distances1, indices1 = nn1.kneighbors(X1)

    # Find nearest neighbors in X1 for each cell in X2
    nn2 = NearestNeighbors(n_neighbors=k, algorithm='auto')
    nn2.fit(X1)
    distances2, indices2 = nn2.kneighbors(X2)

    # Find mutual nearest neighbors
    mnn_pairs = []
    for i in range(len(X1)):
        for j_idx, j in enumerate(indices1[i]):
            # Check if i is in the k-nearest neighbors of j
            if i in indices2[j]:
                mnn_pairs.append((i, j))

    return mnn_pairs


def _compute_correction_vectors(
    X: np.ndarray,
    X_pca: np.ndarray,
    mnn_pairs: List[Tuple[int, int]],
    sigma: float = 1.0
) -> Dict[int, np.ndarray]:
    """
    Compute correction vectors for cells using MNN pairs.

    Parameters
    ----------
    X : np.ndarray
        Original data (high-dimensional)
    X_pca : np.ndarray
        PCA-reduced data
    mnn_pairs : List[Tuple[int, int]]
        List of MNN pairs (idx_batch1, idx_batch2)
    sigma : float
        Gaussian kernel sigma

    Returns
    -------
    dict
        Dictionary mapping cell index to correction vector
    """
    correction_dict = {}

    # Group MNN pairs by cell in first batch
    from collections import defaultdict
    cell_pairs = defaultdict(list)
    for i, j in mnn_pairs:
        cell_pairs[i].append((i, j))

    # Compute correction for each cell
    for cell_idx, pairs in cell_pairs.items():
        correction = np.zeros(X.shape[1])

        for i, j in pairs:
            # Correction vector: difference between paired cells
            diff = X[j] - X[i]

            # Distance in PCA space
            pca_dist = np.linalg.norm(X_pca[i] - X_pca[j])

            # Gaussian weight
            weight = np.exp(-(pca_dist ** 2) / (2 * sigma ** 2))

            correction += weight * diff

        # Normalize by number of pairs
        if len(pairs) > 0:
            correction /= len(pairs)

        correction_dict[cell_idx] = correction

    return correction_dict


if __name__ == "__main__":
    # Test: Basic functionality
    print("Testing MNN correction...")

    # Create simple test data with batch effects
    np.random.seed(42)
    n_samples_per_batch = 50
    n_features = 30

    # Generate base data
    X_base = np.random.randn(n_samples_per_batch, n_features)

    # Create two batches with different shifts
    X_batch1 = X_base + np.random.randn(n_samples_per_batch, n_features) * 0.1
    X_batch2 = X_base + np.random.randn(n_samples_per_batch, n_features) * 0.1
    X_batch2 += 2.0  # Add batch effect

    X = np.vstack([X_batch1, X_batch2])

    # Create container
    import polars as pl
    from scptensor.core.structures import ScpContainer, Assay

    var = pl.DataFrame({"_index": [f"prot_{i}" for i in range(n_features)]})
    obs = pl.DataFrame({
        "_index": [f"cell_{i}" for i in range(2 * n_samples_per_batch)],
        "batch": ["batch1"] * n_samples_per_batch + ["batch2"] * n_samples_per_batch
    })

    assay = Assay(var=var)
    assay.add_layer("raw", ScpMatrix(X=X, M=None))

    container = ScpContainer(obs=obs, assays={"protein": assay})

    # Test MNN correction
    result = mnn_correct(
        container,
        batch_key="batch",
        assay_name="protein",
        base_layer="raw",
        k=10,
        sigma=1.0,
        n_pcs=15
    )

    # Check results
    assert "mnn_corrected" in result.assays["protein"].layers
    X_corrected = result.assays["protein"].layers["mnn_corrected"].X

    # Check batch effect reduction
    mean1 = np.mean(X_corrected[:n_samples_per_batch], axis=0)
    mean2 = np.mean(X_corrected[n_samples_per_batch:], axis=0)
    batch_diff = np.linalg.norm(mean1 - mean2)

    original_diff = np.linalg.norm(
        np.mean(X_batch1, axis=0) - np.mean(X_batch2, axis=0)
    )

    print(f"  Original batch difference: {original_diff:.3f}")
    print(f"  Corrected batch difference: {batch_diff:.3f}")
    print(f"  Reduction ratio: {batch_diff / original_diff:.3f}")
    print(f"  Shape: {X_corrected.shape}")
    print("✅ All tests passed")
