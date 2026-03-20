"""Mutual Nearest Neighbors (MNN) correction for DIA-based single-cell proteomics data.

MNN identifies pairs of cells from different batches that are mutual nearest
neighbors in the high-dimensional space. These pairs are used to estimate and
correct batch-specific biases.

Algorithm
---------

For each pair of batches, MNN finds cell pairs that are mutual nearest neighbors
in the reduced dimensional space (PCA by default). The correction vector for each
cell is computed as a weighted sum of differences from its MNN pairs:

.. math::

    X_{corrected} = X - \\sum_i w_i \\cdot (X_{pair} - X)

where the weights use a Gaussian kernel:

.. math::

    w_i = \\exp\\left(-\\frac{d_i^2}{2\\sigma^2}\\right)

References
----------
Haghverdi L, et al. Batch effects in single-cell RNA-sequencing data
are corrected by matching mutual nearest neighbors. Nature Biotechnology (2018).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from scptensor.core.exceptions import ScpValueError
from scptensor.core.structures import ScpContainer
from scptensor.integration.base import (
    add_integrated_layer,
    log_integration_operation,
    prepare_integration_input,
    preserve_sparsity,
    register_integrate_method,
    validate_batch_integration_params,
    validate_layer_params,
)


@register_integrate_method("mnn", integration_level="embedding", recommended_for_de=False)
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

    Parameters
    ----------
    container : ScpContainer
        Input container with multiple batches
    batch_key : str
        Column name in obs containing batch labels
    assay_name : str, default="protein"
        Name of the assay to use
    base_layer : str, default="raw"
        Layer to use as input
    new_layer_name : str | None, default="mnn_corrected"
        Name for the corrected layer
    k : int, default=20
        Number of nearest neighbors to find
    sigma : float, default=1.0
        Sigma parameter for Gaussian kernel
    n_pcs : int | None
        Number of principal components to use for MNN search (default: 50)
    use_pca : bool, default=True
        Whether to use PCA for MNN search
    use_anchor_correction : bool, default=True
        Whether to use anchor-based correction (first batch as anchor)

    Returns
    -------
    ScpContainer
        Container with batch-corrected data

    Examples
    --------
    >>> container = integrate_mnn(container, batch_key='batch')
    >>> container = integrate_mnn(container, batch_key='batch', k=30, sigma=1.5)
    """

    # Validate parameters
    if k <= 0:
        raise ScpValueError(f"k must be positive, got {k}.", parameter="k", value=k)
    if sigma <= 0:
        raise ScpValueError(f"sigma must be positive, got {sigma}.", parameter="sigma", value=sigma)

    # Validate assay and layer
    assay, layer = validate_layer_params(container, assay_name, base_layer)
    _, batches, unique_batches, _ = validate_batch_integration_params(
        container, batch_key, assay_name, min_batches=2
    )

    # Get and prepare data
    X, input_was_sparse = prepare_integration_input(layer, context="MNN integration")

    # Compute PCA for efficient neighbor search
    X_pca = _compute_pca_for_mnn(X, n_pcs) if use_pca else X

    # Apply correction
    X_corrected = _apply_mnn_correction(
        X, X_pca, batches, unique_batches, k, sigma, use_anchor_correction
    )

    # Preserve sparsity if appropriate
    X_corrected = preserve_sparsity(X_corrected, input_was_sparse)

    # Create new layer
    add_integrated_layer(assay, new_layer_name or "mnn_corrected", X_corrected, layer)

    # Log operation
    return log_integration_operation(
        container,
        action="integration_mnn",
        method_name="mnn",
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

    # Use progressive anchor correction by default to match MNN integration semantics.
    if use_anchor_correction and len(unique_batches) >= 2:
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
    batch1: Any,
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
    """Apply anchor-based MNN correction for multiple batches."""
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

        # Keep pair ordering as (target_cell, reference_cell) for correction.
        mnn_pairs_global = [(batch_idx[p[1]], anchor_idx[p[0]]) for p in mnn_pairs]
        batch_corrections = _compute_batch_correction(X, X_pca, mnn_pairs_global, batch_idx, sigma)

        for i, local_idx in enumerate(batch_idx):
            if i in batch_corrections:
                X_corrected[local_idx] -= batch_corrections[i]

    return X_corrected


def _find_mnn_pairs(
    X1: np.ndarray,
    X2: np.ndarray,
    k: int = 20,
) -> list[tuple[int, int]]:
    """Find mutual nearest neighbors between two batches."""
    nn1 = NearestNeighbors(n_neighbors=min(k, len(X2)), algorithm="auto")
    nn1.fit(X2)
    indices1 = nn1.kneighbors(X1, return_distance=False)

    nn2 = NearestNeighbors(n_neighbors=min(k, len(X1)), algorithm="auto")
    nn2.fit(X1)
    indices2 = nn2.kneighbors(X2, return_distance=False)

    mnn_pairs = []
    for i, neighbors in enumerate(indices1):
        for j in neighbors:
            if i in indices2[j]:
                mnn_pairs.append((i, j))

    return mnn_pairs


def _compute_correction_vectors(
    X: np.ndarray,
    X_pca: np.ndarray,
    mnn_pairs: list[tuple[int, int]],
    sigma: float = 1.0,
) -> dict[int, np.ndarray]:
    """Compute correction vectors for cells using MNN pairs."""
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
    mnn_pairs: list[tuple[int, int]],
    batch_idx: np.ndarray,
    sigma: float = 1.0,
) -> dict[int, np.ndarray]:
    """Compute correction vectors for a batch relative to anchor."""
    global_to_local_batch = {global_idx: i for i, global_idx in enumerate(batch_idx)}

    cell_pairs = defaultdict(list)
    for batch_global, anchor_global in mnn_pairs:
        batch_local = global_to_local_batch[batch_global]
        cell_pairs[batch_local].append((batch_global, anchor_global))

    correction_dict = {}
    for batch_local, pairs in cell_pairs.items():
        correction, total_weight = _compute_weighted_correction(X, X_pca, pairs, sigma)
        correction_dict[batch_local] = correction / total_weight if total_weight > 0 else correction

    return correction_dict


def _compute_weighted_correction(
    X: np.ndarray,
    X_pca: np.ndarray,
    pairs: list[tuple[int, int]],
    sigma: float,
) -> tuple[np.ndarray, float]:
    """Compute weighted correction vector from MNN pairs."""
    correction = np.zeros(X.shape[1])
    total_weight = 0.0

    for i, j in pairs:
        # Pairs are ordered as (target_cell, reference_cell).
        diff = X[i] - X[j]
        pca_dist = np.linalg.norm(X_pca[i] - X_pca[j])
        weight = np.exp(-(pca_dist**2) / (2 * sigma**2))

        correction += weight * diff
        total_weight += weight

    return correction, total_weight


__all__ = ["integrate_mnn"]
