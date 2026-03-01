"""K-Means clustering for single-cell proteomics data.

This module provides K-means clustering, aligned with scanpy's API.

Reference:
    Lloyd, S. (1982). Least squares quantization in PCM.
    IEEE Transactions on Information Theory, 28(2), 129-137.
"""

from __future__ import annotations

from typing import Literal

from sklearn.cluster import KMeans as SKLearnKMeans

from scptensor.cluster.base import (
    _add_labels_to_obs,
    _get_default_key,
    _prepare_matrix,
    _validate_assay_layer,
)
from scptensor.core.exceptions import ScpValueError
from scptensor.core.jit_ops import NUMBA_AVAILABLE, kmeans_core_numba, kmeans_plusplus_init_numba
from scptensor.core.structures import ScpContainer


def cluster_kmeans(
    container: ScpContainer,
    assay_name: str = "pca",
    base_layer: str = "X",
    n_clusters: int = 5,
    random_state: int = 42,
    backend: Literal["sklearn", "numba"] = "sklearn",
    key_added: str | None = None,
) -> ScpContainer:
    """K-Means clustering.

    Performs k-means clustering on the specified assay layer.

    Parameters
    ----------
    container : ScpContainer
        Input container.
    assay_name : str, default="pca"
        Name of assay to use.
    base_layer : str, default="X"
        Name of layer to use.
    n_clusters : int, default=5
        Number of clusters.
    random_state : int, default=42
        Random seed.
    backend : {"sklearn", "numba"}, default="sklearn"
        Clustering backend.
    key_added : str or None, optional
        Column name for results. If None, auto-generated.

    Returns
    -------
    ScpContainer
        Container with clustering results in obs.

    Raises
    ------
    ScpValueError
        If n_clusters or backend invalid.
    ImportError
        If backend='numba' but numba not installed.

    Examples
    --------
    >>> from scptensor import create_test_container
    >>> from scptensor.cluster import cluster_kmeans
    >>> container = create_test_container()
    >>> result = cluster_kmeans(container, n_clusters=10)
    >>> "kmeans_k10" in result.obs.columns
    True
    """
    # Validate parameters
    if n_clusters <= 0:
        raise ScpValueError(
            f"n_clusters must be positive, got {n_clusters}.",
            parameter="n_clusters",
            value=n_clusters,
        )
    if backend not in ("sklearn", "numba"):
        raise ScpValueError(
            f"backend must be 'sklearn' or 'numba', got '{backend}'.",
            parameter="backend",
            value=backend,
        )

    # Get and prepare data
    assay, X = _validate_assay_layer(container, assay_name, base_layer)
    X_dense = _prepare_matrix(X)

    # Run clustering
    if backend == "numba":
        if not NUMBA_AVAILABLE:
            raise ImportError("backend='numba' requires numba. Install with: pip install numba")
        init_centers = kmeans_plusplus_init_numba(X_dense, k=n_clusters, seed=random_state)
        _, labels, _ = kmeans_core_numba(
            X_dense,
            k=n_clusters,
            max_iter=300,
            tol=1e-4,
            init_centers=init_centers,
            rng_seed=random_state,
        )
    else:
        labels = SKLearnKMeans(
            n_clusters=n_clusters, random_state=random_state, n_init="auto"
        ).fit_predict(X_dense)

    # Generate key
    if key_added is None:
        key_added = _get_default_key("kmeans", {"n_clusters": n_clusters})

    # Add to obs
    new_container = _add_labels_to_obs(container, labels, key_added)

    # Log operation
    new_container.log_operation(
        action="cluster_kmeans",
        params={
            "assay": assay_name,
            "layer": base_layer,
            "n_clusters": n_clusters,
            "backend": backend,
        },
        description=f"K-Means (k={n_clusters}, backend={backend}) on {assay_name}/{base_layer}.",
    )

    return new_container


__all__ = ["cluster_kmeans"]
