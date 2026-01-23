"""K-Means clustering with flexible storage options.

Backend Options:
    - sklearn: Reliable, well-tested, O(n*k*i*d) complexity
    - numba: 2-5x faster on large datasets, same complexity (requires numba)

Storage Options:
    - obs: Store cluster labels as column in obs (simple, for visualization/annotation)
    - assay: Store as new assay with one-hot encoding (for downstream analysis)
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import polars as pl
from sklearn.cluster import KMeans as SKLearnKMeans

from scptensor.core.exceptions import AssayNotFoundError, LayerNotFoundError, ScpValueError
from scptensor.core.jit_ops import NUMBA_AVAILABLE, kmeans_core_numba, kmeans_plusplus_init_numba
from scptensor.core.structures import Assay, ScpContainer, ScpMatrix


def cluster_kmeans(
    container: ScpContainer,
    assay_name: str = "pca",
    base_layer: str = "X",
    n_clusters: int = 5,
    random_state: int = 42,
    backend: Literal["sklearn", "numba"] = "sklearn",
    max_iter: int = 300,
    tol: float = 1e-4,
    storage: Literal["obs", "assay"] = "obs",
    col_name: str | None = None,
    new_assay_name: str = "cluster_kmeans",
    key_added: str | None = None,
) -> ScpContainer:
    """K-Means clustering with flexible storage and backend selection.

    Parameters
    ----------
    container : ScpContainer
        Input container.
    assay_name : str, default="pca"
        Name of the assay to use for clustering.
    base_layer : str, default="X"
        Name of the layer within the assay.
    n_clusters : int, default=5
        Number of clusters (k).
    random_state : int, default=42
        Random seed for reproducibility.
    backend : {"sklearn", "numba"}, default="sklearn"
        Clustering backend to use.
    max_iter : int, default=300
        Maximum number of iterations. Only used for numba backend.
    tol : float, default=1e-4
        Convergence tolerance. Only used for numba backend.
    storage : {"obs", "assay"}, default="obs"
        How to store clustering results:
        - "obs": Store as column in obs (simple, for visualization/annotation)
        - "assay": Store as new assay with one-hot encoding (for downstream analysis)
    col_name : str | None, default=None
        Column name for obs storage. If None, uses "kmeans_k{n_clusters}".
        Only used when storage="obs".
    new_assay_name : str, default="cluster_kmeans"
        Name for the new clustering assay.
        Only used when storage="assay".
    key_added : str | None, default=None
        If provided and storage="assay", also adds cluster labels to obs.

    Returns
    -------
    ScpContainer
        New container with clustering results stored according to storage parameter.

    Raises
    ------
    ScpValueError
        If n_clusters is not positive or backend is invalid.
    ImportError
        If backend='numba' but numba is not installed.

    Notes
    -----
    Choose storage mode based on your use case:

    - **storage="obs"** (default): Simple storage as a column in obs.
      Best for: Visualization, filtering, quick annotation
      Example: Plotting clusters in UMAP, filtering cells by cluster

    - **storage="assay"**: Creates new assay with one-hot encoded cluster matrix.
      Best for: Downstream analysis, cluster-specific operations
      Example: Computing cluster-specific statistics, using clusters as features

    Examples
    --------
    >>> # Store as column in obs (default, simplest)
    >>> container = cluster_kmeans(container, n_clusters=3)
    >>> "kmeans_k3" in container.obs.columns
    True

    >>> # Store with custom column name
    >>> container = cluster_kmeans(container, n_clusters=5, col_name="my_clusters")
    >>> "my_clusters" in container.obs.columns
    True

    >>> # Store as assay with one-hot encoding
    >>> container = cluster_kmeans(container, n_clusters=3, storage="assay")
    >>> "cluster_kmeans" in container.assays
    True

    >>> # Store as assay and also add to obs
    >>> container = cluster_kmeans(
    ...     container, n_clusters=3, storage="assay", key_added="cluster_labels"
    ... )
    >>> "cluster_kmeans" in container.assays and "cluster_labels" in container.obs.columns
    True

    >>> # Using numba backend for speed
    >>> container = cluster_kmeans(container, n_clusters=5, backend="numba")
    """
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
    if storage not in ("obs", "assay"):
        raise ScpValueError(
            f"storage must be 'obs' or 'assay', got '{storage}'.",
            parameter="storage",
            value=storage,
        )
    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)
    assay = container.assays[assay_name]
    if base_layer not in assay.layers:
        raise LayerNotFoundError(base_layer, assay_name)

    X = assay.layers[base_layer].X
    from scipy import sparse

    if sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)

    if backend == "numba":
        if not NUMBA_AVAILABLE:
            raise ImportError("backend='numba' requires numba. Install with: pip install numba")
        init_centers = kmeans_plusplus_init_numba(X, k=n_clusters, seed=random_state)
        centers, labels, inertia = kmeans_core_numba(
            X,
            k=n_clusters,
            max_iter=max_iter,
            tol=tol,
            init_centers=init_centers,
            rng_seed=random_state,
        )
    else:
        labels = SKLearnKMeans(
            n_clusters=n_clusters, random_state=random_state, n_init="auto"
        ).fit_predict(X)

    # Store results based on storage parameter
    if storage == "obs":
        # Simple obs storage (like basic.py)
        if col_name is None:
            col_name = f"kmeans_k{n_clusters}"
        new_obs = container.obs.with_columns(pl.Series(col_name, labels).cast(pl.String))
        new_container = ScpContainer(
            obs=new_obs, assays=container.assays, history=list(container.history)
        )
        new_container.log_operation(
            action="cluster_kmeans",
            params={
                "assay": assay_name,
                "layer": base_layer,
                "n_clusters": n_clusters,
                "backend": backend,
                "max_iter": max_iter if backend == "numba" else "default",
                "tol": tol if backend == "numba" else "default",
                "storage": storage,
                "col_name": col_name,
            },
            description=f"K-Means clustering (k={n_clusters}, backend={backend}, storage=obs) on {assay_name}/{base_layer}.",
        )
    else:
        # Assay storage with one-hot encoding (current kmeans.py behavior)
        n_samples = len(labels)
        one_hot = np.zeros((n_samples, n_clusters), dtype=np.float32)
        one_hot[np.arange(n_samples), labels] = 1.0
        M_cluster = np.zeros_like(one_hot, dtype=np.int8)

        var_cluster = pl.DataFrame(
            {
                "_index": [f"Cluster_{i}" for i in range(n_clusters)],
                "cluster_id": [f"Cluster_{i}" for i in range(n_clusters)],
            }
        )
        matrix_cluster = ScpMatrix(X=one_hot, M=M_cluster)
        assay_cluster = Assay(
            var=var_cluster, layers={"binary": matrix_cluster}, feature_id_col="cluster_id"
        )

        new_assays = dict(container.assays)
        new_assays[new_assay_name] = assay_cluster

        new_obs = container.obs
        if key_added:
            new_obs = container.obs.with_columns(pl.Series(key_added, labels).cast(pl.String))

        new_container = ScpContainer(
            obs=new_obs, assays=new_assays, history=list(container.history)
        )
        new_container.log_operation(
            action="cluster_kmeans",
            params={
                "source_assay": assay_name,
                "source_layer": base_layer,
                "n_clusters": n_clusters,
                "backend": backend,
                "max_iter": max_iter if backend == "numba" else "default",
                "tol": tol if backend == "numba" else "default",
                "storage": storage,
                "new_assay_name": new_assay_name,
                "key_added": key_added,
            },
            description=f"K-Means (k={n_clusters}, backend={backend}, storage=assay) on {assay_name}/{base_layer}.",
        )

    return new_container


if __name__ == "__main__":
    print("Testing cluster_kmeans with storage parameter support...")
    print(f"Numba available: {NUMBA_AVAILABLE}")

    np.random.seed(42)
    X_test = np.random.randn(100, 10)
    obs_test = pl.DataFrame({"_index": [f"cell_{i}" for i in range(100)]})
    var_test = pl.DataFrame({"_index": [f"PC_{i}" for i in range(10)]})
    test_assay = Assay(var=var_test, layers={"X": ScpMatrix(X=X_test)})
    test_container = ScpContainer(obs=obs_test, assays={"pca": test_assay})

    print("\n[Test 1] storage='obs' mode (default)")
    result_obs = cluster_kmeans(test_container, n_clusters=3)
    assert "kmeans_k3" in result_obs.obs.columns
    assert result_obs.obs["kmeans_k3"].n_unique() <= 3
    assert len(result_obs.history) == 1
    assert result_obs.n_samples == test_container.n_samples
    print("  PASS")

    print("\n[Test 2] storage='obs' with custom col_name")
    result_custom = cluster_kmeans(test_container, n_clusters=5, col_name="my_clusters")
    assert "my_clusters" in result_custom.obs.columns
    assert result_custom.obs["my_clusters"].n_unique() <= 5
    print("  PASS")

    print("\n[Test 3] storage='assay' mode")
    result_assay = cluster_kmeans(test_container, n_clusters=3, storage="assay")
    assert "cluster_kmeans" in result_assay.assays
    assert result_assay.assays["cluster_kmeans"].n_features == 3
    assert result_assay.assays["cluster_kmeans"].layers["binary"].X.shape[0] == 100
    print("  PASS")

    print("\n[Test 4] storage='assay' with key_added")
    result_both = cluster_kmeans(
        test_container, n_clusters=4, storage="assay", key_added="cluster_labels"
    )
    assert "cluster_kmeans" in result_both.assays
    assert "cluster_labels" in result_both.obs.columns
    assert result_both.obs["cluster_labels"].n_unique() <= 4
    print("  PASS")

    print("\n[Test 5] One-hot encoding structure (assay mode)")
    result = cluster_kmeans(test_container, n_clusters=5, storage="assay")
    one_hot_matrix = result.assays["cluster_kmeans"].layers["binary"].X
    assert one_hot_matrix.shape == (100, 5)
    assert np.allclose(one_hot_matrix.sum(axis=1), 1.0)
    print("  PASS")

    if NUMBA_AVAILABLE:
        print("\n[Test 6] Numba backend with storage='obs'")
        result_numba_obs = cluster_kmeans(
            test_container, n_clusters=3, backend="numba", storage="obs"
        )
        assert "kmeans_k3" in result_numba_obs.obs.columns
        assert result_numba_obs.obs["kmeans_k3"].n_unique() <= 3
        print("  PASS")

        print("\n[Test 7] Numba backend with storage='assay'")
        result_numba_assay = cluster_kmeans(
            test_container, n_clusters=3, backend="numba", storage="assay"
        )
        assert "cluster_kmeans" in result_numba_assay.assays
        assert result_numba_assay.assays["cluster_kmeans"].n_features == 3
        print("  PASS")

        print("\n[Test 8] Numba with custom parameters")
        result_custom = cluster_kmeans(
            test_container, n_clusters=3, backend="numba", max_iter=50, tol=1e-3, storage="obs"
        )
        assert result_custom.history[0].params["max_iter"] == 50
        assert result_custom.history[0].params["tol"] == 1e-3
        print("  PASS")
    else:
        print("\n[Test 6-8] Numba backend tests - SKIPPED (not installed)")

    print("\n[Test 9] Invalid storage parameter")
    try:
        cluster_kmeans(test_container, n_clusters=3, storage="invalid")  # type: ignore[arg-type]
        raise AssertionError("Should have raised ScpValueError")
    except ScpValueError as e:
        assert "storage" in str(e).lower()
        print("  PASS")

    print("\n[Test 10] Invalid n_clusters parameter")
    try:
        cluster_kmeans(test_container, n_clusters=0)
        raise AssertionError("Should have raised ScpValueError")
    except ScpValueError as e:
        assert "n_clusters" in str(e).lower()
        print("  PASS")

    print("\n[Test 11] Invalid backend parameter")
    try:
        cluster_kmeans(test_container, n_clusters=3, backend="invalid")  # type: ignore[arg-type]
        raise AssertionError("Should have raised ScpValueError")
    except ScpValueError as e:
        assert "backend" in str(e).lower()
        print("  PASS")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


# Deprecated alias for backward compatibility
def _deprecated_wrapper(old_name: str, new_name: str, func):
    """Create deprecated wrapper."""

    def wrapper(*args, **kwargs):
        warnings.warn(
            f"{old_name}() is deprecated. Use {new_name}() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    return wrapper


run_kmeans = _deprecated_wrapper("run_kmeans", "cluster_kmeans", cluster_kmeans)
