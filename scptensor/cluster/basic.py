"""Basic clustering algorithms.

Backend Options:
    - sklearn: Reliable, well-tested, O(n*k*i*d) complexity
    - numba: 2-5x faster on large datasets, same complexity (requires numba)
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl
from sklearn.cluster import KMeans as SKLearnKMeans

from scptensor.core.exceptions import AssayNotFoundError, LayerNotFoundError, ScpValueError
from scptensor.core.jit_ops import NUMBA_AVAILABLE, kmeans_core_numba, kmeans_plusplus_init_numba
from scptensor.core.structures import ScpContainer


def cluster_kmeans(
    container: ScpContainer,
    assay_name: str = "pca",
    base_layer: str = "X",
    n_clusters: int = 5,
    random_state: int = 42,
    backend: Literal["sklearn", "numba"] = "sklearn",
    max_iter: int = 300,
    tol: float = 1e-4,
) -> ScpContainer:
    """K-Means clustering with backend selection.

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

    Returns
    -------
    ScpContainer
        New container with kmeans column added to obs.

    Raises
    ------
    ScpValueError
        If n_clusters is not positive or backend is invalid.
    ImportError
        If backend='numba' but numba is not installed.

    Examples
    --------
    >>> container = cluster_kmeans(container, n_clusters=3, backend="sklearn")
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
        init_centers = kmeans_plusplus_init_numba(
            X, k=n_clusters, seed=random_state
        )
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
        },
        description=f"K-Means clustering (k={n_clusters}, backend={backend}) on {assay_name}/{base_layer}.",
    )
    return new_container


if __name__ == "__main__":
    import numpy as np
    from scptensor.core.structures import Assay, ScpContainer, ScpMatrix

    print("Testing cluster_kmeans with backend parameter support...")
    print(f"Numba available: {NUMBA_AVAILABLE}")

    np.random.seed(42)
    X_test = np.random.randn(100, 10)
    obs_test = pl.DataFrame({"_index": [f"cell_{i}" for i in range(100)]})
    var_test = pl.DataFrame({"_index": [f"PC_{i}" for i in range(10)]})
    test_assay = Assay(var=var_test, layers={"X": ScpMatrix(X=X_test)})
    test_container = ScpContainer(obs=obs_test, assays={"pca": test_assay})

    print("\n[Test 1] Sklearn backend (default)")
    result_sklearn = cluster_kmeans(test_container, n_clusters=3)
    assert "kmeans_k3" in result_sklearn.obs.columns
    assert result_sklearn.obs["kmeans_k3"].n_unique() <= 3
    assert len(result_sklearn.history) == 1
    assert result_sklearn.n_samples == test_container.n_samples
    print("  PASS")

    if NUMBA_AVAILABLE:
        print("\n[Test 2] Numba backend")
        result_numba = cluster_kmeans(test_container, n_clusters=3, backend="numba")
        assert "kmeans_k3" in result_numba.obs.columns
        assert result_numba.obs["kmeans_k3"].n_unique() <= 3
        assert len(result_numba.history) == 1
        assert result_numba.n_samples == test_container.n_samples
        print("  PASS")

        print("\n[Test 3] Numba with custom parameters")
        result_custom = cluster_kmeans(
            test_container, n_clusters=3, backend="numba", max_iter=50, tol=1e-3
        )
        assert result_custom.history[0].params["max_iter"] == 50
        assert result_custom.history[0].params["tol"] == 1e-3
        print("  PASS")
    else:
        print("\n[Test 2] Numba backend - SKIPPED (not installed)")

    print("\n[Test 4] Invalid backend parameter")
    try:
        cluster_kmeans(test_container, n_clusters=3, backend="invalid")  # type: ignore[arg-type]
        raise AssertionError("Should have raised ScpValueError")
    except ScpValueError as e:
        assert "backend" in str(e).lower()
        print("  PASS")

    print("\n[Test 5] Invalid n_clusters parameter")
    try:
        cluster_kmeans(test_container, n_clusters=0)
        raise AssertionError("Should have raised ScpValueError")
    except ScpValueError as e:
        assert "n_clusters" in str(e).lower()
        print("  PASS")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
