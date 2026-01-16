"""Basic clustering algorithms.

This module provides simple clustering implementations for single-cell
proteomics data, including K-Means clustering.
"""

from __future__ import annotations

import polars as pl
from sklearn.cluster import KMeans as SKLearnKMeans

from scptensor.core.exceptions import AssayNotFoundError, LayerNotFoundError, ScpValueError
from scptensor.core.structures import ScpContainer


def _validate_clustering_params(n_clusters: int) -> None:
    """Validate clustering parameters.

    Parameters
    ----------
    n_clusters : int
        Number of clusters must be positive.

    Raises
    ------
    ScpValueError
        If n_clusters is not positive.
    """
    if n_clusters <= 0:
        raise ScpValueError(
            f"n_clusters must be positive, got {n_clusters}.",
            parameter="n_clusters",
            value=n_clusters,
        )


def _get_input_matrix(
    container: ScpContainer,
    assay_name: str,
    base_layer: str,
) -> tuple[object, str, str]:
    """Extract input matrix from container with validation.

    Parameters
    ----------
    container : ScpContainer
        Input container.
    assay_name : str
        Name of the assay.
    base_layer : str
        Name of the layer.

    Returns
    -------
    tuple
        (X_matrix, assay_name, base_layer) - matrix and validated names.

    Raises
    ------
    AssayNotFoundError
        If assay does not exist.
    LayerNotFoundError
        If layer does not exist.
    """
    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]
    if base_layer not in assay.layers:
        raise LayerNotFoundError(base_layer, assay_name)

    return assay.layers[base_layer].X, assay_name, base_layer


def cluster_kmeans(
    container: ScpContainer,
    assay_name: str = "pca",
    base_layer: str = "X",
    n_clusters: int = 5,
    random_state: int = 42,
) -> ScpContainer:
    """K-Means clustering.

    Performs K-Means clustering on the specified assay layer and adds
    cluster labels to the container's observation metadata (obs).

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

    Returns
    -------
    ScpContainer
        New container with kmeans column added to obs.

    Raises
    ------
    AssayNotFoundError
        If the specified assay does not exist.
    LayerNotFoundError
        If the specified layer does not exist.
    ScpValueError
        If n_clusters is not positive.

    Examples
    --------
    >>> container = kmeans(container, n_clusters=3)
    >>> print(container.obs["kmeans_k3"])
    """
    _validate_clustering_params(n_clusters)
    X, _, _ = _get_input_matrix(container, assay_name, base_layer)

    model = SKLearnKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto",
    )
    labels = model.fit_predict(X)

    col_name = f"kmeans_k{n_clusters}"
    new_obs = container.obs.with_columns(pl.Series(col_name, labels).cast(pl.String))

    new_container = ScpContainer(
        obs=new_obs,
        assays=container.assays,
        history=list(container.history),
    )

    new_container.log_operation(
        action="cluster_kmeans",
        params={"assay": assay_name, "layer": base_layer, "n_clusters": n_clusters},
        description=f"K-Means clustering (k={n_clusters}) on {assay_name}/{base_layer}.",
    )

    return new_container


if __name__ == "__main__":
    import numpy as np

    from scptensor.core.structures import Assay, ScpContainer, ScpMatrix

    np.random.seed(42)
    X_test = np.random.randn(100, 10)

    obs_test = pl.DataFrame({"_index": [f"cell_{i}" for i in range(100)]})
    var_test = pl.DataFrame({"_index": [f"PC_{i}" for i in range(10)]})

    test_assay = Assay(
        var=var_test,
        layers={"X": ScpMatrix(X=X_test)},
    )

    test_container = ScpContainer(obs=obs_test, assays={"pca": test_assay})

    result = cluster_kmeans(test_container, n_clusters=3)

    assert "kmeans_k3" in result.obs.columns
    assert result.obs["kmeans_k3"].n_unique() <= 3
    assert len(result.history) == 1
    assert result.n_samples == test_container.n_samples

    print("All tests passed.")
