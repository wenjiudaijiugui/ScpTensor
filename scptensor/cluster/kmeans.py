"""K-Means clustering with assay storage.

This module provides K-Means clustering that stores results as a new assay
containing one-hot encoded cluster assignments, following the project's
immutable pattern design.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.cluster import KMeans as SKLearnKMeans

from scptensor.core.exceptions import AssayNotFoundError
from scptensor.core.exceptions import LayerNotFoundError
from scptensor.core.exceptions import ScpValueError
from scptensor.core.structures import Assay
from scptensor.core.structures import ScpContainer
from scptensor.core.structures import ScpMatrix


def _validate_params(n_clusters: int) -> None:
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


def _get_input_data(
    container: ScpContainer,
    assay_name: str,
    base_layer: str,
) -> tuple[np.ndarray, str, str]:
    """Extract and validate input matrix from container.

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
        (X, assay_name, base_layer) - data matrix and validated names.

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


def _create_one_hot_encoding(
    labels: np.ndarray,
    n_clusters: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create one-hot encoding from cluster labels.

    Parameters
    ----------
    labels : ndarray
        Cluster labels for each sample.
    n_clusters : int
        Total number of clusters.

    Returns
    -------
    tuple
        (one_hot, mask) - one-hot encoded matrix and zero mask.
    """
    n_samples = len(labels)
    one_hot = np.zeros((n_samples, n_clusters), dtype=np.float32)
    one_hot[np.arange(n_samples), labels] = 1.0

    mask = np.zeros_like(one_hot, dtype=np.int8)
    return one_hot, mask


def run_kmeans(
    container: ScpContainer,
    assay_name: str = "pca",
    base_layer: str = "X",
    new_assay_name: str = "cluster_kmeans",
    n_clusters: int = 5,
    key_added: str | None = None,
    random_state: int = 42,
) -> ScpContainer:
    """Run K-Means clustering and store results as a new assay.

    This function performs K-Means clustering on the specified assay layer
    and creates a new assay containing one-hot encoded cluster assignments.
    Optionally adds cluster labels to the observation metadata.

    Parameters
    ----------
    container : ScpContainer
        Input container.
    assay_name : str, default="pca"
        Input assay name (typically the PCA assay).
    base_layer : str, default="X"
        Input layer name.
    new_assay_name : str, default="cluster_kmeans"
        Name for the new clustering assay.
    n_clusters : int, default=5
        Number of clusters (k).
    key_added : str | None, default=None
        If provided, adds cluster labels to obs with this key.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    ScpContainer
        New container with clustering assay added (and obs updated if key_added).

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
    >>> container = run_kmeans(container, n_clusters=5, key_added="kmeans")
    >>> "cluster_kmeans" in container.assays
    True
    """
    _validate_params(n_clusters)
    X, _, _ = _get_input_data(container, assay_name, base_layer)

    kmeans = SKLearnKMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    kmeans.fit(X)
    labels = kmeans.labels_

    one_hot, M_cluster = _create_one_hot_encoding(labels, n_clusters)

    var_cluster = pl.DataFrame({
        "_index": [f"Cluster_{i}" for i in range(n_clusters)],
        "cluster_id": [f"Cluster_{i}" for i in range(n_clusters)],
    })

    matrix_cluster = ScpMatrix(X=one_hot, M=M_cluster)

    assay_cluster = Assay(
        var=var_cluster,
        layers={"binary": matrix_cluster},
        feature_id_col="cluster_id",
    )

    new_assays = dict(container.assays)
    new_assays[new_assay_name] = assay_cluster

    new_obs = container.obs
    if key_added:
        new_obs = container.obs.with_columns(
            pl.Series(key_added, labels).cast(pl.String)
        )

    new_container = ScpContainer(
        obs=new_obs,
        assays=new_assays,
        history=list(container.history),
    )

    new_container.log_operation(
        action="run_kmeans",
        params={
            "source_assay": assay_name,
            "source_layer": base_layer,
            "n_clusters": n_clusters,
            "key_added": key_added,
        },
        description=f"K-Means (k={n_clusters}) on {assay_name}/{base_layer}.",
    )

    return new_container


if __name__ == "__main__":
    import numpy as np

    from scptensor.core.structures import ScpContainer, Assay, ScpMatrix

    np.random.seed(42)
    X_test = np.random.randn(100, 10)

    obs_test = pl.DataFrame({"_index": [f"cell_{i}" for i in range(100)]})
    var_test = pl.DataFrame({"_index": [f"PC_{i}" for i in range(10)]})

    test_assay = Assay(
        var=var_test,
        layers={"X": ScpMatrix(X=X_test)},
    )

    test_container = ScpContainer(obs=obs_test, assays={"pca": test_assay})

    result = run_kmeans(test_container, n_clusters=3, key_added="kmeans")

    assert "cluster_kmeans" in result.assays
    assert "kmeans" in result.obs.columns
    assert result.assays["cluster_kmeans"].n_features == 3
    assert len(result.history) == 1

    print("All tests passed.")
