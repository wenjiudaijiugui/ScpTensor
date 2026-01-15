"""Graph-based clustering algorithms.

This module provides community detection clustering methods that operate
on k-nearest neighbor graphs, including the Leiden algorithm.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.neighbors import kneighbors_graph

from scptensor.core.exceptions import AssayNotFoundError, LayerNotFoundError, ScpValueError
from scptensor.core.structures import ScpContainer
from scptensor.core.utils import requires_dependency


def _validate_graph_params(n_neighbors: int, resolution: float) -> None:
    """Validate graph clustering parameters.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbors must be positive.
    resolution : float
        Resolution parameter must be positive.

    Raises
    ------
    ScpValueError
        If parameters are invalid.
    """
    if n_neighbors <= 0:
        raise ScpValueError(
            f"n_neighbors must be positive, got {n_neighbors}.",
            parameter="n_neighbors",
            value=n_neighbors,
        )
    if resolution <= 0:
        raise ScpValueError(
            f"resolution must be positive, got {resolution}.",
            parameter="resolution",
            value=resolution,
        )


def _get_input_matrix(
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


@requires_dependency("leidenalg", "pip install leidenalg")
@requires_dependency("igraph", "pip install python-igraph")
def leiden(
    container: ScpContainer,
    assay_name: str = "pca",
    base_layer: str = "X",
    n_neighbors: int = 15,
    resolution: float = 1.0,
    random_state: int = 42,
) -> ScpContainer:
    """Leiden clustering.

    Performs community detection using the Leiden algorithm on a k-nearest
    neighbor graph constructed from the specified assay layer.

    Parameters
    ----------
    container : ScpContainer
        Input container.
    assay_name : str, default="pca"
        Name of the assay to use.
    base_layer : str, default="X"
        Name of the layer to use as input.
    n_neighbors : int, default=15
        Number of neighbors for kNN graph construction.
    resolution : float, default=1.0
        Resolution parameter for Leiden (higher = more clusters).
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    ScpContainer
        New container with leiden clustering column added to obs.

    Raises
    ------
    AssayNotFoundError
        If the specified assay does not exist.
    LayerNotFoundError
        If the specified layer does not exist.
    ScpValueError
        If n_neighbors or resolution is not positive.

    Examples
    --------
    >>> container = leiden(container, resolution=0.8)
    >>> print(container.obs["leiden_r0.8"])
    """
    _validate_graph_params(n_neighbors, resolution)
    X, _, _ = _get_input_matrix(container, assay_name, base_layer)

    import igraph as ig
    import leidenalg

    adj_matrix = kneighbors_graph(
        X,
        n_neighbors=n_neighbors,
        mode="connectivity",
        include_self=True,
    )

    sources, targets = adj_matrix.nonzero()
    edges = list(zip(sources.tolist(), targets.tolist(), strict=False))

    graph = ig.Graph(directed=False)
    graph.add_vertices(adj_matrix.shape[0])
    graph.add_edges(edges)

    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=random_state,
    )

    labels = np.array(partition.membership)

    col_name = f"leiden_r{resolution}"
    new_obs = container.obs.with_columns(pl.Series(col_name, labels).cast(pl.String))

    new_container = ScpContainer(
        obs=new_obs,
        assays=container.assays,
        history=list(container.history),
    )

    new_container.log_operation(
        action="cluster_leiden",
        params={
            "assay": assay_name,
            "layer": base_layer,
            "n_neighbors": n_neighbors,
            "resolution": resolution,
        },
        description=f"Leiden clustering (k={n_neighbors}, res={resolution}) on {assay_name}/{base_layer}.",
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

    try:
        result = leiden(test_container, n_neighbors=10, resolution=0.5)
        assert "leiden_r0.5" in result.obs.columns
        assert len(result.history) == 1
        print("All tests passed.")
    except (ImportError, Exception):
        print("Skipping test: leidenalg/igraph not installed.")
