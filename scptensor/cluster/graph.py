"""Graph-based clustering for DIA-based single-cell proteomics data.

This module provides Leiden clustering, aligned with scanpy's tl.leiden API.

Reference:
    Traag, V. A., Waltman, L., & van Eck, N. J. (2019).
    From Louvain to Leiden: guaranteeing well-connected communities.
    Scientific Reports, 9(1), 5233.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import kneighbors_graph

from scptensor.cluster.base import (
    _add_labels_to_obs,
    _get_default_key,
    _prepare_matrix,
    _validate_assay_layer,
)
from scptensor.core.assay_alias import resolve_assay_name
from scptensor.core.exceptions import ScpValueError
from scptensor.core.structures import ScpContainer


def cluster_leiden(
    container: ScpContainer,
    assay_name: str = "reduce_pca",
    base_layer: str = "X",
    n_neighbors: int = 15,
    resolution: float = 1.0,
    random_state: int = 42,
    key_added: str | None = None,
) -> ScpContainer:
    """Leiden clustering.

    Performs community detection using the Leiden algorithm on a
    k-nearest neighbor graph.

    Parameters
    ----------
    container : ScpContainer
        Input container.
    assay_name : str, default="reduce_pca"
        Name of assay to use.
    base_layer : str, default="X"
        Name of layer to use.
    n_neighbors : int, default=15
        Number of neighbors for kNN graph.
    resolution : float, default=1.0
        Resolution parameter (higher = more clusters).
    random_state : int, default=42
        Random seed.
    key_added : str or None, optional
        Column name for results. If None, auto-generated.

    Returns
    -------
    ScpContainer
        Container with clustering results in obs.

    Raises
    ------
    ScpValueError
        If n_neighbors or resolution invalid.
    ImportError
        If leidenalg/igraph not installed.

    Examples
    --------
    >>> from scptensor import create_test_container
    >>> from scptensor.cluster import cluster_leiden
    >>> container = create_test_container()
    >>> result = cluster_leiden(container, resolution=0.8)
    >>> "leiden_r0.8" in result.obs.columns
    True
    """
    # Validate parameters
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

    # Get and prepare data
    resolved_assay_name = resolve_assay_name(container, assay_name)
    _, X = _validate_assay_layer(container, resolved_assay_name, base_layer)
    X_dense = _prepare_matrix(X)
    effective_n_neighbors = min(n_neighbors, X_dense.shape[0])

    # Build kNN graph
    adj_matrix = kneighbors_graph(
        X_dense,
        n_neighbors=effective_n_neighbors,
        mode="connectivity",
        include_self=True,
    )

    # Run Leiden
    import igraph as ig
    import leidenalg

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

    # Generate key
    if key_added is None:
        key_added = _get_default_key("leiden", {"resolution": resolution})

    # Add to obs
    new_container = _add_labels_to_obs(container, labels, key_added)

    # Log operation
    new_container.log_operation(
        action="cluster_leiden",
        params={
            "source_assay": resolved_assay_name,
            "source_layer": base_layer,
            "output_key": key_added,
            "n_neighbors": effective_n_neighbors,
            "resolution": resolution,
        },
        description=(
            f"Leiden (k={effective_n_neighbors}, res={resolution}) on "
            f"{resolved_assay_name}/{base_layer}."
        ),
    )

    return new_container


__all__ = ["cluster_leiden"]
