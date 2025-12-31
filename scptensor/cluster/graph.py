from typing import Optional
import polars as pl
from sklearn.neighbors import kneighbors_graph
from scptensor.core.structures import ScpContainer
from scptensor.core.utils import requires_dependency

@requires_dependency('leidenalg', 'pip install leidenalg')
@requires_dependency('igraph', 'pip install python-igraph')
def leiden(
    container: ScpContainer,
    assay_name: str = 'pca',
    base_layer: str = 'X',
    n_neighbors: int = 15,
    resolution: float = 1.0,
    random_state: int = 42
) -> ScpContainer:
    """
    Leiden clustering.
    """
    import leidenalg
    import igraph as ig
    
    if assay_name not in container.assays:
        raise ValueError(f"Assay '{assay_name}' not found.")
    
    assay = container.assays[assay_name]
    if base_layer not in assay.layers:
         raise ValueError(f"Layer '{base_layer}' not found in assay '{assay_name}'.")

    X = assay.layers[base_layer].X
    
    # 1. Construct kNN graph
    # Returns sparse matrix
    adj_matrix = kneighbors_graph(X, n_neighbors=n_neighbors, mode='connectivity', include_self=True)
    
    # 2. Convert to igraph
    sources, targets = adj_matrix.nonzero()
    g = ig.Graph(directed=False)
    g.add_vertices(adj_matrix.shape[0])
    g.add_edges(list(zip(sources, targets)))
    
    # 3. Run Leiden
    partition = leidenalg.find_partition(
        g, 
        leidenalg.RBConfigurationVertexPartition, 
        resolution_parameter=resolution,
        seed=random_state
    )
    
    labels = np.array(partition.membership)
    
    # Add to obs
    new_obs = container.obs.with_columns(
        pl.Series(f"leiden_r{resolution}", labels).cast(pl.Utf8)
    )
    
    new_container = ScpContainer(
        obs=new_obs,
        assays=container.assays,
        history=list(container.history)
    )
    
    new_container.log_operation(
        action="cluster_leiden",
        params={"n_neighbors": n_neighbors, "resolution": resolution},
        description=f"Leiden clustering (res={resolution})."
    )
    
    return new_container
