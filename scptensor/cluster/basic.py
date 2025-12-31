from typing import Optional
import polars as pl
from sklearn.cluster import KMeans
from scptensor.core.structures import ScpContainer

def kmeans(
    container: ScpContainer,
    assay_name: str = 'pca',
    base_layer: str = 'X',
    n_clusters: int = 5,
    random_state: int = 42
) -> ScpContainer:
    """
    K-Means clustering.
    Adds 'kmeans' column to obs.
    """
    if assay_name not in container.assays:
        raise ValueError(f"Assay '{assay_name}' not found.")
    
    assay = container.assays[assay_name]
    if base_layer not in assay.layers:
         raise ValueError(f"Layer '{base_layer}' not found in assay '{assay_name}'.")

    X = assay.layers[base_layer].X
    
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    
    # Add to obs
    # Cast to str to be categorical-like
    new_obs = container.obs.with_columns(
        pl.Series(f"kmeans_k{n_clusters}", labels).cast(pl.Utf8)
    )
    
    new_container = ScpContainer(
        obs=new_obs,
        assays=container.assays,
        history=list(container.history)
    )
    
    new_container.log_operation(
        action="cluster_kmeans",
        params={"n_clusters": n_clusters},
        description=f"KMeans clustering (k={n_clusters})."
    )
    
    return new_container
