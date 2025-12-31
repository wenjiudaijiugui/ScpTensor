import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from scptensor.core.structures import ScpContainer, Assay, ScpMatrix

def run_kmeans(
    container: ScpContainer,
    assay_name: str = "pca",
    base_layer: str = "X",
    new_assay_name: str = "cluster_kmeans",
    n_clusters: int = 5,
    key_added: str = None,
    random_state: int = 42
) -> ScpContainer:
    """
    Run K-Means clustering.
    
    Args:
        container: ScpContainer object.
        assay_name: Input assay name (usually 'pca').
        base_layer: Input layer name.
        new_assay_name: Name for the new Clustering assay.
        n_clusters: Number of clusters (k).
        key_added: If provided, adds cluster labels to obs with this key.
        random_state: Seed.
        
    Returns:
        The modified ScpContainer with a new Assay containing cluster probabilities/one-hot.
    """
    if assay_name not in container.assays:
        raise ValueError(f"Assay '{assay_name}' not found.")
    
    assay = container.assays[assay_name]
    input_matrix = assay.layers[base_layer]
    X = input_matrix.X
    
    # Run K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(X)
    
    # We can store:
    # 1. Hard labels (in obs)
    # 2. Distance/Probability (in a new Assay layer)
    
    # Store labels in obs
    labels = kmeans.labels_
    
    if key_added:
        # Update obs using Polars
        container.obs = container.obs.with_columns(
            pl.Series(name=key_added, values=labels).cast(pl.Utf8)
        )
    # Update obs using Polars
    # Note: container.obs is a Polars DataFrame. We need to append a column.
    # Since container.obs is immutable-ish in design (or we replace it), 
    # we should create a new DataFrame.
    
    # However, Python objects are mutable. We can try to modify it or replace it.
    # The design says "Immutable Flow" for *Filter/Subset*. 
    # For adding columns, it might be allowed, but let's stick to returning new container if we modify obs strictly,
    # OR we follow the "Project" pattern: "Generate a new Assay".
    
    # The design document says:
    # | 细胞注释 | assays['prediction'] | Index: T-Cell... | probability (概率), binary (Mask) |
    
    # So we should create a new Assay for clusters.
    
    # 1. Create One-Hot encoding or Probability (distance-based)
    # Let's use One-Hot for hard clustering
    one_hot = np.zeros((X.shape[0], n_clusters), dtype=np.float32)
    one_hot[np.arange(X.shape[0]), labels] = 1.0
    
    # 2. Create var
    var_cluster = pl.DataFrame({
        "cluster_id": [f"Cluster_{i}" for i in range(n_clusters)]
    })
    
    # 3. Create Matrix
    M_cluster = np.zeros_like(one_hot, dtype=np.int8)
    
    matrix_cluster = ScpMatrix(X=one_hot, M=M_cluster)
    
    assay_cluster = Assay(
        var=var_cluster, 
        layers={
            "binary": matrix_cluster,
        },
        feature_id_col="cluster_id"
    )
    
    container.add_assay(new_assay_name, assay_cluster)
    
    # Optional: Also add to obs for convenience?
    # The design emphasizes Assay storage. Let's stick to Assay for now.
    
    container.log_operation(
        action="run_kmeans",
        params={
            "source_assay": assay_name,
            "source_layer": base_layer,
            "n_clusters": n_clusters
        },
        description=f"K-Means (k={n_clusters}) clustering performed on {assay_name}/{base_layer}."
    )
    
    return container
