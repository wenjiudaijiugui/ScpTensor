
from typing import Optional, Union, Type, List, Dict
import numpy as np
import polars as pl
import umap as umap_learn
import scipy.sparse as sp
from scptensor.core.structures import ScpContainer, ScpMatrix, Assay

def umap_transform(
    container: ScpContainer,
    assay_name: str,
    base_layer_name: str,
    new_assay_name: str = "umap",
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: Optional[Union[int, np.random.RandomState]] = 42,
    dtype: Type = np.float64
) -> ScpContainer:
    """
    Perform UMAP on a specific layer of an assay.

    Args:
        container (ScpContainer): The data container.
        assay_name (str): The name of the assay to use.
        base_layer_name (str): The name of the layer in the assay to use as input.
        new_assay_name (str, optional): The name of the new assay to store UMAP results. Defaults to "umap".
        n_components (int, optional): The number of components to compute. Defaults to 2.
        n_neighbors (int, optional): The size of local neighborhood. Defaults to 15.
        min_dist (float, optional): The effective minimum distance between embedded points. Defaults to 0.1.
        metric (str, optional): The metric to use to calculate distance between examples in a feature array. Defaults to "euclidean".
        random_state (int, RandomState, optional): Seed for the random number generator. Defaults to 42.
        dtype (Type, optional): The desired data type for computation. Defaults to np.float64.

    Returns:
        ScpContainer: A new container with the UMAP results added as a new assay.
    """
    if assay_name not in container.assays:
        raise ValueError(f"Assay '{assay_name}' not found.")

    assay = container.assays[assay_name]
    if base_layer_name not in assay.layers:
        raise ValueError(f"Layer '{base_layer_name}' not found in assay '{assay_name}'.")

    input_matrix = assay.layers[base_layer_name]
    X = input_matrix.X

    # Check for NaN or infinite values
    if sp.issparse(X):
        if np.any(np.isnan(X.data)) or np.any(np.isinf(X.data)):
            raise ValueError(
                "Input data contains NaN or infinite values. "
                "ScpTensor UMAP requires a complete data matrix. "
                "Please use an imputed layer (e.g. run imputation first)."
            )
    else:
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError(
                "Input data contains NaN or infinite values. "
                "ScpTensor UMAP requires a complete data matrix. "
                "Please use an imputed layer (e.g. run imputation first)."
            )

    reducer = umap_learn.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state
    )

    embedding = reducer.fit_transform(X)
    
    if dtype != embedding.dtype:
        embedding = embedding.astype(dtype)

    # Create feature metadata for UMAP dimensions
    feature_names = [f"UMAP_{i+1}" for i in range(n_components)]
    var_df = pl.DataFrame({"feature_id": feature_names})

    # Create a new Assay for UMAP results
    new_assay = Assay(
        var=var_df,
        feature_id_col="feature_id"
    )
    
    # Add the embedding as a layer
    # UMAP doesn't change cells, so M should match input cells
    # However, M usually tracks missing values in original features.
    # In UMAP space, the concept of missingness per "UMAP dimension" is different.
    # Typically, if a sample is successfully embedded, it's valid.
    # So we can create a generic "Valid" mask (all zeros)
    
    M = np.zeros(embedding.shape, dtype=np.int8)

    new_assay.add_layer(
        name="embedding",
        matrix=ScpMatrix(X=embedding, M=M)
    )
    
    # Add the new assay to the container
    new_container = container.copy()
    new_container.add_assay(new_assay_name, new_assay)
    
    # Log the operation
    params = {
        "assay_name": assay_name,
        "base_layer_name": base_layer_name,
        "new_assay_name": new_assay_name,
        "n_components": n_components,
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "metric": metric,
        "random_state": random_state if isinstance(random_state, int) else "RandomState",
        "dtype": str(dtype)
    }
    new_container.log_operation(
        action="umap_transform",
        params=params,
        description=f"Performed UMAP on {assay_name}/{base_layer_name}"
    )
    
    return new_container

def umap(
    container: ScpContainer,
    assay_name: str,
    base_layer: str,
    new_assay_name: str = "umap",
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: Optional[Union[int, np.random.RandomState]] = 42,
    dtype: Type = np.float64
) -> ScpContainer:
    return umap_transform(
        container=container,
        assay_name=assay_name,
        base_layer_name=base_layer,
        new_assay_name=new_assay_name,
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        dtype=dtype
    )
