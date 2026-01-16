"""UMAP (Uniform Manifold Approximation and Projection) for dimensionality reduction.

This module provides UMAP embedding for single-cell proteomics data,
using the umap-learn library with ScpTensor integration.

Reference:
    McInnes, L., Healy, J., & Melville, J. (2018).
    UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.
    arXiv:1802.03426
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import umap as umap_learn

from scptensor.core.exceptions import (
    AssayNotFoundError,
    LayerNotFoundError,
    ScpValueError,
    ValidationError,
)
from scptensor.core.structures import Assay, ScpContainer, ScpMatrix

if TYPE_CHECKING:
    from numpy.typing import DTypeLike


def _validate_embedding_params(
    n_components: int,
    n_neighbors: int,
    min_dist: float,
) -> None:
    """Validate UMAP embedding parameters.

    Args:
        n_components: Number of UMAP dimensions
        n_neighbors: Size of local neighborhood
        min_dist: Minimum distance between embedded points

    Raises:
        ScpValueError: If any parameter is invalid
    """
    if n_components <= 0:
        raise ScpValueError(
            f"n_components must be positive, got {n_components}.",
            parameter="n_components",
            value=n_components,
        )
    if n_neighbors <= 0:
        raise ScpValueError(
            f"n_neighbors must be positive, got {n_neighbors}.",
            parameter="n_neighbors",
            value=n_neighbors,
        )
    if not (0.0 <= min_dist < 1.0):
        raise ScpValueError(
            f"min_dist must be in [0, 1), got {min_dist}.",
            parameter="min_dist",
            value=min_dist,
        )


def _check_valid_data(X: np.ndarray) -> None:
    """Check that input data has no NaN or infinite values.

    Args:
        X: Input data matrix

    Raises:
        ValidationError: If data contains NaN or Inf
    """
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValidationError(
            "Input data contains NaN or infinite values. "
            "ScpTensor UMAP requires a complete data matrix. "
            "Please use an imputed layer (e.g. run imputation first).",
            field="X",
        )


def reduce_umap(
    container: ScpContainer,
    assay_name: str,
    base_layer: str,
    new_assay_name: str = "umap",
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int | np.random.RandomState | None = 42,
    dtype: DTypeLike = np.float64,
) -> ScpContainer:
    """Perform UMAP dimensionality reduction on a specific assay layer.

    UMAP (Uniform Manifold Approximation and Projection) is a manifold learning
    technique that preserves both local and global structure in high-dimensional
    data. This function creates a new assay containing the UMAP embedding.

    Parameters
    ----------
    container : ScpContainer
        The data container containing the input assay.
    assay_name : str
        Name of the assay to transform.
    base_layer : str
        Name of the layer within the assay to use as input.
    new_assay_name : str, optional
        Name for the new assay storing UMAP results. Default is "umap".
    n_components : int, optional
        Number of UMAP dimensions to compute. Default is 2.
    n_neighbors : int, optional
        Size of the local neighborhood for UMAP. Larger values preserve more
        global structure. Default is 15.
    min_dist : float, optional
        Effective minimum distance between embedded points. Smaller values
        create tighter clusters. Default is 0.1.
    metric : str, optional
        Distance metric to use. Default is "euclidean".
    random_state : int or RandomState or None, optional
        Random seed for reproducibility. Default is 42.
    dtype : type or dtype, optional
        Data type for computation. Default is np.float64.

    Returns
    -------
    ScpContainer
        A new container with the UMAP results added as a new assay.

    Raises
    ------
    AssayNotFoundError
        If the specified assay does not exist.
    LayerNotFoundError
        If the specified layer does not exist in the assay.
    ScpValueError
        If n_neighbors, min_dist, or n_components parameters are invalid.
    ValidationError
        If the input data contains NaN or infinite values.

    Examples
    --------
    >>> from scptensor import create_test_container
    >>> from scptensor.dim_reduction import reduce_umap
    >>> container = create_test_container()
    >>> result = reduce_umap(container, "proteins", "imputed", n_components=2)
    >>> result = reduce_umap(
    ...     container, "proteins", "imputed",
    ...     n_neighbors=30, min_dist=0.2
    ... )
    """
    # Validate parameters early
    _validate_embedding_params(n_components, n_neighbors, min_dist)

    # Validate assay and layer exist
    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]
    if base_layer not in assay.layers:
        raise LayerNotFoundError(base_layer, assay_name)

    # Extract input data
    input_matrix = assay.layers[base_layer]
    X = input_matrix.X

    # Validate data completeness
    _check_valid_data(X)

    # Configure and fit UMAP
    reducer = umap_learn.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )

    embedding = reducer.fit_transform(X)

    if np.dtype(dtype) != embedding.dtype:
        embedding = embedding.astype(dtype)

    # Create feature metadata for UMAP dimensions
    feature_names = [f"UMAP_{i + 1}" for i in range(n_components)]
    var_df = pl.DataFrame({"feature_id": feature_names})

    # Create new assay with embedding layer
    new_assay = Assay(var=var_df, feature_id_col="feature_id")

    # UMAP embeddings are always valid (no missingness in reduced space)
    M = np.zeros(embedding.shape, dtype=np.int8)
    new_assay.add_layer(name="embedding", matrix=ScpMatrix(X=embedding, M=M))

    # Create new container with updated assays
    new_container = container.copy()
    new_container.add_assay(new_assay_name, new_assay)

    # Log operation for provenance tracking
    random_state_str = str(random_state) if isinstance(random_state, int) else "RandomState"
    new_container.log_operation(
        action="reduce_umap",
        params={
            "assay_name": assay_name,
            "base_layer": base_layer,
            "new_assay_name": new_assay_name,
            "n_components": n_components,
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "metric": metric,
            "random_state": random_state_str,
            "dtype": str(dtype),
        },
        description=f"Performed UMAP on {assay_name}/{base_layer}",
    )

    return new_container


__all__ = ["reduce_umap"]
