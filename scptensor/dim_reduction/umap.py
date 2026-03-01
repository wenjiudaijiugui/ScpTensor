"""UMAP (Uniform Manifold Approximation and Projection) for dimensionality reduction.

This module provides UMAP embedding for single-cell proteomics data,
aligned with scanpy's tl.umap API.

Reference:
    McInnes, L., Healy, J., & Melville, J. (2018).
    UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.
    arXiv:1802.03426
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from scptensor.core.structures import Assay, ScpContainer, ScpMatrix
from scptensor.dim_reduction.base import (
    _check_no_nan_inf,
    _prepare_matrix,
    _validate_assay_layer,
)

if TYPE_CHECKING:
    from numpy.typing import DTypeLike


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
    """Perform UMAP dimensionality reduction.

    UMAP is a manifold learning technique that preserves both local
    and global structure in high-dimensional data.

    Parameters
    ----------
    container : ScpContainer
        The data container.
    assay_name : str
        Name of the assay to transform.
    base_layer : str
        Name of the layer within the assay.
    new_assay_name : str, optional
        Name for the new assay. Default is "umap".
    n_components : int, optional
        Number of UMAP dimensions. Default is 2.
    n_neighbors : int, optional
        Size of local neighborhood. Default is 15.
    min_dist : float, optional
        Minimum distance between points. Default is 0.1.
    metric : str, optional
        Distance metric. Default is "euclidean".
    random_state : int or RandomState or None, optional
        Random seed. Default is 42.
    dtype : dtype or type, optional
        Data type. Default is np.float64.

    Returns
    -------
    ScpContainer
        Container with UMAP results.

    Raises
    ------
    AssayNotFoundError
        If assay does not exist.
    LayerNotFoundError
        If layer does not exist.
    ValueError
        If parameters invalid or data has NaN/Inf.

    Examples
    --------
    >>> from scptensor import create_test_container
    >>> from scptensor.dim_reduction import reduce_umap
    >>> container = create_test_container()
    >>> result = reduce_umap(container, "proteins", "imputed")

    Notes
    -----
    Input data should be imputed (no missing values) for best results.
    """
    # Validate parameters
    if n_components <= 0:
        raise ValueError(f"n_components must be positive, got {n_components}")
    if n_neighbors <= 0:
        raise ValueError(f"n_neighbors must be positive, got {n_neighbors}")
    if not (0.0 <= min_dist < 1.0):
        raise ValueError(f"min_dist must be in [0, 1), got {min_dist}")

    # Validate assay and layer
    _validate_assay_layer(container, assay_name, base_layer)

    assay = container.assays[assay_name]
    X = assay.layers[base_layer].X

    # Check data completeness
    _check_no_nan_inf(X)

    # Prepare data
    X_dense = _prepare_matrix(X, dtype=np.dtype(dtype))

    # Fit UMAP
    import umap as umap_learn

    reducer = umap_learn.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )

    embedding = reducer.fit_transform(X_dense)

    # Create feature metadata
    feature_names = [f"UMAP_{i + 1}" for i in range(n_components)]
    var_df = pl.DataFrame({"feature_id": feature_names})

    # Create assay
    M = np.zeros(embedding.shape, dtype=np.int8)
    matrix = ScpMatrix(X=embedding, M=M)

    new_assay = Assay(var=var_df, layers={"X": matrix}, feature_id_col="feature_id")

    # Create new container
    new_container = container.copy()
    new_container.add_assay(new_assay_name, new_assay)

    # Log operation
    new_container.log_operation(
        action="reduce_umap",
        params={
            "assay_name": assay_name,
            "base_layer": base_layer,
            "n_components": n_components,
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "metric": metric,
        },
        description=f"UMAP on {assay_name}/{base_layer} (n_neighbors={n_neighbors}).",
    )

    return new_container


__all__ = ["reduce_umap"]
