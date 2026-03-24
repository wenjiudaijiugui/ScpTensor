"""UMAP (Uniform Manifold Approximation and Projection) for dimensionality reduction.

This module provides UMAP embedding for DIA-based single-cell proteomics data,
aligned with scanpy's tl.umap API.

Reference:
    McInnes, L., Healy, J., & Melville, J. (2018).
    UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.
    arXiv:1802.03426
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from scptensor.core._structure_assay import Assay
from scptensor.core._structure_container import ScpContainer
from scptensor.core._structure_matrix import ScpMatrix
from scptensor.core.assay_alias import resolve_assay_name
from scptensor.core.exceptions import MissingDependencyError
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
    if n_neighbors <= 1:
        raise ValueError(f"n_neighbors must be > 1, got {n_neighbors}")
    if not (0.0 <= min_dist <= 1.0):
        raise ValueError(f"min_dist must be in [0, 1], got {min_dist}")

    # Validate assay and layer
    resolved_assay_name = resolve_assay_name(container, assay_name)
    _, X = _validate_assay_layer(container, resolved_assay_name, base_layer)

    # Check data completeness
    _check_no_nan_inf(X)

    # Prepare data
    X_dense = _prepare_matrix(X, dtype=np.dtype(dtype))
    n_samples = X_dense.shape[0]
    if n_samples < 2:
        raise ValueError(f"UMAP requires at least 2 samples, got {n_samples}")
    if n_neighbors >= n_samples:
        raise ValueError(
            f"n_neighbors ({n_neighbors}) must be < n_samples ({n_samples}) to avoid truncation.",
        )

    # Fit UMAP.
    # Some umap-learn builds emit ImportWarning for optional ParametricUMAP/TensorFlow.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ImportWarning, module=r"umap(\.|$)")
        try:
            import umap as umap_learn
        except ImportError as exc:
            raise MissingDependencyError("umap-learn") from exc

    umap_kwargs: dict[str, object] = {
        "n_neighbors": n_neighbors,
        "n_components": n_components,
        "min_dist": min_dist,
        "metric": metric,
        "random_state": random_state,
    }
    if random_state is not None:
        # Avoid UMAP warning about forced single-threading when a seed is set.
        umap_kwargs["n_jobs"] = 1

    reducer = umap_learn.UMAP(
        **umap_kwargs,
    )

    embedding = reducer.fit_transform(X_dense)

    # Create feature metadata
    feature_names = [f"UMAP_{i + 1}" for i in range(n_components)]
    var_df = pl.DataFrame({"feature_id": feature_names})

    # Create assay
    M = np.zeros(embedding.shape, dtype=np.int8)
    matrix = ScpMatrix(X=embedding, M=M)

    new_assay = Assay(var=var_df, layers={"X": matrix}, feature_id_col="feature_id")

    # Create new container (shallow structure copy, no assay deep-copy)
    new_assays = {**container.assays, new_assay_name: new_assay}
    new_container = ScpContainer(
        obs=container.obs,
        assays=new_assays,
        history=list(container.history),
        sample_id_col=container.sample_id_col,
    )

    # Log operation
    new_container.log_operation(
        action="reduce_umap",
        params={
            "source_assay": resolved_assay_name,
            "source_layer": base_layer,
            "target_assay": new_assay_name,
            "n_components": n_components,
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "metric": metric,
        },
        description=f"UMAP on {resolved_assay_name}/{base_layer} (n_neighbors={n_neighbors}).",
    )

    return new_container


__all__ = ["reduce_umap"]
