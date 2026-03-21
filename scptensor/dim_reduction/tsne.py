"""t-SNE (t-distributed Stochastic Neighbor Embedding) for dimensionality reduction.

This module provides t-SNE embedding for DIA-based single-cell proteomics data,
aligned with scanpy's tl.tsne API.

Reference:
    van der Maaten, L., & Hinton, G. (2008).
    Visualizing Data using t-SNE.
    Journal of Machine Learning Research, 9(Nov), 2579-2605.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl

from scptensor.core.assay_alias import resolve_assay_name
from scptensor.core.structures import Assay, ScpContainer, ScpMatrix
from scptensor.dim_reduction.base import (
    _check_no_nan_inf,
    _prepare_matrix,
    _validate_assay_layer,
)

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

LearningRateType = float | Literal["auto"]
InitType = Literal["pca", "random"]
MethodType = Literal["barnes_hut", "exact"]


def reduce_tsne(
    container: ScpContainer,
    assay_name: str,
    base_layer: str,
    new_assay_name: str = "tsne",
    n_components: int = 2,
    perplexity: float = 30.0,
    early_exaggeration: float = 12.0,
    learning_rate: LearningRateType = "auto",
    metric: str = "euclidean",
    init: InitType = "pca",
    method: MethodType = "barnes_hut",
    max_iter: int = 1000,
    random_state: int | np.random.RandomState | None = 42,
    dtype: DTypeLike = np.float64,
) -> ScpContainer:
    """Perform t-SNE dimensionality reduction.

    Parameters
    ----------
    container : ScpContainer
        The data container.
    assay_name : str
        Name of the assay to transform.
    base_layer : str
        Name of the layer within the assay.
    new_assay_name : str, optional
        Name for the new assay. Default is "tsne".
    n_components : int, optional
        Number of embedding dimensions. Default is 2.
    perplexity : float, optional
        Effective neighborhood size. Must be < n_samples. Default is 30.0.
    early_exaggeration : float, optional
        Controls tightness of clusters in early optimization. Default is 12.0.
    learning_rate : float or {"auto"}, optional
        Learning rate for optimization. Default is "auto".
    metric : str, optional
        Distance metric. Default is "euclidean".
    init : {"pca", "random"}, optional
        Embedding initialization method. Default is "pca".
    method : {"barnes_hut", "exact"}, optional
        Optimization method. Default is "barnes_hut".
    max_iter : int, optional
        Maximum number of optimization iterations. Must be >= 250. Default is 1000.
    random_state : int or RandomState or None, optional
        Random seed. Default is 42.
    dtype : dtype or type, optional
        Data type. Default is np.float64.

    Returns
    -------
    ScpContainer
        Container with t-SNE results.

    Raises
    ------
    AssayNotFoundError
        If assay does not exist.
    LayerNotFoundError
        If layer does not exist.
    ValueError
        If parameters are invalid or data has NaN/Inf.
    """
    valid_init = {"pca", "random"}
    valid_method = {"barnes_hut", "exact"}

    if n_components <= 0:
        raise ValueError(f"n_components must be positive, got {n_components}")
    if perplexity <= 0:
        raise ValueError(f"perplexity must be positive, got {perplexity}")
    if early_exaggeration <= 0:
        raise ValueError(f"early_exaggeration must be positive, got {early_exaggeration}")
    if max_iter < 250:
        raise ValueError(f"max_iter must be >= 250, got {max_iter}")
    if learning_rate != "auto" and learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive or 'auto', got {learning_rate}")
    if init not in valid_init:
        raise ValueError(f"init must be one of {sorted(valid_init)}, got {init!r}")
    if method not in valid_method:
        raise ValueError(f"method must be one of {sorted(valid_method)}, got {method!r}")

    if method == "barnes_hut" and n_components > 3:
        raise ValueError(f"n_components must be <= 3 when method='barnes_hut', got {n_components}")

    resolved_assay_name = resolve_assay_name(container, assay_name)
    _, X = _validate_assay_layer(container, resolved_assay_name, base_layer)
    _check_no_nan_inf(X)

    X_dense = _prepare_matrix(X, dtype=np.dtype(dtype))
    n_samples, n_features = X_dense.shape
    if n_samples < 2:
        raise ValueError(f"t-SNE requires at least 2 samples, got {n_samples}")

    if perplexity >= n_samples:
        raise ValueError(f"perplexity ({perplexity}) must be < n_samples ({n_samples})")
    if init == "pca" and n_components > min(n_samples, n_features):
        raise ValueError(
            "init='pca' requires n_components <= min(n_samples, n_features). "
            f"Got n_components={n_components}, min_dim={min(n_samples, n_features)}."
        )

    from sklearn.manifold import TSNE

    reducer = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        early_exaggeration=early_exaggeration,
        learning_rate=learning_rate,
        metric=metric,
        init=init,
        method=method,
        max_iter=max_iter,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(X_dense)

    feature_names = [f"TSNE_{i + 1}" for i in range(n_components)]
    var_df = pl.DataFrame({"feature_id": feature_names})

    M = np.zeros(embedding.shape, dtype=np.int8)
    matrix = ScpMatrix(X=embedding, M=M)
    new_assay = Assay(var=var_df, layers={"X": matrix}, feature_id_col="feature_id")

    new_assays = {**container.assays, new_assay_name: new_assay}
    new_container = ScpContainer(
        obs=container.obs,
        assays=new_assays,
        history=list(container.history),
        sample_id_col=container.sample_id_col,
    )

    new_container.log_operation(
        action="reduce_tsne",
        params={
            "source_assay": resolved_assay_name,
            "source_layer": base_layer,
            "target_assay": new_assay_name,
            "n_components": n_components,
            "perplexity": perplexity,
            "early_exaggeration": early_exaggeration,
            "learning_rate": learning_rate,
            "metric": metric,
            "init": init,
            "method": method,
            "max_iter": max_iter,
        },
        description=f"t-SNE on {resolved_assay_name}/{base_layer} (perplexity={perplexity}).",
    )

    return new_container


__all__ = ["reduce_tsne"]
