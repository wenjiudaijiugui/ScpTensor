"""Nonlinear integration methods for DIA-based single-cell proteomics data.

Harmony uses an iterative clustering and correction approach to remove
batch effects while preserving biological variation.

Algorithm
---------

Harmony iteratively:
1. Assigns cells to clusters using current cell embeddings
2. Computes cluster-specific batch effects
3. Corrects cell embeddings to maximize diversity
4. Repeats until convergence

The diversity penalty encourages each cluster to contain cells from
all batches in equal proportion:

.. math::

    \\mathcal{L} = \\mathcal{L}_{cluster} + \\theta \\cdot \\mathcal{L}_{diversity}

References
----------
Korsunsky I, et al. Fast, sensitive and accurate integration of
single-cell data with Harmony. Nature Methods (2019).
"""

from __future__ import annotations

from scptensor.core.exceptions import MissingDependencyError
from scptensor.core.sparse_utils import is_sparse_matrix
from scptensor.core.structures import ScpContainer, ScpMatrix
from scptensor.integration.base import (
    get_integrate_method_info,
    prepare_integration_data,
    preserve_sparsity,
    register_integrate_method,
    validate_batch_integration_params,
    validate_embedding_input,
)


@register_integrate_method("harmony", integration_level="embedding", recommended_for_de=False)
@register_integrate_method("nonlinear", integration_level="embedding", recommended_for_de=False)
def integrate_harmony(
    container: ScpContainer,
    batch_key: str,
    assay_name: str = "protein",
    base_layer: str = "pca",
    new_layer_name: str | None = "harmony",
    theta: float | None = None,
    lamb: float | None = None,
    sigma: float = 0.1,
    nclust: int | None = None,
    max_iter_harmony: int = 10,
    max_iter_cluster: int = 20,
    epsilon_cluster: float = 1e-5,
    epsilon_harmony: float = 1e-4,
) -> ScpContainer:
    """Harmony integration for batch effect correction.

    Parameters
    ----------
    container : ScpContainer
        Input container with multiple batches
    batch_key : str
        Column name in obs containing batch labels
    assay_name : str, default="protein"
        Name of the assay to process
    base_layer : str, default="pca"
        Name of the layer to use as input
    new_layer_name : str | None, default="harmony"
        Name for the new layer with corrected data
    theta : float | None
        Clustering penalty parameter (default: 2.0)
    lamb : float | None
        Ridge regularization penalty (default: 1.0)
    sigma : float, default=0.1
        Bandwidth parameter for clustering kernel
    nclust : int | None
        Number of clusters (default: auto-detect)
    max_iter_harmony : int, default=10
        Maximum iterations for Harmony
    max_iter_cluster : int, default=20
        Maximum iterations for clustering
    epsilon_cluster : float, default=1e-5
        Convergence threshold for clustering
    epsilon_harmony : float, default=1e-4
        Convergence threshold for Harmony

    Returns
    -------
    ScpContainer
        Container with batch-corrected layer

    Examples
    --------
    >>> container = integrate_harmony(
    ...     container,
    ...     batch_key='batch',
    ...     assay_name='pca',
    ...     base_layer='X',
    ... )
    >>> container = integrate_harmony(
    ...     container,
    ...     batch_key='batch',
    ...     assay_name='protein',
    ...     base_layer='pca',
    ...     theta=3,
    ...     nclust=15,
    ... )
    """
    assay, layer = validate_embedding_input(
        container,
        assay_name,
        base_layer,
        method_name="Harmony integration",
    )
    _, _, unique_batches, _ = validate_batch_integration_params(
        container, batch_key, assay_name, min_batches=2, min_samples_per_batch=2
    )

    try:
        import harmonypy as hm
    except ImportError as exc:
        raise MissingDependencyError("harmonypy") from exc

    X = layer.X
    input_was_sparse = is_sparse_matrix(X)
    X_dense = prepare_integration_data(X, context="Harmony integration")
    meta_data = container.obs.to_pandas()

    harmony_params = {
        "theta": theta if theta is not None else 2.0,
        "lamb": lamb if lamb is not None else 1.0,
        "sigma": sigma,
        "nclust": nclust,
        "max_iter_harmony": max_iter_harmony,
        "max_iter_cluster": max_iter_cluster,
        "epsilon_cluster": epsilon_cluster,
        "epsilon_harmony": epsilon_harmony,
    }

    ho = hm.run_harmony(X_dense, meta_data, batch_key, **harmony_params)

    res = ho.Z_corr.T
    res = preserve_sparsity(res, input_was_sparse)

    new_matrix = ScpMatrix(
        X=res,
        M=layer.M.copy() if layer.M is not None else None,
    )
    assay.add_layer(new_layer_name or "harmony", new_matrix)

    method_info = get_integrate_method_info("harmony")
    container.log_operation(
        action="integration_harmony",
        params={
            "batch_key": batch_key,
            "theta": harmony_params["theta"],
            "lamb": harmony_params["lamb"],
            "sigma": harmony_params["sigma"],
            "nclust": harmony_params["nclust"],
            "n_batches": len(unique_batches),
            "integration_level": method_info.integration_level,
            "recommended_for_de": method_info.recommended_for_de,
        },
        description=(
            f"Harmony integration (theta={harmony_params['theta']}, "
            f"lamb={harmony_params['lamb']}) on layer '{base_layer}'."
        ),
    )

    return container


__all__ = ["integrate_harmony"]
