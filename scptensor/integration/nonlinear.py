"""Nonlinear integration methods for single-cell proteomics data.

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

import numpy as np

from scptensor.core.sparse_utils import is_sparse_matrix
from scptensor.core.structures import ScpMatrix
from scptensor.core.utils import requires_dependency
from scptensor.integration.base import (
    prepare_integration_data,
    preserve_sparsity,
    register_integrate_method,
    validate_batch_integration_params,
    validate_layer_params,
)


@register_integrate_method("harmony")
@register_integrate_method("nonlinear")
@requires_dependency("harmonypy", "pip install harmonypy")
def integrate_harmony(
    container,
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
) -> "ScpContainer":
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
    >>> container = integrate_harmony(container, batch_key='batch')
    >>> container = integrate_harmony(container, batch_key='batch', theta=3, nclust=15)
    """
    import harmonypy as hm

    from scptensor.core.structures import ScpContainer

    # Validate assay and layer
    assay, layer = validate_layer_params(container, assay_name, base_layer)
    obs_df, batches, unique_batches, batch_counts = validate_batch_integration_params(
        container, batch_key, assay_name, min_batches=2, min_samples_per_batch=2
    )

    # Get and prepare data
    X = layer.X
    input_was_sparse = is_sparse_matrix(X)
    X_dense = prepare_integration_data(X)

    # Prepare metadata for Harmony
    meta_data = container.obs.to_pandas()

    # Set default values for Harmony parameters
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

    # Run Harmony
    ho = hm.run_harmony(X_dense, meta_data, batch_key, **harmony_params)

    # Harmony returns (n_pcs, n_samples), transpose to (n_samples, n_pcs)
    res = ho.Z_corr.T

    # Preserve sparsity if appropriate
    res = preserve_sparsity(res, input_was_sparse)

    # Create new layer
    M_input = layer.M
    new_matrix = ScpMatrix(
        X=res,
        M=M_input.copy() if M_input is not None else None,
    )
    container.assays[assay_name].add_layer(new_layer_name or "harmony", new_matrix)

    # Log operation
    container.log_operation(
        action="integration_harmony",
        params={
            "batch_key": batch_key,
            "theta": harmony_params["theta"],
            "lamb": harmony_params["lamb"],
            "sigma": harmony_params["sigma"],
            "nclust": harmony_params["nclust"],
            "n_batches": len(unique_batches),
        },
        description=f"Harmony integration (theta={harmony_params['theta']}, "
        f"lamb={harmony_params['lamb']}) on layer '{base_layer}'.",
    )

    return container


def harmony(
    container,
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
) -> "ScpContainer":
    """Harmony integration for batch effect correction.

    .. deprecated:: 0.1.0
        Use :func:`integrate_harmony` instead.
    """
    import warnings

    warnings.warn(
        "'harmony' is deprecated, use 'integrate_harmony' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return integrate_harmony(
        container=container,
        batch_key=batch_key,
        assay_name=assay_name,
        base_layer=base_layer,
        new_layer_name=new_layer_name,
        theta=theta,
        lamb=lamb,
        sigma=sigma,
        nclust=nclust,
        max_iter_harmony=max_iter_harmony,
        max_iter_cluster=max_iter_cluster,
        epsilon_cluster=epsilon_cluster,
        epsilon_harmony=epsilon_harmony,
    )


__all__ = ["integrate_harmony", "harmony"]
