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

import inspect

import numpy as np

from scptensor.core.exceptions import MissingDependencyError, ScpValueError
from scptensor.core.structures import ScpContainer
from scptensor.integration.base import (
    add_integrated_layer,
    log_integration_operation,
    prepare_integration_input,
    preserve_sparsity,
    register_integrate_method,
    validate_batch_integration_params,
    validate_embedding_input,
)


def _resolve_harmony_nclust(n_samples: int, nclust: int | None) -> int:
    """Mirror harmonypy auto-cluster selection while avoiding invalid zero clusters."""
    if nclust is None:
        return max(1, int(min(round(n_samples / 30.0), 100)))
    if nclust < 1:
        raise ScpValueError(
            f"nclust must be positive or None, got {nclust}.",
            parameter="nclust",
            value=nclust,
        )
    return int(nclust)


def _normalize_harmony_sigma(sigma: float | list[float] | np.ndarray, nclust: int) -> np.ndarray:
    """Return a per-cluster sigma vector accepted by current harmonypy releases."""
    if not isinstance(sigma, list | np.ndarray):
        sigma_value = float(sigma)
        if sigma_value <= 0:
            raise ScpValueError(
                f"sigma must be positive, got {sigma_value}.",
                parameter="sigma",
                value=sigma,
            )
        return np.repeat(sigma_value, nclust).astype(np.float64)

    sigma_arr = np.asarray(sigma, dtype=np.float64).reshape(-1)
    if sigma_arr.size != nclust:
        raise ScpValueError(
            "Harmony sigma must be a positive scalar or have exactly one value per cluster. "
            f"Got len(sigma)={sigma_arr.size}, nclust={nclust}.",
            parameter="sigma",
            value=sigma,
        )
    if np.any(sigma_arr <= 0):
        raise ScpValueError(
            "Harmony sigma values must all be positive.",
            parameter="sigma",
            value=sigma,
        )
    return sigma_arr


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

    X_dense, input_was_sparse = prepare_integration_input(
        layer,
        context="Harmony integration",
    )
    meta_data = container.obs.to_pandas()
    resolved_nclust = _resolve_harmony_nclust(meta_data.shape[0], nclust)
    sigma_vector = _normalize_harmony_sigma(sigma, resolved_nclust)
    logged_sigma: float | list[float]
    if not isinstance(sigma, list | np.ndarray):
        logged_sigma = float(sigma)
    else:
        logged_sigma = sigma_vector.tolist()

    harmony_params = {
        "theta": theta if theta is not None else 2.0,
        "lamb": lamb if lamb is not None else 1.0,
        "sigma": sigma_vector,
        "nclust": resolved_nclust,
        "max_iter_harmony": max_iter_harmony,
        "epsilon_cluster": epsilon_cluster,
        "epsilon_harmony": epsilon_harmony,
    }

    # harmonypy versions disagree on the clustering-iteration kwarg name:
    # older releases expose ``max_iter_kmeans``, while newer wrappers may use
    # ``max_iter_cluster``. Detect the installed signature instead of pinning
    # this wrapper to one harmonypy release line.
    run_harmony_params = inspect.signature(hm.run_harmony).parameters
    if "max_iter_cluster" in run_harmony_params:
        harmony_params["max_iter_cluster"] = max_iter_cluster
    elif "max_iter_kmeans" in run_harmony_params:
        harmony_params["max_iter_kmeans"] = max_iter_cluster

    ho = hm.run_harmony(X_dense, meta_data, batch_key, **harmony_params)

    res = ho.Z_corr
    res = preserve_sparsity(res, input_was_sparse)

    add_integrated_layer(assay, new_layer_name or "harmony", res, layer)

    return log_integration_operation(
        container,
        action="integration_harmony",
        method_name="harmony",
        params={
            "batch_key": batch_key,
            "theta": harmony_params["theta"],
            "lamb": harmony_params["lamb"],
            "sigma": logged_sigma,
            "nclust": resolved_nclust,
            "n_batches": len(unique_batches),
        },
        description=(
            f"Harmony integration (theta={harmony_params['theta']}, "
            f"lamb={harmony_params['lamb']}) on layer '{base_layer}'."
        ),
    )


__all__ = ["integrate_harmony"]
