"""Scanorama integration for DIA-based single-cell proteomics data.

Scanorama performs efficient batch correction and integration using mutual
nearest neighbors alignment. It is particularly effective for large-scale
datasets with many batches.

Algorithm
---------

Scanorama finds mutual nearest neighbors between batches and performs
simultaneous alignment and correction:

1. Finds mutual nearest neighbors between all batch pairs
2. Performs mutual nearest neighbors alignment
3. Corrects and integrates data in a single step

The alignment parameter sigma controls the aggressiveness of correction.

References
----------
Hie B, et al. Efficient integration of heterogeneous single-cell
transcriptomics data using Scanorama. Nature Biotechnology (2019).
"""

from __future__ import annotations

import numpy as np

from scptensor.core.exceptions import ScpValueError
from scptensor.core.structures import ScpContainer
from scptensor.core.utils import requires_dependency
from scptensor.integration.base import (
    add_integrated_layer,
    log_integration_operation,
    prepare_integration_input,
    preserve_sparsity,
    register_integrate_method,
    validate_batch_integration_params,
    validate_layer_params,
)


@register_integrate_method("scanorama", integration_level="embedding", recommended_for_de=False)
@requires_dependency("scanorama", "pip install scanorama")
def integrate_scanorama(
    container: ScpContainer,
    batch_key: str,
    assay_name: str = "protein",
    base_layer: str = "raw",
    new_layer_name: str | None = "scanorama",
    sigma: float = 15.0,
    alpha: float = 0.1,
    knn: int | None = None,
    approx: bool = True,
    return_dimred: bool = False,
    dimred: int | None = None,
) -> ScpContainer:
    """Scanorama integration for batch effect correction.

    Parameters
    ----------
    container : ScpContainer
        Input container with multiple batches
    batch_key : str
        Column name in obs containing batch labels
    assay_name : str, default="protein"
        Assay to use for integration
    base_layer : str, default="raw"
        Layer to use as input
    new_layer_name : str | None, default="scanorama"
        Name for the corrected layer
    sigma : float, default=15.0
        Alignment parameter (higher = more aggressive)
    alpha : float, default=0.1
        Mixture weight parameter
    knn : int | None
        Number of nearest neighbors (default: auto-detect)
    approx : bool, default=True
        Use approximate nearest neighbors
    return_dimred : bool, default=False
        Return dimensionality-reduced data
    dimred : int | None
        Number of dimensions for reduction

    Returns
    -------
    ScpContainer
        Container with integrated and corrected data

    Examples
    --------
    >>> container = integrate_scanorama(container, batch_key='batch')
    >>> container = integrate_scanorama(container, batch_key='batch', sigma=20.0)
    """
    import scanorama

    # Validate parameters
    if sigma <= 0:
        raise ScpValueError(f"sigma must be positive, got {sigma}.", parameter="sigma", value=sigma)
    if not (0 < alpha < 1):
        raise ScpValueError(
            f"alpha must be in (0, 1), got {alpha}.", parameter="alpha", value=alpha
        )
    if knn is not None and knn <= 0:
        raise ScpValueError(f"knn must be positive or None, got {knn}.", parameter="knn", value=knn)

    # Validate assay and layer
    assay, layer = validate_layer_params(container, assay_name, base_layer)
    _, batches, unique_batches, _ = validate_batch_integration_params(
        container, batch_key, assay_name, min_batches=2
    )

    # Get and prepare data
    X, input_was_sparse = prepare_integration_input(layer, context="Scanorama integration")

    # Prepare data list for scanorama
    datasets_list = [X[batches == b] for b in unique_batches]

    # Set default knn based on data size
    knn = knn or max(5, min(20, X.shape[0] // len(unique_batches) - 1))

    # Integrate using scanorama
    integrated = scanorama.correct(
        datasets_list,
        ds_names=[str(b) for b in unique_batches],
        return_dimred=return_dimred,
        return_dense=True,
        sigma=sigma,
        alpha=alpha,
        knn=knn,
        approx=approx,
        dimred=dimred,
    )

    X_corrected = np.vstack(integrated)

    # Preserve sparsity if appropriate
    if not return_dimred:
        X_corrected = preserve_sparsity(X_corrected, input_was_sparse)

    # Create new layer
    add_integrated_layer(assay, new_layer_name or "scanorama", X_corrected, layer)

    # Log operation
    return log_integration_operation(
        container,
        action="integration_scanorama",
        method_name="scanorama",
        params={
            "batch_key": batch_key,
            "assay": assay_name,
            "sigma": sigma,
            "alpha": alpha,
            "knn": knn,
            "approx": approx,
            "return_dimred": return_dimred,
            "n_batches": len(unique_batches),
        },
        description=f"Scanorama integration (sigma={sigma}, alpha={alpha}) on assay '{assay_name}'.",
    )


__all__ = ["integrate_scanorama"]
