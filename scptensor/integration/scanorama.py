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

from scptensor.core._structure_assay import Assay
from scptensor.core._structure_container import ScpContainer
from scptensor.core.exceptions import ScpValueError
from scptensor.core.utils import requires_dependency
from scptensor.integration.base import (
    add_integrated_layer,
    log_integration_operation,
    prepare_integration_input,
    preserve_sparsity,
    register_integrate_method,
    validate_batch_integration_params,
    validate_layer_context,
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
            f"alpha must be in (0, 1), got {alpha}.",
            parameter="alpha",
            value=alpha,
        )
    if knn is not None and knn <= 0:
        raise ScpValueError(f"knn must be positive or None, got {knn}.", parameter="knn", value=knn)
    if return_dimred:
        raise ScpValueError(
            "ScpTensor's Scanorama wrapper currently cannot store low-dimensional "
            "Scanorama embeddings back into an assay layer because assay layers "
            "must preserve the assay feature width. Use return_dimred=False to "
            "write the batch-corrected matrix, or add a dedicated embedding assay "
            "workflow before exposing low-dimensional Scanorama output.",
            parameter="return_dimred",
            value=return_dimred,
        )

    # Validate assay and layer
    ctx = validate_layer_context(container, assay_name, base_layer)
    assay = ctx.assay
    layer = ctx.layer
    _, batches, unique_batches, _ = validate_batch_integration_params(
        container,
        batch_key,
        ctx.resolved_assay_name,
        min_batches=2,
    )

    # Get and prepare data
    X, input_was_sparse = prepare_integration_input(layer, context="Scanorama integration")
    if not np.isfinite(X).all():
        raise ScpValueError(
            "Scanorama integration requires a complete matrix with only finite values "
            "(no NaN/Inf values). Please impute or filter missing values before "
            "batch integration.",
            parameter="X",
        )

    # Split by batch for Scanorama, but preserve original sample indices so the
    # corrected matrix can be restored to the container's sample order.
    datasets_list, batch_sample_indices = _split_scanorama_datasets(
        X=X,
        batches=batches,
        unique_batches=unique_batches,
    )
    genes_list = _build_scanorama_genes_list(assay, len(unique_batches))

    # Set default knn based on data size
    if knn is None:
        knn = min(20, max(1, min(dataset.shape[0] for dataset in datasets_list) - 1))

    # Integrate using scanorama
    corrected_datasets, corrected_genes = scanorama.correct(
        datasets_list,
        genes_list,
        **_build_scanorama_correct_kwargs(
            unique_batches=unique_batches,
            return_dimred=return_dimred,
            sigma=sigma,
            alpha=alpha,
            knn=knn,
            approx=approx,
            dimred=dimred,
        ),
    )
    X_corrected = _stack_scanorama_corrected_datasets(
        corrected_datasets=corrected_datasets,
        corrected_genes=corrected_genes,
        feature_ids=genes_list[0],
        batch_sample_indices=batch_sample_indices,
        n_samples=X.shape[0],
    )

    # Preserve sparsity if appropriate
    X_corrected = preserve_sparsity(X_corrected, input_was_sparse)

    # Create new layer
    layer_name = new_layer_name or "scanorama"
    add_integrated_layer(
        assay,
        layer_name,
        X_corrected,
        layer,
        source_assay_name=ctx.resolved_assay_name,
        source_layer_name=base_layer,
        action="integration_scanorama",
    )

    # Log operation
    return log_integration_operation(
        container,
        action="integration_scanorama",
        method_name="scanorama",
        params={
            "batch_key": batch_key,
            "assay": ctx.resolved_assay_name,
            "base_layer": base_layer,
            "new_layer_name": layer_name,
            "sigma": sigma,
            "alpha": alpha,
            "knn": knn,
            "approx": approx,
            "return_dimred": return_dimred,
            "n_batches": len(unique_batches),
        },
        description=(
            f"Scanorama integration (sigma={sigma}, alpha={alpha}) "
            f"on assay '{ctx.resolved_assay_name}'."
        ),
    )


def _build_scanorama_genes_list(assay: Assay, n_batches: int) -> list[list[str]]:
    """Build per-batch feature-id lists for Scanorama's required genes_list argument."""
    feature_ids = [str(feature_id) for feature_id in assay.feature_ids.to_list()]
    return [list(feature_ids) for _ in range(n_batches)]


def _split_scanorama_datasets(
    *,
    X: np.ndarray,
    batches: np.ndarray,
    unique_batches: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Split the matrix by batch and remember original sample positions."""
    datasets: list[np.ndarray] = []
    sample_indices: list[np.ndarray] = []
    for batch in unique_batches:
        idx = np.flatnonzero(batches == batch)
        sample_indices.append(idx)
        datasets.append(X[idx])
    return datasets, sample_indices


def _build_scanorama_correct_kwargs(
    *,
    unique_batches: np.ndarray,
    return_dimred: bool,
    sigma: float,
    alpha: float,
    knn: int,
    approx: bool,
    dimred: int | None,
) -> dict[str, object]:
    """Build kwargs while preserving Scanorama defaults for omitted options."""
    kwargs: dict[str, object] = {
        "ds_names": [str(batch) for batch in unique_batches],
        "return_dimred": return_dimred,
        "return_dense": True,
        "sigma": sigma,
        "alpha": alpha,
        "knn": knn,
        "approx": approx,
    }
    if dimred is not None:
        kwargs["dimred"] = dimred
    return kwargs


def _stack_scanorama_corrected_datasets(
    *,
    corrected_datasets: list[np.ndarray],
    corrected_genes: list[str],
    feature_ids: list[str],
    batch_sample_indices: list[np.ndarray],
    n_samples: int,
) -> np.ndarray:
    """Align Scanorama outputs to assay features and restore original sample order."""
    corrected_gene_names = [str(gene) for gene in corrected_genes]
    feature_name_set = set(feature_ids)
    corrected_name_set = set(corrected_gene_names)
    if corrected_name_set != feature_name_set:
        raise ScpValueError(
            "Scanorama returned a corrected feature set that does not match the "
            "current assay feature IDs. ScpTensor's current Scanorama wrapper "
            "requires all batches to share the same assay feature space so the "
            "corrected matrices can be written back to the existing assay.",
            parameter="feature_ids",
        )

    corrected_positions = {gene: idx for idx, gene in enumerate(corrected_gene_names)}
    reordered_indices = [corrected_positions[feature_id] for feature_id in feature_ids]
    corrected_matrix = np.empty((n_samples, len(feature_ids)), dtype=float)
    for dataset, sample_idx in zip(corrected_datasets, batch_sample_indices, strict=True):
        corrected_matrix[sample_idx] = np.asarray(dataset)[:, reordered_indices]
    return corrected_matrix


__all__ = ["integrate_scanorama"]
