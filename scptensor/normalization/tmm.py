"""Trimmed Mean of M-values (TMM) normalization for ScpTensor.

Reference:
    Robinson, M. D., & Oshlack, A. (2010). A scaling normalization method
    for differential expression analysis of RNA-seq data. Genome Biology, 11(3), R25.
"""

import numpy as np

from scptensor.core.exceptions import (
    AssayNotFoundError,
    LayerNotFoundError,
    ScpValueError,
)
from scptensor.core.structures import ScpContainer, ScpMatrix


def norm_tmm(
    container: ScpContainer,
    assay_name: str = "protein",
    source_layer: str = "raw",
    new_layer_name: str = "tmm_norm",
    reference_sample: int | None = None,
    trim_ratio: float = 0.3,
) -> ScpContainer:
    """
    Trimmed Mean of M-values (TMM) normalization.

    Implementation adapted from edgeR's TMM method for proteomics data.
    This method is robust to composition bias and differentially expressed features.

    Mathematical Formulation:
        M = log2(y_i / y_j)          # Log ratio between samples i and j
        A = 0.5 * log2(y_i * y_j)    # Average log expression
        w = 1 / (A - trim)^2         # Weight function

        For each sample i relative to reference j:
        TMM_i = exp(sum(w_k * M_k) / sum(w_k))

    Parameters
    ----------
    container : ScpContainer
        ScpContainer containing the data.
    assay_name : str, default="protein"
        Name of the assay to process.
    source_layer : str, default="raw"
        Name of the layer to normalize.
    new_layer_name : str, default="tmm_norm"
        Name for the new normalized layer.
    reference_sample : Optional[int], default=None
        Index of reference sample. If None, uses sample with median total intensity.
    trim_ratio : float, default=0.3
        Proportion of extreme M values to trim. Must be in [0, 0.5).

    Returns
    -------
    ScpContainer
        ScpContainer with added TMM-normalized layer.

    Raises
    ------
    ScpValueError
        If trim_ratio or reference_sample parameters are invalid.
    AssayNotFoundError
        If the specified assay does not exist.
    LayerNotFoundError
        If the specified layer does not exist in the assay.

    Examples
    --------
    >>> import polars as pl
    >>> from scptensor.core.structures import ScpContainer, Assay, ScpMatrix
    >>> import numpy as np
    >>> container = ScpContainer(obs=pl.DataFrame({'_index': ['s1', 's2', 's3']}))
    >>> assay = Assay(var=pl.DataFrame({'_index': ['p1', 'p2', 'p3']}))
    >>> assay.add_layer('raw', ScpMatrix(X=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])))
    >>> container.add_assay('protein', assay)
    >>> result = norm_tmm(container)
    >>> 'tmm_norm' in result.assays['protein'].layers
    True
    """
    # Validate parameters (guard clause pattern)
    if not (0 <= trim_ratio < 0.5):
        raise ScpValueError(
            f"Trim ratio must be in [0, 0.5), got {trim_ratio}. "
            "Values closer to 0 trim fewer extremes, values closer to 0.5 trim more.",
            parameter="trim_ratio",
            value=trim_ratio,
        )

    # Validate assay and layer existence
    if assay_name not in container.assays:
        available = ", ".join(f"'{k}'" for k in container.assays.keys())
        raise AssayNotFoundError(
            assay_name,
            hint=f"Available assays: {available}. Use container.list_assays() to see all assays.",
        )

    assay = container.assays[assay_name]
    if source_layer not in assay.layers:
        available = ", ".join(f"'{k}'" for k in assay.layers.keys())
        raise LayerNotFoundError(
            source_layer,
            assay_name,
            hint=f"Available layers in assay '{assay_name}': {available}. "
            f"Use assay.list_layers() to see all layers.",
        )

    input_layer = assay.layers[source_layer]
    X = input_layer.X.copy()
    n_samples, n_features = X.shape

    # Validate reference_sample bounds
    if reference_sample is not None and (reference_sample < 0 or reference_sample >= n_samples):
        raise ScpValueError(
            f"Reference sample index must be in [0, {n_samples}), got {reference_sample}. "
            f"Use 0 for the first sample or None to auto-select.",
            parameter="reference_sample",
            value=reference_sample,
        )

    # Handle zeros to avoid log(0)
    X_safe = np.where(X == 0, 1e-8, X)

    # Auto-select reference: sample with median total intensity
    if reference_sample is None:
        sample_totals = np.sum(X_safe, axis=1)
        reference_sample = np.argsort(sample_totals)[n_samples // 2]

    ref_data = X_safe[reference_sample, :]

    # Pre-compute scaling factors for all samples
    scaling_factors = np.ones(n_samples)

    # Vectorized computation where possible
    for i in range(n_samples):
        if i == reference_sample:
            continue

        sample_data = X_safe[i, :]

        # Filter to features with positive values in both samples
        valid_mask = (sample_data > 0) & (ref_data > 0)
        if not valid_mask.any():
            continue

        sample_valid = sample_data[valid_mask]
        ref_valid = ref_data[valid_mask]

        # Calculate M (log ratio) and A (average log expression)
        M = np.log2(sample_valid / ref_valid)
        A = 0.5 * np.log2(sample_valid * ref_valid)

        # Trim extremes based on A values
        n_keep = max(2, int(len(M) * (1 - trim_ratio)))
        sort_idx = np.argsort(A)
        trim_start = n_keep // 2
        trim_end = len(M) - n_keep // 2

        if trim_start >= trim_end:
            continue

        trimmed_idx = sort_idx[trim_start:trim_end]
        M_trimmed = M[trimmed_idx]
        A_trimmed = A[trimmed_idx]

        # Calculate weights: inverse variance of A
        A_mean = np.mean(A_trimmed)
        weights = 1.0 / (A_trimmed - A_mean) ** 2
        weights[~np.isfinite(weights)] = 1.0

        # Weighted mean of M values
        weighted_M = np.sum(weights * M_trimmed) / np.sum(weights)
        scaling_factors[i] = 2**weighted_M

    # Apply scaling factors (broadcasting)
    X_normalized = X / scaling_factors[:, np.newaxis]

    # Create new layer
    new_matrix = ScpMatrix(
        X=X_normalized,
        M=input_layer.M.copy() if input_layer.M is not None else None,
    )
    assay.add_layer(new_layer_name, new_matrix)

    container.log_operation(
        action="normalization_tmm",
        params={
            "assay": assay_name,
            "source_layer": source_layer,
            "new_layer_name": new_layer_name,
            "reference_sample": reference_sample,
            "trim_ratio": trim_ratio,
        },
        description=f"TMM normalization on layer '{source_layer}' -> '{new_layer_name}'.",
    )

    return container
