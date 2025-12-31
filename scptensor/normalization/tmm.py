from typing import Optional
import numpy as np
from scptensor.core.structures import ScpContainer, ScpMatrix

def tmm_normalization(
    container: ScpContainer,
    assay_name: str = 'protein',
    base_layer_name: str = 'raw',
    new_layer_name: str = "tmm_norm",
    reference_sample: Optional[int] = None,
    trim_ratio: float = 0.3
) -> ScpContainer:
    """
    Trimmed Mean of M-values (TMM) normalization.

    Implementation adapted from edgeR's TMM method for proteomics data.
    This method is robust to composition bias and differentially expressed features.

    Mathematical Formulation:
        M = log2(y_i / y_j)    # Log ratio between samples i and j
        A = 0.5 * log2(y_i * y_j)  # Average log expression
        w = 1 / (A - trim)^2       # Weight function

        For each sample i relative to reference j:
        TMM_i = exp(sum(w_k * M_k) / sum(w_k))

    Reference:
        Robinson, M. D., & Oshlack, A. (2010).
        A scaling normalization method for differential expression analysis
        of RNA-seq data. Genome Biology, 11(3), R25.

    Args:
        container: ScpContainer containing the data
        assay_name: Name of the assay to process
        base_layer_name: Name of the layer to normalize
        new_layer_name: Name for the new normalized layer
        reference_sample: Index of reference sample (if None, use sample with median total intensity)
        trim_ratio: Proportion of extreme M values to trim (default 0.3)

    Returns:
        ScpContainer with added TMM-normalized layer
    """
    if assay_name not in container.assays:
        raise ValueError(f"Assay '{assay_name}' not found.")

    assay = container.assays[assay_name]
    if base_layer_name not in assay.layers:
        raise ValueError(f"Layer '{base_layer_name}' not found in assay '{assay_name}'.")

    X = assay.layers[base_layer_name].X.copy()
    n_samples, n_features = X.shape

    # Handle zeros by adding small constant to avoid log(0)
    X_safe = np.where(X == 0, 1e-8, X)

    # Select reference sample (sample with median total intensity)
    if reference_sample is None:
        sample_totals = np.sum(X_safe, axis=1)
        reference_sample = np.argsort(sample_totals)[n_samples // 2]

    ref_data = X_safe[reference_sample, :]

    # Calculate TMM scaling factors for each sample
    scaling_factors = np.ones(n_samples)
    scaling_factors[reference_sample] = 1.0  # Reference sample has factor of 1

    for i in range(n_samples):
        if i == reference_sample:
            continue

        sample_data = X_safe[i, :]

        # Remove features with zero values in either sample
        valid_mask = (sample_data > 0) & (ref_data > 0)
        if np.sum(valid_mask) == 0:
            scaling_factors[i] = 1.0
            continue

        sample_valid = sample_data[valid_mask]
        ref_valid = ref_data[valid_mask]

        # Calculate M and A values
        M = np.log2(sample_valid / ref_valid)
        A = 0.5 * np.log2(sample_valid * ref_valid)

        # Trim extreme M values
        n_keep = int(len(M) * (1 - trim_ratio))
        if n_keep < 2:  # Ensure we have enough features
            n_keep = 2

        # Sort by A and trim extremes
        sort_idx = np.argsort(A)
        trimmed_idx = sort_idx[n_keep//2 : len(M) - n_keep//2]

        if len(trimmed_idx) == 0:
            scaling_factors[i] = 1.0
            continue

        M_trimmed = M[trimmed_idx]
        A_trimmed = A[trimmed_idx]

        # Calculate weights (inverse of A variance)
        weights = 1.0 / (A_trimmed - np.mean(A_trimmed))**2
        weights[~np.isfinite(weights)] = 1.0  # Handle division by zero

        # Calculate weighted mean of M values
        weighted_M = np.sum(weights * M_trimmed) / np.sum(weights)
        scaling_factors[i] = 2 ** weighted_M

    # Apply scaling factors
    X_normalized = X / scaling_factors[:, np.newaxis]

    new_matrix = ScpMatrix(X=X_normalized, M=assay.layers[base_layer_name].M.copy())
    container.assays[assay_name].add_layer(new_layer_name, new_matrix)

    container.log_operation(
        action="normalization_tmm",
        params={
            "assay": assay_name,
            "reference_sample": reference_sample,
            "trim_ratio": trim_ratio
        },
        description=f"TMM normalization on layer '{base_layer_name}' -> '{new_layer_name}'."
    )

    return container