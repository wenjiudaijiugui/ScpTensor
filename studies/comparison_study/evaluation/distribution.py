"""
Data distribution evaluation metrics.

This module implements metrics to assess changes in data distribution
after pipeline processing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    pass


def compute_sparsity(container: Any) -> float:
    """
    Compute data sparsity (fraction of missing values).

    Parameters
    ----------
    container : ScpContainer
        Data container

    Returns
    -------
    float
        Sparsity fraction between 0 and 1
    """
    X = _get_data_matrix(container)

    # Try to use mask matrix if available
    if hasattr(container, "assays") and container.assays:
        assay_name = list(container.assays.keys())[0]
        assay = container.assays[assay_name]

        # Get the main layer
        layer_name = "log" if "log" in assay.layers else "X"
        if layer_name in assay.layers:
            M = assay.layers[layer_name].M

            # Mask codes > 0 indicate missing or imputed
            if M is not None:
                if hasattr(M, "toarray"):
                    M = M.toarray()
                n_missing = np.sum(M > 0)
                total = M.size
                return float(n_missing / total) if total > 0 else 0.0

    # Fallback: count zeros as missing
    return float(np.sum(X == 0) / X.size)


def compute_statistics(container: Any) -> dict[str, float]:
    """
    Compute statistical properties of the data.

    Parameters
    ----------
    container : ScpContainer
        Data container

    Returns
    -------
    dict[str, float]
        Dictionary with statistical properties:
        - mean: Mean value
        - std: Standard deviation
        - skewness: Skewness (asymmetry)
        - kurtosis: Kurtosis (tailedness)
        - cv: Coefficient of variation
        - median: Median value
        - mad: Median absolute deviation
    """
    X = _get_data_matrix(container)

    # Exclude zeros (missing values)
    X_valid = X[X != 0]

    if X_valid.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "cv": 0.0,
            "median": 0.0,
            "mad": 0.0,
        }

    # Flatten for scipy stats
    X_valid_flat = X_valid.flatten()

    mean_val = float(np.mean(X_valid))
    std_val = float(np.std(X_valid))
    median_val = float(np.median(X_valid))

    # Compute MAD (median absolute deviation)
    mad_val = float(np.median(np.abs(X_valid_flat - median_val)))

    return {
        "mean": mean_val,
        "std": std_val,
        "skewness": float(stats.skew(X_valid_flat)),
        "kurtosis": float(stats.kurtosis(X_valid_flat)),
        "cv": std_val / mean_val if mean_val != 0 else 0.0,
        "median": median_val,
        "mad": mad_val,
    }


def distribution_test(
    original: Any,
    result: Any,
) -> tuple[float, float]:
    """
    Perform Kolmogorov-Smirnov test to compare distributions.

    The KS test compares the empirical distributions of two samples.
    A low p-value (< 0.05) suggests the distributions are significantly different.

    Parameters
    ----------
    original : ScpContainer
        Original container
    result : ScpContainer
        Processed container

    Returns
    -------
    statistic : float
        KS statistic (0 = identical, 1 = completely different)
    pvalue : float
        P-value for the test (smaller = more significant difference)

    Examples
    --------
    >>> stat, pval = distribution_test(original, processed)
    >>> if pval < 0.05:
    ...     print("Distributions are significantly different")
    """
    X_orig = _get_data_matrix(original)
    X_result = _get_data_matrix(result)

    # Flatten and exclude zeros (missing values)
    X_orig_valid = X_orig[X_orig != 0].flatten()
    X_result_valid = X_result[X_result != 0].flatten()

    if X_orig_valid.size == 0 or X_result_valid.size == 0:
        return 0.0, 1.0

    # Perform KS test
    statistic, pvalue = stats.ks_2samp(X_orig_valid, X_result_valid)

    return float(statistic), float(pvalue)


def compute_quantiles(
    container: Any,
    q: list[float] | tuple[float, ...] | None = None,
) -> dict[str, float]:
    """
    Compute quantiles of the data distribution.

    Parameters
    ----------
    container : ScpContainer
        Data container
    q : list[float] or tuple[float, ...], optional
        Quantiles to compute (default: [0.25, 0.5, 0.75])

    Returns
    -------
    dict[str, float]
        Dictionary with quantile values (e.g., 'q25', 'q50', 'q75')
    """
    if q is None:
        q = (0.25, 0.5, 0.75)

    X = _get_data_matrix(container)
    X_valid = X[X != 0]

    if X_valid.size == 0:
        return {f"q{int(qi * 100)}": 0.0 for qi in q}

    quantiles = np.quantile(X_valid, q)

    return {f"q{int(qi * 100)}": float(qv) for qi, qv in zip(q, quantiles, strict=False)}


def compute_distribution_similarity(
    original: Any,
    result: Any,
    method: str = "wasserstein",
) -> dict[str, float]:
    """
    Compute distribution similarity metrics between original and result.

    Parameters
    ----------
    original : ScpContainer
        Original container
    result : ScpContainer
        Processed container
    method : str, default "wasserstein"
        Distance metric to use. Options: "wasserstein", "energy", "ks"

    Returns
    -------
    dict[str, float]
        Dictionary with similarity metrics:
        - distance: Computed distance (lower = more similar)
        - normalized_distance: Distance normalized by data range

    Examples
    --------
    >>> # Using Wasserstein distance (Earth Mover's Distance)
    >>> sim = compute_distribution_similarity(orig, proc, method="wasserstein")
    >>> print(f"Distance: {sim['distance']:.4f}")
    """
    X_orig = _get_data_matrix(original)
    X_result = _get_data_matrix(result)

    # Flatten and exclude zeros
    X_orig_valid = X_orig[X_orig != 0].flatten()
    X_result_valid = X_result[X_result != 0].flatten()

    if X_orig_valid.size == 0 or X_result_valid.size == 0:
        return {"distance": 0.0, "normalized_distance": 0.0}

    # Compute distance based on method
    if method == "wasserstein":
        # Earth Mover's Distance
        distance = stats.wasserstein_distance(X_orig_valid, X_result_valid)
    elif method == "energy":
        # Energy distance
        distance = stats.energy_distance(X_orig_valid, X_result_valid)
    elif method == "ks":
        # Kolmogorov-Smirnov statistic
        distance, _ = stats.ks_2samp(X_orig_valid, X_result_valid)
    else:
        msg = f"Unknown method: {method}. Use 'wasserstein', 'energy', or 'ks'"
        raise ValueError(msg)

    # Normalize by data range
    data_range = np.max(X_orig_valid) - np.min(X_orig_valid)
    normalized_distance = distance / data_range if data_range > 0 else 0.0

    return {
        "distance": float(distance),
        "normalized_distance": float(normalized_distance),
    }


def _get_data_matrix(container: Any) -> np.ndarray:
    """
    Helper to get dense data matrix.

    Returns
    -------
    np.ndarray
        Dense data matrix of shape (n_cells, n_features)
    """
    if not hasattr(container, "assays") or not container.assays:
        raise ValueError("Container has no assays")

    # Get first assay
    assay_name = list(container.assays.keys())[0]
    assay = container.assays[assay_name]

    # Prefer normalized layer
    layer_name = "log" if "log" in assay.layers else "X"
    if layer_name not in assay.layers:
        if assay.layers:
            layer_name = list(assay.layers.keys())[0]
        else:
            raise ValueError("Assay has no layers")

    X = assay.layers[layer_name].X

    # Convert sparse to dense
    if hasattr(X, "toarray"):
        return X.toarray()
    return X
