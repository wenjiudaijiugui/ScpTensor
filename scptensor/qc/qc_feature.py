"""Feature-level Quality Control.

Handles QC for proteins/features including:
- Missingness analysis (Missing Rate vs Detection Rate)
- Coefficient of Variation (CV) filtering
- Feature quality metrics calculation
"""

import numpy as np
import polars as pl

from scptensor.core.structures import ScpContainer
from scptensor.qc._utils import (
    compute_detection_stats,
    count_detected,
    filter_features_with_provenance,
    resolve_assay,
    resolve_layer,
    validate_threshold,
)
from scptensor.qc.metrics import compute_cv


def calculate_feature_qc_metrics(
    container: ScpContainer,
    assay_name: str = "protein",
    layer_name: str | None = None,
) -> ScpContainer:
    """Calculate quality control metrics for features.

    Computes:
    - missing_rate: Proportion of missing values per feature
    - detection_rate: Complement of missing_rate
    - mean_expression: Mean intensity of detected values
    - cv: Coefficient of Variation

    Parameters
    ----------
    container : ScpContainer
        ScpContainer containing the assay to analyze.
    assay_name : str, default="protein"
        Name of the assay containing feature-level data.
    layer_name : str, optional
        Name of the layer to use. If None, uses first available layer.

    Returns
    -------
    ScpContainer
        ScpContainer with QC metrics added to assay.var.

    Examples
    --------
    >>> result = calculate_feature_qc_metrics(container)
    >>> result.assays['protein'].var[['missing_rate', 'detection_rate', 'cv']]
    """
    resolved_assay_name, assay = resolve_assay(container, assay_name)
    layer_name, layer = resolve_layer(
        assay,
        assay_name=resolved_assay_name,
        layer_name=layer_name,
        fallback_to_first=True,
    )

    # Compute detection statistics
    n_detected, detection_rate, means = compute_detection_stats(layer.X, M=layer.M)
    missing_rate = 1.0 - detection_rate

    # Compute CV
    cv = compute_cv(layer.X, axis=0)

    # Create new metrics DataFrame
    new_metrics = pl.DataFrame(
        {
            "missing_rate": missing_rate,
            "detection_rate": detection_rate,
            "mean_expression": means,
            "cv": cv,
        }
    )

    # Merge with existing var
    current_var = assay.var
    new_var = current_var.hstack(new_metrics)

    new_container = container.copy()
    new_container.assays[resolved_assay_name].var = new_var

    new_container.log_operation(
        action="calculate_feature_qc_metrics",
        params={
            "assay": resolved_assay_name,
            "layer": layer_name,
            "n_features": assay.n_features,
            "n_samples": container.n_samples,
        },
        description=(
            f"Calculated QC metrics for {assay.n_features} features "
            f"across {container.n_samples} samples in assay '{resolved_assay_name}'. "
            f"Metrics: missing_rate (mean={np.mean(missing_rate):.3f}), "
            f"detection_rate (mean={np.mean(detection_rate):.3f}), "
            f"cv (mean={np.nanmean(cv):.3f})."
        ),
    )

    return new_container


def filter_features_by_missingness(
    container: ScpContainer,
    assay_name: str = "protein",
    layer_name: str | None = None,
    max_missing_rate: float = 0.5,
) -> ScpContainer:
    """Filter features based on missing rate.

    Removes features with high missing rates to improve data quality.
    Missing rate = proportion of samples where feature is not detected.

    Parameters
    ----------
    container : ScpContainer
        ScpContainer containing the assay to filter.
    assay_name : str, default="protein"
        Name of the assay containing feature-level data.
    layer_name : str, optional
        Name of the layer to use. If None, uses first available layer.
    max_missing_rate : float, default=0.5
        Maximum acceptable missing rate [0, 1].
        Recommended: 0.2-0.3 (stringent), 0.5 (moderate), 0.7 (lenient).

    Returns
    -------
    ScpContainer
        ScpContainer with low-quality features removed.

    Examples
    --------
    >>> result = filter_features_by_missingness(container, max_missing_rate=0.5)
    >>> result.assays['protein'].n_features
    4
    """
    validate_threshold(max_missing_rate, "max_missing_rate", min_val=0.0, max_val=1.0)
    resolved_assay_name, assay = resolve_assay(container, assay_name)

    _, layer = resolve_layer(
        assay,
        assay_name=resolved_assay_name,
        layer_name=layer_name,
        fallback_to_first=True,
    )

    n_samples = layer.X.shape[0]

    # Compute missing rate
    n_detected = count_detected(layer.X, layer.M, axis=0)
    missing_rate = 1.0 - (n_detected / n_samples)

    # Create filter mask
    keep_mask = missing_rate <= max_missing_rate
    keep_indices = np.where(keep_mask)[0]

    n_removed = assay.n_features - len(keep_indices)
    return filter_features_with_provenance(
        container,
        resolved_assay_name,
        keep_indices,
        action="filter_features_by_missingness",
        params={
            "assay": resolved_assay_name,
            "n_removed": n_removed,
            "n_total": assay.n_features,
            "max_missing_rate": max_missing_rate,
        },
        description=(
            f"Filtered {n_removed}/{assay.n_features} features from "
            f"{resolved_assay_name} by missingness "
            f"(max_missing_rate={max_missing_rate})."
        ),
    )


def filter_features_by_cv(
    container: ScpContainer,
    assay_name: str = "protein",
    layer_name: str | None = None,
    max_cv: float = 1.0,
    min_mean: float = 1e-6,
) -> ScpContainer:
    """Filter features based on Coefficient of Variation (CV).

    Removes features with high CV to improve measurement stability.
    CV = standard deviation / mean (dimensionless metric).

    Parameters
    ----------
    container : ScpContainer
        ScpContainer containing the assay to filter.
    assay_name : str, default="protein"
        Name of the assay containing feature-level data.
    layer_name : str, optional
        Name of the layer to use. If None, uses first available layer.
    max_cv : float, default=1.0
        Maximum acceptable CV.
        Recommended: 0.3-0.5 (stringent), 1.0 (lenient).
    min_mean : float, default=1e-6
        Minimum mean value for CV calculation.

    Returns
    -------
    ScpContainer
        ScpContainer with high-CV features removed.

    Examples
    --------
    >>> result = filter_features_by_cv(container, max_cv=0.5)
    >>> result.assays['protein'].n_features
    3
    """
    if max_cv <= 0:
        from scptensor.core.exceptions import ScpValueError

        raise ScpValueError(
            f"max_cv must be positive, got {max_cv}.",
            parameter="max_cv",
            value=max_cv,
        )

    resolved_assay_name, assay = resolve_assay(container, assay_name)

    _, layer = resolve_layer(
        assay,
        assay_name=resolved_assay_name,
        layer_name=layer_name,
        fallback_to_first=True,
    )

    X = layer.X
    cv = compute_cv(X, axis=0, min_mean=min_mean)

    # Create filter mask
    keep_mask = (cv <= max_cv) & (~np.isnan(cv))
    keep_indices = np.where(keep_mask)[0]

    n_removed = assay.n_features - len(keep_indices)
    return filter_features_with_provenance(
        container,
        resolved_assay_name,
        keep_indices,
        action="filter_features_by_cv",
        params={
            "assay": resolved_assay_name,
            "n_removed": n_removed,
            "n_total": assay.n_features,
            "max_cv": max_cv,
            "min_mean": min_mean,
        },
        description=(
            f"Filtered {n_removed}/{assay.n_features} features from "
            f"{resolved_assay_name} by CV (max_cv={max_cv})."
        ),
    )
