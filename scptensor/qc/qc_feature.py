"""Feature-level Quality Control.

Handles QC for proteins/features including:
- Missingness analysis (Missing Rate vs Detection Rate)
- Coefficient of Variation (CV) filtering
- Feature quality metrics calculation
"""

import numpy as np
import polars as pl
import scipy.sparse as sp

from scptensor.core.filtering import FilterCriteria
from scptensor.core.structures import ScpContainer
from scptensor.qc._utils import (
    compute_detection_stats,
    log_filtering_operation,
    validate_assay,
    validate_layer,
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
    assay = validate_assay(container, assay_name)

    if layer_name is None:
        layer = next(iter(assay.layers.values()))
    else:
        validate_layer(assay, layer_name)
        layer = assay.layers[layer_name]

    X = layer.X

    # Compute detection statistics
    n_detected, detection_rate, means = compute_detection_stats(X)
    missing_rate = 1.0 - detection_rate

    # Compute CV
    cv = compute_cv(X, axis=0)

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

    # Create new assay with updated var
    new_assay = assay.subset(np.arange(assay.n_features), copy_data=False)
    new_assay.var = new_var

    # Create new container
    new_assays = {
        name: new_assay if name == assay_name else a for name, a in container.assays.items()
    }

    new_container = ScpContainer(
        obs=container.obs,
        assays=new_assays,
        links=container.links,
        history=container.history,
        sample_id_col=container.sample_id_col,
    )

    new_container.log_operation(
        action="calculate_feature_qc_metrics",
        params={
            "assay": assay_name,
            "layer": layer_name or next(iter(assay.layers.keys())),
            "n_features": assay.n_features,
            "n_samples": container.n_samples,
        },
        description=(
            f"Calculated QC metrics for {assay.n_features} features "
            f"across {container.n_samples} samples in assay '{assay_name}'. "
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
    validate_threshold(max_missing_rate, "max_missing_rate")
    assay = validate_assay(container, assay_name)

    if layer_name is None:
        layer = next(iter(assay.layers.values()))
    else:
        validate_layer(assay, layer_name)
        layer = assay.layers[layer_name]

    X = layer.X
    n_samples = X.shape[0]

    # Compute missing rate
    if sp.issparse(X):
        n_detected = X.getnnz(axis=0)
    else:
        n_detected = np.sum((X > 0) & (~np.isnan(X)), axis=0)

    missing_rate = 1.0 - (n_detected / n_samples)

    # Create filter mask
    keep_mask = missing_rate <= max_missing_rate
    keep_indices = np.where(keep_mask)[0]

    # Apply filtering
    criteria = FilterCriteria.by_indices(keep_indices)
    new_container = container.filter_features(assay_name, criteria)

    # Log provenance
    n_removed = assay.n_features - len(keep_indices)
    new_container = log_filtering_operation(
        new_container,
        "filter_features_by_missingness",
        assay_name,
        n_removed,
        assay.n_features,
        {"max_missing_rate": max_missing_rate},
    )

    return new_container


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

    assay = validate_assay(container, assay_name)

    if layer_name is None:
        layer = next(iter(assay.layers.values()))
    else:
        validate_layer(assay, layer_name)
        layer = assay.layers[layer_name]

    X = layer.X
    cv = compute_cv(X, axis=0, min_mean=min_mean)

    # Create filter mask
    keep_mask = (cv <= max_cv) & (~np.isnan(cv))
    keep_indices = np.where(keep_mask)[0]

    # Apply filtering
    criteria = FilterCriteria.by_indices(keep_indices)
    new_container = container.filter_features(assay_name, criteria)

    # Log provenance
    n_removed = assay.n_features - len(keep_indices)
    new_container = log_filtering_operation(
        new_container,
        "filter_features_by_cv",
        assay_name,
        n_removed,
        assay.n_features,
        {"max_cv": max_cv, "min_mean": min_mean},
    )

    return new_container
