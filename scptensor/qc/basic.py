"""Basic Quality Control operations for single-cell proteomics data."""

import numpy as np
import polars as pl
import scipy.sparse as sp

from scptensor.core.exceptions import AssayNotFoundError, LayerNotFoundError, ScpValueError
from scptensor.core.structures import ScpContainer


def _get_layer(assay, layer_name: str | None = None):
    """Get a layer from assay, defaulting to 'raw' or first available layer.

    Args:
        assay: The Assay object.
        layer_name: Specific layer name to retrieve, or None for default.

    Returns:
        The ScpMatrix layer.

    Raises:
        LayerNotFoundError: If no layers exist in the assay.
    """
    from scptensor.core.exceptions import LayerNotFoundError

    if layer_name:
        if layer_name not in assay.layers:
            raise LayerNotFoundError(layer_name, "<assay>")
        return assay.layers[layer_name]

    return assay.layers.get("raw") or next(iter(assay.layers.values()), None)


def qc_basic(
    container: ScpContainer,
    assay_name: str = "protein",
    min_features: int = 200,
    min_cells: int = 3,
    detection_threshold: float = 0.0,
    new_layer_name: str | None = None,
) -> ScpContainer:
    """
    Perform basic Quality Control (QC) on samples and features.

    This function filters samples with too few detected features and
    features detected in too few samples.

    Parameters
    ----------
    container : ScpContainer
        The ScpContainer object.
    assay_name : str, default "protein"
        Name of the assay to perform QC on.
    min_features : int, default 200
        Minimum number of features required for a cell to be kept.
    min_cells : int, default 3
        Minimum number of cells a feature must be detected in.
    detection_threshold : float, default 0.0
        Threshold for a value to be considered detected.
    new_layer_name : str | None, default None
        Unused parameter (kept for API compatibility).

    Returns
    -------
    ScpContainer
        A new ScpContainer with filtered samples and features.

    Raises
    ------
    AssayNotFoundError
        If the specified assay does not exist.
    ScpValueError
        If min_features or min_cells parameters are invalid.

    Examples
    --------
    >>> container = qc_basic(
    ...     container,
    ...     assay_name="protein",
    ...     min_features=200,
    ...     min_cells=3
    ... )
    """
    # Validate parameters
    if min_features < 0:
        raise ScpValueError(
            f"min_features must be non-negative, got {min_features}.",
            parameter="min_features",
            value=min_features,
        )
    if min_cells < 0:
        raise ScpValueError(
            f"min_cells must be non-negative, got {min_cells}.",
            parameter="min_cells",
            value=min_cells,
        )

    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]
    layer = _get_layer(assay)
    X = layer.X

    # 1. Sample QC: Filter cells with too few features
    n_features_per_cell = np.sum(detection_threshold < X, axis=1)
    keep_samples_mask = n_features_per_cell >= min_features
    keep_samples_indices = np.where(keep_samples_mask)[0]

    container_filtered_samples = container.filter_samples(sample_indices=keep_samples_indices)

    # 2. Feature QC: Filter features detected in too few cells
    assay_filtered = container_filtered_samples.assays[assay_name]
    layer_filtered = _get_layer(assay_filtered)
    X_filtered = layer_filtered.X

    n_cells_per_feature = np.sum(X_filtered > detection_threshold, axis=0)
    keep_features_mask = n_cells_per_feature >= min_cells
    keep_features_indices = np.where(keep_features_mask)[0]

    container_final = container_filtered_samples.filter_features(
        assay_name, feature_indices=keep_features_indices
    )

    # Log QC stats
    n_samples_removed = container.n_samples - container_final.n_samples
    n_features_removed = assay.n_features - container_final.assays[assay_name].n_features

    container_final.log_operation(
        action="qc_basic",
        params={"assay": assay_name, "min_features": min_features, "min_cells": min_cells},
        description=f"Removed {n_samples_removed} samples and {n_features_removed} features.",
    )

    return container_final


def qc_score(
    container: ScpContainer,
    assay_name: str = "protein",
    layer_name: str = "raw",
    weights: dict[str, float] | None = None,
    detection_threshold: float = 0.0,
) -> ScpContainer:
    """Compute a comprehensive data quality score for each sample.

    The quality score combines multiple QC metrics into a single value [0, 1],
    where higher values indicate better quality.

    Metrics included:
    - Detection rate: Proportion of features detected
    - Total intensity: Sum of all intensities (normalized)
    - Missing rate: Proportion of missing values (lower is better)
    - Coefficient of variation: Variability relative to mean

    Parameters
    ----------
    container : ScpContainer
        Input container with data to score.
    assay_name : str, default "protein"
        Name of assay containing the layer to score.
    layer_name : str, default "raw"
        Name of layer to use for score calculation.
    weights : dict[str, float] | None, default None
        Custom weights for each metric. Default is None for equal weights.
        Expected keys: 'detection_rate', 'total_intensity', 'missing_rate', 'cv'.
    detection_threshold : float, default 0.0
        Value threshold for considering a value as detected.

    Returns
    -------
    ScpContainer
        Container with quality scores added to obs:
        - 'quality_score': Overall quality score [0, 1]
        - 'quality_detection_rate': Detection rate component
        - 'quality_total_intensity': Total intensity component
        - 'quality_missing_rate': Missing rate component (inverted)
        - 'quality_cv': Coefficient of variation component

    Raises
    ------
    AssayNotFoundError
        If assay_name does not exist.
    LayerNotFoundError
        If layer_name does not exist in the assay.
    ScpValueError
        If weights are invalid.

    Examples
    --------
    >>> container = qc_score(container, assay_name="protein")
    >>> scores = container.obs['quality_score'].to_numpy()
    >>> # Filter low-quality samples
    >>> high_quality_mask = scores > 0.5
    """
    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]
    if layer_name not in assay.layers:
        available = ", ".join(f"'{k}'" for k in assay.layers.keys())
        raise LayerNotFoundError(
            layer_name,
            assay_name,
            hint=f"Layer '{layer_name}' not found in assay '{assay_name}'. "
            f"Available layers: {available}.",
        )

    # Set default weights
    default_weights = {
        "detection_rate": 0.3,
        "total_intensity": 0.2,
        "missing_rate": 0.3,
        "cv": 0.2,
    }
    if weights is not None:
        # Validate and merge weights
        for key, value in weights.items():
            if key not in default_weights:
                raise ScpValueError(
                    f"Unknown weight key '{key}'. Valid keys: {list(default_weights.keys())}",
                    parameter="weights",
                    value=weights,
                )
            if value < 0:
                raise ScpValueError(
                    f"Weights must be non-negative, got {value} for '{key}'.",
                    parameter="weights",
                    value=weights,
                )
        default_weights.update(weights)

    # Normalize weights
    total_weight = sum(default_weights.values())
    weights = {k: v / total_weight for k, v in default_weights.items()}

    X = assay.layers[layer_name].X
    n_samples, n_features = X.shape

    # Handle sparse matrices
    if sp.issparse(X):
        X = X.toarray()  # type: ignore[union-attr]

    # Replace values below threshold with NaN for calculations
    X_detected = X.copy()
    X_detected[X_detected <= detection_threshold] = np.nan

    # 1. Detection rate: proportion of features detected per sample
    n_detected_per_sample = np.sum(detection_threshold < X, axis=1)
    detection_rate = n_detected_per_sample / n_features

    # 2. Total intensity (normalized)
    total_intensity = np.nansum(X, axis=1)
    if total_intensity.max() > total_intensity.min():
        total_intensity_norm = (total_intensity - total_intensity.min()) / (
            total_intensity.max() - total_intensity.min()
        )
    else:
        total_intensity_norm = np.ones_like(total_intensity)

    # 3. Missing rate (inverted - lower missing = higher score)
    missing_rate = 1.0 - detection_rate
    missing_rate_score = 1.0 - missing_rate  # Invert so lower missing = higher score

    # 4. Coefficient of variation (normalized)
    means = np.nanmean(X_detected, axis=1)
    stds = np.nanstd(X_detected, axis=1)
    cv = np.divide(stds, means, out=np.zeros_like(stds), where=means > 0)
    # Cap CV at reasonable values and normalize
    cv = np.clip(cv, 0, 2) / 2.0

    # Compute weighted quality score
    quality_score = (
        weights["detection_rate"] * detection_rate
        + weights["total_intensity"] * total_intensity_norm
        + weights["missing_rate"] * missing_rate_score
        + weights["cv"] * cv
    )

    # Clip to [0, 1]
    quality_score = np.clip(quality_score, 0, 1)

    # Add results to obs
    new_obs = container.obs.with_columns(
        pl.Series("quality_score", quality_score),
        pl.Series("quality_detection_rate", detection_rate),
        pl.Series("quality_total_intensity", total_intensity_norm),
        pl.Series("quality_missing_rate", missing_rate_score),
        pl.Series("quality_cv", cv),
    )

    new_container = ScpContainer(
        obs=new_obs,
        assays=container.assays,
        links=list(container.links),
        history=list(container.history),
        sample_id_col=container.sample_id_col,
    )

    mean_quality = float(np.mean(quality_score))
    new_container.log_operation(
        action="qc_score",
        params={"assay": assay_name, "layer_name": layer_name, "weights": weights},
        description=f"Computed quality scores (mean: {mean_quality:.3f}).",
    )

    return new_container


def compute_feature_variance(
    container: ScpContainer,
    assay_name: str = "protein",
    layer_name: str = "raw",
    detection_threshold: float = 0.0,
) -> ScpContainer:
    """Compute variance statistics for all features.

    This function calculates variance-related metrics for each feature,
    which are useful for feature selection and quality assessment.

    Parameters
    ----------
    container : ScpContainer
        Input container with data to analyze.
    assay_name : str, default "protein"
        Name of assay containing the layer.
    layer_name : str, default "raw"
        Name of layer to use for calculation.
    detection_threshold : float, default 0.0
        Value threshold for considering a value as detected.

    Returns
    -------
    ScpContainer
        Container with feature variance metrics added to assay var:
        - 'feature_variance': Variance of feature values
        - 'feature_std': Standard deviation
        - 'feature_mean': Mean value
        - 'feature_cv': Coefficient of variation
        - 'feature_iqr': Interquartile range

    Raises
    ------
    AssayNotFoundError
        If assay_name does not exist.
    LayerNotFoundError
        If layer_name does not exist in the assay.

    Examples
    --------
    >>> container = compute_feature_variance(container, assay_name="protein")
    >>> variances = container.assays['protein'].var['feature_variance'].to_numpy()
    """
    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]
    if layer_name not in assay.layers:
        available = ", ".join(f"'{k}'" for k in assay.layers.keys())
        raise LayerNotFoundError(
            layer_name,
            assay_name,
            hint=f"Layer '{layer_name}' not found in assay '{assay_name}'. "
            f"Available layers: {available}.",
        )

    X = assay.layers[layer_name].X

    # Handle sparse matrices
    if sp.issparse(X):
        X = X.toarray()  # type: ignore[union-attr]

    n_features = X.shape[1]

    # Replace values below threshold with NaN
    X_clean = X.copy()
    X_clean[X_clean <= detection_threshold] = np.nan

    # Compute statistics for each feature
    feature_means = np.nanmean(X_clean, axis=0)
    feature_stds = np.nanstd(X_clean, axis=0)
    feature_variances = np.nanvar(X_clean, axis=0)
    feature_cv = np.divide(
        feature_stds,
        feature_means,
        out=np.zeros_like(feature_stds),
        where=feature_means > 0,
    )

    # Compute IQR
    feature_q25 = np.nanpercentile(X_clean, 25, axis=0)
    feature_q75 = np.nanpercentile(X_clean, 75, axis=0)
    feature_iqr = feature_q75 - feature_q25

    # Replace NaN values with 0
    feature_means = np.nan_to_num(feature_means, nan=0.0)
    feature_stds = np.nan_to_num(feature_stds, nan=0.0)
    feature_variances = np.nan_to_num(feature_variances, nan=0.0)
    feature_cv = np.nan_to_num(feature_cv, nan=0.0)
    feature_iqr = np.nan_to_num(feature_iqr, nan=0.0)

    # Add results to var
    new_var = assay.var.with_columns(
        pl.Series("feature_variance", feature_variances),
        pl.Series("feature_std", feature_stds),
        pl.Series("feature_mean", feature_means),
        pl.Series("feature_cv", feature_cv),
        pl.Series("feature_iqr", feature_iqr),
    )

    # Create new container with updated var
    new_assay = assay.subset(np.arange(assay.n_features), copy_data=False)
    new_assay.var = new_var

    new_assays = {
        name: new_assay if name == assay_name else a for name, a in container.assays.items()
    }

    new_container = ScpContainer(
        obs=container.obs,
        assays=new_assays,
        links=list(container.links),
        history=list(container.history),
        sample_id_col=container.sample_id_col,
    )

    new_container.log_operation(
        action="compute_feature_variance",
        params={"assay": assay_name, "layer_name": layer_name},
        description=f"Computed variance statistics for {n_features} features.",
    )

    return new_container


def compute_feature_missing_rate(
    container: ScpContainer,
    assay_name: str = "protein",
    layer_name: str = "raw",
    detection_threshold: float = 0.0,
) -> ScpContainer:
    """Compute missing rate statistics for all features.

    This function calculates missing data metrics for each feature,
    which are useful for quality assessment and filtering decisions.

    Parameters
    ----------
    container : ScpContainer
        Input container with data to analyze.
    assay_name : str, default "protein"
        Name of assay containing the layer.
    layer_name : str, default "raw"
        Name of layer to use for calculation.
    detection_threshold : float, default 0.0
        Value threshold for considering a value as detected.

    Returns
    -------
    ScpContainer
        Container with feature missing rate metrics added to assay var:
        - 'feature_missing_rate': Proportion of samples where feature is missing
        - 'feature_n_detected': Number of samples where feature is detected
        - 'feature_prevalence': Proportion of samples where feature is detected
        - 'feature_detection_rate': Alias for prevalence (1 - missing_rate)

    Raises
    ------
    AssayNotFoundError
        If assay_name does not exist.
    LayerNotFoundError
        If layer_name does not exist in the assay.

    Examples
    --------
    >>> container = compute_feature_missing_rate(container, assay_name="protein")
    >>> missing_rates = container.assays['protein'].var['feature_missing_rate'].to_numpy()
    """
    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]
    if layer_name not in assay.layers:
        available = ", ".join(f"'{k}'" for k in assay.layers.keys())
        raise LayerNotFoundError(
            layer_name,
            assay_name,
            hint=f"Layer '{layer_name}' not found in assay '{assay_name}'. "
            f"Available layers: {available}.",
        )

    X = assay.layers[layer_name].X
    n_samples = X.shape[0]

    # Calculate detection for each feature
    if sp.issparse(X):
        n_detected = np.array(X.getnnz(axis=0)).flatten()  # type: ignore[union-attr]
    else:
        n_detected = np.sum(detection_threshold < X, axis=0)

    # Compute metrics
    missing_rate = 1.0 - (n_detected / n_samples)
    prevalence = n_detected / n_samples
    detection_rate = prevalence.copy()

    # Add results to var
    new_var = assay.var.with_columns(
        pl.Series("feature_missing_rate", missing_rate),
        pl.Series("feature_n_detected", n_detected),
        pl.Series("feature_prevalence", prevalence),
        pl.Series("feature_detection_rate", detection_rate),
    )

    # Create new container with updated var
    new_assay = assay.subset(np.arange(assay.n_features), copy_data=False)
    new_assay.var = new_var

    new_assays = {
        name: new_assay if name == assay_name else a for name, a in container.assays.items()
    }

    new_container = ScpContainer(
        obs=container.obs,
        assays=new_assays,
        links=list(container.links),
        history=list(container.history),
        sample_id_col=container.sample_id_col,
    )

    mean_missing = float(np.mean(missing_rate))
    new_container.log_operation(
        action="compute_feature_missing_rate",
        params={"assay": assay_name, "layer_name": layer_name},
        description=f"Computed missing rate (mean: {mean_missing:.3f}).",
    )

    return new_container
