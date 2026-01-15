"""Bivariate QC metrics for single-cell proteomics data.

This module provides methods for analyzing relationships between samples
and detecting outliers based on pairwise correlations and other bivariate
statistics.
"""

from typing import Literal

import numpy as np
import polars as pl
import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from scptensor.core.exceptions import AssayNotFoundError, LayerNotFoundError, ScpValueError
from scptensor.core.structures import ScpContainer


def compute_pairwise_correlation(
    container: ScpContainer,
    assay_name: str = "protein",
    layer_name: str = "raw",
    method: Literal["pearson", "spearman"] = "spearman",
    detection_threshold: float = 0.0,
) -> ScpContainer:
    """Compute pairwise correlation between all samples.

    This function calculates the correlation matrix between all samples,
    which can be used to identify samples with unusual expression patterns
    or to detect batch effects.

    Parameters
    ----------
    container : ScpContainer
        Input container with data to analyze.
    assay_name : str, default "protein"
        Name of assay containing the layer.
    layer_name : str, default "raw"
        Name of layer to use for correlation calculation.
    method : {"pearson", "spearman"}, default "spearman"
        Correlation method.
    detection_threshold : float, default 0.0
        Value below which is considered missing.

    Returns
    -------
    ScpContainer
        Container with pairwise correlation matrix stored in obs
        as 'pairwise_correlation' (flattened upper triangle).

    Raises
    ------
    AssayNotFoundError
        If assay_name does not exist.
    LayerNotFoundError
        If layer_name does not exist in the assay.
    ScpValueError
        If method is invalid.

    Examples
    --------
    >>> container = compute_pairwise_correlation(
    ...     container, assay_name="protein", method="spearman"
    ... )
    >>> # Get correlation matrix
    >>> cor_matrix = container.obs['pairwise_correlation'][0]
    """
    if method not in ("pearson", "spearman"):
        raise ScpValueError(
            f"method must be 'pearson' or 'spearman', got '{method}'.",
            parameter="method",
            value=method,
        )

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
        X = X.toarray()

    # Replace values below threshold with NaN for proper correlation
    X_copy = X.copy()
    X_copy[X_copy <= detection_threshold] = np.nan

    n_samples = X_copy.shape[0]

    # Compute correlation matrix
    if method == "spearman":
        # Spearman correlation (rank-based)
        corr_matrix = np.ones((n_samples, n_samples))

        # Compute upper triangle (vectorized where possible)
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                # Mask out pairs where both have NaN
                mask = ~(np.isnan(X_copy[i]) | np.isnan(X_copy[j]))
                if mask.sum() > 3:  # Need at least 3 pairs
                    corr, _ = spearmanr(X_copy[i][mask], X_copy[j][mask])
                    if not np.isnan(corr):
                        corr_matrix[i, j] = corr
                        corr_matrix[j, i] = corr
    else:
        # Pearson correlation
        from scipy.stats import pearsonr

        corr_matrix = np.ones((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                mask = ~(np.isnan(X_copy[i]) | np.isnan(X_copy[j]))
                if mask.sum() > 3:
                    corr, _ = pearsonr(X_copy[i][mask], X_copy[j][mask])
                    if not np.isnan(corr):
                        corr_matrix[i, j] = corr
                        corr_matrix[j, i] = corr

    # Store as list of matrices (one per batch/sample group if applicable)
    # Convert to list of lists for Polars compatibility and replicate for all rows
    corr_list = corr_matrix.tolist()
    # Replicate for all samples to match DataFrame height
    corr_list_replicated = [corr_list] * n_samples
    new_obs = container.obs.with_columns(
        pl.Series("pairwise_correlation", corr_list_replicated)
    )

    new_container = ScpContainer(
        obs=new_obs,
        assays=container.assays,
        links=list(container.links),
        history=list(container.history),
        sample_id_col=container.sample_id_col,
    )

    new_container.log_operation(
        action="compute_pairwise_correlation",
        params={"assay": assay_name, "layer_name": layer_name, "method": method},
        description=f"Computed {method} correlation matrix ({n_samples}x{n_samples}).",
    )

    return new_container


def detect_outlier_samples(
    container: ScpContainer,
    assay_name: str = "protein",
    layer_name: str = "raw",
    method: Literal["median_absolute_deviation", "zscore", "iqr", "isolation"] = "median_absolute_deviation",
    threshold: float = 3.0,
    metric: Literal["mean_correlation", "median_distance", "total_intensity"] = "mean_correlation",
    random_state: int = 42,
) -> ScpContainer:
    """Detect outlier samples based on their relationship to other samples.

    This function identifies samples that are outliers based on their
    correlation with other samples or their distance to the median sample.

    Parameters
    ----------
    container : ScpContainer
        Input container with data to analyze.
    assay_name : str, default "protein"
        Name of assay containing the layer.
    layer_name : str, default "raw"
        Name of layer to use for detection.
    method : {"median_absolute_deviation", "zscore", "iqr", "isolation"}, default "median_absolute_deviation"
        Outlier detection method:
        - 'median_absolute_deviation': Uses MAD of mean correlations
        - 'zscore': Uses z-score of mean correlations
        - 'iqr': Uses interquartile range
        - 'isolation': Uses Isolation Forest
    threshold : float, default 3.0
        Threshold for outlier detection.
        For MAD/zscore, typically 3.0. For IQR, multiplier for IQR.
    metric : {"mean_correlation", "median_distance", "total_intensity"}, default "mean_correlation"
        Metric to use for outlier detection:
        - 'mean_correlation': Mean correlation to other samples
        - 'median_distance': Median distance to other samples
        - 'total_intensity': Total intensity (sum of all values)
    random_state : int, default 42
        Random state for isolation forest.

    Returns
    -------
    ScpContainer
        Container with outlier detection results added to obs:
        - 'is_correlation_outlier': Boolean indicating outlier status
        - 'outlier_metric_value': The metric value used for detection

    Raises
    ------
    AssayNotFoundError
        If assay_name does not exist.
    LayerNotFoundError
        If layer_name does not exist in the assay.
    ScpValueError
        If parameters are invalid.

    Examples
    --------
    >>> container = detect_outlier_samples(
    ...     container, assay_name="protein", method="median_absolute_deviation"
    ... )
    >>> outliers = container.obs['is_correlation_outlier'].to_numpy()
    >>> n_outliers = outliers.sum()
    """
    valid_methods = ("median_absolute_deviation", "zscore", "iqr", "isolation")
    if method not in valid_methods:
        raise ScpValueError(
            f"method must be one of {valid_methods}, got '{method}'.",
            parameter="method",
            value=method,
        )

    valid_metrics = ("mean_correlation", "median_distance", "total_intensity")
    if metric not in valid_metrics:
        raise ScpValueError(
            f"metric must be one of {valid_metrics}, got '{metric}'.",
            parameter="metric",
            value=metric,
        )

    if threshold <= 0:
        raise ScpValueError(
            f"threshold must be positive, got {threshold}.",
            parameter="threshold",
            value=threshold,
        )

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
        X = X.toarray()

    n_samples = X.shape[0]

    # Calculate metric values
    if metric == "mean_correlation":
        # Compute mean correlation for each sample
        from scipy.stats import spearmanr

        metric_values = np.zeros(n_samples)
        for i in range(n_samples):
            correlations = []
            for j in range(n_samples):
                if i != j:
                    mask = ~(np.isnan(X[i]) | np.isnan(X[j]))
                    if mask.sum() > 3:
                        corr, _ = spearmanr(X[i][mask], X[j][mask])
                        if not np.isnan(corr):
                            correlations.append(corr)
            metric_values[i] = np.mean(correlations) if correlations else 0

    elif metric == "median_distance":
        # Compute median distance to other samples
        distances = pdist(X, metric="euclidean")
        dist_matrix = squareform(distances)
        metric_values = np.median(dist_matrix, axis=1)

    else:  # total_intensity
        metric_values = np.nansum(X, axis=1)

    # Detect outliers based on method
    if method == "median_absolute_deviation":
        median = np.median(metric_values)
        mad = np.median(np.abs(metric_values - median))
        # Modified Z-score using MAD
        modified_z_scores = 0.6745 * (metric_values - median) / (mad + 1e-10)
        is_outlier = np.abs(modified_z_scores) > threshold

    elif method == "zscore":
        mean_val = np.mean(metric_values)
        std_val = np.std(metric_values)
        z_scores = (metric_values - mean_val) / (std_val + 1e-10)
        is_outlier = np.abs(z_scores) > threshold

    elif method == "iqr":
        q1 = np.percentile(metric_values, 25)
        q3 = np.percentile(metric_values, 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        is_outlier = (metric_values < lower_bound) | (metric_values > upper_bound)

    else:  # isolation
        from sklearn.ensemble import IsolationForest

        # Reshape for sklearn
        X_reshaped = metric_values.reshape(-1, 1)
        iso_forest = IsolationForest(
            contamination=min(0.1, threshold / 10),
            random_state=random_state,
            n_jobs=-1,
        )
        predictions = iso_forest.fit_predict(X_reshaped)
        is_outlier = predictions == -1

    # Add results to obs
    new_obs = container.obs.with_columns(
        pl.Series("is_correlation_outlier", is_outlier),
        pl.Series("outlier_metric_value", metric_values),
    )

    new_container = ScpContainer(
        obs=new_obs,
        assays=container.assays,
        links=list(container.links),
        history=list(container.history),
        sample_id_col=container.sample_id_col,
    )

    n_outliers = int(np.sum(is_outlier))
    new_container.log_operation(
        action="detect_outlier_samples",
        params={
            "assay": assay_name,
            "layer_name": layer_name,
            "method": method,
            "metric": metric,
            "threshold": threshold,
        },
        description=f"Detected {n_outliers} outlier samples using {method} on {metric}.",
    )

    return new_container


def compute_sample_similarity_network(
    container: ScpContainer,
    assay_name: str = "protein",
    layer_name: str = "raw",
    n_neighbors: int = 5,
    metric: str = "correlation",
) -> ScpContainer:
    """Compute a sample similarity network based on correlation or distance.

    This function creates a k-nearest neighbors graph where edges connect
    similar samples. Useful for visualization and detecting outliers.

    Parameters
    ----------
    container : ScpContainer
        Input container with data to analyze.
    assay_name : str, default "protein"
        Name of assay containing the layer.
    layer_name : str, default "raw"
        Name of layer to use for network construction.
    n_neighbors : int, default 5
        Number of nearest neighbors for each sample.
    metric : {"correlation", "euclidean", "cosine"}, default "correlation"
        Similarity metric.

    Returns
    -------
    ScpContainer
        Container with network information added to obs:
        - 'similarity_neighbors': List of neighbor indices for each sample
        - 'similarity_scores': List of similarity scores for each neighbor

    Raises
    ------
    AssayNotFoundError
        If assay_name does not exist.
    LayerNotFoundError
        If layer_name does not exist in the assay.

    Examples
    --------
    >>> container = compute_sample_similarity_network(
    ...     container, assay_name="protein", n_neighbors=5
    ... )
    >>> neighbors = container.obs['similarity_neighbors'][0]
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
        X = X.toarray()

    # Replace NaN with 0 for distance calculation
    X_clean = np.nan_to_num(X, nan=0.0)

    n_samples = X.shape[0]

    if n_neighbors >= n_samples:
        n_neighbors = n_samples - 1

    # Compute similarity matrix
    if metric == "correlation":
        # Use correlation as similarity (higher = more similar)
        from scipy.stats import spearmanr

        similarity_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i, n_samples):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    corr, _ = spearmanr(X_clean[i], X_clean[j])
                    similarity_matrix[i, j] = corr if not np.isnan(corr) else 0
                    similarity_matrix[j, i] = similarity_matrix[i, j]

    elif metric == "euclidean":
        # Convert distance to similarity (inverse)
        distances = pdist(X_clean, metric="euclidean")
        dist_matrix = squareform(distances)
        max_dist = np.max(dist_matrix[dist_matrix < np.inf])
        similarity_matrix = 1 - (dist_matrix / (max_dist + 1e-10))

    else:  # cosine
        distances = pdist(X_clean, metric="cosine")
        dist_matrix = squareform(distances)
        similarity_matrix = 1 - dist_matrix

    # Find k-nearest neighbors for each sample
    neighbors_list = []
    scores_list = []

    for i in range(n_samples):
        # Get similarities (exclude self)
        similarities = similarity_matrix[i].copy()
        similarities[i] = -1  # Exclude self

        # Get top k neighbors
        top_indices = np.argsort(similarities)[-n_neighbors:][::-1]
        top_scores = similarities[top_indices]

        neighbors_list.append(top_indices.tolist())
        scores_list.append(top_scores.tolist())

    # Add results to obs
    new_obs = container.obs.with_columns(
        pl.Series("similarity_neighbors", neighbors_list),
        pl.Series("similarity_scores", scores_list),
    )

    new_container = ScpContainer(
        obs=new_obs,
        assays=container.assays,
        links=list(container.links),
        history=list(container.history),
        sample_id_col=container.sample_id_col,
    )

    new_container.log_operation(
        action="compute_sample_similarity_network",
        params={
            "assay": assay_name,
            "layer_name": layer_name,
            "n_neighbors": n_neighbors,
            "metric": metric,
        },
        description=f"Computed similarity network with {n_neighbors} neighbors per sample.",
    )

    return new_container
