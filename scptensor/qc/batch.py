"""Batch effect detection and QC metrics for single-cell proteomics data.

This module provides methods for detecting and quantifying batch effects
in single-cell proteomics data, which is critical for ensuring data quality
before downstream analysis.
"""

from typing import Literal

import numpy as np
import polars as pl
import scipy.sparse as sp
from scipy.stats import f_oneway, kruskal

from scptensor.core.exceptions import AssayNotFoundError, LayerNotFoundError, ScpValueError
from scptensor.core.structures import ScpContainer


def compute_batch_metrics(
    container: ScpContainer,
    assay_name: str = "protein",
    layer_name: str = "raw",
    batch_col: str | None = None,
    detection_threshold: float = 0.0,
) -> ScpContainer:
    """Compute batch-level QC metrics for each batch in the dataset.

    This function calculates various quality metrics for each batch,
    including total intensity, missing rate, and feature prevalence.

    Parameters
    ----------
    container : ScpContainer
        Input container with data to analyze.
    assay_name : str, default "protein"
        Name of assay containing the layer.
    layer_name : str, default "raw"
        Name of layer to use for metric calculation.
    batch_col : str | None, default None
        Column name in obs containing batch information.
        If None, looks for 'batch' column.
    detection_threshold : float, default 0.0
        Value threshold for considering a value as detected.

    Returns
    -------
    ScpContainer
        Container with batch metrics added to obs as 'batch_metrics' dict
        and per-sample batch alignment metrics.

    Raises
    ------
    AssayNotFoundError
        If assay_name does not exist.
    LayerNotFoundError
        If layer_name does not exist in the assay.
    ScpValueError
        If batch column is not found.

    Examples
    --------
    >>> container = compute_batch_metrics(
    ...     container, assay_name="protein", batch_col="batch"
    ... )
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

    # Determine batch column
    batch_column = batch_col or "batch"

    if batch_column not in container.obs.columns:
        raise ScpValueError(
            f"Batch column '{batch_column}' not found in obs. "
            f"Available columns: {container.obs.columns}",
            parameter="batch_col",
            value=batch_column,
        )

    X = assay.layers[layer_name].X

    # Handle sparse matrices
    if sp.issparse(X):
        X = X.toarray()

    # Get batch assignments
    batches = container.obs[batch_column].to_numpy()
    unique_batches = np.unique(batches)

    # Calculate per-sample metrics
    n_features = X.shape[1]
    detected_mask = X > detection_threshold

    n_detected_per_sample = detected_mask.sum(axis=1)
    total_intensity_per_sample = X.sum(axis=1)
    missing_rate_per_sample = 1.0 - (n_detected_per_sample / n_features)

    # Calculate batch-level statistics
    batch_stats = {}
    for batch in unique_batches:
        batch_mask = batches == batch
        batch_samples = np.where(batch_mask)[0]

        batch_stats[str(batch)] = {
            "n_samples": int(batch_mask.sum()),
            "mean_n_detected": float(n_detected_per_sample[batch_mask].mean()),
            "std_n_detected": float(n_detected_per_sample[batch_mask].std()),
            "mean_total_intensity": float(total_intensity_per_sample[batch_mask].mean()),
            "std_total_intensity": float(total_intensity_per_sample[batch_mask].std()),
            "mean_missing_rate": float(missing_rate_per_sample[batch_mask].mean()),
        }

    # Compute batch alignment score (how well each sample aligns with its batch)
    batch_alignment_scores = np.zeros(X.shape[0])
    for batch in unique_batches:
        batch_mask = batches == batch
        batch_indices = np.where(batch_mask)[0]

        if len(batch_indices) > 1:
            # Use median of batch as reference
            batch_median = np.median(X[batch_indices], axis=0)
            for idx in batch_indices:
                # Correlation with batch median
                mask = ~(np.isnan(X[idx]) | np.isnan(batch_median))
                if mask.sum() > 3:
                    from scipy.stats import spearmanr

                    corr, _ = spearmanr(X[idx][mask], batch_median[mask])
                    batch_alignment_scores[idx] = corr if not np.isnan(corr) else 0

    # Add results to obs
    new_obs = container.obs.with_columns(
        pl.Series("batch_n_detected", n_detected_per_sample),
        pl.Series("batch_total_intensity", total_intensity_per_sample),
        pl.Series("batch_missing_rate", missing_rate_per_sample),
        pl.Series("batch_alignment_score", batch_alignment_scores),
        pl.Series("batch_id", batches),
    )

    new_container = ScpContainer(
        obs=new_obs,
        assays=container.assays,
        links=list(container.links),
        history=list(container.history),
        sample_id_col=container.sample_id_col,
    )

    new_container.log_operation(
        action="compute_batch_metrics",
        params={
            "assay": assay_name,
            "layer_name": layer_name,
            "batch_col": batch_column,
        },
        description=f"Computed batch metrics for {len(unique_batches)} batches.",
    )

    return new_container


def detect_batch_effects(
    container: ScpContainer,
    assay_name: str = "protein",
    layer_name: str = "raw",
    batch_col: str | None = None,
    test: Literal["anova", "kruskal"] = "kruskal",
    n_features_max: int = 100,
    detection_threshold: float = 0.0,
    random_state: int = 42,
) -> ScpContainer:
    """Detect batch effects using statistical tests on feature intensities.

    This function tests whether there are significant differences in feature
    intensities between batches using ANOVA or Kruskal-Wallis tests.

    Parameters
    ----------
    container : ScpContainer
        Input container with data to analyze.
    assay_name : str, default "protein"
        Name of assay containing the layer.
    layer_name : str, default "raw"
        Name of layer to use for detection.
    batch_col : str | None, default None
        Column name in obs containing batch information.
    test : {"anova", "kruskal"}, default "kruskal"
        Statistical test to use.
    n_features_max : int, default 100
        Maximum number of features to test (most variable).
    detection_threshold : float, default 0.0
        Value threshold for considering a value as detected.
        random_state : int, default 42
        Random state for feature selection.

    Returns
    -------
    ScpContainer
        Container with batch effect detection results:
        - 'batch_effect_detected': Boolean indicating if batch effect was detected
        - 'batch_effect_p_values': Array of p-values for tested features
        - 'batch_effect_score': Overall batch effect score (0-1)

    Raises
    ------
    AssayNotFoundError
        If assay_name does not exist.
    LayerNotFoundError
        If layer_name does not exist in the assay.
    ScpValueError
        If batch column is not found or test is invalid.

    Examples
    --------
    >>> container = detect_batch_effects(
    ...     container, assay_name="protein", batch_col="batch"
    ... )
    >>> has_batch_effect = container.obs['batch_effect_detected'][0]
    """
    if test not in ("anova", "kruskal"):
        raise ScpValueError(
            f"test must be 'anova' or 'kruskal', got '{test}'.",
            parameter="test",
            value=test,
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

    # Determine batch column
    batch_column = batch_col or "batch"

    if batch_column not in container.obs.columns:
        raise ScpValueError(
            f"Batch column '{batch_column}' not found in obs. "
            f"Available columns: {container.obs.columns}",
            parameter="batch_col",
            value=batch_column,
        )

    X = assay.layers[layer_name].X

    # Handle sparse matrices
    if sp.issparse(X):
        X = X.toarray()

    batches = container.obs[batch_column].to_numpy()
    unique_batches = np.unique(batches)

    if len(unique_batches) < 2:
        # Only one batch, no batch effect possible
        new_obs = container.obs.with_columns(
            pl.Series("batch_effect_detected", [False]),
            pl.Series("batch_effect_score", [0.0]),
            pl.Series("batch_effect_n_significant", [0]),
        )

        new_container = ScpContainer(
            obs=new_obs,
            assays=container.assays,
            links=list(container.links),
            history=list(container.history),
            sample_id_col=container.sample_id_col,
        )

        new_container.log_operation(
            action="detect_batch_effects",
            params={"assay": assay_name, "layer_name": layer_name, "test": test},
            description="Only one batch found, no batch effect detected.",
        )

        return new_container

    # Replace values below threshold with NaN
    X_clean = X.copy()
    X_clean[X_clean <= detection_threshold] = np.nan

    # Select most variable features
    feature_variances = np.nanvar(X_clean, axis=0)
    n_features_to_test = min(n_features_max, X_clean.shape[1])

    # Handle NaN variances (features with all missing)
    valid_variance_indices = np.where(~np.isnan(feature_variances))[0]

    if len(valid_variance_indices) == 0:
        # All features have zero variance
        selected_indices = np.arange(n_features_to_test)
    else:
        top_variance_indices = valid_variance_indices[
            np.argsort(feature_variances[valid_variance_indices])[-n_features_to_test:]
        ]
        selected_indices = top_variance_indices

    # Run statistical tests
    p_values = []
    n_significant = 0

    for feat_idx in selected_indices:
        feature_data = X_clean[:, feat_idx]

        # Get data for each batch
        batch_data = []
        for batch in unique_batches:
            batch_mask = batches == batch
            batch_values = feature_data[batch_mask]
            # Remove NaN values
            batch_values = batch_values[~np.isnan(batch_values)]
            if len(batch_values) >= 2:  # Need at least 2 samples per batch
                batch_data.append(batch_values)

        if len(batch_data) >= 2 and all(len(d) >= 2 for d in batch_data):
            try:
                if test == "anova":
                    stat, p_value = f_oneway(*batch_data)
                else:  # kruskal
                    stat, p_value = kruskal(*batch_data)

                if not np.isnan(p_value):
                    p_values.append(p_value)
                    if p_value < 0.05:
                        n_significant += 1
            except Exception:
                pass

    # Calculate batch effect score
    if p_values:
        # Proportion of features with significant batch effect
        batch_effect_score = n_significant / len(p_values)
        batch_effect_detected = n_significant > len(p_values) * 0.1  # More than 10% significant
    else:
        batch_effect_score = 0.0
        batch_effect_detected = False

    # Add results to obs (same value for all samples in a container)
    n_samples = container.n_samples
    new_obs = container.obs.with_columns(
        pl.Series("batch_effect_detected", [batch_effect_detected] * n_samples),
        pl.Series("batch_effect_score", [batch_effect_score] * n_samples),
        pl.Series("batch_effect_n_significant", [n_significant] * n_samples),
        pl.Series("batch_effect_n_tested", [len(p_values)] * n_samples),
    )

    new_container = ScpContainer(
        obs=new_obs,
        assays=container.assays,
        links=list(container.links),
        history=list(container.history),
        sample_id_col=container.sample_id_col,
    )

    new_container.log_operation(
        action="detect_batch_effects",
        params={
            "assay": assay_name,
            "layer_name": layer_name,
            "batch_col": batch_column,
            "test": test,
        },
        description=f"Batch effect detection: {n_significant}/{len(p_values)} features significant. "
        f"Score: {batch_effect_score:.3f}",
    )

    return new_container


def compute_batch_pca(
    container: ScpContainer,
    assay_name: str = "protein",
    layer_name: str = "raw",
    batch_col: str | None = None,
    n_components: int = 10,
    detection_threshold: float = 0.0,
    random_state: int = 42,
) -> ScpContainer:
    """Compute PCA colored by batch to visualize batch effects.

    This function performs PCA on the data and returns coordinates
    that can be used to visualize batch effects.

    Parameters
    ----------
    container : ScpContainer
        Input container with data to analyze.
    assay_name : str, default "protein"
        Name of assay containing the layer.
    layer_name : str, default "raw"
        Name of layer to use for PCA.
    batch_col : str | None, default None
        Column name in obs containing batch information.
    n_components : int, default 10
        Number of PCA components to compute.
    detection_threshold : float, default 0.0
        Value threshold for considering a value as detected.
    random_state : int, default 42
        Random state for PCA.

    Returns
    -------
    ScpContainer
        Container with PCA coordinates added to obs:
        - 'batch_pc1', 'batch_pc2', etc.: PCA component values
        - 'batch_pca_explained_variance': Explained variance ratio

    Raises
    ------
    AssayNotFoundError
        If assay_name does not exist.
    LayerNotFoundError
        If layer_name does not exist in the assay.
    ScpValueError
        If batch column is not found.

    Examples
    --------
    >>> container = compute_batch_pca(
    ...     container, assay_name="protein", n_components=2
    ... )
    >>> pc1 = container.obs['batch_pc1'].to_numpy()
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

    # Determine batch column
    batch_column = batch_col or "batch"

    if batch_column not in container.obs.columns:
        raise ScpValueError(
            f"Batch column '{batch_column}' not found in obs. "
            f"Available columns: {container.obs.columns}",
            parameter="batch_col",
            value=batch_column,
        )

    X = assay.layers[layer_name].X

    # Handle sparse matrices
    if sp.issparse(X):
        X = X.toarray()

    # Impute missing values for PCA
    X_clean = X.copy()
    X_clean[X_clean <= detection_threshold] = 0

    # Perform PCA
    from sklearn.decomposition import PCA

    n_components_actual = min(n_components, X_clean.shape[0], X_clean.shape[1])

    pca = PCA(n_components=n_components_actual, random_state=random_state)
    pca_result = pca.fit_transform(X_clean)

    # Add PCA coordinates to obs
    obs_data = {}
    for i in range(n_components_actual):
        obs_data[f"batch_pc{i + 1}"] = pca_result[:, i]

    # Add explained variance
    explained_variance = pca.explained_variance_ratio_.tolist()

    new_obs = container.obs.with_columns(
        **{k: pl.Series(v) for k, v in obs_data.items()}
    )

    new_container = ScpContainer(
        obs=new_obs,
        assays=container.assays,
        links=list(container.links),
        history=list(container.history),
        sample_id_col=container.sample_id_col,
    )

    new_container.log_operation(
        action="compute_batch_pca",
        params={
            "assay": assay_name,
            "layer_name": layer_name,
            "batch_col": batch_column,
            "n_components": n_components_actual,
        },
        description=f"Computed batch PCA with {n_components_actual} components. "
        f"PC1 explains {explained_variance[0]*100:.1f}% of variance.",
    )

    return new_container
