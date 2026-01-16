"""Variability statistics for single-cell proteomics data QC.

This module provides coefficient of variation (CV) related quality control metrics:
- CV computation: Feature-wise coefficient of variation
- Technical replicate CV: Assess experimental reproducibility
- Batch CV: Within-batch and between-batch variability
- CV filtering: Filter features by CV threshold

The coefficient of variation is a standardized measure of dispersion:
    CV = Standard Deviation / Mean

References
----------
Vanderaa, C., & Gatto, L. (2023). Revisiting the Thorny Issue of Missing
Values in Single-Cell Proteomics. arXiv:2304.06654
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
import scipy.sparse as sp

from scptensor.core.exceptions import AssayNotFoundError, LayerNotFoundError, ScpValueError
from scptensor.core.structures import Assay, ScpContainer


@dataclass
class CVReport:
    """Coefficient of variation report.

    Attributes
    ----------
    feature_cv : np.ndarray
        Coefficient of variation for each feature.
    mean_cv : float
        Mean CV across all features.
    median_cv : float
        Median CV across all features.
    cv_by_group : dict[str, np.ndarray] | None
        CV values computed per group (if group_by provided).
    within_batch_cv : dict[str, float] | None
        Within-batch CV for each batch (if batch_col provided).
    between_batch_cv : float | None
        Between-batch CV (if batch_col provided).
    high_cv_features : list | None
        List of feature indices with high CV (exceeding threshold).
    low_quality_samples : list | None
        List of sample indices with high missing rate.
    """

    feature_cv: np.ndarray
    mean_cv: float
    median_cv: float
    cv_by_group: dict[str, np.ndarray] | None = None
    within_batch_cv: dict[str, float] | None = None
    between_batch_cv: float | None = None
    high_cv_features: list | None = None
    low_quality_samples: list | None = None

    # Container reference (for accessing assays)
    _container: ScpContainer | None = None

    @property
    def assays(self):
        """Access to container's assays for backward compatibility.

        This property allows CVReport to be used similarly to ScpContainer
        in test code that expects .assays attribute.
        """
        if self._container is None:
            raise AttributeError("CVReport has no container reference")
        return self._container.assays

    @property
    def history(self):
        """Access to container's history."""
        if self._container is None:
            raise AttributeError("CVReport has no container reference")
        return self._container.history


def _compute_cv_matrix(
    data: np.ndarray | sp.spmatrix,
    axis: int = 0,
    min_mean: float = 1e-6,
) -> np.ndarray:
    """Compute coefficient of variation along specified axis.

    Args:
        data: Data matrix (dense or sparse).
        axis: Axis along which to compute CV (0 for features, 1 for samples).
        min_mean: Minimum mean value to avoid division by zero.

    Returns:
        Array of CV values.
    """
    if sp.issparse(data):
        if axis == 0:
            # Compute CV per feature (column)
            means = np.asarray(data.mean(axis=0)).flatten()
            # Compute E[X^2] for variance calculation
            data_squared = data.copy()
            if hasattr(data_squared, "power"):
                data_squared = data_squared.power(2)
            else:
                # Fallback for sparse matrices without power method
                data_squared = data.multiply(data)  # type: ignore[union-attr]
            mean_squares = np.asarray(data_squared.mean(axis=0)).flatten()
            stds = np.sqrt(np.maximum(0, mean_squares - means**2))
        else:
            # Compute CV per sample (row)
            means = np.asarray(data.mean(axis=1)).flatten()
            data_squared = data.copy()
            if hasattr(data_squared, "power"):
                data_squared = data_squared.power(2)
            else:
                data_squared = data.multiply(data)  # type: ignore[union-attr]
            mean_squares = np.asarray(data_squared.mean(axis=1)).flatten()
            stds = np.sqrt(np.maximum(0, mean_squares - means**2))
    else:
        means = np.mean(data, axis=axis)
        stds = np.std(data, axis=axis)

    # Avoid division by zero
    means_safe = np.where(np.abs(means) < min_mean, min_mean, means)

    return stds / means_safe


def compute_cv(
    container: ScpContainer,
    assay_name: str = "protein",
    layer_name: str = "raw",
    group_by: str | None = None,
    min_mean: float = 1e-6,
) -> CVReport:
    """Compute coefficient of variation for each feature.

    The coefficient of variation (CV) measures the relative variability of
    each feature across samples. Features with high CV may indicate poor
    measurement quality or biological heterogeneity.

    Parameters
    ----------
    container : ScpContainer
        Input container with data to analyze.
    assay_name : str, default "protein"
        Name of assay containing the layer.
    layer_name : str, default "raw"
        Name of layer to use for calculation.
    group_by : str | None, default None
        Column name in obs to group samples by. If provided, CV is
        computed separately for each group.
    min_mean : float, default 1e-6
        Minimum mean value to avoid division by zero.

    Returns
    -------
    CVReport
        Report containing CV statistics for features.

    Raises
    ------
    AssayNotFoundError
        If assay_name does not exist.
    LayerNotFoundError
        If layer_name does not exist in the assay.
    ScpValueError
        If group_by column does not exist in obs or min_mean is negative.

    Examples
    --------
    >>> report = compute_cv(container, assay_name="protein")
    >>> print(f"Mean CV: {report.mean_cv:.3f}")
    >>> # With grouping
    >>> report = compute_cv(container, group_by="batch")
    >>> for group, cvs in report.cv_by_group.items():
    ...     print(f"{group}: {np.mean(cvs):.3f}")
    """
    if min_mean < 0:
        raise ScpValueError(
            f"min_mean must be non-negative, got {min_mean}.",
            parameter="min_mean",
            value=min_mean,
        )

    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]
    if layer_name not in assay.layers:
        available = ", ".join(f"'{k}'" for k in assay.layers)
        raise LayerNotFoundError(
            layer_name,
            assay_name,
            hint=f"Layer '{layer_name}' not found in assay '{assay_name}'. "
            f"Available layers: {available}.",
        )

    data = assay.layers[layer_name].X

    if group_by is None or group_by not in container.obs.columns:
        if group_by is not None:
            raise ScpValueError(
                f"Column '{group_by}' not found in obs. "
                f"Available columns: {list(container.obs.columns)}",
                parameter="group_by",
                value=group_by,
            )
        # Compute CV across all samples
        feature_cv = _compute_cv_matrix(data, axis=0, min_mean=min_mean)
        cv_by_group = None
    else:
        # Compute CV per group
        groups = container.obs[group_by].to_numpy()
        unique_groups = np.unique(groups)

        cv_by_group = {}
        group_cv_arrays = []

        for group in unique_groups:
            group_mask = groups == group
            group_data = data[group_mask, :] if sp.issparse(data) else data[group_mask, :]

            group_cv = _compute_cv_matrix(group_data, axis=0, min_mean=min_mean)
            cv_by_group[str(group)] = group_cv
            group_cv_arrays.append(group_cv)

        # Use overall mean CV as the main feature_cv
        feature_cv = np.mean(group_cv_arrays, axis=0)

    # Compute summary statistics
    mean_cv = float(np.mean(feature_cv))
    median_cv = float(np.median(feature_cv))

    # Add cv column to var
    new_var = assay.var.with_columns(pl.Series("cv", feature_cv))

    # Create new assay with updated var
    new_assay = Assay(
        var=new_var,
        layers=assay.layers,
        feature_id_col=assay.feature_id_col,
    )

    # Create new container
    new_container = ScpContainer(
        obs=container.obs,
        assays={**container.assays, assay_name: new_assay},
        links=list(container.links),
        history=list(container.history),
        sample_id_col=container.sample_id_col,
    )

    # Log operation
    group_suffix = f" grouped by {group_by}" if group_by else ""
    new_container.log_operation(
        action="compute_cv",
        params={
            "assay": assay_name,
            "layer": layer_name,
            "group_by": group_by,
            "min_mean": min_mean,
        },
        description=f"Computed CV for {len(feature_cv)} features{group_suffix}. "
        f"Mean CV: {mean_cv:.3f}, Median CV: {median_cv:.3f}.",
    )

    return CVReport(
        feature_cv=feature_cv,
        mean_cv=mean_cv,
        median_cv=median_cv,
        cv_by_group=cv_by_group,
        _container=new_container,
    )


def compute_technical_replicate_cv(
    container: ScpContainer,
    assay_name: str = "protein",
    layer_name: str = "raw",
    replicate_col: str = "replicate",
    aggregate: Literal["mean", "median", "max"] = "mean",
    min_mean: float = 1e-6,
) -> CVReport:
    """Compute CV within technical replicate groups.

    This function assesses experimental reproducibility by computing the CV
    within groups of technical replicates. Low CV within replicates indicates
    good measurement reproducibility.

    Parameters
    ----------
    container : ScpContainer
        Input container with data to analyze.
    assay_name : str, default "protein"
        Name of assay containing the layer.
    layer_name : str, default "raw"
        Name of layer to use for calculation.
    replicate_col : str, default "replicate"
        Column name in obs identifying technical replicate groups.
    aggregate : {"mean", "median", "max"}, default "mean"
        Method for aggregating CV across replicates.
    min_mean : float, default 1e-6
        Minimum mean value to avoid division by zero.

    Returns
    -------
    CVReport
        Report containing CV statistics for technical replicates.
        Adds the following columns to var:
        - 'tech_rep_cv': Aggregated CV value per feature
        - 'tech_rep_cv_std': Standard deviation of CV across replicates
        - 'tech_rep_n_groups': Number of replicate groups

    Raises
    ------
    AssayNotFoundError
        If assay_name does not exist.
    LayerNotFoundError
        If layer_name does not exist in the assay.
    ScpValueError
        If replicate_col does not exist in obs or aggregate is invalid.

    Examples
    --------
    >>> report = compute_technical_replicate_cv(
    ...     container, replicate_col="replicate_id"
    ... )
    >>> print(f"Mean CV: {report.mean_cv:.3f}")
    >>> # With median aggregation
    >>> report = compute_technical_replicate_cv(
    ...     container, replicate_col="replicate_id", aggregate="median"
    ... )
    """
    # Validate aggregate parameter
    valid_aggregates = ["mean", "median", "max"]
    if aggregate not in valid_aggregates:
        raise ScpValueError(
            f"aggregate must be one of {valid_aggregates}, got '{aggregate}'.",
            parameter="aggregate",
            value=aggregate,
        )

    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]
    if layer_name not in assay.layers:
        available = ", ".join(f"'{k}'" for k in assay.layers)
        raise LayerNotFoundError(
            layer_name,
            assay_name,
            hint=f"Layer '{layer_name}' not found in assay '{assay_name}'. "
            f"Available layers: {available}.",
        )

    if replicate_col not in container.obs.columns:
        raise ScpValueError(
            f"Column '{replicate_col}' not found in obs. "
            f"Available columns: {list(container.obs.columns)}",
            parameter="replicate_col",
            value=replicate_col,
        )

    data = assay.layers[layer_name].X
    replicates = container.obs[replicate_col].to_numpy()
    unique_replicates = np.unique(replicates)
    n_features = data.shape[1]

    # Compute CV for each replicate group
    cv_by_group = {}
    replicate_cv_arrays = []

    for rep_id in unique_replicates:
        rep_mask = replicates == rep_id
        rep_data = data[rep_mask, :] if sp.issparse(data) else data[rep_mask, :]

        # Only compute CV if we have multiple samples in the replicate
        if np.sum(rep_mask) > 1:
            rep_cv = _compute_cv_matrix(rep_data, axis=0, min_mean=min_mean)
            cv_by_group[str(rep_id)] = rep_cv
            replicate_cv_arrays.append(rep_cv)
        else:
            # Single sample: CV is 0
            rep_cv = np.zeros(n_features)
            cv_by_group[str(rep_id)] = rep_cv
            replicate_cv_arrays.append(rep_cv)

    # Stack replicate CVs for aggregation
    cv_matrix = np.array(replicate_cv_arrays)  # shape: (n_replicates, n_features)

    # Compute aggregated CV based on aggregate parameter
    if aggregate == "mean":
        feature_cv = np.mean(cv_matrix, axis=0)
    elif aggregate == "median":
        feature_cv = np.median(cv_matrix, axis=0)
    else:  # aggregate == "max"
        feature_cv = np.max(cv_matrix, axis=0)

    # Compute statistics
    mean_cv = float(np.mean(feature_cv))
    median_cv = float(np.median(feature_cv))

    # Compute per-feature standard deviation across replicates
    tech_rep_cv_std = np.std(cv_matrix, axis=0)

    # Add columns to var (include replicate_cv for backward compatibility)
    new_var = assay.var.with_columns(
        [
            pl.Series("tech_rep_cv", feature_cv),
            pl.Series("replicate_cv", feature_cv),  # Alias for backward compatibility
            pl.Series("tech_rep_cv_std", tech_rep_cv_std),
            pl.Series("tech_rep_n_groups", [len(unique_replicates)] * n_features),
        ]
    )

    # Create new assay with updated var
    new_assay = Assay(
        var=new_var,
        layers=assay.layers,
        feature_id_col=assay.feature_id_col,
    )

    # Create new container
    new_container = ScpContainer(
        obs=container.obs,
        assays={**container.assays, assay_name: new_assay},
        links=list(container.links),
        history=list(container.history),
        sample_id_col=container.sample_id_col,
    )

    # Log operation
    new_container.log_operation(
        action="compute_technical_replicate_cv",
        params={
            "assay": assay_name,
            "layer": layer_name,
            "replicate_col": replicate_col,
            "aggregate": aggregate,
            "min_mean": min_mean,
        },
        description=f"Computed technical replicate CV for {len(unique_replicates)} groups. "
        f"Aggregation: {aggregate}, Mean CV: {mean_cv:.3f}.",
    )

    return CVReport(
        feature_cv=feature_cv,
        mean_cv=mean_cv,
        median_cv=median_cv,
        cv_by_group=cv_by_group,
        _container=new_container,
    )


def compute_batch_cv(
    container: ScpContainer,
    assay_name: str = "protein",
    layer_name: str = "raw",
    batch_col: str = "batch",
    min_mean: float = 1e-6,
    high_cv_threshold: float = 0.5,
) -> CVReport:
    """Compute within-batch and between-batch CV for batch effect assessment.

    This function computes CV statistics to assess batch effects:
    - Within-batch CV: Variability of features within each batch
    - Between-batch CV: Variability of batch means across batches

    Parameters
    ----------
    container : ScpContainer
        Input container with data to analyze.
    assay_name : str, default "protein"
        Name of assay containing the layer.
    layer_name : str, default "raw"
        Name of layer to use for calculation.
    batch_col : str, default "batch"
        Column name in obs identifying batch groups.
    min_mean : float, default 1e-6
        Minimum mean value to avoid division by zero.
    high_cv_threshold : float, default 0.5
        Threshold for identifying high CV features. Features with CV above
        this threshold are added to high_cv_features list.

    Returns
    -------
    CVReport
        Report containing within-batch and between-batch CV statistics.
        Adds the following columns to var:
        - 'within_batch_cv': Within-batch CV per feature
        - 'between_batch_cv': Between-batch CV per feature

    Raises
    ------
    AssayNotFoundError
        If assay_name does not exist.
    LayerNotFoundError
        If layer_name does not exist in the assay.
    ScpValueError
        If batch_col does not exist in obs or high_cv_threshold is negative.

    Examples
    --------
    >>> report = compute_batch_cv(container, batch_col="batch")
    >>> print(f"Within-batch CV: {report.within_batch_cv}")
    >>> print(f"Between-batch CV: {report.between_batch_cv:.3f}")
    >>> # With custom threshold
    >>> report = compute_batch_cv(container, batch_col="batch", high_cv_threshold=0.3)
    """
    if high_cv_threshold < 0:
        raise ScpValueError(
            f"high_cv_threshold must be non-negative, got {high_cv_threshold}.",
            parameter="high_cv_threshold",
            value=high_cv_threshold,
        )

    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]
    if layer_name not in assay.layers:
        available = ", ".join(f"'{k}'" for k in assay.layers)
        raise LayerNotFoundError(
            layer_name,
            assay_name,
            hint=f"Layer '{layer_name}' not found in assay '{assay_name}'. "
            f"Available layers: {available}.",
        )

    if batch_col not in container.obs.columns:
        raise ScpValueError(
            f"Column '{batch_col}' not found in obs. "
            f"Available columns: {list(container.obs.columns)}",
            parameter="batch_col",
            value=batch_col,
        )

    data = assay.layers[layer_name].X
    batches = container.obs[batch_col].to_numpy()
    unique_batches = np.unique(batches)
    n_features = data.shape[1]

    # Compute within-batch CV per feature
    within_batch_cv_arrays = []
    batch_means = []

    for batch_id in unique_batches:
        batch_mask = batches == batch_id
        batch_data = data[batch_mask, :] if sp.issparse(data) else data[batch_mask, :]

        # Compute mean for this batch (for between-batch CV)
        if sp.issparse(batch_data):
            batch_mean = np.asarray(batch_data.mean(axis=0)).flatten()
        else:
            batch_mean = np.mean(batch_data, axis=0)
        batch_means.append(batch_mean)

        # Compute within-batch CV per feature
        if np.sum(batch_mask) > 1:
            batch_cv = _compute_cv_matrix(batch_data, axis=0, min_mean=min_mean)
            within_batch_cv_arrays.append(batch_cv)
        else:
            within_batch_cv_arrays.append(np.zeros(n_features))

    # Stack within-batch CVs
    within_batch_cv_matrix = np.array(within_batch_cv_arrays)

    # Compute mean within-batch CV per feature
    within_batch_cv_per_feature = np.mean(within_batch_cv_matrix, axis=0)

    # Compute summary within-batch CV (mean of means)
    within_batch_cv_summary = {
        str(batch_id): float(np.mean(within_batch_cv_arrays[i]))
        for i, batch_id in enumerate(unique_batches)
    }

    # Compute between-batch CV per feature (CV of batch means)
    batch_means_array = np.array(batch_means)
    between_batch_cv_per_feature = _compute_cv_matrix(batch_means_array, axis=0, min_mean=min_mean)
    between_batch_cv = float(np.mean(between_batch_cv_per_feature))

    # Compute overall feature CV
    feature_cv = _compute_cv_matrix(data, axis=0, min_mean=min_mean)

    # Identify high CV features
    high_cv_features = np.where(feature_cv > high_cv_threshold)[0].tolist()

    # Add columns to var
    new_var = assay.var.with_columns(
        [
            pl.Series("within_batch_cv", within_batch_cv_per_feature),
            pl.Series("between_batch_cv", between_batch_cv_per_feature),
        ]
    )

    # Create new assay with updated var
    new_assay = Assay(
        var=new_var,
        layers=assay.layers,
        feature_id_col=assay.feature_id_col,
    )

    # Create new container
    new_container = ScpContainer(
        obs=container.obs,
        assays={**container.assays, assay_name: new_assay},
        links=list(container.links),
        history=list(container.history),
        sample_id_col=container.sample_id_col,
    )

    # Log operation
    new_container.log_operation(
        action="compute_batch_cv",
        params={
            "assay": assay_name,
            "layer": layer_name,
            "batch_col": batch_col,
            "min_mean": min_mean,
            "high_cv_threshold": high_cv_threshold,
        },
        description=f"Computed batch CV for {len(unique_batches)} batches. "
        f"Between-batch CV: {between_batch_cv:.3f}, "
        f"High CV features: {len(high_cv_features)}.",
    )

    return CVReport(
        feature_cv=feature_cv,
        mean_cv=float(np.mean(feature_cv)),
        median_cv=float(np.median(feature_cv)),
        within_batch_cv=within_batch_cv_summary,
        between_batch_cv=between_batch_cv,
        high_cv_features=high_cv_features,
        _container=new_container,
    )


def filter_by_cv(
    container: ScpContainer | CVReport,
    assay_name: str = "protein",
    layer_name: str = "raw",
    cv_threshold: float = 0.3,
    keep_filtered: bool = False,
    max_cv: float | None = None,
    min_mean: float = 1e-6,
) -> ScpContainer:
    """Filter features by coefficient of variation threshold.

    Features with CV above the threshold are filtered. A typical CV threshold
    of 0.3 (30%) is used for quality proteomics data.

    Parameters
    ----------
    container : ScpContainer | CVReport
        Input container with data to analyze, or a CVReport from compute_cv.
    assay_name : str, default "protein"
        Name of assay containing the layer.
    layer_name : str, default "raw"
        Name of layer to use for calculation.
    cv_threshold : float, default 0.3
        Maximum allowed CV. Features with CV > threshold are filtered.
    keep_filtered : bool, default False
        If True, keep all features but add 'cv_filtered' column to var
        indicating which features were filtered. If False, remove
        high CV features from the assay.
    max_cv : float | None, default None
        Alias for cv_threshold. If provided, overrides cv_threshold.
    min_mean : float, default 1e-6
        Minimum mean value to avoid division by zero.

    Returns
    -------
    ScpContainer
        Container with filtered features. If keep_filtered=True,
        all features are retained with a 'cv_filtered' column added.
        If keep_filtered=False, high CV features are removed.

    Raises
    ------
    AssayNotFoundError
        If assay_name does not exist.
    LayerNotFoundError
        If layer_name does not exist in the assay.
    ScpValueError
        If cv_threshold is negative.

    Examples
    --------
    >>> # Remove high CV features
    >>> container = filter_by_cv(container, cv_threshold=0.3)
    >>> print(f"Remaining features: {container.assays['protein'].n_features}")
    >>> # Keep all features but mark filtered ones
    >>> container = filter_by_cv(container, cv_threshold=0.3, keep_filtered=True)
    >>> print(container.assays['protein'].var['cv_filtered'])
    """
    # Use max_cv if provided
    threshold = max_cv if max_cv is not None else cv_threshold

    if threshold < 0:
        raise ScpValueError(
            f"cv_threshold must be non-negative, got {threshold}.",
            parameter="cv_threshold",
            value=threshold,
        )

    # Check if container is a CVReport
    if isinstance(container, CVReport):
        report = container
        working_container = report._container
        if working_container is None:
            raise ScpValueError(
                "CVReport has no container reference. Pass a ScpContainer instead.",
                parameter="container",
                value=container,
            )
    else:
        # Compute CV
        report = compute_cv(
            container,
            assay_name=assay_name,
            layer_name=layer_name,
            min_mean=min_mean,
        )
        working_container = report._container

    # Determine features to keep (passing CV threshold)
    feature_mask = report.feature_cv <= threshold

    if keep_filtered:
        # Keep all features, just mark them
        new_var = working_container.assays[assay_name].var.with_columns(  # type: ignore[union-attr]
            pl.Series("cv_filtered", ~feature_mask)
        )
        new_assay = Assay(
            var=new_var,
            layers=working_container.assays[assay_name].layers,  # type: ignore[union-attr]
            feature_id_col=working_container.assays[assay_name].feature_id_col,  # type: ignore[union-attr]
        )
        new_container = ScpContainer(
            obs=working_container.obs,  # type: ignore[union-attr]
            assays={**working_container.assays, assay_name: new_assay},  # type: ignore[union-attr]
            links=list(working_container.links),  # type: ignore[union-attr]
            history=list(working_container.history),  # type: ignore[union-attr]
            sample_id_col=working_container.sample_id_col,  # type: ignore[union-attr]
        )

        # Log operation
        n_passed = np.sum(feature_mask)
        n_filtered = np.sum(~feature_mask)
        new_container.log_operation(
            action="filter_by_cv",
            params={
                "assay": assay_name,
                "layer": layer_name,
                "cv_threshold": threshold,
                "keep_filtered": True,
            },
            description=f"Marked {n_filtered} features with CV > {threshold} as filtered. "
            f"{n_passed} features passed CV filter.",
        )

        return new_container
    else:
        # Remove high CV features
        feature_indices = np.where(feature_mask)[0]

        # Use container.filter_features method
        new_container = working_container.filter_features(  # type: ignore[union-attr]
            assay_name=assay_name,
            feature_indices=feature_indices,
        )

        # Log operation
        n_passed = np.sum(feature_mask)
        n_filtered = np.sum(~feature_mask)
        new_container.log_operation(
            action="filter_by_cv",
            params={
                "assay": assay_name,
                "layer": layer_name,
                "cv_threshold": threshold,
                "keep_filtered": False,
            },
            description=f"Filtered {n_filtered} features with CV > {threshold}. "
            f"{n_passed} features passed CV filter.",
        )

        return new_container


__all__ = [
    "compute_cv",
    "compute_technical_replicate_cv",
    "compute_batch_cv",
    "filter_by_cv",
    "CVReport",
]
