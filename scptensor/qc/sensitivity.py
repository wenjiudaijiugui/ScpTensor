"""Sensitivity metrics for single-cell proteomics data QC.

This module provides sensitivity-related quality control metrics:
- Total sensitivity: Total number of unique features detected across all samples
- Local sensitivity: Number of features detected per sample
- Completeness: Proportion of non-missing values
- Jaccard index: Feature overlap similarity between sample pairs
- Cumulative sensitivity: Feature saturation analysis

References
----------
Vanderaa, C., & Gatto, L. (2023). Revisiting the Thorny Issue of Missing
Values in Single-Cell Proteomics. arXiv:2304.06654
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform

from scptensor.core.exceptions import AssayNotFoundError, LayerNotFoundError, ScpValueError
from scptensor.core.structures import ScpContainer


@dataclass
class QCMetrics:
    """Quality control metrics summary.

    Attributes
    ----------
    n_features_per_sample : np.ndarray
        Number of detected features per sample.
    completeness_per_sample : np.ndarray
        Data completeness proportion (0-1) for each sample.
    total_features : int
        Total number of unique features in the dataset.
    mean_sensitivity : float
        Mean number of features detected per sample.
    estimated_total_sensitivity : int | None
        Estimated total features if more sampling were done.
    group_stats : dict[str, dict] | None
        Group-level statistics if group_by was provided.
    """

    n_features_per_sample: np.ndarray
    completeness_per_sample: np.ndarray
    total_features: int
    mean_sensitivity: float
    estimated_total_sensitivity: int | None = None
    group_stats: dict[str, dict] | None = None


@dataclass
class CumulativeSensitivityResult:
    """Result of cumulative sensitivity analysis.

    Attributes
    ----------
    sample_sizes : np.ndarray
        Number of samples at each step.
    cumulative_features : np.ndarray
        Cumulative unique features at each step.
    saturation_point : int | None
        Estimated sample size where feature saturation occurs.
    """

    sample_sizes: np.ndarray
    cumulative_features: np.ndarray
    saturation_point: int | None = None


def _get_layer(assay, layer_name: str | None = None):
    """Get a layer from assay, defaulting to 'raw' or first available layer.

    Args:
        assay: The Assay object.
        layer_name: Specific layer name to retrieve, or None for default.

    Returns
    -------
        The ScpMatrix layer.

    Raises
    ------
        LayerNotFoundError: If no layers exist in the assay.
    """
    from scptensor.core.exceptions import LayerNotFoundError

    if layer_name:
        if layer_name not in assay.layers:
            raise LayerNotFoundError(layer_name, "<assay>")
        return assay.layers[layer_name]

    return assay.layers.get("raw") or next(iter(assay.layers.values()), None)


def _get_detection_matrix(
    data: np.ndarray | sp.spmatrix,
    detection_threshold: float = 0.0,
) -> np.ndarray:
    """Convert data matrix to binary detection matrix.

    Args:
        data: Data matrix (dense or sparse).
        detection_threshold: Threshold for considering a value as detected.

    Returns
    -------
        Boolean matrix where True indicates detected feature.
    """
    if sp.issparse(data):
        if detection_threshold == 0.0:
            # For zero threshold, use non-zero elements
            return data > 0
        else:
            # Convert to dense for threshold comparison
            data_dense = data.toarray() if sp.issparse(data) else data  # type: ignore[union-attr]
            return data_dense > detection_threshold
    else:
        return detection_threshold < data


def compute_sensitivity(
    container: ScpContainer,
    assay_name: str = "protein",
    layer_name: str = "raw",
    detection_threshold: float = 0.0,
) -> QCMetrics:
    """Compute total and local sensitivity metrics.

    Total sensitivity is the total number of unique features detected across
    all samples. Local sensitivity is the number of features detected per sample.

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
    QCMetrics
        Metrics object containing sensitivity information.

    Raises
    ------
    AssayNotFoundError
        If assay_name does not exist.
    LayerNotFoundError
        If layer_name does not exist in the assay.

    Examples
    --------
    >>> metrics = compute_sensitivity(container, assay_name="protein")
    >>> print(f"Total features: {metrics.total_features}")
    >>> print(f"Mean features per sample: {metrics.mean_sensitivity}")
    """
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
    n_samples, n_features = data.shape

    # Get detection matrix
    detected = _get_detection_matrix(data, detection_threshold)

    # Compute local sensitivity (features per sample)
    if sp.issparse(detected):
        n_features_per_sample = np.array(detected.getnnz(axis=1)).flatten()  # type: ignore[attr-defined]
        # For sparse, use getnnz to count non-zeros per column
        any_detected = np.array(detected.getnnz(axis=0)).flatten() > 0  # type: ignore[attr-defined]
    else:
        n_features_per_sample = np.sum(detected, axis=1)
        any_detected = np.any(detected, axis=0)

    total_features = int(np.sum(any_detected))

    # Compute completeness (proportion of non-missing values)
    completeness_per_sample = n_features_per_sample / n_features

    return QCMetrics(
        n_features_per_sample=n_features_per_sample,
        completeness_per_sample=completeness_per_sample,
        total_features=total_features,
        mean_sensitivity=float(np.mean(n_features_per_sample)),
        estimated_total_sensitivity=None,
    )


def compute_completeness(
    container: ScpContainer,
    assay_name: str = "protein",
    layer_name: str = "raw",
    detection_threshold: float = 0.0,
) -> np.ndarray:
    """Compute data completeness for each sample.

    Completeness is the proportion of features that are detected (non-missing)
    for each sample.

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
    np.ndarray
        Completeness values [0, 1] for each sample.

    Raises
    ------
    AssayNotFoundError
        If assay_name does not exist.
    LayerNotFoundError
        If layer_name does not exist in the assay.

    Examples
    --------
    >>> completeness = compute_completeness(container, assay_name="protein")
    >>> mean_completeness = np.mean(completeness)
    """
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
    n_features = data.shape[1]

    # Get detection matrix
    detected = _get_detection_matrix(data, detection_threshold)

    # Compute completeness per sample
    if sp.issparse(detected):
        n_detected = np.array(detected.getnnz(axis=1)).flatten()  # type: ignore[attr-defined]
    else:
        n_detected = np.sum(detected, axis=1)

    return n_detected / n_features


def compute_jaccard_index(
    container: ScpContainer,
    assay_name: str = "protein",
    layer_name: str = "raw",
    detection_threshold: float = 0.0,
) -> np.ndarray:
    """Compute Jaccard index between all sample pairs.

    The Jaccard index measures feature overlap similarity between two samples:
        J(A, B) = |A intersect B| / |A union B|

    where A and B are the sets of detected features for each sample.

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
    np.ndarray
        Square matrix of Jaccard indices with shape (n_samples, n_samples).
        Values range from 0 (no overlap) to 1 (identical feature sets).

    Raises
    ------
    AssayNotFoundError
        If assay_name does not exist.
    LayerNotFoundError
        If layer_name does not exist in the assay.

    Examples
    --------
    >>> jaccard = compute_jaccard_index(container, assay_name="protein")
    >>> # Mean similarity between all sample pairs
    >>> mean_similarity = np.mean(jaccard[np.triu_indices_from(jaccard, k=1)])
    """
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

    # Get detection matrix as boolean
    detected = _get_detection_matrix(data, detection_threshold)

    # Convert to dense if sparse for efficient computation
    if sp.issparse(detected):
        detected = detected.toarray()  # type: ignore[attr-defined]

    # Compute Jaccard distance (1 - Jaccard similarity) using pdist
    # Jaccard distance = 1 - (intersection / union)
    jaccard_dist = pdist(detected, metric="jaccard")
    jaccard_sim = 1 - squareform(jaccard_dist)

    return jaccard_sim


def compute_cumulative_sensitivity(
    container: ScpContainer,
    assay_name: str = "protein",
    layer_name: str = "raw",
    detection_threshold: float = 0.0,
    batch_col: str | None = None,
    n_steps: int = 20,
    seed: int | None = None,
) -> CumulativeSensitivityResult:
    """Compute cumulative sensitivity curve for assessing feature saturation.

    This function computes the cumulative number of unique features detected
    as more samples are added. This helps assess whether the feature space
    has been adequately sampled or if more samples would discover new features.

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
    batch_col : str | None, default None
        If provided, compute cumulative sensitivity within each batch.
    n_steps : int, default 20
        Number of steps in the cumulative curve.
    seed : int | None, default None
        Random seed for sample ordering.

    Returns
    -------
    CumulativeSensitivityResult
        Object containing cumulative sensitivity curve and saturation estimate.

    Raises
    ------
    AssayNotFoundError
        If assay_name does not exist.
    LayerNotFoundError
        If layer_name does not exist in the assay.

    Examples
    --------
    >>> result = compute_cumulative_sensitivity(container, assay_name="protein")
    >>> # Check if saturation is reached
    >>> if result.saturation_point:
    ...     print(f"Saturation at {result.saturation_point} samples")
    >>> else:
    ...     print("Saturation not reached within sample size")
    """
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
    n_samples = data.shape[0]

    if n_steps < 2:
        raise ScpValueError(
            f"n_steps must be at least 2, got {n_steps}.",
            parameter="n_steps",
            value=n_steps,
        )

    if n_steps > n_samples:
        n_steps = n_samples

    # Get detection matrix
    detected = _get_detection_matrix(data, detection_threshold)

    # Convert to dense if needed
    if sp.issparse(detected):
        detected = detected.toarray()  # type: ignore[attr-defined]

    # Create sample indices
    rng = np.random.default_rng(seed)
    sample_order = rng.permutation(n_samples)

    # Generate step sizes
    step_sizes = np.linspace(1, n_samples, n_steps, dtype=int)

    # Compute cumulative features at each step
    cumulative_features_list: list[int] = []
    for step_size in step_sizes:
        selected_indices = sample_order[:step_size]
        selected_detected = detected[selected_indices, :]
        unique_features = np.any(selected_detected, axis=0)
        cumulative_features_list.append(int(np.sum(unique_features)))

    cumulative_features = np.array(cumulative_features_list, dtype=np.int64)

    # Estimate saturation point
    # Saturation is reached when the increase is less than 1% over previous steps
    saturation_point = None
    if len(cumulative_features) > 5:
        for i in range(5, len(cumulative_features)):
            increase = cumulative_features[i] - cumulative_features[i - 1]
            if increase == 0:
                saturation_point = int(step_sizes[i])
                break

    return CumulativeSensitivityResult(
        sample_sizes=step_sizes,
        cumulative_features=cumulative_features,
        saturation_point=saturation_point,
    )


def qc_report_metrics(
    container: ScpContainer,
    assay_name: str = "protein",
    layer_name: str = "raw",
    group_by: str | None = None,
    detection_threshold: float = 0.0,
) -> ScpContainer:
    """Compute core QC metrics and add to container.

    This is the main entry point for sensitivity QC analysis. It computes
    and adds several metrics to the container's obs DataFrame.

    Parameters
    ----------
    container : ScpContainer
        Input container with data to analyze.
    assay_name : str, default "protein"
        Name of assay containing the layer.
    layer_name : str, default "raw"
        Name of layer to use for calculation.
    group_by : str | None, default None
        Column name in obs to group samples by. If provided, group-level
        statistics are also computed.
    detection_threshold : float, default 0.0
        Value threshold for considering a value as detected.

    Returns
    -------
    ScpContainer
        New container with QC metrics added to obs:
        - 'n_detected_features': Number of features detected per sample
        - 'total_features': Total features in dataset
        - 'completeness': Data completeness proportion (0-1)
        - 'local_sensitivity': Local sensitivity (same as n_detected_features)

        If group_by is provided, also adds:
        - '{group_by}_mean_features': Mean features per group
        - '{group_by}_total_features': Total features per group

    Raises
    ------
    AssayNotFoundError
        If assay_name does not exist.
    LayerNotFoundError
        If layer_name does not exist in the assay.
    ScpValueError
        If group_by column does not exist in obs.

    Examples
    --------
    >>> container = qc_report_metrics(container, assay_name="protein")
    >>> print(container.obs['n_detected_features'])
    >>> # With grouping
    >>> container = qc_report_metrics(container, group_by="batch")
    """
    # Compute sensitivity metrics
    metrics = compute_sensitivity(
        container,
        assay_name=assay_name,
        layer_name=layer_name,
        detection_threshold=detection_threshold,
    )

    # Prepare new columns for obs
    new_columns: dict[str, Any] = {
        "n_detected_features": metrics.n_features_per_sample,
        "total_features": np.full(container.n_samples, metrics.total_features, dtype=object),
        "completeness": metrics.completeness_per_sample,
        "local_sensitivity": metrics.n_features_per_sample,
    }

    # Add group-level statistics if requested
    if group_by is not None:
        if group_by not in container.obs.columns:
            raise ScpValueError(
                f"Column '{group_by}' not found in obs. "
                f"Available columns: {list(container.obs.columns)}",
                parameter="group_by",
                value=group_by,
            )

        # Create temporary DataFrame with group info and computed metrics
        obs_dict = container.obs.to_pandas()
        temp_df = obs_dict.copy()
        temp_df["n_detected_features"] = metrics.n_features_per_sample

        # Compute group statistics
        group_stats = temp_df.groupby(group_by).agg(
            mean_features=("n_detected_features", "mean"),
            total_features=("n_detected_features", "sum"),
        )

        # Map group statistics back to samples
        group_means = obs_dict[group_by].map(group_stats["mean_features"]).to_numpy()
        group_totals = obs_dict[group_by].map(group_stats["total_features"]).to_numpy()

        new_columns[f"{group_by}_mean_features"] = group_means
        new_columns[f"{group_by}_total_features"] = group_totals

    # Create new obs DataFrame
    new_obs = container.obs
    for col_name, col_data in new_columns.items():
        new_obs = new_obs.with_columns(pl.Series(col_name, col_data))

    # Create new container
    new_container = ScpContainer(
        obs=new_obs,
        assays=container.assays,
        links=list(container.links),
        history=list(container.history),
        sample_id_col=container.sample_id_col,
    )

    # Log operation
    group_suffix = f" grouped by {group_by}" if group_by else ""
    new_container.log_operation(
        action="qc_report_metrics",
        params={
            "assay": assay_name,
            "layer": layer_name,
            "group_by": group_by,
            "detection_threshold": detection_threshold,
        },
        description=f"Computed QC metrics{group_suffix}: "
        f"total_features={metrics.total_features}, "
        f"mean_sensitivity={metrics.mean_sensitivity:.2f}, "
        f"mean_completeness={np.mean(metrics.completeness_per_sample):.3f}.",
    )

    return new_container


__all__ = [
    "compute_sensitivity",
    "compute_completeness",
    "compute_jaccard_index",
    "compute_cumulative_sensitivity",
    "qc_report_metrics",
    "QCMetrics",
    "CumulativeSensitivityResult",
]
