"""
Advanced Quality Control methods for single-cell proteomics data.

This module provides advanced QC operations including:
- Feature filtering: Filter features based on missing rate, variance, prevalence
- Sample filtering: Filter samples based on various quality metrics
- Contaminant detection: Detect potential contaminant proteins
- Doublet detection: Detect potential multiplets/doublets

All methods follow the functional pattern - they return new containers/layers
rather than modifying in-place.
"""

import re
from typing import Literal

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

from scptensor.core.exceptions import AssayNotFoundError, LayerNotFoundError, ScpValueError
from scptensor.core.structures import ScpContainer, ScpMatrix

# Default contaminant patterns for proteomics
_DEFAULT_CONTAMINANT_PATTERNS = [
    r"^KRT\d+",  # Keratins
    r"Keratin",
    r"Trypsin",
    r"Albumin",
    r"ALB_",  # Albumin
    r"IG[HKL]",  # Immunoglobulins
    r"^HBA[12]",  # Hemoglobin alpha
    r"^HBB",  # Hemoglobin beta
    r"Hemoglobin",
]


def _create_container_with_updated_var(
    container: ScpContainer,
    assay_name: str,
    new_var: pl.DataFrame,
) -> ScpContainer:
    """Create a new container with updated var DataFrame for the specified assay.

    Args:
        container: Original container.
        assay_name: Name of assay to update.
        new_var: New var DataFrame.

    Returns:
        New container with updated var.
    """
    assay = container.assays[assay_name]
    new_assay = assay.subset(np.arange(assay.n_features), copy_data=False)
    new_assay.var = new_var

    new_assays = {
        name: new_assay if name == assay_name else a for name, a in container.assays.items()
    }

    return ScpContainer(
        obs=container.obs,
        assays=new_assays,
        links=list(container.links),
        history=list(container.history),
        sample_id_col=container.sample_id_col,
    )


def _sparse_row_means(data: np.ndarray, indptr: np.ndarray, n_rows: int) -> np.ndarray:
    """Compute means for each row in a CSR matrix.

    Args:
        data: Non-zero values.
        indptr: CSR index pointers.
        n_rows: Number of rows.

    Returns:
        Array of means per row.
    """
    result = np.zeros(n_rows)
    for i in range(n_rows):
        start, end = indptr[i], indptr[i + 1]
        if end > start:
            result[i] = np.mean(data[start:end])
    return result


def _sparse_row_medians(data: np.ndarray, indptr: np.ndarray, n_rows: int) -> np.ndarray:
    """Compute medians for each row in a CSR matrix.

    Args:
        data: Non-zero values.
        indptr: CSR index pointers.
        n_rows: Number of rows.

    Returns:
        Array of medians per row.
    """
    result = np.zeros(n_rows)
    for i in range(n_rows):
        start, end = indptr[i], indptr[i + 1]
        if end > start:
            result[i] = np.median(data[start:end])
    return result


def filter_features_by_missing_rate(
    container: ScpContainer,
    assay_name: str = "protein",
    layer: str = "raw",
    max_missing_rate: float = 0.5,
    detection_threshold: float = 0.0,
    inplace: bool = False,
) -> ScpContainer:
    """
    Filter features with excessive missing values.

    Features missing in more than max_missing_rate proportion of samples are removed.

    Args:
        container: The ScpContainer object.
        assay_name: Name of the assay to filter features in.
        layer: Layer to use for filtering.
        max_missing_rate: Maximum proportion of missing values allowed (0-1).
        detection_threshold: Value threshold for considering a value as detected.
        inplace: If True, return a new container with filtered features.
                  If False, return the original container with filter results in obs.

    Returns:
        A new ScpContainer with filtered features, or the original container
        with filter statistics added to assay var.

    Raises:
        AssayNotFoundError: If the specified assay does not exist.
        LayerNotFoundError: If the specified layer does not exist.
        ScpValueError: If max_missing_rate is not in [0, 1].

    Examples:
        >>> container = filter_features_by_missing_rate(
        ...     container, assay_name="protein", max_missing_rate=0.3
        ... )
    """
    if not (0 <= max_missing_rate <= 1):
        raise ScpValueError(
            f"max_missing_rate must be in [0, 1], got {max_missing_rate}.",
            parameter="max_missing_rate",
            value=max_missing_rate,
        )

    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]
    if layer not in assay.layers:
        raise LayerNotFoundError(layer, assay_name)

    matrix = assay.layers[layer]
    X = matrix.X

    # Calculate missing rate for each feature
    if sp.issparse(X):
        X.shape[0]
        n_detected = X.getnnz(axis=0)  # Number of non-zero entries per column
    else:
        n_detected = np.sum(detection_threshold < X, axis=0)

    missing_rate = 1.0 - (n_detected / X.shape[0])

    # Determine features to keep
    keep_mask = missing_rate <= max_missing_rate
    keep_indices = np.where(keep_mask)[0]

    if inplace:
        new_container = container.filter_features(assay_name, feature_indices=keep_indices)
        n_removed = assay.n_features - new_container.assays[assay_name].n_features
        new_container.log_operation(
            action="filter_features_by_missing_rate",
            params={
                "assay": assay_name,
                "layer": layer,
                "max_missing_rate": max_missing_rate,
            },
            description=f"Removed {n_removed} features with high missing rate.",
        )
        return new_container

    # Add filter statistics to var
    filter_col = f"keep_missing_rate_{max_missing_rate}"
    new_var = assay.var.with_columns(
        pl.Series("missing_rate", missing_rate),
        pl.Series(filter_col, keep_mask),
    )
    return _create_container_with_updated_var(container, assay_name, new_var)


def filter_features_by_variance(
    container: ScpContainer,
    assay_name: str = "protein",
    layer: str = "raw",
    min_variance: float = 0.01,
    top_n: int | None = None,
    inplace: bool = False,
) -> ScpContainer:
    """
    Filter features with low variance.

    Features with variance below min_variance are removed, or only top_n features
    with highest variance are kept.

    Args:
        container: The ScpContainer object.
        assay_name: Name of the assay to filter features in.
        layer: Layer to use for filtering.
        min_variance: Minimum variance threshold. Features below this are removed.
        top_n: If specified, keep only the top N features by variance.
        inplace: If True, return a new container with filtered features.
                  If False, return the original container with variance in var.

    Returns:
        A new ScpContainer with filtered features, or the original container
        with variance statistics added to assay var.

    Raises:
        AssayNotFoundError: If the specified assay does not exist.
        LayerNotFoundError: If the specified layer does not exist.
        ScpValueError: If parameters are invalid.

    Examples:
        >>> # Keep features with variance > 0.1
        >>> container = filter_features_by_variance(
        ...     container, assay_name="protein", min_variance=0.1
        ... )
        >>> # Keep top 1000 most variable features
        >>> container = filter_features_by_variance(
        ...     container, assay_name="protein", top_n=1000
        ... )
    """
    if min_variance < 0:
        raise ScpValueError(
            f"min_variance must be non-negative, got {min_variance}.",
            parameter="min_variance",
            value=min_variance,
        )

    if top_n is not None and top_n <= 0:
        raise ScpValueError(
            f"top_n must be positive, got {top_n}.",
            parameter="top_n",
            value=top_n,
        )

    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]
    if layer not in assay.layers:
        raise LayerNotFoundError(layer, assay_name)

    X = assay.layers[layer].X

    # Calculate variance for each feature
    if sp.issparse(X):
        # For sparse matrices, use efficient variance computation
        X = X.tocsc()
        mean = np.array(X.mean(axis=0)).flatten()
        mean_sq = np.array(X.power(2).mean(axis=0)).flatten()
        variance = mean_sq - mean**2
        variance = np.nan_to_num(variance, nan=0.0)
    else:
        variance = np.var(X, axis=0)

    if top_n is not None:
        # Keep top N features by variance
        top_indices = np.argsort(variance)[-top_n:][::-1]
        keep_indices = top_indices
        keep_mask = np.zeros(len(variance), dtype=bool)
        keep_mask[top_indices] = True
    else:
        # Keep features above variance threshold
        keep_mask = variance >= min_variance
        keep_indices = np.where(keep_mask)[0]

    if inplace:
        new_container = container.filter_features(assay_name, feature_indices=keep_indices)
        n_removed = assay.n_features - new_container.assays[assay_name].n_features
        new_container.log_operation(
            action="filter_features_by_variance",
            params={
                "assay": assay_name,
                "layer": layer,
                "min_variance": min_variance,
                "top_n": top_n,
            },
            description=f"Removed {n_removed} features with low variance.",
        )
        return new_container

    # Add variance statistics to var
    var_col_name = "feature_variance"
    filter_col_name = f"keep_variance_{min_variance}"
    new_var = assay.var.with_columns(
        pl.Series(var_col_name, variance),
        pl.Series(filter_col_name, keep_mask),
    )
    return _create_container_with_updated_var(container, assay_name, new_var)


def filter_features_by_prevalence(
    container: ScpContainer,
    assay_name: str = "protein",
    layer: str = "raw",
    min_prevalence: int = 3,
    min_prevalence_ratio: float | None = None,
    detection_threshold: float = 0.0,
    inplace: bool = False,
) -> ScpContainer:
    """
    Filter features based on prevalence (number/ratio of samples where detected).

    Features detected in fewer than min_prevalence samples or less than
    min_prevalence_ratio of samples are removed.

    Args:
        container: The ScpContainer object.
        assay_name: Name of the assay to filter features in.
        layer: Layer to use for filtering.
        min_prevalence: Minimum number of samples where feature must be detected.
        min_prevalence_ratio: Minimum ratio of samples (0-1). Overrides min_prevalence if set.
        detection_threshold: Value threshold for considering a value as detected.
        inplace: If True, return a new container with filtered features.

    Returns:
        A new ScpContainer with filtered features, or the original container
        with prevalence statistics added to assay var.

    Raises:
        AssayNotFoundError: If the specified assay does not exist.
        LayerNotFoundError: If the specified layer does not exist.
        ScpValueError: If parameters are invalid.

    Examples:
        >>> # Keep features detected in at least 3 samples
        >>> container = filter_features_by_prevalence(
        ...     container, assay_name="protein", min_prevalence=3
        ... )
        >>> # Keep features detected in at least 10% of samples
        >>> container = filter_features_by_prevalence(
        ...     container, assay_name="protein", min_prevalence_ratio=0.1
        ... )
    """
    if min_prevalence < 0:
        raise ScpValueError(
            f"min_prevalence must be non-negative, got {min_prevalence}.",
            parameter="min_prevalence",
            value=min_prevalence,
        )

    if min_prevalence_ratio is not None and not (0 <= min_prevalence_ratio <= 1):
        raise ScpValueError(
            f"min_prevalence_ratio must be in [0, 1], got {min_prevalence_ratio}.",
            parameter="min_prevalence_ratio",
            value=min_prevalence_ratio,
        )

    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]
    if layer not in assay.layers:
        raise LayerNotFoundError(layer, assay_name)

    X = assay.layers[layer].X
    n_samples = X.shape[0]

    # Calculate prevalence for each feature
    if sp.issparse(X):
        prevalence = np.array(X.getnnz(axis=0)).flatten()
    else:
        prevalence = np.sum(detection_threshold < X, axis=0)

    # Determine threshold
    if min_prevalence_ratio is not None:
        threshold = int(min_prevalence_ratio * n_samples)
    else:
        threshold = min_prevalence

    keep_mask = prevalence >= threshold
    keep_indices = np.where(keep_mask)[0]

    if inplace:
        new_container = container.filter_features(assay_name, feature_indices=keep_indices)
        n_removed = assay.n_features - new_container.assays[assay_name].n_features
        new_container.log_operation(
            action="filter_features_by_prevalence",
            params={
                "assay": assay_name,
                "layer": layer,
                "min_prevalence": threshold,
            },
            description=f"Removed {n_removed} features with low prevalence.",
        )
        return new_container

    # Add prevalence statistics to var
    new_var = assay.var.with_columns(
        pl.Series("prevalence", prevalence),
        pl.Series(f"keep_prevalence_{threshold}", keep_mask),
    )
    return _create_container_with_updated_var(container, assay_name, new_var)


def filter_samples_by_total_count(
    container: ScpContainer,
    assay_name: str = "protein",
    layer: str = "raw",
    min_total: float = 500.0,
    max_total: float | None = None,
    detection_threshold: float = 0.0,
    inplace: bool = False,
) -> ScpContainer:
    """
    Filter samples based on total protein count/intensity.

    Samples with total count below min_total or above max_total are removed.

    Args:
        container: The ScpContainer object.
        assay_name: Name of the assay to use.
        layer: Layer to use for filtering.
        min_total: Minimum total count/intensity.
        max_total: Maximum total count/intensity (optional).
        detection_threshold: Minimum value to include in count.
        inplace: If True, filter samples. If False, add statistics to obs.

    Returns:
        A new ScpContainer with filtered samples, or the original container
        with count statistics added to obs.

    Raises:
        AssayNotFoundError: If the specified assay does not exist.
        LayerNotFoundError: If the specified layer does not exist.
        ScpValueError: If parameters are invalid.

    Examples:
        >>> # Remove samples with total intensity < 1000
        >>> container = filter_samples_by_total_count(
        ...     container, assay_name="protein", min_total=1000, inplace=True
        ... )
    """
    if min_total < 0:
        raise ScpValueError(
            f"min_total must be non-negative, got {min_total}.",
            parameter="min_total",
            value=min_total,
        )

    if max_total is not None and max_total < min_total:
        raise ScpValueError(
            f"max_total must be >= min_total, got {max_total} < {min_total}.",
            parameter="max_total",
            value=max_total,
        )

    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]
    if layer not in assay.layers:
        raise LayerNotFoundError(layer, assay_name)

    X = assay.layers[layer].X

    # Calculate total count per sample
    if sp.issparse(X):
        total_counts = np.array(X.sum(axis=1)).flatten()
    else:
        # Only count values above threshold
        X_masked = np.where(detection_threshold < X, X, 0)
        total_counts = np.sum(X_masked, axis=1)

    # Determine samples to keep
    if max_total is not None:
        keep_mask = (total_counts >= min_total) & (total_counts <= max_total)
    else:
        keep_mask = total_counts >= min_total

    keep_indices = np.where(keep_mask)[0]

    if inplace:
        new_container = container.filter_samples(sample_indices=keep_indices)

        n_removed = container.n_samples - new_container.n_samples
        new_container.log_operation(
            action="filter_samples_by_total_count",
            params={
                "assay": assay_name,
                "layer": layer,
                "min_total": min_total,
                "max_total": max_total,
            },
            description=f"Removed {n_removed} samples by total count.",
        )
        return new_container
    else:
        # Add count statistics to obs
        col_name = "total_count"
        filter_col = f"keep_total_min_{min_total}"

        new_obs = container.obs.with_columns(
            pl.Series(col_name, total_counts),
            pl.Series(filter_col, keep_mask),
        )

        new_container = ScpContainer(
            obs=new_obs,
            assays=container.assays,
            links=list(container.links),
            history=list(container.history),
            sample_id_col=container.sample_id_col,
        )
        return new_container


def filter_samples_by_missing_rate(
    container: ScpContainer,
    assay_name: str = "protein",
    layer: str = "raw",
    max_missing_rate: float = 0.5,
    detection_threshold: float = 0.0,
    inplace: bool = False,
) -> ScpContainer:
    """
    Filter samples based on missing rate (proportion of missing values).

    Samples with missing rate above max_missing_rate are removed.

    Args:
        container: The ScpContainer object.
        assay_name: Name of the assay to use.
        layer: Layer to use for filtering.
        max_missing_rate: Maximum proportion of missing values allowed (0-1).
        detection_threshold: Value threshold for considering a value as detected.
        inplace: If True, filter samples. If False, add statistics to obs.

    Returns:
        A new ScpContainer with filtered samples, or the original container
        with missing rate statistics added to obs.

    Raises:
        AssayNotFoundError: If the specified assay does not exist.
        LayerNotFoundError: If the specified layer does not exist.
        ScpValueError: If parameters are invalid.

    Examples:
        >>> # Remove samples with >50% missing values
        >>> container = filter_samples_by_missing_rate(
        ...     container, assay_name="protein", max_missing_rate=0.5, inplace=True
        ... )
    """
    if not (0 <= max_missing_rate <= 1):
        raise ScpValueError(
            f"max_missing_rate must be in [0, 1], got {max_missing_rate}.",
            parameter="max_missing_rate",
            value=max_missing_rate,
        )

    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]
    if layer not in assay.layers:
        raise LayerNotFoundError(layer, assay_name)

    X = assay.layers[layer].X
    n_features = X.shape[1]

    # Calculate missing rate per sample
    if sp.issparse(X):
        n_detected = np.array(X.getnnz(axis=1)).flatten()
    else:
        n_detected = np.sum(detection_threshold < X, axis=1)

    missing_rate = 1.0 - (n_detected / n_features)

    # Determine samples to keep
    keep_mask = missing_rate <= max_missing_rate
    keep_indices = np.where(keep_mask)[0]

    if inplace:
        new_container = container.filter_samples(sample_indices=keep_indices)

        n_removed = container.n_samples - new_container.n_samples
        new_container.log_operation(
            action="filter_samples_by_missing_rate",
            params={
                "assay": assay_name,
                "layer": layer,
                "max_missing_rate": max_missing_rate,
            },
            description=f"Removed {n_removed} samples with high missing rate.",
        )
        return new_container
    else:
        # Add missing rate statistics to obs
        col_name = "missing_rate"
        filter_col = f"keep_missing_rate_{max_missing_rate}"

        new_obs = container.obs.with_columns(
            pl.Series(col_name, missing_rate),
            pl.Series(filter_col, keep_mask),
        )

        new_container = ScpContainer(
            obs=new_obs,
            assays=container.assays,
            links=list(container.links),
            history=list(container.history),
            sample_id_col=container.sample_id_col,
        )
        return new_container


def detect_contaminant_proteins(
    container: ScpContainer,
    assay_name: str = "protein",
    layer: str = "raw",
    contaminant_patterns: list[str] | None = None,
    min_prevalence: int = 3,
    detection_threshold: float = 0.0,
) -> ScpContainer:
    """
    Detect potential contaminant proteins based on naming patterns.

    For proteomics, common contaminants include:
    - Keratins (KRT_* proteins)
    - Trypsin
    - Albumin
    - Immunoglobulins
    - Hemoglobin

    Args:
        container: The ScpContainer object.
        assay_name: Name of the assay to use.
        layer: Layer to use for detection.
        contaminant_patterns: List of regex patterns for contaminant names.
                             If None, uses default common proteomics contaminants.
        min_prevalence: Minimum number of samples where protein must be detected.
        detection_threshold: Value threshold for considering a value as detected.

    Returns:
        ScpContainer with contaminant detection results added to assay var.

    Raises:
        AssayNotFoundError: If the specified assay does not exist.
        LayerNotFoundError: If the specified layer does not exist.

    Examples:
        >>> container = detect_contaminant_proteins(
        ...     container, assay_name="protein"
        ... )
        >>> # Get list of detected contaminants
        >>> contaminants = container.assays['protein'].var.filter(
        ...     pl.col('is_contaminant') == True
        ... )
    """
    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]
    if layer not in assay.layers:
        raise LayerNotFoundError(layer, assay_name)

    X = assay.layers[layer].X
    patterns = contaminant_patterns or _DEFAULT_CONTAMINANT_PATTERNS

    # Get feature IDs and check against patterns (vectorized regex matching)
    feature_ids = assay.feature_ids.to_numpy().astype(str)
    is_contaminant = np.zeros(len(feature_ids), dtype=bool)

    compiled_patterns = [re.compile(p, re.IGNORECASE) for p in patterns]
    for i, feat_id in enumerate(feature_ids):
        for pattern in compiled_patterns:
            if pattern.search(feat_id):
                is_contaminant[i] = True
                break

    # Calculate prevalence for each potential contaminant
    if sp.issparse(X):
        prevalence = np.array(X.getnnz(axis=0)).flatten()
    else:
        prevalence = np.sum(detection_threshold < X, axis=0)

    # Only mark as contaminant if also prevalent enough
    is_detected_contaminant = is_contaminant & (prevalence >= min_prevalence)

    # Calculate contaminant content per sample (vectorized)
    if sp.issparse(X):
        X_dense = X.toarray()
    else:
        X_dense = X

    contaminant_content = X_dense[:, is_contaminant].sum(axis=1)
    total_intensity = X_dense.sum(axis=1)
    contaminant_ratio = np.divide(
        contaminant_content,
        total_intensity,
        out=np.zeros_like(contaminant_content),
        where=total_intensity > 0,
    )

    # Add results to var
    new_var = assay.var.with_columns(
        pl.Series("is_contaminant", is_detected_contaminant),
        pl.Series("contaminant_prevalence", np.where(is_contaminant, prevalence, 0)),
    )

    # Add contaminant content to obs
    new_obs = container.obs.with_columns(
        pl.Series("contaminant_content", contaminant_content),
        pl.Series("contaminant_ratio", contaminant_ratio),
    )

    new_container = _create_container_with_updated_var(container, assay_name, new_var)
    new_container.obs = new_obs

    n_contaminants = int(np.sum(is_detected_contaminant))
    new_container.log_operation(
        action="detect_contaminant_proteins",
        params={
            "assay": assay_name,
            "layer": layer,
            "patterns": patterns,
            "min_prevalence": min_prevalence,
        },
        description=f"Detected {n_contaminants} contaminant proteins.",
    )

    return new_container


def detect_doublets(
    container: ScpContainer,
    assay_name: str = "protein",
    layer: str = "raw",
    method: Literal["knn", "isolation", "hybrid"] = "knn",
    n_neighbors: int = 15,
    expected_doublet_rate: float = 0.1,
    random_state: int = 42,
) -> ScpContainer:
    """
    Detect potential doublets (multiplets) in single-cell proteomics data.

    Doublets are samples that may contain material from multiple cells.
    This is a common issue in single-cell experiments.

    Methods:
    - 'knn': Uses local density (distance to k-nearest neighbors) to identify
             samples in sparse regions that may be doublets.
    - 'isolation': Uses Isolation Forest to detect anomalies.
    - 'hybrid': Combines both methods for more robust detection.

    Args:
        container: The ScpContainer object.
        assay_name: Name of the assay to use.
        layer: Layer to use for detection.
        method: Detection method to use.
        n_neighbors: Number of neighbors for KNN-based detection.
        expected_doublet_rate: Expected proportion of doublets (0-1).
        random_state: Random state for reproducibility.

    Returns:
        ScpContainer with doublet detection results added to obs:
        - 'is_doublet': Boolean indicating if sample is a predicted doublet
        - 'doublet_score': Doublet probability/score

    Raises:
        AssayNotFoundError: If the specified assay does not exist.
        LayerNotFoundError: If the specified layer does not exist.
        ScpValueError: If parameters are invalid.

    Examples:
        >>> container = detect_doublets(
        ...     container, assay_name="protein", method="knn"
        ... )
        >>> # Remove predicted doublets
        >>> doublet_mask = container.obs['is_doublet'].to_numpy()
        >>> container_clean = container.filter_samples(
        ...     np.where(~doublet_mask)[0]
        ... )
    """
    if not (0 < expected_doublet_rate < 0.5):
        raise ScpValueError(
            f"expected_doublet_rate must be in (0, 0.5), got {expected_doublet_rate}.",
            parameter="expected_doublet_rate",
            value=expected_doublet_rate,
        )

    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]
    if layer not in assay.layers:
        raise LayerNotFoundError(layer, assay_name)

    X = assay.layers[layer].X

    # Handle sparse matrices
    if sp.issparse(X):
        X_for_analysis = X.toarray()
    else:
        X_for_analysis = X

    # Impute missing values for distance calculation
    X_clean = np.nan_to_num(X_for_analysis, nan=0.0)

    scores = np.zeros(container.n_samples)

    if method in ("knn", "hybrid"):
        # KNN-based detection using local density
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=-1)
        nbrs.fit(X_clean)
        distances, _ = nbrs.kneighbors(X_clean)

        # Use average distance to k-nearest neighbors as doublet score
        # Doublets often have larger distances (in sparse regions)
        knn_score = np.mean(distances[:, 1:], axis=1)
        knn_score = (knn_score - knn_score.min()) / (knn_score.max() - knn_score.min() + 1e-10)

        if method == "knn":
            scores = knn_score

    if method in ("isolation", "hybrid"):
        # Isolation Forest based detection
        from sklearn.ensemble import IsolationForest

        iso_forest = IsolationForest(
            contamination=expected_doublet_rate,
            random_state=random_state,
            n_jobs=-1,
        )
        iso_forest.fit_predict(X_clean)
        iso_score = -iso_forest.score_samples(X_clean)  # Negative because lower is abnormal
        iso_score = (iso_score - iso_score.min()) / (iso_score.max() - iso_score.min() + 1e-10)

        if method == "isolation":
            scores = iso_score
        else:  # hybrid
            scores = 0.5 * knn_score + 0.5 * iso_score

    # Determine doublet threshold based on expected rate
    threshold = np.quantile(scores, 1.0 - expected_doublet_rate)
    is_doublet = scores > threshold

    # Add results to obs
    new_obs = container.obs.with_columns(
        pl.Series("is_doublet", is_doublet),
        pl.Series("doublet_score", scores),
    )

    new_container = ScpContainer(
        obs=new_obs,
        assays=container.assays,
        links=list(container.links),
        history=list(container.history),
        sample_id_col=container.sample_id_col,
    )

    n_doublets = int(np.sum(is_doublet))
    new_container.log_operation(
        action="detect_doublets",
        params={
            "assay": assay_name,
            "layer": layer,
            "method": method,
            "expected_rate": expected_doublet_rate,
        },
        description=f"Detected {n_doublets} potential doublets using {method} method.",
    )

    return new_container


def calculate_qc_metrics(
    container: ScpContainer,
    assay_name: str = "protein",
    layer: str = "raw",
    detection_threshold: float = 0.0,
) -> ScpContainer:
    """
    Calculate comprehensive QC metrics for samples and features.

    This function adds multiple QC metrics to the container:
    Sample metrics (in obs):
    - n_detected: Number of detected features per sample
    - total_intensity: Total intensity per sample
    - missing_rate: Proportion of missing values per sample
    - mean_intensity: Mean intensity of detected values per sample
    - median_intensity: Median intensity of detected values per sample

    Feature metrics (in var):
    - n_detected: Number of samples where feature is detected
    - prevalence: Proportion of samples where feature is detected
    - mean_intensity: Mean intensity across samples
    - variance: Variance across samples

    Args:
        container: The ScpContainer object.
        assay_name: Name of the assay to calculate metrics for.
        layer: Layer to use for calculation.
        detection_threshold: Value threshold for considering a value as detected.

    Returns:
        ScpContainer with QC metrics added to obs and var.

    Raises:
        AssayNotFoundError: If the specified assay does not exist.
        LayerNotFoundError: If the specified layer does not exist.

    Examples:
        >>> container = calculate_qc_metrics(container, assay_name="protein")
        >>> # Access sample metrics
        >>> n_detected = container.obs['n_detected'].to_numpy()
        >>> # Access feature metrics
        >>> prevalence = container.assays['protein'].var['prevalence'].to_numpy()
    """
    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]
    if layer not in assay.layers:
        raise LayerNotFoundError(layer, assay_name)

    X = assay.layers[layer].X
    n_samples, n_features = X.shape

    if sp.issparse(X):
        X_csr = X.tocsr()
        X_csc = X.tocsc()

        # Sample metrics - vectorized where possible
        n_detected_samples = np.diff(X_csr.indptr)
        total_intensity_samples = np.array(X_csr.sum(axis=1)).flatten()
        missing_rate_samples = 1.0 - (n_detected_samples / n_features)

        mean_intensity_samples = _sparse_row_means(X_csr.data, X_csr.indptr, n_samples)
        median_intensity_samples = _sparse_row_medians(X_csr.data, X_csr.indptr, n_samples)

        # Feature metrics - vectorized where possible
        n_detected_features = np.diff(X_csc.indptr)
        prevalence_features = n_detected_features / n_samples
        np.array(X_csc.sum(axis=0)).flatten()

        mean_intensity_features = np.zeros(n_features)
        variance_features = np.zeros(n_features)
        for j in range(n_features):
            start, end = X_csc.indptr[j], X_csc.indptr[j + 1]
            if end > start:
                col_data = X_csc.data[start:end]
                mean_intensity_features[j] = np.mean(col_data)
                variance_features[j] = np.var(col_data)
    else:
        # Dense matrix operations - fully vectorized
        detected_mask = detection_threshold < X

        # Sample metrics
        n_detected_samples = detected_mask.sum(axis=1)
        total_intensity_samples = X.sum(axis=1)
        missing_rate_samples = 1.0 - (n_detected_samples / n_features)

        # Use masked arrays for mean/median of detected values only
        X_masked = np.where(detected_mask, X, np.nan)
        mean_intensity_samples = np.nanmean(X_masked, axis=1)
        median_intensity_samples = np.nanmedian(X_masked, axis=1)
        mean_intensity_samples = np.nan_to_num(mean_intensity_samples, nan=0.0)
        median_intensity_samples = np.nan_to_num(median_intensity_samples, nan=0.0)

        # Feature metrics - fully vectorized
        n_detected_features = detected_mask.sum(axis=0)
        prevalence_features = n_detected_features / n_samples
        X.sum(axis=0)

        # Transpose mask for column-wise operations
        X_masked_t = np.where(detected_mask.T, X.T, np.nan)
        mean_intensity_features = np.nanmean(X_masked_t, axis=1)
        variance_features = np.nanvar(X_masked_t, axis=1)
        mean_intensity_features = np.nan_to_num(mean_intensity_features, nan=0.0)
        variance_features = np.nan_to_num(variance_features, nan=0.0)

    # Add sample metrics to obs
    new_obs = container.obs.with_columns(
        pl.Series("n_detected", n_detected_samples),
        pl.Series("total_intensity", total_intensity_samples),
        pl.Series("missing_rate", missing_rate_samples),
        pl.Series("mean_intensity", mean_intensity_samples),
        pl.Series("median_intensity", median_intensity_samples),
    )

    # Add feature metrics to var
    new_var = assay.var.with_columns(
        pl.Series("n_detected", n_detected_features),
        pl.Series("prevalence", prevalence_features),
        pl.Series("mean_intensity", mean_intensity_features),
        pl.Series("variance", variance_features),
    )

    new_container = _create_container_with_updated_var(container, assay_name, new_var)
    new_container.obs = new_obs

    new_container.log_operation(
        action="calculate_qc_metrics",
        params={"assay": assay_name, "layer": layer},
        description="Calculated comprehensive QC metrics.",
    )

    return new_container


if __name__ == "__main__":
    import sys

    print("Testing advanced QC methods...")
    print()

    # Create test data
    np.random.seed(42)
    n_samples = 100
    n_features = 50

    # Create synthetic data with some missing values
    X_dense = np.random.exponential(1.0, size=(n_samples, n_features))
    missing_mask = np.random.random((n_samples, n_features)) < 0.3
    X_dense[missing_mask] = 0

    # Add some contaminants (first 5 features)
    X_dense[:, :5] *= 2  # Higher abundance for contaminants

    # Create container
    import polars as pl

    obs = pl.DataFrame(
        {
            "_index": [f"sample_{i}" for i in range(n_samples)],
        }
    )

    var = pl.DataFrame(
        {
            "_index": [f"protein_{i}" for i in range(n_features)],
            "name": [f"KRT{i}" if i < 5 else f"PROT{i}" for i in range(n_features)],
        }
    )

    from scptensor.core.structures import Assay, ScpMatrix

    assay = Assay(var=var, layers={"raw": ScpMatrix(X=X_dense, M=None)})
    container = ScpContainer(obs=obs, assays={"protein": assay})

    # Test 1: calculate_qc_metrics
    print("Test 1: calculate_qc_metrics")
    container = calculate_qc_metrics(container, assay_name="protein")
    assert "n_detected" in container.obs.columns
    assert "total_intensity" in container.obs.columns
    assert "missing_rate" in container.obs.columns
    assert "prevalence" in container.assays["protein"].var.columns
    assert "mean_intensity" in container.assays["protein"].var.columns
    print("  QC metrics added successfully")

    # Test 2: filter_features_by_missing_rate (non-inplace)
    print("Test 2: filter_features_by_missing_rate")
    container = filter_features_by_missing_rate(
        container, assay_name="protein", max_missing_rate=0.5, inplace=False
    )
    assert "missing_rate" in container.assays["protein"].var.columns
    print("  Missing rate filter calculated successfully")

    # Test 3: filter_features_by_variance (non-inplace)
    print("Test 3: filter_features_by_variance")
    container = filter_features_by_variance(
        container, assay_name="protein", min_variance=0.1, inplace=False
    )
    assert "feature_variance" in container.assays["protein"].var.columns
    print("  Variance filter calculated successfully")

    # Test 4: filter_features_by_prevalence (non-inplace)
    print("Test 4: filter_features_by_prevalence")
    container = filter_features_by_prevalence(
        container, assay_name="protein", min_prevalence=10, inplace=False
    )
    assert "prevalence" in container.assays["protein"].var.columns
    print("  Prevalence filter calculated successfully")

    # Test 5: filter_samples_by_total_count (non-inplace)
    print("Test 5: filter_samples_by_total_count")
    container = filter_samples_by_total_count(
        container, assay_name="protein", min_total=10, inplace=False
    )
    assert "total_count" in container.obs.columns
    print("  Total count filter calculated successfully")

    # Test 6: filter_samples_by_missing_rate (non-inplace)
    print("Test 6: filter_samples_by_missing_rate")
    container = filter_samples_by_missing_rate(
        container, assay_name="protein", max_missing_rate=0.5, inplace=False
    )
    assert "missing_rate" in container.obs.columns
    print("  Sample missing rate filter calculated successfully")

    # Test 7: detect_contaminant_proteins
    print("Test 7: detect_contaminant_proteins")
    container = detect_contaminant_proteins(container, assay_name="protein")
    assert "is_contaminant" in container.assays["protein"].var.columns
    assert "contaminant_content" in container.obs.columns
    n_contaminants = container.assays["protein"].var["is_contaminant"].sum()
    print(f"  Detected {n_contaminants} contaminant proteins")

    # Test 8: detect_doublets
    print("Test 8: detect_doublets")
    container = detect_doublets(container, assay_name="protein", method="knn")
    assert "is_doublet" in container.obs.columns
    assert "doublet_score" in container.obs.columns
    n_doublets = container.obs["is_doublet"].sum()
    print(f"  Detected {n_doublets} potential doublets")

    # Test 9: Inplace filtering
    print("Test 9: Inplace feature filtering")
    original_n_features = container.assays["protein"].n_features
    container_filtered = filter_features_by_missing_rate(
        container, assay_name="protein", max_missing_rate=0.2, inplace=True
    )
    assert container_filtered.assays["protein"].n_features < original_n_features
    print(
        f"  Features reduced from {original_n_features} to {container_filtered.assays['protein'].n_features}"
    )

    # Test 10: Inplace sample filtering
    print("Test 10: Inplace sample filtering")
    original_n_samples = container.n_samples
    container_filtered_samples = filter_samples_by_missing_rate(
        container, assay_name="protein", max_missing_rate=0.3, inplace=True
    )
    assert container_filtered_samples.n_samples <= original_n_samples
    print(f"  Samples reduced from {original_n_samples} to {container_filtered_samples.n_samples}")

    # Test 11: Top N features by variance
    print("Test 11: Filter top N features by variance")
    container_top_n = filter_features_by_variance(
        container, assay_name="protein", top_n=20, inplace=True
    )
    assert container_top_n.assays["protein"].n_features == 20
    print("  Kept top 20 features by variance")

    # Test 12: Sparse matrix support
    print("Test 12: Sparse matrix support")
    X_sparse = sp.csr_matrix(X_dense)
    assay_sparse = Assay(var=var, layers={"raw": ScpMatrix(X=X_sparse, M=None)})
    container_sparse = ScpContainer(obs=obs, assays={"protein": assay_sparse})
    container_sparse = calculate_qc_metrics(container_sparse, assay_name="protein")
    assert "n_detected" in container_sparse.obs.columns
    assert "prevalence" in container_sparse.assays["protein"].var.columns
    print("  Sparse matrix support works correctly")

    print()
    print("=" * 60)
    print("All advanced QC tests passed successfully!")
    print("=" * 60)
    sys.exit(0)
