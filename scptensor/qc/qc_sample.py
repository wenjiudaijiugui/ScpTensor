"""Sample-level Quality Control.

Handles QC for cells/samples including:
- Basic metric calculation (ID count, total intensity, detection rate)
- Outlier detection (using robust MAD statistics)
- Doublet detection based on intensity outliers
- Batch effect assessment
"""

from typing import cast

import numpy as np
import polars as pl
import scipy.sparse as sp

from scptensor.core.filtering import FilterCriteria
from scptensor.core.structures import ScpContainer
from scptensor.qc._utils import (
    count_detected,
    resolve_assay,
    validate_layer,
)
from scptensor.qc.metrics import is_outlier_mad


def calculate_sample_qc_metrics(
    container: ScpContainer,
    assay_name: str = "protein",
    layer_name: str = "raw",
) -> ScpContainer:
    """Calculate quality control metrics for samples.

    Computes:
    - n_features: Number of detected features per sample
    - total_intensity: Sum of intensity values (library size)
    - log1p_total_intensity: Log-transformed total intensity

    Parameters
    ----------
    container : ScpContainer
        ScpContainer containing sample data.
    assay_name : str, default="protein"
        Name of the assay to use for QC metric calculation.
    layer_name : str, default="raw"
        Name of the layer to analyze.

    Returns
    -------
    ScpContainer
        New ScpContainer with QC metrics added to obs.
        Columns: n_features_{assay_name}, total_intensity_{assay_name},
        log1p_total_intensity_{assay_name}.

    Examples
    --------
    >>> result = calculate_sample_qc_metrics(container)
    >>> result.obs[['n_features_protein', 'total_intensity_protein']]
    """
    resolved_assay_name, assay = resolve_assay(container, assay_name)
    validate_layer(assay, layer_name, assay_name=resolved_assay_name)

    layer = assay.layers[layer_name]
    X = layer.X

    # Calculate metrics based on matrix type
    if sp.issparse(X):
        total_intensity = np.array(cast(sp.spmatrix, X).sum(axis=1)).flatten()
    else:
        total_intensity = np.nansum(X, axis=1)
    n_features = count_detected(layer.X, layer.M, axis=1)

    log1p_total = np.log1p(total_intensity)

    # Create metrics DataFrame with assay-specific column names
    metrics_df = pl.DataFrame(
        {
            f"n_features_{resolved_assay_name}": n_features,
            f"total_intensity_{resolved_assay_name}": total_intensity,
            f"log1p_total_intensity_{resolved_assay_name}": log1p_total,
        }
    )

    new_container = container.copy()
    new_container.obs = container.obs.hstack(metrics_df)
    return new_container


def filter_low_quality_samples(
    container: ScpContainer,
    assay_name: str = "protein",
    min_features: int = 100,
    nmads: float = 3.0,
    use_mad: bool = True,
) -> ScpContainer:
    """Filter low-quality samples based on feature detection count.

    Removes samples with insufficient detected features using:
    1. Hard threshold: n_features >= min_features
    2. MAD-based outlier detection (optional): removes lower-tail outliers

    Parameters
    ----------
    container : ScpContainer
        ScpContainer containing sample data to filter.
    assay_name : str, default="protein"
        Name of the assay to use for filtering.
    min_features : int, default=100
        Hard threshold for minimum detected features.
    nmads : float, default=3.0
        Number of MADs for statistical outlier detection.
    use_mad : bool, default=True
        Whether to apply MAD-based outlier detection.

    Returns
    -------
    ScpContainer
        New ScpContainer with low-quality samples removed.

    Examples
    --------
    >>> result = filter_low_quality_samples(container, min_features=5, use_mad=False)
    >>> result.n_samples
    4
    """
    resolved_assay_name, assay = resolve_assay(container, assay_name)

    layer = assay.layers.get("raw", next(iter(assay.layers.values())))
    X = layer.X

    # Calculate number of detected features per sample
    n_features = count_detected(layer.X, layer.M, axis=1)

    # Apply hard threshold filter
    keep_mask = n_features >= min_features

    # Apply MAD-based outlier detection (lower tail only)
    if use_mad:
        outliers = is_outlier_mad(
            n_features.astype(float),
            nmads=nmads,
            direction="lower",
        )
        keep_mask = keep_mask & (~outliers)

    keep_indices = np.where(keep_mask)[0]

    criteria = FilterCriteria.by_indices(keep_indices)
    new_container = container.filter_samples(criteria)

    # Log provenance
    n_removed = container.n_samples - len(keep_indices)
    filter_desc = (
        f"Removed {n_removed}/{container.n_samples} samples from assay '{resolved_assay_name}'. "
        f"Filtering: n_features >= {min_features}"
    )
    if use_mad:
        filter_desc += f" AND not lower-outlier by >{nmads} MADs."
    else:
        filter_desc += " (hard threshold only)."

    new_container.log_operation(
        action="filter_low_quality_samples",
        params={
            "assay": resolved_assay_name,
            "min_features": min_features,
            "use_mad": use_mad,
            "nmads": nmads,
        },
        description=filter_desc,
    )

    return new_container


def filter_doublets_mad(
    container: ScpContainer,
    assay_name: str = "protein",
    nmads: float = 3.0,
) -> ScpContainer:
    """Filter potential doublets using MAD-based outlier detection.

    Doublets (multiple cells analyzed as one) exhibit abnormally high
    total intensity. Uses upper-tail outlier detection on log-transformed
    total intensity using robust MAD statistics.

    Parameters
    ----------
    container : ScpContainer
        ScpContainer containing sample data.
    assay_name : str, default="protein"
        Name of the assay to use for doublet detection.
    nmads : float, default=3.0
        Number of MADs for doublet detection.
        Recommended: 2.0 (aggressive), 3.0 (standard), 4.0 (conservative).

    Returns
    -------
    ScpContainer
        New ScpContainer with potential doublets removed.

    Examples
    --------
    >>> result = filter_doublets_mad(container, nmads=2.0)
    >>> result.n_samples
    3
    """
    resolved_assay_name, assay = resolve_assay(container, assay_name)

    layer = assay.layers.get("raw", next(iter(assay.layers.values())))
    X = layer.X

    # Calculate total intensity for each sample
    if sp.issparse(X):
        total_intensity = np.array(X.sum(axis=1)).flatten()
    else:
        total_intensity = np.nansum(X, axis=1)

    # Transform to log space for better outlier detection
    log_lib_size = np.log1p(total_intensity)

    # Detect upper-tail outliers (high intensity = potential doublets)
    is_doublet = is_outlier_mad(
        log_lib_size,
        nmads=nmads,
        direction="upper",
    )

    keep_indices = np.where(~is_doublet)[0]

    criteria = FilterCriteria.by_indices(keep_indices)
    new_container = container.filter_samples(criteria)

    # Log provenance
    n_removed = container.n_samples - len(keep_indices)
    new_container.log_operation(
        action="filter_doublets_mad",
        params={
            "assay": resolved_assay_name,
            "nmads": nmads,
            "method": "MAD_upper_tail",
        },
        description=(
            f"Removed {n_removed}/{container.n_samples} samples as potential doublets "
            f"(high intensity outliers >{nmads} MADs) from assay '{resolved_assay_name}'."
        ),
    )

    return new_container


def assess_batch_effects(
    container: ScpContainer,
    batch_col: str,
    assay_name: str = "protein",
) -> pl.DataFrame:
    """Assess batch effects by calculating QC metrics per batch.

    Computes summary statistics for QC metrics grouped by batch identifier.

    Parameters
    ----------
    container : ScpContainer
        ScpContainer containing sample data with batch information in obs.
    batch_col : str
        Column name in container.obs containing batch identifiers.
    assay_name : str, default="protein"
        Name of the assay to analyze.

    Returns
    -------
    pl.DataFrame
        Summary statistics per batch with columns:
        - batch: Batch identifier
        - n_cells: Number of samples in batch
        - median_features: Median number of detected features
        - std_features: Standard deviation of feature counts
        - median_intensity: Median total intensity

    Examples
    --------
    >>> summary = assess_batch_effects(container, batch_col='batch')
    >>> summary
    shape: (2, 5)
    ┌───────┬────────┬────────────────┬─────────────┬─────────────────┐
    │ batch ┆ n_cells ┆ median_features ┆ std_features ┆ median_intensity │
    │ ---   ┆ ---    ┆ ---             ┆ ---         ┆ ---             │
    │ str   ┆ u32    ┆ f64             ┆ f64         ┆ f64             │
    ╞═══════╪════════╪═════════════════╪═════════════╪═════════════════╡
    │ A     ┆ 10     ┆ 15.0            ┆ 2.5         ┆ 100.0           │
    │ B     ┆ 10     ┆ 16.0            ┆ 2.0         ┆ 110.0           │
    └───────┴────────┴────────────────┴─────────────┴─────────────────┘
    """
    from scptensor.core.exceptions import ScpValueError

    resolved_assay_name, assay = resolve_assay(container, assay_name)

    # Validate batch column exists
    if batch_col not in container.obs.columns:
        available = ", ".join(f"'{col}'" for col in container.obs.columns)
        raise ScpValueError(
            f"Batch column '{batch_col}' not found in container.obs. "
            f"Available columns: {available}.",
            parameter="batch_col",
            value=batch_col,
        )

    layer = assay.layers.get("raw", next(iter(assay.layers.values())))
    X = layer.X

    # Calculate QC metrics for each sample
    if sp.issparse(X):
        X_sparse = cast(sp.spmatrix, X)
        total_intensity = np.array(X_sparse.sum(axis=1)).flatten()
    else:
        total_intensity = np.nansum(X, axis=1)
    n_features = count_detected(layer.X, layer.M, axis=1)

    # Create temporary DataFrame with batch identifiers and metrics
    temp_df = container.obs.select(batch_col).with_columns(
        [
            pl.Series("n_features", n_features),
            pl.Series("total_intensity", total_intensity),
        ]
    )

    # Calculate summary statistics per batch
    summary = (
        temp_df.group_by(batch_col)
        .agg(
            [
                pl.col("n_features").count().alias("n_cells"),
                pl.col("n_features").median().alias("median_features"),
                pl.col("n_features").std().alias("std_features"),
                pl.col("total_intensity").median().alias("median_intensity"),
            ]
        )
        .sort(batch_col)
    )

    return summary
