"""Sample-level Quality Control.

Handles QC for cells/samples including:
- Basic metric calculation (ID count, total intensity, detection rate)
- Outlier detection (using robust MAD statistics)
- Doublet detection based on intensity outliers
- Batch effect assessment
"""

import numpy as np
import polars as pl

from scptensor.core.structures import ScpContainer
from scptensor.qc._utils import (
    compute_sample_qc_vectors,
    filter_samples_with_provenance,
    resolve_assay,
    resolve_layer,
    validate_column_exists,
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
    _, layer = resolve_layer(assay, assay_name=resolved_assay_name, layer_name=layer_name)
    n_features, total_intensity = compute_sample_qc_vectors(layer)
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
    new_container.obs = new_container.obs.hstack(metrics_df)
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
    _, layer = resolve_layer(
        assay,
        assay_name=resolved_assay_name,
        preferred_layer="raw",
        fallback_to_first=True,
    )
    n_features, _ = compute_sample_qc_vectors(layer)

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

    n_removed = container.n_samples - len(keep_indices)
    filter_desc = (
        f"Removed {n_removed}/{container.n_samples} samples from assay '{resolved_assay_name}'. "
        f"Filtering: n_features >= {min_features}"
    )
    if use_mad:
        filter_desc += f" AND not lower-outlier by >{nmads} MADs."
    else:
        filter_desc += " (hard threshold only)."

    return filter_samples_with_provenance(
        container,
        keep_indices,
        action="filter_low_quality_samples",
        params={
            "assay": resolved_assay_name,
            "min_features": min_features,
            "use_mad": use_mad,
            "nmads": nmads,
        },
        description=filter_desc,
    )


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
    _, layer = resolve_layer(
        assay,
        assay_name=resolved_assay_name,
        preferred_layer="raw",
        fallback_to_first=True,
    )
    _, total_intensity = compute_sample_qc_vectors(layer)

    # Transform to log space for better outlier detection
    log_lib_size = np.log1p(total_intensity)

    # Detect upper-tail outliers (high intensity = potential doublets)
    is_doublet = is_outlier_mad(
        log_lib_size,
        nmads=nmads,
        direction="upper",
    )

    keep_indices = np.where(~is_doublet)[0]

    n_removed = container.n_samples - len(keep_indices)
    return filter_samples_with_provenance(
        container,
        keep_indices,
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
    resolved_assay_name, assay = resolve_assay(container, assay_name)
    validate_column_exists(container.obs, batch_col)
    _, layer = resolve_layer(
        assay,
        assay_name=resolved_assay_name,
        preferred_layer="raw",
        fallback_to_first=True,
    )
    n_features, total_intensity = compute_sample_qc_vectors(layer)

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
