"""PSM-level Quality Control.

Handles filtering and QC for Peptide Spectrum Matches (PSMs):
- Contaminant filtering (keratins, trypsin, albumin, etc.)
- PIF (Parent Ion Fraction) filtering for co-elution detection
- FDR control via PEP to q-value conversion
- Sample-to-carrier ratio computation
- Reference channel normalization
"""

from typing import Literal

import numpy as np
import polars as pl
import scipy.sparse as sp

from scptensor.core.filtering import FilterCriteria
from scptensor.core.structures import ScpContainer, ScpMatrix
from scptensor.qc._utils import validate_assay, validate_column_exists, validate_threshold

# Default contaminant patterns for proteomics
DEFAULT_CONTAMINANT_PATTERNS = [
    r"^KRT\d+",  # Keratins (skin contaminants)
    r"Keratin",  # Generic keratin
    r"Trypsin",  # Digestion enzyme
    r"Albumin",  # Serum albumin
    r"ALB_",  # Albumin variants
    r"IG[HKL]",  # Immunoglobulins (antibodies)
    r"^HBA[12]",  # Hemoglobin alpha
    r"^HBB",  # Hemoglobin beta
    r"Hemoglobin",  # Generic hemoglobin
    r"^CON__",  # MaxQuant contaminants prefix
    r"^REV__",  # MaxQuant reverse/decoy prefix
]


def filter_psms_by_pif(
    container: ScpContainer,
    assay_name: str = "peptide",
    min_pif: float = 0.8,
    pif_col: str = "pif",
) -> ScpContainer:
    """Filter PSMs by Parent Ion Fraction (PIF) to remove co-elution interference.

    PIF measures the purity of precursor ion isolation. Low PIF values indicate
    co-eluting peptides or interference, compromising quantification accuracy.

    Parameters
    ----------
    container : ScpContainer
        ScpContainer containing PSM-level data with PIF scores in var.
    assay_name : str, default="peptide"
        Name of the assay containing PSM data.
    min_pif : float, default=0.8
        Minimum PIF threshold [0, 1].
        Recommended: 0.7 (low stringency), 0.8 (standard), 0.9 (high stringency).
    pif_col : str, default="pif"
        Column name in assay.var containing PIF scores.

    Returns
    -------
    ScpContainer
        ScpContainer with low-purity PSMs removed.

    Examples
    --------
    >>> result = filter_psms_by_pif(container, min_pif=0.8)
    >>> result.assays['peptide'].n_features
    2
    """
    validate_threshold(min_pif, "min_pif")
    assay = validate_assay(container, assay_name)
    validate_column_exists(assay, pif_col, "PIF column", "assay.var")

    pif_series = assay.var[pif_col]
    keep_mask = (pif_series >= min_pif).fill_null(False)
    keep_indices = np.where(keep_mask.to_numpy())[0]

    criteria = FilterCriteria.by_indices(keep_indices)
    new_container = container.filter_features(assay_name, criteria)

    # Log provenance
    n_removed = assay.n_features - len(keep_indices)
    percent_removed = (n_removed / assay.n_features * 100) if assay.n_features > 0 else 0

    new_container.log_operation(
        action="filter_psms_by_pif",
        params={"assay": assay_name, "min_pif": min_pif, "pif_col": pif_col},
        description=(
            f"Removed {n_removed}/{assay.n_features} PSMs ({percent_removed:.1f}%) "
            f"with {pif_col} < {min_pif} from assay '{assay_name}'."
        ),
    )

    return new_container


def filter_contaminants(
    container: ScpContainer,
    assay_name: str = "peptide",
    feature_col: str = "gene_names",
    patterns: list[str] | None = None,
) -> ScpContainer:
    """Filter contaminant features based on regex pattern matching.

    Removes common laboratory contaminants including keratins, trypsin,
    albumin, hemoglobins, and immunoglobulins.

    Parameters
    ----------
    container : ScpContainer
        ScpContainer containing feature metadata.
    assay_name : str, default="peptide"
        Name of the assay containing features to filter.
    feature_col : str, default="gene_names"
        Column name in assay.var containing feature names.
    patterns : list of str, optional
        List of regex patterns for contaminants. If None, uses
        DEFAULT_CONTAMINANT_PATTERNS.

    Returns
    -------
    ScpContainer
        ScpContainer with contaminant features removed.

    Examples
    --------
    >>> result = filter_contaminants(container, feature_col='gene_names')
    >>> result.assays['peptide'].n_features
    3

    Custom patterns:
    >>> custom = [r'^FLAG_', r'^HA_']
    >>> result = filter_contaminants(container, patterns=custom)
    """
    assay = validate_assay(container, assay_name)
    validate_column_exists(assay, feature_col, "Feature column", "assay.var")

    regex_patterns = patterns or DEFAULT_CONTAMINANT_PATTERNS
    combined_pattern = "|".join(f"({p})" for p in regex_patterns)

    is_contaminant = assay.var[feature_col].str.contains(combined_pattern).fill_null(False)
    keep_mask = ~is_contaminant
    keep_indices = np.where(keep_mask.to_numpy())[0]

    criteria = FilterCriteria.by_indices(keep_indices)
    new_container = container.filter_features(assay_name, criteria)

    # Log provenance
    n_removed = assay.n_features - len(keep_indices)
    percent_removed = (n_removed / assay.n_features * 100) if assay.n_features > 0 else 0

    new_container.log_operation(
        action="filter_contaminants",
        params={
            "assay": assay_name,
            "feature_col": feature_col,
            "patterns_count": len(regex_patterns),
            "patterns": regex_patterns[:5],
        },
        description=(
            f"Removed {n_removed}/{assay.n_features} features ({percent_removed:.1f}%) "
            f"matching {len(regex_patterns)} contaminant patterns from assay '{assay_name}'."
        ),
    )

    return new_container


def pep_to_qvalue(
    pep: np.ndarray,
    method: Literal["storey", "bh"] = "storey",
    lambda_param: float = 0.5,
) -> np.ndarray:
    """Convert Posterior Error Probability (PEP) to q-values for FDR control.

    Two methods:
    - Storey's method: Estimates π0 for increased power (default)
    - Benjamini-Hochberg (BH): More conservative (π0 = 1)

    Parameters
    ----------
    pep : np.ndarray
        1D array of PEP values in range [0, 1].
    method : {"storey", "bh"}, default="storey"
        Method for q-value calculation.
    lambda_param : float, default=0.5
        Lambda parameter for Storey's π0 estimation [0, 1).

    Returns
    -------
    np.ndarray
        1D array of q-values, monotonic and bounded by [0, 1].

    Examples
    --------
    >>> pep = np.array([0.001, 0.01, 0.05, 0.1, 0.5, 0.9])
    >>> qvals = pep_to_qvalue(pep, method="storey")
    >>> significant = qvals < 0.01
    """
    if method not in ("storey", "bh"):
        raise ValueError(f"method must be 'storey' or 'bh', got '{method}'.")

    if not 0 <= lambda_param < 1:
        raise ValueError(f"lambda_param must be in [0, 1), got {lambda_param}.")

    pep = np.asarray(pep, dtype=np.float64)
    nan_mask = np.isnan(pep)
    pep[nan_mask] = 1.0
    pep = np.clip(pep, 0, 1)

    n = len(pep)
    if n == 0:
        return np.array([], dtype=np.float64)

    sort_indices = np.argsort(pep)
    sorted_pep = pep[sort_indices]
    ranks = np.arange(1, n + 1)

    if method == "storey":
        pi0 = np.sum(pep > lambda_param) / (n * (1 - lambda_param))
        pi0 = min(max(pi0, 1.0 / n), 1.0)
        qvals = sorted_pep * pi0 * n / ranks
    else:  # bh
        qvals = sorted_pep * n / ranks

    # Enforce monotonicity
    qvals = np.minimum.accumulate(qvals[::-1])[::-1]
    qvals = np.clip(qvals, 0, 1)

    # Return to original order
    qvals_unsorted = np.empty_like(qvals)
    qvals_unsorted[sort_indices] = qvals
    qvals_unsorted[nan_mask] = np.nan

    return qvals_unsorted


def filter_psms_by_qvalue(
    container: ScpContainer,
    assay_name: str = "peptide",
    qvalue_threshold: float = 0.01,
    qvalue_col: str = "qvalue",
) -> ScpContainer:
    """Filter PSMs by q-value threshold for FDR control.

    Retains PSMs with q-value below threshold, controlling FDR.

    Parameters
    ----------
    container : ScpContainer
        ScpContainer containing PSM-level data with q-values in var.
    assay_name : str, default="peptide"
        Name of the assay containing PSM data.
    qvalue_threshold : float, default=0.01
        Maximum q-value [0, 1].
        Recommended: 0.01 (1% FDR, stringent), 0.05 (5% FDR, moderate).
    qvalue_col : str, default="qvalue"
        Column name in assay.var containing q-values.

    Returns
    -------
    ScpContainer
        ScpContainer with high-q-value PSMs removed.

    Examples
    --------
    >>> result = filter_psms_by_qvalue(container, qvalue_threshold=0.01)
    >>> result.assays['peptide'].n_features
    2
    """
    validate_threshold(qvalue_threshold, "qvalue_threshold")
    assay = validate_assay(container, assay_name)
    validate_column_exists(assay, qvalue_col, "Q-value column", "assay.var")

    qvalue_series = assay.var[qvalue_col]
    keep_mask = (qvalue_series < qvalue_threshold).fill_null(False)
    keep_indices = np.where(keep_mask.to_numpy())[0]

    criteria = FilterCriteria.by_indices(keep_indices)
    new_container = container.filter_features(assay_name, criteria)

    # Log provenance
    n_removed = assay.n_features - len(keep_indices)
    percent_removed = (n_removed / assay.n_features * 100) if assay.n_features > 0 else 0

    new_container.log_operation(
        action="filter_psms_by_qvalue",
        params={
            "assay": assay_name,
            "qvalue_threshold": qvalue_threshold,
            "qvalue_col": qvalue_col,
        },
        description=(
            f"Removed {n_removed}/{assay.n_features} PSMs ({percent_removed:.1f}%) "
            f"with {qvalue_col} >= {qvalue_threshold} from assay '{assay_name}'. "
            f"FDR controlled at {qvalue_threshold * 100:.1f}%."
        ),
    )

    return new_container


def compute_sample_carrier_ratio(
    container: ScpContainer,
    assay_name: str = "peptide",
    layer_name: str = "raw",
    carrier_identifier: str = "Carrier",
    sample_identifier: str | None = None,
    max_scr: float = 0.1,
) -> ScpContainer:
    """Compute Sample-to-Carrier Ratio (SCR) for isobaric labeling experiments.

    SCR = sample_intensity / (sample_intensity + carrier_intensity)
    High SCR (>0.1) may indicate overloaded sample or carrier issues.

    Parameters
    ----------
    container : ScpContainer
        ScpContainer with intensity data from isobaric labeling experiment.
    assay_name : str, default="peptide"
        Name of the assay containing PSM or protein data.
    layer_name : str, default="raw"
        Name of the layer containing intensity values.
    carrier_identifier : str, default="Carrier"
        String identifier for carrier channels in sample names.
    sample_identifier : str, optional
        String identifier for sample channels. If None, all non-carriers are samples.
    max_scr : float, default=0.1
        Maximum acceptable SCR threshold.

    Returns
    -------
    ScpContainer
        ScpContainer with SCR metrics added to obs:
        - scr_median: Median SCR across features
        - scr_mean: Mean SCR across features
        - scr_high_psm_count: Number of features with SCR > max_scr

    Examples
    --------
    >>> result = compute_sample_carrier_ratio(container, carrier_identifier="Carrier")
    >>> result.obs[['scr_median', 'scr_mean', 'scr_high_psm_count']]
    """
    from scptensor.core.exceptions import ScpValueError

    assay = validate_assay(container, assay_name)

    if layer_name not in assay.layers:
        available = ", ".join(f"'{k}'" for k in assay.layers.keys())
        raise ScpValueError(
            f"Layer '{layer_name}' not found. Available: {available}.",
            parameter="layer_name",
            value=layer_name,
        )

    sample_names = container.obs["_index"].to_numpy()
    is_carrier = np.array([carrier_identifier in name for name in sample_names])
    carrier_indices = np.where(is_carrier)[0]

    if len(carrier_indices) == 0:
        raise ScpValueError(
            f"No carrier channels found with identifier '{carrier_identifier}'.",
            parameter="carrier_identifier",
            value=carrier_identifier,
        )

    if sample_identifier is not None:
        is_sample = np.array([sample_identifier in name for name in sample_names])
    else:
        is_sample = ~is_carrier

    sample_indices = np.where(is_sample)[0]

    if len(sample_indices) == 0:
        raise ScpValueError(
            "No sample channels found.",
            parameter="sample_identifier",
            value=sample_identifier,
        )

    layer = assay.layers[layer_name]
    X = layer.X

    if sp.issparse(X):
        X = X.toarray()

    carrier_signal = X[carrier_indices, :].sum(axis=0)

    scr_median = []
    scr_mean = []
    scr_high_count = []

    for sample_idx in sample_indices:
        sample_signal = X[sample_idx, :]
        total_signal = sample_signal + carrier_signal

        with np.errstate(divide="ignore", invalid="ignore"):
            scr = np.where(total_signal > 0, sample_signal / total_signal, 0.0)

        scr_median.append(np.nanmedian(scr))
        scr_mean.append(np.nanmean(scr))
        scr_high_count.append(np.sum(scr > max_scr))

    scr_median_all = np.full(len(sample_names), np.nan)
    scr_mean_all = np.full(len(sample_names), np.nan)
    scr_high_count_all = np.full(len(sample_names), 0, dtype=int)

    scr_median_all[sample_indices] = scr_median
    scr_mean_all[sample_indices] = scr_mean
    scr_high_count_all[sample_indices] = scr_high_count

    new_container = container.copy()
    new_container.obs = new_container.obs.with_columns(
        [
            pl.Series("scr_median", scr_median_all),
            pl.Series("scr_mean", scr_mean_all),
            pl.Series("scr_high_psm_count", scr_high_count_all),
        ]
    )

    new_container.log_operation(
        action="compute_sample_carrier_ratio",
        params={
            "assay": assay_name,
            "layer": layer_name,
            "carrier_identifier": carrier_identifier,
            "max_scr": max_scr,
        },
        description=(
            f"Computed SCR for {len(sample_indices)} samples. "
            f"Identified {sum(1 for m in scr_median if m > max_scr)} with SCR > {max_scr}."
        ),
    )

    return new_container


def compute_median_cv(
    container: ScpContainer,
    assay_name: str = "peptide",
    layer_name: str = "raw",
    cv_threshold: float = 0.65,
) -> ScpContainer:
    """Compute median coefficient of variation (CV) across samples.

    CV = σ / μ measures technical variability. High median CV indicates
    poor sample quality or technical issues.

    Parameters
    ----------
    container : ScpContainer
        ScpContainer with intensity data.
    assay_name : str, default="peptide"
        Name of the assay containing data.
    layer_name : str, default="raw"
        Name of the layer containing intensity values.
    cv_threshold : float, default=0.65
        Threshold for flagging high CV samples.

    Returns
    -------
    ScpContainer
        ScpContainer with CV metrics added to obs:
        - median_cv: Median CV across features
        - is_high_cv: Boolean flag if median_cv > cv_threshold

    Examples
    --------
    >>> result = compute_median_cv(container, cv_threshold=0.5)
    >>> result.obs[['median_cv', 'is_high_cv']]
    """
    from scptensor.qc.metrics import compute_cv

    assay = validate_assay(container, assay_name)

    if layer_name not in assay.layers:
        from scptensor.core.exceptions import ScpValueError

        available = ", ".join(f"'{k}'" for k in assay.layers.keys())
        raise ScpValueError(
            f"Layer '{layer_name}' not found. Available: {available}.",
            parameter="layer_name",
            value=layer_name,
        )

    layer = assay.layers[layer_name]
    X = layer.X

    cv_values = compute_cv(X.T, axis=0)

    new_container = container.copy()
    new_container.obs = new_container.obs.with_columns(
        [
            pl.Series("median_cv", cv_values),
            pl.Series("is_high_cv", cv_values > cv_threshold),
        ]
    )

    n_high_cv = int(np.sum(cv_values > cv_threshold))
    median_cv_all = float(np.nanmedian(cv_values))

    new_container.log_operation(
        action="compute_median_cv",
        params={
            "assay": assay_name,
            "layer": layer_name,
            "cv_threshold": cv_threshold,
        },
        description=(
            f"Computed median CV. Median across samples: {median_cv_all:.3f}. "
            f"Identified {n_high_cv} samples with CV > {cv_threshold}."
        ),
    )

    return new_container


def divide_by_reference(
    container: ScpContainer,
    assay_name: str = "peptide",
    layer_name: str = "raw",
    reference_channel: str = "Reference",
    new_layer_name: str = "reference_normalized",
    epsilon: float = 1e-6,
) -> ScpContainer:
    """Normalize channels by dividing by reference channel.

    Used in isobaric labeling experiments (TMT, iTRAQ) to remove
    technical variation between runs.

    Parameters
    ----------
    container : ScpContainer
        ScpContainer with intensity data including reference channel.
    assay_name : str, default="peptide"
        Name of the assay containing data.
    layer_name : str, default="raw"
        Name of the source layer containing raw intensities.
    reference_channel : str, default="Reference"
        Name of the reference channel in obs['_index'].
    new_layer_name : str, default="reference_normalized"
        Name for the new normalized layer.
    epsilon : float, default=1e-6
        Small constant to avoid division by zero.

    Returns
    -------
    ScpContainer
        ScpContainer with new layer containing normalized data.

    Examples
    --------
    >>> result = divide_by_reference(container, reference_channel="Reference")
    >>> X_norm = result.assays['peptide'].layers['reference_normalized'].X
    """
    from scptensor.core.exceptions import ScpValueError

    assay = validate_assay(container, assay_name)

    if layer_name not in assay.layers:
        available = ", ".join(f"'{k}'" for k in assay.layers.keys())
        raise ScpValueError(
            f"Layer '{layer_name}' not found. Available: {available}.",
            parameter="layer_name",
            value=layer_name,
        )

    sample_names = container.obs["_index"].to_numpy()
    reference_matches = np.where(sample_names == reference_channel)[0]

    if len(reference_matches) == 0:
        available = ", ".join(f"'{name}'" for name in sample_names[:5])
        raise ScpValueError(
            f"Reference channel '{reference_channel}' not found. Available: {available}.",
            parameter="reference_channel",
            value=reference_channel,
        )

    if len(reference_matches) > 1:
        raise ScpValueError(
            f"Multiple channels match '{reference_channel}'. Reference must be unique.",
            parameter="reference_channel",
            value=reference_channel,
        )

    reference_idx = reference_matches[0]

    source_layer = assay.layers[layer_name]
    X_source = source_layer.X
    M_source = source_layer.M

    if sp.issparse(X_source):
        X_source = X_source.toarray()

    reference_intensity = X_source[reference_idx, :].reshape(1, -1)
    X_normalized = X_source / (reference_intensity + epsilon)

    new_layer = ScpMatrix(X=X_normalized, M=M_source)

    new_container = container.copy()
    new_container.assays[assay_name].layers[new_layer_name] = new_layer

    n_samples, n_features = X_source.shape
    median_ref_intensity = float(np.nanmedian(reference_intensity))

    new_container.log_operation(
        action="divide_by_reference",
        params={
            "assay": assay_name,
            "source_layer": layer_name,
            "reference_channel": reference_channel,
            "new_layer": new_layer_name,
            "epsilon": epsilon,
        },
        description=(
            f"Normalized {n_samples} channels by reference '{reference_channel}'. "
            f"Created layer '{new_layer_name}' with {n_features} features. "
            f"Reference median intensity: {median_ref_intensity:.2f}."
        ),
    )

    return new_container
