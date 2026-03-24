"""PSM-level Quality Control.

Handles filtering and QC for Peptide Spectrum Matches (PSMs):
- Contaminant filtering (keratins, trypsin, albumin, etc.)
- PIF (Parent Ion Fraction) filtering for co-elution detection
- FDR control via PEP to q-value conversion
- Sample-level CV summary across peptide/PSM features
"""

from typing import Literal

import numpy as np
import polars as pl

from scptensor.core._structure_container import ScpContainer
from scptensor.core.types import JsonValue
from scptensor.qc._utils import (
    filter_features_with_provenance,
    resolve_assay,
    validate_column_exists,
    validate_threshold,
)

# Internal default contaminant patterns for proteomics. The stable contract is
# `patterns=None` in `filter_contaminants`, not direct reuse of this regex list.
_DEFAULT_CONTAMINANT_PATTERNS = [
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
    assay_name: str = "peptides",
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
    assay_name : str, default="peptides"
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
    >>> result.assays['peptides'].n_features
    2

    """
    validate_threshold(min_pif, "min_pif", min_val=0.0, max_val=1.0)
    resolved_assay_name, assay = resolve_assay(container, assay_name)
    validate_column_exists(assay.var, pif_col, "assay.var")

    pif_series = assay.var[pif_col]
    keep_mask = (pif_series >= min_pif).fill_null(False)
    keep_indices = np.where(keep_mask.to_numpy())[0]

    n_removed = assay.n_features - len(keep_indices)
    percent_removed = (n_removed / assay.n_features * 100) if assay.n_features > 0 else 0

    return filter_features_with_provenance(
        container,
        resolved_assay_name,
        keep_indices,
        action="filter_psms_by_pif",
        params={"assay": resolved_assay_name, "min_pif": min_pif, "pif_col": pif_col},
        description=(
            f"Removed {n_removed}/{assay.n_features} PSMs ({percent_removed:.1f}%) "
            f"with {pif_col} < {min_pif} from assay '{resolved_assay_name}'."
        ),
    )


def filter_contaminants(
    container: ScpContainer,
    assay_name: str = "peptides",
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
    assay_name : str, default="peptides"
        Name of the assay containing features to filter.
    feature_col : str, default="gene_names"
        Column name in assay.var containing feature names.
    patterns : list of str, optional
        List of regex patterns for contaminants. If None, uses the module's
        built-in default contaminant patterns.

    Returns
    -------
    ScpContainer
        ScpContainer with contaminant features removed.

    Examples
    --------
    >>> result = filter_contaminants(container, feature_col='gene_names')
    >>> result.assays['peptides'].n_features
    3

    Custom patterns:
    >>> custom = [r'^FLAG_', r'^HA_']
    >>> result = filter_contaminants(container, patterns=custom)

    """
    resolved_assay_name, assay = resolve_assay(container, assay_name)
    validate_column_exists(assay.var, feature_col, "assay.var")

    regex_patterns = patterns or _DEFAULT_CONTAMINANT_PATTERNS
    combined_pattern = "|".join(f"({p})" for p in regex_patterns)

    is_contaminant = assay.var[feature_col].str.contains(combined_pattern).fill_null(False)
    keep_mask = ~is_contaminant
    keep_indices = np.where(keep_mask.to_numpy())[0]

    n_removed = assay.n_features - len(keep_indices)
    percent_removed = (n_removed / assay.n_features * 100) if assay.n_features > 0 else 0

    pattern_preview: list[JsonValue] = [str(p) for p in regex_patterns[:5]]
    return filter_features_with_provenance(
        container,
        resolved_assay_name,
        keep_indices,
        action="filter_contaminants",
        params={
            "assay": resolved_assay_name,
            "feature_col": feature_col,
            "patterns_count": len(regex_patterns),
            "patterns": pattern_preview,
        },
        description=(
            f"Removed {n_removed}/{assay.n_features} features ({percent_removed:.1f}%) "
            f"matching {len(regex_patterns)} contaminant patterns "
            f"from assay '{resolved_assay_name}'."
        ),
    )


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
        pi0 = float(np.sum(pep > lambda_param) / (n * (1 - lambda_param)))
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
    assay_name: str = "peptides",
    qvalue_threshold: float = 0.01,
    qvalue_col: str = "qvalue",
) -> ScpContainer:
    """Filter PSMs by q-value threshold for FDR control.

    Retains PSMs with q-value below threshold, controlling FDR.

    Parameters
    ----------
    container : ScpContainer
        ScpContainer containing PSM-level data with q-values in var.
    assay_name : str, default="peptides"
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
    >>> result.assays['peptides'].n_features
    2

    """
    validate_threshold(qvalue_threshold, "qvalue_threshold", min_val=0.0, max_val=1.0)
    resolved_assay_name, assay = resolve_assay(container, assay_name)
    validate_column_exists(assay.var, qvalue_col, "assay.var")

    qvalue_series = assay.var[qvalue_col]
    keep_mask = (qvalue_series < qvalue_threshold).fill_null(False)
    keep_indices = np.where(keep_mask.to_numpy())[0]

    n_removed = assay.n_features - len(keep_indices)
    percent_removed = (n_removed / assay.n_features * 100) if assay.n_features > 0 else 0

    return filter_features_with_provenance(
        container,
        resolved_assay_name,
        keep_indices,
        action="filter_psms_by_qvalue",
        params={
            "assay": resolved_assay_name,
            "qvalue_threshold": qvalue_threshold,
            "qvalue_col": qvalue_col,
        },
        description=(
            f"Removed {n_removed}/{assay.n_features} PSMs ({percent_removed:.1f}%) "
            f"with {qvalue_col} >= {qvalue_threshold} from assay '{resolved_assay_name}'. "
            f"FDR controlled at {qvalue_threshold * 100:.1f}%."
        ),
    )


def _compute_sample_cv_summary(
    container: ScpContainer,
    assay_name: str,
    layer_name: str,
    cv_threshold: float,
    *,
    action: str,
    cv_col: str,
    flag_col: str,
    alias_note: str | None = None,
) -> ScpContainer:
    """Internal helper for sample-level CV summary annotation."""
    from scptensor.qc.metrics import compute_cv

    resolved_assay_name, assay = resolve_assay(container, assay_name)

    if layer_name not in assay.layers:
        from scptensor.core.exceptions import ScpValueError

        available = ", ".join(f"'{k}'" for k in assay.layers.keys())
        raise ScpValueError(
            f"Layer '{layer_name}' not found. Available: {available}.",
            parameter="layer_name",
            value=layer_name,
        )

    layer = assay.layers[layer_name]
    cv_values = compute_cv(layer.X.T, axis=0)

    new_container = container.copy()
    new_container.obs = new_container.obs.with_columns(
        [
            pl.Series(cv_col, cv_values),
            pl.Series(flag_col, cv_values > cv_threshold),
        ],
    )

    n_high_cv = int(np.sum(cv_values > cv_threshold))
    median_cv_all = float(np.nanmedian(cv_values))
    description_prefix = "Computed sample CV across features."
    if alias_note is not None:
        description_prefix = f"Computed sample CV across features via legacy alias '{alias_note}'."

    new_container.log_operation(
        action=action,
        params={
            "assay": resolved_assay_name,
            "layer": layer_name,
            "cv_threshold": cv_threshold,
        },
        description=(
            f"{description_prefix} Median across samples: {median_cv_all:.3f}. "
            f"Identified {n_high_cv} samples with CV > {cv_threshold}."
        ),
    )

    return new_container


def compute_sample_cv(
    container: ScpContainer,
    assay_name: str = "peptides",
    layer_name: str = "raw",
    cv_threshold: float = 0.65,
) -> ScpContainer:
    """Compute per-sample coefficient of variation (CV) across features.

    CV = σ / μ measures technical variability. High sample CV indicates
    poor sample quality or technical issues.

    Parameters
    ----------
    container : ScpContainer
        ScpContainer with intensity data.
    assay_name : str, default="peptides"
        Name of the assay containing data.
    layer_name : str, default="raw"
        Name of the layer containing intensity values.
    cv_threshold : float, default=0.65
        Threshold for flagging high CV samples.

    Returns
    -------
    ScpContainer
        ScpContainer with sample-level CV metrics added to obs:
        - sample_cv: CV for each sample across features
        - is_high_sample_cv: Boolean flag if sample_cv > cv_threshold

    Examples
    --------
    >>> result = compute_sample_cv(container, cv_threshold=0.5)
    >>> result.obs[['sample_cv', 'is_high_sample_cv']]

    """
    return _compute_sample_cv_summary(
        container,
        assay_name,
        layer_name,
        cv_threshold,
        action="compute_sample_cv",
        cv_col="sample_cv",
        flag_col="is_high_sample_cv",
    )


__all__ = [
    "filter_psms_by_pif",
    "filter_contaminants",
    "pep_to_qvalue",
    "filter_psms_by_qvalue",
    "compute_sample_cv",
]
