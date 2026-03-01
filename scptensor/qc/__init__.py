"""Quality Control (QC) module for single-cell proteomics.

Provides hierarchical QC workflow:
1. PSM Level (qc_psm): Filter raw peptide matches (PIF, Contaminants, FDR)
2. Sample Level (qc_sample): Filter cells (Empty wells, Doublets via MAD)
3. Feature Level (qc_feature): Filter proteins (Missingness, CV)
"""

from scptensor.qc import qc_feature, qc_psm, qc_sample
from scptensor.qc.qc_feature import (
    calculate_feature_qc_metrics,
    filter_features_by_cv,
    filter_features_by_missingness,
)
from scptensor.qc.qc_psm import (
    DEFAULT_CONTAMINANT_PATTERNS,
    compute_median_cv,
    compute_sample_carrier_ratio,
    divide_by_reference,
    filter_contaminants,
    filter_psms_by_pif,
    filter_psms_by_qvalue,
    pep_to_qvalue,
)
from scptensor.qc.qc_sample import (
    assess_batch_effects,
    calculate_sample_qc_metrics,
    filter_doublets_mad,
    filter_low_quality_samples,
)

__all__ = [
    # Submodules
    "qc_psm",
    "qc_sample",
    "qc_feature",
    # PSM functions
    "filter_contaminants",
    "filter_psms_by_pif",
    "filter_psms_by_qvalue",
    "pep_to_qvalue",
    "compute_sample_carrier_ratio",
    "compute_median_cv",
    "divide_by_reference",
    "DEFAULT_CONTAMINANT_PATTERNS",
    # Sample functions
    "calculate_sample_qc_metrics",
    "filter_low_quality_samples",
    "filter_doublets_mad",
    "assess_batch_effects",
    # Feature functions
    "calculate_feature_qc_metrics",
    "filter_features_by_missingness",
    "filter_features_by_cv",
]
