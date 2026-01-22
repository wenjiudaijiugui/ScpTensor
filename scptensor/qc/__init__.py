"""Quality Control (QC) module for single-cell proteomics.

This module provides a hierarchical QC workflow:
1. PSM Level (qc_psm): Filter raw peptide matches (PIF, Contaminants).
2. Sample Level (qc_sample): Filter cells (Empty wells, Doublets via MAD).
3. Feature Level (qc_feature): Filter proteins (Missingness, CV).

Common Usage:
    >>> from scptensor.qc import qc_psm, qc_sample, qc_feature
    >>>
    >>> # 1. PSM QC
    >>> container = qc_psm.filter_contaminants(container)
    >>> container = qc_psm.filter_psms_by_pif(container, min_pif=0.8)
    >>>
    >>> # 2. Sample QC
    >>> container = qc_sample.calculate_sample_qc_metrics(container)
    >>> container = qc_sample.filter_low_quality_samples(container, nmads=3)
    >>> container = qc_sample.filter_doublets_mad(container, nmads=3)
    >>>
    >>> # 3. Feature QC
    >>> container = qc_feature.filter_features_by_missingness(container, max_missing_rate=0.5)
"""

from scptensor.qc import qc_feature, qc_psm, qc_sample
from scptensor.qc.qc_feature import (
    calculate_feature_qc_metrics,
    filter_features_by_cv,
    filter_features_by_missingness,
)
from scptensor.qc.qc_psm import (
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
    # Functions
    "filter_contaminants",
    "filter_psms_by_pif",
    "pep_to_qvalue",
    "filter_psms_by_qvalue",
    "compute_sample_carrier_ratio",
    "compute_median_cv",
    "divide_by_reference",
    "calculate_sample_qc_metrics",
    "filter_low_quality_samples",
    "filter_doublets_mad",
    "assess_batch_effects",
    "calculate_feature_qc_metrics",
    "filter_features_by_missingness",
    "filter_features_by_cv",
]
