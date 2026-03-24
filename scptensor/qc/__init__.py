"""Quality Control (QC) module for DIA-based single-cell proteomics.

Provides hierarchical QC workflow:
1. Sample Level (qc_sample): Filter cells (Empty wells, Doublets via MAD)
2. Feature Level (qc_feature): Filter proteins (Missingness, CV)

Note:
----
Peptide/PSM-level QC helpers are kept in source form but are not part of the
stable preprocessing contract. Import them from
``from scptensor.experimental import qc_psm`` rather than the stable
``scptensor.qc`` namespace.

"""

from scptensor.qc import qc_feature, qc_sample
from scptensor.qc.qc_feature import (
    calculate_feature_qc_metrics,
    filter_features_by_cv,
    filter_features_by_missingness,
)
from scptensor.qc.qc_sample import (
    assess_batch_effects,
    calculate_sample_qc_metrics,
    filter_doublets_mad,
    filter_low_quality_samples,
)

__all__ = [
    # Submodules
    "qc_sample",
    "qc_feature",
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
