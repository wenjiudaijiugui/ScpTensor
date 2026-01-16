"""Quality Control module for single-cell proteomics data.

This module provides comprehensive QC operations including:
- Basic QC: Filter samples/features by detection rates
- Advanced QC: Feature filtering, contaminant detection, doublet detection
- Bivariate analysis: Correlation-based QC metrics
- Batch effect detection: Statistical tests for batch effects
- Sensitivity metrics: Total/local feature detection, completeness, Jaccard index
"""

from scptensor.qc.advanced import (
    calculate_qc_metrics,
    detect_contaminants,
    detect_doublets,
    filter_features_missing,
    filter_features_prevalence,
    filter_features_variance,
    filter_samples_count,
    filter_samples_missing,
)
from scptensor.qc.basic import (
    compute_feature_missing_rate,
    compute_feature_variance,
    qc_basic,
    qc_score,
)
from scptensor.qc.batch import (
    compute_batch_pca,
    detect_batch_effects,
    qc_batch_metrics,
)
from scptensor.qc.bivariate import (
    compute_pairwise_correlation,
    compute_sample_similarity_network,
    detect_outlier_samples,
)
from scptensor.qc.missing import (
    analyze_missing_types,
    compute_missing_stats,
    report_missing_values,
)
from scptensor.qc.outlier import detect_outliers
from scptensor.qc.sensitivity import (
    compute_completeness,
    compute_cumulative_sensitivity,
    compute_jaccard_index,
    compute_sensitivity,
    qc_report_metrics,
)

# Variability statistics
from scptensor.qc.variability import (
    CVReport,
    compute_batch_cv,
    compute_cv,
    compute_technical_replicate_cv,
    filter_by_cv,
)

__all__ = [
    # Basic QC
    "qc_basic",
    "qc_score",
    "compute_feature_variance",
    "compute_feature_missing_rate",
    "detect_outliers",
    # Advanced QC - filtering
    "filter_features_missing",
    "filter_features_variance",
    "filter_features_prevalence",
    "filter_samples_count",
    "filter_samples_missing",
    # Advanced QC - detection
    "detect_contaminants",
    "detect_doublets",
    # Advanced QC - metrics
    "calculate_qc_metrics",
    # Bivariate analysis
    "compute_pairwise_correlation",
    "detect_outlier_samples",
    "compute_sample_similarity_network",
    # Batch QC
    "qc_batch_metrics",
    "detect_batch_effects",
    "compute_batch_pca",
    # Sensitivity metrics
    "compute_sensitivity",
    "compute_completeness",
    "compute_jaccard_index",
    "compute_cumulative_sensitivity",
    "qc_report_metrics",
    # Missing value analysis
    "analyze_missing_types",
    "compute_missing_stats",
    "report_missing_values",
    # Variability statistics
    "CVReport",
    "compute_cv",
    "compute_technical_replicate_cv",
    "compute_batch_cv",
    "filter_by_cv",
]
