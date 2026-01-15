"""Quality Control module for single-cell proteomics data.

This module provides comprehensive QC operations including:
- Basic QC: Filter samples/features by detection rates
- Advanced QC: Feature filtering, contaminant detection, doublet detection
- Bivariate analysis: Correlation-based QC metrics
- Batch effect detection: Statistical tests for batch effects

Basic Methods (new names):
    - qc_basic: Perform basic quality control calculations
    - qc_score: Compute comprehensive quality score for samples
    - compute_feature_variance: Compute variance statistics for features
    - compute_feature_missing_rate: Compute missing rate statistics for features

Basic Methods (deprecated - use new names):
    - basic_qc: Use qc_basic instead
    - compute_quality_score: Use qc_score instead

Advanced Feature Filtering (new names):
    - filter_features_missing: Filter features with excessive missing values
    - filter_features_variance: Filter features with low variance
    - filter_features_prevalence: Filter features based on detection prevalence

Advanced Feature Filtering (deprecated):
    - filter_features_by_missing_rate: Use filter_features_missing instead
    - filter_features_by_variance: Use filter_features_variance instead
    - filter_features_by_prevalence: Use filter_features_prevalence instead

Advanced Sample Filtering (new names):
    - filter_samples_count: Filter samples based on total intensity
    - filter_samples_missing: Filter samples with excessive missing values

Advanced Sample Filtering (deprecated):
    - filter_samples_by_total_count: Use filter_samples_count instead
    - filter_samples_by_missing_rate: Use filter_samples_missing instead

Detection Methods:
    - detect_outliers: Detect outlier samples using statistical methods
    - detect_contaminants: Detect potential contaminant proteins (new name)
    - detect_contaminant_proteins: Use detect_contaminants instead (deprecated)
    - detect_doublets: Detect potential doublets/multiplets

Bivariate Analysis:
    - compute_pairwise_correlation: Compute pairwise sample correlations
    - detect_outlier_samples: Detect outliers based on sample relationships
    - compute_sample_similarity_network: Build k-nearest neighbor similarity graph

Batch Effect Detection (new names):
    - qc_batch_metrics: Compute batch-level QC metrics

Batch Effect Detection (deprecated):
    - compute_batch_metrics: Use qc_batch_metrics instead

Other Batch Functions:
    - detect_batch_effects: Detect batch effects using statistical tests
    - compute_batch_pca: Compute PCA for batch effect visualization

Metrics:
    - calculate_qc_metrics: Calculate comprehensive QC metrics for samples and features
"""

from scptensor.qc.advanced import (
    calculate_qc_metrics,
    detect_contaminant_proteins,
    detect_contaminants,
    detect_doublets,
    filter_features_by_missing_rate,
    filter_features_by_prevalence,
    filter_features_by_variance,
    filter_features_missing,
    filter_features_prevalence,
    filter_features_variance,
    filter_samples_by_missing_rate,
    filter_samples_by_total_count,
    filter_samples_count,
    filter_samples_missing,
)
from scptensor.qc.basic import (
    basic_qc,
    compute_feature_missing_rate,
    compute_feature_variance,
    compute_quality_score,
    qc_basic,
    qc_score,
)
from scptensor.qc.batch import (
    compute_batch_metrics,
    compute_batch_pca,
    detect_batch_effects,
    qc_batch_metrics,
)
from scptensor.qc.bivariate import (
    compute_pairwise_correlation,
    compute_sample_similarity_network,
    detect_outlier_samples,
)
from scptensor.qc.outlier import detect_outliers

__all__ = [
    # Basic QC (new names)
    "qc_basic",
    "qc_score",
    # Basic QC (backward compatibility)
    "basic_qc",
    "compute_quality_score",
    # Computation functions (kept as-is)
    "compute_feature_variance",
    "compute_feature_missing_rate",
    "detect_outliers",
    # Advanced QC - filtering (new names)
    "filter_features_missing",
    "filter_features_variance",
    "filter_features_prevalence",
    "filter_samples_count",
    "filter_samples_missing",
    # Advanced QC - filtering (backward compatibility)
    "filter_features_by_missing_rate",
    "filter_features_by_variance",
    "filter_features_by_prevalence",
    "filter_samples_by_total_count",
    "filter_samples_by_missing_rate",
    # Advanced QC - detection (new names)
    "detect_contaminants",
    "detect_doublets",
    # Advanced QC - detection (backward compatibility)
    "detect_contaminant_proteins",
    # Advanced QC - metrics
    "calculate_qc_metrics",
    # Bivariate analysis
    "compute_pairwise_correlation",
    "detect_outlier_samples",
    "compute_sample_similarity_network",
    # Batch QC (new names)
    "qc_batch_metrics",
    # Batch QC (backward compatibility)
    "compute_batch_metrics",
    # Other batch functions
    "detect_batch_effects",
    "compute_batch_pca",
]
