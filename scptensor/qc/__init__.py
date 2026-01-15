"""Quality Control module for single-cell proteomics data.

This module provides methods for performing quality control on SCP data,
including basic QC metrics, outlier detection, and advanced filtering methods.

Basic Methods:
    - basic_qc: Perform basic quality control calculations
    - compute_quality_score: Compute comprehensive quality score for samples
    - compute_feature_variance: Compute variance statistics for features
    - compute_feature_missing_rate: Compute missing rate statistics for features

Advanced Feature Filtering:
    - filter_features_by_missing_rate: Filter features with excessive missing values
    - filter_features_by_variance: Filter features with low variance
    - filter_features_by_prevalence: Filter features based on detection prevalence

Advanced Sample Filtering:
    - filter_samples_by_total_count: Filter samples based on total intensity
    - filter_samples_by_missing_rate: Filter samples with excessive missing values

Detection Methods:
    - detect_outliers: Detect outlier samples using statistical methods
    - detect_contaminant_proteins: Detect potential contaminant proteins
    - detect_doublets: Detect potential doublets/multiplets

Bivariate Analysis:
    - compute_pairwise_correlation: Compute pairwise sample correlations
    - detect_outlier_samples: Detect outliers based on sample relationships
    - compute_sample_similarity_network: Build k-nearest neighbor similarity graph

Batch Effect Detection:
    - compute_batch_metrics: Compute batch-level QC metrics
    - detect_batch_effects: Detect batch effects using statistical tests
    - compute_batch_pca: Compute PCA for batch effect visualization

Metrics:
    - calculate_qc_metrics: Calculate comprehensive QC metrics for samples and features
"""

from scptensor.qc.advanced import (
    calculate_qc_metrics,
    detect_contaminant_proteins,
    detect_doublets,
    filter_features_by_missing_rate,
    filter_features_by_prevalence,
    filter_features_by_variance,
    filter_samples_by_missing_rate,
    filter_samples_by_total_count,
)
from scptensor.qc.batch import (
    compute_batch_metrics,
    compute_batch_pca,
    detect_batch_effects,
)
from scptensor.qc.basic import (
    basic_qc,
    compute_feature_missing_rate,
    compute_feature_variance,
    compute_quality_score,
)
from scptensor.qc.bivariate import (
    compute_pairwise_correlation,
    compute_sample_similarity_network,
    detect_outlier_samples,
)
from scptensor.qc.outlier import detect_outliers

__all__ = [
    # Basic methods
    "basic_qc",
    "compute_quality_score",
    "compute_feature_variance",
    "compute_feature_missing_rate",
    "detect_outliers",
    # Advanced feature filtering
    "filter_features_by_missing_rate",
    "filter_features_by_variance",
    "filter_features_by_prevalence",
    # Advanced sample filtering
    "filter_samples_by_total_count",
    "filter_samples_by_missing_rate",
    # Detection methods
    "detect_contaminant_proteins",
    "detect_doublets",
    # Metrics
    "calculate_qc_metrics",
    # Bivariate analysis
    "compute_pairwise_correlation",
    "detect_outlier_samples",
    "compute_sample_similarity_network",
    # Batch effect detection
    "compute_batch_metrics",
    "detect_batch_effects",
    "compute_batch_pca",
]
