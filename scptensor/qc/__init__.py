"""Quality Control module for single-cell proteomics data.

This module provides methods for performing quality control on SCP data,
including basic QC metrics, outlier detection, and advanced filtering methods.

Basic Methods:
    - basic_qc: Perform basic quality control calculations
    - detect_outliers: Detect outlier samples using statistical methods

Advanced Feature Filtering:
    - filter_features_by_missing_rate: Filter features with excessive missing values
    - filter_features_by_variance: Filter features with low variance
    - filter_features_by_prevalence: Filter features based on detection prevalence

Advanced Sample Filtering:
    - filter_samples_by_total_count: Filter samples based on total intensity
    - filter_samples_by_missing_rate: Filter samples with excessive missing values

Detection Methods:
    - detect_contaminant_proteins: Detect potential contaminant proteins
    - detect_doublets: Detect potential doublets/multiplets

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
from scptensor.qc.basic import basic_qc
from scptensor.qc.outlier import detect_outliers

__all__ = [
    # Basic methods
    "basic_qc",
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
]
