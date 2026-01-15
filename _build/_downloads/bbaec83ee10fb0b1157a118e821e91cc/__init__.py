"""
Quality Control module for single-cell proteomics data.

This module provides methods for performing quality control on SCP data,
including basic QC metrics and outlier detection.

Available Methods:
- basic_qc: Perform basic quality control calculations
- detect_outliers: Detect outlier samples using statistical methods
"""

from scptensor.qc.basic import basic_qc
from scptensor.qc.outlier import detect_outliers

__all__ = [
    "basic_qc",
    "detect_outliers",
]
