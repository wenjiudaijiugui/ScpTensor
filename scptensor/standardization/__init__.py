"""Standardization module for DIA-based single-cell proteomics data.

This module provides feature-scaling methods that transform data to a common
statistical scale.
"""

from .zscore import zscore

__all__ = ["zscore"]
