"""Harmony integration wrapper for DIA-based single-cell proteomics data.

This module re-exports the Harmony implementation from nonlinear.py
with the new integrate_* naming convention.

Reference
---------
Korsunsky I, et al. Fast, sensitive and accurate integration of
single-cell data with Harmony. Nature Methods (2019).
"""

from scptensor.integration.nonlinear import integrate_harmony

__all__ = ["integrate_harmony"]
