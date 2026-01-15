"""
Real data comparison tests module.

This module contains utilities for converting real single-cell proteomics
data to h5ad format for comparison testing between ScpTensor and other tools.
"""

from .data_prep import (
    convert_to_h5ad,
    load_experimental_design,
    match_samples_to_design,
    parse_spectronaut_tsv,
)
from .comparison_suite import (
    ComparisonSuite,
    run_phase1_tests,
)

__all__ = [
    "parse_spectronaut_tsv",
    "load_experimental_design",
    "match_samples_to_design",
    "convert_to_h5ad",
    "ComparisonSuite",
    "run_phase1_tests",
]
