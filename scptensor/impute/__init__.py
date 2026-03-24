"""Imputation methods for DIA-based single-cell proteomics data.

This module provides various missing value imputation algorithms:
- none: no-op passthrough (explicit no-imputation baseline)
- zero: fill with zeros
- row_mean: fill with sample-wise row means
- row_median: fill with sample-wise row medians
- half_row_min: fill with fraction * sample-wise row minimum
- knn: k-Nearest Neighbors imputation
- missforest: Random Forest (MissForest) imputation
- bpca: Bayesian PCA imputation
- lls: Local Least Squares imputation
- iterative_svd: Iterative low-rank SVD imputation
- softimpute: Nuclear-norm regularized matrix completion (optional dependency)
- qrilc: Quantile Regression Imputation of Left-Censored Data
- minprob: Probabilistic minimum imputation (MNAR)

Unified interface:
    Use `impute()` function for method-agnostic imputation.

Registry and mechanism helpers live in ``scptensor.impute.base`` and are not
re-exported from the package root.
"""

# Import individual methods
# Import unified interface
from .base import impute
from .baseline import (
    impute_half_row_min,
    impute_none,
    impute_row_mean,
    impute_row_median,
    impute_zero,
)
from .bpca import impute_bpca
from .knn import impute_knn
from .lls import impute_lls
from .minprob import impute_minprob
from .missforest import impute_mf
from .qrilc import impute_qrilc
from .svd import impute_iterative_svd, impute_softimpute

__all__ = [
    # Individual methods
    "impute_none",
    "impute_zero",
    "impute_row_mean",
    "impute_row_median",
    "impute_half_row_min",
    "impute_knn",
    "impute_bpca",
    "impute_mf",
    "impute_lls",
    "impute_iterative_svd",
    "impute_softimpute",
    "impute_qrilc",
    "impute_minprob",
    # Unified interface
    "impute",
]
