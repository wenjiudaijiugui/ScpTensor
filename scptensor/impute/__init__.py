"""
Imputation methods for single-cell proteomics data.

This module provides various missing value imputation algorithms:
- knn: k-Nearest Neighbors imputation
- missforest: Random Forest (MissForest) imputation
- bpca: Bayesian PCA imputation
- lls: Local Least Squares imputation
- qrilc: Quantile Regression Imputation of Left-Censored Data
- minprob: Probabilistic minimum imputation (MNAR)

Unified interface:
    Use `impute()` function for method-agnostic imputation.
"""

# Import individual methods
# Import unified interface
from .base import impute, list_impute_methods
from .bpca import impute_bpca
from .knn import impute_knn
from .lls import impute_lls
from .minprob import impute_minprob
from .missforest import impute_mf
from .qrilc import impute_qrilc

__all__ = [
    # Individual methods
    "impute_knn",
    "impute_bpca",
    "impute_mf",
    "impute_lls",
    "impute_qrilc",
    "impute_minprob",
    # Unified interface
    "impute",
    "list_impute_methods",
]
