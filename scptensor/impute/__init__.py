"""Imputation methods for single-cell proteomics data.

This module provides various missing value imputation algorithms:
- impute_knn: k-Nearest Neighbors imputation
- impute_ppca: Probabilistic PCA imputation
- impute_svd: Iterative SVD imputation
- impute_mf: Random Forest (MissForest) imputation

Backward Compatibility:
========================
The following deprecated aliases are maintained for backward compatibility:
- knn -> impute_knn
- ppca -> impute_ppca
- svd_impute -> impute_svd
- missforest -> impute_mf

These will be removed in version 1.0.0.
"""

from .knn import impute_knn, knn
from .missforest import impute_mf, missforest
from .ppca import impute_ppca, ppca
from .svd import impute_svd, svd_impute

__all__ = [
    # New API (impute_* prefix)
    "impute_knn",
    "impute_ppca",
    "impute_svd",
    "impute_mf",
    # Deprecated aliases (for backward compatibility)
    "knn",
    "ppca",
    "svd_impute",
    "missforest",
]
