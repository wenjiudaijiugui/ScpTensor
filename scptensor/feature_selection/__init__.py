"""Feature selection methods for single-cell proteomics data.

This module provides various feature selection techniques commonly used
in single-cell analysis to identify the most informative proteins/peptides.

Available Methods
-----------------
- HVG (Highly Variable Genes/Proteins): Select based on coefficient of variation
- Dropout-based: Filter features by missing data rate
- VST (Variance Stabilizing Transformation): Seurat-style variable feature selection
- Dispersion-based: Normalized dispersion ranking
- Model-based: Random forest or variance threshold importance
- PCA loading-based: Select by principal component contribution

Examples
--------
>>> from scptensor.feature_selection import select_hvg, select_by_dropout
>>>
>>> # Select top 2000 highly variable proteins
>>> container_hvg = select_hvg(container, n_top_features=2000)
>>>
>>> # Filter features with high dropout rate
>>> container_filtered = select_by_dropout(container, max_dropout_rate=0.5)
"""

from scptensor.feature_selection.dropout import (
    get_dropout_stats,
    select_by_dropout,
)
from scptensor.feature_selection.hvg import select_hvg
from scptensor.feature_selection.model import (
    select_by_model_importance,
    select_by_pca_loadings,
)
from scptensor.feature_selection.vst import (
    select_by_dispersion,
    select_by_vst,
)

__all__ = [
    # HVG selection
    "select_hvg",
    # Dropout-based selection
    "select_by_dropout",
    "get_dropout_stats",
    # VST-based selection
    "select_by_vst",
    "select_by_dispersion",
    # Model-based selection
    "select_by_model_importance",
    "select_by_pca_loadings",
]
