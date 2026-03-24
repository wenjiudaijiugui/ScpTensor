"""Metrics module for automatic method selection.

This module provides various metrics for evaluating data processing
effectiveness in the automatic method selection system.

Available Metrics
-----------------
Quality Metrics (quality.py)
    cv_stability
        Coefficient of variation stability across features
    skewness_improvement
        Improvement in distribution skewness after transformation
    kurtosis_improvement
        Improvement in distribution kurtosis after transformation
    dynamic_range
        Appropriateness of data dynamic range
    outlier_ratio
        Proportion of non-outlier values

Batch Effect Metrics (batch.py)
    batch_asw
        Batch average silhouette width (1 - ASW, higher is better)
    bio_asw
        Biological group average silhouette width (higher is better)
    batch_mixing_score
        Heuristic local batch-mixing proxy
    kbet_score
        Fixed-k kBET acceptance rate
    ilisi_score
        Perplexity-weighted iLISI summary

Clustering Metrics (clustering.py)
    silhouette_score
        Silhouette coefficient for clustering quality
    calinski_harabasz_score
        Calinski-Harabasz index (normalized)
    davies_bouldin_score
        Davies-Bouldin index (1 - DB, higher is better)
    clustering_stability
        Clustering stability through subsampling

All metrics return values in the range [0, 1], where higher values
indicate better quality.

Example:
-------
>>> import numpy as np
>>> from scptensor.autoselect.metrics import (
...     cv_stability, outlier_ratio,
...     batch_asw, bio_asw,
...     silhouette_score, clustering_stability
... )
>>> X = np.random.randn(100, 10)
>>> batch_labels = np.repeat([0, 1], 50)
>>> bio_labels = np.repeat([0, 1, 2, 3], 25)
>>> stability = cv_stability(X)
>>> batch_score = batch_asw(X, batch_labels)
>>> bio_score = bio_asw(X, bio_labels)
>>> cluster_score = silhouette_score(X, bio_labels)
>>> print(f"Stability: {stability:.3f}, Batch: {batch_score:.3f}")

"""

from __future__ import annotations

from scptensor.autoselect.metrics.batch import (
    batch_asw,
    batch_mixing_score,
    bio_asw,
    ilisi_score,
    kbet_score,
)
from scptensor.autoselect.metrics.clustering import (
    calinski_harabasz_score,
    clustering_stability,
    davies_bouldin_score,
    silhouette_score,
)
from scptensor.autoselect.metrics.quality import (
    cv_stability,
    dynamic_range,
    kurtosis_improvement,
    outlier_ratio,
    skewness_improvement,
)

__all__ = [
    # Quality metrics
    "cv_stability",
    "skewness_improvement",
    "kurtosis_improvement",
    "dynamic_range",
    "outlier_ratio",
    # Batch effect metrics
    "batch_asw",
    "bio_asw",
    "batch_mixing_score",
    "kbet_score",
    "ilisi_score",
    # Clustering metrics
    "silhouette_score",
    "calinski_harabasz_score",
    "davies_bouldin_score",
    "clustering_stability",
]
