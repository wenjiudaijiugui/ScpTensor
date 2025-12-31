from typing import Optional, Union
import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors

def compute_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Silhouette Score.
    Range: [-1, 1].
    """
    if len(np.unique(labels)) < 2:
        return np.nan
    # Sample if too large to save time? For now, full compute.
    if X.shape[0] > 10000:
        # Downsample for speed
        indices = np.random.choice(X.shape[0], 10000, replace=False)
        return silhouette_score(X[indices], labels[indices])
    return silhouette_score(X, labels)

def compute_batch_mixing(X: np.ndarray, batch_labels: np.ndarray) -> float:
    """
    Evaluate batch mixing using 1 - |Silhouette Score|.
    
    Silhouette Score for batch labels:
    - High positive: Batches are well separated (Bad for integration).
    - Near 0: Batches are overlapping (Good for integration).
    - Negative: Batches are mixed (Good).
    
    We want a metric where Higher is Better.
    
    Let's use: 1 - abs(silhouette_score)
    If silhouette is 0.5 (separated), score is 0.5.
    If silhouette is 0.0 (mixed), score is 1.0.
    """
    sil = compute_silhouette(X, batch_labels)
    if np.isnan(sil):
        return 0.0
    
    # Ideally we want silhouette to be close to 0.
    # But usually it's positive if batches are distinct.
    # If batches are perfectly mixed, expected silhouette is 0.
    
    return 1.0 - abs(sil)

def compute_cluster_separation(X: np.ndarray, cluster_labels: np.ndarray) -> float:
    """
    Evaluate cluster separation using Silhouette Score.
    Higher is better.
    """
    sil = compute_silhouette(X, cluster_labels)
    if np.isnan(sil):
        return 0.0
    return sil

def compute_ari(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Adjusted Rand Index.
    """
    return adjusted_rand_score(labels_true, labels_pred)

def compute_reconstruction_error(X_orig: np.ndarray, X_recon: np.ndarray) -> float:
    """
    Root Mean Squared Error.
    """
    return np.sqrt(np.nanmean((X_orig - X_recon) ** 2))
