"""Clustering metrics evaluator for algorithm quality assessment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import spmatrix
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from .biological import BaseEvaluator

if TYPE_CHECKING:
    from collections.abc import Sequence

ArrayFloat = NDArray[np.float64]
ArrayInt = NDArray[np.intp]


def compute_clustering_ari(
    labels_true: ArrayInt | Sequence[int],
    labels_pred: ArrayInt | Sequence[int],
) -> float:
    """Compute Adjusted Rand Index (ARI) for clustering evaluation.

    The Adjusted Rand Index measures the similarity between two clusterings,
    adjusted for the chance grouping of elements. It ranges from -1 to 1,
    where 1 indicates perfect match, 0 indicates random labeling, and
    negative values indicate worse than random.

    Parameters
    ----------
    labels_true : ArrayInt | Sequence[int]
        True cluster labels (ground truth)
    labels_pred : ArrayInt | Sequence[int]
        Predicted cluster labels

    Returns
    -------
    float
        ARI value in range [-1, 1], where 1 indicates perfect agreement

    Raises
    ------
    ValueError
        If labels_true and labels_pred have different lengths

    Examples
    --------
    >>> true_labels = np.array([0, 0, 1, 1, 2, 2])
    >>> pred_labels = np.array([0, 0, 1, 2, 2, 2])
    >>> ari = compute_clustering_ari(true_labels, pred_labels)
    >>> print(f"ARI: {ari:.3f}")
    """
    labels_true_arr = np.asarray(labels_true).ravel()
    labels_pred_arr = np.asarray(labels_pred).ravel()

    if labels_true_arr.shape != labels_pred_arr.shape:
        msg = (
            f"Shape mismatch: labels_true={labels_true_arr.shape}, "
            f"labels_pred={labels_pred_arr.shape}"
        )
        raise ValueError(msg)

    if labels_true_arr.size == 0:
        return np.nan

    try:
        return float(adjusted_rand_score(labels_true_arr, labels_pred_arr))
    except Exception as e:
        msg = f"Failed to compute ARI: {e!s}"
        raise ValueError(msg) from e


def compute_clustering_nmi(
    labels_true: ArrayInt | Sequence[int],
    labels_pred: ArrayInt | Sequence[int],
    average_method: str = "arithmetic",
) -> float:
    """Compute Normalized Mutual Information (NMI) for clustering evaluation.

    NMI measures the mutual information between two clusterings, normalized
    by the entropy of each clustering. It ranges from 0 to 1, where 1 indicates
    perfect match and 0 indicates no mutual information.

    Parameters
    ----------
    labels_true : ArrayInt | Sequence[int]
        True cluster labels (ground truth)
    labels_pred : ArrayInt | Sequence[int]
        Predicted cluster labels
    average_method : str, default='arithmetic'
        Method for averaging normalization term. Options:
        - 'arithmetic': Arithmetic mean
        - 'geometric': Geometric mean
        - 'min': Minimum of entropies
        - 'max': Maximum of entropies

    Returns
    -------
    float
        NMI value in range [0, 1], where 1 indicates perfect agreement

    Raises
    ------
    ValueError
        If labels_true and labels_pred have different lengths
        If average_method is not valid

    Examples
    --------
    >>> true_labels = np.array([0, 0, 1, 1, 2, 2])
    >>> pred_labels = np.array([0, 0, 1, 2, 2, 2])
    >>> nmi = compute_clustering_nmi(true_labels, pred_labels)
    >>> print(f"NMI: {nmi:.3f}")
    """
    valid_average_methods = {"arithmetic", "geometric", "min", "max"}
    if average_method not in valid_average_methods:
        msg = f"Invalid average_method: {average_method}. Must be one of {valid_average_methods}"
        raise ValueError(msg)

    labels_true_arr = np.asarray(labels_true).ravel()
    labels_pred_arr = np.asarray(labels_pred).ravel()

    if labels_true_arr.shape != labels_pred_arr.shape:
        msg = (
            f"Shape mismatch: labels_true={labels_true_arr.shape}, "
            f"labels_pred={labels_pred_arr.shape}"
        )
        raise ValueError(msg)

    if labels_true_arr.size == 0:
        return np.nan

    try:
        return float(
            normalized_mutual_info_score(
                labels_true_arr,
                labels_pred_arr,
                average_method=average_method,
            )
        )
    except Exception as e:
        msg = f"Failed to compute NMI: {e!s}"
        raise ValueError(msg) from e


def compute_clustering_silhouette(
    X: ArrayFloat | spmatrix,
    labels: ArrayInt | Sequence[int],
    metric: str = "euclidean",
) -> float:
    """Compute Silhouette Score for clustering evaluation.

    The Silhouette Score measures how well-separated the clusters are.
    It ranges from -1 to 1, where 1 indicates well-separated clusters,
    0 indicates overlapping clusters, and -1 indicates incorrectly clustered data.

    Parameters
    ----------
    X : ArrayFloat | spmatrix
        Feature matrix of shape (n_samples, n_features)
    labels : ArrayInt | Sequence[int]
        Cluster labels
    metric : str, default='euclidean'
        Distance metric to use. Common options:
        - 'euclidean': Euclidean distance
        - 'manhattan': Manhattan distance
        - 'cosine': Cosine distance
        - 'correlation': Correlation distance

    Returns
    -------
    float
        Mean silhouette coefficient in range [-1, 1],
        where 1 indicates well-clustered data

    Raises
    ------
    ValueError
        If X and labels have incompatible shapes
        If number of unique labels is less than 2 or more than n_samples - 1

    Examples
    --------
    >>> X = np.random.randn(100, 10)
    >>> labels = np.random.randint(0, 3, 100)
    >>> score = compute_clustering_silhouette(X, labels)
    >>> print(f"Silhouette Score: {score:.3f}")
    """
    X_arr = np.asarray(X) if not isinstance(X, spmatrix) else X
    labels_arr = np.asarray(labels).ravel()

    if X_arr.shape[0] != labels_arr.shape[0]:
        msg = (
            f"Shape mismatch: X has {X_arr.shape[0]} samples, "
            f"labels has {labels_arr.shape[0]} samples"
        )
        raise ValueError(msg)

    unique_labels = np.unique(labels_arr)
    n_unique = len(unique_labels)

    if n_unique < 2:
        msg = f"Silhouette score requires at least 2 clusters, got {n_unique} cluster(s)"
        raise ValueError(msg)

    if n_unique >= X_arr.shape[0]:
        msg = (
            f"Number of clusters ({n_unique}) must be less than "
            f"number of samples ({X_arr.shape[0]})"
        )
        raise ValueError(msg)

    try:
        # Handle sparse matrices efficiently
        if isinstance(X_arr, spmatrix):
            # Convert to dense for silhouette if not too large
            if X_arr.shape[0] * X_arr.shape[1] <= 10000:
                X_arr = X_arr.toarray()
            else:
                # For large sparse matrices, use precomputed distance matrix
                # to avoid memory issues
                X_arr = X_arr.toarray()

        return float(silhouette_score(X_arr, labels_arr, metric=metric))
    except Exception as e:
        msg = f"Failed to compute silhouette score: {e!s}"
        raise ValueError(msg) from e


def compare_pca_variance_explained(
    variance_scptensor: ArrayFloat | Sequence[float],
    variance_scanpy: ArrayFloat | Sequence[float],
    n_components: int = 50,
) -> dict[str, float]:
    """Compare PCA variance explained between ScpTensor and Scanpy.

    Computes similarity metrics between variance explained curves from two
    different implementations. Useful for validating that PCA implementations
    produce comparable results.

    Parameters
    ----------
    variance_scptensor : ArrayFloat | Sequence[float]
        Variance explained ratio from ScpTensor PCA
    variance_scanpy : ArrayFloat | Sequence[float]
        Variance explained ratio from Scanpy PCA
    n_components : int, default=50
        Number of principal components to compare

    Returns
    -------
    dict[str, float]
        Dictionary containing comparison metrics:
        - 'pearson_correlation': Pearson correlation coefficient
        - 'spearman_correlation': Spearman correlation coefficient
        - 'mse': Mean squared error
        - 'mae': Mean absolute error
        - 'max_error': Maximum absolute error
        - 'relative_error': Relative error (L2 norm)

    Raises
    ------
    ValueError
        If variance arrays have different lengths
        If n_components is larger than array lengths

    Examples
    --------
    >>> var_scpt = np.array([0.3, 0.2, 0.15, 0.1, 0.05])
    >>> var_scanpy = np.array([0.31, 0.19, 0.16, 0.09, 0.06])
    >>> metrics = compare_pca_variance_explained(var_scpt, var_scanpy, n_components=5)
    >>> print(f"Pearson correlation: {metrics['pearson_correlation']:.3f}")
    """
    var_scpt_arr = np.asarray(variance_scptensor).ravel()
    var_scan_arr = np.asarray(variance_scanpy).ravel()

    if var_scpt_arr.shape != var_scan_arr.shape:
        msg = (
            f"Shape mismatch: variance_scptensor={var_scpt_arr.shape}, "
            f"variance_scanpy={var_scan_arr.shape}"
        )
        raise ValueError(msg)

    n_comp = min(n_components, len(var_scpt_arr))
    var_scpt_trunc = var_scpt_arr[:n_comp]
    var_scan_trunc = var_scan_arr[:n_comp]

    result: dict[str, float] = {}

    # Pearson correlation
    try:
        result["pearson_correlation"] = float(np.corrcoef(var_scpt_trunc, var_scan_trunc)[0, 1])
    except Exception:
        result["pearson_correlation"] = np.nan

    # Spearman correlation
    try:
        result["spearman_correlation"] = float(spearmanr(var_scpt_trunc, var_scan_trunc)[0])
    except Exception:
        result["spearman_correlation"] = np.nan

    # Mean squared error
    try:
        result["mse"] = float(np.mean((var_scpt_trunc - var_scan_trunc) ** 2))
    except Exception:
        result["mse"] = np.nan

    # Mean absolute error
    try:
        result["mae"] = float(np.mean(np.abs(var_scpt_trunc - var_scan_trunc)))
    except Exception:
        result["mae"] = np.nan

    # Maximum absolute error
    try:
        result["max_error"] = float(np.max(np.abs(var_scpt_trunc - var_scan_trunc)))
    except Exception:
        result["max_error"] = np.nan

    # Relative error (L2 norm)
    try:
        norm_scpt = np.linalg.norm(var_scpt_trunc)
        if norm_scpt > 0:
            result["relative_error"] = float(
                np.linalg.norm(var_scpt_trunc - var_scan_trunc) / norm_scpt
            )
        else:
            result["relative_error"] = np.nan
    except Exception:
        result["relative_error"] = np.nan

    return result


def compare_umap_embedding_quality(
    embedding_scptensor: ArrayFloat,
    embedding_scanpy: ArrayFloat,
    labels: ArrayInt | Sequence[int] | None = None,
) -> dict[str, float]:
    """Compare UMAP embedding quality between ScpTensor and Scanpy.

    Computes similarity metrics between UMAP embeddings from two different
    implementations. Can optionally evaluate label consistency in the embedding space.

    Parameters
    ----------
    embedding_scptensor : ArrayFloat
        UMAP embedding from ScpTensor, shape (n_samples, n_components)
    embedding_scanpy : ArrayFloat
        UMAP embedding from Scanpy, shape (n_samples, n_components)
    labels : ArrayInt | Sequence[int] | None, default=None
        Optional cluster labels for evaluating label consistency

    Returns
    -------
    dict[str, float]
        Dictionary containing comparison metrics:
        - 'procrustes_distance': Procrustes distance between embeddings
        - 'pearson_correlation': Mean correlation of coordinates
        - 'local_structure_preservation': Fraction of nearest neighbors preserved
        - 'label_consistency_ari': ARI of label-based neighborhoods (if labels provided)
        - 'label_consistency_nmi': NMI of label-based neighborhoods (if labels provided)

    Raises
    ------
    ValueError
        If embeddings have different shapes
        If labels length doesn't match embedding samples

    Examples
    --------
    >>> embed_scpt = np.random.randn(100, 2)
    >>> embed_scanpy = embed_scpt + np.random.randn(100, 2) * 0.1
    >>> metrics = compare_umap_embedding_quality(embed_scpt, embed_scanpy)
    >>> print(f"Procrustes distance: {metrics['procrustes_distance']:.3f}")
    """
    embed_scpt = np.asarray(embedding_scptensor)
    embed_scan = np.asarray(embedding_scanpy)

    if embed_scpt.shape != embed_scan.shape:
        msg = (
            f"Shape mismatch: embedding_scptensor={embed_scpt.shape}, "
            f"embedding_scanpy={embed_scan.shape}"
        )
        raise ValueError(msg)

    n_samples = embed_scpt.shape[0]

    if labels is not None:
        labels_arr = np.asarray(labels).ravel()
        if labels_arr.shape[0] != n_samples:
            msg = (
                f"Labels length ({labels_arr.shape[0]}) doesn't match "
                f"number of samples ({n_samples})"
            )
            raise ValueError(msg)

    result: dict[str, float] = {}

    # Procrustes distance (after optimal alignment)
    try:
        # Center embeddings
        embed_scpt_centered = embed_scpt - embed_scpt.mean(axis=0)
        embed_scan_centered = embed_scan - embed_scan.mean(axis=0)

        # Compute Frobenius norm difference
        diff_norm = np.linalg.norm(embed_scpt_centered - embed_scan_centered, "fro")
        norm_scpt = np.linalg.norm(embed_scpt_centered, "fro")
        norm_scan = np.linalg.norm(embed_scan_centered, "fro")

        if norm_scpt > 0 and norm_scan > 0:
            result["procrustes_distance"] = float(diff_norm / (norm_scpt * norm_scan) ** 0.5)
        else:
            result["procrustes_distance"] = np.nan
    except Exception:
        result["procrustes_distance"] = np.nan

    # Mean Pearson correlation of coordinates
    try:
        correlations = []
        for dim in range(embed_scpt.shape[1]):
            corr = np.corrcoef(embed_scpt[:, dim], embed_scan[:, dim])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
        result["pearson_correlation"] = float(np.mean(correlations)) if correlations else np.nan
    except Exception:
        result["pearson_correlation"] = np.nan

    # Local structure preservation (nearest neighbors overlap)
    try:
        n_neighbors = min(15, n_samples - 1)
        # Compute pairwise distances
        dist_scpt = cdist(embed_scpt, embed_scpt)
        dist_scan = cdist(embed_scan, embed_scan)

        # Find nearest neighbors (excluding self)
        nn_scpt = np.argsort(dist_scpt, axis=1)[:, 1 : n_neighbors + 1]
        nn_scan = np.argsort(dist_scan, axis=1)[:, 1 : n_neighbors + 1]

        # Compute overlap
        overlaps = []
        for i in range(n_samples):
            overlap = len(set(nn_scpt[i]) & set(nn_scan[i]))
            overlaps.append(overlap / n_neighbors)

        result["local_structure_preservation"] = float(np.mean(overlaps))
    except Exception:
        result["local_structure_preservation"] = np.nan

    # Label consistency (if labels provided)
    if labels is not None:
        labels_arr = np.asarray(labels).ravel()

        # Compute neighborhood labels consistency
        try:
            n_neighbors = min(15, n_samples - 1)
            dist_scpt = cdist(embed_scpt, embed_scpt)
            dist_scan = cdist(embed_scan, embed_scan)

            # Get nearest neighbors for each embedding
            nn_scpt = np.argsort(dist_scpt, axis=1)[:, 1 : n_neighbors + 1]
            nn_scan = np.argsort(dist_scan, axis=1)[:, 1 : n_neighbors + 1]

            # Predict labels based on neighborhood majority vote
            pred_from_scpt = np.array(
                [np.argmax(np.bincount(labels_arr[nn])) if len(nn) > 0 else -1 for nn in nn_scpt]
            )
            pred_from_scan = np.array(
                [np.argmax(np.bincount(labels_arr[nn])) if len(nn) > 0 else -1 for nn in nn_scan]
            )

            # Compute consistency metrics
            result["label_consistency_ari"] = float(
                adjusted_rand_score(pred_from_scpt, pred_from_scan)
            )
            result["label_consistency_nmi"] = float(
                normalized_mutual_info_score(pred_from_scpt, pred_from_scan)
            )
        except Exception:
            result["label_consistency_ari"] = np.nan
            result["label_consistency_nmi"] = np.nan

    return result


class ClusteringEvaluator(BaseEvaluator):
    """Evaluator for clustering quality metrics.

    Supports comprehensive evaluation of clustering results including:
    - Adjusted Rand Index (ARI) for clustering similarity
    - Normalized Mutual Information (NMI) for information-theoretic similarity
    - Silhouette Score for cluster separation quality
    - PCA variance explained comparison
    - UMAP embedding quality comparison

    Attributes
    ----------
    supported_metrics : list[str]
        List of supported metric names

    Examples
    --------
    >>> evaluator = ClusteringEvaluator()
    >>> true_labels = np.array([0, 0, 1, 1, 2, 2])
    >>> pred_labels = np.array([0, 0, 1, 2, 2, 2])
    >>> metrics = evaluator.evaluate_clustering(true_labels, pred_labels)
    >>> print(f"ARI: {metrics['ari']:.3f}")
    """

    __slots__ = ("supported_metrics",)

    def __init__(self) -> None:
        """Initialize the clustering evaluator."""
        self.supported_metrics = [
            "ari",
            "nmi",
            "silhouette",
            "pca_variance",
            "umap_quality",
        ]

    def evaluate_clustering(
        self,
        labels_true: ArrayInt | Sequence[int],
        labels_pred: ArrayInt | Sequence[int],
        X: ArrayFloat | spmatrix | None = None,
        metrics: list[str] | None = None,
        **kwargs: object,
    ) -> dict[str, float]:
        """Evaluate clustering quality using specified metrics.

        Parameters
        ----------
        labels_true : ArrayInt | Sequence[int]
            True cluster labels (ground truth)
        labels_pred : ArrayInt | Sequence[int]
            Predicted cluster labels
        X : ArrayFloat | spmatrix | None, default=None
            Feature matrix, required for silhouette score
        metrics : list[str] | None, default=None
            List of metrics to compute. If None, computes all supported metrics
            that can be computed with the provided data
        **kwargs : object
            Additional parameters passed to metric functions:
            - average_method: str for NMI average method
            - metric: str for silhouette distance metric

        Returns
        -------
        dict[str, float]
            Dictionary mapping metric names to their computed values

        Raises
        ------
        ValueError
            If requested metric is not supported
            If required data is missing for a metric

        Examples
        --------
        >>> evaluator = ClusteringEvaluator()
        >>> labels_true = np.array([0, 0, 1, 1])
        >>> labels_pred = np.array([0, 0, 1, 1])
        >>> X = np.random.randn(4, 10)
        >>> metrics = evaluator.evaluate_clustering(
        ...     labels_true, labels_pred, X,
        ...     metrics=['ari', 'nmi', 'silhouette']
        ... )
        """
        if metrics is None:
            # Auto-detect which metrics to compute
            metrics_to_compute = []
            if X is not None:
                metrics_to_compute = ["ari", "nmi", "silhouette"]
            else:
                metrics_to_compute = ["ari", "nmi"]
        else:
            # Validate requested metrics
            for metric in metrics:
                if metric not in self.supported_metrics:
                    msg = (
                        f"Unsupported metric: {metric}. Supported metrics: {self.supported_metrics}"
                    )
                    raise ValueError(msg)
            metrics_to_compute = metrics

        results: dict[str, float] = {}

        # Compute ARI
        if "ari" in metrics_to_compute:
            try:
                results["ari"] = compute_clustering_ari(labels_true, labels_pred)
            except Exception as e:
                results["ari"] = np.nan
                if kwargs.get("raise_errors", False):
                    msg = f"Failed to compute ARI: {e!s}"
                    raise ValueError(msg) from e

        # Compute NMI
        if "nmi" in metrics_to_compute:
            try:
                average_method = kwargs.get("average_method", "arithmetic")
                results["nmi"] = compute_clustering_nmi(
                    labels_true,
                    labels_pred,
                    average_method=str(average_method),
                )
            except Exception as e:
                results["nmi"] = np.nan
                if kwargs.get("raise_errors", False):
                    msg = f"Failed to compute NMI: {e!s}"
                    raise ValueError(msg) from e

        # Compute Silhouette
        if "silhouette" in metrics_to_compute:
            if X is None:
                results["silhouette"] = np.nan
                if kwargs.get("raise_errors", False):
                    msg = "Feature matrix X is required for silhouette score"
                    raise ValueError(msg)
            else:
                try:
                    metric = kwargs.get("metric", "euclidean")
                    results["silhouette"] = compute_clustering_silhouette(
                        X,
                        labels_pred,
                        metric=str(metric),
                    )
                except Exception as e:
                    results["silhouette"] = np.nan
                    if kwargs.get("raise_errors", False):
                        msg = f"Failed to compute silhouette score: {e!s}"
                        raise ValueError(msg) from e

        return results

    def evaluate_pca_variance(
        self,
        variance_scptensor: ArrayFloat | Sequence[float],
        variance_scanpy: ArrayFloat | Sequence[float],
        n_components: int = 50,
    ) -> dict[str, float]:
        """Evaluate PCA variance explained comparison.

        Parameters
        ----------
        variance_scptensor : ArrayFloat | Sequence[float]
            Variance explained ratio from ScpTensor PCA
        variance_scanpy : ArrayFloat | Sequence[float]
            Variance explained ratio from Scanpy PCA
        n_components : int, default=50
            Number of principal components to compare

        Returns
        -------
        dict[str, float]
            Dictionary containing comparison metrics

        Examples
        --------
        >>> evaluator = ClusteringEvaluator()
        >>> var_scpt = np.array([0.3, 0.2, 0.15])
        >>> var_scanpy = np.array([0.31, 0.19, 0.16])
        >>> metrics = evaluator.evaluate_pca_variance(var_scpt, var_scanpy, n_components=3)
        >>> print(f"Pearson correlation: {metrics['pearson_correlation']:.3f}")
        """
        return compare_pca_variance_explained(
            variance_scptensor,
            variance_scanpy,
            n_components=n_components,
        )

    def evaluate_umap_quality(
        self,
        embedding_scptensor: ArrayFloat,
        embedding_scanpy: ArrayFloat,
        labels: ArrayInt | Sequence[int] | None = None,
    ) -> dict[str, float]:
        """Evaluate UMAP embedding quality comparison.

        Parameters
        ----------
        embedding_scptensor : ArrayFloat
            UMAP embedding from ScpTensor
        embedding_scanpy : ArrayFloat
            UMAP embedding from Scanpy
        labels : ArrayInt | Sequence[int] | None, default=None
            Optional cluster labels for evaluating label consistency

        Returns
        -------
        dict[str, float]
            Dictionary containing embedding quality metrics

        Examples
        --------
        >>> evaluator = ClusteringEvaluator()
        >>> embed_scpt = np.random.randn(100, 2)
        >>> embed_scanpy = embed_scpt + np.random.randn(100, 2) * 0.1
        >>> metrics = evaluator.evaluate_umap_quality(embed_scpt, embed_scanpy)
        >>> print(f"Procrustes distance: {metrics['procrustes_distance']:.3f}")
        """
        return compare_umap_embedding_quality(
            embedding_scptensor,
            embedding_scanpy,
            labels=labels,
        )


__all__ = [
    "compute_clustering_ari",
    "compute_clustering_nmi",
    "compute_clustering_silhouette",
    "compare_pca_variance_explained",
    "compare_umap_embedding_quality",
    "ClusteringEvaluator",
]
