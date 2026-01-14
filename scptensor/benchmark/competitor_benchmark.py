"""Competitor benchmark module for comparing ScpTensor against similar tools.

This module provides reference implementations of common operations using:
- scanpy: Single-cell RNA-seq analysis framework
- scikit-learn: General machine learning library
- numpy: Raw numerical computing
- scipy: Scientific computing
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import psutil
from scipy import sparse, stats
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer

from scptensor.core.structures import ScpContainer


# =============================================================================
# Context Manager for Resource Tracking
# =============================================================================


class _ResourceTracker:
    """Track memory usage during operations."""

    __slots__ = ("start_memory",)

    def __init__(self) -> None:
        self.start_memory = 0.0

    def start(self) -> None:
        """Start tracking memory."""
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024

    def stop(self) -> float:
        """Stop tracking and return memory used in MB.

        Returns
        -------
        float
            Memory used in MB.
        """
        current = psutil.Process().memory_info().rss / 1024 / 1024
        return max(0.0, current - self.start_memory)


def get_valid_data(X: np.ndarray, M: np.ndarray | None = None) -> np.ndarray:
    """Get valid (non-missing) data from matrix.

    Parameters
    ----------
    X : np.ndarray
        Data matrix.
    M : np.ndarray | None
        Mask matrix (0=valid, non-zero=missing). If None, all data considered valid.

    Returns
    -------
    np.ndarray
        X with missing values replaced by NaN for computation.
    """
    if M is None:
        return X

    X_nan = X.copy().astype(float)
    X_nan[M > 0] = np.nan
    return X_nan


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(slots=True)
class CompetitorResult:
    """Result from a competitor method run.

    Attributes
    ----------
    competitor_name : str
        Name of the competitor method.
    operation : str
        Operation performed.
    runtime_seconds : float
        Runtime in seconds.
    memory_usage_mb : float
        Memory usage in MB.
    result_array : np.ndarray
        Result data.
    parameters : dict[str, Any]
        Parameters used.
    timestamp : str
        ISO timestamp of run.
    """

    competitor_name: str
    operation: str
    runtime_seconds: float
    memory_usage_mb: float
    result_array: np.ndarray
    parameters: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: _get_timestamp())


def _get_timestamp() -> str:
    """Get current timestamp in ISO format."""
    import time
    return time.strftime("%Y-%m-%d %H:%M:%S")


# =============================================================================
# Normalization Competitors
# =============================================================================


class NumpyLogNormalize:
    """Raw numpy implementation of log normalization."""

    name = "numpy_log"

    @staticmethod
    def run(
        X: np.ndarray,
        M: np.ndarray | None = None,
        base: float = 2.0,
        offset: float = 1.0,
    ) -> tuple[np.ndarray, float, float]:
        """Run log normalization using numpy.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        M : np.ndarray | None
            Mask matrix.
        base : float
            Logarithm base.
        offset : float
            Offset to add before log.

        Returns
        -------
        tuple[np.ndarray, float, float]
            (result_array, runtime_seconds, memory_mb)
        """
        tracker = _ResourceTracker()
        tracker.start()

        import time
        start_time = time.time()

        X_valid = get_valid_data(X, M)
        X_log = np.log(X_valid + offset) / np.log(base)
        X_log = np.nan_to_num(X_log, nan=0.0)

        runtime = time.time() - start_time
        memory = tracker.stop()

        return X_log, runtime, memory


class SklearnStandardScaler:
    """scikit-learn StandardScaler implementation."""

    name = "sklearn_standard"

    @staticmethod
    def run(X: np.ndarray, M: np.ndarray | None = None) -> tuple[np.ndarray, float, float]:
        """Run z-score standardization using sklearn.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        M : np.ndarray | None
            Mask matrix.

        Returns
        -------
        tuple[np.ndarray, float, float]
            (result_array, runtime_seconds, memory_mb)
        """
        from sklearn.preprocessing import StandardScaler

        tracker = _ResourceTracker()
        tracker.start()

        import time
        start_time = time.time()

        X_valid = get_valid_data(X, M)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_valid)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        runtime = time.time() - start_time
        memory = tracker.stop()

        return X_scaled, runtime, memory


class ScipyZScore:
    """scipy.stats zscore implementation."""

    name = "scipy_zscore"

    @staticmethod
    def run(
        X: np.ndarray, M: np.ndarray | None = None, axis: int = 0
    ) -> tuple[np.ndarray, float, float]:
        """Run z-score using scipy.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        M : np.ndarray | None
            Mask matrix.
        axis : int
            Axis along which to compute z-scores.

        Returns
        -------
        tuple[np.ndarray, float, float]
            (result_array, runtime_seconds, memory_mb)
        """
        tracker = _ResourceTracker()
        tracker.start()

        import time
        start_time = time.time()

        X_valid = get_valid_data(X, M)
        X_masked = np.ma.masked_invalid(X_valid)
        X_z = stats.zscore(X_masked, axis=axis, nan_policy="omit")

        if isinstance(X_z, np.ma.MaskedArray):
            X_z = X_z.filled(0.0)

        X_z = np.nan_to_num(X_z, nan=0.0)

        runtime = time.time() - start_time
        memory = tracker.stop()

        return X_z, runtime, memory


# =============================================================================
# Imputation Competitors
# =============================================================================


class SklearnKNNImputer:
    """scikit-learn KNN imputer."""

    name = "sklearn_knn"

    @staticmethod
    def run(
        X: np.ndarray, M: np.ndarray | None = None, n_neighbors: int = 5
    ) -> tuple[np.ndarray, float, float]:
        """Run KNN imputation using sklearn.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        M : np.ndarray | None
            Mask matrix.
        n_neighbors : int
            Number of neighbors for KNN.

        Returns
        -------
        tuple[np.ndarray, float, float]
            (result_array, runtime_seconds, memory_mb)
        """
        tracker = _ResourceTracker()
        tracker.start()

        import time
        start_time = time.time()

        if M is None:
            runtime = time.time() - start_time
            memory = tracker.stop()
            return X.copy(), runtime, memory

        X_masked = X.copy().astype(float)
        X_masked[M > 0] = np.nan

        imputer = KNNImputer(n_neighbors=n_neighbors)
        X_imputed = imputer.fit_transform(X_masked)

        runtime = time.time() - start_time
        memory = tracker.stop()

        return X_imputed, runtime, memory


class NumpyMeanImputer:
    """Simple numpy mean imputation."""

    name = "numpy_mean"

    @staticmethod
    def run(X: np.ndarray, M: np.ndarray | None = None) -> tuple[np.ndarray, float, float]:
        """Run mean imputation using numpy.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        M : np.ndarray | None
            Mask matrix.

        Returns
        -------
        tuple[np.ndarray, float, float]
            (result_array, runtime_seconds, memory_mb)
        """
        tracker = _ResourceTracker()
        tracker.start()

        import time
        start_time = time.time()

        if M is None:
            runtime = time.time() - start_time
            memory = tracker.stop()
            return X.copy(), runtime, memory

        X_masked = X.copy().astype(float)
        X_masked[M > 0] = np.nan

        col_means = np.nanmean(X_masked, axis=0)
        missing_indices = np.where(M > 0)
        X_masked[missing_indices] = np.take(col_means, missing_indices[1])

        runtime = time.time() - start_time
        memory = tracker.stop()

        return X_masked, runtime, memory


class ScipySVDImputer:
    """SVD-based imputation using scipy."""

    name = "scipy_svd"

    @staticmethod
    def run(
        X: np.ndarray,
        M: np.ndarray | None = None,
        n_components: int = 10,
        max_iter: int = 100,
    ) -> tuple[np.ndarray, float, float]:
        """Run SVD-based imputation.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        M : np.ndarray | None
            Mask matrix.
        n_components : int
            Number of SVD components.
        max_iter : int
            Maximum iterations.

        Returns
        -------
        tuple[np.ndarray, float, float]
            (result_array, runtime_seconds, memory_mb)
        """
        tracker = _ResourceTracker()
        tracker.start()

        import time
        start_time = time.time()

        if M is None:
            runtime = time.time() - start_time
            memory = tracker.stop()
            return X.copy(), runtime, memory

        X_masked = X.copy().astype(float)
        X_masked[M > 0] = np.nan

        # Initialize with column means
        X_imputed = X_masked.copy()
        col_means = np.nanmean(X_masked, axis=0)

        for i in range(X_masked.shape[1]):
            missing_mask = np.isnan(X_masked[:, i])
            if np.any(missing_mask):
                X_imputed[missing_mask, i] = col_means[i]

        for _ in range(max_iter):
            X_mean = X_imputed - np.nanmean(X_imputed, axis=0)

            try:
                U, s, Vt = np.linalg.svd(X_mean, full_matrices=False)
                S_diag = np.zeros_like(X_mean)
                np.fill_diagonal(S_diag, s[:n_components])
                X_reconstructed = (
                    U[:, :n_components]
                    @ S_diag[:n_components, :n_components]
                    @ Vt[:n_components, :]
                )
                X_reconstructed = X_reconstructed + np.nanmean(X_imputed, axis=0)

                missing_mask = np.isnan(X_masked)
                X_imputed[missing_mask] = X_reconstructed[missing_mask]

            except np.linalg.LinAlgError:
                break

        runtime = time.time() - start_time
        memory = tracker.stop()

        return X_imputed, runtime, memory


# =============================================================================
# Dimensionality Reduction Competitors
# =============================================================================


class SklearnPCA:
    """scikit-learn PCA implementation."""

    name = "sklearn_pca"

    @staticmethod
    def run(
        X: np.ndarray, M: np.ndarray | None = None, n_components: int = 50
    ) -> tuple[np.ndarray, float, float]:
        """Run PCA using sklearn.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        M : np.ndarray | None
            Mask matrix.
        n_components : int
            Number of principal components.

        Returns
        -------
        tuple[np.ndarray, float, float]
            (result_array, runtime_seconds, memory_mb)
        """
        tracker = _ResourceTracker()
        tracker.start()

        import time
        start_time = time.time()

        X_valid = get_valid_data(X, M)

        if np.any(np.isnan(X_valid)):
            col_means = np.nanmean(X_valid, axis=0)
            col_means = np.nan_to_num(col_means, nan=0.0)
            for i in range(X_valid.shape[1]):
                X_valid[np.isnan(X_valid[:, i]), i] = col_means[i]

        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_valid)

        runtime = time.time() - start_time
        memory = tracker.stop()

        return X_pca, runtime, memory


class SparsePCA:
    """Sparse PCA implementation for sparse matrices."""

    name = "sparse_pca"

    @staticmethod
    def run(
        X: np.ndarray, M: np.ndarray | None = None, n_components: int = 50
    ) -> tuple[np.ndarray, float, float]:
        """Run PCA on sparse matrix.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        M : np.ndarray | None
            Mask matrix.
        n_components : int
            Number of components.

        Returns
        -------
        tuple[np.ndarray, float, float]
            (result_array, runtime_seconds, memory_mb)
        """
        from sklearn.decomposition import TruncatedSVD

        tracker = _ResourceTracker()
        tracker.start()

        import time
        start_time = time.time()

        if M is not None:
            valid_mask = M == 0
            X_sparse = sparse.csr_matrix(X * valid_mask)
        else:
            X_sparse = sparse.csr_matrix(X)

        pca = TruncatedSVD(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X_sparse)

        runtime = time.time() - start_time
        memory = tracker.stop()

        return X_pca, runtime, memory


# =============================================================================
# Clustering Competitors
# =============================================================================


class SklearnKMeans:
    """scikit-learn KMeans clustering."""

    name = "sklearn_kmeans"

    @staticmethod
    def run(
        X: np.ndarray, M: np.ndarray | None = None, n_clusters: int = 5
    ) -> tuple[np.ndarray, float, float]:
        """Run K-means clustering using sklearn.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        M : np.ndarray | None
            Mask matrix.
        n_clusters : int
            Number of clusters.

        Returns
        -------
        tuple[np.ndarray, float, float]
            (labels, runtime_seconds, memory_mb)
        """
        from sklearn.cluster import KMeans

        tracker = _ResourceTracker()
        tracker.start()

        import time
        start_time = time.time()

        X_valid = get_valid_data(X, M)

        if np.any(np.isnan(X_valid)):
            col_means = np.nanmean(X_valid, axis=0)
            for i in range(X_valid.shape[1]):
                X_valid[np.isnan(X_valid[:, i]), i] = col_means[i]

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_valid)

        runtime = time.time() - start_time
        memory = tracker.stop()

        return labels, runtime, memory


# =============================================================================
# Competitor Registry
# =============================================================================


COMPETITOR_REGISTRY: dict[str, type] = {
    # Normalization
    "numpy_log": NumpyLogNormalize,
    "sklearn_standard": SklearnStandardScaler,
    "scipy_zscore": ScipyZScore,
    # Imputation
    "sklearn_knn": SklearnKNNImputer,
    "numpy_mean": NumpyMeanImputer,
    "scipy_svd": ScipySVDImputer,
    # Dimensionality reduction
    "sklearn_pca": SklearnPCA,
    "sparse_pca": SparsePCA,
    # Clustering
    "sklearn_kmeans": SklearnKMeans,
}


def get_competitor(name: str) -> type:
    """Get competitor class by name.

    Parameters
    ----------
    name : str
        Competitor name.

    Returns
    -------
    type
        Competitor class.

    Raises
    ------
    ValueError
        If competitor not found.
    """
    if name not in COMPETITOR_REGISTRY:
        available = ", ".join(COMPETITOR_REGISTRY.keys())
        raise ValueError(f"Unknown competitor: {name}. Available: {available}")
    return COMPETITOR_REGISTRY[name]


def list_competitors() -> list[str]:
    """List all available competitors.

    Returns
    -------
    list[str]
        List of competitor names.
    """
    return list(COMPETITOR_REGISTRY.keys())


def get_competitors_by_operation(operation: str) -> dict[str, type]:
    """Get competitors grouped by operation type.

    Parameters
    ----------
    operation : str
        Operation type (normalization, imputation, dim_reduction, clustering).

    Returns
    -------
    dict[str, type]
        Dictionary of competitor name to class.
    """
    operation_map = {
        "normalization": ["numpy_log", "sklearn_standard", "scipy_zscore"],
        "imputation": ["sklearn_knn", "numpy_mean", "scipy_svd"],
        "dim_reduction": ["sklearn_pca", "sparse_pca"],
        "clustering": ["sklearn_kmeans"],
    }

    competitors = operation_map.get(operation, [])
    return {name: COMPETITOR_REGISTRY[name] for name in competitors if name in COMPETITOR_REGISTRY}


# =============================================================================
# Scanpy-style Operations
# =============================================================================


class ScanpyStyleOps:
    """Scanpy-style operations implementation.

    Note: This doesn't use AnnData but implements similar algorithms
    for direct comparison with ScpTensor.
    """

    name = "scanpy_style"

    @staticmethod
    def normalize_total(
        X: np.ndarray,
        M: np.ndarray | None = None,
        target_sum: float = 1e4,
    ) -> tuple[np.ndarray, float, float]:
        """Scanpy-style total count normalization.

        Equivalent to sc.pp.normalize_total in scanpy.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        M : np.ndarray | None
            Mask matrix.
        target_sum : float
            Target total count per sample.

        Returns
        -------
        tuple[np.ndarray, float, float]
            (result_array, runtime_seconds, memory_mb)
        """
        tracker = _ResourceTracker()
        tracker.start()

        import time
        start_time = time.time()

        X_valid = get_valid_data(X, M)

        if M is not None:
            X_masked = X.copy()
            X_masked[M > 0] = 0
            totals = np.sum(X_masked, axis=1, keepdims=True)
        else:
            totals = np.sum(X_valid, axis=1, keepdims=True)

        totals = np.where(totals > 0, totals, 1.0)
        X_normalized = X_valid / totals * target_sum
        X_normalized = np.nan_to_num(X_normalized, nan=0.0)

        runtime = time.time() - start_time
        memory = tracker.stop()

        return X_normalized, runtime, memory

    @staticmethod
    def log1p(
        X: np.ndarray, M: np.ndarray | None = None
    ) -> tuple[np.ndarray, float, float]:
        """Scanpy-style log1p transformation.

        Equivalent to sc.pp.log1p in scanpy.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        M : np.ndarray | None
            Mask matrix.

        Returns
        -------
        tuple[np.ndarray, float, float]
            (result_array, runtime_seconds, memory_mb)
        """
        tracker = _ResourceTracker()
        tracker.start()

        import time
        start_time = time.time()

        X_valid = get_valid_data(X, M)
        X_log = np.log1p(X_valid)
        X_log = np.nan_to_num(X_log, nan=0.0)

        runtime = time.time() - start_time
        memory = tracker.stop()

        return X_log, runtime, memory

    @staticmethod
    def scale(
        X: np.ndarray,
        M: np.ndarray | None = None,
        max_value: float | None = None,
    ) -> tuple[np.ndarray, float, float]:
        """Scanpy-style scaling to unit variance and zero mean.

        Equivalent to sc.pp.scale in scanpy.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        M : np.ndarray | None
            Mask matrix.
        max_value : float | None
            Maximum value to clip to.

        Returns
        -------
        tuple[np.ndarray, float, float]
            (result_array, runtime_seconds, memory_mb)
        """
        tracker = _ResourceTracker()
        tracker.start()

        import time
        start_time = time.time()

        X_valid = get_valid_data(X, M)

        if np.any(np.isnan(X_valid)):
            col_means = np.nanmean(X_valid, axis=0)
            col_stds = np.nanstd(X_valid, axis=0)
            for i in range(X_valid.shape[1]):
                X_valid[np.isnan(X_valid[:, i]), i] = col_means[i]
        else:
            col_means = np.mean(X_valid, axis=0)
            col_stds = np.std(X_valid, axis=0)

        X_scaled = (X_valid - col_means) / (col_stds + 1e-8)

        if max_value is not None:
            X_scaled = np.clip(X_scaled, -max_value, max_value)

        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        runtime = time.time() - start_time
        memory = tracker.stop()

        return X_scaled, runtime, memory

    @staticmethod
    def pca(
        X: np.ndarray,
        M: np.ndarray | None = None,
        n_comps: int = 50,
        svd_solver: str = "arpack",
    ) -> tuple[np.ndarray, float, float]:
        """Scanpy-style PCA.

        Equivalent to sc.tl.pca in scanpy.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        M : np.ndarray | None
            Mask matrix.
        n_comps : int
            Number of components.
        svd_solver : str
            SVD solver to use.

        Returns
        -------
        tuple[np.ndarray, float, float]
            (result_array, runtime_seconds, memory_mb)
        """
        tracker = _ResourceTracker()
        tracker.start()

        import time
        start_time = time.time()

        X_valid = get_valid_data(X, M)

        if np.any(np.isnan(X_valid)):
            col_means = np.nanmean(X_valid, axis=0)
            for i in range(X_valid.shape[1]):
                X_valid[np.isnan(X_valid[:, i]), i] = col_means[i]

        pca = PCA(n_components=n_comps, svd_solver=svd_solver, random_state=0)
        X_pca = pca.fit_transform(X_valid)

        runtime = time.time() - start_time
        memory = tracker.stop()

        return X_pca, runtime, memory
