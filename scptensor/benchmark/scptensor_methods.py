"""ScpTensor method implementations for competitor benchmarking.

This module provides wrappers around ScpTensor methods to allow
fair comparison with competitor implementations in benchmarks.

The wrappers extract raw arrays, apply the operation, and return
results in a standardized format for comparison.
"""

import numpy as np
import psutil

from scptensor.core.structures import Assay, ScpContainer, ScpMatrix
from scptensor.impute.knn import knn as scptensor_knn
from scptensor.impute.svd import svd_impute as scptensor_svd_impute
from scptensor.normalization.log import log_normalize as scptensor_log_normalize

# =============================================================================
# Resource Tracking
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
        """Stop tracking and return memory used in MB."""
        current = psutil.Process().memory_info().rss / 1024 / 1024
        return max(0.0, current - self.start_memory)


def _get_valid_data(X: np.ndarray, M: np.ndarray | None = None) -> np.ndarray:
    """Get valid (non-missing) data from matrix.

    Parameters
    ----------
    X : np.ndarray
        Data matrix.
    M : np.ndarray | None
        Mask matrix (0=valid, non-zero=missing).

    Returns
    -------
    np.ndarray
        X with missing values replaced by NaN.
    """
    if M is None:
        return X

    X_nan = X.copy().astype(float)
    X_nan[M > 0] = np.nan
    return X_nan


def _create_container_from_data(X: np.ndarray, M: np.ndarray | None = None) -> ScpContainer:
    """Create a minimal ScpContainer from data arrays.

    Parameters
    ----------
    X : np.ndarray
        Data matrix.
    M : np.ndarray | None
        Mask matrix.

    Returns
    -------
    ScpContainer
        Container with protein assay.
    """
    import polars as pl

    n_samples, n_features = X.shape

    obs = pl.DataFrame(
        {
            "_index": [f"S{i:03d}" for i in range(n_samples)],
            "sample_id": [f"S{i:03d}" for i in range(n_samples)],
        }
    )

    var = pl.DataFrame(
        {
            "_index": [f"P{i:04d}" for i in range(n_features)],
            "protein_id": [f"P{i:04d}" for i in range(n_features)],
        }
    )

    matrix = ScpMatrix(X=X.copy(), M=M.copy() if M is not None else None)
    assay = Assay(var=var, layers={"raw": matrix}, feature_id_col="protein_id")

    return ScpContainer(
        assays={"protein": assay},
        obs=obs,
        sample_id_col="sample_id",
    )


# =============================================================================
# ScpTensor Method Wrappers
# =============================================================================


class ScpTensorLogNormalize:
    """ScpTensor log normalization wrapper."""

    name = "scptensor_log"

    @staticmethod
    def run(
        X: np.ndarray,
        M: np.ndarray | None = None,
        base: float = 2.0,
        offset: float = 1.0,
    ) -> tuple[np.ndarray, float, float]:
        """Run log normalization using ScpTensor.

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

        # Create container
        container = _create_container_from_data(X, M)

        # Apply log normalization
        result_container = scptensor_log_normalize(
            container,
            assay_name="protein",
            base_layer="raw",
            new_layer_name="log",
            base=base,
            offset=offset,
        )

        # Extract result
        result_matrix = result_container.assays["protein"].layers["log"]
        X_result = result_matrix.X

        runtime = time.time() - start_time
        memory = tracker.stop()

        return X_result, runtime, memory


class ScpTensorKNNImputer:
    """ScpTensor KNN imputation wrapper."""

    name = "scptensor_knn"

    @staticmethod
    def run(
        X: np.ndarray,
        M: np.ndarray | None = None,
        n_neighbors: int = 5,
    ) -> tuple[np.ndarray, float, float]:
        """Run KNN imputation using ScpTensor.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        M : np.ndarray | None
            Mask matrix.
        n_neighbors : int
            Number of neighbors.

        Returns
        -------
        tuple[np.ndarray, float, float]
            (result_array, runtime_seconds, memory_mb)
        """
        tracker = _ResourceTracker()
        tracker.start()

        import time

        start_time = time.time()

        # Convert mask to NaN format for ScpTensor
        X_nan = _get_valid_data(X, M)

        # Create container
        container = _create_container_from_data(X_nan, None)

        # Apply KNN imputation
        result_container = scptensor_knn(
            container,
            assay_name="protein",
            base_layer="raw",
            new_layer_name="imputed_knn",
            k=n_neighbors,
            batch_size=500,
        )

        # Extract result
        result_matrix = result_container.assays["protein"].layers["imputed_knn"]
        X_result = result_matrix.X

        runtime = time.time() - start_time
        memory = tracker.stop()

        return X_result, runtime, memory


class ScpTensorSVDImputer:
    """ScpTensor SVD imputation wrapper."""

    name = "scptensor_svd"

    @staticmethod
    def run(
        X: np.ndarray,
        M: np.ndarray | None = None,
        n_components: int = 10,
        max_iter: int = 100,
    ) -> tuple[np.ndarray, float, float]:
        """Run SVD imputation using ScpTensor.

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

        # Convert mask to NaN format for ScpTensor
        X_nan = _get_valid_data(X, M)

        # Create container
        container = _create_container_from_data(X_nan, None)

        # Apply SVD imputation
        result_container = scptensor_svd_impute(
            container,
            assay_name="protein",
            base_layer="raw",
            new_layer_name="imputed_svd",
            n_components=n_components,
            max_iter=max_iter,
        )

        # Extract result
        result_matrix = result_container.assays["protein"].layers["imputed_svd"]
        X_result = result_matrix.X

        runtime = time.time() - start_time
        memory = tracker.stop()

        return X_result, runtime, memory


class ScpTensorPCA:
    """ScpTensor PCA wrapper."""

    name = "scptensor_pca"

    @staticmethod
    def run(
        X: np.ndarray,
        M: np.ndarray | None = None,
        n_components: int = 50,
    ) -> tuple[np.ndarray, float, float]:
        """Run PCA using ScpTensor.

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
        from scptensor.dim_reduction.pca import pca as scptensor_pca

        tracker = _ResourceTracker()
        tracker.start()

        import time

        start_time = time.time()

        # Handle missing values
        X_valid = _get_valid_data(X, M)
        if np.any(np.isnan(X_valid)):
            # Simple mean imputation for PCA
            col_means = np.nanmean(X_valid, axis=0)
            col_means = np.nan_to_num(col_means, nan=0.0)
            for i in range(X_valid.shape[1]):
                X_valid[np.isnan(X_valid[:, i]), i] = col_means[i]

        # Create container
        container = _create_container_from_data(X_valid, None)

        # Apply PCA
        result_container = scptensor_pca(
            container,
            assay_name="protein",
            base_layer_name="raw",
            new_assay_name="pca",
            n_components=n_components,
            center=True,
            scale=False,
        )

        # Extract result
        result_matrix = result_container.assays["pca"].layers["scores"]
        X_result = result_matrix.X

        runtime = time.time() - start_time
        memory = tracker.stop()

        return X_result, runtime, memory


class ScpTensorKMeans:
    """ScpTensor KMeans clustering wrapper."""

    name = "scptensor_kmeans"

    @staticmethod
    def run(
        X: np.ndarray,
        M: np.ndarray | None = None,
        n_clusters: int = 5,
    ) -> tuple[np.ndarray, float, float]:
        """Run K-means clustering using ScpTensor.

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
        from scptensor.cluster.kmeans import run_kmeans as scptensor_kmeans

        tracker = _ResourceTracker()
        tracker.start()

        import time

        start_time = time.time()

        # Handle missing values
        X_valid = _get_valid_data(X, M)
        if np.any(np.isnan(X_valid)):
            col_means = np.nanmean(X_valid, axis=0)
            col_means = np.nan_to_num(col_means, nan=0.0)
            for i in range(X_valid.shape[1]):
                X_valid[np.isnan(X_valid[:, i]), i] = col_means[i]

        # Create container with PCA-like structure
        import polars as pl

        n_samples = X_valid.shape[0]
        obs = pl.DataFrame(
            {
                "_index": [f"S{i:03d}" for i in range(n_samples)],
                "sample_id": [f"S{i:03d}" for i in range(n_samples)],
            }
        )

        var = pl.DataFrame(
            {
                "_index": [f"D{i:04d}" for i in range(X_valid.shape[1])],
                "dim_id": [f"D{i:04d}" for i in range(X_valid.shape[1])],
            }
        )

        matrix = ScpMatrix(X=X_valid.copy(), M=None)
        assay = Assay(var=var, layers={"X": matrix}, feature_id_col="dim_id")

        container = ScpContainer(
            assays={"input": assay},
            obs=obs,
            sample_id_col="sample_id",
        )

        # Apply KMeans
        result_container = scptensor_kmeans(
            container,
            assay_name="input",
            base_layer="X",
            n_clusters=n_clusters,
            random_state=42,
        )

        # Extract cluster labels from one-hot encoding
        cluster_matrix = result_container.assays["cluster_kmeans"].layers["binary"]
        one_hot = cluster_matrix.X
        labels = np.argmax(one_hot, axis=1)

        runtime = time.time() - start_time
        memory = tracker.stop()

        return labels, runtime, memory


# =============================================================================
# Registry
# =============================================================================


SCPTENSOR_METHODS: dict[str, type] = {
    "scptensor_log": ScpTensorLogNormalize,
    "scptensor_knn": ScpTensorKNNImputer,
    "scptensor_svd": ScpTensorSVDImputer,
    "scptensor_pca": ScpTensorPCA,
    "scptensor_kmeans": ScpTensorKMeans,
}


def get_scptensor_method(name: str) -> type:
    """Get ScpTensor method class by name.

    Parameters
    ----------
    name : str
        Method name.

    Returns
    -------
    type
        Method class.

    Raises
    ------
    ValueError
        If method not found.
    """
    if name not in SCPTENSOR_METHODS:
        available = ", ".join(SCPTENSOR_METHODS.keys())
        raise ValueError(f"Unknown ScpTensor method: {name}. Available: {available}")
    return SCPTENSOR_METHODS[name]


def list_scptensor_methods() -> list[str]:
    """List all available ScpTensor methods.

    Returns
    -------
    list[str]
        List of method names.
    """
    return list(SCPTENSOR_METHODS.keys())


if __name__ == "__main__":
    # Test the wrappers

    print("Testing ScpTensor method wrappers...")

    # Create test data
    np.random.seed(42)
    X_test = np.random.rand(50, 100) * 10
    M_test = np.zeros_like(X_test, dtype=np.int8)
    M_test[np.random.choice(X_test.size, 200, replace=False)] = 1

    # Test log normalization
    print("\n1. Testing log normalization...")
    X_log, t_log, m_log = ScpTensorLogNormalize.run(X_test, M_test)
    print(f"   Result shape: {X_log.shape}, Time: {t_log:.4f}s, Memory: {m_log:.2f}MB")

    # Test KNN imputation
    print("\n2. Testing KNN imputation...")
    X_imp, t_imp, m_imp = ScpTensorKNNImputer.run(X_test, M_test, n_neighbors=5)
    print(f"   Result shape: {X_imp.shape}, Time: {t_imp:.4f}s, Memory: {m_imp:.2f}MB")

    # Test PCA
    print("\n3. Testing PCA...")
    X_pca, t_pca, m_pca = ScpTensorPCA.run(X_test, M_test, n_components=10)
    print(f"   Result shape: {X_pca.shape}, Time: {t_pca:.4f}s, Memory: {m_pca:.2f}MB")

    # Test KMeans
    print("\n4. Testing KMeans...")
    labels, t_km, m_km = ScpTensorKMeans.run(X_test, M_test, n_clusters=3)
    print(f"   Unique labels: {np.unique(labels)}, Time: {t_km:.4f}s, Memory: {m_km:.2f}MB")

    print("\nAll ScpTensor method wrappers tested successfully!")
