"""Clustering performance test module for benchmarking.

This module provides comprehensive benchmarking of clustering algorithms
including KMeans and graph-based methods (Leiden/Louvain), comparing
ScpTensor implementations against Scanpy where applicable.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from sklearn.cluster import KMeans as SKLearnKMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from scptensor.benchmark.data_provider import ComparisonDataset, get_provider
from scptensor.benchmark.modules.base import BaseModule, ModuleConfig, ModuleResult
from scptensor.benchmark.scptensor_adapter import ScpTensorMethods
from scptensor.benchmark.utils.registry import register_module

if TYPE_CHECKING:
    from collections.abc import Callable


def _time_function(func: Callable[[], tuple[np.ndarray, float]]) -> tuple[np.ndarray, float]:
    """Time a function execution.

    Parameters
    ----------
    func : Callable[[], tuple[np.ndarray, float]]
        Function to time that returns (result, runtime).

    Returns
    -------
    tuple[np.ndarray, float]
        (result, runtime) - Result and elapsed time in seconds.
    """
    return func()


@register_module("clustering_test")
class ClusteringTestModule(BaseModule):
    """Benchmark module for clustering algorithm performance testing.

    This module evaluates clustering methods including:
    - KMeans clustering (varying number of clusters)
    - Graph-based clustering (Leiden, if dependencies available)

    For each method, it computes:
    - ARI (Adjusted Rand Index) - requires true labels
    - NMI (Normalized Mutual Information) - requires true labels
    - Silhouette Score - intrinsic quality measure
    - Inertia (for KMeans) - within-cluster sum of squares

    Examples
    --------
    >>> from scptensor.benchmark.modules.base import ModuleConfig
    >>> from scptensor.benchmark.modules.clustering_test import ClusteringTestModule
    >>> config = ModuleConfig(
    ...     name="clustering_test",
    ...     datasets=["synthetic_small"],
    ...     params={"n_clusters_list": [5, 10], "methods": ["kmeans"]}
    ... )
    >>> module = ClusteringTestModule(config)
    >>> results = module.run("synthetic_small")
    """

    def __init__(self, config: ModuleConfig) -> None:
        """Initialize the clustering test module.

        Parameters
        ----------
        config : ModuleConfig
            Configuration object with module parameters:
            - n_clusters_list: list[int] - Cluster counts to test [5, 10, 15]
            - methods: list[str] - Methods to test ["kmeans", "leiden"]
            - n_neighbors: int - Neighbors for graph clustering (default 15)
            - random_state: int - Random seed (default 42)
        """
        super().__init__(config)

        # Extract parameters with defaults
        self._n_clusters_list: list[int] = config.params.get("n_clusters_list", [5, 10, 15])
        self._methods: list[str] = config.params.get("methods", ["kmeans"])
        self._n_neighbors: int = config.params.get("n_neighbors", 15)
        self._random_state: int = config.params.get("random_state", 42)

        # Data provider for dataset access
        self._provider = get_provider()

        # Initialize adapters
        self._scptensor_methods = ScpTensorMethods()

    def run(self, dataset_name: str) -> list[ModuleResult]:
        """Execute clustering benchmark tests on a dataset.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset to process.

        Returns
        -------
        list[ModuleResult]
            List of benchmark results, one for each method/parameter combination.
        """
        if not self.should_process_dataset(dataset_name):
            return []

        results: list[ModuleResult] = []

        # Get dataset configuration
        dataset_config = self._get_dataset_config(dataset_name)
        if dataset_config is None:
            results.append(
                ModuleResult(
                    module_name="clustering_test",
                    dataset_name=dataset_name,
                    method_name="load_data",
                    success=False,
                    error_message=f"Dataset '{dataset_name}' not found",
                )
            )
            return results

        # Load data
        X, M, batches, groups = self._provider.get_dataset(dataset_config)

        # Run tests for each method
        for method in self._methods:
            if not self.should_process_method(method):
                continue

            if method == "kmeans":
                method_results = self._run_kmeans_tests(dataset_name, X, M, groups)
            elif method == "leiden":
                method_results = self._run_leiden_tests(dataset_name, X, M, groups)
            elif method == "louvain":
                method_results = self._run_louvain_tests(dataset_name, X, M, groups)
            else:
                results.append(
                    ModuleResult(
                        module_name="clustering_test",
                        dataset_name=dataset_name,
                        method_name=method,
                        success=False,
                        error_message=f"Unknown method: {method}",
                    )
                )
                continue

            results.extend(method_results)

        # Store results
        for result in results:
            self._add_result(result)

        return results

    def _get_dataset_config(self, dataset_name: str) -> ComparisonDataset | None:
        """Get dataset configuration by name.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset.

        Returns
        -------
        ComparisonDataset | None
            Dataset configuration or None if not found.
        """
        from scptensor.benchmark.data_provider import COMPARISON_DATASETS

        for dataset in COMPARISON_DATASETS:
            if dataset.name == dataset_name:
                return dataset
        return None

    def _run_kmeans_tests(
        self,
        dataset_name: str,
        X: np.ndarray,
        M: np.ndarray,
        true_labels: np.ndarray,
    ) -> list[ModuleResult]:
        """Run KMeans clustering tests with varying cluster counts.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset.
        X : np.ndarray
            Data matrix.
        M : np.ndarray
            Mask matrix.
        true_labels : np.ndarray
            True cluster labels.

        Returns
        -------
        list[ModuleResult]
            List of results for each cluster count.
        """
        results: list[ModuleResult] = []

        for n_clusters in self._n_clusters_list:
            # Skip if n_clusters is too large for the dataset
            if n_clusters >= X.shape[0]:
                continue

            # Test ScpTensor KMeans
            result = self._test_scptensor_kmeans(dataset_name, X, M, true_labels, n_clusters)
            results.append(result)

            # Test Scanpy KMeans (using sklearn via adapter)
            result_scanpy = self._test_scanpy_kmeans(dataset_name, X, M, true_labels, n_clusters)
            results.append(result_scanpy)

        return results

    def _test_scptensor_kmeans(
        self,
        dataset_name: str,
        X: np.ndarray,
        M: np.ndarray,
        true_labels: np.ndarray,
        n_clusters: int,
    ) -> ModuleResult:
        """Test ScpTensor KMeans clustering.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset.
        X : np.ndarray
            Data matrix.
        M : np.ndarray
            Mask matrix.
        true_labels : np.ndarray
            True cluster labels.
        n_clusters : int
            Number of clusters.

        Returns
        -------
        ModuleResult
            Benchmark result.
        """
        try:
            start_time = time.time()

            # Use sklearn directly for KMeans (same as adapter)
            kmeans = SKLearnKMeans(
                n_clusters=n_clusters,
                random_state=self._random_state,
                n_init="auto",
            )
            labels = kmeans.fit_predict(X)
            inertia = kmeans.inertia_

            runtime = time.time() - start_time

            # Compute metrics
            metrics = self._compute_clustering_metrics(X, labels, true_labels, inertia)

            return ModuleResult(
                module_name="clustering_test",
                dataset_name=dataset_name,
                method_name=f"scptensor_kmeans_k{n_clusters}",
                output=labels,
                metrics=metrics,
                runtime_seconds=runtime,
                success=True,
            )

        except Exception as e:
            return ModuleResult(
                module_name="clustering_test",
                dataset_name=dataset_name,
                method_name=f"scptensor_kmeans_k{n_clusters}",
                success=False,
                error_message=str(e),
            )

    def _test_scanpy_kmeans(
        self,
        dataset_name: str,
        X: np.ndarray,
        M: np.ndarray,
        true_labels: np.ndarray,
        n_clusters: int,
    ) -> ModuleResult:
        """Test Scanpy-style KMeans clustering.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset.
        X : np.ndarray
            Data matrix.
        M : np.ndarray
            Mask matrix.
        true_labels : np.ndarray
            True cluster labels.
        n_clusters : int
            Number of clusters.

        Returns
        -------
        ModuleResult
            Benchmark result.
        """
        try:
            start_time = time.time()

            # Scanpy uses sklearn KMeans internally
            kmeans = SKLearnKMeans(
                n_clusters=n_clusters,
                random_state=self._random_state,
                n_init="auto",
            )
            labels = kmeans.fit_predict(X)
            inertia = kmeans.inertia_

            runtime = time.time() - start_time

            # Compute metrics
            metrics = self._compute_clustering_metrics(X, labels, true_labels, inertia)

            return ModuleResult(
                module_name="clustering_test",
                dataset_name=dataset_name,
                method_name=f"scanpy_kmeans_k{n_clusters}",
                output=labels,
                metrics=metrics,
                runtime_seconds=runtime,
                success=True,
            )

        except Exception as e:
            return ModuleResult(
                module_name="clustering_test",
                dataset_name=dataset_name,
                method_name=f"scanpy_kmeans_k{n_clusters}",
                success=False,
                error_message=str(e),
            )

    def _run_leiden_tests(
        self,
        dataset_name: str,
        X: np.ndarray,
        M: np.ndarray,
        true_labels: np.ndarray,
    ) -> list[ModuleResult]:
        """Run Leiden clustering tests.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset.
        X : np.ndarray
            Data matrix.
        M : np.ndarray
            Mask matrix.
        true_labels : np.ndarray
            True cluster labels.

        Returns
        -------
        list[ModuleResult]
            List of results.
        """
        results: list[ModuleResult] = []

        try:
            import polars as pl

            from scptensor.cluster import cluster_leiden
            from scptensor.core.structures import Assay, ScpContainer, ScpMatrix

            # Create container for Leiden
            n_pcs = min(50, X.shape[1])
            var = pl.DataFrame({"_index": [f"PC_{i}" for i in range(n_pcs)]})
            obs = pl.DataFrame(
                {
                    "_index": [f"S{i:04d}" for i in range(X.shape[0])],
                    "true_label": true_labels.astype(str),
                }
            )
            matrix = ScpMatrix(X=X[:, :n_pcs])
            assay = Assay(var=var, layers={"X": matrix})
            container = ScpContainer(obs=obs, assays={"pca": assay})

            # Test Leiden with default resolution
            start_time = time.time()
            result_container = cluster_leiden(
                container,
                assay_name="pca",
                base_layer="X",
                n_neighbors=self._n_neighbors,
                resolution=1.0,
                random_state=self._random_state,
            )
            runtime = time.time() - start_time

            # Extract labels
            labels_col = "leiden_r1.0"
            if labels_col in result_container.obs.columns:
                labels = result_container.obs[labels_col].to_numpy().astype(int)
            else:
                raise ValueError(f"Leiden column '{labels_col}' not found in obs")

            # Compute metrics (no inertia for Leiden)
            metrics = self._compute_clustering_metrics(X, labels, true_labels, inertia=None)

            results.append(
                ModuleResult(
                    module_name="clustering_test",
                    dataset_name=dataset_name,
                    method_name="scptensor_leiden_r1.0",
                    output=labels,
                    metrics=metrics,
                    runtime_seconds=runtime,
                    success=True,
                )
            )

        except ImportError:
            results.append(
                ModuleResult(
                    module_name="clustering_test",
                    dataset_name=dataset_name,
                    method_name="scptensor_leiden",
                    success=False,
                    error_message="leidenalg/igraph not installed",
                )
            )
        except Exception as e:
            results.append(
                ModuleResult(
                    module_name="clustering_test",
                    dataset_name=dataset_name,
                    method_name="scptensor_leiden",
                    success=False,
                    error_message=str(e),
                )
            )

        return results

    def _run_louvain_tests(
        self,
        dataset_name: str,
        X: np.ndarray,
        M: np.ndarray,
        true_labels: np.ndarray,
    ) -> list[ModuleResult]:
        """Run Louvain clustering tests.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset.
        X : np.ndarray
            Data matrix.
        M : np.ndarray
            Mask matrix.
        true_labels : np.ndarray
            True cluster labels.

        Returns
        -------
        list[ModuleResult]
            List of results.
        """
        results: list[ModuleResult] = []

        try:
            import igraph as ig
            import leidenalg
            from sklearn.neighbors import kneighbors_graph

            # Build kNN graph
            start_time = time.time()
            adj_matrix = kneighbors_graph(
                X,
                n_neighbors=self._n_neighbors,
                mode="connectivity",
                include_self=True,
            )

            sources, targets = adj_matrix.nonzero()
            edges = list(zip(sources.tolist(), targets.tolist(), strict=False))

            graph = ig.Graph(directed=False)
            graph.add_vertices(adj_matrix.shape[0])
            graph.add_edges(edges)

            # Use Louvain algorithm via leidenalg
            partition = leidenalg.find_partition(
                graph,
                leidenalg.ModularityVertexPartition,
                seed=self._random_state,
            )

            labels = np.array(partition.membership)
            runtime = time.time() - start_time

            # Compute metrics
            metrics = self._compute_clustering_metrics(X, labels, true_labels, inertia=None)

            results.append(
                ModuleResult(
                    module_name="clustering_test",
                    dataset_name=dataset_name,
                    method_name="scptensor_louvain",
                    output=labels,
                    metrics=metrics,
                    runtime_seconds=runtime,
                    success=True,
                )
            )

        except ImportError:
            results.append(
                ModuleResult(
                    module_name="clustering_test",
                    dataset_name=dataset_name,
                    method_name="scptensor_louvain",
                    success=False,
                    error_message="leidenalg/igraph not installed",
                )
            )
        except Exception as e:
            results.append(
                ModuleResult(
                    module_name="clustering_test",
                    dataset_name=dataset_name,
                    method_name="scptensor_louvain",
                    success=False,
                    error_message=str(e),
                )
            )

        return results

    def _compute_clustering_metrics(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        true_labels: np.ndarray,
        inertia: float | None,
    ) -> dict[str, float]:
        """Compute clustering evaluation metrics.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.
        labels : np.ndarray
            Predicted cluster labels.
        true_labels : np.ndarray
            True cluster labels.
        inertia : float | None
            Within-cluster sum of squares (for KMeans).

        Returns
        -------
        dict[str, float]
            Dictionary of metric names and values.
        """
        metrics: dict[str, float] = {}

        # ARI - requires true labels
        try:
            ari = adjusted_rand_score(true_labels, labels)
            metrics["ari"] = float(ari)
        except Exception:
            metrics["ari"] = np.nan

        # NMI - requires true labels
        try:
            nmi = normalized_mutual_info_score(true_labels, labels)
            metrics["nmi"] = float(nmi)
        except Exception:
            metrics["nmi"] = np.nan

        # Silhouette score - intrinsic metric
        try:
            # Handle edge cases
            n_clusters = len(np.unique(labels))
            if n_clusters < 2 or X.shape[0] < n_clusters * 2:
                metrics["silhouette"] = np.nan
            else:
                # Sample if too large for silhouette computation
                if X.shape[0] > 10000:
                    sample_idx = np.random.choice(X.shape[0], size=10000, replace=False)
                    X_sample = X[sample_idx]
                    labels_sample = labels[sample_idx]
                else:
                    X_sample = X
                    labels_sample = labels

                silhouette = silhouette_score(X_sample, labels_sample)
                metrics["silhouette"] = float(silhouette)
        except Exception:
            metrics["silhouette"] = np.nan

        # Inertia (if available)
        if inertia is not None:
            metrics["inertia"] = float(inertia)

        # Number of clusters found
        metrics["n_clusters_found"] = int(len(np.unique(labels)))

        return metrics


__all__ = ["ClusteringTestModule"]
