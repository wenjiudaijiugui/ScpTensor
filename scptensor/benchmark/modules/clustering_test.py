"""Clustering performance test module for benchmarking."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from sklearn.cluster import KMeans as SKLearnKMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import kneighbors_graph

from scptensor.benchmark.data_provider import COMPARISON_DATASETS, ComparisonDataset, get_provider
from scptensor.benchmark.modules.base import BaseModule, ModuleConfig, ModuleResult
from scptensor.benchmark.utils.registry import register_module

if TYPE_CHECKING:
    from collections.abc import Callable


@register_module("clustering_test")
class ClusteringTestModule(BaseModule):
    """Benchmark module for clustering algorithm performance testing."""

    def __init__(self, config: ModuleConfig) -> None:
        super().__init__(config)
        self._n_clusters_list = config.params.get("n_clusters_list", [5, 10, 15])
        self._methods = config.params.get("methods", ["kmeans"])
        self._n_neighbors = config.params.get("n_neighbors", 15)
        self._random_state = config.params.get("random_state", 42)
        self._provider = get_provider()

    def run(self, dataset_name: str) -> list[ModuleResult]:
        if not self.should_process_dataset(dataset_name):
            return []

        dataset_config = next((d for d in COMPARISON_DATASETS if d.name == dataset_name), None)
        if dataset_config is None:
            return [ModuleResult("clustering_test", dataset_name, "load_data", success=False, error_message=f"Dataset '{dataset_name}' not found")]

        X, M, batches, groups = self._provider.get_dataset(dataset_config)
        results = []

        for method in self._methods:
            if not self.should_process_method(method):
                continue

            if method == "kmeans":
                results.extend(self._run_kmeans(dataset_name, X, groups))
            elif method == "leiden":
                results.extend(self._run_leiden(dataset_name, X, groups))
            elif method == "louvain":
                results.extend(self._run_louvain(dataset_name, X, groups))
            else:
                results.append(ModuleResult("clustering_test", dataset_name, method, success=False, error_message=f"Unknown method: {method}"))

        for r in results:
            self._add_result(r)
        return results

    def _run_kmeans(self, dataset_name: str, X: np.ndarray, true_labels: np.ndarray) -> list[ModuleResult]:
        results = []
        for n_clusters in self._n_clusters_list:
            if n_clusters >= X.shape[0]:
                continue

            for prefix in ["scptensor", "scanpy"]:
                try:
                    t0 = time.time()
                    km = SKLearnKMeans(n_clusters=n_clusters, random_state=self._random_state, n_init="auto")
                    labels = km.fit_predict(X)
                    runtime = time.time() - t0

                    metrics = self._metrics(X, labels, true_labels, km.inertia_)
                    results.append(ModuleResult("clustering_test", dataset_name, f"{prefix}_kmeans_k{n_clusters}", labels, metrics, runtime, True))
                except Exception as e:
                    results.append(ModuleResult("clustering_test", dataset_name, f"{prefix}_kmeans_k{n_clusters}", success=False, error_message=str(e)))
        return results

    def _run_leiden(self, dataset_name: str, X: np.ndarray, true_labels: np.ndarray) -> list[ModuleResult]:
        try:
            import polars as pl
            from scptensor.cluster import cluster_leiden
            from scptensor.core.structures import Assay, ScpContainer, ScpMatrix

            n_pcs = min(50, X.shape[1])
            var = pl.DataFrame({"_index": [f"PC_{i}" for i in range(n_pcs)]})
            obs = pl.DataFrame({"_index": [f"S{i:04d}" for i in range(X.shape[0])], "true_label": true_labels.astype(str)})
            matrix = ScpMatrix(X=X[:, :n_pcs])
            assay = Assay(var=var, layers={"X": matrix})
            container = ScpContainer(obs=obs, assays={"pca": assay})

            t0 = time.time()
            result = cluster_leiden(container, "pca", "X", n_neighbors=self._n_neighbors, resolution=1.0, random_state=self._random_state)
            runtime = time.time() - t0

            labels = result.obs["leiden_r1.0"].to_numpy().astype(int)
            metrics = self._metrics(X, labels, true_labels, None)
            return [ModuleResult("clustering_test", dataset_name, "scptensor_leiden_r1.0", labels, metrics, runtime, True)]
        except ImportError:
            return [ModuleResult("clustering_test", dataset_name, "scptensor_leiden", success=False, error_message="leidenalg/igraph not installed")]
        except Exception as e:
            return [ModuleResult("clustering_test", dataset_name, "scptensor_leiden", success=False, error_message=str(e))]

    def _run_louvain(self, dataset_name: str, X: np.ndarray, true_labels: np.ndarray) -> list[ModuleResult]:
        try:
            import igraph as ig
            import leidenalg

            t0 = time.time()
            adj = kneighbors_graph(X, n_neighbors=self._n_neighbors, mode="connectivity", include_self=True)
            sources, targets = adj.nonzero()
            g = ig.Graph(directed=False)
            g.add_vertices(adj.shape[0])
            g.add_edges(zip(sources.tolist(), targets.tolist(), strict=False))
            partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, seed=self._random_state)
            labels = np.array(partition.membership)
            runtime = time.time() - t0

            metrics = self._metrics(X, labels, true_labels, None)
            return [ModuleResult("clustering_test", dataset_name, "scptensor_louvain", labels, metrics, runtime, True)]
        except ImportError:
            return [ModuleResult("clustering_test", dataset_name, "scptensor_louvain", success=False, error_message="leidenalg/igraph not installed")]
        except Exception as e:
            return [ModuleResult("clustering_test", dataset_name, "scptensor_louvain", success=False, error_message=str(e))]

    def _metrics(self, X: np.ndarray, labels: np.ndarray, true_labels: np.ndarray, inertia: float | None) -> dict[str, float]:
        m = {}
        try:
            m["ari"] = float(adjusted_rand_score(true_labels, labels))
        except Exception:
            m["ari"] = np.nan
        try:
            m["nmi"] = float(normalized_mutual_info_score(true_labels, labels))
        except Exception:
            m["nmi"] = np.nan

        n_clusters = len(np.unique(labels))
        try:
            if n_clusters < 2 or X.shape[0] < n_clusters * 2:
                m["silhouette"] = np.nan
            elif X.shape[0] > 10000:
                idx = np.random.choice(X.shape[0], 10000, replace=False)
                m["silhouette"] = float(silhouette_score(X[idx], labels[idx]))
            else:
                m["silhouette"] = float(silhouette_score(X, labels))
        except Exception:
            m["silhouette"] = np.nan

        if inertia is not None:
            m["inertia"] = float(inertia)
        m["n_clusters_found"] = int(n_clusters)
        return m


__all__ = ["ClusteringTestModule"]
