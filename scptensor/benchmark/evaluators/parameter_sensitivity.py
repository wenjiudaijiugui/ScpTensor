"""Parameter sensitivity evaluator for algorithm robustness assessment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

if TYPE_CHECKING:
    pass

from .biological import BaseEvaluator

try:
    import igraph as ig
    import leidenalg
except ImportError:
    ig = None
    leidenalg = None

ArrayFloat = NDArray[np.float64]
ArrayInt = NDArray[np.intp]

_DEFAULT_PARAM_SPECS = {
    "n_clusters": {"default": 5.0, "range": (2.0, 50.0), "values": [3.0, 5.0, 8.0, 10.0, 15.0, 20.0]},
    "n_neighbors": {"default": 15.0, "range": (3.0, 100.0), "values": [5.0, 10.0, 15.0, 20.0, 30.0, 50.0]},
    "n_pcs": {"default": 30.0, "range": (2.0, 100.0), "values": [5.0, 10.0, 15.0, 20.0, 30.0, 50.0]},
    "resolution": {"default": 1.0, "range": (0.1, 5.0), "values": [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]},
}


class ParameterSensitivityEvaluator(BaseEvaluator):
    """Evaluator for parameter sensitivity analysis."""

    def __init__(self, param_name: str, param_values: list[float] | None = None, n_jobs: int = 1) -> None:
        if param_name not in _DEFAULT_PARAM_SPECS:
            raise ValueError(f"Unsupported parameter '{param_name}'. Supported: {list(_DEFAULT_PARAM_SPECS)}")
        self.param_name = param_name
        self.param_spec = _DEFAULT_PARAM_SPECS[param_name]
        self.param_values = param_values or self.param_spec["values"]
        self._scores = {}

    def evaluate(self, X: ArrayFloat, labels: ArrayInt | None = None, batches: ArrayInt | None = None, **kwargs) -> dict:
        X = np.asarray(X, dtype=np.float64)
        self._scores = {}

        test_fn = {
            "n_clusters": self._test_n_clusters,
            "n_neighbors": self._test_n_neighbors,
            "n_pcs": self._test_n_pcs,
            "resolution": self._test_resolution,
        }.get(self.param_name)

        if test_fn is None:
            raise ValueError(f"No test method for '{self.param_name}'")

        for v in self.param_values:
            self._scores[v] = test_fn(X, v, labels, **kwargs)

        scores = np.array(list(self._scores.values()))
        values = np.array(list(self._scores.keys()))
        mean, std = np.mean(scores), np.std(scores)
        sensitivity = 0.0 if mean == 0 else min(1.0, std / (abs(mean) + 1e-10))

        norm_scores = (scores - scores.min()) / (scores.max() - scores.min()) if scores.max() > scores.min() else np.zeros_like(scores)
        stability = (1.0 - sensitivity) * 0.7 + norm_scores.max() * 0.3

        return {
            "sensitivity_score": float(sensitivity),
            "stability_score": float(stability),
            "optimal_value": float(values[np.argmax(scores)]),
            "metric_variance": float(std**2),
            "metric_mean": float(mean),
        }

    def _test_n_clusters(self, X: ArrayFloat, k: float, labels: ArrayInt | None = None, **kwargs) -> float:
        k = int(k)
        if k < 2 or k >= X.shape[0]:
            return 0.0
        try:
            pred = KMeans(n_clusters=k, random_state=kwargs.get("random_state", 42), n_init="auto").fit_predict(X)
            if len(np.unique(pred)) != k or X.shape[0] < k * 2:
                return 0.0
            return float(silhouette_score(X, pred))
        except Exception:
            return 0.0

    def _test_n_neighbors(self, X: ArrayFloat, k: float, labels: ArrayInt | None = None, **kwargs) -> float:
        k = int(min(k, X.shape[0] - 1))
        if k < 1:
            return 0.0
        try:
            nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
            dists, idxs = nn.kneighbors(X)
            neighbor_vals = X[idxs[:, 1:]]
            reconstructed = neighbor_vals.mean(axis=1)
            errors = np.linalg.norm(X - reconstructed, axis=1)
            scale = np.linalg.norm(X, axis=1).mean()
            return float(-errors.mean() / (scale if scale > 0 else 1.0))
        except Exception:
            return 0.0

    def _test_n_pcs(self, X: ArrayFloat, n: float, labels: ArrayInt | None = None, **kwargs) -> float:
        n = int(min(n, X.shape[1], X.shape[0] - 1))
        if n < 1:
            return 0.0
        try:
            return float(PCA(n_components=n).fit(X).explained_variance_ratio_.sum())
        except Exception:
            return 0.0

    def _test_resolution(self, X: ArrayFloat, res: float, labels: ArrayInt | None = None, **kwargs) -> float:
        if ig is None or leidenalg is None:
            return self._test_n_clusters(X, max(2, int(res * 10)), labels, **kwargs)
        try:
            from sklearn.neighbors import kneighbors_graph
            n_neighbors = min(kwargs.get("n_neighbors", 15), X.shape[0] - 1)
            adj = kneighbors_graph(X, n_neighbors=n_neighbors, mode="connectivity", include_self=True)
            sources, targets = adj.nonzero()
            g = ig.Graph(directed=False)
            g.add_vertices(X.shape[0])
            g.add_edges(zip(sources.tolist(), targets.tolist(), strict=False))
            partition = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=res, seed=kwargs.get("random_state", 42))
            pred = np.array(partition.membership)
            if len(np.unique(pred)) < 2:
                return 0.0
            return float(silhouette_score(X, pred))
        except Exception:
            return self._test_n_clusters(X, max(2, int(res * 10)), labels, **kwargs)


def evaluate_parameter_sensitivity(X: ArrayFloat, param_name: str, param_values: list[float] | None = None, labels: ArrayInt | None = None, n_jobs: int = 1, **kwargs) -> dict:
    return ParameterSensitivityEvaluator(param_name, param_values, n_jobs).evaluate(X, labels, **kwargs)


def get_supported_parameters() -> list[str]:
    return list(_DEFAULT_PARAM_SPECS.keys())


def get_parameter_spec(param_name: str) -> dict | None:
    return _DEFAULT_PARAM_SPECS.get(param_name)


__all__ = [
    "ParameterSensitivityEvaluator",
    "SensitivityResult",
    "evaluate_parameter_sensitivity",
    "get_supported_parameters",
    "get_parameter_spec",
]


class SensitivityResult:
    """Placeholder for sensitivity result compatibility."""
    pass
