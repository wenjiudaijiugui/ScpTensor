"""Parameter sensitivity evaluator for algorithm robustness assessment.

This module provides metrics for evaluating how sensitive algorithms are
to parameter changes. It helps identify optimal parameter values and assess
the stability of analysis methods.

The evaluator supports common algorithm parameters:
- n_clusters: Number of clusters for clustering algorithms
- n_neighbors: Number of neighbors for KNN/imputation
- n_pcs: Number of principal components for dimensionality reduction
- resolution: Resolution parameter for Leiden/Louvain clustering
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

if TYPE_CHECKING:
    pass

# Import BaseEvaluator from biological module
from .biological import BaseEvaluator

# Try to import graph clustering for resolution parameter testing
try:
    import igraph as ig
    import leidenalg

    _GRAPH_CLUSTERING_AVAILABLE = True
except ImportError:
    _GRAPH_CLUSTERING_AVAILABLE = False

# =============================================================================
# Type Aliases
# =============================================================================

ArrayFloat = NDArray[np.float64]
ArrayInt = NDArray[np.intp]


# =============================================================================
# Parameter Registry
# =============================================================================

@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """Specification for a parameter sensitivity test.

    Attributes
    ----------
    name : str
        Parameter name (e.g., "n_clusters", "n_neighbors").
    default_value : float
        Default parameter value.
    value_range : tuple[float, float]
        Valid range for the parameter (min, max).
    is_integer : bool
        Whether the parameter should be integer-valued.
    recommended_values : list[float] | None
        List of recommended values to test (optional).
    """
    name: str
    default_value: float
    value_range: tuple[float, float]
    is_integer: bool = True
    recommended_values: list[float] | None = None


# Default parameter specifications
_DEFAULT_PARAM_SPECS: dict[str, ParameterSpec] = {
    "n_clusters": ParameterSpec(
        name="n_clusters",
        default_value=5.0,
        value_range=(2.0, 50.0),
        is_integer=True,
        recommended_values=[3.0, 5.0, 8.0, 10.0, 15.0, 20.0],
    ),
    "n_neighbors": ParameterSpec(
        name="n_neighbors",
        default_value=15.0,
        value_range=(3.0, 100.0),
        is_integer=True,
        recommended_values=[5.0, 10.0, 15.0, 20.0, 30.0, 50.0],
    ),
    "n_pcs": ParameterSpec(
        name="n_pcs",
        default_value=30.0,
        value_range=(2.0, 100.0),
        is_integer=True,
        recommended_values=[5.0, 10.0, 15.0, 20.0, 30.0, 50.0],
    ),
    "resolution": ParameterSpec(
        name="resolution",
        default_value=1.0,
        value_range=(0.1, 5.0),
        is_integer=False,
        recommended_values=[0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
    ),
}


# =============================================================================
# Result Data Classes
# =============================================================================


@dataclass(slots=True)
class SensitivityResult:
    """Result of parameter sensitivity evaluation.

    Attributes
    ----------
    param_name : str
        Name of the tested parameter.
    sensitivity_score : float
        Parameter sensitivity score (0-1). Higher means more sensitive.
    stability_score : float
        Algorithm stability score (0-1). Higher means more stable.
    optimal_value : float
        Recommended optimal parameter value.
    metric_variance : float
        Variance of metrics across parameter values.
    metric_mean : float
        Mean metric value across all parameter values.
    all_scores : dict[float, float]
        Mapping of parameter values to metric scores.
    """
    param_name: str
    sensitivity_score: float
    stability_score: float
    optimal_value: float
    metric_variance: float
    metric_mean: float
    all_scores: dict[float, float] = field(default_factory=dict)


# =============================================================================
# Parameter Sensitivity Evaluator
# =============================================================================


class ParameterSensitivityEvaluator(BaseEvaluator):
    """Evaluator for parameter sensitivity analysis.

    This evaluator assesses how algorithm performance varies with different
    parameter values, helping to identify optimal parameters and evaluate
    algorithm robustness.

    For each parameter value tested, a quality metric is computed:
    - For n_clusters: silhouette score
    - For n_neighbors: reconstruction error (negative for maximization)
    - For n_pcs: explained variance ratio
    - For resolution: silhouette score (for graph clustering)

    Metrics
    -------
    - sensitivity_score: Parameter sensitivity (0-1). Higher means algorithm
      output changes more with parameter changes.
    - stability_score: Algorithm stability (0-1). Higher means more consistent
      results across similar parameter values.
    - optimal_value: Recommended parameter value based on analysis.
    - metric_variance: Variance of quality metrics across parameter values.
    - metric_mean: Mean quality metric across all tested values.

    Parameters
    ----------
    param_name : str
        Name of the parameter to evaluate. Must be one of: "n_clusters",
        "n_neighbors", "n_pcs", "resolution".
    param_values : list[float] | None, default=None
        List of parameter values to test. If None, uses recommended values
        for the parameter.
    n_jobs : int, default=1
        Number of parallel jobs for computation.

    Attributes
    ----------
    param_name : str
        Name of the parameter being evaluated.
    param_values : list[float]
        Parameter values to test.
    param_spec : ParameterSpec
        Specification for the parameter.

    Raises
    ------
    ValueError
        If param_name is not a supported parameter.

    Examples
    --------
    >>> import numpy as np
    >>> from scptensor.benchmark.evaluators import ParameterSensitivityEvaluator
    >>>
    >>> X = np.random.randn(100, 20)
    >>> evaluator = ParameterSensitivityEvaluator("n_clusters", [3, 5, 8, 10])
    >>> result = evaluator.evaluate(X)
    >>> print(f"Sensitivity: {result['sensitivity_score']:.3f}")
    >>> print(f"Optimal k: {result['optimal_value']:.0f}")
    """

    __slots__ = ("param_name", "param_values", "param_spec", "n_jobs", "_all_scores")

    # Mapping of parameter names to test methods
    _TEST_METHODS = {
        "n_clusters": "_test_n_clusters",
        "n_neighbors": "_test_n_neighbors",
        "n_pcs": "_test_n_pcs",
        "resolution": "_test_resolution",
    }

    def __init__(
        self,
        param_name: str,
        param_values: list[float] | None = None,
        n_jobs: int = 1,
    ) -> None:
        """Initialize the parameter sensitivity evaluator.

        Parameters
        ----------
        param_name : str
            Name of the parameter to evaluate. Must be one of: "n_clusters",
            "n_neighbors", "n_pcs", "resolution".
        param_values : list[float] | None, default=None
            List of parameter values to test. If None, uses recommended values.
        n_jobs : int, default=1
            Number of parallel jobs for computation.

        Raises
        ------
        ValueError
            If param_name is not supported.
        """
        if param_name not in _DEFAULT_PARAM_SPECS:
            supported = ", ".join(_DEFAULT_PARAM_SPECS.keys())
            raise ValueError(
                f"Unsupported parameter '{param_name}'. "
                f"Supported parameters: {supported}"
            )

        self.param_name = param_name
        self.param_spec = _DEFAULT_PARAM_SPECS[param_name]

        # Use provided values or recommended defaults
        if param_values is None:
            self.param_values = self.param_spec.recommended_values or [self.param_spec.default_value]
        else:
            self.param_values = param_values

        self.n_jobs = max(1, n_jobs)
        self._all_scores: dict[float, float] = {}

    def evaluate(
        self,
        X: ArrayFloat,
        labels: ArrayInt | None = None,
        batches: ArrayInt | None = None,
        **kwargs,
    ) -> dict[str, float]:
        """Evaluate parameter sensitivity on the given data.

        Parameters
        ----------
        X : ArrayFloat
            Data matrix of shape (n_samples, n_features).
        labels : ArrayInt | None, default=None
            True cluster labels (optional, for validation).
        batches : ArrayInt | None, default=None
            Batch labels (not used in sensitivity evaluation).
        **kwargs
            Additional parameters including:
            - random_state: Random seed (default=42)
            - true_n_clusters: True number of clusters (for validation)

        Returns
        -------
        dict[str, float]
            Dictionary containing:
            - "sensitivity_score": Parameter sensitivity (0-1)
            - "stability_score": Algorithm stability (0-1)
            - "optimal_value": Recommended optimal parameter value
            - "metric_variance": Variance of metrics across values
            - "metric_mean": Mean metric across all values
        """
        X = np.asarray(X, dtype=np.float64)
        self._all_scores = {}

        # Get the test method for this parameter
        test_method_name = self._TEST_METHODS.get(self.param_name)
        if test_method_name is None:
            raise ValueError(f"No test method for parameter '{self.param_name}'")

        test_method = getattr(self, test_method_name)

        # Test each parameter value
        for value in self.param_values:
            score = test_method(X, value, labels=labels, **kwargs)
            self._all_scores[value] = score

        # Compute sensitivity metrics
        return self._compute_sensitivity_metrics()

    def _compute_sensitivity_metrics(self) -> dict[str, float]:
        """Compute sensitivity and stability metrics from collected scores.

        Returns
        -------
        dict[str, float]
            Dictionary of sensitivity metrics.
        """
        if not self._all_scores:
            return {
                "sensitivity_score": 0.0,
                "stability_score": 0.0,
                "optimal_value": self.param_spec.default_value,
                "metric_variance": 0.0,
                "metric_mean": 0.0,
            }

        values = np.array(list(self._all_scores.keys()))
        scores = np.array(list(self._all_scores.values()))

        # Normalize scores to [0, 1] if needed
        if np.max(scores) > np.min(scores):
            normalized_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        else:
            normalized_scores = np.zeros_like(scores)

        # Sensitivity: coefficient of variation (normalized by mean)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        variance = std_score**2

        if mean_score != 0:
            sensitivity_score = min(1.0, std_score / (abs(mean_score) + 1e-10))
        else:
            sensitivity_score = 0.0

        # Stability: inverse of sensitivity (1 - sensitivity)
        # But also consider if scores are generally high (good performance)
        stability_score = 1.0 - sensitivity_score
        if np.max(normalized_scores) > 0:
            stability_score = stability_score * 0.7 + np.max(normalized_scores) * 0.3

        # Optimal value: argmax of scores
        optimal_idx = np.argmax(scores)
        optimal_value = float(values[optimal_idx])

        return {
            "sensitivity_score": float(sensitivity_score),
            "stability_score": float(stability_score),
            "optimal_value": optimal_value,
            "metric_variance": float(variance),
            "metric_mean": float(mean_score),
        }

    def _test_n_clusters(
        self,
        X: ArrayFloat,
        n_clusters: float,
        labels: ArrayInt | None = None,
        **kwargs,
    ) -> float:
        """Test clustering quality with given number of clusters.

        Parameters
        ----------
        X : ArrayFloat
            Data matrix.
        n_clusters : float
            Number of clusters to test.
        labels : ArrayInt | None, default=None
            True labels for validation (optional).
        **kwargs
            Additional parameters.

        Returns
        -------
        float
            Clustering quality score (silhouette or ARI).
        """
        k = int(n_clusters)
        n_samples = X.shape[0]

        # Validate k
        if k < 2 or k >= n_samples:
            return 0.0

        random_state = kwargs.get("random_state", 42)

        try:
            # Perform KMeans clustering
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
            pred_labels = kmeans.fit_predict(X)

            # Check if we got the expected number of clusters
            n_found = len(np.unique(pred_labels))
            if n_found != k:
                return -0.5  # Penalty for not finding expected clusters

            # Compute silhouette score
            if n_found < 2 or n_samples < n_found * 2:
                return 0.0

            try:
                silhouette = silhouette_score(X, pred_labels)
                return float(silhouette)
            except Exception:
                return 0.0

        except Exception:
            return 0.0

    def _test_n_neighbors(
        self,
        X: ArrayFloat,
        n_neighbors: float,
        labels: ArrayInt | None = None,
        **kwargs,
    ) -> float:
        """Test KNN reconstruction quality with given number of neighbors.

        Parameters
        ----------
        X : ArrayFloat
            Data matrix.
        n_neighbors : float
            Number of neighbors to test.
        labels : ArrayInt | None, default=None
            Not used.
        **kwargs
            Additional parameters.

        Returns
        -------
        float
            Negative reconstruction error (higher is better).
        """
        k = int(min(n_neighbors, X.shape[0] - 1))

        if k < 1:
            return 0.0

        try:
            nn = NearestNeighbors(n_neighbors=k + 1)
            nn.fit(X)
            distances, indices = nn.kneighbors(X)

            # Compute reconstruction error: how well can we reconstruct
            # each point from its neighbors?
            reconstruction_errors = []
            for i in range(X.shape[0]):
                # Skip self (first neighbor is self due to include_self)
                neighbor_indices = indices[i, 1:] if k + 1 <= indices.shape[1] else indices[i, :-1]
                neighbor_values = X[neighbor_indices]

                # Reconstruct as mean of neighbors
                reconstructed = np.mean(neighbor_values, axis=0)
                error = np.linalg.norm(X[i] - reconstructed)
                reconstruction_errors.append(error)

            mean_error = np.mean(reconstruction_errors)

            # Return negative error (higher is better)
            # Normalize by data scale
            data_scale = np.linalg.norm(X, axis=1).mean()
            if data_scale > 0:
                return -mean_error / data_scale
            return -mean_error

        except Exception:
            return 0.0

    def _test_n_pcs(
        self,
        X: ArrayFloat,
        n_pcs: float,
        labels: ArrayInt | None = None,
        **kwargs,
    ) -> float:
        """Test PCA explained variance with given number of components.

        Parameters
        ----------
        X : ArrayFloat
            Data matrix.
        n_pcs : float
            Number of principal components to test.
        labels : ArrayInt | None, default=None
            Not used.
        **kwargs
            Additional parameters.

        Returns
        -------
        float
            Cumulative explained variance ratio.
        """
        n_components = int(min(n_pcs, X.shape[1], X.shape[0] - 1))

        if n_components < 1:
            return 0.0

        try:
            pca = PCA(n_components=n_components)
            pca.fit(X)

            # Return cumulative explained variance
            explained_var = np.sum(pca.explained_variance_ratio_)
            return float(explained_var)

        except Exception:
            return 0.0

    def _test_resolution(
        self,
        X: ArrayFloat,
        resolution: float,
        labels: ArrayInt | None = None,
        **kwargs,
    ) -> float:
        """Test graph clustering quality with given resolution.

        Parameters
        ----------
        X : ArrayFloat
            Data matrix.
        resolution : float
            Resolution parameter for Leiden clustering.
        labels : ArrayInt | None, default=None
            True labels for validation.
        **kwargs
            Additional parameters including:
            - n_neighbors: Number of neighbors for graph (default=15)
            - random_state: Random seed (default=42)

        Returns
        -------
        float
            Clustering quality score.
        """
        if not _GRAPH_CLUSTERING_AVAILABLE:
            # Fallback: use KMeans with n_clusters derived from resolution
            n_clusters = max(2, int(resolution * 10))
            return self._test_n_clusters(X, n_clusters, labels=labels, **kwargs)

        n_neighbors = kwargs.get("n_neighbors", 15)
        random_state = kwargs.get("random_state", 42)
        n_samples = X.shape[0]

        if n_neighbors >= n_samples:
            n_neighbors = max(5, n_samples // 10)

        try:
            from sklearn.neighbors import kneighbors_graph

            # Build kNN graph
            adj_matrix = kneighbors_graph(
                X,
                n_neighbors=n_neighbors,
                mode="connectivity",
                include_self=True,
            )

            sources, targets = adj_matrix.nonzero()
            edges = list(zip(sources.tolist(), targets.tolist(), strict=False))

            graph = ig.Graph(directed=False)
            graph.add_vertices(n_samples)
            graph.add_edges(edges)

            # Use Leiden with given resolution
            partition = leidenalg.find_partition(
                graph,
                leidenalg.RBConfigurationVertexPartition,
                resolution_parameter=resolution,
                seed=random_state,
            )

            pred_labels = np.array(partition.membership)
            n_found = len(np.unique(pred_labels))

            if n_found < 2 or n_samples < n_found * 2:
                return 0.0

            # Compute silhouette score
            silhouette = silhouette_score(X, pred_labels)
            return float(silhouette)

        except Exception:
            # Fallback to KMeans-based approximation
            n_clusters = max(2, int(resolution * 10))
            return self._test_n_clusters(X, n_clusters, labels=labels, **kwargs)

    def get_detailed_result(self) -> SensitivityResult:
        """Get detailed sensitivity analysis result.

        Returns
        -------
        SensitivityResult
            Detailed result with all scores and metadata.
        """
        metrics = self._compute_sensitivity_metrics()

        return SensitivityResult(
            param_name=self.param_name,
            sensitivity_score=metrics["sensitivity_score"],
            stability_score=metrics["stability_score"],
            optimal_value=metrics["optimal_value"],
            metric_variance=metrics["metric_variance"],
            metric_mean=metrics["metric_mean"],
            all_scores=dict(self._all_scores),
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def evaluate_parameter_sensitivity(
    X: ArrayFloat,
    param_name: str,
    param_values: list[float] | None = None,
    labels: ArrayInt | None = None,
    n_jobs: int = 1,
    **kwargs,
) -> dict[str, float]:
    """Convenience function to evaluate parameter sensitivity.

    Parameters
    ----------
    X : ArrayFloat
        Data matrix of shape (n_samples, n_features).
    param_name : str
        Name of the parameter to evaluate. Must be one of: "n_clusters",
        "n_neighbors", "n_pcs", "resolution".
    param_values : list[float] | None, default=None
        List of parameter values to test. If None, uses recommended values.
    labels : ArrayInt | None, default=None
        True cluster labels (optional, for validation).
    n_jobs : int, default=1
        Number of parallel jobs for computation.
    **kwargs
        Additional parameters passed to the evaluator.

    Returns
    -------
    dict[str, float]
        Dictionary containing:
        - "sensitivity_score": Parameter sensitivity (0-1)
        - "stability_score": Algorithm stability (0-1)
        - "optimal_value": Recommended optimal parameter value
        - "metric_variance": Variance of metrics across values
        - "metric_mean": Mean metric across all values

    Examples
    --------
    >>> import numpy as np
    >>> from scptensor.benchmark.evaluators import evaluate_parameter_sensitivity
    >>>
    >>> X = np.random.randn(100, 20)
    >>> result = evaluate_parameter_sensitivity(X, "n_clusters", [3, 5, 8, 10])
    >>> print(f"Optimal k: {result['optimal_value']:.0f}")
    """
    evaluator = ParameterSensitivityEvaluator(
        param_name=param_name,
        param_values=param_values,
        n_jobs=n_jobs,
    )
    return evaluator.evaluate(X, labels=labels, **kwargs)


def get_supported_parameters() -> list[str]:
    """Get list of supported parameter names.

    Returns
    -------
    list[str]
        List of supported parameter names for sensitivity analysis.

    Examples
    --------
    >>> from scptensor.benchmark.evaluators.parameter_sensitivity import (
    ...     get_supported_parameters
    ... )
    >>> params = get_supported_parameters()
    >>> print(params)  # ['n_clusters', 'n_neighbors', 'n_pcs', 'resolution']
    """
    return list(_DEFAULT_PARAM_SPECS.keys())


def get_parameter_spec(param_name: str) -> ParameterSpec | None:
    """Get specification for a parameter.

    Parameters
    ----------
    param_name : str
        Name of the parameter.

    Returns
    -------
    ParameterSpec | None
        Parameter specification or None if not found.

    Examples
    --------
    >>> from scptensor.benchmark.evaluators.parameter_sensitivity import (
    ...     get_parameter_spec
    ... )
    >>> spec = get_parameter_spec("n_clusters")
    >>> print(spec.default_value)  # 5.0
    """
    return _DEFAULT_PARAM_SPECS.get(param_name)


__all__ = [
    "BaseEvaluator",
    "ParameterSensitivityEvaluator",
    "SensitivityResult",
    "ParameterSpec",
    "evaluate_parameter_sensitivity",
    "get_supported_parameters",
    "get_parameter_spec",
]
