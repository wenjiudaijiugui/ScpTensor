"""Clustering evaluator for automatic method selection.

This module provides an evaluator for testing and comparing different
clustering methods.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from scptensor.autoselect.evaluators.base import BaseEvaluator

if TYPE_CHECKING:
    from scptensor.autoselect.core import EvaluationResult
    from scptensor.core.structures import ScpContainer


class ClusteringEvaluator(BaseEvaluator):
    """Evaluator for clustering methods.

    This evaluator tests various clustering methods and evaluates their
    performance using metrics such as silhouette score, Calinski-Harabasz
    index, Davies-Bouldin index, and clustering stability.

    Attributes
    ----------
    stage_name : str
        Name of the analysis stage ("cluster")
    methods : dict[str, Callable]
        Dictionary of clustering methods to test
    metric_weights : dict[str, float]
        Weights for evaluation metrics

    Examples
    --------
    >>> evaluator = ClusteringEvaluator(n_clusters=5)
    >>> result_container, report = evaluator.run_all(
    ...     container=data,
    ...     assay_name="pca",
    ...     source_layer="X"
    ... )
    """

    def __init__(
        self,
        n_clusters: int = 5,
        resolution: float = 1.0,
        n_neighbors: int = 15,
        random_state: int = 42,
    ):
        """Initialize the clustering evaluator.

        Parameters
        ----------
        n_clusters : int, optional
            Number of clusters for K-means, by default 5
        resolution : float, optional
            Resolution parameter for Leiden, by default 1.0
        n_neighbors : int, optional
            Number of neighbors for graph-based methods, by default 15
        random_state : int, optional
            Random seed for reproducibility, by default 42
        """
        self._n_clusters = n_clusters
        self._resolution = resolution
        self._n_neighbors = n_neighbors
        self._random_state = random_state
        self._available_methods: dict[str, Callable] | None = None
        self._metric_assay_name = "pca"
        self._metric_source_layer = "X"

    def _get_available_methods(self) -> dict[str, Callable]:
        """Get available clustering methods, checking for optional dependencies.

        Returns
        -------
        dict[str, Callable]
            Dictionary of available methods
        """
        from scptensor.autoselect.evaluators.base import create_wrapper

        if self._available_methods is not None:
            return self._available_methods

        methods: dict[str, Callable] = {}

        # K-means is always available
        try:
            from scptensor.cluster import cluster_kmeans

            methods["kmeans"] = create_wrapper(
                cluster_kmeans,
                source_layer_param="base_layer",
                layer_namer=lambda _, __: f"kmeans_k{self._n_clusters}",
                n_clusters=self._n_clusters,
                random_state=self._random_state,
                key_added=f"kmeans_k{self._n_clusters}",
            )
        except ImportError:
            pass

        # Leiden requires leidenalg and igraph
        try:
            from scptensor.cluster import cluster_leiden

            methods["leiden"] = create_wrapper(
                cluster_leiden,
                source_layer_param="base_layer",
                layer_namer=lambda _, __: f"leiden_r{self._resolution}",
                n_neighbors=self._n_neighbors,
                resolution=self._resolution,
                random_state=self._random_state,
                key_added=f"leiden_r{self._resolution}",
            )
        except ImportError:
            pass

        self._available_methods = methods
        return methods

    @property
    def stage_name(self) -> str:
        """Return the name of the analysis stage.

        Returns
        -------
        str
            Stage name ("cluster")
        """
        return "cluster"

    @property
    def methods(self) -> dict[str, Callable]:
        """Return dictionary of available clustering methods.

        Returns
        -------
        dict[str, Callable]
            Dictionary mapping method names to their implementation functions.
            Only methods with installed dependencies are included.
        """
        return self._get_available_methods()

    @property
    def metric_weights(self) -> dict[str, float]:
        """Return weights for each evaluation metric.

        Returns
        -------
        dict[str, float]
            Dictionary mapping metric names to their weights
        """
        return {
            "silhouette": 0.30,
            "calinski_harabasz": 0.25,
            "davies_bouldin": 0.20,
            "stability": 0.25,
        }

    def compute_metrics(
        self,
        container: ScpContainer,
        original_container: ScpContainer,
        layer_name: str,
    ) -> dict[str, float]:
        """Compute evaluation metrics for clustering results.

        Parameters
        ----------
        container : ScpContainer
            Container with clustering results in obs
        original_container : ScpContainer
            Original container (for getting the data matrix)
        layer_name : str
            Name of the clustering key in obs (e.g., "kmeans_k5")

        Returns
        -------
        dict[str, float]
            Dictionary mapping metric names to their scores (0.0 to 1.0)
        """
        import numpy as np

        # Check if clustering result exists
        if layer_name not in container.obs.columns:
            return dict.fromkeys(self.metric_weights, 0.0)

        labels = container.obs[layer_name].to_numpy()

        # Get the data matrix from the assay/layer actually used for clustering.
        assay_name = getattr(self, "_metric_assay_name", "pca")
        layer = getattr(self, "_metric_source_layer", "X")

        if assay_name not in container.assays:
            return dict.fromkeys(self.metric_weights, 0.0)

        assay = container.assays[assay_name]
        if layer not in assay.layers:
            return dict.fromkeys(self.metric_weights, 0.0)

        x_matrix = assay.layers[layer].X
        if hasattr(x_matrix, "toarray"):
            x_matrix = x_matrix.toarray()

        # Check for minimum requirements
        if len(np.unique(labels)) < 2:
            return dict.fromkeys(self.metric_weights, 0.0)

        # Compute metrics
        scores: dict[str, float] = {}

        # Silhouette score
        scores["silhouette"] = self._compute_silhouette(x_matrix, labels)

        # Calinski-Harabasz index
        scores["calinski_harabasz"] = self._compute_calinski_harabasz(x_matrix, labels)

        # Davies-Bouldin index (inverted)
        scores["davies_bouldin"] = self._compute_davies_bouldin(x_matrix, labels)

        # Stability score
        scores["stability"] = self._compute_stability(x_matrix, labels)

        return scores

    def _compute_silhouette(self, x_data: np.ndarray, labels: np.ndarray) -> float:
        """Compute silhouette score (higher is better)."""
        from sklearn.metrics import silhouette_score

        try:
            # Handle NaN
            valid_mask = ~np.isnan(x_data).any(axis=1)
            if not np.any(valid_mask):
                return 0.0

            x_clean = x_data[valid_mask]
            labels_clean = labels[valid_mask]

            # Check for minimum clusters
            if len(np.unique(labels_clean)) < 2:
                return 0.0

            # Subsample for efficiency
            if len(x_clean) > 5000:
                idx = np.random.choice(len(x_clean), 5000, replace=False)
                x_sub = x_clean[idx]
                labels_sub = labels_clean[idx]
            else:
                x_sub = x_clean
                labels_sub = labels_clean

            score = silhouette_score(x_sub, labels_sub)
            # Clip to [0, 1] - negative values indicate poor clustering
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.0

    def _compute_calinski_harabasz(self, x_data: np.ndarray, labels: np.ndarray) -> float:
        """Compute Calinski-Harabasz index (higher is better, normalized)."""
        from sklearn.metrics import calinski_harabasz_score

        try:
            # Handle NaN
            valid_mask = ~np.isnan(x_data).any(axis=1)
            if not np.any(valid_mask):
                return 0.0

            x_clean = x_data[valid_mask]
            labels_clean = labels[valid_mask]
            n_samples = len(x_clean)

            # Check for minimum clusters
            n_clusters = len(np.unique(labels_clean))
            if n_clusters < 2 or n_samples <= n_clusters:
                return 0.0

            ch_score = calinski_harabasz_score(x_clean, labels_clean)

            # Normalize using sigmoid-like transformation
            # CH scores vary widely, use soft normalization
            scaling_factor = 100.0
            normalized = ch_score / (ch_score + scaling_factor)

            return float(np.clip(normalized, 0.0, 1.0))
        except Exception:
            return 0.0

    def _compute_davies_bouldin(self, x_data: np.ndarray, labels: np.ndarray) -> float:
        """Compute Davies-Bouldin index (lower is better, returns 1-db_normalized)."""
        from sklearn.metrics import davies_bouldin_score

        try:
            # Handle NaN
            valid_mask = ~np.isnan(x_data).any(axis=1)
            if not np.any(valid_mask):
                return 0.0

            x_clean = x_data[valid_mask]
            labels_clean = labels[valid_mask]

            # Check for minimum clusters
            if len(np.unique(labels_clean)) < 2:
                return 0.0

            db_score = davies_bouldin_score(x_clean, labels_clean)

            # Normalize: 1 / (1 + db) maps [0, inf) to (0, 1]
            # db = 0 (perfect) -> score = 1.0
            normalized = 1.0 / (1.0 + db_score)

            return float(np.clip(normalized, 0.0, 1.0))
        except Exception:
            return 0.0

    def _compute_stability(
        self,
        x_data: np.ndarray,
        labels: np.ndarray,
        n_subsamples: int = 10,
        subsample_ratio: float = 0.8,
    ) -> float:
        """Compute clustering stability through subsampling."""
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score

        try:
            # Handle NaN
            valid_mask = ~np.isnan(x_data).any(axis=1)
            if not np.any(valid_mask):
                return 0.0

            x_clean = x_data[valid_mask]
            labels_clean = labels[valid_mask]
            n_samples = len(x_clean)

            # Check for minimum requirements
            n_clusters = len(np.unique(labels_clean))
            if n_clusters < 2:
                return 0.0

            min_samples_for_subsample = max(4, int(1 / (1 - subsample_ratio + 1e-10)))
            if n_samples < min_samples_for_subsample:
                return 0.5  # Not enough samples for stability test

            subsample_size = max(int(n_samples * subsample_ratio), 2)
            if subsample_size <= n_clusters:
                return 0.5

            # Subsample and re-cluster
            rng = np.random.RandomState(self._random_state)
            ari_scores = []

            for _ in range(n_subsamples):
                # Create random subsample
                subsample_indices = rng.choice(n_samples, size=subsample_size, replace=False)

                x_sub = x_clean[subsample_indices]
                labels_sub_original = labels_clean[subsample_indices]

                # Re-cluster using K-means
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    random_state=rng.randint(0, 2**31),
                    n_init=10,
                )
                labels_sub_new = kmeans.fit_predict(x_sub)

                # Calculate ARI
                ari = adjusted_rand_score(labels_sub_original, labels_sub_new)
                ari_scores.append(ari)

            # Return average ARI, clipped to [0, 1]
            avg_ari = np.mean(ari_scores)
            return float(np.clip(avg_ari, 0.0, 1.0))
        except Exception:
            return 0.5

    def _collect_success_artifact(
        self,
        result_container: ScpContainer,
        assay_name: str,
        result_name: str,
    ):
        """Return the clustering labels stored in obs for a successful method."""
        del assay_name
        return result_container.obs[result_name]

    def _attach_success_artifact(
        self,
        result_container: ScpContainer,
        assay_name: str,
        result_name: str,
        artifact,
    ) -> None:
        """Attach clustering labels to the final container obs."""
        del assay_name
        result_container.obs = result_container.obs.with_columns(
            pl.Series(name=result_name, values=artifact)
        )

    def evaluate_method(
        self,
        container: ScpContainer,
        method_name: str,
        method_func: Callable,
        assay_name: str = "pca",
        source_layer: str = "X",
        **kwargs,
    ) -> tuple[ScpContainer | None, EvaluationResult]:
        """Evaluate a single clustering method.

        Overrides base method to handle obs-based results instead of layer-based.

        Parameters
        ----------
        container : ScpContainer
            Input container to process
        method_name : str
            Name of the method to evaluate
        method_func : Callable
            Method implementation function
        assay_name : str, optional
            Name of assay to process, by default "pca"
        source_layer : str, optional
            Name of source layer, by default "X"
        **kwargs
            Additional parameters passed to the method

        Returns
        -------
        tuple[ScpContainer | None, EvaluationResult]
            Tuple of (result_container, evaluation_result).
        """
        import time

        from scptensor.autoselect.core import EvaluationResult

        # Determine the key that will be added to obs
        if method_name == "kmeans":
            key_added = f"kmeans_k{self._n_clusters}"
        elif method_name == "leiden":
            key_added = f"leiden_r{self._resolution}"
        else:
            key_added = f"{method_name}_result"

        # Track execution time
        start_time = time.perf_counter()
        result_container: ScpContainer | None = None
        error_msg: str | None = None
        self._metric_assay_name = assay_name
        self._metric_source_layer = source_layer

        try:
            # Execute method on a copy of container
            result_container = method_func(
                container=container.copy(),
                assay_name=assay_name,
                source_layer=source_layer,
                **kwargs,
            )
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            result_container = None

        execution_time = time.perf_counter() - start_time

        # Compute metrics if method succeeded
        if result_container is not None and error_msg is None:
            try:
                # For clustering, layer_name is the obs column name
                scores = self.compute_metrics(
                    container=result_container,
                    original_container=container,
                    layer_name=key_added,
                )
            except Exception as e:
                error_msg = f"Metric computation failed: {type(e).__name__}: {str(e)}"
                scores = dict.fromkeys(self.metric_weights, 0.0)
        else:
            scores = dict.fromkeys(self.metric_weights, 0.0)

        # Compute overall score
        overall_score = 0.0 if error_msg is not None else self.compute_overall_score(scores)

        # Clear evaluation context to avoid leaking across methods.
        self._metric_assay_name = "pca"
        self._metric_source_layer = "X"

        # Create evaluation result
        eval_result = EvaluationResult(
            method_name=method_name,
            scores=scores,
            overall_score=overall_score,
            execution_time=execution_time,
            layer_name=key_added,  # For clustering, this is the obs column name
            error=error_msg,
        )

        return result_container, eval_result


__all__ = ["ClusteringEvaluator"]
