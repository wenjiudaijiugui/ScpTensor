"""Dimensionality reduction evaluator for automatic method selection.

This module provides an evaluator for testing and comparing different
dimensionality reduction methods.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from scptensor.autoselect.evaluators.base import BaseEvaluator

if TYPE_CHECKING:
    from scptensor.autoselect.core import EvaluationResult, StageReport
    from scptensor.core.structures import ScpContainer


class DimReductionEvaluator(BaseEvaluator):
    """Evaluator for dimensionality reduction methods.

    This evaluator tests various dimensionality reduction methods and evaluates
    their performance using metrics such as variance explained, reconstruction
    error, local structure preservation, and clustering potential.

    Attributes
    ----------
    stage_name : str
        Name of the analysis stage ("reduce")
    methods : dict[str, Callable]
        Dictionary of reduction methods to test
    metric_weights : dict[str, float]
        Weights for evaluation metrics

    Examples
    --------
    >>> evaluator = DimReductionEvaluator(n_components=50)
    >>> result_container, report = evaluator.run_all(
    ...     container=data,
    ...     assay_name="proteins",
    ...     source_layer="imputed"
    ... )
    """

    def __init__(
        self,
        n_components: int = 50,
        n_neighbors: int = 15,
        random_state: int = 42,
    ):
        """Initialize the dimensionality reduction evaluator.

        Parameters
        ----------
        n_components : int, optional
            Number of components to reduce to, by default 50
        n_neighbors : int, optional
            Number of neighbors for UMAP, by default 15
        random_state : int, optional
            Random seed for reproducibility, by default 42
        """
        self._n_components = n_components
        self._n_neighbors = n_neighbors
        self._random_state = random_state
        self._available_methods: dict[str, Callable] | None = None
        self._metric_source_layer: str | None = None

    def _get_available_methods(self) -> dict[str, Callable]:
        """Get available reduction methods, checking for optional dependencies.

        Returns
        -------
        dict[str, Callable]
            Dictionary of available methods
        """
        from scptensor.autoselect.evaluators.base import create_wrapper

        if self._available_methods is not None:
            return self._available_methods

        methods: dict[str, Callable] = {}

        # PCA is always available
        try:
            from scptensor.dim_reduction import reduce_pca

            methods["pca"] = create_wrapper(
                reduce_pca,
                source_layer_param="base_layer",
                layer_namer=lambda _, __: "pca",
                new_assay_name="pca",
                n_components=self._n_components,
                random_state=self._random_state,
            )
        except ImportError:
            pass

        # UMAP requires umap-learn
        try:
            from scptensor.dim_reduction import reduce_umap

            methods["umap"] = create_wrapper(
                reduce_umap,
                source_layer_param="base_layer",
                layer_namer=lambda _, __: "umap",
                new_assay_name="umap",
                n_components=min(self._n_components, 2),  # UMAP usually 2D
                n_neighbors=self._n_neighbors,
                random_state=self._random_state,
            )
        except ImportError:
            pass

        # t-SNE is available via scikit-learn
        try:
            from scptensor.dim_reduction import reduce_tsne

            methods["tsne"] = create_wrapper(
                reduce_tsne,
                source_layer_param="base_layer",
                layer_namer=lambda _, __: "tsne",
                new_assay_name="tsne",
                n_components=min(self._n_components, 2),
                random_state=self._random_state,
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
            Stage name ("reduce")
        """
        return "reduce"

    @property
    def methods(self) -> dict[str, Callable]:
        """Return dictionary of available reduction methods.

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
            "variance_explained": 0.30,
            "reconstruction_error": 0.25,
            "local_structure": 0.25,
            "clustering_potential": 0.20,
        }

    def compute_metrics(
        self,
        container: ScpContainer,
        original_container: ScpContainer,
        layer_name: str,
    ) -> dict[str, float]:
        """Compute evaluation metrics for a reduced dimensionality assay.

        Parameters
        ----------
        container : ScpContainer
            Container with the reduced data assay
        original_container : ScpContainer
            Original container before reduction (for comparison)
        layer_name : str
            Name of the assay to evaluate (e.g., "pca", "umap")

        Returns
        -------
        dict[str, float]
            Dictionary mapping metric names to their scores (0.0 to 1.0)
        """

        # Check if assay exists
        if layer_name not in container.assays:
            return dict.fromkeys(self.metric_weights, 0.0)

        reduced_assay = container.assays[layer_name]
        if "X" not in reduced_assay.layers:
            return dict.fromkeys(self.metric_weights, 0.0)

        # Get reduced data
        x_reduced = reduced_assay.layers["X"].X
        if hasattr(x_reduced, "toarray"):
            x_reduced = x_reduced.toarray()

        # Get original data
        if "proteins" not in original_container.assays:
            return dict.fromkeys(self.metric_weights, 0.0)

        original_assay = original_container.assays["proteins"]
        # Prefer the actual input layer used by this evaluation.
        source_layer = getattr(self, "_metric_source_layer", None)
        if source_layer not in original_assay.layers:
            source_layer = None
            for ln in ["imputed", "normalized", "raw"]:
                if ln in original_assay.layers:
                    source_layer = ln
                    break

        if source_layer is None:
            return dict.fromkeys(self.metric_weights, 0.5)

        x_original = original_assay.layers[source_layer].X
        if hasattr(x_original, "toarray"):
            x_original = x_original.toarray()

        # Compute metrics
        scores: dict[str, float] = {}

        # Variance explained (for PCA, get from var; for others, estimate)
        scores["variance_explained"] = self._compute_variance_explained(
            x_original, x_reduced, reduced_assay
        )

        # Reconstruction error
        scores["reconstruction_error"] = self._compute_reconstruction_error(x_original, x_reduced)

        # Local structure preservation
        scores["local_structure"] = self._compute_local_structure(x_original, x_reduced)

        # Clustering potential
        scores["clustering_potential"] = self._compute_clustering_potential(x_reduced)

        return scores

    def _compute_variance_explained(
        self,
        x_original: np.ndarray,
        x_reduced: np.ndarray,
        reduced_assay,
    ) -> float:
        """Compute variance explained score."""
        try:
            # Check if explained variance is stored in assay var
            if (
                hasattr(reduced_assay, "var")
                and "explained_variance_ratio" in reduced_assay.var.columns
            ):
                ratios = reduced_assay.var["explained_variance_ratio"].to_numpy()
                cumulative = np.cumsum(ratios)
                # Target: 80% variance explained with n_components
                target = 0.8
                achieved = min(cumulative[-1], 1.0)
                # Score based on how close we are to target
                return float(min(achieved / target, 1.0))

            # Estimate variance explained using reconstruction
            # For methods without explicit variance (like UMAP)
            # Use a proxy based on neighbor preservation
            return 0.7  # Default reasonable score
        except Exception:
            return 0.5

    def _compute_reconstruction_error(self, x_original: np.ndarray, x_reduced: np.ndarray) -> float:
        """Compute reconstruction error score (inverted, higher is better)."""
        from scipy.stats import pearsonr

        try:
            # For PCA, we can compute actual reconstruction
            # For other methods, use distance correlation as proxy

            # Center original data
            x_centered = x_original - np.nanmean(x_original, axis=0)

            # Compute total variance
            total_var = np.nansum(x_centered**2)
            if total_var < 1e-10:
                return 0.5

            # For PCA-like methods, we can approximate reconstruction quality
            # using the correlation structure
            # Higher correlation = lower reconstruction error = higher score

            # Sample pairs for efficiency
            n_samples = min(1000, x_original.shape[0])
            if x_original.shape[0] > n_samples:
                idx = np.random.choice(x_original.shape[0], n_samples, replace=False)
                x_orig_sample = x_original[idx]
                x_red_sample = x_reduced[idx]
            else:
                x_orig_sample = x_original
                x_red_sample = x_reduced

            # Compute distance matrices
            n = len(x_orig_sample)
            if n < 10:
                return 0.5

            # Sample pairs
            n_pairs = min(1000, n * (n - 1) // 2)
            pairs = []
            for _ in range(n_pairs):
                i, j = np.random.choice(n, 2, replace=False)
                pairs.append((i, j))

            dist_orig: list[float] = []
            dist_red: list[float] = []
            for i, j in pairs:
                dist_orig.append(float(np.linalg.norm(x_orig_sample[i] - x_orig_sample[j])))
                dist_red.append(float(np.linalg.norm(x_red_sample[i] - x_red_sample[j])))

            dist_orig_arr = np.array(dist_orig)
            dist_red_arr = np.array(dist_red)

            # Normalize distances
            if np.std(dist_orig_arr) < 1e-10 or np.std(dist_red_arr) < 1e-10:
                return 0.5

            corr, _ = pearsonr(dist_orig_arr, dist_red_arr)
            if np.isnan(corr):
                return 0.5

            # Map correlation to [0, 1]
            return float(np.clip((corr + 1) / 2, 0.0, 1.0))
        except Exception:
            return 0.5

    def _compute_local_structure(
        self, x_original: np.ndarray, x_reduced: np.ndarray, k: int = 15
    ) -> float:
        """Compute local structure preservation score using kNN agreement."""
        from sklearn.neighbors import NearestNeighbors

        try:
            k = min(k, x_original.shape[0] - 1, x_reduced.shape[0] - 1)
            if k < 1:
                return 0.5

            # Handle NaN in original
            valid_mask = ~np.isnan(x_original).any(axis=1)
            if not np.any(valid_mask):
                return 0.5

            x_orig_clean = x_original[valid_mask]
            x_red_clean = x_reduced[valid_mask]

            n_samples = len(x_orig_clean)
            if n_samples < k + 1:
                return 0.5

            # Find kNN in original space
            nn_orig = NearestNeighbors(n_neighbors=k + 1)
            nn_orig.fit(x_orig_clean)
            _, indices_orig = nn_orig.kneighbors(x_orig_clean)

            # Find kNN in reduced space
            nn_red = NearestNeighbors(n_neighbors=k + 1)
            nn_red.fit(x_red_clean)
            _, indices_red = nn_red.kneighbors(x_red_clean)

            # Compute Jaccard similarity of neighborhoods
            similarities = []
            for i in range(n_samples):
                neighbors_orig = set(indices_orig[i, 1:])  # Exclude self
                neighbors_red = set(indices_red[i, 1:])
                if len(neighbors_orig) == 0:
                    continue
                jaccard = len(neighbors_orig & neighbors_red) / len(neighbors_orig | neighbors_red)
                similarities.append(jaccard)

            if not similarities:
                return 0.5

            return float(np.mean(similarities))
        except Exception:
            return 0.5

    def _compute_clustering_potential(self, x_reduced: np.ndarray) -> float:
        """Compute clustering potential using silhouette analysis."""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        try:
            # Handle NaN
            valid_mask = ~np.isnan(x_reduced).any(axis=1)
            if not np.any(valid_mask):
                return 0.5

            x_clean = x_reduced[valid_mask]
            n_samples = len(x_clean)

            if n_samples < 10:
                return 0.5

            # Subsample for efficiency
            if n_samples > 1000:
                idx = np.random.choice(n_samples, 1000, replace=False)
                x_sample = x_clean[idx]
            else:
                x_sample = x_clean

            # Try different numbers of clusters
            best_score = 0.0
            for n_clusters in [3, 5, 7, 10]:
                if n_clusters >= len(x_sample):
                    continue
                try:
                    kmeans = KMeans(
                        n_clusters=n_clusters,
                        random_state=self._random_state,
                        n_init=10,
                    )
                    labels = kmeans.fit_predict(x_sample)
                    score = silhouette_score(x_sample, labels)
                    best_score = max(best_score, score)
                except Exception:
                    continue

            return float(np.clip(best_score, 0.0, 1.0))
        except Exception:
            return 0.5

    def evaluate_method(
        self,
        container: ScpContainer,
        method_name: str,
        method_func: Callable,
        assay_name: str = "proteins",
        source_layer: str = "raw",
        **kwargs,
    ) -> tuple[ScpContainer | None, EvaluationResult]:
        """Evaluate a single dimensionality reduction method.

        Overrides base method to handle assay-based results instead of layer-based.

        Parameters
        ----------
        container : ScpContainer
            Input container to process
        method_name : str
            Name of the method to evaluate
        method_func : Callable
            Method implementation function
        assay_name : str, optional
            Name of assay to process, by default "proteins"
        source_layer : str, optional
            Name of source layer, by default "raw"
        **kwargs
            Additional parameters passed to the method

        Returns
        -------
        tuple[ScpContainer | None, EvaluationResult]
            Tuple of (result_container, evaluation_result).
        """
        import time

        from scptensor.autoselect.core import EvaluationResult

        # The result assay name is the method name (e.g., "pca", "umap")
        new_assay_name = method_name

        # Track execution time
        start_time = time.perf_counter()
        result_container: ScpContainer | None = None
        error_msg: str | None = None
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
                # For dim reduction, layer_name is the assay name
                scores = self.compute_metrics(
                    container=result_container,
                    original_container=container,
                    layer_name=new_assay_name,
                )
            except Exception as e:
                error_msg = f"Metric computation failed: {type(e).__name__}: {str(e)}"
                scores = dict.fromkeys(self.metric_weights, 0.0)
        else:
            scores = dict.fromkeys(self.metric_weights, 0.0)

        # Compute overall score
        overall_score = 0.0 if error_msg is not None else self.compute_overall_score(scores)

        # Clear evaluation context to avoid leaking across methods.
        self._metric_source_layer = None

        # Create evaluation result
        eval_result = EvaluationResult(
            method_name=method_name,
            scores=scores,
            overall_score=overall_score,
            execution_time=execution_time,
            layer_name=new_assay_name,  # For dim reduction, this is the assay name
            error=error_msg,
        )

        return result_container, eval_result

    def run_all(
        self,
        container: ScpContainer,
        assay_name: str = "proteins",
        source_layer: str = "raw",
        keep_all: bool = False,
        **kwargs,
    ) -> tuple[ScpContainer, StageReport]:
        """Run all dimensionality reduction methods and select the best one.

        Overrides base method to handle assay-based results.

        Parameters
        ----------
        container : ScpContainer
            Input container to process
        assay_name : str, optional
            Name of assay to process, by default "proteins"
        source_layer : str, optional
            Name of source layer, by default "raw"
        keep_all : bool, optional
            If True, keep all result assays; if False, keep only best
        **kwargs
            Additional parameters passed to all methods

        Returns
        -------
        tuple[ScpContainer, StageReport]
            Tuple of (result_container, stage_report).
        """
        from scptensor.autoselect.core import StageReport

        n_repeats, confidence_level, strategy, method_kwargs = self._extract_eval_controls(kwargs)

        # Initialize report
        report = StageReport(
            stage_name=self.stage_name,
            stage_key=self.stage_name,
            metric_weights=self.get_metric_weights(),
            selection_strategy=strategy,
            n_repeats=n_repeats,
            confidence_level=confidence_level,
        )
        results: list = []

        # Store successful results: method_name -> (assay_name, assay_object)
        from scptensor.core.structures import Assay

        successful_assays: dict[str, tuple[str, Assay]] = {}

        # Evaluate each method
        for method_name, method_func in self.methods.items():
            result_container, eval_result = self.evaluate_method_repeated(
                container=container,
                method_name=method_name,
                method_func=method_func,
                assay_name=assay_name,
                source_layer=source_layer,
                n_repeats=n_repeats,
                confidence_level=confidence_level,
                **method_kwargs,
            )

            results.append(eval_result)

            # Store successful result assay
            if result_container is not None and eval_result.error is None:
                assay_name_result = eval_result.layer_name
                assay_obj = result_container.assays[assay_name_result]
                successful_assays[method_name] = (assay_name_result, assay_obj)

        # Update report with all results
        report.results = results
        self._apply_selection_scores(report.results, strategy)

        # Find best method
        successful_results = [r for r in results if r.error is None]

        if successful_results:
            best_result = self._select_best_result(successful_results)
            report.best_method = best_result.method_name
            report.best_result = best_result
            report.recommendation_reason = (
                f"Best '{strategy}' selection score "
                f"({best_result.selection_score if best_result.selection_score is not None else best_result.overall_score:.4f}) "
                f"from {len(successful_results)} successful methods (n_repeats={n_repeats})."
            )

            # Create result container with appropriate assays
            result_container = container.copy()

            if keep_all:
                # Add all successful result assays
                for method_name, (assay_name_result, assay_obj) in successful_assays.items():
                    if method_name != best_result.method_name:
                        result_container.assays[assay_name_result] = assay_obj

            # Add the best assay
            best_assay_name = best_result.layer_name
            best_assay = successful_assays[best_result.method_name][1]
            result_container.assays[best_assay_name] = best_assay

            return result_container, report
        else:
            # All methods failed
            report.best_method = ""
            report.best_result = None
            report.recommendation_reason = "All methods failed"

            return container.copy(), report


__all__ = ["DimReductionEvaluator"]
