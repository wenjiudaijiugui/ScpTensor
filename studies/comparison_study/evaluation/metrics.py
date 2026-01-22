"""
Main evaluation module for pipeline comparison.

This module provides the main PipelineEvaluator class that orchestrates
all evaluation metrics across different dimensions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

import numpy as np


class PipelineEvaluator:
    """
    Main evaluator for pipeline performance assessment.

    This class orchestrates evaluation across multiple dimensions:
    - Batch effect removal
    - Computational performance
    - Data distribution changes
    - Data structure preservation

    Parameters
    ----------
    config : Mapping[str, Any]
        Evaluation configuration from evaluation_config.yaml
    save_intermediate : bool, default True
        Whether to save intermediate results

    Examples
    --------
    >>> config = {
    ...     "batch_effects": {"enabled": True, "kbet": {"k": 25}},
    ...     "performance": {"enabled": True},
    ...     "distribution": {"enabled": True},
    ...     "structure": {"enabled": True}
    ... }
    >>> evaluator = PipelineEvaluator(config)
    >>> metrics = evaluator.evaluate(
    ...     original_container=container,
    ...     result_container=result,
    ...     runtime=120.5,
    ...     memory_peak=4.2,
    ...     pipeline_name="scptensor",
    ...     dataset_name="test_data"
    ... )
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        save_intermediate: bool = True,
    ) -> None:
        """Initialize the evaluator with configuration."""
        self.config = config
        self.save_intermediate = save_intermediate
        self.metrics: dict[str, Any] = {}

    def evaluate(
        self,
        original_container: Any,
        result_container: Any,
        runtime: float,
        memory_peak: float,
        pipeline_name: str,
        dataset_name: str,
    ) -> dict[str, Any]:
        """
        Run complete evaluation on pipeline results.

        Parameters
        ----------
        original_container : ScpContainer
            Original input container
        result_container : ScpContainer
            Processed container after pipeline
        runtime : float
            Total runtime in seconds
        memory_peak : float
            Peak memory usage in GB
        pipeline_name : str
            Name of the pipeline being evaluated
        dataset_name : str
            Name of the dataset used

        Returns
        -------
        dict[str, Any]
            Dictionary containing all evaluation metrics organized by dimension
        """
        results: dict[str, Any] = {
            "pipeline_name": pipeline_name,
            "dataset_name": dataset_name,
            "runtime": runtime,
            "memory_peak": memory_peak,
        }

        # Evaluate each dimension if enabled
        if self.config.get("batch_effects", {}).get("enabled", True):
            results["batch_effects"] = self._evaluate_batch_effects(result_container)

        if self.config.get("performance", {}).get("enabled", True):
            results["performance"] = self._evaluate_performance(
                runtime,
                memory_peak,
                original_container,
            )

        if self.config.get("distribution", {}).get("enabled", True):
            results["distribution"] = self._evaluate_distribution(
                original_container,
                result_container,
            )

        if self.config.get("structure", {}).get("enabled", True):
            results["structure"] = self._evaluate_structure(
                original_container,
                result_container,
            )

        self.metrics = results
        return results

    def _evaluate_batch_effects(self, container: Any) -> dict[str, float | str]:
        """Evaluate batch effect removal metrics."""
        from .batch_effects import (
            compute_kbet,
            compute_lisi,
            compute_mixing_entropy,
            compute_variance_ratio,
        )

        config = self.config.get("batch_effects", {})
        results: dict[str, float | str] = {}

        try:
            if config.get("kbet", {}).get("enabled", True):
                k = config["kbet"].get("k", 25)
                results["kbet"] = compute_kbet(container, k=k)
        except Exception as e:
            results["kbet_error"] = str(e)

        try:
            if config.get("lisi", {}).get("enabled", True):
                k = config["lisi"].get("k", 25)
                results["lisi"] = compute_lisi(container, k=k)
        except Exception as e:
            results["lisi_error"] = str(e)

        try:
            if config.get("mixing_entropy", {}).get("enabled", True):
                k_neighbors = config["mixing_entropy"].get("k_neighbors", 25)
                results["mixing_entropy"] = compute_mixing_entropy(
                    container,
                    k_neighbors=k_neighbors,
                )
        except Exception as e:
            results["mixing_entropy_error"] = str(e)

        try:
            if config.get("variance_ratio", {}).get("enabled", True):
                results["variance_ratio"] = compute_variance_ratio(container)
        except Exception as e:
            results["variance_ratio_error"] = str(e)

        return results

    def _evaluate_performance(
        self,
        runtime: float,
        memory_peak: float,
        container: Any,
    ) -> dict[str, float]:
        """Evaluate computational performance metrics."""
        from .performance import compute_efficiency_score

        # Get data dimensions
        n_cells, n_features = self._get_container_dimensions(container)

        efficiency_metrics = compute_efficiency_score(
            runtime=runtime,
            memory=memory_peak,
            n_cells=n_cells,
            n_features=n_features,
        )

        return {
            "runtime_seconds": runtime,
            "memory_gb": memory_peak,
            **efficiency_metrics,
        }

    def _evaluate_distribution(
        self,
        original: Any,
        result: Any,
    ) -> dict[str, float | str]:
        """Evaluate data distribution changes."""
        from .distribution import (
            compute_sparsity,
            compute_statistics,
            distribution_test,
        )

        config = self.config.get("distribution", {})
        results: dict[str, float | str] = {}

        try:
            if config.get("sparsity", {}).get("enabled", True):
                orig_sparsity = compute_sparsity(original)
                result_sparsity = compute_sparsity(result)
                results["sparsity_original"] = orig_sparsity
                results["sparsity_result"] = result_sparsity
                results["sparsity_change"] = result_sparsity - orig_sparsity
        except Exception as e:
            results["sparsity_error"] = str(e)

        try:
            if config.get("statistics", {}).get("enabled", True):
                metrics_to_compute = config["statistics"].get(
                    "metrics",
                    ["mean", "std", "skewness", "kurtosis", "cv"],
                )

                orig_stats = compute_statistics(original)
                result_stats = compute_statistics(result)

                for metric in metrics_to_compute:
                    if metric in orig_stats and metric in result_stats:
                        results[f"{metric}_original"] = orig_stats[metric]
                        results[f"{metric}_result"] = result_stats[metric]
                        results[f"{metric}_change"] = result_stats[metric] - orig_stats[metric]
        except Exception as e:
            results["statistics_error"] = str(e)

        try:
            if config.get("distribution_test", {}).get("enabled", True):
                ks_stat, ks_pval = distribution_test(original, result)
                results["ks_statistic"] = ks_stat
                results["ks_pvalue"] = ks_pval
        except Exception as e:
            results["distribution_test_error"] = str(e)

        return results

    def _evaluate_structure(
        self,
        original: Any,
        result: Any,
    ) -> dict[str, float | str]:
        """Evaluate data structure preservation."""
        from .structure import (
            compute_distance_preservation,
            compute_global_structure,
            compute_nn_consistency,
            compute_pca_variance,
        )

        config = self.config.get("structure", {})
        results: dict[str, float | str] = {}

        try:
            if config.get("pca_variance", {}).get("enabled", True):
                n_components = config["pca_variance"].get("n_components", 10)
                variance_explained = compute_pca_variance(result, n_components=n_components)
                results["pca_variance_cumulative"] = float(np.sum(variance_explained))
                for i, var in enumerate(variance_explained):
                    results[f"pca_variance_pc{i + 1}"] = float(var)
        except Exception as e:
            results["pca_variance_error"] = str(e)

        try:
            if config.get("nn_consistency", {}).get("enabled", True):
                k = config["nn_consistency"].get("k", 10)
                results["nn_consistency"] = compute_nn_consistency(original, result, k=k)
        except Exception as e:
            results["nn_consistency_error"] = str(e)

        try:
            if config.get("distance_preservation", {}).get("enabled", True):
                method = config["distance_preservation"].get("method", "spearman")
                results["distance_correlation"] = compute_distance_preservation(
                    original,
                    result,
                    method=method,
                )
        except Exception as e:
            results["distance_preservation_error"] = str(e)

        try:
            if config.get("global_structure", {}).get("enabled", True):
                global_metrics = compute_global_structure(original, result)
                results.update(global_metrics)
        except Exception as e:
            results["global_structure_error"] = str(e)

        return results

    def _get_container_dimensions(self, container: Any) -> tuple[int, int]:
        """
        Helper to get container dimensions.

        Returns
        -------
        tuple[int, int]
            (n_cells, n_features)
        """
        try:
            if hasattr(container, "assays") and container.assays:
                # Get first assay
                assay_name = list(container.assays.keys())[0]
                assay = container.assays[assay_name]

                # Get the main layer
                layer_name = "log" if "log" in assay.layers else "X"
                if layer_name in assay.layers:
                    X = assay.layers[layer_name].X
                    if hasattr(X, "shape"):
                        return X.shape[:2]

            # Fallback to obs
            if hasattr(container, "obs"):
                n_cells = container.obs.height
                return (n_cells, 1000)  # Default feature count
        except Exception:
            pass

        return (1000, 1000)  # Default dimensions

    def get_summary(self) -> dict[str, Any]:
        """
        Get a summary of evaluation results.

        Returns
        -------
        dict[str, Any]
            Summary of key metrics across all dimensions
        """
        if not self.metrics:
            return {"status": "No evaluation results available"}

        summary: dict[str, Any] = {
            "pipeline": self.metrics.get("pipeline_name", "unknown"),
            "dataset": self.metrics.get("dataset_name", "unknown"),
            "runtime_s": self.metrics.get("runtime", 0),
            "memory_gb": self.metrics.get("memory_peak", 0),
        }

        # Add batch effect summary
        if "batch_effects" in self.metrics:
            be = self.metrics["batch_effects"]
            summary["batch_effect_kbet"] = be.get("kbet", 0)
            summary["batch_effect_lisi"] = be.get("lisi", 0)

        # Add structure summary
        if "structure" in self.metrics:
            st = self.metrics["structure"]
            summary["nn_consistency"] = st.get("nn_consistency", 0)
            summary["distance_correlation"] = st.get("distance_correlation", 0)

        return summary
