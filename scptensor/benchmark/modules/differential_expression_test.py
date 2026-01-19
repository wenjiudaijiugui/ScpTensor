"""Differential expression performance test module for benchmarking.

This module provides comprehensive benchmarking of differential expression
methods including t-test (Welch's), Wilcoxon rank-sum test, and optionally
DESeq2-style methods.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from scptensor.benchmark.data_provider import ComparisonDataset, get_provider
from scptensor.benchmark.modules.base import BaseModule, ModuleConfig, ModuleResult
from scptensor.benchmark.scptensor_adapter import _create_container
from scptensor.benchmark.utils.registry import register_module
from scptensor.core.structures import ScpContainer
from scptensor.diff_expr import diff_expr_mannwhitney, diff_expr_ttest
from scptensor.diff_expr.nonparametric import diff_expr_wilcoxon

if TYPE_CHECKING:
    from collections.abc import Callable


def _time_function(func: Callable[[], tuple]) -> tuple:
    """Time a function execution.

    Parameters
    ----------
    func : Callable[[], tuple]
        Function to time that returns (result, runtime).

    Returns
    -------
    tuple
        Result and elapsed time in seconds.
    """
    return func()


@register_module("differential_expression_test")
class DifferentialExpressionTestModule(BaseModule):
    """Benchmark module for differential expression performance testing.

    This module evaluates differential expression methods including:
    - Welch's t-test (unequal variance t-test)
    - Mann-Whitney U test (Wilcoxon rank-sum)
    - Wilcoxon rank-sum test (paired or unpaired)

    For each method, it computes:
    - Runtime (seconds)
    - Number of significant features (at alpha threshold)
    - Mean and median p-values
    - False discovery rate (FDR) adjusted statistics
    - P-value distribution metrics

    Examples
    --------
    >>> from scptensor.benchmark.modules.base import ModuleConfig
    >>> from scptensor.benchmark.modules.differential_expression_test import (
    ...     DifferentialExpressionTestModule
    ... )
    >>> config = ModuleConfig(
    ...     name="differential_expression_test",
    ...     datasets=["synthetic_small"],
    ...     params={"methods": ["t_test", "wilcoxon"], "alpha": 0.05}
    ... )
    >>> module = DifferentialExpressionTestModule(config)
    >>> results = module.run("synthetic_small")
    """

    def __init__(self, config: ModuleConfig) -> None:
        """Initialize the differential expression test module.

        Parameters
        ----------
        config : ModuleConfig
            Configuration object with module parameters:
            - methods: list[str] - Methods to test ["t_test", "wilcoxon", "mannwhitney"]
            - alpha: float - Significance threshold (default 0.05)
            - group1: str | None - First group name (default None, uses first group)
            - group2: str | None - Second group name (default None, uses second group)
            - log2_fc_offset: float - Offset for log2 FC calculation (default 1.0)
            - missing_strategy: str - How to handle missing values (default "ignore")
        """
        super().__init__(config)

        # Extract parameters with defaults
        self._methods: list[str] = config.params.get("methods", ["t_test", "wilcoxon"])
        self._alpha: float = config.params.get("alpha", 0.05)
        self._group1: str | None = config.params.get("group1", None)
        self._group2: str | None = config.params.get("group2", None)
        self._log2_fc_offset: float = config.params.get("log2_fc_offset", 1.0)
        self._missing_strategy: str = config.params.get("missing_strategy", "ignore")

        # Data provider for dataset access
        self._provider = get_provider()

    def run(self, dataset_name: str) -> list[ModuleResult]:
        """Execute differential expression benchmark tests on a dataset.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset to process.

        Returns
        -------
        list[ModuleResult]
            List of benchmark results, one for each method tested.
        """
        if not self.should_process_dataset(dataset_name):
            return []

        results: list[ModuleResult] = []

        # Get dataset configuration
        dataset_config = self._get_dataset_config(dataset_name)
        if dataset_config is None:
            results.append(
                ModuleResult(
                    module_name="differential_expression_test",
                    dataset_name=dataset_name,
                    method_name="load_data",
                    success=False,
                    error_message=f"Dataset '{dataset_name}' not found",
                )
            )
            return results

        # Load data and create container
        container = self._create_container_from_config(dataset_config)

        # Determine group names
        group1, group2 = self._get_group_names(container)

        # Run tests for each method
        for method in self._methods:
            if not self.should_process_method(method):
                continue

            if method == "t_test":
                result = self._run_t_test(dataset_name, container, group1, group2)
            elif method == "wilcoxon":
                result = self._run_wilcoxon(dataset_name, container, group1, group2)
            elif method == "mannwhitney":
                result = self._run_mannwhitney(dataset_name, container, group1, group2)
            else:
                results.append(
                    ModuleResult(
                        module_name="differential_expression_test",
                        dataset_name=dataset_name,
                        method_name=method,
                        success=False,
                        error_message=f"Unknown method: {method}",
                    )
                )
                continue

            results.append(result)

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

    def _create_container_from_config(self, config: ComparisonDataset) -> ScpContainer:
        """Create a ScpContainer from dataset configuration.

        Parameters
        ----------
        config : ComparisonDataset
            Dataset configuration.

        Returns
        -------
        ScpContainer
            Container with loaded data.
        """
        X, M, batches, groups = self._provider.get_dataset(config)
        return _create_container(X, M, batch_labels=batches, group_labels=groups)

    def _get_group_names(self, container: ScpContainer) -> tuple[str, str]:
        """Get group names for comparison.

        Parameters
        ----------
        container : ScpContainer
            Container with group labels.

        Returns
        -------
        tuple[str, str]
            (group1, group2) names.
        """
        groups = container.obs["group"].to_numpy()
        unique_groups = sorted(set(groups))

        if len(unique_groups) < 2:
            raise ValueError(f"Need at least 2 groups for DE test, found {len(unique_groups)}")

        g1 = self._group1 if self._group1 is not None else str(unique_groups[0])
        g2 = self._group2 if self._group2 is not None else str(unique_groups[1])

        return g1, g2

    def _run_t_test(
        self,
        dataset_name: str,
        container: ScpContainer,
        group1: str,
        group2: str,
    ) -> ModuleResult:
        """Run Welch's t-test for differential expression.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset.
        container : ScpContainer
            Data container.
        group1, group2 : str
            Group names to compare.

        Returns
        -------
        ModuleResult
            Benchmark result.
        """
        try:
            start_time = time.time()

            result = diff_expr_ttest(
                container,
                assay_name="protein",
                group_col="group",
                group1=group1,
                group2=group2,
                layer_name="raw",
                equal_var=False,  # Welch's t-test
                missing_strategy=self._missing_strategy,
                log2_fc_offset=self._log2_fc_offset,
            )

            runtime = time.time() - start_time

            # Compute metrics
            metrics = self._compute_de_metrics(result)

            return ModuleResult(
                module_name="differential_expression_test",
                dataset_name=dataset_name,
                method_name="t_test_welch",
                output=result.p_values,
                metrics=metrics,
                runtime_seconds=runtime,
                success=True,
            )

        except Exception as e:
            return ModuleResult(
                module_name="differential_expression_test",
                dataset_name=dataset_name,
                method_name="t_test_welch",
                success=False,
                error_message=str(e),
            )

    def _run_wilcoxon(
        self,
        dataset_name: str,
        container: ScpContainer,
        group1: str,
        group2: str,
    ) -> ModuleResult:
        """Run Wilcoxon rank-sum test for differential expression.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset.
        container : ScpContainer
            Data container.
        group1, group2 : str
            Group names to compare.

        Returns
        -------
        ModuleResult
            Benchmark result.
        """
        try:
            start_time = time.time()

            result = diff_expr_wilcoxon(
                container,
                assay_name="protein",
                layer="raw",
                groupby="group",
                group1=group1,
                group2=group2,
                paired=False,
                missing_strategy=self._missing_strategy,
                log2_fc_offset=self._log2_fc_offset,
            )

            runtime = time.time() - start_time

            # Compute metrics
            metrics = self._compute_de_metrics(result)

            return ModuleResult(
                module_name="differential_expression_test",
                dataset_name=dataset_name,
                method_name="wilcoxon",
                output=result.p_values,
                metrics=metrics,
                runtime_seconds=runtime,
                success=True,
            )

        except Exception as e:
            return ModuleResult(
                module_name="differential_expression_test",
                dataset_name=dataset_name,
                method_name="wilcoxon",
                success=False,
                error_message=str(e),
            )

    def _run_mannwhitney(
        self,
        dataset_name: str,
        container: ScpContainer,
        group1: str,
        group2: str,
    ) -> ModuleResult:
        """Run Mann-Whitney U test for differential expression.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset.
        container : ScpContainer
            Data container.
        group1, group2 : str
            Group names to compare.

        Returns
        -------
        ModuleResult
            Benchmark result.
        """
        try:
            start_time = time.time()

            result = diff_expr_mannwhitney(
                container,
                assay_name="protein",
                group_col="group",
                group1=group1,
                group2=group2,
                layer_name="raw",
                alternative="two-sided",
                missing_strategy=self._missing_strategy,
                log2_fc_offset=self._log2_fc_offset,
            )

            runtime = time.time() - start_time

            # Compute metrics
            metrics = self._compute_de_metrics(result)

            return ModuleResult(
                module_name="differential_expression_test",
                dataset_name=dataset_name,
                method_name="mannwhitney",
                output=result.p_values,
                metrics=metrics,
                runtime_seconds=runtime,
                success=True,
            )

        except Exception as e:
            return ModuleResult(
                module_name="differential_expression_test",
                dataset_name=dataset_name,
                method_name="mannwhitney",
                success=False,
                error_message=str(e),
            )

    def _compute_de_metrics(self, result) -> dict[str, float]:
        """Compute differential expression metrics from result.

        Parameters
        ----------
        result : DiffExprResult
            Differential expression result object.

        Returns
        -------
        dict[str, float]
            Dictionary of metric names and values.
        """
        metrics: dict[str, float] = {}

        p_values = result.p_values
        p_values_adj = result.p_values_adj
        log2_fc = result.log2_fc

        # Filter out NaN values for statistics
        valid_mask = ~np.isnan(p_values)
        p_valid = p_values[valid_mask]

        if len(p_valid) > 0:
            # P-value distribution statistics
            metrics["mean_pval"] = float(np.mean(p_valid))
            metrics["median_pval"] = float(np.median(p_valid))
            metrics["min_pval"] = float(np.min(p_valid))
            metrics["max_pval"] = float(np.max(p_valid))

            # Standard deviation of p-values
            metrics["std_pval"] = float(np.std(p_valid))

        # Number of significant features
        significant_mask = ~np.isnan(p_values_adj) & (p_values_adj < self._alpha)
        n_significant = int(np.sum(significant_mask))
        metrics["n_significant"] = float(n_significant)
        metrics["n_tested"] = float(np.sum(~np.isnan(p_values)))

        # Proportion of significant features
        if metrics["n_tested"] > 0:
            metrics["prop_significant"] = n_significant / metrics["n_tested"]

        # FDR summary statistics
        valid_adj_mask = ~np.isnan(p_values_adj)
        if valid_adj_mask.sum() > 0:
            p_adj_valid = p_values_adj[valid_adj_mask]
            metrics["mean_fdr"] = float(np.mean(p_adj_valid))
            metrics["median_fdr"] = float(np.median(p_adj_valid))

        # Log2 fold change statistics
        valid_fc_mask = ~np.isnan(log2_fc)
        if valid_fc_mask.sum() > 0:
            fc_valid = log2_fc[valid_fc_mask]
            metrics["mean_log2_fc"] = float(np.mean(fc_valid))
            metrics["median_log2_fc"] = float(np.median(fc_valid))
            metrics["max_abs_log2_fc"] = float(np.max(np.abs(fc_valid)))

        # Number of features with large fold changes
        if valid_fc_mask.sum() > 0:
            large_fc_mask = np.abs(log2_fc) > 1.0  # 2-fold change
            metrics["n_large_fc"] = float(np.sum(large_fc_mask & ~np.isnan(log2_fc)))

        return metrics


__all__ = ["DifferentialExpressionTestModule"]
