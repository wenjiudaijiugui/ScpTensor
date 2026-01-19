"""Differential expression performance test module."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from scptensor.benchmark.data_provider import get_provider
from scptensor.benchmark.modules.base import BaseModule, ModuleConfig, ModuleResult
from scptensor.benchmark.scptensor_adapter import _create_container
from scptensor.benchmark.utils.registry import register_module
from scptensor.diff_expr import diff_expr_mannwhitney, diff_expr_ttest
from scptensor.diff_expr.nonparametric import diff_expr_wilcoxon

if TYPE_CHECKING:
    from collections.abc import Callable


def _time_function(func: Callable[[], tuple]) -> tuple:
    return func()


@register_module("differential_expression_test")
class DifferentialExpressionTestModule(BaseModule):
    def __init__(self, config: ModuleConfig) -> None:
        super().__init__(config)
        self._methods: list[str] = config.params.get("methods", ["t_test", "wilcoxon"])
        self._alpha: float = config.params.get("alpha", 0.05)
        self._group1: str | None = config.params.get("group1", None)
        self._group2: str | None = config.params.get("group2", None)
        self._log2_fc_offset: float = config.params.get("log2_fc_offset", 1.0)
        self._missing_strategy: str = config.params.get("missing_strategy", "ignore")
        self._provider = get_provider()

    def run(self, dataset_name: str) -> list[ModuleResult]:
        if not self.should_process_dataset(dataset_name):
            return []

        results: list[ModuleResult] = []

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

        container = self._create_container_from_config(dataset_config)
        group1, group2 = self._get_group_names(container)

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

        for result in results:
            self._add_result(result)

        return results

    def _get_dataset_config(self, dataset_name: str):
        from scptensor.benchmark.data_provider import COMPARISON_DATASETS
        for dataset in COMPARISON_DATASETS:
            if dataset.name == dataset_name:
                return dataset
        return None

    def _create_container_from_config(self, config) -> ScpContainer:
        X, M, batches, groups = self._provider.get_dataset(config)
        return _create_container(X, M, batch_labels=batches, group_labels=groups)

    def _get_group_names(self, container) -> tuple[str, str]:
        groups = container.obs["group"].to_numpy()
        unique_groups = sorted(set(groups))
        if len(unique_groups) < 2:
            raise ValueError(f"Need at least 2 groups for DE test, found {len(unique_groups)}")
        g1 = self._group1 if self._group1 is not None else str(unique_groups[0])
        g2 = self._group2 if self._group2 is not None else str(unique_groups[1])
        return g1, g2

    def _run_t_test(self, dataset_name: str, container, group1: str, group2: str) -> ModuleResult:
        try:
            start_time = time.time()
            result = diff_expr_ttest(
                container,
                assay_name="protein",
                group_col="group",
                group1=group1,
                group2=group2,
                layer_name="raw",
                equal_var=False,
                missing_strategy=self._missing_strategy,
                log2_fc_offset=self._log2_fc_offset,
            )
            runtime = time.time() - start_time
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

    def _run_wilcoxon(self, dataset_name: str, container, group1: str, group2: str) -> ModuleResult:
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

    def _run_mannwhitney(self, dataset_name: str, container, group1: str, group2: str) -> ModuleResult:
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
        metrics: dict[str, float] = {}
        p_values = result.p_values
        p_values_adj = result.p_values_adj
        log2_fc = result.log2_fc

        valid_mask = ~np.isnan(p_values)
        p_valid = p_values[valid_mask]

        if len(p_valid) > 0:
            metrics["mean_pval"] = float(np.mean(p_valid))
            metrics["median_pval"] = float(np.median(p_valid))
            metrics["min_pval"] = float(np.min(p_valid))
            metrics["max_pval"] = float(np.max(p_valid))
            metrics["std_pval"] = float(np.std(p_valid))

        significant_mask = ~np.isnan(p_values_adj) & (p_values_adj < self._alpha)
        n_significant = int(np.sum(significant_mask))
        metrics["n_significant"] = float(n_significant)
        metrics["n_tested"] = float(np.sum(~np.isnan(p_values)))

        if metrics["n_tested"] > 0:
            metrics["prop_significant"] = n_significant / metrics["n_tested"]

        valid_adj_mask = ~np.isnan(p_values_adj)
        if valid_adj_mask.sum() > 0:
            p_adj_valid = p_values_adj[valid_adj_mask]
            metrics["mean_fdr"] = float(np.mean(p_adj_valid))
            metrics["median_fdr"] = float(np.median(p_adj_valid))

        valid_fc_mask = ~np.isnan(log2_fc)
        if valid_fc_mask.sum() > 0:
            fc_valid = log2_fc[valid_fc_mask]
            metrics["mean_log2_fc"] = float(np.mean(fc_valid))
            metrics["median_log2_fc"] = float(np.median(fc_valid))
            metrics["max_abs_log2_fc"] = float(np.max(np.abs(fc_valid)))

        if valid_fc_mask.sum() > 0:
            metrics["n_large_fc"] = float(np.sum((np.abs(log2_fc) > 1.0) & ~np.isnan(log2_fc)))

        return metrics


__all__ = ["DifferentialExpressionTestModule"]
