"""Comparison engine for ScpTensor vs Scanpy benchmarks.

This module executes comparative benchmarks and aggregates results.
"""

from __future__ import annotations

import dataclasses
import json
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import psutil

from scptensor.benchmark.data_provider import ComparisonDataset, get_provider
from scptensor.benchmark.method_registry import get_comparison_pairs
from scptensor.benchmark.scanpy_adapter import SCANPY_AVAILABLE, get_scanpy_methods
from scptensor.benchmark.scptensor_adapter import ScpTensorMethods

if TYPE_CHECKING:
    from collections.abc import Callable


class MetricType(Enum):
    RUNTIME = "runtime"
    MEMORY = "memory"
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    ARI = "ari"
    NMI = "nmi"
    SILHOUETTE = "silhouette"
    MSE = "mse"
    MAE = "mae"
    VARIANCE_EXPLAINED = "variance_explained"
    CORRELATION = "correlation"


@dataclasses.dataclass
class MethodResult:
    method_name: str
    framework: str
    dataset_name: str
    runtime_seconds: float
    memory_mb: float
    success: bool
    error_message: str | None = None
    output: np.ndarray | None = None
    metrics: dict = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class ComparisonResult:
    comparison_name: str
    dataset_name: str
    scptensor_result: MethodResult | None = None
    scanpy_result: MethodResult | None = None
    comparison_metrics: dict = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _json_serializer(obj: Any) -> Any:
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return str(obj)


def compute_accuracy_metrics(
    output: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    output_flat = output.ravel()
    reference_flat = reference.ravel()

    if mask is not None and mask.size == output.size:
        valid_mask = mask.ravel() == 0
        output_flat = output_flat[valid_mask]
        reference_flat = reference_flat[valid_mask]

    diff = output_flat - reference_flat
    metrics = {
        "mse": float(np.mean(diff ** 2)),
        "mae": float(np.mean(np.abs(diff))),
    }

    if len(output_flat) > 1:
        std_out, std_ref = np.std(output_flat), np.std(reference_flat)
        if std_out > 0 and std_ref > 0:
            metrics["correlation"] = float(np.corrcoef(output_flat, reference_flat)[0, 1])
        else:
            metrics["correlation"] = 0.0

        if np.any(reference_flat != 0):
            metrics["relative_error"] = float(np.mean(np.abs(diff) / (np.abs(reference_flat) + 1e-10)))
        else:
            metrics["relative_error"] = metrics["mae"]
    else:
        metrics["correlation"] = 0.0
        metrics["relative_error"] = metrics["mae"]

    return metrics


def compute_clustering_metrics(
    labels1: np.ndarray,
    labels2: np.ndarray,
    X: np.ndarray | None = None,
) -> dict[str, float]:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

    metrics = {
        "ari": float(adjusted_rand_score(labels1, labels2)),
        "nmi": float(normalized_mutual_info_score(labels1, labels2)),
    }

    if X is not None:
        try:
            metrics["silhouette"] = float(silhouette_score(X, labels1))
        except Exception:
            metrics["silhouette"] = 0.0

    return metrics


def compute_variance_metrics(
    output: np.ndarray,
    input_data: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    if mask is not None:
        valid_mask = mask == 0
        output_valid = output[valid_mask]
        input_valid = input_data[valid_mask]
    else:
        output_valid, input_valid = output.ravel(), input_data.ravel()

    input_var = float(np.var(input_valid))
    output_var = float(np.var(output_valid))

    metrics = {
        "variance_ratio": (output_var / input_var) if input_var > 0 else 1.0,
    }

    signal_mean = float(np.mean(output_valid))
    noise_std = float(np.std(output_valid - input_valid))
    metrics["snr_db"] = 20 * np.log10(abs(signal_mean) / (noise_std + 1e-10)) if noise_std > 0 else float("inf")

    return metrics


def _parse_method_result(result: tuple | np.ndarray, runtime: float = 0.0) -> tuple[np.ndarray | None, float, dict]:
    output, rt, metrics = None, runtime, {}

    if isinstance(result, tuple):
        if len(result) == 2:
            output, rt = result
        elif len(result) >= 3:
            output = result[0]
            rt = result[-1]
            if len(result) >= 3 and isinstance(result[1], np.ndarray):
                metrics["variance_ratio"] = result[1]
    else:
        output = result

    return output, rt, metrics


class ComparisonEngine:
    def __init__(self, output_dir: str | Path = "benchmark_results") -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(exist_ok=True, parents=True)
        self._data_provider = get_provider()
        self._scptensor_methods = ScpTensorMethods()
        self._scanpy_methods = get_scanpy_methods() if SCANPY_AVAILABLE else None
        self._results: dict[str, list[ComparisonResult]] = {}

    def run_shared_comparison(
        self,
        comparison_name: str,
        dataset: ComparisonDataset,
        **params,
    ) -> ComparisonResult:
        X, M, batches, groups = self._data_provider.get_dataset(dataset)

        pair = get_comparison_pairs().get(comparison_name)
        if pair is None:
            raise ValueError(f"Unknown comparison: {comparison_name}")

        scptensor_method_name, scanpy_method_name = pair

        scptensor_result = self._run_scptensor_method(scptensor_method_name, X, M, dataset.name, **params)

        if self._scanpy_methods is not None:
            scanpy_result = self._run_scanpy_method(scanpy_method_name, X, M, dataset.name, batches, groups, **params)
        else:
            scanpy_result = None

        comparison_metrics: dict[str, Any] = {}

        if scptensor_result.success and scanpy_result and scanpy_result.success:
            comparison_metrics["speedup"] = (
                scanpy_result.runtime_seconds / scptensor_result.runtime_seconds
                if scanpy_result.runtime_seconds > 0
                else float("inf")
            )

            comparison_metrics["memory_ratio"] = (
                scptensor_result.memory_mb / scanpy_result.memory_mb
                if scanpy_result.memory_mb > 0
                else 1.0
            )

            if (
                scptensor_result.output is not None
                and scanpy_result.output is not None
                and scptensor_result.output.shape == scanpy_result.output.shape
            ):
                comparison_metrics.update(compute_accuracy_metrics(scptensor_result.output, scanpy_result.output, M))

        result = ComparisonResult(
            comparison_name=comparison_name,
            dataset_name=dataset.name,
            scptensor_result=scptensor_result,
            scanpy_result=scanpy_result,
            comparison_metrics=comparison_metrics,
        )

        self._results.setdefault(comparison_name, []).append(result)
        return result

    def _run_scptensor_method(
        self,
        method_name: str,
        X: np.ndarray,
        M: np.ndarray | None,
        dataset_name: str,
        **params,
    ) -> MethodResult:
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024

        try:
            method = getattr(self._scptensor_methods, method_name, None)
            if method is None:
                return MethodResult(
                    method_name=method_name,
                    framework="scptensor",
                    dataset_name=dataset_name,
                    runtime_seconds=0.0,
                    memory_mb=0.0,
                    success=False,
                    error_message=f"Method {method_name} not found",
                )

            import time
            start = time.time()
            result = method(X, M, **params)
            runtime = time.time() - start

            output, runtime, metrics = _parse_method_result(result, runtime)
            memory_mb = max(0.0, psutil.Process().memory_info().rss / 1024 / 1024 - start_memory)

            return MethodResult(
                method_name=method_name,
                framework="scptensor",
                dataset_name=dataset_name,
                runtime_seconds=runtime,
                memory_mb=memory_mb,
                success=True,
                output=output,
                metrics=metrics,
            )
        except Exception as e:
            return MethodResult(
                method_name=method_name,
                framework="scptensor",
                dataset_name=dataset_name,
                runtime_seconds=0.0,
                memory_mb=0.0,
                success=False,
                error_message=str(e),
            )

    def _run_scanpy_method(
        self,
        method_name: str,
        X: np.ndarray,
        M: np.ndarray | None,
        dataset_name: str,
        batches: np.ndarray | None = None,
        groups: np.ndarray | None = None,
        **params,
    ) -> MethodResult:
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024

        try:
            method = getattr(self._scanpy_methods, method_name, None)
            if method is None:
                return MethodResult(
                    method_name=method_name,
                    framework="scanpy",
                    dataset_name=dataset_name,
                    runtime_seconds=0.0,
                    memory_mb=0.0,
                    success=False,
                    error_message=f"Method {method_name} not found",
                )

            import time
            start = time.time()
            result = method(X, M, **params)
            runtime = time.time() - start

            output, runtime, metrics = _parse_method_result(result, runtime)
            memory_mb = max(0.0, psutil.Process().memory_info().rss / 1024 / 1024 - start_memory)

            return MethodResult(
                method_name=method_name,
                framework="scanpy",
                dataset_name=dataset_name,
                runtime_seconds=runtime,
                memory_mb=memory_mb,
                success=True,
                output=output,
                metrics=metrics,
            )
        except Exception as e:
            return MethodResult(
                method_name=method_name,
                framework="scanpy",
                dataset_name=dataset_name,
                runtime_seconds=0.0,
                memory_mb=0.0,
                success=False,
                error_message=str(e),
            )

    def export_results(self) -> Path:
        export_data: dict[str, Any] = {
            "results": {
                name: [r.to_dict() for r in results]
                for name, results in self._results.items()
            },
            "summary": self._compute_summary(),
        }

        output_path = self._output_dir / "comparison_results.json"
        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2, default=_json_serializer)

        return output_path

    def _compute_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {}

        for name, results in self._results.items():
            if not results:
                continue

            scptensor_times = [r.scptensor_result.runtime_seconds for r in results if r.scptensor_result]
            scanpy_times = [
                r.scanpy_result.runtime_seconds
                for r in results
                if r.scanpy_result and r.scanpy_result.success
            ]

            speedups = [
                r.comparison_metrics.get("speedup", 1.0)
                for r in results
                if "speedup" in r.comparison_metrics
            ]

            summary[name] = {
                "n_comparisons": len(results),
                "scptensor_avg_time": float(np.mean(scptensor_times)) if scptensor_times else 0.0,
                "scanpy_avg_time": float(np.mean(scanpy_times)) if scanpy_times else 0.0,
                "avg_speedup": float(np.mean(speedups)) if speedups else 1.0,
            }

        return summary


def get_engine(output_dir: str | Path = "benchmark_results") -> ComparisonEngine:
    return ComparisonEngine(output_dir=output_dir)
