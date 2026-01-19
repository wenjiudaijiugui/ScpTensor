"""Performance evaluator for benchmarking algorithm runtime characteristics."""

from __future__ import annotations

import dataclasses
import time
import tracemalloc
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from typing import TypeVar

    T_co = TypeVar("T_co", covariant=True)

ArrayFloat = NDArray[np.float64]

_MB = 1024 * 1024


@dataclasses.dataclass(frozen=True, slots=True)
class PerformanceResult:
    runtime: float
    runtime_std: float
    memory_usage: float
    memory_usage_std: float
    throughput: float
    throughput_std: float
    cpu_time: float
    cpu_time_std: float
    n_runs: int
    n_samples: int | None

    def to_dict(self) -> dict[str, float]:
        return {
            "runtime": self.runtime,
            "runtime_std": self.runtime_std,
            "memory_usage": self.memory_usage,
            "memory_usage_std": self.memory_usage_std,
            "throughput": self.throughput,
            "throughput_std": self.throughput_std,
            "cpu_time": self.cpu_time,
            "cpu_time_std": self.cpu_time_std,
        }


class PerformanceEvaluator:
    """Evaluator for computational performance metrics."""

    __slots__ = ("n_runs", "warmup_runs", "track_memory", "track_cpu")

    def __init__(
        self,
        n_runs: int = 3,
        warmup_runs: int = 1,
        track_memory: bool = True,
        track_cpu: bool = True,
    ) -> None:
        self.n_runs = max(1, n_runs)
        self.warmup_runs = max(0, warmup_runs)
        self.track_memory = track_memory
        self.track_cpu = track_cpu

    def evaluate(
        self,
        func: Callable[..., T_co],
        X: ArrayFloat | Any,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, float]:
        for _ in range(self.warmup_runs):
            try:
                func(X, *args, **kwargs)
            except Exception:
                pass

        runtimes: list[float] = []
        memory_usages: list[float] = []
        cpu_times: list[float] = []

        for _ in range(self.n_runs):
            if self.track_memory:
                tracemalloc.start()
                start_mem = tracemalloc.get_traced_memory()[0]

            start_time = time.perf_counter()
            start_cpu = time.process_time()

            try:
                func(X, *args, **kwargs)
            except Exception as e:
                if self.track_memory:
                    tracemalloc.stop()
                raise RuntimeError(f"Function execution failed: {e}") from e

            end_time = time.perf_counter()
            end_cpu = time.process_time()

            if self.track_memory:
                current_mem = tracemalloc.get_traced_memory()[0]
                tracemalloc.stop()
                memory_usages.append(max(0.0, (current_mem - start_mem) / _MB))

            runtimes.append(end_time - start_time)
            if self.track_cpu:
                cpu_times.append(end_cpu - start_cpu)

        rt = np.array(runtimes)
        runtime_mean = float(rt.mean())
        runtime_std = float(rt.std(ddof=1)) if len(rt) > 1 else 0.0

        if self.track_memory and memory_usages:
            mem = np.array(memory_usages)
            memory_mean = float(mem.mean())
            memory_std = float(mem.std(ddof=1)) if len(mem) > 1 else 0.0
        else:
            memory_mean = 0.0
            memory_std = 0.0

        if self.track_cpu and cpu_times:
            cpu = np.array(cpu_times)
            cpu_mean = float(cpu.mean())
            cpu_std = float(cpu.std(ddof=1)) if len(cpu) > 1 else 0.0
        else:
            cpu_mean = 0.0
            cpu_std = 0.0

        n_samples = self._get_n_samples(X)
        throughput_mean = n_samples / runtime_mean if runtime_mean > 0 and n_samples else 0.0
        throughput_std = float((n_samples / rt).std(ddof=1)) if len(rt) > 1 and n_samples else 0.0

        return {
            "runtime": runtime_mean,
            "runtime_std": runtime_std,
            "memory_usage": memory_mean,
            "memory_usage_std": memory_std,
            "throughput": throughput_mean,
            "throughput_std": throughput_std,
            "cpu_time": cpu_mean,
            "cpu_time_std": cpu_std,
        }

    def evaluate_scalability(
        self,
        func: Callable[..., T_co],
        sizes: list[int],
        n_features: int = 50,
        **kwargs: Any,
    ) -> dict[int, dict[str, float]]:
        return {size: self.evaluate(func, np.random.randn(size, n_features), **kwargs) for size in sizes}

    def compare(
        self,
        funcs: dict[str, Callable[..., T_co]],
        X: ArrayFloat | Any,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, dict[str, float]]:
        results = {}
        nan_metrics = {k: np.nan for k in ["runtime", "runtime_std", "memory_usage", "memory_usage_std", "throughput", "throughput_std", "cpu_time", "cpu_time_std"]}

        for name, func in funcs.items():
            try:
                results[name] = self.evaluate(func, X, *args, **kwargs)
            except Exception as e:
                results[name] = {**nan_metrics, "error": str(e)}

        return results

    def to_performance_result(
        self, metrics: dict[str, float], n_samples: int | None = None
    ) -> PerformanceResult:
        return PerformanceResult(
            runtime=metrics.get("runtime", 0.0),
            runtime_std=metrics.get("runtime_std", 0.0),
            memory_usage=metrics.get("memory_usage", 0.0),
            memory_usage_std=metrics.get("memory_usage_std", 0.0),
            throughput=metrics.get("throughput", 0.0),
            throughput_std=metrics.get("throughput_std", 0.0),
            cpu_time=metrics.get("cpu_time", 0.0),
            cpu_time_std=metrics.get("cpu_time_std", 0.0),
            n_runs=self.n_runs,
            n_samples=n_samples,
        )


    @staticmethod
    def _get_n_samples(X: Any) -> int | None:
        if isinstance(X, np.ndarray):
            return len(X) if X.ndim == 1 else X.shape[0]
        if hasattr(X, "__len__"):
            return len(X)
        return None


def evaluate_performance(
    func: Callable[..., T_co],
    X: ArrayFloat | Any,
    *args: Any,
    n_runs: int = 3,
    warmup_runs: int = 1,
    track_memory: bool = True,
    track_cpu: bool = True,
    **kwargs: Any,
) -> dict[str, float]:
    return PerformanceEvaluator(n_runs, warmup_runs, track_memory, track_cpu).evaluate(func, X, *args, **kwargs)


def benchmark_scalability(
    func: Callable[..., T_co],
    sizes: list[int],
    n_features: int = 50,
    n_runs: int = 3,
    **kwargs: Any,
) -> dict[int, dict[str, float]]:
    return PerformanceEvaluator(n_runs=n_runs).evaluate_scalability(func, sizes, n_features, **kwargs)


__all__ = ["PerformanceEvaluator", "PerformanceResult", "evaluate_performance", "benchmark_scalability"]
