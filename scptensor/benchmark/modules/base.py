"""Base classes for benchmark modules."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class ModuleConfig:
    """Configuration for a benchmark module."""
    name: str
    enabled: bool = True
    datasets: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ModuleResult:
    """Result from a single benchmark module execution."""
    module_name: str
    dataset_name: str
    method_name: str
    output: np.ndarray | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    runtime_seconds: float = 0.0
    memory_mb: float = 0.0
    success: bool = True
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_name": self.module_name,
            "dataset_name": self.dataset_name,
            "method_name": self.method_name,
            "output": self.output.tolist() if self.output is not None else None,
            "metrics": self.metrics,
            "runtime_seconds": self.runtime_seconds,
            "memory_mb": self.memory_mb,
            "success": self.success,
            "error_message": self.error_message,
        }


class BaseModule(ABC):
    """Abstract base class for benchmark modules."""

    def __init__(self, config: ModuleConfig) -> None:
        self._config = config
        self._results: list[ModuleResult] = []

    @property
    def config(self) -> ModuleConfig:
        return self._config

    @abstractmethod
    def run(self, dataset_name: str) -> list[ModuleResult]:
        raise NotImplementedError

    def get_results(self) -> list[ModuleResult]:
        return self._results.copy()

    def clear_results(self) -> None:
        self._results.clear()

    def _add_result(self, result: ModuleResult) -> None:
        self._results.append(result)

    def is_enabled(self) -> bool:
        return self._config.enabled

    def should_process_dataset(self, dataset_name: str) -> bool:
        return not self._config.datasets or dataset_name in self._config.datasets

    def should_process_method(self, method_name: str) -> bool:
        return not self._config.methods or method_name in self._config.methods
