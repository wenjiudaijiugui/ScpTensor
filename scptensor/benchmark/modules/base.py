"""
Base classes for benchmark modules.

This module provides the foundational abstractions for creating modular
benchmark components that can be easily extended and composed.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class ModuleConfig:
    """Configuration for a benchmark module.

    A ModuleConfig defines the behavior and scope of a benchmark module,
    including which datasets and methods to operate on.

    Attributes
    ----------
    name : str
        Unique identifier for the module.
    enabled : bool, default=True
        Whether the module is active in benchmark runs.
    datasets : list[str], default=[]
        List of dataset names to process. Empty list means all datasets.
    methods : list[str], default=[]
        List of method names to benchmark. Empty list means all methods.
    params : dict[str, Any], default={}
        Additional module-specific parameters.
    """

    name: str
    enabled: bool = True
    datasets: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ModuleResult:
    """Result from a single benchmark module execution.

    Encapsulates the output, metrics, and metadata from running a benchmark
    module on a specific dataset-method combination.

    Attributes
    ----------
    module_name : str
        Name of the module that produced this result.
    dataset_name : str
        Name of the dataset processed.
    method_name : str
        Name of the method applied.
    output : np.ndarray | None, default=None
        Computed output array, if applicable.
    metrics : dict[str, float], default={}
        Computed metric names and values.
    runtime_seconds : float, default=0.0
        Execution time in seconds.
    memory_mb : float, default=0.0
        Peak memory usage in megabytes.
    success : bool, default=True
        Whether execution completed successfully.
    error_message : str | None, default=None
        Error message if execution failed.
    """

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
        """Convert result to a dictionary for serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary representation of the result. NumPy arrays are
            converted to lists for JSON compatibility.
        """
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
    """Abstract base class for benchmark modules.

    BaseModule defines the interface that all benchmark modules must implement.
    Subclasses should implement the :meth:`run` method to execute their
    specific benchmark logic.

    Parameters
    ----------
    config : ModuleConfig
        Configuration object controlling module behavior.

    Examples
    --------
    >>> from scptensor.benchmark.modules.base import BaseModule, ModuleConfig
    >>>
    >>> class MyModule(BaseModule):
    ...     def run(self, dataset_name: str) -> list[ModuleResult]:
    ...         # Implementation here
    ...         return results
    >>>
    >>> config = ModuleConfig(name="my_module", datasets=["dataset1"])
    >>> module = MyModule(config)
    >>> results = module.run("dataset1")
    """

    def __init__(self, config: ModuleConfig) -> None:
        """Initialize the module with configuration.

        Parameters
        ----------
        config : ModuleConfig
            Configuration object for this module instance.
        """
        self._config = config
        self._results: list[ModuleResult] = []

    @property
    def config(self) -> ModuleConfig:
        """Get the module configuration.

        Returns
        -------
        ModuleConfig
            The configuration object for this module.
        """
        return self._config

    @abstractmethod
    def run(self, dataset_name: str) -> list[ModuleResult]:
        """Execute the benchmark module on a dataset.

        This method must be implemented by subclasses to define the
        specific benchmark logic.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset to process.

        Returns
        -------
        list[ModuleResult]
            List of results from running the module, one per method
            or configuration combination tested.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement run()")

    def get_results(self) -> list[ModuleResult]:
        """Get all accumulated results from module runs.

        Returns
        -------
        list[ModuleResult]
            List of all results collected from previous run() calls.
        """
        return self._results.copy()

    def clear_results(self) -> None:
        """Clear all accumulated results.

        Resets the internal results storage to an empty list.
        """
        self._results.clear()

    def _add_result(self, result: ModuleResult) -> None:
        """Add a result to the internal storage.

        Parameters
        ----------
        result : ModuleResult
            Result to add to storage.
        """
        self._results.append(result)

    def is_enabled(self) -> bool:
        """Check if the module is enabled.

        Returns
        -------
        bool
            True if the module is enabled in its configuration.
        """
        return self._config.enabled

    def should_process_dataset(self, dataset_name: str) -> bool:
        """Check if a dataset should be processed by this module.

        A dataset is processed if the module's datasets list is empty
        or if the dataset name is explicitly listed.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset to check.

        Returns
        -------
        bool
            True if the dataset should be processed.
        """
        return not self._config.datasets or dataset_name in self._config.datasets

    def should_process_method(self, method_name: str) -> bool:
        """Check if a method should be processed by this module.

        A method is processed if the module's methods list is empty
        or if the method name is explicitly listed.

        Parameters
        ----------
        method_name : str
            Name of the method to check.

        Returns
        -------
        bool
            True if the method should be processed.
        """
        return not self._config.methods or method_name in self._config.methods
