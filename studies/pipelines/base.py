"""Pipeline base class for single-cell proteomics analysis.

This module provides the base infrastructure for implementing different
analysis pipelines that can be compared in a standardized framework.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import yaml


def load_pipeline_config(pipeline_name: str) -> dict[str, Any]:
    """
    Load pipeline configuration from YAML file.

    Parameters
    ----------
    pipeline_name : str
        Name of the pipeline (e.g., 'pipeline_a')

    Returns
    -------
    dict
        Configuration dictionary with steps and parameters

    Raises
    ------
    FileNotFoundError
        If configuration file is not found
    KeyError
        If pipeline_name is not in configuration

    Examples
    --------
    >>> config = load_pipeline_config("pipeline_a")
    >>> config["name"]
    'Classic Pipeline'
    """
    config_path = Path(__file__).parent.parent / "configs" / "pipeline_configs.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        configs = yaml.safe_load(f)

    if pipeline_name not in configs:
        available = list(configs.keys())
        raise KeyError(
            f"Pipeline '{pipeline_name}' not found in configuration. "
            f"Available pipelines: {available}"
        )

    return configs[pipeline_name]


class BasePipeline(ABC):
    """
    Abstract base class for analysis pipelines.

    This class provides the infrastructure for implementing different
    single-cell proteomics analysis pipelines with standardized execution
    logging and configuration management.

    Parameters
    ----------
    name : str
        Pipeline name (e.g., "Classic Pipeline")
    config : Dict[str, Any]
        Pipeline configuration dictionary with steps and parameters
    random_seed : int, optional
        Random seed for reproducibility, default 42

    Attributes
    ----------
    name : str
        Pipeline name
    config : Dict[str, Any]
        Pipeline configuration
    random_seed : int
        Random seed for reproducibility
    steps_log : list
        Execution log with timing information for each step

    Examples
    --------
    >>> from scptensor import create_test_container
    >>> pipeline = PipelineA(name="Pipeline A", config={})
    >>> result_container = pipeline.run(container)
    >>> print(pipeline.get_execution_log())
    """

    def __init__(self, name: str, config: dict[str, Any], random_seed: int = 42):
        """
        Initialize pipeline.

        Parameters
        ----------
        name : str
            Pipeline name
        config : Dict[str, Any]
            Pipeline configuration
        random_seed : int, optional
            Random seed for reproducibility
        """
        self.name = name
        self.config = config
        self.random_seed = random_seed
        self.steps_log: list[dict[str, Any]] = []

    @abstractmethod
    def run(self, container: Any) -> Any:
        """
        Execute the complete analysis pipeline.

        This method must be implemented by subclasses to define the
        specific analysis steps for each pipeline variant.

        Parameters
        ----------
        container : ScpContainer
            Input data container

        Returns
        -------
        ScpContainer
            Processed data container with all analysis results

        Raises
        ------
        NotImplementedError
            If subclass does not implement this method
        """
        raise NotImplementedError("Subclasses must implement run() method")

    def get_config(self) -> dict[str, Any]:
        """
        Return pipeline configuration.

        Returns
        -------
        Dict[str, Any]
            Pipeline configuration dictionary
        """
        return self.config

    def get_execution_log(self) -> list[dict[str, Any]]:
        """
        Return execution log with timing information.

        Returns
        -------
        list[Dict[str, Any]]
            List of step execution records with 'step' and 'runtime' keys
        """
        return self.steps_log

    def _log_step(self, step_name: str, runtime: float) -> None:
        """
        Log a step execution with timing.

        Parameters
        ----------
        step_name : str
            Name of the step (e.g., "qc", "normalization")
        runtime : float
            Execution time in seconds
        """
        self.steps_log.append({"step": step_name, "runtime": runtime})

    def _execute_step(self, step_name: str, step_func: callable, *args: Any, **kwargs: Any) -> Any:
        """
        Execute a pipeline step with timing and error handling.

        Parameters
        ----------
        step_name : str
            Name of the step for logging
        step_func : callable
            Function to execute
        *args : Any
            Positional arguments to pass to step_func
        **kwargs : Any
            Keyword arguments to pass to step_func

        Returns
        -------
        Any
            Result from step_func

        Raises
        ------
        Exception
            If step_func raises an exception
        """
        start_time = time.time()
        try:
            result = step_func(*args, **kwargs)
            runtime = time.time() - start_time
            self._log_step(step_name, runtime)
            return result
        except Exception as e:
            runtime = time.time() - start_time
            self._log_step(f"{step_name}_failed", runtime)
            raise RuntimeError(
                f"Pipeline step '{step_name}' failed after {runtime:.2f}s: {str(e)}"
            ) from e
