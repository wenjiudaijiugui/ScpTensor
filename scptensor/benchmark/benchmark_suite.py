"""
Main benchmark orchestrator for comparing methods and parameters.
"""

import warnings
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Union
from scptensor.core.structures import ScpContainer
from .core import BenchmarkResults, MethodRunResult
from .metrics import MetricsEngine, TechnicalMetrics, BiologicalMetrics, ComputationalMetrics
from .parameter_grid import ParameterGrid, MethodConfig


class BenchmarkSuite:
    """
    Main orchestrator for systematic method comparison and parameter optimization.

    Provides two main modes:
    1. Method comparison: Compare different methods with optimal parameters
    2. Parameter optimization: Find best parameters for a specific method
    """

    def __init__(
        self,
        methods: Dict[str, MethodConfig],
        datasets: List[ScpContainer],
        parameter_grids: Optional[Dict[str, Dict[str, List[Any]]]] = None,
        random_seed: int = 42
    ):
        """
        Initialize benchmark suite.

        Args:
            methods: Dictionary of method_name -> MethodConfig
            datasets: List of datasets to test on
            parameter_grids: Parameter grids for optimization (optional)
            random_seed: Random seed for reproducibility
        """
        self.methods = methods
        self.datasets = {f"dataset_{i}": dataset for i, dataset in enumerate(datasets)}
        self.parameter_grids = parameter_grids or {}
        self.random_seed = random_seed
        self.metrics_engine = MetricsEngine()

        # Validate methods
        self._validate_methods()

    def _validate_methods(self):
        """Validate that all methods are properly configured."""
        for method_name, method_config in self.methods.items():
            if not hasattr(method_config.method_class, '__call__'):
                raise ValueError(f"Method {method_name} must be callable")

    def run_method_comparison(
        self,
        optimize_params: bool = True,
        datasets: Optional[List[str]] = None,
        methods: Optional[List[str]] = None
    ) -> BenchmarkResults:
        """
        Compare different methods with their optimal parameters.

        Args:
            optimize_params: Whether to optimize parameters before comparison
            datasets: List of dataset names to include (None = all)
            methods: List of method names to include (None = all)

        Returns:
            BenchmarkResults with comparison data
        """
        datasets_to_use = datasets or list(self.datasets.keys())
        methods_to_use = methods or list(self.methods.keys())

        results = BenchmarkResults()

        # Add datasets to results
        for dataset_name in datasets_to_use:
            results.add_dataset(dataset_name, self.datasets[dataset_name])

        for method_name in methods_to_use:
            if method_name not in self.methods:
                warnings.warn(f"Method {method_name} not found, skipping")
                continue

            for dataset_name in datasets_to_use:
                dataset = self.datasets[dataset_name]

                # Find optimal parameters if requested
                if optimize_params:
                    best_params = self._optimize_method_parameters(
                        method_name, dataset_name
                    )
                else:
                    best_params = self.methods[method_name].default_parameters

                # Run method with optimal parameters
                result = self._run_method_on_dataset(
                    method_name, dataset_name, best_params
                )

                results.add_run(result)

        return results

    def run_parameter_optimization(
        self,
        method_name: str,
        parameter_grid: Optional[Dict[str, List[Any]]] = None,
        datasets: Optional[List[str]] = None
    ) -> BenchmarkResults:
        """
        Optimize parameters for a specific method across multiple parameter combinations.

        Args:
            method_name: Name of method to optimize
            parameter_grid: Parameter combinations to test (uses registered grid if None)
            datasets: List of dataset names to include (None = all)

        Returns:
            BenchmarkResults with parameter optimization data
        """
        if method_name not in self.methods:
            raise ValueError(f"Method {method_name} not found")

        parameter_grid = parameter_grid or self.parameter_grids.get(method_name, {})
        if not parameter_grid:
            # Use default parameters
            parameter_grid = {k: [v] for k, v in self.methods[method_name].default_parameters.items()}

        datasets_to_use = datasets or list(self.datasets.keys())

        results = BenchmarkResults()

        # Add datasets to results
        for dataset_name in datasets_to_use:
            results.add_dataset(dataset_name, self.datasets[dataset_name])

        # Generate all parameter combinations
        param_grid = ParameterGrid(parameter_grid)

        for dataset_name in datasets_to_use:
            dataset = self.datasets[dataset_name]

            for params in param_grid.generate_combinations(strategy='grid'):
                # Run method with current parameters
                result = self._run_method_on_dataset(
                    method_name, dataset_name, params
                )

                results.add_run(result)

        return results

    def _optimize_method_parameters(
        self,
        method_name: str,
        dataset_name: str,
        n_trials: int = 50
    ) -> Dict[str, Any]:
        """
        Find optimal parameters for a method using grid search.

        Args:
            method_name: Name of method to optimize
            dataset_name: Name of dataset to optimize on
            n_trials: Maximum number of parameter combinations to try

        Returns:
            Dictionary of optimal parameters
        """
        if method_name not in self.parameter_grids:
            return self.methods[method_name].default_parameters

        dataset = self.datasets[dataset_name]
        param_grid = ParameterGrid(self.parameter_grids[method_name])

        # Generate parameter combinations (limit to n_trials)
        combinations = param_grid.generate_combinations(strategy='grid')
        if len(combinations) > n_trials:
            combinations = combinations[:n_trials]

        best_score = -np.inf
        best_params = self.methods[method_name].default_parameters

        for params in combinations:
            try:
                # Run method with current parameters
                result = self._run_method_on_dataset(method_name, dataset_name, params)

                # Compute optimization score (weighted combination of metrics)
                score = self._compute_optimization_score(result)

                if score > best_score:
                    best_score = score
                    best_params = params

            except Exception as e:
                warnings.warn(f"Parameter combination failed: {params}, error: {e}")
                continue

        return best_params

    def _compute_optimization_score(self, result: MethodRunResult) -> float:
        """
        Compute optimization score for parameter selection.

        Uses a weighted combination of metrics:
        - 40% biological signal preservation
        - 30% technical quality
        - 20% computational efficiency (inverse)
        - 10% data recovery
        """
        biological_score = 0.0
        if result.biological_scores:
            biological_score = (
                result.biological_scores.group_separation * 0.5 +
                result.biological_scores.biological_signal_preservation * 0.3 +
                result.biological_scores.clustering_consistency * 0.2
            )

        technical_score = (
            result.technical_scores.data_recovery_rate * 0.3 +
            result.technical_scores.batch_mixing_score * 0.3 +
            result.technical_scores.signal_to_noise_ratio * 0.2 +
            result.technical_scores.variance_preservation * 0.2
        )

        # Computational efficiency (inverse - lower is better)
        computational_score = 1.0 / (1.0 + result.computational_scores.runtime_seconds)

        overall_score = (
            biological_score * 0.4 +
            technical_score * 0.3 +
            computational_score * 0.2 +
            result.technical_scores.data_recovery_rate * 0.1
        )

        return overall_score

    def _run_method_on_dataset(
        self,
        method_name: str,
        dataset_name: str,
        parameters: Dict[str, Any]
    ) -> MethodRunResult:
        """
        Run a method on a dataset with specific parameters.

        Args:
            method_name: Name of method to run
            dataset_name: Name of dataset to run on
            parameters: Parameters to use

        Returns:
            MethodRunResult with metrics
        """
        # Set random seed for reproducibility
        np.random.seed(self.random_seed + hash(method_name + dataset_name + str(parameters)) % 10000)

        # Get method and dataset
        method_config = self.methods[method_name]
        method_class = method_config.method_class
        input_container = self.datasets[dataset_name]

        # Validate parameters
        method_config.validate_parameters(parameters)

        # Start timing
        self.metrics_engine.start_timing()

        # Copy input container to avoid modification
        input_copy = self._deep_copy_container(input_container)

        try:
            # Run method
            output_container = method_class(input_copy, **parameters)

            # Stop timing
            runtime, memory = self.metrics_engine.stop_timing()

            # Compute metrics
            technical_scores = self.metrics_engine.evaluate_technical(
                input_copy, output_container
            )

            biological_scores = self.metrics_engine.evaluate_biological(
                output_container
            )

            computational_scores = self.metrics_engine.evaluate_computational(
                runtime, memory
            )

            # Create result
            result = MethodRunResult(
                method_name=method_name,
                parameters=parameters,
                dataset_name=dataset_name,
                input_container=input_copy,
                output_container=output_container,
                runtime_seconds=runtime,
                memory_usage_mb=memory,
                technical_scores=technical_scores,
                biological_scores=biological_scores,
                computational_scores=computational_scores,
                random_seed=self.random_seed
            )

            return result

        except Exception as e:
            # Stop timing even if method failed
            self.metrics_engine.stop_timing()
            raise RuntimeError(f"Method {method_name} failed on dataset {dataset_name}: {e}")

    def _deep_copy_container(self, container: ScpContainer) -> ScpContainer:
        """Create a deep copy of a ScpContainer."""
        # This is a simplified deep copy - in practice, you might want more sophisticated copying
        import copy
        return copy.deepcopy(container)