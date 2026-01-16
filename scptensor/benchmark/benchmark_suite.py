"""
Main benchmark orchestrator for comparing methods and parameters.
"""

from typing import Any

import numpy as np

from scptensor.core.structures import ScpContainer

from .core import BenchmarkResults, MethodRunResult
from .metrics import MetricsEngine
from .parameter_grid import MethodConfig, ParameterGrid


class BenchmarkSuite:
    """Main orchestrator for systematic method comparison and parameter optimization.

    Provides two main modes:
    1. Method comparison: Compare different methods with optimal parameters
    2. Parameter optimization: Find best parameters for a specific method

    Examples
    --------
    >>> from scptensor.benchmark import BenchmarkSuite, MethodConfig
    >>> methods = {"my_method": MethodConfig(method_class=my_func, ...)}
    >>> datasets = [container1, container2]
    >>> suite = BenchmarkSuite(methods, datasets)
    >>> results = suite.run_method_comparison()
    """

    __slots__ = (
        "methods",
        "datasets",
        "parameter_grids",
        "random_seed",
        "metrics_engine",
    )

    def __init__(
        self,
        methods: dict[str, MethodConfig],
        datasets: list[ScpContainer],
        parameter_grids: dict[str, dict[str, list[Any]]] | None = None,
        random_seed: int = 42,
    ):
        """Initialize benchmark suite.

        Parameters
        ----------
        methods : dict[str, MethodConfig]
            Dictionary mapping method names to configurations.
        datasets : list[ScpContainer]
            List of datasets to test on.
        parameter_grids : dict[str, dict[str, list[Any]]] | None
            Parameter grids for optimization.
        random_seed : int
            Random seed for reproducibility.

        Raises
        ------
        ValueError
            If any method class is not callable.
        """
        self.methods = methods
        self.datasets = {f"dataset_{i}": ds for i, ds in enumerate(datasets)}
        self.parameter_grids = parameter_grids or {}
        self.random_seed = random_seed
        self.metrics_engine = MetricsEngine()

        self._validate_methods()

    def _validate_methods(self) -> None:
        """Validate that all methods are properly configured.

        Raises
        ------
        ValueError
            If any method_class is not callable.
        """
        for name, config in self.methods.items():
            if not callable(config.method_class):
                raise ValueError(f"Method {name} must be callable")

    def run_method_comparison(
        self,
        optimize_params: bool = True,
        datasets: list[str] | None = None,
        methods: list[str] | None = None,
    ) -> BenchmarkResults:
        """Compare different methods with their optimal parameters.

        Parameters
        ----------
        optimize_params : bool
            Whether to optimize parameters before comparison.
        datasets : list[str] | None
            Dataset names to include (None = all).
        methods : list[str] | None
            Method names to include (None = all).

        Returns
        -------
        BenchmarkResults
            Comparison results.
        """
        datasets_to_use = datasets or list(self.datasets.keys())
        methods_to_use = methods or list(self.methods.keys())

        results = BenchmarkResults()

        for name in datasets_to_use:
            results.add_dataset(name, self.datasets[name])

        for method_name in methods_to_use:
            if method_name not in self.methods:
                continue

            method_config = self.methods[method_name]
            default_params = method_config.default_parameters

            for dataset_name in datasets_to_use:
                params = (
                    self._optimize_method_parameters(method_name, dataset_name)
                    if optimize_params
                    else default_params
                )
                result = self._run_method_on_dataset(method_name, dataset_name, params)
                results.add_run(result)

        return results

    def run_parameter_optimization(
        self,
        method_name: str,
        parameter_grid: dict[str, list[Any]] | None = None,
        datasets: list[str] | None = None,
    ) -> BenchmarkResults:
        """Optimize parameters for a specific method.

        Parameters
        ----------
        method_name : str
            Name of method to optimize.
        parameter_grid : dict[str, list[Any]] | None
            Parameter combinations to test (uses registered grid if None).
        datasets : list[str] | None
            Dataset names to include (None = all).

        Returns
        -------
        BenchmarkResults
            Parameter optimization results.

        Raises
        ------
        ValueError
            If method not found.
        """
        if method_name not in self.methods:
            raise ValueError(f"Method {method_name} not found")

        parameter_grid = parameter_grid or self.parameter_grids.get(method_name)
        if not parameter_grid:
            parameter_grid = {
                k: [v] for k, v in self.methods[method_name].default_parameters.items()
            }

        datasets_to_use = datasets or list(self.datasets.keys())
        results = BenchmarkResults()

        for name in datasets_to_use:
            results.add_dataset(name, self.datasets[name])

        param_grid = ParameterGrid(parameter_grid)  # type: ignore[arg-type]
        combinations = param_grid.generate_combinations(strategy="grid")

        for dataset_name in datasets_to_use:
            for params in combinations:
                result = self._run_method_on_dataset(method_name, dataset_name, params)
                results.add_run(result)

        return results

    def _optimize_method_parameters(
        self, method_name: str, dataset_name: str, n_trials: int = 50
    ) -> dict[str, Any]:
        """Find optimal parameters for a method using grid search.

        Parameters
        ----------
        method_name : str
            Name of method to optimize.
        dataset_name : str
            Name of dataset to optimize on.
        n_trials : int
            Maximum number of parameter combinations to try.

        Returns
        -------
        dict[str, Any]
            Optimal parameter dictionary.
        """
        if method_name not in self.parameter_grids:
            return self.methods[method_name].default_parameters

        param_grid = ParameterGrid(self.parameter_grids[method_name])  # type: ignore[arg-type]
        combinations = param_grid.generate_combinations(strategy="grid")[:n_trials]

        best_score = -np.inf
        best_params = self.methods[method_name].default_parameters

        for params in combinations:
            try:
                result = self._run_method_on_dataset(method_name, dataset_name, params)
                score = self._compute_optimization_score(result)
                if score > best_score:
                    best_score, best_params = score, params
            except Exception:
                continue

        return best_params

    def _compute_optimization_score(self, result: MethodRunResult) -> float:
        """Compute optimization score for parameter selection.

        Uses a weighted combination of metrics:
        - 40% biological signal preservation
        - 30% technical quality
        - 20% computational efficiency (inverse)
        - 10% data recovery

        Parameters
        ----------
        result : MethodRunResult
            Result to score.

        Returns
        -------
        float
            Optimization score.
        """
        bio = result.biological_scores
        biological_score = (
            (
                bio.group_separation * 0.5
                + bio.biological_signal_preservation * 0.3
                + bio.clustering_consistency * 0.2
            )
            if bio
            else 0.0
        )

        tech = result.technical_scores
        technical_score = (
            tech.data_recovery_rate * 0.3
            + tech.batch_mixing_score * 0.3
            + tech.signal_to_noise_ratio * 0.2
            + tech.variance_preservation * 0.2
        )

        comp_score = 1.0 / (1.0 + result.computational_scores.runtime_seconds)

        return (
            biological_score * 0.4
            + technical_score * 0.3
            + comp_score * 0.2
            + tech.data_recovery_rate * 0.1
        )

    def _run_method_on_dataset(
        self, method_name: str, dataset_name: str, parameters: dict[str, Any]
    ) -> MethodRunResult:
        """Run a method on a dataset with specific parameters.

        Parameters
        ----------
        method_name : str
            Name of method to run.
        dataset_name : str
            Name of dataset to run on.
        parameters : dict[str, Any]
            Parameters to use.

        Returns
        -------
        MethodRunResult
            Result with computed metrics.

        Raises
        ------
        RuntimeError
            If method execution fails.
        """
        # Seed for reproducibility
        np.random.seed(
            self.random_seed + hash(method_name + dataset_name + str(parameters)) % 10000
        )

        method_config = self.methods[method_name]
        method_config.validate_parameters(parameters)

        input_copy = self._deep_copy_container(self.datasets[dataset_name])

        self.metrics_engine.start_timing()
        try:
            output_container = method_config.method_class(input_copy, **parameters)
            runtime, memory = self.metrics_engine.stop_timing()

            return MethodRunResult(
                method_name=method_name,
                parameters=parameters,
                dataset_name=dataset_name,
                input_container=input_copy,
                output_container=output_container,
                runtime_seconds=runtime,
                memory_usage_mb=memory,
                technical_scores=self.metrics_engine.evaluate_technical(
                    input_copy, output_container
                ),
                biological_scores=self.metrics_engine.evaluate_biological(output_container),
                computational_scores=self.metrics_engine.evaluate_computational(runtime, memory),
                random_seed=self.random_seed,
            )
        except Exception as e:
            self.metrics_engine.stop_timing()
            raise RuntimeError(f"Method {method_name} failed on dataset {dataset_name}: {e}") from e

    @staticmethod
    def _deep_copy_container(container: ScpContainer) -> ScpContainer:
        """Create a deep copy of a ScpContainer.

        Parameters
        ----------
        container : ScpContainer
            Container to copy.

        Returns
        -------
        ScpContainer
            Deep copy of the container.
        """
        import copy

        return copy.deepcopy(container)
