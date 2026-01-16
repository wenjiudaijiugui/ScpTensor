"""
Core benchmarking framework classes.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from scptensor.core.structures import ScpContainer


@dataclass(frozen=True, slots=True)
class TechnicalMetrics:
    """Technical quality metrics for benchmark evaluation.

    Attributes
    ----------
    data_recovery_rate : float
        Proportion of missing values successfully handled.
    variance_preservation : float
        How well original variance structure is preserved.
    sparsity_preservation : float
        Preservation of original sparsity patterns.
    batch_mixing_score : float
        How well batches are mixed (higher = better).
    signal_to_noise_ratio : float
        Ratio of biological signal to technical noise.
    missing_value_pattern_score : float
        Quality of missing value handling.
    """

    data_recovery_rate: float
    variance_preservation: float
    sparsity_preservation: float
    batch_mixing_score: float
    signal_to_noise_ratio: float
    missing_value_pattern_score: float


@dataclass(frozen=True, slots=True)
class BiologicalMetrics:
    """Biological fidelity metrics for benchmark evaluation.

    Attributes
    ----------
    group_separation : float
        Separation between biological groups (e.g., silhouette score).
    biological_signal_preservation : float
        Preservation of true biological signals.
    clustering_consistency : float
        Consistency of clustering with ground truth.
    biological_variance_explained : float
        Percentage of variance explained by biological factors.
    differential_expression_concordance : float | None
        DE analysis consistency, if available.
    """

    group_separation: float
    biological_signal_preservation: float
    clustering_consistency: float
    biological_variance_explained: float
    differential_expression_concordance: float | None = None


@dataclass(frozen=True, slots=True)
class ComputationalMetrics:
    """Computational efficiency metrics.

    Attributes
    ----------
    runtime_seconds : float
        Total runtime in seconds.
    memory_usage_mb : float
        Peak memory usage in MB.
    scalability_factor : float
        How runtime scales with data size.
    convergence_iterations : int | None
        Number of iterations to convergence, if applicable.
    cpu_utilization_percent : float | None
        Average CPU usage, if available.
    """

    runtime_seconds: float
    memory_usage_mb: float
    scalability_factor: float
    convergence_iterations: int | None = None
    cpu_utilization_percent: float | None = None


@dataclass(slots=True)
class MethodRunResult:
    """Results from a single method run on a dataset.

    Attributes
    ----------
    method_name : str
        Name of the method that was run.
    parameters : dict[str, Any]
        Parameters used for the method run.
    dataset_name : str
        Name of the dataset used.
    input_container : ScpContainer
        Input data container.
    output_container : ScpContainer
        Output data container after processing.
    runtime_seconds : float
        Runtime in seconds.
    memory_usage_mb : float
        Memory usage in MB.
    technical_scores : TechnicalMetrics
        Technical quality metrics.
    biological_scores : BiologicalMetrics | None
        Biological quality metrics, if available.
    computational_scores : ComputationalMetrics
        Computational efficiency metrics.
    random_seed : int
        Random seed used for reproducibility.
    timestamp : str
        ISO timestamp of the run.
    software_versions : dict[str, str]
        Software version information.
    description : str | None
        Optional description of the run.
    """

    method_name: str
    parameters: dict[str, Any]
    dataset_name: str
    input_container: ScpContainer
    output_container: ScpContainer
    runtime_seconds: float
    memory_usage_mb: float
    technical_scores: TechnicalMetrics
    biological_scores: BiologicalMetrics | None
    computational_scores: ComputationalMetrics
    random_seed: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    software_versions: dict[str, str] = field(default_factory=dict)
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for easy serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary representation of the result.
        """
        return {
            "method_name": self.method_name,
            "parameters": self.parameters,
            "dataset_name": self.dataset_name,
            "runtime_seconds": self.runtime_seconds,
            "memory_usage_mb": self.memory_usage_mb,
            "technical_scores": asdict(self.technical_scores),
            "biological_scores": asdict(self.biological_scores)
            if self.biological_scores
            else None,
            "computational_scores": asdict(self.computational_scores),
            "random_seed": self.random_seed,
            "timestamp": self.timestamp,
            "software_versions": self.software_versions,
            "description": self.description,
        }


class BenchmarkResults:
    """Container for multiple benchmark runs with analysis capabilities."""

    __slots__ = ("runs", "datasets", "metadata", "_method_cache", "_dataset_cache")

    def __init__(self) -> None:
        self.runs: list[MethodRunResult] = []
        self.datasets: dict[str, ScpContainer] = {}
        self.metadata: dict[str, Any] = {}
        self._method_cache: list[str] | None = None
        self._dataset_cache: list[str] | None = None

    def add_run(self, result: MethodRunResult) -> None:
        """Add a new run result.

        Parameters
        ----------
        result : MethodRunResult
            Result to add.
        """
        self.runs.append(result)
        self._invalidate_cache()

    def add_dataset(self, name: str, container: ScpContainer) -> None:
        """Add a dataset reference.

        Parameters
        ----------
        name : str
            Dataset name.
        container : ScpContainer
            Dataset container.
        """
        self.datasets[name] = container

    def filter_by_method(self, method_name: str) -> list[MethodRunResult]:
        """Get all runs for a specific method.

        Parameters
        ----------
        method_name : str
            Method name to filter by.

        Returns
        -------
        list[MethodRunResult]
            Filtered results.
        """
        return [r for r in self.runs if r.method_name == method_name]

    def filter_by_dataset(self, dataset_name: str) -> list[MethodRunResult]:
        """Get all runs for a specific dataset.

        Parameters
        ----------
        dataset_name : str
            Dataset name to filter by.

        Returns
        -------
        list[MethodRunResult]
            Filtered results.
        """
        return [r for r in self.runs if r.dataset_name == dataset_name]

    def get_methods(self) -> list[str]:
        """Get list of all tested methods.

        Returns
        -------
        list[str]
            Unique method names.
        """
        if self._method_cache is None:
            self._method_cache = list({r.method_name for r in self.runs})
        return self._method_cache

    def get_datasets(self) -> list[str]:
        """Get list of all tested datasets.

        Returns
        -------
        list[str]
            Unique dataset names.
        """
        if self._dataset_cache is None:
            self._dataset_cache = list({r.dataset_name for r in self.runs})
        return self._dataset_cache

    def _invalidate_cache(self) -> None:
        """Invalidate cached method and dataset lists."""
        self._method_cache = None
        self._dataset_cache = None

    def get_parameter_sensitivity(
        self, method_name: str, dataset_name: str | None = None
    ) -> pd.DataFrame:
        """Analyze parameter sensitivity for a specific method.

        Parameters
        ----------
        method_name : str
            Method name to analyze.
        dataset_name : str | None
            Dataset name to filter by, if any.

        Returns
        -------
        pd.DataFrame
            DataFrame with parameters as columns and metrics as values.

        Raises
        ------
        ValueError
            If no runs found for the specified method/dataset.
        """
        runs = self.filter_by_method(method_name)
        if dataset_name:
            runs = [r for r in runs if r.dataset_name == dataset_name]

        if not runs:
            raise ValueError(
                f"No runs found for method {method_name}"
                + (f" and dataset {dataset_name}" if dataset_name else "")
            )

        data = [
            {
                **r.parameters,
                "data_recovery_rate": r.technical_scores.data_recovery_rate,
                "group_separation": r.biological_scores.group_separation
                if r.biological_scores
                else np.nan,
                "runtime_seconds": r.computational_scores.runtime_seconds,
                "memory_usage_mb": r.computational_scores.memory_usage_mb,
            }
            for r in runs
        ]

        return pd.DataFrame(data)

    def get_method_comparison(self, dataset_name: str | None = None) -> pd.DataFrame:
        """Get method comparison summary.

        Parameters
        ----------
        dataset_name : str | None
            Dataset name to filter by, if any.

        Returns
        -------
        pd.DataFrame
            DataFrame with methods as rows and aggregated metrics as columns.

        Raises
        ------
        ValueError
            If no runs found.
        """
        runs = self.runs.copy()
        if dataset_name:
            runs = [r for r in runs if r.dataset_name == dataset_name]

        if not runs:
            raise ValueError(
                "No runs found" + (f" for dataset {dataset_name}" if dataset_name else "")
            )

        methods = self.get_methods()
        method_data = {}

        for method in methods:
            method_runs = [r for r in runs if r.method_name == method]
            n_runs = len(method_runs)

            # Extract scores with vectorized operations
            runtimes = np.array([r.computational_scores.runtime_seconds for r in method_runs])
            memory = np.array([r.computational_scores.memory_usage_mb for r in method_runs])
            data_recovery = np.array([r.technical_scores.data_recovery_rate for r in method_runs])
            batch_mixing = np.array([r.technical_scores.batch_mixing_score for r in method_runs])

            bio_scores = [
                r.biological_scores.group_separation for r in method_runs if r.biological_scores
            ]
            group_sep = np.nanmean(bio_scores) if bio_scores else np.nan

            method_data[method] = {
                "mean_runtime_seconds": float(np.mean(runtimes)),
                "mean_memory_usage_mb": float(np.mean(memory)),
                "mean_data_recovery_rate": float(np.mean(data_recovery)),
                "mean_group_separation": float(group_sep) if not np.isnan(group_sep) else 0.0,
                "mean_batch_mixing_score": float(np.mean(batch_mixing)),
                "num_runs": n_runs,
            }

        return pd.DataFrame.from_dict(method_data, orient="index")

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all results to a single DataFrame.

        Returns
        -------
        pd.DataFrame
            Flattened DataFrame with all result data.
        """
        if not self.runs:
            return pd.DataFrame()

        data = [r.to_dict() for r in self.runs]
        df = pd.DataFrame(data)

        # Expand nested dictionaries
        technical_df = pd.json_normalize(df["technical_scores"]).add_prefix("technical_")
        biological_df = pd.json_normalize(df["biological_scores"]).add_prefix("biological_")
        computational_df = pd.json_normalize(df["computational_scores"]).add_prefix(
            "computational_"
        )

        return pd.concat(
            [
                df.drop(["technical_scores", "biological_scores", "computational_scores"], axis=1),
                technical_df,
                biological_df,
                computational_df,
            ],
            axis=1,
        )

    def export_data(self, format: str = "csv", filepath: str | None = None) -> str:
        """Export results data.

        Parameters
        ----------
        format : str
            Export format: 'csv', 'parquet', or 'json'.
        filepath : str | None
            Output filepath. Auto-generated if None.

        Returns
        -------
        str
            Path to exported file.

        Raises
        ------
        ValueError
            If format is not supported.
        """
        df = self.to_dataframe()

        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"benchmark_results_{timestamp}.{format}"

        export_handlers = {
            "csv": lambda p: df.to_csv(p, index=False),
            "parquet": lambda p: df.to_parquet(p, index=False),
            "json": lambda p: df.to_json(p, orient="records", indent=2),
        }

        if format not in export_handlers:
            raise ValueError(
                f"Unsupported export format: {format}. Choose from {list(export_handlers)}"
            )

        export_handlers[format](filepath)
        return filepath
