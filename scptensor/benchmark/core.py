"""
Core benchmarking framework classes.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import numpy as np
import pandas as pd
from scptensor.core.structures import ScpContainer


@dataclass
class TechnicalMetrics:
    """Technical quality metrics for benchmark evaluation."""

    data_recovery_rate: float  # Proportion of missing values successfully handled
    variance_preservation: float  # How well original variance structure is preserved
    sparsity_preservation: float  # Preservation of original sparsity patterns
    batch_mixing_score: float  # How well batches are mixed (higher = better)
    signal_to_noise_ratio: float  # Ratio of biological signal to technical noise
    missing_value_pattern_score: float  # Quality of missing value handling


@dataclass
class BiologicalMetrics:
    """Biological fidelity metrics for benchmark evaluation."""

    group_separation: float  # Separation between biological groups (e.g., silhouette score)
    biological_signal_preservation: float  # Preservation of true biological signals
    clustering_consistency: float  # Consistency of clustering with ground truth
    biological_variance_explained: float  # % of variance explained by biological factors
    differential_expression_concordance: Optional[float] = None  # DE analysis consistency


@dataclass
class ComputationalMetrics:
    """Computational efficiency metrics."""

    runtime_seconds: float  # Total runtime in seconds
    memory_usage_mb: float  # Peak memory usage in MB
    scalability_factor: float  # How runtime scales with data size
    convergence_iterations: Optional[int] = None  # Number of iterations to convergence
    cpu_utilization_percent: Optional[float] = None  # Average CPU usage


@dataclass
class MethodRunResult:
    """Results from a single method run on a dataset."""

    method_name: str
    parameters: Dict[str, Any]
    dataset_name: str

    # Data containers
    input_container: ScpContainer
    output_container: ScpContainer

    # Computational information
    runtime_seconds: float
    memory_usage_mb: float

    # Quality metrics
    technical_scores: TechnicalMetrics
    biological_scores: Optional[BiologicalMetrics]
    computational_scores: ComputationalMetrics

    # Metadata
    random_seed: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    software_versions: Dict[str, str] = field(default_factory=dict)
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy serialization."""
        return {
            'method_name': self.method_name,
            'parameters': self.parameters,
            'dataset_name': self.dataset_name,
            'runtime_seconds': self.runtime_seconds,
            'memory_usage_mb': self.memory_usage_mb,
            'technical_scores': self.technical_scores.__dict__,
            'biological_scores': self.biological_scores.__dict__ if self.biological_scores else None,
            'computational_scores': self.computational_scores.__dict__,
            'random_seed': self.random_seed,
            'timestamp': self.timestamp,
            'software_versions': self.software_versions,
            'description': self.description
        }


class BenchmarkResults:
    """Container for multiple benchmark runs with analysis capabilities."""

    def __init__(self):
        self.runs: List[MethodRunResult] = []
        self.datasets: Dict[str, ScpContainer] = {}
        self.metadata: Dict[str, Any] = {}

    def add_run(self, result: MethodRunResult):
        """Add a new run result."""
        self.runs.append(result)

    def add_dataset(self, name: str, container: ScpContainer):
        """Add a dataset reference."""
        self.datasets[name] = container

    def filter_by_method(self, method_name: str) -> List[MethodRunResult]:
        """Get all runs for a specific method."""
        return [run for run in self.runs if run.method_name == method_name]

    def filter_by_dataset(self, dataset_name: str) -> List[MethodRunResult]:
        """Get all runs for a specific dataset."""
        return [run for run in self.runs if run.dataset_name == dataset_name]

    def get_methods(self) -> List[str]:
        """Get list of all tested methods."""
        return list(set(run.method_name for run in self.runs))

    def get_datasets(self) -> List[str]:
        """Get list of all tested datasets."""
        return list(set(run.dataset_name for run in self.runs))

    def get_parameter_sensitivity(self, method_name: str, dataset_name: Optional[str] = None) -> pd.DataFrame:
        """
        Analyze parameter sensitivity for a specific method.

        Returns DataFrame with parameters as columns and metrics as values.
        """
        runs = self.filter_by_method(method_name)
        if dataset_name:
            runs = [run for run in runs if run.dataset_name == dataset_name]

        if not runs:
            raise ValueError(f"No runs found for method {method_name}" +
                           (f" and dataset {dataset_name}" if dataset_name else ""))

        # Extract parameters and metrics
        data = []
        for run in runs:
            row = run.parameters.copy()
            row.update({
                'data_recovery_rate': run.technical_scores.data_recovery_rate,
                'group_separation': run.biological_scores.group_separation if run.biological_scores else np.nan,
                'runtime_seconds': run.computational_scores.runtime_seconds,
                'memory_usage_mb': run.computational_scores.memory_usage_mb,
            })
            data.append(row)

        return pd.DataFrame(data)

    def get_method_comparison(self, dataset_name: Optional[str] = None) -> pd.DataFrame:
        """
        Get method comparison summary.

        Returns DataFrame with methods as rows and aggregated metrics as columns.
        """
        runs = self.runs.copy()
        if dataset_name:
            runs = [run for run in runs if run.dataset_name == dataset_name]

        if not runs:
            raise ValueError(f"No runs found" +
                           (f" for dataset {dataset_name}" if dataset_name else ""))

        # Aggregate by method (take mean across parameter variations)
        method_data = {}
        for method in self.get_methods():
            method_runs = [run for run in runs if run.method_name == method]

            method_data[method] = {
                'mean_runtime_seconds': np.mean([run.computational_scores.runtime_seconds for run in method_runs]),
                'mean_memory_usage_mb': np.mean([run.computational_scores.memory_usage_mb for run in method_runs]),
                'mean_data_recovery_rate': np.mean([run.technical_scores.data_recovery_rate for run in method_runs]),
                'mean_group_separation': np.nanmean([run.biological_scores.group_separation
                                                   for run in method_runs if run.biological_scores]),
                'mean_batch_mixing_score': np.mean([run.technical_scores.batch_mixing_score for run in method_runs]),
                'num_runs': len(method_runs)
            }

        return pd.DataFrame.from_dict(method_data, orient='index')

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all results to a single DataFrame."""
        data = [run.to_dict() for run in self.runs]
        df = pd.DataFrame(data)

        # Expand nested dictionaries
        technical_df = pd.json_normalize(df['technical_scores']).add_prefix('technical_')
        biological_df = pd.json_normalize(df['biological_scores']).add_prefix('biological_')
        computational_df = pd.json_normalize(df['computational_scores']).add_prefix('computational_')

        result = pd.concat([
            df.drop(['technical_scores', 'biological_scores', 'computational_scores'], axis=1),
            technical_df,
            biological_df,
            computational_df
        ], axis=1)

        return result

    def export_data(self, format: str = 'csv', filepath: Optional[str] = None) -> str:
        """Export results data."""
        df = self.to_dataframe()

        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"benchmark_results_{timestamp}.{format}"

        if format == 'csv':
            df.to_csv(filepath, index=False)
        elif format == 'parquet':
            df.to_parquet(filepath, index=False)
        elif format == 'json':
            df.to_json(filepath, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")

        return filepath