"""Tests for scptensor.benchmark module.

This module contains comprehensive tests for benchmark core classes:
- TechnicalMetrics, BiologicalMetrics, ComputationalMetrics
- MethodRunResult
- BenchmarkResults
- MetricsEngine
- SyntheticDataset
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from scptensor.benchmark.core import (
    BenchmarkResults,
    BiologicalMetrics,
    ComputationalMetrics,
    MethodRunResult,
    TechnicalMetrics,
)
from scptensor.benchmark.metrics import MetricsEngine
from scptensor.benchmark.synthetic_data import SyntheticDataset, create_benchmark_datasets
from scptensor.core import Assay, ScpContainer, ScpMatrix

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_container():
    """Create a sample container for testing."""
    np.random.seed(42)
    n_samples = 50
    n_features = 100

    obs = pl.DataFrame(
        {
            "_index": [f"S{i:03d}" for i in range(n_samples)],
            "batch": [f"Batch{i % 3}" for i in range(n_samples)],
            "group": [f"Group{i % 2}" for i in range(n_samples)],
        }
    )

    var = pl.DataFrame(
        {
            "_index": [f"P{i:04d}" for i in range(n_features)],
            "protein_id": [f"P{i:04d}" for i in range(n_features)],
        }
    )

    # Create data with some missing values
    X = np.random.rand(n_samples, n_features) * 10
    M = np.zeros((n_samples, n_features), dtype=np.int8)

    # Add some missing values
    missing_indices = np.random.choice(
        n_samples * n_features, size=int(n_samples * n_features * 0.3), replace=False
    )
    rows = missing_indices // n_features
    cols = missing_indices % n_features
    M[rows, cols] = np.random.choice([1, 2], size=len(rows))  # MBR or LOD
    X[rows, cols] = 0

    matrix = ScpMatrix(X=X, M=M)
    assay = Assay(var=var, layers={"raw": matrix}, feature_id_col="protein_id")

    return ScpContainer(obs=obs, assays={"protein": assay}, sample_id_col="_index")


@pytest.fixture
def sample_container_processed(sample_container):
    """Create a container with processed data layer."""
    input_matrix = sample_container.assays["protein"].layers["raw"]
    X_processed = input_matrix.X * 1.1  # Simple transformation
    M_processed = input_matrix.M.copy() if input_matrix.M is not None else None

    processed_matrix = ScpMatrix(X=X_processed, M=M_processed)
    sample_container.assays["protein"].layers["processed"] = processed_matrix

    return sample_container


@pytest.fixture
def metrics_engine():
    """Create a MetricsEngine instance."""
    return MetricsEngine()


# =============================================================================
# Tests for TechnicalMetrics
# =============================================================================


class TestTechnicalMetrics:
    """Tests for TechnicalMetrics dataclass."""

    def test_technical_metrics_creation(self):
        """Test creating TechnicalMetrics with valid values."""
        metrics = TechnicalMetrics(
            data_recovery_rate=0.85,
            variance_preservation=0.92,
            sparsity_preservation=0.95,
            batch_mixing_score=0.75,
            signal_to_noise_ratio=3.5,
            missing_value_pattern_score=0.88,
        )
        assert metrics.data_recovery_rate == 0.85
        assert metrics.variance_preservation == 0.92
        assert metrics.sparsity_preservation == 0.95
        assert metrics.batch_mixing_score == 0.75
        assert metrics.signal_to_noise_ratio == 3.5
        assert metrics.missing_value_pattern_score == 0.88

    def test_technical_metrics_zero_values(self):
        """Test TechnicalMetrics with zero values."""
        metrics = TechnicalMetrics(
            data_recovery_rate=0.0,
            variance_preservation=0.0,
            sparsity_preservation=0.0,
            batch_mixing_score=0.0,
            signal_to_noise_ratio=0.0,
            missing_value_pattern_score=0.0,
        )
        assert metrics.data_recovery_rate == 0.0
        assert metrics.signal_to_noise_ratio == 0.0

    def test_technical_metrics_max_values(self):
        """Test TechnicalMetrics with maximum values."""
        metrics = TechnicalMetrics(
            data_recovery_rate=1.0,
            variance_preservation=1.0,
            sparsity_preservation=1.0,
            batch_mixing_score=1.0,
            signal_to_noise_ratio=100.0,
            missing_value_pattern_score=1.0,
        )
        assert metrics.data_recovery_rate == 1.0
        assert metrics.signal_to_noise_ratio == 100.0

    def test_technical_metrics_dict_conversion(self):
        """Test converting TechnicalMetrics to dict."""
        metrics = TechnicalMetrics(
            data_recovery_rate=0.85,
            variance_preservation=0.92,
            sparsity_preservation=0.95,
            batch_mixing_score=0.75,
            signal_to_noise_ratio=3.5,
            missing_value_pattern_score=0.88,
        )
        d = metrics.__dict__
        assert isinstance(d, dict)
        assert len(d) == 6
        assert d["data_recovery_rate"] == 0.85


# =============================================================================
# Tests for BiologicalMetrics
# =============================================================================


class TestBiologicalMetrics:
    """Tests for BiologicalMetrics dataclass."""

    def test_biological_metrics_creation(self):
        """Test creating BiologicalMetrics with valid values."""
        metrics = BiologicalMetrics(
            group_separation=0.75,
            biological_signal_preservation=0.88,
            clustering_consistency=0.92,
            biological_variance_explained=65.5,
            differential_expression_concordance=0.80,
        )
        assert metrics.group_separation == 0.75
        assert metrics.biological_signal_preservation == 0.88
        assert metrics.clustering_consistency == 0.92
        assert metrics.biological_variance_explained == 65.5
        assert metrics.differential_expression_concordance == 0.80

    def test_biological_metrics_without_de(self):
        """Test BiologicalMetrics without differential expression."""
        metrics = BiologicalMetrics(
            group_separation=0.75,
            biological_signal_preservation=0.88,
            clustering_consistency=0.92,
            biological_variance_explained=65.5,
        )
        assert metrics.differential_expression_concordance is None

    def test_biological_metrics_negative_values(self):
        """Test BiologicalMetrics with negative values."""
        metrics = BiologicalMetrics(
            group_separation=-0.1,
            biological_signal_preservation=0.5,
            clustering_consistency=0.0,
            biological_variance_explained=10.0,
        )
        assert metrics.group_separation == -0.1

    def test_biological_metrics_high_separation(self):
        """Test BiologicalMetrics with high separation score."""
        metrics = BiologicalMetrics(
            group_separation=0.95,
            biological_signal_preservation=0.98,
            clustering_consistency=1.0,
            biological_variance_explained=85.0,
        )
        assert metrics.group_separation == 0.95
        assert metrics.clustering_consistency == 1.0


# =============================================================================
# Tests for ComputationalMetrics
# =============================================================================


class TestComputationalMetrics:
    """Tests for ComputationalMetrics dataclass."""

    def test_computational_metrics_creation(self):
        """Test creating ComputationalMetrics with valid values."""
        metrics = ComputationalMetrics(
            runtime_seconds=45.2,
            memory_usage_mb=256.5,
            scalability_factor=1.2,
            convergence_iterations=100,
            cpu_utilization_percent=85.0,
        )
        assert metrics.runtime_seconds == 45.2
        assert metrics.memory_usage_mb == 256.5
        assert metrics.scalability_factor == 1.2
        assert metrics.convergence_iterations == 100
        assert metrics.cpu_utilization_percent == 85.0

    def test_computational_metrics_minimal(self):
        """Test ComputationalMetrics with only required fields."""
        metrics = ComputationalMetrics(
            runtime_seconds=10.0,
            memory_usage_mb=128.0,
            scalability_factor=1.0,
        )
        assert metrics.convergence_iterations is None
        assert metrics.cpu_utilization_percent is None

    def test_computational_metrics_zero_runtime(self):
        """Test ComputationalMetrics with near-zero runtime."""
        metrics = ComputationalMetrics(
            runtime_seconds=0.001,
            memory_usage_mb=64.0,
            scalability_factor=1.0,
        )
        assert metrics.runtime_seconds == 0.001


# =============================================================================
# Tests for MethodRunResult
# =============================================================================


class TestMethodRunResult:
    """Tests for MethodRunResult class."""

    @pytest.fixture
    def sample_result(self, sample_container):
        """Create a sample MethodRunResult."""
        technical = TechnicalMetrics(
            data_recovery_rate=0.85,
            variance_preservation=0.92,
            sparsity_preservation=0.95,
            batch_mixing_score=0.75,
            signal_to_noise_ratio=3.5,
            missing_value_pattern_score=0.88,
        )
        biological = BiologicalMetrics(
            group_separation=0.75,
            biological_signal_preservation=0.88,
            clustering_consistency=0.92,
            biological_variance_explained=65.5,
        )
        computational = ComputationalMetrics(
            runtime_seconds=45.2,
            memory_usage_mb=256.5,
            scalability_factor=1.0,
        )

        return MethodRunResult(
            method_name="test_method",
            parameters={"param1": 1.0, "param2": "value"},
            dataset_name="test_dataset",
            input_container=sample_container,
            output_container=sample_container,
            runtime_seconds=45.2,
            memory_usage_mb=256.5,
            technical_scores=technical,
            biological_scores=biological,
            computational_scores=computational,
            random_seed=42,
        )

    def test_method_run_result_creation(self, sample_result):
        """Test creating MethodRunResult."""
        assert sample_result.method_name == "test_method"
        assert sample_result.dataset_name == "test_dataset"
        assert sample_result.random_seed == 42
        assert len(sample_result.parameters) == 2

    def test_method_run_result_to_dict(self, sample_result):
        """Test to_dict method."""
        d = sample_result.to_dict()
        assert isinstance(d, dict)
        assert d["method_name"] == "test_method"
        assert "technical_scores" in d
        assert "biological_scores" in d
        assert "computational_scores" in d

    def test_method_run_result_without_biological(self, sample_container):
        """Test MethodRunResult without biological scores."""
        technical = TechnicalMetrics(
            data_recovery_rate=0.85,
            variance_preservation=0.92,
            sparsity_preservation=0.95,
            batch_mixing_score=0.75,
            signal_to_noise_ratio=3.5,
            missing_value_pattern_score=0.88,
        )
        computational = ComputationalMetrics(
            runtime_seconds=45.2,
            memory_usage_mb=256.5,
            scalability_factor=1.0,
        )

        result = MethodRunResult(
            method_name="test_method",
            parameters={},
            dataset_name="test_dataset",
            input_container=sample_container,
            output_container=sample_container,
            runtime_seconds=45.2,
            memory_usage_mb=256.5,
            technical_scores=technical,
            biological_scores=None,
            computational_scores=computational,
            random_seed=42,
        )
        assert result.biological_scores is None

    def test_method_run_result_timestamp(self, sample_result):
        """Test that timestamp is generated."""
        assert sample_result.timestamp is not None
        assert isinstance(sample_result.timestamp, str)

    def test_method_run_result_software_versions(self, sample_result):
        """Test software_versions dict."""
        assert isinstance(sample_result.software_versions, dict)
        # Can be empty by default
        assert len(sample_result.software_versions) == 0


# =============================================================================
# Tests for BenchmarkResults
# =============================================================================


class TestBenchmarkResults:
    """Tests for BenchmarkResults class."""

    def test_benchmark_results_init(self):
        """Test BenchmarkResults initialization."""
        results = BenchmarkResults()
        assert results.runs == []
        assert results.datasets == {}
        assert results.metadata == {}

    def test_benchmark_results_add_run(self, sample_container):
        """Test adding a run result."""
        results = BenchmarkResults()

        technical = TechnicalMetrics(
            data_recovery_rate=0.85,
            variance_preservation=0.92,
            sparsity_preservation=0.95,
            batch_mixing_score=0.75,
            signal_to_noise_ratio=3.5,
            missing_value_pattern_score=0.88,
        )
        biological = BiologicalMetrics(
            group_separation=0.75,
            biological_signal_preservation=0.88,
            clustering_consistency=0.92,
            biological_variance_explained=65.5,
        )
        computational = ComputationalMetrics(
            runtime_seconds=45.2,
            memory_usage_mb=256.5,
            scalability_factor=1.0,
        )

        result = MethodRunResult(
            method_name="test_method",
            parameters={"param1": 1.0},
            dataset_name="test_dataset",
            input_container=sample_container,
            output_container=sample_container,
            runtime_seconds=45.2,
            memory_usage_mb=256.5,
            technical_scores=technical,
            biological_scores=biological,
            computational_scores=computational,
            random_seed=42,
        )

        results.add_run(result)
        assert len(results.runs) == 1
        assert results.runs[0].method_name == "test_method"

    def test_benchmark_results_add_dataset(self, sample_container):
        """Test adding a dataset."""
        results = BenchmarkResults()
        results.add_dataset("test_dataset", sample_container)
        assert "test_dataset" in results.datasets
        assert results.datasets["test_dataset"] is sample_container

    def test_benchmark_results_filter_by_method(self, sample_container):
        """Test filtering results by method name."""
        results = BenchmarkResults()

        for i in range(3):
            technical = TechnicalMetrics(
                data_recovery_rate=0.8 + i * 0.05,
                variance_preservation=0.9,
                sparsity_preservation=0.95,
                batch_mixing_score=0.75,
                signal_to_noise_ratio=3.5,
                missing_value_pattern_score=0.88,
            )
            biological = BiologicalMetrics(
                group_separation=0.75,
                biological_signal_preservation=0.88,
                clustering_consistency=0.92,
                biological_variance_explained=65.5,
            )
            computational = ComputationalMetrics(
                runtime_seconds=45.2,
                memory_usage_mb=256.5,
                scalability_factor=1.0,
            )

            result = MethodRunResult(
                method_name=f"method_{i % 2}",
                parameters={"param": i},
                dataset_name="test_dataset",
                input_container=sample_container,
                output_container=sample_container,
                runtime_seconds=45.2,
                memory_usage_mb=256.5,
                technical_scores=technical,
                biological_scores=biological,
                computational_scores=computational,
                random_seed=42,
            )
            results.add_run(result)

        method_0_runs = results.filter_by_method("method_0")
        assert len(method_0_runs) == 2

    def test_benchmark_results_filter_by_dataset(self, sample_container):
        """Test filtering results by dataset name."""
        results = BenchmarkResults()

        technical = TechnicalMetrics(
            data_recovery_rate=0.85,
            variance_preservation=0.92,
            sparsity_preservation=0.95,
            batch_mixing_score=0.75,
            signal_to_noise_ratio=3.5,
            missing_value_pattern_score=0.88,
        )
        biological = BiologicalMetrics(
            group_separation=0.75,
            biological_signal_preservation=0.88,
            clustering_consistency=0.92,
            biological_variance_explained=65.5,
        )
        computational = ComputationalMetrics(
            runtime_seconds=45.2,
            memory_usage_mb=256.5,
            scalability_factor=1.0,
        )

        result1 = MethodRunResult(
            method_name="test_method",
            parameters={},
            dataset_name="dataset1",
            input_container=sample_container,
            output_container=sample_container,
            runtime_seconds=45.2,
            memory_usage_mb=256.5,
            technical_scores=technical,
            biological_scores=biological,
            computational_scores=computational,
            random_seed=42,
        )

        result2 = MethodRunResult(
            method_name="test_method",
            parameters={},
            dataset_name="dataset2",
            input_container=sample_container,
            output_container=sample_container,
            runtime_seconds=45.2,
            memory_usage_mb=256.5,
            technical_scores=technical,
            biological_scores=biological,
            computational_scores=computational,
            random_seed=42,
        )

        results.add_run(result1)
        results.add_run(result2)

        dataset1_runs = results.filter_by_dataset("dataset1")
        assert len(dataset1_runs) == 1
        assert dataset1_runs[0].dataset_name == "dataset1"

    def test_benchmark_results_get_methods(self, sample_container):
        """Test getting list of unique methods."""
        results = BenchmarkResults()

        for method_name in ["method1", "method2", "method1"]:
            technical = TechnicalMetrics(
                data_recovery_rate=0.85,
                variance_preservation=0.92,
                sparsity_preservation=0.95,
                batch_mixing_score=0.75,
                signal_to_noise_ratio=3.5,
                missing_value_pattern_score=0.88,
            )
            biological = BiologicalMetrics(
                group_separation=0.75,
                biological_signal_preservation=0.88,
                clustering_consistency=0.92,
                biological_variance_explained=65.5,
            )
            computational = ComputationalMetrics(
                runtime_seconds=45.2,
                memory_usage_mb=256.5,
                scalability_factor=1.0,
            )

            result = MethodRunResult(
                method_name=method_name,
                parameters={},
                dataset_name="test_dataset",
                input_container=sample_container,
                output_container=sample_container,
                runtime_seconds=45.2,
                memory_usage_mb=256.5,
                technical_scores=technical,
                biological_scores=biological,
                computational_scores=computational,
                random_seed=42,
            )
            results.add_run(result)

        methods = results.get_methods()
        assert set(methods) == {"method1", "method2"}

    def test_benchmark_results_get_datasets(self, sample_container):
        """Test getting list of unique datasets."""
        results = BenchmarkResults()

        for dataset_name in ["dataset1", "dataset2", "dataset1"]:
            technical = TechnicalMetrics(
                data_recovery_rate=0.85,
                variance_preservation=0.92,
                sparsity_preservation=0.95,
                batch_mixing_score=0.75,
                signal_to_noise_ratio=3.5,
                missing_value_pattern_score=0.88,
            )
            biological = BiologicalMetrics(
                group_separation=0.75,
                biological_signal_preservation=0.88,
                clustering_consistency=0.92,
                biological_variance_explained=65.5,
            )
            computational = ComputationalMetrics(
                runtime_seconds=45.2,
                memory_usage_mb=256.5,
                scalability_factor=1.0,
            )

            result = MethodRunResult(
                method_name="test_method",
                parameters={},
                dataset_name=dataset_name,
                input_container=sample_container,
                output_container=sample_container,
                runtime_seconds=45.2,
                memory_usage_mb=256.5,
                technical_scores=technical,
                biological_scores=biological,
                computational_scores=computational,
                random_seed=42,
            )
            results.add_run(result)

        datasets = results.get_datasets()
        assert set(datasets) == {"dataset1", "dataset2"}

    def test_benchmark_results_to_dataframe(self, sample_container):
        """Test converting results to DataFrame."""
        results = BenchmarkResults()

        technical = TechnicalMetrics(
            data_recovery_rate=0.85,
            variance_preservation=0.92,
            sparsity_preservation=0.95,
            batch_mixing_score=0.75,
            signal_to_noise_ratio=3.5,
            missing_value_pattern_score=0.88,
        )
        biological = BiologicalMetrics(
            group_separation=0.75,
            biological_signal_preservation=0.88,
            clustering_consistency=0.92,
            biological_variance_explained=65.5,
        )
        computational = ComputationalMetrics(
            runtime_seconds=45.2,
            memory_usage_mb=256.5,
            scalability_factor=1.0,
        )

        result = MethodRunResult(
            method_name="test_method",
            parameters={"param1": 1.0},
            dataset_name="test_dataset",
            input_container=sample_container,
            output_container=sample_container,
            runtime_seconds=45.2,
            memory_usage_mb=256.5,
            technical_scores=technical,
            biological_scores=biological,
            computational_scores=computational,
            random_seed=42,
        )

        results.add_run(result)
        df = results.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "method_name" in df.columns
        assert "technical_data_recovery_rate" in df.columns

    def test_benchmark_results_export_csv(self, sample_container, tmp_path):
        """Test exporting results to CSV."""
        results = BenchmarkResults()

        technical = TechnicalMetrics(
            data_recovery_rate=0.85,
            variance_preservation=0.92,
            sparsity_preservation=0.95,
            batch_mixing_score=0.75,
            signal_to_noise_ratio=3.5,
            missing_value_pattern_score=0.88,
        )
        biological = BiologicalMetrics(
            group_separation=0.75,
            biological_signal_preservation=0.88,
            clustering_consistency=0.92,
            biological_variance_explained=65.5,
        )
        computational = ComputationalMetrics(
            runtime_seconds=45.2,
            memory_usage_mb=256.5,
            scalability_factor=1.0,
        )

        result = MethodRunResult(
            method_name="test_method",
            parameters={"param1": 1.0},
            dataset_name="test_dataset",
            input_container=sample_container,
            output_container=sample_container,
            runtime_seconds=45.2,
            memory_usage_mb=256.5,
            technical_scores=technical,
            biological_scores=biological,
            computational_scores=computational,
            random_seed=42,
        )

        results.add_run(result)

        filepath = results.export_data(format="csv", filepath=str(tmp_path / "results.csv"))
        assert filepath.endswith("results.csv")

        # Verify file was created
        import os

        assert os.path.exists(filepath)

    def test_benchmark_results_get_parameter_sensitivity(self, sample_container):
        """Test getting parameter sensitivity."""
        results = BenchmarkResults()

        for param_val in [0.1, 0.5, 1.0]:
            technical = TechnicalMetrics(
                data_recovery_rate=0.8 + param_val * 0.1,
                variance_preservation=0.92,
                sparsity_preservation=0.95,
                batch_mixing_score=0.75,
                signal_to_noise_ratio=3.5,
                missing_value_pattern_score=0.88,
            )
            biological = BiologicalMetrics(
                group_separation=0.75,
                biological_signal_preservation=0.88,
                clustering_consistency=0.92,
                biological_variance_explained=65.5,
            )
            computational = ComputationalMetrics(
                runtime_seconds=45.2,
                memory_usage_mb=256.5,
                scalability_factor=1.0,
            )

            result = MethodRunResult(
                method_name="test_method",
                parameters={"alpha": param_val},
                dataset_name="test_dataset",
                input_container=sample_container,
                output_container=sample_container,
                runtime_seconds=45.2,
                memory_usage_mb=256.5,
                technical_scores=technical,
                biological_scores=biological,
                computational_scores=computational,
                random_seed=42,
            )
            results.add_run(result)

        df = results.get_parameter_sensitivity("test_method")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "alpha" in df.columns


# =============================================================================
# Tests for MetricsEngine
# =============================================================================


class TestMetricsEngine:
    """Tests for MetricsEngine class."""

    def test_metrics_engine_init(self, metrics_engine):
        """Test MetricsEngine initialization."""
        assert metrics_engine.start_time is None
        assert metrics_engine.start_memory is None

    def test_start_timing(self, metrics_engine):
        """Test starting timing."""
        metrics_engine.start_timing()
        assert metrics_engine.start_time is not None
        assert metrics_engine.start_memory is not None

    def test_stop_timing(self, metrics_engine):
        """Test stopping timing."""
        metrics_engine.start_timing()
        runtime, memory = metrics_engine.stop_timing()
        assert runtime >= 0
        assert memory >= 0
        assert metrics_engine.start_time is None

    def test_stop_timing_without_start(self, metrics_engine):
        """Test stopping timing without starting raises error."""
        with pytest.raises(RuntimeError, match="Timing not started"):
            metrics_engine.stop_timing()

    def test_timing_round_trip(self, metrics_engine):
        """Test timing a short operation."""
        import time

        metrics_engine.start_timing()
        time.sleep(0.01)  # 10ms
        runtime, memory = metrics_engine.stop_timing()

        assert runtime >= 0.01
        assert runtime < 1.0  # Should not take too long

    def test_evaluate_technical(self, metrics_engine, sample_container, sample_container_processed):
        """Test evaluating technical metrics."""
        metrics = metrics_engine.evaluate_technical(
            input_container=sample_container,
            output_container=sample_container_processed,
            assay_name="protein",
        )

        assert isinstance(metrics, TechnicalMetrics)
        assert 0 <= metrics.data_recovery_rate <= 1
        assert 0 <= metrics.variance_preservation <= 1
        assert 0 <= metrics.sparsity_preservation <= 1
        assert 0 <= metrics.batch_mixing_score <= 1
        assert metrics.signal_to_noise_ratio >= 0

    def test_evaluate_technical_different_assay(self, metrics_engine, sample_container):
        """Test evaluating technical metrics with default assay."""
        # Add processed layer
        input_matrix = sample_container.assays["protein"].layers["raw"]
        X_processed = input_matrix.X * 1.1
        processed_matrix = ScpMatrix(X=X_processed, M=input_matrix.M)
        sample_container.assays["protein"].layers["processed"] = processed_matrix

        metrics = metrics_engine.evaluate_technical(
            input_container=sample_container,
            output_container=sample_container,
        )

        assert isinstance(metrics, TechnicalMetrics)

    def test_evaluate_biological(self, metrics_engine, sample_container_processed):
        """Test evaluating biological metrics."""
        metrics = metrics_engine.evaluate_biological(
            output_container=sample_container_processed,
            assay_name="protein",
        )

        assert isinstance(metrics, BiologicalMetrics)
        assert -1 <= metrics.group_separation <= 1
        assert 0 <= metrics.biological_signal_preservation <= 1
        assert -1 <= metrics.clustering_consistency <= 1
        assert 0 <= metrics.biological_variance_explained <= 100

    def test_evaluate_biological_with_groups(self, metrics_engine, sample_container_processed):
        """Test evaluating biological metrics with explicit groups."""
        groups = np.array([0, 1, 0, 1, 0] * 10)  # 50 samples

        metrics = metrics_engine.evaluate_biological(
            output_container=sample_container_processed,
            ground_truth_groups=groups,
            assay_name="protein",
        )

        assert isinstance(metrics, BiologicalMetrics)
        # With groups provided, should have valid separation
        assert -1 <= metrics.group_separation <= 1

    def test_evaluate_computational(self, metrics_engine):
        """Test evaluating computational metrics."""
        metrics = metrics_engine.evaluate_computational(
            runtime_seconds=45.2,
            memory_usage_mb=256.5,
        )

        assert isinstance(metrics, ComputationalMetrics)
        assert metrics.runtime_seconds == 45.2
        assert metrics.memory_usage_mb == 256.5
        assert metrics.scalability_factor == 1.0

    def test_get_mask_statistics_helper(self, metrics_engine, sample_container):
        """Test _compute_data_recovery helper."""
        input_matrix = sample_container.assays["protein"].layers["raw"]

        # Test that missing values are detected
        from scptensor.core import MatrixOps

        stats = MatrixOps.get_mask_statistics(input_matrix)
        assert isinstance(stats, dict)
        assert "VALID" in stats
        assert "MBR" in stats or "LOD" in stats


# =============================================================================
# Tests for SyntheticDataset
# =============================================================================


class TestSyntheticDataset:
    """Tests for SyntheticDataset class."""

    def test_synthetic_dataset_init(self):
        """Test SyntheticDataset initialization."""
        dataset = SyntheticDataset(
            n_samples=100,
            n_features=500,
            n_groups=3,
            n_batches=2,
            missing_rate=0.3,
            batch_effect_strength=0.2,
            group_effect_strength=0.5,
            signal_to_noise_ratio=2.0,
            random_seed=42,
        )

        assert dataset.n_samples == 100
        assert dataset.n_features == 500
        assert dataset.n_groups == 3
        assert dataset.n_batches == 2
        assert dataset.missing_rate == 0.3

    def test_synthetic_dataset_generate(self):
        """Test generating synthetic dataset."""
        dataset = SyntheticDataset(
            n_samples=50,
            n_features=100,
            n_groups=2,
            n_batches=2,
            missing_rate=0.3,
            random_seed=42,
        )

        container = dataset.generate()

        assert isinstance(container, ScpContainer)
        assert container.n_samples == 50
        assert "protein" in container.assays
        assert len(container.assays["protein"].var) == 100

    def test_synthetic_dataset_obs_columns(self):
        """Test that obs has required columns."""
        dataset = SyntheticDataset(
            n_samples=50,
            n_features=100,
            random_seed=42,
        )

        container = dataset.generate()

        assert "sample_id" in container.obs.columns
        assert "group" in container.obs.columns
        assert "batch" in container.obs.columns

    def test_synthetic_dataset_var_columns(self):
        """Test that var has required columns."""
        dataset = SyntheticDataset(
            n_samples=50,
            n_features=100,
            random_seed=42,
        )

        container = dataset.generate()
        var = container.assays["protein"].var

        assert "protein_id" in var.columns
        assert "protein_class" in var.columns

    def test_synthetic_dataset_missing_values(self):
        """Test that missing values are introduced."""
        dataset = SyntheticDataset(
            n_samples=50,
            n_features=100,
            missing_rate=0.3,
            random_seed=42,
        )

        container = dataset.generate()
        matrix = container.assays["protein"].layers["raw"]

        if matrix.M is not None:
            missing_count = np.sum(matrix.M != 0)
            total_count = matrix.M.size
            actual_rate = missing_count / total_count
            # Should be close to target rate (within tolerance)
            assert abs(actual_rate - 0.3) < 0.15

    def test_synthetic_dataset_groups(self):
        """Test that groups are generated correctly."""
        dataset = SyntheticDataset(
            n_samples=50,
            n_features=100,
            n_groups=3,
            random_seed=42,
        )

        container = dataset.generate()
        unique_groups = set(container.obs["group"].to_list())

        assert len(unique_groups) == 3

    def test_synthetic_dataset_batches(self):
        """Test that batches are generated correctly."""
        dataset = SyntheticDataset(
            n_samples=50,
            n_features=100,
            n_batches=2,
            random_seed=42,
        )

        container = dataset.generate()
        unique_batches = set(container.obs["batch"].to_list())

        assert len(unique_batches) == 2

    def test_synthetic_dataset_reproducibility(self):
        """Test that same seed produces identical results."""
        dataset1 = SyntheticDataset(n_samples=50, n_features=100, random_seed=42)
        dataset2 = SyntheticDataset(n_samples=50, n_features=100, random_seed=42)

        container1 = dataset1.generate()
        container2 = dataset2.generate()

        X1 = container1.assays["protein"].layers["raw"].X
        X2 = container2.assays["protein"].layers["raw"].X

        np.testing.assert_array_equal(X1, X2)

    def test_synthetic_dataset_get_ground_truth(self):
        """Test getting ground truth labels."""
        dataset = SyntheticDataset(
            n_samples=50,
            n_features=100,
            n_groups=2,
            n_batches=2,
            random_seed=42,
        )

        ground_truth = dataset.get_ground_truth()

        assert "groups" in ground_truth
        assert "batches" in ground_truth
        assert "group_labels" in ground_truth
        assert "batch_labels" in ground_truth
        assert len(ground_truth["groups"]) == 50

    def test_synthetic_dataset_zero_missing_rate(self):
        """Test with zero missing rate."""
        dataset = SyntheticDataset(
            n_samples=50,
            n_features=100,
            missing_rate=0.0,
            random_seed=42,
        )

        container = dataset.generate()
        matrix = container.assays["protein"].layers["raw"]

        if matrix.M is not None:
            assert np.all(matrix.M == 0)


# =============================================================================
# Tests for create_benchmark_datasets
# =============================================================================


class TestCreateBenchmarkDatasets:
    """Tests for create_benchmark_datasets function."""

    def test_create_benchmark_datasets_returns_list(self):
        """Test that function returns list of datasets."""
        datasets = create_benchmark_datasets()

        assert isinstance(datasets, list)
        assert len(datasets) == 4

    def test_create_benchmark_datasets_containers(self):
        """Test that all items are ScpContainers."""
        datasets = create_benchmark_datasets()

        for dataset in datasets:
            assert isinstance(dataset, ScpContainer)

    def test_create_benchmark_datasets_sizes(self):
        """Test that datasets have expected sizes."""
        datasets = create_benchmark_datasets()

        # Dataset 1: Small, high signal
        assert datasets[0].n_samples == 50
        assert datasets[0].assays["protein"].n_features == 200

        # Dataset 2: Medium size
        assert datasets[1].n_samples == 100
        assert datasets[1].assays["protein"].n_features == 500

        # Dataset 3: Large
        assert datasets[2].n_samples == 200
        assert datasets[2].assays["protein"].n_features == 1000

    def test_create_benchmark_datasets_structure(self):
        """Test that all datasets have required structure."""
        datasets = create_benchmark_datasets()

        for dataset in datasets:
            assert "protein" in dataset.assays
            assert "raw" in dataset.assays["protein"].layers
            assert "sample_id" in dataset.obs.columns
            assert "group" in dataset.obs.columns
            assert "batch" in dataset.obs.columns


# =============================================================================
# Run tests if executed directly
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
