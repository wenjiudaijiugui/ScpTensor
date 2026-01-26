"""Streamlined integration tests for comparison study modules."""

from pathlib import Path

import numpy as np

from studies.comparison_study.comparison_engine import compare_pipelines
from studies.comparison_study.data_generation import generate_small_dataset, generate_synthetic_data
from studies.comparison_study.metrics import (
    calculate_all_metrics,
    calculate_asw,
    calculate_clisi,
    calculate_ilisi,
    calculate_kbet,
)
from studies.comparison_study.plotting import plot_batch_effects


def test_data_generation():
    """Test data generation."""
    container = generate_small_dataset(seed=42)

    assert container is not None
    assert container.n_samples == 1000
    assert container.n_features == 1000
    assert "proteins" in container.assays
    assert "raw" in container.assays["proteins"].layers
    assert "batch" in container.obs.columns
    assert "cell_type" in container.obs.columns


def test_custom_data_generation():
    """Test custom data generation parameters."""
    container = generate_synthetic_data(
        n_samples=500,
        n_features=200,
        n_batches=3,
        missing_rate=0.4,
        batch_effect=1.0,
        n_cell_types=4,
        random_seed=123,
    )

    assert container.n_samples == 500
    assert container.n_features == 200
    assert len(container.obs["batch"].unique()) == 3
    assert len(container.obs["cell_type"].unique()) == 4


def test_kbet_calculation():
    """Test kBET metric calculation."""
    container = generate_small_dataset(seed=42)
    X = container.assays["proteins"].layers["raw"].X
    batch_labels = container.obs["batch"].to_numpy()

    # Add batch effects for testing
    X_with_batch = X.copy()
    batch_shifts = np.random.randn(2, X.shape[1]) * 2.0
    for i in range(len(batch_labels)):
        X_with_batch[i] += batch_shifts[batch_labels[i]]

    kbet_score = calculate_kbet(X_with_batch, batch_labels, k=25)

    assert isinstance(kbet_score, float)
    assert 0.0 <= kbet_score <= 1.0


def test_ilisi_calculation():
    """Test iLISI metric calculation."""
    container = generate_small_dataset(seed=42)
    X = container.assays["proteins"].layers["raw"].X
    batch_labels = container.obs["batch"].to_numpy()

    ilisi_score = calculate_ilisi(X, batch_labels, k=20)

    assert isinstance(ilisi_score, float)
    assert ilisi_score >= 0.0


def test_clisi_calculation():
    """Test cLISI metric calculation."""
    container = generate_small_dataset(seed=42)
    X = container.assays["proteins"].layers["raw"].X
    cell_labels = container.obs["cell_type"].to_numpy()

    clisi_score = calculate_clisi(X, cell_labels, k=20)

    assert isinstance(clisi_score, float)
    assert clisi_score >= 0.0


def test_asw_calculation():
    """Test ASW metric calculation."""
    container = generate_small_dataset(seed=42)
    X = container.assays["proteins"].layers["raw"].X
    cell_labels = container.obs["cell_type"].to_numpy()

    asw_score = calculate_asw(X, cell_labels)

    assert isinstance(asw_score, float)
    assert -1.0 <= asw_score <= 1.0


def test_calculate_all_metrics():
    """Test comprehensive metrics calculation."""
    container = generate_small_dataset(seed=42)
    X = container.assays["proteins"].layers["raw"].X
    batch_labels = container.obs["batch"].to_numpy()
    cell_labels = container.obs["cell_type"].to_numpy()

    metrics = calculate_all_metrics(X, batch_labels, cell_labels, k=25)

    assert isinstance(metrics, dict)
    assert "kbet" in metrics
    assert "ilisi" in metrics
    assert "clisi" in metrics
    assert "asw" in metrics

    assert all(isinstance(v, float) for v in metrics.values())


def test_visualization(tmp_path: Path):
    """Test visualization generation."""
    container = generate_small_dataset(seed=42)
    X = container.assays["proteins"].layers["raw"].X
    batch_labels = container.obs["batch"].to_numpy()
    cell_labels = container.obs["cell_type"].to_numpy()

    # Calculate metrics
    metrics = calculate_all_metrics(X, batch_labels, cell_labels, k=25)

    # Create results dict
    results_dict = {
        "scptensor": {
            "kbet_score": metrics["kbet"],
            "ilisi_score": metrics["ilisi"],
            "clisi_score": metrics["clisi"],
            "asw_score": metrics["asw"],
        }
    }

    # Test plotting
    output_path = tmp_path / "test_batch_effects.png"
    result_path = plot_batch_effects(results_dict, output_path=output_path)

    assert result_path.exists()
    assert result_path == output_path


def test_end_to_end_comparison(tmp_path: Path):
    """Test end-to-end pipeline comparison."""
    # Generate dataset
    container = generate_small_dataset(seed=42)

    datasets = {"small": container}

    # Define dummy pipelines
    def dummy_pipeline1(data):
        return data

    def dummy_pipeline2(data):
        return data

    pipelines = {
        "pipeline1": dummy_pipeline1,
        "pipeline2": dummy_pipeline2,
    }

    # Run comparison
    results = compare_pipelines(pipelines, datasets)

    assert isinstance(results, dict)
    assert "pipeline1" in results
    assert "pipeline2" in results
    assert "small" in results["pipeline1"]
    assert "small" in results["pipeline2"]

    # Check structure
    for pipeline_result in results.values():
        for dataset_result in pipeline_result.values():
            assert "scores" in dataset_result
            assert "runtime" in dataset_result
            assert isinstance(dataset_result["scores"], dict)
            assert isinstance(dataset_result["runtime"], float)


def test_metrics_edge_cases():
    """Test metrics with edge case inputs."""
    # Single batch
    X_single = np.random.randn(50, 20)
    batch_single = np.zeros(50, dtype=int)

    kbet_single = calculate_kbet(X_single, batch_single, k=10)
    assert kbet_single == 0.0  # No batch mixing with single batch

    # Small dataset
    X_small = np.random.randn(10, 5)
    batch_small = np.random.randint(0, 2, 10)

    kbet_small = calculate_kbet(X_small, batch_small, k=20)
    assert isinstance(kbet_small, float)


def test_data_generation_reproducibility():
    """Test data generation reproducibility with same seed."""
    container1 = generate_small_dataset(seed=42)
    container2 = generate_small_dataset(seed=42)

    X1 = container1.assays["proteins"].layers["raw"].X
    X2 = container2.assays["proteins"].layers["raw"].X

    np.testing.assert_array_equal(X1, X2)

    batch1 = container1.obs["batch"].to_numpy()
    batch2 = container2.obs["batch"].to_numpy()

    np.testing.assert_array_equal(batch1, batch2)
