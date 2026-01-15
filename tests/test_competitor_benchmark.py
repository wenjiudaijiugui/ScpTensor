"""Test the competitor benchmark functionality."""

from scptensor.benchmark import (
    CompetitorBenchmarkSuite,
    SyntheticDataset,
)


def test_synthetic_dataset_generation() -> None:
    """Test synthetic dataset generation for benchmarking."""
    dataset = SyntheticDataset(
        n_samples=50,
        n_features=100,
        n_groups=2,
        n_batches=2,
        missing_rate=0.3,
        random_seed=42,
    ).generate()

    assert dataset.n_samples == 50
    assert dataset.assays["protein"].n_features == 100
    print("test_synthetic_dataset_generation PASSED")


def test_log_normalization_benchmark() -> None:
    """Test log normalization benchmark."""
    dataset = SyntheticDataset(
        n_samples=50,
        n_features=100,
        n_groups=2,
        n_batches=2,
        missing_rate=0.3,
        random_seed=42,
    ).generate()

    suite = CompetitorBenchmarkSuite()
    suite.datasets = [dataset]

    result = suite._run_single_benchmark(dataset, "log_normalization")

    assert result.operation == "log_normalization"
    assert result.speedup_factor > 0
    assert 0 <= result.accuracy_correlation <= 1.0
    print(f"test_log_normalization_benchmark PASSED (speedup={result.speedup_factor:.2f}x)")


def test_pca_benchmark() -> None:
    """Test PCA benchmark."""
    dataset = SyntheticDataset(
        n_samples=50,
        n_features=100,
        n_groups=2,
        n_batches=2,
        missing_rate=0.3,
        random_seed=42,
    ).generate()

    suite = CompetitorBenchmarkSuite()
    suite.datasets = [dataset]

    result = suite._run_single_benchmark(dataset, "pca")

    assert result.operation == "pca"
    assert result.speedup_factor > 0
    assert 0 <= result.accuracy_correlation <= 1.0
    print(f"test_pca_benchmark PASSED (speedup={result.speedup_factor:.2f}x)")


def test_knn_imputation_benchmark() -> None:
    """Test KNN imputation benchmark."""
    dataset = SyntheticDataset(
        n_samples=50,
        n_features=100,
        n_groups=2,
        n_batches=2,
        missing_rate=0.3,
        random_seed=42,
    ).generate()

    suite = CompetitorBenchmarkSuite()
    suite.datasets = [dataset]

    result = suite._run_single_benchmark(dataset, "knn_imputation")

    assert result.operation == "knn_imputation"
    assert result.speedup_factor > 0
    assert 0 <= result.accuracy_correlation <= 1.0
    print(f"test_knn_imputation_benchmark PASSED (speedup={result.speedup_factor:.2f}x)")


def test_benchmark_summary() -> None:
    """Test benchmark summary computation."""
    datasets = [
        SyntheticDataset(
            n_samples=50,
            n_features=100,
            n_groups=2,
            n_batches=2,
            missing_rate=0.3,
            random_seed=42 + i,
        ).generate()
        for i in range(2)
    ]

    suite = CompetitorBenchmarkSuite()
    suite.datasets = datasets

    # Run multiple benchmarks
    results = []
    for dataset in datasets:
        results.append(suite._run_single_benchmark(dataset, "log_normalization"))
        results.append(suite._run_single_benchmark(dataset, "pca"))

    assert len(results) == 4
    print(f"test_benchmark_summary PASSED ({len(results)} benchmark results)")


if __name__ == "__main__":
    print("Running competitor benchmark tests...")
    print()

    test_synthetic_dataset_generation()
    test_log_normalization_benchmark()
    test_pca_benchmark()
    test_knn_imputation_benchmark()
    test_benchmark_summary()

    print()
    print("All competitor benchmark tests PASSED!")
