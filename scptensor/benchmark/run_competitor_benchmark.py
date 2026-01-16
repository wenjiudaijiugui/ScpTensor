#!/usr/bin/env python3
"""Comprehensive competitor benchmark runner for ScpTensor.

This script runs benchmarks comparing ScpTensor against common single-cell
analysis tools including scikit-learn, scipy, and numpy implementations.

Key areas benchmarked:
- Imputation algorithms (KNN, SVD)
- Dimensionality reduction (PCA)
- Clustering (KMeans)
- Normalization methods (Log, Z-score)

Usage:
    python -m scptensor.benchmark.run_competitor_benchmark
    python -m scptensor.benchmark.run_competitor_benchmark --quick
    python -m scptensor.benchmark.run_competitor_benchmark --output-dir results
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


# Import benchmark classes (deferred to avoid circular imports)
def _get_classes():
    """Import benchmark classes at runtime to avoid circular imports."""
    from scptensor.benchmark.competitor_suite import ComparisonResult, CompetitorBenchmarkSuite
    from scptensor.benchmark.competitor_viz import CompetitorResultVisualizer
    from scptensor.benchmark.synthetic_data import SyntheticDataset

    return CompetitorBenchmarkSuite, ComparisonResult, CompetitorResultVisualizer, SyntheticDataset


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run competitor benchmarks for ScpTensor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full benchmark suite
  python -m scptensor.benchmark.run_competitor_benchmark

  # Run quick benchmark (smaller datasets)
  python -m scptensor.benchmark.run_competitor_benchmark --quick

  # Specify custom output directory
  python -m scptensor.benchmark.run_competitor_benchmark --output-dir my_results

  # Run only specific operations
  python -m scptensor.benchmark.run_competitor_benchmark --operations pca knn_imputation
        """,
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark with smaller datasets",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="competitor_benchmark_results",
        help="Output directory for results (default: competitor_benchmark_results)",
    )

    parser.add_argument(
        "--operations",
        type=str,
        nargs="+",
        choices=[
            "log_normalization",
            "zscore_normalization",
            "knn_imputation",
            "svd_imputation",
            "pca",
            "kmeans_clustering",
        ],
        default=None,
        help="Specific operations to benchmark (default: all)",
    )

    parser.add_argument(
        "--dataset-sizes",
        type=int,
        nargs=2,
        default=None,
        metavar=("N_SAMPLES", "N_FEATURES"),
        help="Custom dataset size (e.g., --dataset-sizes 100 500)",
    )

    parser.add_argument(
        "--n-repeats",
        type=int,
        default=1,
        help="Number of times to repeat each benchmark (default: 1)",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating visualization plots",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    return parser.parse_args()


def create_datasets(
    quick: bool = False,
    custom_size: tuple[int, int] | None = None,
) -> list:
    """Create benchmark datasets.

    Parameters
    ----------
    quick : bool
        If True, create smaller datasets for quick testing.
    custom_size : tuple[int, int] | None
        Custom dataset size (n_samples, n_features).

    Returns
    -------
    list
        List of ScpContainer datasets.
    """
    SyntheticDataset = _get_classes()[3]

    if custom_size:
        n_samples, n_features = custom_size
        return [
            SyntheticDataset(
                n_samples=n_samples,
                n_features=n_features,
                n_groups=2,
                n_batches=2,
                missing_rate=0.3,
                random_seed=42,
            ).generate()
        ]

    if quick:
        return [
            SyntheticDataset(
                n_samples=50,
                n_features=200,
                n_groups=2,
                n_batches=2,
                missing_rate=0.3,
                random_seed=42,
            ).generate()
        ]

    # Standard benchmark sizes
    configs = [
        (50, 200, 2, 2, 0.3, 42),  # Small
        (100, 500, 3, 3, 0.3, 123),  # Medium
        (200, 1000, 4, 4, 0.35, 456),  # Large
    ]

    datasets = []
    for n_samples, n_features, n_groups, n_batches, missing_rate, seed in configs:
        datasets.append(
            SyntheticDataset(
                n_samples=n_samples,
                n_features=n_features,
                n_groups=n_groups,
                n_batches=n_batches,
                missing_rate=missing_rate,
                random_seed=seed,
            ).generate()
        )

    return datasets


def print_header(text: str, width: int = 70) -> None:
    """Print a formatted header.

    Parameters
    ----------
    text : str
        Header text.
    width : int
        Total width.
    """
    padding = (width - len(text) - 2) // 2
    print("=" * width)
    print(" " * padding + text)
    print("=" * width)


def print_operation_header(operation: str) -> None:
    """Print a header for an operation.

    Parameters
    ----------
    operation : str
        Operation name.
    """
    print(f"\n{'-' * 60}")
    print(f"BENCHMARKING: {operation}")
    print(f"{'-' * 60}")


def print_result_summary(result) -> None:
    """Print summary of a benchmark result.

    Parameters
    ----------
    result : ComparisonResult
        Benchmark result.
    """
    speedup_str = f"{result.speedup_factor:.2f}x"
    if result.speedup_factor > 1.2:
        symbol = "+"
    elif result.speedup_factor < 0.8:
        symbol = "-"
    else:
        symbol = "="

    print(
        f"  {symbol} {result.competitor_name:20s}: "
        f"Speedup: {speedup_str:>8s}, "
        f"Memory: {result.memory_ratio:.2f}x, "
        f"Accuracy: {result.accuracy_correlation:.3f}"
    )
    print(
        f"     Runtime: ScpTensor={result.scptensor_time * 1000:.1f}ms, "
        f"Competitor={result.competitor_time * 1000:.1f}ms"
    )


def run_benchmark_suite(
    datasets: list,
    operations: list[str] | None = None,
    n_repeats: int = 1,
    verbose: bool = False,
) -> tuple[dict, dict]:
    """Run the complete benchmark suite.

    Parameters
    ----------
    datasets : list
        List of datasets to benchmark.
    operations : list[str] | None
        Operations to benchmark.
    n_repeats : int
        Number of repeats.
    verbose : bool
        Verbose output.

    Returns
    -------
    tuple
        (results_dict, summaries_dict)
    """
    CompetitorBenchmarkSuite = _get_classes()[0]

    if operations is None:
        operations = [
            "log_normalization",
            "zscore_normalization",
            "knn_imputation",
            "svd_imputation",
            "pca",
        ]

    suite = CompetitorBenchmarkSuite(verbose=verbose)
    suite.datasets = datasets

    results_dict = {}
    summaries_dict = {}

    for operation in operations:
        print_operation_header(operation)

        op_results = []
        for repeat in range(n_repeats):
            if n_repeats > 1:
                print(f"\n  Repeat {repeat + 1}/{n_repeats}...")

            for i, dataset in enumerate(datasets):
                n_samples = dataset.n_samples
                n_features = dataset.assays["protein"].n_features
                if verbose or i == 0:
                    print(f"  Dataset {i + 1}: {n_samples} samples x {n_features} features")

                try:
                    result = suite._run_single_benchmark(dataset, operation)
                    op_results.append(result)
                except Exception as e:
                    print(f"    ERROR: {e}")
                    continue

                if verbose or i == 0:
                    print_result_summary(result)

        results_dict[operation] = op_results

        # Compute summary
        if op_results:
            speedups = [r.speedup_factor for r in op_results]
            memory_ratios = [r.memory_ratio for r in op_results]
            accuracies = [r.accuracy_correlation for r in op_results]

            summaries_dict[operation] = {
                "n_comparisons": len(op_results),
                "mean_speedup": float(np.mean(speedups)),
                "std_speedup": float(np.std(speedups)),
                "min_speedup": float(np.min(speedups)),
                "max_speedup": float(np.max(speedups)),
                "mean_memory_ratio": float(np.mean(memory_ratios)),
                "mean_accuracy": float(np.mean(accuracies)),
                "winner": "scptensor"
                if np.mean(speedups) > 1.1
                else ("competitor" if np.mean(speedups) < 0.9 else "mixed"),
            }

    return results_dict, summaries_dict


def save_results(
    results_dict: dict,
    summaries_dict: dict,
    output_dir: Path,
    datasets: list,
) -> Path:
    """Save benchmark results to JSON file.

    Parameters
    ----------
    results_dict : dict
        Results dictionary.
    summaries_dict : dict
        Summaries dictionary.
    output_dir : Path
        Output directory.
    datasets : list
        Datasets used.

    Returns
    -------
    Path
        Path to saved results file.
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    # Convert results to serializable format
    serializable: dict[str, Any] = {
        op: [
            {
                "operation": r.operation,
                "scptensor_time": r.scptensor_time,
                "competitor_time": r.competitor_time,
                "scptensor_memory": r.scptensor_memory,
                "competitor_memory": r.competitor_memory,
                "speedup_factor": r.speedup_factor,
                "memory_ratio": r.memory_ratio,
                "accuracy_correlation": r.accuracy_correlation,
                "competitor_name": r.competitor_name,
                "parameters": r.parameters,
                "timestamp": r.timestamp,
            }
            for r in results
        ]
        for op, results in results_dict.items()
    }

    serializable["_summary"] = summaries_dict  # type: ignore[assignment]
    serializable["_metadata"] = {  # type: ignore[assignment]
        "timestamp": datetime.now().isoformat(),
        "n_datasets": len(datasets),
        "dataset_sizes": [(d.n_samples, d.assays["protein"].n_features) for d in datasets],
    }

    results_file = output_dir / "competitor_benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(serializable, f, indent=2)

    return results_file


def print_final_summary(summaries_dict: dict, output_dir: Path) -> None:
    """Print final benchmark summary.

    Parameters
    ----------
    summaries_dict : dict
        Summaries dictionary.
    output_dir : Path
        Output directory.
    """
    print_header("BENCHMARK SUMMARY")

    for op, summary in summaries_dict.items():
        print(f"\n{op.replace('_', ' ').title()}:")
        print(f"  Comparisons: {summary['n_comparisons']}")
        print(f"  Mean Speedup: {summary['mean_speedup']:.3f}x (+/- {summary['std_speedup']:.3f})")
        print(f"  Speedup Range: {summary['min_speedup']:.3f}x - {summary['max_speedup']:.3f}x")
        print(f"  Mean Memory Ratio: {summary['mean_memory_ratio']:.3f}x")
        print(f"  Mean Accuracy: {summary['mean_accuracy']:.3f}")
        print(f"  Winner: {summary['winner'].upper()}")

    print("\n" + "=" * 70)
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


def main() -> int:
    """Main entry point for the benchmark runner.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    """
    args = parse_arguments()

    print_header("ScpTensor Competitor Benchmark")

    print("\nConfiguration:")
    print(f"  Mode: {'Quick' if args.quick else 'Full'}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Operations: {args.operations or 'All'}")
    print(f"  Repeats: {args.n_repeats}")

    # Create datasets
    print("\nGenerating datasets...")
    start_time = time.time()
    datasets = create_datasets(quick=args.quick, custom_size=args.dataset_sizes)
    dataset_time = time.time() - start_time
    print(f"  Created {len(datasets)} dataset(s) in {dataset_time:.2f}s")

    for i, d in enumerate(datasets):
        print(f"    {i + 1}. {d.n_samples} samples x {d.assays['protein'].n_features} features")

    # Run benchmarks
    print("\nRunning benchmarks...")
    benchmark_start = time.time()

    try:
        results_dict, summaries_dict = run_benchmark_suite(
            datasets=datasets,
            operations=args.operations,
            n_repeats=args.n_repeats,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"\nERROR during benchmarking: {e}")
        import traceback

        traceback.print_exc()
        return 1

    benchmark_time = time.time() - benchmark_start
    print(f"\nBenchmarking completed in {benchmark_time:.2f}s")

    # Save results
    output_dir = Path(args.output_dir)
    results_file = save_results(results_dict, summaries_dict, output_dir, datasets)
    print(f"\nResults saved to: {results_file}")

    # Generate visualizations
    if not args.no_plots:
        print("\nGenerating visualizations...")
        try:
            CompetitorResultVisualizer = _get_classes()[2]
            viz = CompetitorResultVisualizer()
            viz.results = results_dict
            viz.summaries = summaries_dict

            plots_dir = output_dir / "plots"
            plots = viz.create_all_plots(output_dir=plots_dir)

            print(f"Generated {len(plots)} visualization plots:")
            for plot_type, plot_path in plots.items():
                print(f"  - {plot_type}: {plot_path}")
        except Exception as e:
            print(f"WARNING: Could not generate all plots: {e}")
            import traceback

            traceback.print_exc()

    # Print summary
    print_final_summary(summaries_dict, output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
