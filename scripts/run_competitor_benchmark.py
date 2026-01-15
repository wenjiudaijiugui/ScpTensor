#!/usr/bin/env python3
"""Comprehensive competitor benchmark runner for ScpTensor.

Compares ScpTensor performance against competing tools:
- scanpy-style operations
- scikit-learn
- scipy
- numpy

Benchmarked operations:
1. Normalization (log, z-score)
2. Imputation (KNN, SVD)
3. Dimensionality reduction (PCA)

Usage:
    python scripts/run_competitor_benchmark.py
    python scripts/run_competitor_benchmark.py --output results/
    python scripts/run_competitor_benchmark.py --quick
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

# Add project root to path dynamically
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scptensor.benchmark import (
    CompetitorBenchmarkSuite,
    ComparisonResult,
    SyntheticDataset,
)

# Dataset configurations for comprehensive benchmarking
DATASET_CONFIGS: Final = [
    {
        "name": "small",
        "n_samples": 50,
        "n_features": 200,
        "n_groups": 2,
        "n_batches": 2,
        "missing_rate": 0.2,
        "signal_to_noise_ratio": 3.0,
        "random_seed": 42,
    },
    {
        "name": "medium",
        "n_samples": 100,
        "n_features": 500,
        "n_groups": 3,
        "n_batches": 3,
        "missing_rate": 0.3,
        "signal_to_noise_ratio": 2.0,
        "random_seed": 123,
    },
    {
        "name": "large",
        "n_samples": 200,
        "n_features": 1000,
        "n_groups": 4,
        "n_batches": 4,
        "missing_rate": 0.35,
        "signal_to_noise_ratio": 1.5,
        "random_seed": 456,
    },
]

# Operations to benchmark
OPERATIONS: Final = [
    "log_normalization",
    "zscore_normalization",
    "knn_imputation",
    "svd_imputation",
    "pca",
]


def create_datasets(quick: bool = False) -> list:
    """Create test datasets for benchmarking.

    Args:
        quick: If True, create only small dataset

    Returns:
        List of ScpContainer datasets
    """
    if quick:
        config = DATASET_CONFIGS[0]
        return [SyntheticDataset(**config).generate()]

    return [SyntheticDataset(**config).generate() for config in DATASET_CONFIGS]


def run_benchmark_suite(
    output_dir: str | Path = "competitor_benchmark_results",
    quick: bool = False,
) -> dict:
    """Run the full competitor benchmark suite.

    Args:
        output_dir: Directory to save results
        quick: If True, run quick version with single dataset

    Returns:
        Dictionary of benchmark results
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print("=" * 70)
    print("ScpTensor Competitor Benchmark Suite")
    print("=" * 70)

    # Initialize suite
    suite = CompetitorBenchmarkSuite(output_dir=str(output_dir), verbose=True)

    # Generate datasets
    print("\n1. Generating test datasets...")
    datasets = create_datasets(quick)
    print(f"   Created {len(datasets)} dataset(s)")

    for i, ds in enumerate(datasets):
        assay = ds.assays.get("protein", list(ds.assays.values())[0])
        print(f"   Dataset {i + 1}: {ds.n_samples} samples x {assay.n_features} features")

    # Run benchmarks
    print("\n2. Running benchmarks...")
    results = suite.run_all_benchmarks(datasets=datasets, operations=OPERATIONS)

    # Print summary
    print("\n3. Benchmark Summary")
    suite.print_summary()

    # Export results
    print("\n4. Saving results...")
    results_file = suite.save_results()
    csv_file = output_path / "benchmark_results.csv"

    df = suite.export_to_dataframe()
    df.to_csv(csv_file, index=False)
    print(f"   CSV: {csv_file}")

    # Generate markdown report
    summary_file = output_path / "benchmark_summary.md"
    generate_markdown_report(suite, summary_file)

    return results


def generate_markdown_report(
    suite: CompetitorBenchmarkSuite,
    output_path: Path,
) -> None:
    """Generate a markdown report of benchmark results.

    Args:
        suite: Benchmark suite with results
        output_path: Destination for report file
    """
    summaries = suite.summarize_results()
    df = suite.export_to_dataframe()

    lines = [
        "# ScpTensor Competitor Benchmark Report\n",
        f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "## Summary\n",
        "Performance comparison against competing tools.\n",
        "## Results by Operation\n",
    ]

    # Operation summaries
    for operation, summary in summaries.items():
        name = operation.replace("_", " ").title()
        lines.extend([
            f"### {name}\n",
            f"- **Winner:** {summary.winner.upper()}",
            f"- **Mean Speedup:** {summary.mean_speedup:.3f}x (+/- {summary.std_speedup:.3f})",
            f"- **Memory Ratio:** {summary.mean_memory_ratio:.3f}",
            f"- **Accuracy Correlation:** {summary.mean_accuracy:.3f}",
            f"- **Comparisons:** {summary.n_comparisons}",
            "",
        ])

    # Detailed results table
    lines.extend([
        "## Detailed Results\n",
        "| Operation | Dataset | ScpTensor (ms) | Competitor (ms) | Speedup | Accuracy |",
        "|-----------|---------|----------------|-----------------|---------|----------|",
    ])

    for _, row in df.iterrows():
        lines.append(
            f"| {row['operation']} | {row['dataset_index']} | "
            f"{row['scptensor_time_ms']:.2f} | {row['competitor_time_ms']:.2f} | "
            f"{row['speedup_factor']:.3f}x | {row['accuracy_correlation']:.3f} |"
        )

    # Methodology
    lines.extend([
        "\n## Methodology\n",
        "- **Test Data:** Synthetic single-cell proteomics datasets",
        "- **Metrics:** Runtime, memory usage, output correlation",
        "- **Speedup > 1.0:** ScpTensor is faster",
        "- **Speedup < 1.0:** Competitor is faster",
        "- **Accuracy:** Correlation between ScpTensor and competitor outputs\n",
        "## Competitors\n",
        "- **numpy:** Raw numerical computing",
        "- **scikit-learn:** Machine learning library",
        "- **scipy:** Scientific computing",
        "- **scanpy-style:** Single-cell analysis patterns",
    ])

    output_path.write_text("\n".join(lines))
    print(f"   Markdown: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run ScpTensor competitor benchmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        "-o",
        default="competitor_benchmark_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--quick",
        "-q",
        action="store_true",
        help="Run quick benchmark (1 dataset)",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    args = parse_args()

    try:
        run_benchmark_suite(output_dir=args.output, quick=args.quick)
        print("\nBenchmark completed successfully!")
        return 0
    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
