"""Pipeline comparison runner - simplified version.

Usage:
    python -m studies.run_comparison --test
    python -m studies.run_comparison --pipelines classic batch_corrected
    python -m studies.run_comparison --full
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

from scptensor import create_test_container

from .evaluation.core import evaluate_pipeline, monitor_performance
from .pipelines.executor import get_available_pipelines, run_pipeline


def generate_test_data(size: str = "small") -> Any:
    """Generate test data container."""
    n_cells = {"small": 100, "medium": 500, "large": 1000}.get(size, 100)
    n_proteins = {"small": 500, "medium": 1000, "large": 2000}.get(size, 500)

    return create_test_container(n_cells=n_cells, n_proteins=n_proteins)


def run_comparison(
    pipelines: list[str] | None = None,
    data_size: str = "small",
    output_dir: str = "studies/outputs",
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Run pipeline comparison.

    Parameters
    ----------
    pipelines : list of str, optional
        Pipeline names to run. None = all pipelines
    data_size : str
        "small", "medium", or "large"
    output_dir : str
        Output directory for results
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Comparison results
    """
    if pipelines is None:
        pipelines = get_available_pipelines()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate data
    if verbose:
        print(f"Generating {data_size} test data...")

    container = generate_test_data(data_size)

    results = {}

    for pipeline_name in pipelines:
        if verbose:
            print(f"\nRunning pipeline: {pipeline_name}")

        try:
            with monitor_performance() as perf:
                result_container, log = run_pipeline(
                    container, pipeline_name=pipeline_name, verbose=verbose
                )

            # Evaluate
            metrics = evaluate_pipeline(
                original=container,
                result=result_container,
                runtime=perf["runtime"],
                memory_peak=perf["memory_peak"],
                pipeline_name=pipeline_name,
                dataset_name=data_size,
            )

            results[pipeline_name] = {
                "status": "success",
                "metrics": metrics,
                "log": log,
            }

        except Exception as e:
            results[pipeline_name] = {
                "status": "failed",
                "error": str(e),
            }
            if verbose:
                print(f"  ERROR: {e}")

    # Save results
    results_file = output_path / "results" / f"comparison_{data_size}.pkl"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "wb") as f:
        pickle.dump(results, f)

    if verbose:
        print(f"\nResults saved to: {results_file}")

    return results


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Run pipeline comparison")
    parser.add_argument(
        "--pipelines",
        nargs="+",
        choices=get_available_pipelines(),
        help="Pipelines to run (default: all)",
    )
    parser.add_argument(
        "--size", choices=["small", "medium", "large"], default="small", help="Data size"
    )
    parser.add_argument("--output", default="studies/outputs", help="Output directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--test", action="store_true", help="Quick test mode (small data, 2 pipelines)"
    )
    parser.add_argument(
        "--full", action="store_true", help="Full comparison (all sizes, all pipelines)"
    )

    args = parser.parse_args()

    if args.test:
        pipelines = ["classic", "batch_corrected"]
        size = "small"
    elif args.full:
        pipelines = None
        # Run all sizes
        all_results = {}
        for size in ["small", "medium", "large"]:
            print(f"\n{'=' * 50}")
            print(f"Running size: {size}")
            print("=" * 50)
            all_results[size] = run_comparison(
                pipelines=None, data_size=size, output_dir=args.output, verbose=args.verbose
            )
        return all_results
    else:
        pipelines = args.pipelines
        size = args.size

    return run_comparison(
        pipelines=pipelines, data_size=size, output_dir=args.output, verbose=args.verbose
    )


if __name__ == "__main__":
    main()
