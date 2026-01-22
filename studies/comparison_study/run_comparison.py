#!/usr/bin/env python3
"""
Main runner script for pipeline comparison study.

This script executes the complete pipeline comparison experiment, including:
- Loading/generating datasets
- Running all pipelines on all datasets
- Evaluating performance across multiple dimensions
- Generating visualizations
- Creating comprehensive report

Usage
-----
Basic usage:
    python run_comparison.py

Full experiment with all datasets:
    python run_comparison.py --full

Quick test with small dataset only:
    python run_comparison.py --test

Custom configuration:
    python run_comparison.py --config custom_config.yaml

Options
-------
--full : Run complete experiment (all datasets, multiple repeats)
--test : Quick test mode (small dataset, single repeat)
--config PATH : Path to custom configuration file
--output DIR : Output directory for results
--no-cache : Regenerate datasets even if cached
--repeats N : Number of repeats per experiment (default: 3)
--verbose : Enable verbose output
"""

from __future__ import annotations

# Setup Python path
import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import argparse
import pickle
import time
from typing import Any

import yaml


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run pipeline comparison study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Run complete experiment (all datasets, multiple repeats)",
    )

    parser.add_argument("--test", action="store_true", help="Quick test mode (small dataset only)")

    parser.add_argument(
        "--config", type=str, default=None, help="Path to custom configuration file"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="docs/comparison_study/outputs",
        help="Output directory for results",
    )

    parser.add_argument(
        "--no-cache", action="store_true", help="Regenerate datasets even if cached"
    )

    parser.add_argument("--repeats", type=int, default=3, help="Number of repeats per experiment")

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    return parser.parse_args()


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """
    Load configuration from YAML file.

    Parameters
    ----------
    config_path : str, optional
        Path to configuration file. If None, uses default config.

    Returns
    -------
    dict
        Configuration dictionary with 'pipeline' and 'evaluation' keys
    """
    if config_path is None:
        # Use default configs
        base_dir = Path(__file__).parent

        pipeline_config_path = base_dir / "configs" / "pipeline_configs.yaml"
        eval_config_path = base_dir / "configs" / "evaluation_config.yaml"

        configs = {}

        with open(pipeline_config_path) as f:
            configs["pipeline"] = yaml.safe_load(f)

        with open(eval_config_path) as f:
            configs["evaluation"] = yaml.safe_load(f)

        return configs
    else:
        # Load custom config
        with open(config_path) as f:
            return yaml.safe_load(f)


def setup_output_directory(output_dir: str) -> Path:
    """
    Create and setup output directory structure.

    Parameters
    ----------
    output_dir : str
        Output directory path

    Returns
    -------
    Path
        Path object for output directory
    """
    output_path = Path(output_dir)

    # Create subdirectories
    (output_path / "results").mkdir(parents=True, exist_ok=True)
    (output_path / "figures").mkdir(parents=True, exist_ok=True)
    (output_path / "data_cache").mkdir(parents=True, exist_ok=True)

    return output_path


def load_datasets(use_cache: bool = True, verbose: bool = False) -> dict[str, Any]:
    """
    Load or generate all datasets.

    Parameters
    ----------
    use_cache : bool
        Whether to use cached datasets if available
    verbose : bool
        Whether to print progress messages

    Returns
    -------
    dict
        Dictionary of dataset name → ScpContainer
    """
    from docs.comparison_study.data import cache_datasets, load_all_datasets, load_cached_datasets

    cache_dir = "docs/comparison_study/outputs/data_cache"

    # Try to load from cache
    if use_cache:
        try:
            if verbose:
                print("Loading cached datasets...")
            datasets = load_cached_datasets(cache_dir)
            if verbose:
                print("✓ Loaded cached datasets")
            return datasets
        except FileNotFoundError:
            if verbose:
                print("No cached datasets found, generating new ones...")

    # Generate new datasets
    if verbose:
        print("Generating synthetic datasets...")

    datasets = load_all_datasets()

    # Cache datasets
    if verbose:
        print("Caching datasets...")

    cache_datasets(datasets, cache_dir)

    if verbose:
        print("✓ Generated and cached datasets")

    return datasets


def initialize_pipelines(config: dict[str, Any]) -> list:
    """
    Initialize all pipeline instances.

    Parameters
    ----------
    config : dict
        Configuration dictionary

    Returns
    -------
    list
        List of pipeline instances
    """
    from docs.comparison_study.pipelines import (
        PipelineA,
        PipelineB,
        PipelineC,
        PipelineD,
        PipelineE,
    )

    pipelines = [PipelineA(), PipelineB(), PipelineC(), PipelineD(), PipelineE()]

    return pipelines


def run_single_pipeline(
    pipeline: Any,
    container: Any,
    evaluator: Any,
    pipeline_name: str,
    dataset_name: str,
    repeat: int,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Run a single pipeline on a dataset and evaluate results.

    Parameters
    ----------
    pipeline : BasePipeline
        Pipeline instance to run
    container : ScpContainer
        Input data container
    evaluator : PipelineEvaluator
        Evaluator instance
    pipeline_name : str
        Name of the pipeline
    dataset_name : str
        Name of the dataset
    repeat : int
        Repeat number
    verbose : bool
        Whether to print progress

    Returns
    -------
    dict
        Evaluation results
    """
    if verbose:
        print(f"  Running {pipeline_name} on {dataset_name} (repeat {repeat + 1})...")

    # Monitor performance
    from docs.comparison_study.evaluation.performance import monitor_performance

    result_container = None
    runtime = 0.0
    memory_peak = 0.0

    try:
        with monitor_performance() as perf:
            # Run pipeline
            result_container = pipeline.run(container)

        # Read performance metrics AFTER context exits
        runtime = perf["runtime"]
        memory_peak = perf["memory_peak"]

    except Exception as e:
        if verbose:
            print(f"    ✗ Pipeline failed: {e}")
        # If failed, still try to get performance metrics
        if "perf" in locals():
            runtime = perf.get("runtime", 0.0)
            memory_peak = perf.get("memory_peak", 0.0)
        return {
            "pipeline_name": pipeline_name,
            "dataset_name": dataset_name,
            "repeat": repeat,
            "error": str(e),
            "runtime": runtime,
            "memory_peak": memory_peak,
        }

    # Evaluate results
    try:
        results = evaluator.evaluate(
            original_container=container,
            result_container=result_container,
            runtime=runtime,
            memory_peak=memory_peak,
            pipeline_name=pipeline_name,
            dataset_name=dataset_name,
        )

        results["repeat"] = repeat

        if verbose:
            print(f"    ✓ Completed in {runtime:.2f}s, {memory_peak:.2f}GB")

    except Exception as e:
        if verbose:
            print(f"    ✗ Evaluation failed: {e}")
        results = {
            "pipeline_name": pipeline_name,
            "dataset_name": dataset_name,
            "repeat": repeat,
            "error": str(e),
            "runtime": runtime,
            "memory_peak": memory_peak,
        }

    return results


def run_complete_experiment(
    datasets: dict[str, Any],
    pipelines: list,
    config: dict[str, Any],
    output_dir: Path,
    dataset_names: list[str] | None = None,
    n_repeats: int = 3,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Run complete comparison experiment.

    Parameters
    ----------
    datasets : dict
        Dictionary of dataset name → container
    pipelines : list
        List of pipeline instances
    config : dict
        Configuration dictionary
    output_dir : Path
        Output directory
    dataset_names : list, optional
        List of dataset names to use. If None, uses all datasets.
    n_repeats : int
        Number of repeats per experiment
    verbose : bool
        Whether to print progress messages

    Returns
    -------
    dict
        All evaluation results
    """
    from docs.comparison_study.evaluation import PipelineEvaluator

    # Initialize evaluator
    evaluator = PipelineEvaluator(config["evaluation"])

    # Filter datasets if specified
    if dataset_names is not None:
        datasets = {name: datasets[name] for name in dataset_names if name in datasets}

    # Results storage
    all_results = {}

    # Total number of experiments
    n_experiments = len(datasets) * len(pipelines) * n_repeats
    experiment_count = 0

    if verbose:
        print(f"\nRunning {n_experiments} experiments...")
        print("=" * 60)

    # Run experiments
    for dataset_name, container in datasets.items():
        for pipeline in pipelines:
            pipeline_name = pipeline.name.replace(" ", "_").lower()

            for repeat in range(n_repeats):
                experiment_count += 1

                if verbose:
                    print(f"\n[{experiment_count}/{n_experiments}]")

                # Run pipeline and evaluate
                results = run_single_pipeline(
                    pipeline=pipeline,
                    container=container,
                    evaluator=evaluator,
                    pipeline_name=pipeline_name,
                    dataset_name=dataset_name,
                    repeat=repeat,
                    verbose=verbose,
                )

                # Store results
                key = f"{dataset_name}_{pipeline_name}_r{repeat}"
                all_results[key] = results

                # Save intermediate results
                results_path = output_dir / "results" / f"{key}.pkl"
                with open(results_path, "wb") as f:
                    pickle.dump(results, f)

    if verbose:
        print("\n" + "=" * 60)
        print("✓ All experiments completed!")

    return all_results


def aggregate_results(results: dict[str, Any]) -> dict[str, Any]:
    """
    Aggregate results across repeats.

    Parameters
    ----------
    results : dict
        All individual results

    Returns
    -------
    dict
        Aggregated results (mean ± std across repeats)
    """
    import numpy as np

    aggregated = {}

    # Group results by pipeline and dataset
    groups = {}
    for key, result in results.items():
        if "error" in result:
            continue

        # Key format: dataset_pipeline_name_r0
        # We want: dataset_pipeline_name (without _r{repeat} suffix)
        parts = key.rsplit("_r", 1)  # Split from the right at "_r"
        base_key = parts[0]  # Get the part before "_r{repeat}"

        if base_key not in groups:
            groups[base_key] = []
        groups[base_key].append(result)

    # Compute statistics
    for base_key, group_results in groups.items():
        # Aggregate each metric
        agg_result = {
            "pipeline_name": group_results[0]["pipeline_name"],
            "dataset_name": group_results[0]["dataset_name"],
            "n_repeats": len(group_results),
        }

        # Aggregate nested metrics
        for dimension in ["batch_effects", "performance", "distribution", "structure"]:
            if dimension in group_results[0]:
                agg_result[dimension] = {}

                for metric_name in group_results[0][dimension].keys():
                    values = [
                        r[dimension][metric_name]
                        for r in group_results
                        if dimension in r and metric_name in r[dimension]
                    ]

                    if values:
                        agg_result[dimension][metric_name] = {
                            "mean": float(np.mean(values)),
                            "std": float(np.std(values)),
                            "min": float(np.min(values)),
                            "max": float(np.max(values)),
                        }

        aggregated[base_key] = agg_result

    return aggregated


def print_experiment_summary(
    results: dict[str, Any],
    datasets: list[str],
    pipelines: list[str],
    n_repeats: int,
    total_runtime: float,
) -> None:
    """
    Print experiment summary to console.

    Parameters
    ----------
    results : dict
        All evaluation results
    datasets : list
        Dataset names used
    pipelines : list
        Pipeline names used
    n_repeats : int
        Number of repeats
    total_runtime : float
        Total runtime in seconds
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Total runtime: {total_runtime / 60:.2f} minutes")
    print(f"Datasets used: {datasets}")
    print(f"Pipelines tested: {len(pipelines)}")
    print(f"Repeats per experiment: {n_repeats}")

    # Count successful vs failed
    successful = sum(1 for r in results.values() if "error" not in r)
    failed = sum(1 for r in results.values() if "error" in r)

    print(f"Total experiments: {len(results)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")

    if failed > 0:
        print("\nFailed experiments:")
        for key, result in results.items():
            if "error" in result:
                print(f"  - {key}: {result['error']}")


def main() -> int:
    """Main entry point for comparison study."""
    # Parse arguments
    args = parse_arguments()

    # Determine mode
    if args.test:
        dataset_names = ["small"]
        n_repeats = 1
    elif args.full:
        dataset_names = None  # Use all datasets
        n_repeats = args.repeats
    else:
        # Default: medium dataset, single repeat
        dataset_names = ["medium"]
        n_repeats = 1

    # Setup
    if args.verbose:
        print("\n" + "=" * 60)
        print("Single-Cell Proteomics Pipeline Comparison Study")
        print("=" * 60)

    # Load configuration
    if args.verbose:
        print("\nLoading configuration...")

    config = load_config(args.config)

    # Setup output directory
    output_dir = setup_output_directory(args.output)

    if args.verbose:
        print(f"Output directory: {output_dir}")

    # Load datasets
    datasets = load_datasets(use_cache=not args.no_cache, verbose=args.verbose)

    # Initialize pipelines
    if args.verbose:
        print("\nInitializing pipelines...")

    pipelines = initialize_pipelines(config)

    if args.verbose:
        print(f"✓ Initialized {len(pipelines)} pipelines:")
        for pipeline in pipelines:
            print(f"  - {pipeline.name}")

    # Run experiments
    start_time = time.time()

    results = run_complete_experiment(
        datasets=datasets,
        pipelines=pipelines,
        config=config,
        output_dir=output_dir,
        dataset_names=dataset_names,
        n_repeats=n_repeats,
        verbose=args.verbose,
    )

    total_runtime = time.time() - start_time

    # Aggregate results (even for single repeat, for consistent key format)
    if n_repeats >= 1:
        if args.verbose and n_repeats > 1:
            print("\nAggregating results across repeats...")
        aggregated = aggregate_results(results)
    else:
        aggregated = results

    # Generate visualizations
    if args.verbose:
        print("\nGenerating visualizations...")

    from docs.comparison_study.visualization import generate_all_figures

    try:
        figures = generate_all_figures(
            results=aggregated if n_repeats > 1 else results,
            config=config["evaluation"],
            output_dir=str(output_dir / "figures"),
        )

        if args.verbose:
            print(f"✓ Generated {len(figures)} figures")

    except Exception as e:
        if args.verbose:
            print(f"✗ Failed to generate figures: {e}")
        figures = []

    # Generate report
    if args.verbose:
        print("\nGenerating report...")

    from docs.comparison_study.visualization import ReportGenerator

    try:
        generator = ReportGenerator(config=config["evaluation"], output_dir=str(output_dir))

        report_path = generator.generate_report(
            results=aggregated if n_repeats > 1 else results, figures=figures
        )

        if args.verbose:
            print(f"✓ Report generated: {report_path}")

    except Exception as e:
        if args.verbose:
            print(f"✗ Failed to generate report: {e}")

    # Save complete results
    results_file = output_dir / "results" / "complete_results.pkl"
    with open(results_file, "wb") as f:
        pickle.dump(
            {
                "results": results,
                "aggregated": aggregated,
                "config": config,
                "total_runtime": total_runtime,
            },
            f,
        )

    # Print summary
    if args.verbose:
        print_experiment_summary(
            results=results,
            datasets=list(datasets.keys()),
            pipelines=[p.name for p in pipelines],
            n_repeats=n_repeats,
            total_runtime=total_runtime,
        )

        print(f"\nResults saved to: {output_dir}")
        print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
