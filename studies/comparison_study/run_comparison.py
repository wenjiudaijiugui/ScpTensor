#!/usr/bin/env python3
"""
Simplified pipeline comparison main runner script.

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

import argparse  # noqa: E402
import pickle  # noqa: E402
import time  # noqa: E402
from typing import Any  # noqa: E402

import yaml  # noqa: E402

# =============================================================================
# Import Streamlined Modules
# =============================================================================
# Add current directory to path for imports
_comparison_dir = Path(__file__).parent
if str(_comparison_dir) not in sys.path:
    sys.path.insert(0, str(_comparison_dir))

from studies.comparison_study.comparison_engine import (  # noqa: E402
    compare_pipelines,
    generate_comparison_report,
)
from studies.comparison_study.data_generation import (  # noqa: E402
    generate_large_dataset,
    generate_medium_dataset,
    generate_small_dataset,
)
from studies.comparison_study.plotting import (  # noqa: E402
    plot_batch_effects,
    plot_performance_comparison,
    plot_radar_chart,
)

# =============================================================================
# Command Line Interface
# =============================================================================


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


def load_datasets(
    use_cache: bool = True, verbose: bool = False, dataset_names: list[str] | None = None
) -> dict[str, Any]:
    """
    Load or generate specified datasets.

    Parameters
    ----------
    use_cache : bool
        Whether to use cached datasets if available
    verbose : bool
        Whether to print progress messages
    dataset_names : list, optional
        List of dataset names to generate. If None, generates all datasets.

    Returns
    -------
    dict
        Dictionary of dataset name → ScpContainer
    """
    datasets = {}

    # Determine which datasets to generate
    if dataset_names is None:
        dataset_names = ["small", "medium", "large"]

    # Generate datasets
    for name in dataset_names:
        if verbose:
            print(f"Generating {name} dataset...")

        if name == "small":
            datasets[name] = generate_small_dataset(seed=42)
        elif name == "medium":
            datasets[name] = generate_medium_dataset(seed=42)
        elif name == "large":
            datasets[name] = generate_large_dataset(seed=42)
        else:
            if verbose:
                print(f"  Warning: Unknown dataset '{name}', skipping...")

        if verbose:
            print(f"  ✓ Generated {name} dataset")

    return datasets


def initialize_pipelines(config: dict[str, Any]) -> dict[str, Any]:
    """
    Initialize all pipeline instances.

    Parameters
    ----------
    config : dict
        Configuration dictionary

    Returns
    -------
    dict
        Dictionary of pipeline name → pipeline instance
    """
    from studies.comparison_study.pipelines import (
        PipelineA,
        PipelineB,
        PipelineC,
        PipelineD,
        PipelineE,
    )

    pipelines = {
        "PipelineA": PipelineA(),
        "PipelineB": PipelineB(),
        "PipelineC": PipelineC(),
        "PipelineD": PipelineD(),
        "PipelineE": PipelineE(),
    }

    return pipelines


# =============================================================================
# Main Execution Flow
# =============================================================================


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
    datasets = load_datasets(
        use_cache=not args.no_cache, verbose=args.verbose, dataset_names=dataset_names
    )

    # Initialize pipelines
    if args.verbose:
        print("\nInitializing pipelines...")

    pipelines = initialize_pipelines(config)

    if args.verbose:
        print(f"✓ Initialized {len(pipelines)} pipelines:")
        for pipeline_name in pipelines:
            print(f"  - {pipeline_name}")

    # Run comparison using streamlined comparison engine
    if args.verbose:
        print("\nRunning pipeline comparison...")

    start_time = time.time()

    # Use compare_pipelines from comparison_engine
    comparison_results = compare_pipelines(
        pipelines=pipelines,
        datasets=datasets,
        metrics_list=["kbet", "ilisi", "clisi", "asw"],
    )

    total_runtime = time.time() - start_time

    if args.verbose:
        print(f"✓ Comparison completed in {total_runtime:.2f}s")

    # Generate visualizations
    if args.verbose:
        print("\nGenerating visualizations...")

    figures = []

    # Prepare data for plotting
    # Convert results to format expected by plotting functions
    plotting_results = {}
    for pipeline_name, pipeline_results in comparison_results.items():
        plotting_results[pipeline_name] = {}
        for dataset_name, dataset_results in pipeline_results.items():
            scores = dataset_results.get("scores", {})
            plotting_results[pipeline_name][dataset_name] = scores

    try:
        # Plot batch effects for each dataset
        for dataset_name in datasets:
            dataset_results = {
                pipeline_name: pipeline_results.get(dataset_name, {})
                for pipeline_name, pipeline_results in comparison_results.items()
            }

            output_path = output_dir / "figures" / f"{dataset_name}_batch_effects.png"
            fig_path = plot_batch_effects(dataset_results, output_path=output_path)
            figures.append(fig_path)

        # Plot performance comparison
        perf_results = {}
        for pipeline_name, pipeline_results in comparison_results.items():
            total_time = sum(
                dataset_results.get("runtime", 0) for dataset_results in pipeline_results.values()
            )
            perf_results[pipeline_name] = {"execution_time": total_time}

        fig_path = plot_performance_comparison(
            perf_results, output_path=output_dir / "figures" / "performance.png"
        )
        figures.append(fig_path)

        # Plot radar chart (for first dataset)
        first_dataset = list(datasets.keys())[0]
        radar_data = {
            pipeline_name: pipeline_results[first_dataset].get("scores", {})
            for pipeline_name, pipeline_results in comparison_results.items()
        }

        fig_path = plot_radar_chart(
            radar_data, output_path=output_dir / "figures" / "radar_chart.png"
        )
        figures.append(fig_path)

        if args.verbose:
            print(f"✓ Generated {len(figures)} figures")

    except Exception as e:
        if args.verbose:
            print(f"✗ Failed to generate figures: {e}")
            import traceback

            traceback.print_exc()

    # Generate comparison report
    if args.verbose:
        print("\nGenerating comparison report...")

    try:
        report_path = output_dir / "comparison_report.md"
        generate_comparison_report(comparison_results, output_path=str(report_path))

        if args.verbose:
            print(f"✓ Report generated: {report_path}")

    except Exception as e:
        if args.verbose:
            print(f"✗ Failed to generate report: {e}")
            import traceback

            traceback.print_exc()

    # Save complete results
    results_file = output_dir / "results" / "complete_results.pkl"
    with open(results_file, "wb") as f:
        pickle.dump(
            {
                "results": comparison_results,
                "config": config,
                "total_runtime": total_runtime,
            },
            f,
        )

    # Print summary
    if args.verbose:
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        print(f"Total runtime: {total_runtime / 60:.2f} minutes")
        print(f"Datasets used: {list(datasets.keys())}")
        print(f"Pipelines tested: {len(pipelines)}")
        print(f"Repeats per experiment: {n_repeats}")

        # Count successful vs failed
        successful = 0
        for pipeline_results in comparison_results.values():
            for dataset_results in pipeline_results.values():
                if "scores" in dataset_results:
                    successful += 1

        total = sum(len(pipeline_results) for pipeline_results in comparison_results.values())
        print(f"Total experiments: {total}")
        print(f"  Successful: {successful}")

        print(f"\nResults saved to: {output_dir}")
        print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
