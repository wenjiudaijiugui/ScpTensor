#!/usr/bin/env python3
"""
Example script demonstrating how to use the pipeline comparison runner.

This example shows different ways to run the comparison study:
1. Using command-line interface
2. Using Python API directly
3. Customizing the experiment configuration
"""


def example_cli_usage():
    """Example 1: Using command-line interface."""
    print("\n" + "=" * 60)
    print("Example 1: Command-Line Interface")
    print("=" * 60)

    print("\n# Quick test mode")
    print("python docs/comparison_study/run_comparison.py --test --verbose")

    print("\n# Full experiment")
    print("python docs/comparison_study/run_comparison.py --full --repeats 3 --verbose")

    print("\n# Custom configuration")
    print("python docs/comparison_study/run_comparison.py --config custom.yaml --verbose")


def example_python_api():
    """Example 2: Using Python API directly."""
    print("\n" + "=" * 60)
    print("Example 2: Python API")
    print("=" * 60)

    code = """
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from docs.comparison_study.run_comparison import (
    load_config,
    setup_output_directory,
    load_datasets,
    initialize_pipelines,
    run_complete_experiment,
    aggregate_results
)

# Load configuration
config = load_config()

# Setup output directory
output_dir = setup_output_directory("outputs/my_experiment")

# Load datasets (use cache if available)
datasets = load_datasets(use_cache=True, verbose=True)

# Initialize pipelines
pipelines = initialize_pipelines(config)

# Run experiments
results = run_complete_experiment(
    datasets=datasets,
    pipelines=pipelines,
    config=config,
    output_dir=output_dir,
    dataset_names=["small"],  # Only test on small dataset
    n_repeats=3,
    verbose=True
)

# Aggregate results
aggregated = aggregate_results(results)

print(f"✓ Completed {len(results)} experiments")
"""

    print(code)


def example_custom_experiment():
    """Example 3: Custom experiment configuration."""
    print("\n" + "=" * 60)
    print("Example 3: Custom Experiment")
    print("=" * 60)

    code = """
from docs.comparison_study.run_comparison import (
    load_datasets,
    initialize_pipelines,
)
from docs.comparison_study.evaluation import PipelineEvaluator
from docs.comparison_study.evaluation.performance import monitor_performance
import pickle

# Load data
datasets = load_datasets(use_cache=True, verbose=True)

# Initialize specific pipelines only
from docs.comparison_study.pipelines import PipelineA, PipelineB
pipelines = [PipelineA(), PipelineB()]

# Custom evaluator with different weights
config = {
    'batch_effects': 0.4,  # Higher weight for batch correction
    'performance': 0.3,
    'distribution': 0.2,
    'structure': 0.1
}
evaluator = PipelineEvaluator(config)

# Run custom experiment
for dataset_name, container in datasets.items():
    for pipeline in pipelines:
        with monitor_performance() as perf:
            result = pipeline.run(container)

        # Evaluate with custom criteria
        metrics = evaluator.evaluate(
            original_container=container,
            result_container=result,
            runtime=perf['runtime'],
            memory_peak=perf['memory_peak'],
            pipeline_name=pipeline.name,
            dataset_name=dataset_name
        )

        print(f"{pipeline.name} on {dataset_name}:")
        print(f"  Runtime: {perf['runtime']:.2f}s")
        print(f"  Memory: {perf['memory_peak']:.2f}GB")
"""

    print(code)


def example_result_analysis():
    """Example 4: Analyzing results."""
    print("\n" + "=" * 60)
    print("Example 4: Analyzing Results")
    print("=" * 60)

    code = """
import pickle
import pandas as pd
import numpy as np

# Load complete results
with open("docs/comparison_study/outputs/results/complete_results.pkl", "rb") as f:
    data = pickle.load(f)

results = data["results"]
aggregated = data["aggregated"]

# Example 1: Get successful experiments
successful = {k: v for k, v in results.items() if "error" not in v}
print(f"Successful experiments: {len(successful)}/{len(results)}")

# Example 2: Compare pipeline performance
performance_data = []
for key, result in successful.items():
    perf = result.get("performance", {})
    performance_data.append({
        "pipeline": result["pipeline_name"],
        "dataset": result["dataset_name"],
        "runtime": perf.get("runtime", np.nan),
        "memory_peak": perf.get("memory_peak", np.nan)
    })

df = pd.DataFrame(performance_data)
print("\\nPerformance Summary:")
print(df.groupby("pipeline")[["runtime", "memory_peak"]].mean())

# Example 3: Ranking by batch correction
batch_scores = []
for key, result in successful.items():
    batch = result.get("batch_effects", {})
    batch_scores.append({
        "pipeline": result["pipeline_name"],
        "kbet_score": batch.get("kbet_score", np.nan),
        "lisi_score": batch.get("lisi_score", np.nan)
    })

batch_df = pd.DataFrame(batch_scores)
print("\\nBatch Correction Performance:")
print(batch_df.groupby("pipeline").mean())
"""

    print(code)


def example_visualization():
    """Example 5: Custom visualizations."""
    print("\n" + "=" * 60)
    print("Example 5: Custom Visualizations")
    print("=" * 60)

    code = """
from docs.comparison_study.visualization import ComparisonPlotter
import pickle

# Load results
with open("docs/comparison_study/outputs/results/complete_results.pkl", "rb") as f:
    data = pickle.load(f)

config = data["config"]
results = data["aggregated"]

# Create custom plotter
plotter = ComparisonPlotter(
    config=config['evaluation'],
    output_dir="outputs/custom_figures"
)

# Generate specific figures
figures = []

# 1. Batch effects comparison
fig_path = plotter.plot_batch_effects_comparison(results)
figures.append(fig_path)

# 2. Performance comparison
fig_path = plotter.plot_performance_comparison(results)
figures.append(fig_path)

# 3. Custom ranking plot
from docs.comparison_study.visualization import calculate_overall_scores
scores = calculate_overall_scores(results, config['evaluation'])
fig_path = plotter.plot_ranking_barplot(scores)
figures.append(fig_path)

print(f"Generated {len(figures)} custom figures")
"""

    print(code)


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Pipeline Comparison Runner - Usage Examples")
    print("=" * 60)

    example_cli_usage()
    example_python_api()
    example_custom_experiment()
    example_result_analysis()
    example_visualization()

    print("\n" + "=" * 60)
    print("Additional Resources")
    print("=" * 60)
    print("\nFor more information, see:")
    print("  - README.md: Complete usage guide")
    print("  - run_comparison.py --help: Command-line options")
    print("  - configs/: Configuration file examples")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
