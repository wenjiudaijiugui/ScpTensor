#!/usr/bin/env python3
"""
Efficient benchmark workflow example for ScpTensor.

Demonstrates:
1. Method comparison across different normalization methods
2. Parameter optimization for best-performing method
3. Comprehensive visualization and reporting
4. Performance optimization considerations
"""

import time
import warnings
import numpy as np
from pathlib import Path
from scptensor.benchmark import (
    BenchmarkSuite, SyntheticDataset, create_method_configs,
    create_normalization_parameter_grids, ResultsVisualizer
)
from scptensor.core import MatrixOps

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def run_efficient_benchmark():
    """
    Run an efficient benchmark workflow with optimization considerations.

    Performance optimizations:
    1. Pre-generate datasets to avoid repeated computation
    2. Use parameter grids with reasonable sizes
    3. Cache intermediate results
    4. Generate comprehensive report at the end
    """

    print("🚀 Starting Efficient ScpTensor Benchmark Workflow")
    print("=" * 60)

    # Timing the entire workflow
    workflow_start = time.time()

    # Step 1: Generate datasets efficiently (reuse for multiple comparisons)
    print("\n📊 Step 1: Generating synthetic datasets...")
    dataset_start = time.time()

    # Create varied datasets for robust benchmarking
    datasets = [
        SyntheticDataset(
            n_samples=50,
            n_features=200,
            n_groups=2,
            n_batches=2,
            missing_rate=0.2,
            signal_to_noise_ratio=3.0,
            random_seed=42
        ).generate(),
        SyntheticDataset(
            n_samples=100,
            n_features=500,
            n_groups=3,
            n_batches=3,
            missing_rate=0.3,
            signal_to_noise_ratio=2.0,
            random_seed=123
        ).generate()
    ]

    dataset_time = time.time() - dataset_start
    print(f"   ✅ Generated {len(datasets)} datasets in {dataset_time:.2f} seconds")

    # Step 2: Configure methods with optimized parameter grids
    print("\n🔧 Step 2: Configuring methods and parameter grids...")
    config_start = time.time()

    # Get predefined method configurations
    method_configs = create_method_configs()

    # Limit to normalization methods for demonstration
    normalization_methods = {
        name: config for name, config in method_configs.items()
        if 'normalization' in name or name in ['tmm_normalization', 'sample_median_normalization']
    }

    # Create focused parameter grids (not too large for demo)
    parameter_grids = {
        'tmm_normalization': {
            'trim_ratio': [0.1, 0.3, 0.5],  # Reduced from 5 to 3 values
        },
        'upper_quartile_normalization': {
            'percentile': [0.7, 0.75, 0.8],  # Reduced range
        }
        # Other methods use default parameters (no grid)
    }

    config_time = time.time() - config_start
    print(f"   ✅ Configured {len(normalization_methods)} methods in {config_time:.2f} seconds")

    # Step 3: Initialize benchmark suite
    print("\n⚡ Step 3: Initializing benchmark suite...")
    suite = BenchmarkSuite(
        methods=normalization_methods,
        datasets=datasets,
        parameter_grids=parameter_grids,
        random_seed=42
    )
    print("   ✅ Benchmark suite initialized")

    # Step 4: Run method comparison (efficient mode)
    print("\n🏁 Step 4: Running method comparison...")
    comparison_start = time.time()

    # Run comparison without parameter optimization for speed
    comparison_results = suite.run_method_comparison(
        optimize_params=False,  # Skip optimization for faster demo
        methods=['tmm_normalization', 'sample_median_normalization',
                'global_median_normalization', 'upper_quartile_normalization']
    )

    comparison_time = time.time() - comparison_start
    print(f"   ✅ Method comparison completed in {comparison_time:.2f} seconds")
    print(f"   📈 Results: {len(comparison_results.runs)} total runs")

    # Step 5: Parameter optimization for best method (limited scope)
    print("\n🎯 Step 5: Optimizing parameters for best method...")
    optimization_start = time.time()

    # Find best method from comparison
    method_comparison_df = comparison_results.get_method_comparison()
    if not method_comparison_df.empty:
        best_method = method_comparison_df['mean_group_separation'].idxmax()
        print(f"   🏆 Best method: {best_method}")

        # Run limited parameter optimization
        optimization_results = suite.run_parameter_optimization(
            method_name=best_method
        )

        optimization_time = time.time() - optimization_start
        print(f"   ✅ Parameter optimization completed in {optimization_time:.2f} seconds")
        print(f"   📊 Tested {len(optimization_results.runs)} parameter combinations")
    else:
        optimization_results = None
        optimization_time = 0

    # Step 6: Generate comprehensive visualization report
    print("\n📊 Step 6: Generating visualization report...")
    viz_start = time.time()

    visualizer = ResultsVisualizer(style="science")

    # Create output directory
    output_dir = Path("benchmark_example_results")
    output_dir.mkdir(exist_ok=True)

    # Generate comprehensive report
    plot_paths = visualizer.create_comprehensive_report(
        comparison_results,
        output_dir=str(output_dir),
        create_interactive=True
    )

    viz_time = time.time() - viz_start
    print(f"   ✅ Visualizations generated in {viz_time:.2f} seconds")

    # Step 7: Performance analysis
    print("\n📈 Step 7: Performance analysis...")
    workflow_time = time.time() - workflow_start

    # Calculate efficiency metrics
    total_runs = len(comparison_results.runs)
    if optimization_results:
        total_runs += len(optimization_results.runs)

    runs_per_second = total_runs / workflow_time
    avg_memory_usage = np.mean([
        run.computational_scores.memory_usage_mb
        for run in comparison_results.runs
    ])

    # Step 8: Summary report
    print(f"\n🎉 BENCHMARK COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"⏱️  Total workflow time: {workflow_time:.2f} seconds")
    print(f"📊 Total benchmark runs: {total_runs}")
    print(f"⚡ Average speed: {runs_per_second:.2f} runs/second")
    print(f"💾 Average memory usage: {avg_memory_usage:.1f} MB")
    print(f"📁 Results saved to: {output_dir}")

    # Show best performing methods
    if not method_comparison_df.empty:
        print(f"\n🏆 TOP 3 METHODS (by biological performance):")
        top_3 = method_comparison_df.nlargest(3, 'mean_group_separation')
        for i, (method, row) in enumerate(top_3.iterrows(), 1):
            print(f"   {i}. {method}:")
            print(f"      Group Separation: {row['mean_group_separation']:.3f}")
            print(f"      Runtime: {row['mean_runtime_seconds']:.2f}s")
            print(f"      Data Recovery: {row['mean_data_recovery_rate']:.3f}")

    # Show generated files
    print(f"\n📁 Generated Files:")
    for plot_type, file_path in plot_paths.items():
        file_name = Path(file_path).name
        file_size = Path(file_path).stat().st_size / 1024  # KB
        print(f"   {plot_type}: {file_name} ({file_size:.1f} KB)")

    # Performance recommendations
    print(f"\n💡 PERFORMANCE RECOMMENDATIONS:")
    if workflow_time > 60:
        print("   ⚠️  Workflow took >1min. Consider:")
        print("      - Reducing parameter grid size")
        print("      - Using fewer datasets")
        print("      - Running methods in parallel")

    if runs_per_second < 0.5:
        print("   ⚠️  Low efficiency. Consider:")
        print("      - Caching intermediate results")
        print("      - Using optimized method implementations")
        print("      - Reducing dataset size for development")

    return comparison_results, optimization_results, plot_paths


def run_parameter_sensitivity_analysis():
    """
    Demonstrate parameter sensitivity analysis for a specific method.

    This function shows how to analyze how different parameters affect
    method performance, which is crucial for method selection.
    """
    print("\n🔍 PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 40)

    # Create test dataset
    dataset = SyntheticDataset(
        n_samples=50,
        n_features=200,
        n_groups=2,
        n_batches=2,
        missing_rate=0.25,
        random_seed=42
    ).generate()

    # Setup benchmark suite
    method_configs = create_method_configs()
    suite = BenchmarkSuite(
        methods={'tmm_normalization': method_configs['tmm_normalization']},
        datasets=[dataset],
        parameter_grids={
            'tmm_normalization': {
                'trim_ratio': [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
            }
        },
        random_seed=42
    )

    # Run parameter optimization
    results = suite.run_parameter_optimization(
        method_name='tmm_normalization',
        n_trials=20
    )

    # Analyze parameter sensitivity
    sensitivity_df = results.get_parameter_sensitivity('tmm_normalization')

    print(f"📊 Parameter Sensitivity Results:")
    print(f"   Best trim_ratio: {sensitivity_df.loc[sensitivity_df['group_separation'].idxmax(), 'trim_ratio']}")
    print(f"   Performance range: {sensitivity_df['group_separation'].min():.3f} - {sensitivity_df['group_separation'].max():.3f}")

    return results


if __name__ == "__main__":
    """
    Main execution with efficiency monitoring.
    """

    print("🧪 ScpTensor Benchmark Example")
    print("Focused on efficiency and comprehensive analysis")
    print()

    # Run main benchmark workflow
    try:
        comparison_results, optimization_results, plot_paths = run_efficient_benchmark()

        # Optional: Run detailed parameter sensitivity analysis
        # Note: Skip interactive input in automated environments
        print("\n🔍 Skipping parameter sensitivity analysis (run manually if needed)")
        # sensitivity_results = run_parameter_sensitivity_analysis()

        print("\n✅ All benchmarks completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during benchmark: {e}")
        print("Please check the traceback above for details.")
        raise