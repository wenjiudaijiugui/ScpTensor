"""
Test script for visualization module.

This script creates synthetic evaluation results and tests the visualization
module to ensure all plots can be generated.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from docs.comparison_study.visualization import (  # noqa: E402
    ComparisonPlotter,
    ReportGenerator,
    calculate_overall_scores,
    generate_all_figures,
)


def create_synthetic_results() -> dict[str, dict[str, dict[str, float]]]:
    """
    Create synthetic evaluation results for testing.

    Returns
    -------
    dict
        Synthetic results matching the expected structure
    """
    datasets = ["small", "medium", "large"]
    pipelines = ["pipeline_a", "pipeline_b", "pipeline_c", "pipeline_d", "pipeline_e"]

    results: dict[str, dict[str, dict[str, float]]] = {}

    for dataset in datasets:
        for pipeline in pipelines:
            key = f"{dataset}_{pipeline}"

            # Generate varying scores based on pipeline characteristics
            if pipeline == "pipeline_a":  # Classic
                kbet = 0.65 if dataset == "small" else 0.50
                lisi = 0.70 if dataset == "small" else 0.55
                runtime = 60.0 if dataset == "small" else 300.0
                memory = 2.0 if dataset == "small" else 8.0
            elif pipeline == "pipeline_b":  # Batch correction
                kbet = 0.85 if dataset != "small" else 0.70
                lisi = 0.88 if dataset != "small" else 0.75
                runtime = 80.0 if dataset == "small" else 400.0
                memory = 2.5 if dataset == "small" else 10.0
            elif pipeline == "pipeline_c":  # Advanced
                kbet = 0.90 if dataset != "small" else 0.75
                lisi = 0.92 if dataset != "small" else 0.80
                runtime = 120.0 if dataset == "small" else 600.0
                memory = 3.0 if dataset == "small" else 15.0
            elif pipeline == "pipeline_d":  # Performance-optimized
                kbet = 0.75 if dataset != "small" else 0.60
                lisi = 0.78 if dataset != "small" else 0.65
                runtime = 40.0 if dataset == "small" else 150.0
                memory = 1.5 if dataset == "small" else 5.0
            else:  # pipeline_e - Conservative
                kbet = 0.60 if dataset == "small" else 0.45
                lisi = 0.65 if dataset == "small" else 0.50
                runtime = 70.0 if dataset == "small" else 350.0
                memory = 2.2 if dataset == "small" else 9.0

            results[key] = {
                "batch_effects": {
                    "kbet": kbet,
                    "lisi": lisi,
                    "mixing_entropy": 0.75 + (0.1 * pipelines.index(pipeline)),
                    "variance_ratio": 1.0 + (0.1 * (datasets.index(dataset) + 1)),
                },
                "performance": {
                    "runtime_seconds": runtime * (1 + datasets.index(dataset)),
                    "memory_gb": memory * (1 + datasets.index(dataset) * 0.5),
                },
                "distribution": {
                    "sparsity_change": 0.05 * (pipelines.index(pipeline) + 1),
                    "mean_change": 0.1 * (pipelines.index(pipeline) + 1),
                    "std_change": 0.08 * (pipelines.index(pipeline) + 1),
                    "cv_change": 0.12 * (pipelines.index(pipeline) + 1),
                },
                "structure": {
                    "pca_variance_cumulative": 0.70 + (0.02 * pipelines.index(pipeline)),
                    "nn_consistency": 0.85 - (0.02 * pipelines.index(pipeline)),
                    "distance_correlation": 0.90 - (0.01 * pipelines.index(pipeline)),
                    "global_structure": {"centroid_distance": 0.1 * pipelines.index(pipeline)},
                },
            }

    return results


def create_test_config() -> dict[str, dict[str, dict[str, str | float]]]:
    """
    Create test configuration.

    Returns
    -------
    dict
        Test configuration matching evaluation_config.yaml structure
    """
    return {
        "visualization": {
            "figure": {
                "dpi": 300,
                "format": "png",
                "style": ["science", "no-latex"],
                "figsize": [10, 8],
                "colors": {
                    "pipeline_a": "#1f77b4",
                    "pipeline_b": "#ff7f0e",
                    "pipeline_c": "#2ca02c",
                    "pipeline_d": "#d62728",
                    "pipeline_e": "#9467bd",
                },
            },
            "font": {
                "family": "Arial",
                "title_size": 16,
                "label_size": 12,
                "legend_size": 10,
            },
        },
        "scoring": {
            "weights": {
                "batch_effects": 0.25,
                "performance": 0.25,
                "distribution": 0.25,
                "structure": 0.25,
            },
            "grading": {
                "A": {"min_score": 80, "description": "Excellent"},
                "B": {"min_score": 60, "description": "Good"},
                "C": {"min_score": 0, "description": "Acceptable"},
            },
        },
        "report": {
            "metadata": {
                "title": "Pipeline Comparison Report",
                "authors": "ScpTensor Team",
                "version": "1.0.0",
                "date_format": "%Y-%m-%d",
            }
        },
    }


def main() -> int:
    """
    Run visualization tests.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure)
    """
    print("=" * 70)
    print("Testing Visualization Module")
    print("=" * 70)

    # Create test data
    print("\n1. Creating synthetic evaluation results...")
    results = create_synthetic_results()
    print(f"   ✓ Generated {len(results)} dataset-pipeline combinations")

    # Create test config
    print("\n2. Creating test configuration...")
    config = create_test_config()
    print("   ✓ Configuration loaded")

    # Test ComparisonPlotter
    print("\n3. Testing ComparisonPlotter...")
    plotter = ComparisonPlotter(config, output_dir="test_outputs/figures")
    print("   ✓ ComparisonPlotter initialized")

    # Test individual plots
    print("\n4. Testing individual plot generation...")

    try:
        path = plotter.plot_batch_effects_comparison(results)
        print(f"   ✓ Batch effects comparison: {path}")
    except Exception as e:
        print(f"   ✗ Batch effects comparison failed: {e}")
        return 1

    try:
        path = plotter.plot_performance_comparison(results)
        print(f"   ✓ Performance comparison: {path}")
    except Exception as e:
        print(f"   ✗ Performance comparison failed: {e}")
        return 1

    try:
        path = plotter.plot_distribution_comparison(results)
        print(f"   ✓ Distribution comparison: {path}")
    except Exception as e:
        print(f"   ✗ Distribution comparison failed: {e}")
        return 1

    try:
        path = plotter.plot_structure_preservation(results)
        print(f"   ✓ Structure preservation: {path}")
    except Exception as e:
        print(f"   ✗ Structure preservation failed: {e}")
        return 1

    try:
        path = plotter.plot_comprehensive_radar(results)
        print(f"   ✓ Comprehensive radar: {path}")
    except Exception as e:
        print(f"   ✗ Comprehensive radar failed: {e}")
        return 1

    # Test score calculation
    print("\n5. Testing score calculation...")
    try:
        scores = calculate_overall_scores(results, config)
        print("   ✓ Scores calculated:")
        for pipeline, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            grade = "A" if score >= 80 else "B" if score >= 60 else "C"
            print(f"      {pipeline}: {score:.1f}/100 (Grade {grade})")

        # Test ranking plot
        path = plotter.plot_ranking_barplot(scores)
        print(f"   ✓ Ranking barplot: {path}")
    except Exception as e:
        print(f"   ✗ Score calculation or ranking failed: {e}")
        return 1

    # Test generate_all_figures
    print("\n6. Testing generate_all_figures()...")
    try:
        figures = generate_all_figures(results, config, output_dir="test_outputs/figures/all")
        print(f"   ✓ Generated {len(figures)} figures:")
        for fig in figures:
            print(f"      - {fig}")
    except Exception as e:
        print(f"   ✗ generate_all_figures failed: {e}")
        return 1

    # Test ReportGenerator
    print("\n7. Testing ReportGenerator...")
    try:
        generator = ReportGenerator(config, output_dir="test_outputs")
        print("   ✓ ReportGenerator initialized")

        # Get list of generated figures
        import glob

        figure_files = glob.glob("test_outputs/figures/*.png")
        pdf_path = generator.generate_report(
            results, figure_files, save_path="test_outputs/report.pdf"
        )
        print(f"   ✓ Report generated: {pdf_path}")
        print("   Note: Markdown saved to test_outputs/report.md")
        print(
            "         Convert to PDF with: pandoc test_outputs/report.md -o test_outputs/report.pdf"
        )
    except Exception as e:
        print(f"   ✗ Report generation failed: {e}")
        return 1

    print("\n" + "=" * 70)
    print("All tests passed successfully!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - Figures: test_outputs/figures/")
    print("  - Report: test_outputs/report.md")
    print("\nTo view figures:")
    print("  - Open test_outputs/figures/ directory")
    print("\nTo convert report to PDF:")
    print("  - pandoc test_outputs/report.md -o test_outputs/report.pdf")

    return 0


if __name__ == "__main__":
    sys.exit(main())
