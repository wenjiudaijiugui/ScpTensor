"""Example usage of the simplified plotting module.

This script demonstrates how to use the plotting functions
with actual comparison study results.
"""

from pathlib import Path

from plotting import (
    plot_batch_effects,
    plot_clustering_results,
    plot_metrics_heatmap,
    plot_performance_comparison,
    plot_radar_chart,
    plot_umap_comparison,
)


def example_integration_comparison():
    """Example: Compare integration methods."""
    # Simulated results from integration comparison
    integration_results = {
        "ComBat": {
            "framework": "scptensor",
            "kbet_score": 0.85,
            "ilisi_score": 0.72,
            "clisi_score": 0.88,
            "asw_score": 0.15,
            "execution_time": 2.5,
            "memory_usage": 128.5,
        },
        "Harmony": {
            "framework": "scanpy",
            "kbet_score": 0.82,
            "ilisi_score": 0.68,
            "clisi_score": 0.85,
            "asw_score": 0.18,
            "execution_time": 3.2,
            "memory_usage": 145.2,
        },
        "MNN": {
            "framework": "scptensor",
            "kbet_score": 0.78,
            "ilisi_score": 0.65,
            "clisi_score": 0.82,
            "asw_score": 0.22,
            "execution_time": 4.1,
            "memory_usage": 156.8,
        },
        "Scanorama": {
            "framework": "scptensor",
            "kbet_score": 0.80,
            "ilisi_score": 0.70,
            "clisi_score": 0.86,
            "asw_score": 0.20,
            "execution_time": 5.5,
            "memory_usage": 178.3,
        },
    }

    output_dir = Path("outputs/integration_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate all plots
    plot_batch_effects(
        integration_results,
        metrics=["kbet_score", "ilisi_score", "clisi_score", "asw_score"],
        output_path=output_dir / "batch_effects.png",
    )

    plot_performance_comparison(
        integration_results,
        output_path=output_dir / "performance.png",
    )

    metrics_for_radar = {
        method: {
            "kbet_score": data["kbet_score"],
            "ilisi_score": data["ilisi_score"],
            "clisi_score": data["clisi_score"],
            "biological_preservation": 0.90 - 0.05 * idx,  # Simulated
        }
        for idx, (method, data) in enumerate(integration_results.items())
    }

    plot_radar_chart(
        metrics_for_radar,
        output_path=output_dir / "radar_chart.png",
    )

    plot_metrics_heatmap(
        integration_results,
        output_path=output_dir / "metrics_heatmap.png",
    )

    print(f"Integration comparison plots saved to {output_dir}")


def example_imputation_comparison():
    """Example: Compare imputation methods."""

    # Simulated results
    imputation_results = {
        "KNN-ScpTensor": {
            "framework": "scptensor",
            "mse": 0.15,
            "mae": 0.28,
            "correlation": 0.85,
            "execution_time": 3.2,
            "memory_usage": 145.6,
        },
        "KNN-Scanpy": {
            "framework": "scanpy",
            "mse": 0.18,
            "mae": 0.32,
            "correlation": 0.82,
            "execution_time": 4.5,
            "memory_usage": 167.2,
        },
        "MissForest": {
            "framework": "scptensor",
            "mse": 0.12,
            "mae": 0.24,
            "correlation": 0.88,
            "execution_time": 8.7,
            "memory_usage": 234.1,
        },
    }

    output_dir = Path("outputs/imputation_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    plot_performance_comparison(
        imputation_results,
        output_path=output_dir / "performance.png",
    )

    # Radar chart for imputation quality metrics
    quality_metrics = {
        method: {
            "mse": 1.0 - data["mse"],  # Invert MSE (lower is better)
            "mae": 1.0 - data["mae"],  # Invert MAE
            "correlation": data["correlation"],
            "speed": 1.0 / data["execution_time"],
        }
        for method, data in imputation_results.items()
    }

    plot_radar_chart(
        quality_metrics,
        metrics=["mse", "mae", "correlation", "speed"],
        output_path=output_dir / "quality_radar.png",
    )

    print(f"Imputation comparison plots saved to {output_dir}")


def example_umap_visualization():
    """Example: UMAP before/after batch correction."""
    import numpy as np

    # Simulated data
    n_samples = 300
    np.random.seed(42)

    # Before correction (clear batch separation)
    umap_before = np.vstack(
        [
            np.random.randn(100, 2) + [3, 3],  # Batch 0
            np.random.randn(100, 2) + [-3, -3],  # Batch 1
            np.random.randn(100, 2) + [3, -3],  # Batch 2
        ]
    )

    # After correction (well mixed)
    umap_after = np.random.randn(n_samples, 2) * 2

    batch_labels = np.array([0] * 100 + [1] * 100 + [2] * 100)

    output_dir = Path("outputs/umap_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_umap_comparison(
        umap_before,
        umap_after,
        batch_labels,
        method_name="ComBat",
        output_path=output_dir / "umap_before_after.png",
    )

    print(f"UMAP comparison plot saved to {output_dir}")


def example_clustering_visualization():
    """Example: Clustering results visualization."""
    import numpy as np

    # Simulated data with 5 clusters
    n_clusters = 5
    np.random.seed(42)

    # Generate clustered data
    x = np.vstack(
        [np.random.randn(100, 20) + np.random.randn(1, 20) * 2 for _ in range(n_clusters)]
    )

    labels = np.array([i for i in range(n_clusters) for _ in range(100)])

    output_dir = Path("outputs/clustering")
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_clustering_results(
        x,
        labels,
        method="PCA",
        output_path=output_dir / "clustering_pca.png",
    )

    print(f"Clustering plot saved to {output_dir}")


if __name__ == "__main__":
    print("Generating example plots...\n")

    print("1. Integration comparison...")
    example_integration_comparison()

    print("\n2. Imputation comparison...")
    example_imputation_comparison()

    print("\n3. UMAP visualization...")
    example_umap_visualization()

    print("\n4. Clustering visualization...")
    example_clustering_visualization()

    print("\n" + "=" * 60)
    print("All example plots generated successfully!")
    print("Check the outputs/ directory for results.")
    print("=" * 60)
