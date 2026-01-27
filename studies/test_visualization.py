"""Quick test for plotting module."""

import numpy as np
from plotting import (
    plot_batch_effects,
    plot_clustering_results,
    plot_distribution_comparison,
    plot_metrics_heatmap,
    plot_performance_comparison,
    plot_radar_chart,
    plot_umap_comparison,
)

# Create dummy data
np.random.seed(42)
n_samples = 200

# Test data for batch effects
results_dict = {
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
}

# Test batch effects plot
print("Testing batch effects plot...")
plot_batch_effects(results_dict, output_path="test_batch_effects.png")

# Test performance comparison
print("Testing performance comparison...")
plot_performance_comparison(results_dict, output_path="test_performance.png")

# Test radar chart
print("Testing radar chart...")
metrics_data = {
    "ComBat": {
        "kbet_score": 0.85,
        "ilisi_score": 0.72,
        "clisi_score": 0.88,
        "biological_preservation": 0.90,
    },
    "Harmony": {
        "kbet_score": 0.82,
        "ilisi_score": 0.68,
        "clisi_score": 0.85,
        "biological_preservation": 0.87,
    },
}
plot_radar_chart(metrics_data, output_path="test_radar.png")

# Test distribution comparison
print("Testing distribution comparison...")
data_dict = {
    "ComBat": np.random.randn(n_samples, 10),
    "Harmony": np.random.randn(n_samples, 10) * 0.8 + 0.2,
}
plot_distribution_comparison(data_dict, output_path="test_distribution.png")

# Test clustering results
print("Testing clustering results...")
X = np.random.randn(n_samples, 20)
labels = np.random.randint(0, 4, n_samples)
plot_clustering_results(X, labels, output_path="test_clustering.png")

# Test UMAP comparison
print("Testing UMAP comparison...")
umap_before = np.random.randn(n_samples, 2)
batch_labels = np.random.randint(0, 3, n_samples)
umap_after = umap_before + np.random.randn(n_samples, 2) * 0.1
plot_umap_comparison(
    umap_before,
    umap_after,
    batch_labels,
    method_name="ComBat",
    output_path="test_umap_comparison.png",
)

# Test metrics heatmap
print("Testing metrics heatmap...")
plot_metrics_heatmap(results_dict, output_path="test_heatmap.png")

print("\nAll tests completed! Check the generated PNG files:")
print("  - test_batch_effects.png")
print("  - test_performance.png")
print("  - test_radar.png")
print("  - test_distribution.png")
print("  - test_clustering.png")
print("  - test_umap_comparison.png")
print("  - test_heatmap.png")
