# Plotting Module - Quick Start Guide

## File Location

```
studies/comparison_study/plotting.py
```

## Quick Overview

- **518 lines** (under 600-line target)
- **7 pure functions** (no classes)
- **Direct matplotlib calls** (no abstractions)
- **SciencePlots style** (publication quality)
- **Type hints** (Python 3.11+)

## Available Functions

### 1. plot_batch_effects()

Multi-metric batch effect comparison (4 metrics in 2x2 grid).

```python
from plotting import plot_batch_effects

results = {
    "ComBat": {
        "kbet_score": 0.85,
        "ilisi_score": 0.72,
        "clisi_score": 0.88,
        "asw_score": 0.15,
    },
    "Harmony": {
        "kbet_score": 0.82,
        "ilisi_score": 0.68,
        "clisi_score": 0.85,
        "asw_score": 0.18,
    },
}

plot_batch_effects(
    results_dict=results,
    metrics=["kbet_score", "ilisi_score", "clisi_score", "asw_score"],
    output_path="batch_effects.png",
)
```

### 2. plot_performance_comparison()

Execution time and memory usage comparison.

```python
from plotting import plot_performance_comparison

results = {
    "ComBat": {
        "execution_time": 2.5,
        "memory_usage": 128.5,
    },
    "Harmony": {
        "execution_time": 3.2,
        "memory_usage": 145.2,
    },
}

plot_performance_comparison(
    results_dict=results,
    output_path="performance.png",
)
```

### 3. plot_radar_chart()

Comprehensive metrics assessment (normalized to 0-1 scale).

```python
from plotting import plot_radar_chart

metrics = {
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

plot_radar_chart(
    metrics_dict=metrics,
    metrics=["kbet_score", "ilisi_score", "clisi_score", "biological_preservation"],
    output_path="radar.png",
)
```

### 4. plot_distribution_comparison()

Histogram and box plot comparison.

```python
from plotting import plot_distribution_comparison
import numpy as np

data = {
    "ComBat": np.random.randn(1000, 10),
    "Harmony": np.random.randn(1000, 10) * 0.8 + 0.2,
}

plot_distribution_comparison(
    data_dict=data,
    output_path="distribution.png",
)
```

### 5. plot_clustering_results()

PCA/UMAP visualization of clustering results.

```python
from plotting import plot_clustering_results
import numpy as np

X = np.random.randn(500, 20)  # High-dimensional data
labels = np.random.randint(0, 5, 500)  # Cluster labels

plot_clustering_results(
    X=X,
    labels=labels,
    method="PCA",
    output_path="clustering.png",
)
```

### 6. plot_umap_comparison()

UMAP before/after batch correction visualization.

```python
from plotting import plot_umap_comparison
import numpy as np

umap_before = np.random.randn(300, 2)
umap_after = np.random.randn(300, 2) * 0.5  # Better mixed
batch_labels = np.array([0]*100 + [1]*100 + [2]*100)

plot_umap_comparison(
    umap_before=umap_before,
    umap_after=umap_after,
    batch_labels=batch_labels,
    method_name="ComBat",
    output_path="umap_comparison.png",
)
```

### 7. plot_metrics_heatmap()

Heatmap of all metrics across methods.

```python
from plotting import plot_metrics_heatmap

results = {
    "ComBat": {
        "kbet_score": 0.85,
        "ilisi_score": 0.72,
        "clisi_score": 0.88,
        "execution_time": 2.5,
    },
    "Harmony": {
        "kbet_score": 0.82,
        "ilisi_score": 0.68,
        "clisi_score": 0.85,
        "execution_time": 3.2,
    },
}

plot_metrics_heatmap(
    results_dict=results,
    output_path="heatmap.png",
)
```

## Common Usage Patterns

### Pattern 1: Complete Comparison Study

```python
from plotting import (
    plot_batch_effects,
    plot_performance_comparison,
    plot_radar_chart,
    plot_metrics_heatmap,
)
from pathlib import Path

def generate_comparison_plots(results: dict, output_dir: Path):
    """Generate all plots for a comparison study."""
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_batch_effects(results, output_path=output_dir / "batch_effects.png")
    plot_performance_comparison(results, output_path=output_dir / "performance.png")
    plot_radar_chart(results, output_path=output_dir / "radar.png")
    plot_metrics_heatmap(results, output_path=output_dir / "heatmap.png")

# Usage
results = {...}  # Your comparison results
generate_comparison_plots(results, Path("outputs/my_comparison"))
```

### Pattern 2: Integration Method Evaluation

```python
from plotting import plot_batch_effects, plot_umap_comparison

def evaluate_integration_method(method_name: str, results: dict, umap_data: tuple):
    """Evaluate a single integration method."""
    output_dir = Path(f"outputs/{method_name}")

    # Plot metrics
    plot_batch_effects(
        {method_name: results},
        output_path=output_dir / "metrics.png",
    )

    # Plot UMAP
    umap_before, umap_after, batch_labels = umap_data
    plot_umap_comparison(
        umap_before,
        umap_after,
        batch_labels,
        method_name=method_name,
        output_path=output_dir / "umap.png",
    )
```

### Pattern 3: Performance Benchmarking

```python
from plotting import plot_performance_comparison

def benchmark_methods(methods: list[str], results: dict):
    """Benchmark multiple methods."""
    # Extract performance metrics
    perf_data = {
        method: {
            "execution_time": results[method]["time"],
            "memory_usage": results[method]["memory"],
        }
        for method in methods
    }

    plot_performance_comparison(
        perf_data,
        output_path="outputs/benchmark/performance.png",
    )
```

## Configuration

### Style Settings

The module automatically configures SciencePlots style with:
- DPI: 300 (publication quality)
- Font size: 10pt
- Style: "science" + "no-latex"
- Fallback: "seaborn-v0_8-whitegrid" (if SciencePlots unavailable)

### Color Scheme

Fixed colors for frameworks:

```python
COLORS = {
    "scptensor": "#1f77b4",    # Blue
    "scanpy": "#ff7f0e",       # Orange
    "scran": "#2ca02c",        # Green
    "seurat": "#d62728",       # Red
}
```

Customize by modifying the `COLORS` dict in `plotting.py`.

### Output Format

All functions:
- Return `Path` object to saved file
- Create parent directories automatically
- Save at 300 DPI
- Use `bbox_inches="tight"` for clean margins

## Testing

Run the test suite:

```bash
cd studies/comparison_study
uv run python test_visualization.py
```

Run the examples:

```bash
cd studies/comparison_study
uv run python plotting_examples.py
```

## Dependencies

```bash
# Required
uv pip install numpy matplotlib

# Optional (for SciencePlots style)
uv pip install scienceplots

# For clustering visualization
uv pip install scikit-learn
```

## Integration with Comparison Study

### Example: Integration Comparison

```python
# In run_comparison.py or your analysis script
from plotting import plot_batch_effects, plot_performance_comparison

# After running comparison
results = runner.run_integration_comparison()

# Generate plots
output_dir = Path("outputs/integration")
plot_batch_effects(results, output_path=output_dir / "batch_effects.png")
plot_performance_comparison(results, output_path=output_dir / "performance.png")
```

### Example: Framework Comparison

```python
# Compare ScpTensor vs Scanpy vs Scran
from plotting import plot_radar_chart

framework_metrics = {
    "ScpTensor": {
        "kbet_score": 0.85,
        "speed": 0.90,
        "memory_efficiency": 0.88,
        "accuracy": 0.92,
    },
    "Scanpy": {
        "kbet_score": 0.82,
        "speed": 0.75,
        "memory_efficiency": 0.70,
        "accuracy": 0.85,
    },
}

plot_radar_chart(
    framework_metrics,
    output_path="outputs/framework_comparison.png",
)
```

## Troubleshooting

### Issue: SciencePlots not available

**Solution**: Module falls back to seaborn style automatically.

### Issue: Plots look blurry

**Solution**: Ensure DPI is set to 300 (default). Check your matplotlib backend.

### Issue: Colors don't match frameworks

**Solution**: Add framework to `COLORS` dict in `plotting.py`:

```python
COLORS["my_framework"] = "#hexcolor"
```

### Issue: Too many methods on plot

**Solution**: Filter results before plotting:

```python
# Plot only top 5 methods
top_methods = sorted(results, key=lambda x: results[x]["score"])[:5]
filtered_results = {m: results[m] for m in top_methods}
plot_batch_effects(filtered_results, ...)
```

## Performance Tips

1. **Vectorize operations**: Use NumPy arrays instead of lists
2. **Reduce data size**: Downsample large datasets before plotting
3. **Use PNG format**: Default, good balance of quality and size
4. **Batch generation**: Generate all plots in one script to reuse matplotlib session

## File Structure

```
studies/comparison_study/
├── plotting.py                 # Main module (518 lines)
├── test_visualization.py       # Test suite
├── plotting_examples.py        # Usage examples
├── PLOTTING_MODULE_SUMMARY.md  # Detailed summary
└── PLOTTING_QUICK_START.md     # This file
```

## Summary

- **Pure functions**: No classes, no state
- **Simple API**: One function per plot type
- **Type hints**: Full type annotations
- **Tested**: All functions verified
- **Production-ready**: Used in comparison study

For more details, see `PLOTTING_MODULE_SUMMARY.md`.
