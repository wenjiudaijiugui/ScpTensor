# Quick Start Guide - Updated run_comparison.py

**File:** `/home/shenshang/projects/ScpTensor/studies/comparison_study/run_comparison.py`

---

## TL;DR

The `run_comparison.py` file has been updated to use streamlined modules. **All features preserved, 34% less code.**

---

## Quick Commands

```bash
# Navigate to directory
cd /home/shenshang/projects/ScpTensor/studies/comparison_study

# Quick test (small dataset, ~1 minute)
uv run python run_comparison.py --test --verbose

# Medium dataset (default, ~5 minutes)
uv run python run_comparison.py --verbose

# Full experiment (all datasets, 3 repeats, ~30 minutes)
uv run python run_comparison.py --full --repeats 3 --verbose

# Custom output directory
uv run python run_comparison.py --output /tmp/results --verbose
```

---

## What Changed?

### Before (681 lines)
- Duplicate code in multiple places
- Complex caching logic
- Manual pipeline execution
- Hard to maintain

### After (448 lines)
- ✅ Uses streamlined modules
- ✅ No code duplication
- ✅ Clean, modular design
- ✅ Easy to extend

---

## New Module Structure

```
run_comparison.py (main runner)
├── data_generation.py (dataset generation)
├── metrics.py (evaluation metrics)
├── plotting.py (visualization)
└── comparison_engine.py (pipeline comparison)
```

---

## Key Functions

### In `run_comparison.py`:
- `parse_arguments()` - Command-line interface
- `load_config()` - Load YAML config
- `setup_output_directory()` - Create output folders
- `load_datasets()` - Generate datasets (simplified!)
- `initialize_pipelines()` - Setup pipeline instances
- `main()` - Main execution flow

### In `data_generation.py`:
- `generate_small_dataset()` - 1K samples, 1K features
- `generate_medium_dataset()` - 5K samples, 1.5K features, 5 batches
- `generate_large_dataset()` - 20K samples, 2K features, 10 batches

### In `metrics.py`:
- `calculate_kbet()` - Batch mixing score
- `calculate_ilisi()` - Inverse LISI (batch mixing)
- `calculate_clisi()` - Cell type LISI
- `calculate_asw()` - Average silhouette width
- `calculate_all_metrics()` - Run all metrics

### In `plotting.py`:
- `plot_batch_effects()` - Bar charts for metrics
- `plot_radar_chart()` - Radar chart comparison
- `plot_performance_comparison()` - Time/memory comparison
- `plot_umap_comparison()` - UMAP before/after
- `plot_clustering_results()` - Clustering visualization
- `plot_metrics_heatmap()` - Heatmap of metrics

### In `comparison_engine.py`:
- `compare_pipelines()` - Run all pipelines on all datasets
- `generate_comparison_report()` - Create Markdown report
- `rank_methods()` - Rank methods by metric

---

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--full` | Run all datasets, multiple repeats | False |
| `--test` | Quick test (small dataset) | False |
| `--config PATH` | Custom config file | None |
| `--output DIR` | Output directory | `docs/comparison_study/outputs` |
| `--no-cache` | Regenerate datasets | False |
| `--repeats N` | Number of repeats | 3 |
| `--verbose` | Verbose output | False |

---

## Output Structure

```
outputs/
├── results/
│   └── complete_results.pkl          # All results (pickle)
├── figures/
│   ├── small_batch_effects.png        # Batch effect comparison
│   ├── medium_batch_effects.png
│   ├── large_batch_effects.png
│   ├── performance.png                # Time/memory comparison
│   └── radar_chart.png                # Radar chart (first dataset)
└── comparison_report.md               # Markdown report
```

---

## Examples

### Example 1: Quick Test
```bash
# Generate small dataset, run all pipelines once
uv run python run_comparison.py --test --verbose

# Expected output:
# - 1 dataset (small: 1K samples)
# - 5 pipelines
# - 1 repeat
# - Runtime: ~1 minute
```

### Example 2: Default Mode
```bash
# Generate medium dataset, run all pipelines once
uv run python run_comparison.py --verbose

# Expected output:
# - 1 dataset (medium: 5K samples, 5 batches)
# - 5 pipelines
# - 1 repeat
# - Runtime: ~5 minutes
```

### Example 3: Full Experiment
```bash
# Generate all datasets, run all pipelines 3 times
uv run python run_comparison.py --full --repeats 3 --verbose

# Expected output:
# - 3 datasets (small, medium, large)
# - 5 pipelines
# - 3 repeats
# - Runtime: ~30 minutes
```

### Example 4: Custom Output
```bash
# Save results to custom directory
uv run python run_comparison.py --output /tmp/my_results --verbose

# Results saved to:
# /tmp/my_results/results/complete_results.pkl
# /tmp/my_results/figures/*.png
# /tmp/my_results/comparison_report.md
```

---

## Troubleshooting

### Issue: Import Error
```
ImportError: attempted relative import with no known parent package
```

**Solution:** The imports have been fixed. Make sure you're using:
```bash
cd /home/shenshang/projects/ScpTensor/studies/comparison_study
uv run python run_comparison.py --verbose
```

### Issue: Module Not Found
```
ModuleNotFoundError: No module named 'studies.comparison_study.pipelines'
```

**Solution:** Make sure the pipelines module exists:
```bash
ls studies/comparison_study/pipelines/
# Should see: __init__.py, pipeline_a.py, etc.
```

### Issue: Dataset Generation Fails
```
ValueError: cannot reshape array of size X into shape Y
```

**Solution:** Check dataset parameters in `data_generation.py`:
```python
# generate_small_dataset()
n_samples=1000
n_features=1000
```

---

## Verification

Run the verification script to check everything works:

```bash
cd /home/shenshang/projects/ScpTensor/studies/comparison_study
uv run python verify_runner_update.py

# Expected output:
# [1/5] Testing imports...
#   ✓ data_generation imports successful
#   ✓ metrics imports successful
#   ✓ plotting imports successful
#   ✓ comparison_engine imports successful
# [2/5] Checking run_comparison.py syntax...
#   ✓ Syntax check passed
# [3/5] Importing run_comparison module...
#   ✓ Module import successful
# [4/5] Checking key functions...
#   ✓ All 6 required functions present
# [5/5] Testing data generation...
#   ✓ Generated test dataset: 1000 samples, 1 assay(s)
# ✓ All checks passed!
```

---

## Code Snippets

### Generate Dataset Manually
```python
from studies.comparison_study.data_generation import generate_small_dataset

# Generate small dataset
container = generate_small_dataset(seed=42)

# Access data
print(container.n_samples)  # 1000
print(container.obs)  # Sample metadata
print(container.assays['proteins'].layers['raw'].X)  # Data matrix
```

### Calculate Metrics Manually
```python
from studies.comparison_study.metrics import calculate_all_metrics
import numpy as np

# Generate some data
X = np.random.randn(1000, 100)
batch_labels = np.random.randint(0, 5, 1000)
cell_labels = np.random.randint(0, 8, 1000)

# Calculate metrics
metrics = calculate_all_metrics(X, batch_labels, cell_labels)

print(metrics)
# {'kbet': 0.123, 'ilisi': 0.456, 'clisi': 0.789, 'asw': 0.234}
```

### Create Plot Manually
```python
from studies.comparison_study.plotting import plot_batch_effects

# Results from comparison
results = {
    'PipelineA': {'kbet_score': 0.8, 'ilisi_score': 0.7},
    'PipelineB': {'kbet_score': 0.6, 'ilisi_score': 0.9},
}

# Create plot
fig_path = plot_batch_effects(
    results_dict=results,
    output_path='batch_effects.png'
)

print(f"Plot saved to: {fig_path}")
```

---

## Summary

✅ **Updated** - `run_comparison.py` now uses streamlined modules
✅ **Verified** - All checks passed
✅ **Simplified** - 34% less code, no duplication
✅ **Ready** - Use it now!

---

**File:** `/home/shenshang/projects/ScpTensor/studies/comparison_study/run_comparison.py`
**Status:** ✅ UPDATED & VERIFIED
**Date:** 2026-01-26
