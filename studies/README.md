# Studies Module

Pipeline comparison utilities for ScpTensor.

## Quick Start

```bash
# Quick test
python -m studies.run_comparison --test --verbose

# Compare specific pipelines
python -m studies.run_comparison --pipelines classic batch_corrected --verbose

# Full comparison
python -m studies.run_comparison --full --verbose
```

## Available Pipelines

| Pipeline | Description |
|----------|-------------|
| `classic` | QC → Median norm → Log → KNN impute → PCA → K-means |
| `batch_corrected` | Classic + ComBat batch correction |
| `advanced` | MissForest impute + Harmony batch correction |
| `fast` | LLS impute, fewer PCA components |
| `conservative` | No imputation, no batch correction |

## Programmatic Usage

```python
from studies.pipelines.executor import run_pipeline, get_available_pipelines
from studies.evaluation.core import evaluate_pipeline, monitor_performance

# List available pipelines
print(get_available_pipelines())

# Run a pipeline
from scptensor import create_test_container
container = create_test_container()
result, log = run_pipeline(container, "classic", verbose=True)

# Evaluate results
metrics = evaluate_pipeline(container, result,
                            runtime=log["total_time"],
                            memory_peak=0.5)
print(metrics)
```

## Custom Pipelines

```python
from studies.pipelines.executor import run_pipeline

custom_steps = [
    ("qc", {}),
    ("norm_median", {}),
    ("log_transform", {}),
    ("pca", {"n_components": 30}),
]

result, log = run_pipeline(container, steps=custom_steps)
```

## Available Steps

| Step | Function |
|------|----------|
| `qc` | Quality control filtering |
| `norm_median` | Median normalization |
| `norm_mean` | Mean normalization |
| `log_transform` | Logarithmic transformation |
| `impute_knn` | K-nearest neighbors imputation |
| `impute_missforest` | MissForest imputation |
| `impute_lls` | Local least squares imputation |
| `batch_combat` | ComBat batch correction |
| `batch_harmony` | Harmony integration |
| `batch_mnn` | MNN batch correction |
| `pca` | PCA dimensionality reduction |
| `kmeans` | K-means clustering |
