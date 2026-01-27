# Metrics Module - Quick Start Guide

## Import
```python
from studies.comparison_study.metrics import *
```

## Basic Usage
```python
import numpy as np

# Prepare data
X = np.random.randn(100, 20)  # cells × features
batch_labels = np.random.randint(0, 3, 100)  # batch assignments
cell_labels = np.random.randint(0, 5, 100)  # cell types

# Calculate all metrics
metrics = calculate_all_metrics(X, batch_labels, cell_labels)
```

## Available Functions

### Batch Correction
- `calculate_kbet(X, batch_labels, k=25)` → Higher is better (max=1.0)
- `calculate_ilisi(X, batch_labels, k=20)` → Higher = better mixing

### Biological Preservation
- `calculate_clisi(X, cell_labels, k=20)` → Higher = better separation
- `calculate_asw(X, labels)` → Range [-1, 1], higher is better

### Accuracy
- `calculate_mse(X1, X2)` → Mean squared error
- `calculate_mae(X1, X2)` → Mean absolute error
- `calculate_correlation(X1, X2)` → Pearson correlation

### Performance
- `measure_runtime(func, *args)` → Execution time
- `measure_memory_usage(func, *args)` → Peak memory (MB)

### Batch Operations
- `calculate_all_metrics(X, batch, cell, k=25)` → All metrics at once
- `calculate_integration_metrics(X_orig, X_corr, batch, cell)` → Before/after comparison

## File Location
```
/home/shenshang/projects/ScpTensor/studies/comparison_study/metrics.py
```

## Stats
- **169 lines** (66% under 500-line target)
- **78.9% code reduction** from original
- **11 pure functions**
- **100% type hints**
- **Zero new dependencies**

## Full Documentation
See `METRICS_MODULE.md` for detailed API reference and examples.
