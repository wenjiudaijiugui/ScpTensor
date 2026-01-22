# ScpTensor Competitor Benchmark Documentation

**Version:** 0.1.0-alpha
**Last Updated:** 2025-01-14

---

## Overview

This document describes the competitor benchmark framework for ScpTensor, which compares performance and accuracy against established data analysis tools.

## Purpose

The competitor benchmarks serve to:

1. **Validate correctness** - Ensure ScpTensor produces results consistent with established tools
2. **Measure performance** - Quantify runtime and memory efficiency
3. **Identify advantages** - Highlight where ScpTensor provides unique value
4. **Guide optimization** - Find areas where ScpTensor can be improved

## Competitors

ScpTensor is compared against the following tools:

| Tool | Purpose | Website |
|------|---------|---------|
| **numpy** | Numerical computing foundation | https://numpy.org |
| **scipy** | Scientific computing algorithms | https://scipy.org |
| **scikit-learn** | Machine learning algorithms | https://scikit-learn.org |
| **umap-learn** | UMAP dimensionality reduction | https://umap-learn.readthedocs.io |
| **scanpy** | Single-cell analysis (style patterns) | https://scanpy.readthedocs.io |

## Benchmarked Operations

### 1. Normalization

| Operation | ScpTensor | Competitor |
|-----------|-----------|------------|
| Log transform | `scptensor.normalization.log_normalize` | `numpy.log` |
| Z-score | `scptensor.normalization.zscore` | `sklearn.preprocessing.StandardScaler` |
| Total count | `ScanpyStyleOps.normalize_total` | `scanpy.pp.normalize_total` |
| Log1p | `ScanpyStyleOps.log1p` | `scanpy.pp.log1p` |
| Scale | `ScanpyStyleOps.scale` | `scanpy.pp.scale` |

### 2. Imputation

| Operation | ScpTensor | Competitor |
|-----------|-----------|------------|
| KNN imputation | `scptensor.impute.knn` | `sklearn.impute.KNNImputer` |
| SVD imputation | `scptensor.impute.svd_impute` | `scipy.linalg.svd` |
| Mean imputation | `numpy_mean` | `numpy.mean` |

### 3. Dimensionality Reduction

| Operation | ScpTensor | Competitor |
|-----------|-----------|------------|
| PCA | `scptensor.dim_reduction.pca` | `sklearn.decomposition.PCA` |
| Sparse PCA | `scptensor.dim_reduction.pca` (sparse mode) | `sklearn.decomposition.TruncatedSVD` |
| UMAP | `scptensor.dim_reduction.umap` | `umap-learn`, `scanpy_umap` |

### 4. Clustering

| Operation | ScpTensor | Competitor |
|-----------|-----------|------------|
| K-means | `scptensor.cluster.kmeans` | `sklearn.cluster.KMeans` |
| Agglomerative | (planned) | `sklearn.cluster.AgglomerativeClustering` |

## Metrics

### Performance Metrics

- **Runtime:** Execution time in milliseconds
- **Memory:** Peak memory usage in MB
- **Speedup Factor:** `competitor_time / scptensor_time`
  - > 1.0: ScpTensor is faster
  - < 1.0: Competitor is faster

### Accuracy Metrics

- **Correlation:** Pearson correlation between ScpTensor and competitor outputs
  - 1.0: Identical results
  - > 0.95: Excellent agreement
  - > 0.90: Good agreement

## Usage

### Command Line Interface

```bash
# Run full benchmark suite
python -m scptensor.benchmark.run_competitor_benchmark

# Run quick benchmark (smaller datasets)
python -m scptensor.benchmark.run_competitor_benchmark --quick

# Specify output directory
python -m scptensor.benchmark.run_competitor_benchmark --output-dir my_results/

# Run specific operations only
python -m scptensor.benchmark.run_competitor_benchmark --operations pca knn_imputation

# Custom dataset size
python -m scptensor.benchmark.run_competitor_benchmark --dataset-sizes 200 1000

# Run multiple repeats for more reliable measurements
python -m scptensor.benchmark.run_competitor_benchmark --n-repeats 3

# Skip generating plots (results only)
python -m scptensor.benchmark.run_competitor_benchmark --no-plots

# Enable verbose output
python -m scptensor.benchmark.run_competitor_benchmark --verbose
```

### Programmatic Usage

```python
from scptensor.benchmark import (
    CompetitorBenchmarkSuite,
    SyntheticDataset,
)

# Create benchmark suite
suite = CompetitorBenchmarkSuite(output_dir="results")

# Generate test data
datasets = [
    SyntheticDataset(n_samples=100, n_features=500, ...).generate()
]

# Run benchmarks
results = suite.run_all_benchmarks(
    datasets=datasets,
    operations=["log_normalization", "knn_imputation", "pca"]
)

# Print summary
suite.print_summary()

# Save results
suite.save_results()
```

### Visualization

```python
from scptensor.benchmark.competitor_viz import CompetitorResultVisualizer

# Load results
viz = CompetitorResultVisualizer("results/competitor_benchmark_results.json")

# Create all plots
plots = viz.create_all_plots(output_dir="plots")

# Or create individual plots
viz.plot_speedup_comparison("plots/speedup.png")
viz.plot_runtime_comparison("plots/runtime.png")
```

## File Structure

```
scptensor/benchmark/
├── competitor_benchmark.py    # Reference implementations
├── competitor_suite.py        # Benchmark orchestration
├── competitor_viz.py          # Visualization utilities
├── core.py                    # Core benchmark classes
├── metrics.py                 # Metrics computation
├── synthetic_data.py          # Test data generation
└── ...

scripts/
└── run_competitor_benchmark.py  # Main benchmark runner
```

## Interpreting Results

### Speedup Factor

```
speedup_factor = competitor_time / scptensor_time
```

- **speedup > 1.2**: ScpTensor is significantly faster (20%+ improvement)
- **0.8 < speedup < 1.2**: Similar performance (within 20%)
- **speedup < 0.8**: Competitor is significantly faster

### Memory Ratio

```
memory_ratio = scptensor_memory / competitor_memory
```

- **ratio < 0.8**: ScpTensor uses 20%+ less memory
- **0.8 < ratio < 1.2**: Similar memory usage
- **ratio > 1.2**: ScpTensor uses more memory

### Accuracy Correlation

High correlation (>0.95) indicates that ScpTensor produces results consistent with established tools. Lower correlations may indicate:

- Different algorithm implementations
- Numerical precision differences
- Different handling of edge cases

## Example Results

Below is example output from the benchmark suite:

```
======================================================================
COMPETITOR BENCHMARK SUMMARY
======================================================================

Operation: log_normalization
  Comparisons: 3
  Mean Speedup: 1.234 (+/- 0.156)
  Mean Memory Ratio: 0.987
  Mean Accuracy: 1.000
  Winner: SCPTENSOR

Operation: knn_imputation
  Comparisons: 3
  Mean Speedup: 0.876 (+/- 0.089)
  Mean Memory Ratio: 1.056
  Mean Accuracy: 0.998
  Winner: competitor

Operation: pca
  Comparisons: 3
  Mean Speedup: 1.002 (+/- 0.023)
  Mean Memory Ratio: 1.001
  Mean Accuracy: 0.999
  Winner: mixed

======================================================================
```

## Expected Findings

Based on design priorities:

### Where ScpTensor Should Excel

1. **Mask-aware operations** - ScpTensor's mask tracking system avoids data copying
2. **Sparse data handling** - Optimized for sparse protein matrices
3. **Provenance tracking** - Built-in audit trail adds value
4. **Type safety** - Full type annotations catch errors early

### Where Competitors May Excel

1. **Mature implementations** - Tools like scikit-learn have years of optimization
2. **C extensions** - Some operations use compiled code
3. **Specialized algorithms** - Dedicated implementations may be more optimized

## Extending Benchmarks

To add a new competitor:

1. Add implementation class to `competitor_benchmark.py`
2. Register in `COMPETITOR_REGISTRY`
3. Add benchmark method to `CompetitorBenchmarkSuite`
4. Update this documentation

Example:

```python
class MyToolNormalize:
    """My tool's normalization implementation."""

    name = "mytool_normalize"

    @staticmethod
    def run(X: np.ndarray, M: np.ndarray | None = None) -> tuple[np.ndarray, float, float]:
        """Run normalization using my tool."""
        # Implementation here
        return result, runtime, memory

# Register
COMPETITOR_REGISTRY["mytool_normalize"] = MyToolNormalize
```

## Troubleshooting

### Benchmark fails to run

- Ensure all dependencies are installed: `uv pip install -e ".[dev]"`
- Check that test datasets can be generated
- Verify output directory is writable

### Results show low accuracy

- Check that both implementations use the same algorithm
- Verify NaN handling is consistent
- Ensure parameter values are equivalent

### Memory usage is high

- Reduce dataset size for testing
- Check for memory leaks in implementations
- Use sparse matrices for large datasets

## References

- NumPy: https://numpy.org/doc/stable/
- SciPy: https://docs.scipy.org/doc/scipy/
- scikit-learn: https://scikit-learn.org/stable/documentation.html
- Scanpy: https://scanpy.readthedocs.io/
