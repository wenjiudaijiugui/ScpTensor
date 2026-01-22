# Evaluation Metrics Module

Comprehensive evaluation metrics for single-cell proteomics analysis pipeline comparison.

## Overview

This module provides four dimensions of evaluation metrics:

1. **Batch Effect Removal** - Assesses effectiveness of batch correction
2. **Computational Performance** - Measures runtime and memory efficiency
3. **Data Distribution Changes** - Tracks statistical properties
4. **Data Structure Preservation** - Evaluates local and global structure preservation

## Installation

The module is part of the ScpTensor comparison study framework.

```python
import sys
sys.path.insert(0, 'docs/comparison_study')

from evaluation import PipelineEvaluator
```

## Quick Start

### Basic Usage

```python
from evaluation import PipelineEvaluator

# Configure evaluation
config = {
    'batch_effects': {'enabled': True, 'kbet': {'k': 25}},
    'performance': {'enabled': True},
    'distribution': {'enabled': True},
    'structure': {'enabled': True}
}

# Create evaluator
evaluator = PipelineEvaluator(config)

# Run evaluation
results = evaluator.evaluate(
    original_container=original_data,
    result_container=processed_data,
    runtime=120.5,
    memory_peak=4.2,
    pipeline_name="scptensor",
    dataset_name="test_dataset"
)

# Get summary
summary = evaluator.get_summary()
print(summary)
```

### Performance Monitoring

```python
from evaluation import monitor_performance

with monitor_performance() as perf:
    # Run your pipeline
    result = run_pipeline(data)

print(f"Runtime: {perf['runtime']:.2f}s")
print(f"Peak memory: {perf['memory_peak']:.2f} GB")
```

## Metrics Reference

### Batch Effect Metrics

#### kBET (k-nearest neighbour batch effect test)
Measures local batch mixing. Higher scores (closer to 1) indicate better mixing.

```python
from evaluation import compute_kbet

kbet_score = compute_kbet(container, k=25)
# Returns: float between 0 and 1
```

#### LISI (Local Inverse Simpson Index)
Measures local diversity. Higher scores indicate more diverse neighborhoods.

```python
from evaluation import compute_lisi

lisi_score = compute_lisi(container, k=25)
# Returns: float (average LISI across cells)
```

#### Mixing Entropy
Normalized Shannon entropy of batch labels in local neighborhoods.

```python
from evaluation import compute_mixing_entropy

entropy = compute_mixing_entropy(container, k_neighbors=25)
# Returns: float between 0 and 1 (normalized)
```

#### Variance Ratio
Ratio of within-batch to between-batch variance.

```python
from evaluation import compute_variance_ratio

ratio = compute_variance_ratio(container)
# Returns: float (higher = better batch correction)
```

### Performance Metrics

#### Efficiency Score
Normalizes runtime and memory by data size.

```python
from evaluation import compute_efficiency_score

efficiency = compute_efficiency_score(
    runtime=10.0,
    memory=2.0,
    n_cells=1000,
    n_features=100
)
# Returns: dict with time_per_cell, memory_per_cell, etc.
```

#### Complexity Estimation
Estimates algorithmic complexity by fitting to different models.

```python
from evaluation import estimate_complexity

complexity = estimate_complexity(
    runtimes=[0.1, 0.2, 0.4, 0.8],
    data_sizes=[1000, 2000, 4000, 8000]
)
# Returns: {'estimated_complexity': 'linear', 'linear_r2': 0.99, ...}
```

### Distribution Metrics

#### Sparsity
Computes fraction of missing values.

```python
from evaluation import compute_sparsity

sparsity = compute_sparsity(container)
# Returns: float between 0 and 1
```

#### Statistics
Computes statistical properties (mean, std, skewness, kurtosis, CV).

```python
from evaluation import compute_statistics

stats = compute_statistics(container)
# Returns: {'mean': 1.23, 'std': 0.45, 'skewness': -0.2, ...}
```

#### Distribution Test
Kolmogorov-Smirnov test for distribution differences.

```python
from evaluation import distribution_test

statistic, pvalue = distribution_test(original, result)
# Returns: (statistic: float, pvalue: float)
```

### Structure Metrics

#### PCA Variance
Computes variance explained by PCA components.

```python
from evaluation import compute_pca_variance

variance = compute_pca_variance(container, n_components=10)
# Returns: np.ndarray of variance ratios
```

#### NN Consistency
Jaccard similarity of k-nearest neighbors before/after processing.

```python
from evaluation import compute_nn_consistency

consistency = compute_nn_consistency(original, result, k=10)
# Returns: float between 0 and 1
```

#### Distance Preservation
Correlation of pairwise distances before/after processing.

```python
from evaluation import compute_distance_preservation

corr = compute_distance_preservation(original, result, method='spearman')
# Returns: float (correlation coefficient)
```

## Configuration

The evaluation module uses a hierarchical configuration:

```python
config = {
    'batch_effects': {
        'enabled': True,
        'kbet': {'enabled': True, 'k': 25},
        'lisi': {'enabled': True, 'k': 25},
        'mixing_entropy': {'enabled': True, 'k_neighbors': 25},
        'variance_ratio': {'enabled': True}
    },
    'performance': {
        'enabled': True
    },
    'distribution': {
        'enabled': True,
        'sparsity': {'enabled': True},
        'statistics': {'enabled': True, 'metrics': ['mean', 'std', 'cv']},
        'distribution_test': {'enabled': True}
    },
    'structure': {
        'enabled': True,
        'pca_variance': {'enabled': True, 'n_components': 10},
        'nn_consistency': {'enabled': True, 'k': 10},
        'distance_preservation': {'enabled': True, 'method': 'spearman'},
        'global_structure': {'enabled': True}
    }
}
```

## Output Format

The evaluator returns a nested dictionary:

```python
{
    'pipeline_name': str,
    'dataset_name': str,
    'runtime': float,
    'memory_peak': float,
    'batch_effects': {
        'kbet': float,
        'lisi': float,
        'mixing_entropy': float,
        'variance_ratio': float
    },
    'performance': {
        'runtime_seconds': float,
        'memory_gb': float,
        'time_per_cell': float,
        'memory_per_cell': float
    },
    'distribution': {
        'sparsity_original': float,
        'sparsity_result': float,
        'sparsity_change': float,
        'mean_original': float,
        'mean_result': float,
        'ks_statistic': float,
        'ks_pvalue': float
    },
    'structure': {
        'pca_variance_cumulative': float,
        'pca_variance_pc1': float,
        'nn_consistency': float,
        'distance_correlation': float,
        'centroid_distance': float
    }
}
```

## Error Handling

The module includes comprehensive error handling:

- Individual metric failures are caught and reported as error strings
- Missing batch information is handled gracefully
- Edge cases (single batch, empty data) are handled
- Sparse matrices are automatically converted to dense format

## Type Safety

All functions include complete type annotations:

```python
def compute_kbet(container: Any, k: int = 25) -> float:
    """Compute kBET score..."""
    pass

def compute_efficiency_score(
    runtime: float,
    memory: float,
    n_cells: int,
    n_features: int
) -> dict[str, float]:
    """Compute efficiency scores..."""
    pass
```

## Examples

See the main comparison study documentation for complete examples.

## References

- Büttner, M. et al. (2019) Nature Methods - kBET
- Korsunsky, I. et al. (2019) Nature Methods - LISI
- Scikit-learn documentation - PCA and Nearest Neighbors
- SciPy documentation - Statistical tests
