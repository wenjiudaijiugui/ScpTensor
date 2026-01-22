# Data Loading and Preparation Module

This module provides utilities for loading and generating synthetic single-cell proteomics datasets for pipeline comparison testing.

## Overview

The data module consists of two main components:

1. **Data Loading** (`load_datasets.py`): Load real datasets from various file formats
2. **Synthetic Data Generation** (`prepare_synthetic.py`): Generate realistic synthetic datasets

## Installation

This module is part of the ScpTensor comparison study. Ensure you have ScpTensor installed:

```bash
cd /home/shenshang/projects/ScpTensor
uv sync
```

## Quick Start

### Generate Preset Datasets

```python
from docs.comparison_study.data import load_all_datasets

# Load all three preset datasets
datasets = load_all_datasets()

# Access individual datasets
small = datasets["small"]   # 1K cells × 1K proteins, 1 batch
medium = datasets["medium"] # 5K cells × 1.5K proteins, 5 batches
large = datasets["large"]   # 20K cells × 2K proteins, 10 batches
```

### Generate Custom Dataset

```python
from docs.comparison_study.data.prepare_synthetic import generate_synthetic_dataset

# Generate custom dataset
container = generate_synthetic_dataset(
    n_samples=5000,
    n_features=1500,
    n_batches=5,
    sparsity=0.7,
    batch_effect_size=1.5,
    n_cell_types=8,
    random_seed=42
)
```

### Load from Files

```python
from docs.comparison_study.data import load_dataset

# Load from pickle (ScpContainer format)
container = load_dataset("data/my_data.pkl")

# Load from CSV
container = load_dataset("data/my_data.csv")

# Load from h5ad (AnnData format)
container = load_dataset("data/my_data.h5ad")
```

## Dataset Characteristics

### Small Dataset
- **Size**: 1,000 cells × 1,000 proteins
- **Batches**: 1 (no batch effects)
- **Cell Types**: 5
- **Sparsity**: 60%
- **Use Case**: Baseline testing and method development

### Medium Dataset
- **Size**: 5,000 cells × 1,500 proteins
- **Batches**: 5 (moderate batch effects)
- **Cell Types**: 8
- **Sparsity**: 70%
- **Use Case**: Batch correction testing

### Large Dataset
- **Size**: 20,000 cells × 2,000 proteins
- **Batches**: 10 (strong batch effects)
- **Cell Types**: 12
- **Sparsity**: 75%
- **Use Case**: Scalability testing

## Synthetic Data Features

### Cell Type-Specific Expression
- Each cell type has marker proteins with high expression (uniform 8-10)
- Non-marker proteins have basal expression (exponential, scale=2.0)
- Technical noise added (Gaussian, σ=0.5)

### Batch Effects
When `n_batches > 1`, the following effects are added:
- **Location Effect**: Batch-specific shift (normal, σ=effect_size)
- **Scale Effect**: Batch-specific scaling (normal, μ=1.0, σ=0.1×effect_size)

### Missing Values
- Controlled by `sparsity` parameter (fraction of missing values)
- Missing values encoded with mask code 1 (MBR - Missing Between Runs)
- Randomly distributed across the dataset

## Data Structure

Generated datasets follow the ScpTensor hierarchical structure:

```
ScpContainer
├── obs (sample metadata)
│   ├── _index: cell IDs
│   ├── batch: batch labels (0 to n_batches-1)
│   ├── cell_type: cell type labels (0 to n_cell_types-1)
│   └── n_features: number of detected features per cell
└── assays
    └── proteins
        ├── var (feature metadata)
        │   ├── _index: feature IDs (protein_0, protein_1, ...)
        │   ├── n_cells: total number of cells
        │   └── missing_rate: fraction of missing values per feature
        └── layers
            └── raw
                ├── X: expression values (n_samples × n_features)
                └── M: mask codes (0=valid, 1=missing)
```

## Caching Datasets

For large datasets, use caching to avoid regenerating:

```python
from docs.comparison_study.data import cache_datasets, load_cached_datasets

# Generate and cache datasets
datasets = load_all_datasets()
cache_datasets(datasets, cache_dir="outputs/data_cache")

# Load cached datasets later
datasets = load_cached_datasets(cache_dir="outputs/data_cache")
```

## Advanced Usage

### Custom Batch Labels

```python
from docs.comparison_study.data.load_datasets import create_batch_labels

# Even split
labels = create_batch_labels(n_samples=100, n_batches=4)

# Custom sizes
labels = create_batch_labels(
    n_samples=100,
    n_batches=3,
    batch_sizes=[30, 50, 20]
)
```

### Add Batch Effects to Existing Data

```python
from docs.comparison_study.data.load_datasets import add_batch_effects

# Add batch effects
X_batched = add_batch_effects(
    X,
    batch_labels=batch_labels,
    effect_size=1.5
)
```

### CSV Format Requirements

When loading from CSV, metadata columns must be prefixed with `meta_`:

```csv
meta_batch,meta_cell_type,protein1,protein2,protein3
0,0,10.5,8.2,5.1
0,1,12.3,9.1,6.2
1,0,11.1,7.8,5.5
```

This will create:
- `obs`: columns `batch` and `cell_type`
- Data matrix: columns `protein1`, `protein2`, `protein3`

## API Reference

### Data Loading Functions

#### `load_dataset(dataset_path, assay_name="proteins", batch_column="batch")`
Load dataset from file. Auto-detects format from extension.

**Supported formats**: `.pkl`, `.csv`, `.h5ad`, `.h5`

**Returns**: `ScpContainer`

#### `load_from_pickle(pickle_path)`
Load ScpContainer from pickle file.

#### `load_from_csv(csv_path, assay_name="proteins", batch_column="batch")`
Load dataset from CSV file with `meta_` prefix for metadata columns.

#### `load_from_h5ad(h5ad_path, assay_name="proteins", batch_column="batch")`
Convert AnnData h5ad file to ScpContainer.

**Requires**: `anndata` package

### Synthetic Data Functions

#### `generate_synthetic_dataset(...)`
Generate synthetic dataset with full customization.

**Parameters**:
- `n_samples`: Number of cells
- `n_features`: Number of proteins
- `n_batches`: Number of batches (1 = single batch)
- `sparsity`: Fraction of missing values (0-1)
- `batch_effect_size`: Strength of batch effects (0 = none)
- `n_cell_types`: Number of distinct cell types
- `random_seed`: Random seed for reproducibility

**Returns**: `ScpContainer`

#### `generate_small_dataset(random_seed=42)`
Generate small preset dataset (1K × 1K, 1 batch).

#### `generate_medium_dataset(random_seed=42)`
Generate medium preset dataset (5K × 1.5K, 5 batches).

#### `generate_large_dataset(random_seed=42)`
Generate large preset dataset (20K × 2K, 10 batches).

#### `load_all_datasets(config=None)`
Load all three preset datasets.

**Parameters**:
- `config`: Optional dict with custom parameters per dataset

**Returns**: `dict[str, ScpContainer]`

#### `cache_datasets(datasets, cache_dir="outputs/data_cache")`
Cache datasets to disk using pickle format.

#### `load_cached_datasets(cache_dir="outputs/data_cache")`
Load previously cached datasets from disk.

## Examples

See `/docs/comparison_study/examples/demo_data_generation.py` for comprehensive examples.

## Testing

Run unit tests:

```bash
uv run pytest tests/comparison_study/test_data.py -v
```

## Notes

- All random number generation uses `np.random.default_rng()` (new API)
- Datasets are fully reproducible with fixed random seeds
- Mask matrices use int8 dtype for memory efficiency
- Generated data is always non-negative
- Cell type markers are selected deterministically based on feature indices

## Dependencies

- `numpy`: Numerical operations
- `polars`: Dataframe operations
- `scipy`: Sparse matrix support
- `scptensor`: Core data structures

Optional:
- `anndata`: For loading h5ad files

## License

This module is part of ScpTensor and follows the same license.
