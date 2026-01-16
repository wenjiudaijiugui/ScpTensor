# ScpTensor Project Status

**Version:** v0.1.0-beta
**Last Updated:** 2026-01-14
**Project:** Single-Cell Proteomics Analysis Framework

---

## Project Overview

ScpTensor is a Python library designed for the analysis of single-cell proteomics (SCP) data. It provides a unified data structure and comprehensive analysis tools for quality control, normalization, imputation, batch effect correction, dimensionality reduction, clustering, differential expression, and feature selection.

### Key Features

- **Unified Data Structure:** Hierarchical `ScpContainer` -> `Assay` -> `ScpMatrix` design
- **Provenance Tracking:** Built-in operation audit trail with `ProvenanceLog`
- **Sparse Matrix Support:** Efficient handling of sparse SCP data (70-90% missing values)
- **Performance Optimized:** Numba JIT compilation and sparse-aware operations
- **Type Safe:** Full type annotations on public APIs
- **Well Tested:** Comprehensive test suite with pytest (>80% coverage)
- **Differential Expression:** Statistical tests for group comparisons
- **Feature Selection:** HVG, VST, dropout-based, and model-based methods
- **Benchmarking:** Competitor comparison and performance metrics

---

## Current Version: v0.1.0-beta

### Capabilities

#### Data Structures (scptensor.core)
| Component | Description |
|-----------|-------------|
| `ScpContainer` | Top-level container for multi-assay experiments |
| `Assay` | Feature-space specific data layer management |
| `ScpMatrix` | Physical storage with mask-based provenance tracking |
| `ProvenanceLog` | Operation history audit trail |
| `MaskCode` | Data status enumeration (VALID, MBR, LOD, FILTERED, IMPUTED) |

#### Normalization Methods (scptensor.normalization)
| Method | Description |
|--------|-------------|
| `log_normalize` | Log transform (base 2 by default) |
| `sample_median_normalization` | Median centering per sample |
| `sample_mean_normalization` | Mean centering per sample |
| `global_median_normalization` | Global median centering |
| `tmm_normalization` | TMM scaling for between-sample normalization |
| `upper_quartile_normalization` | Upper quartile scaling |
| `zscore_normalize` | Z-score standardization |

#### Imputation Methods (scptensor.impute)
| Method | Description | Complexity | Status |
|--------|-------------|------------|--------|
| `knn` | K-nearest neighbors imputation | O(n^2) | Complete |
| `ppca` | Probabilistic PCA imputation | O(n^3) | Complete |
| `svd_impute` | Singular value decomposition imputation | O(n^3) | Complete |
| `missforest` | Random forest-based imputation | O(n^2 log n) | Complete |
| `qrilc` | Quantile Regression Imputation of Left-Censored data | O(n log n) | Complete |
| `minprob` | Probabilistic minimum imputation (MNAR) | O(n) | Complete |
| `mindet` | Deterministic minimum imputation (MNAR) | O(n) | Complete |
| `lls` | Local Least Squares imputation | O(n^2) | Complete |
| `bpca` | Bayesian PCA imputation | O(n^3) | Complete |
| `nmf` | Non-negative Matrix Factorization imputation | O(n^3) | Complete |

#### Integration Methods (scptensor.integration)
| Method | Description | Dependencies | Status |
|--------|-------------|--------------|--------|
| `combat` | ComBat batch correction (empirical Bayes) | Built-in | Complete |
| `harmony` | Harmony integration (iterative clustering) | harmonypy (optional) | Complete |
| `mnn_correct` | Mutual Nearest Neighbors correction | Built-in | Complete |
| `scanorama_integrate` | Scanorama integration | scanorama (optional) | Complete |
| `nonlinear_integrate` | Nonlinear manifold alignment | Built-in | Complete |

#### Dimensionality Reduction (scptensor.dim_reduction)
| Method | Description | Status |
|--------|-------------|--------|
| `pca` | Principal Component Analysis | Complete |
| `umap` | Uniform Manifold Approximation and Projection | Complete |

#### Clustering (scptensor.cluster)
| Method | Description | Status |
|--------|-------------|--------|
| `run_kmeans` | K-means clustering | Complete |
| `graph_cluster` | Graph clustering (Leiden/Louvain) | Complete |

#### Quality Control (scptensor.qc)
| Method | Description | Status |
|--------|-------------|--------|
| `basic_qc` | Perform basic quality control calculations | Complete |
| `detect_outliers` | Detect outlier samples using statistical methods | Complete |
| `calculate_qc_metrics` | Calculate comprehensive QC metrics | Complete |
| `filter_features_by_missing_rate` | Filter features with excessive missing values | Complete |
| `filter_samples_by_total_count` | Filter samples based on total intensity | Complete |

#### Feature Selection (scptensor.feature_selection)
| Method | Description | Status |
|--------|-------------|--------|
| `select_hvg` | Highly Variable Genes/Proteins selection | Complete |
| `select_by_dropout` | Filter features by missing data rate | Complete |
| `select_by_vst` | Seurat-style VST feature selection | Complete |
| `select_by_dispersion` | Normalized dispersion ranking | Complete |
| `select_by_model_importance` | Random forest importance ranking | Complete |
| `select_by_pca_loadings` | PCA loading-based selection | Complete |

#### Differential Expression (scptensor.diff_expr)
| Method | Description | Status |
|--------|-------------|--------|
| `diff_expr_ttest` | Two-group comparison (Welch's/Student's t-test) | Complete |
| `diff_expr_mannwhitney` | Non-parametric two-group comparison | Complete |
| `diff_expr_anova` | Multi-group ANOVA | Complete |
| `diff_expr_kruskal` | Non-parametric multi-group comparison | Complete |
| `adjust_fdr` | Multiple testing correction (FDR) | Complete |

#### Visualization (scptensor.viz)
| Method | Type | Description |
|--------|------|-------------|
| `scatter`, `heatmap`, `violin` | Static (Matplotlib) | Basic plots with SciencePlots style |
| `scatter_plot`, `violin_plot`, `heatmap_i` | Interactive (Plotly) | Interactive visualizations |
| `embedding` | Recipe | PCA/UMAP embedding visualization |
| `qc_completeness`, `qc_matrix_spy` | Recipe | QC visualization |
| `volcano` | Recipe | Volcano plot for differential expression |

#### Datasets (scptensor.datasets)
| Dataset | Samples | Features | Description |
|----------|---------|----------|-------------|
| `load_toy_example` | ~100 | ~50 | Small synthetic for quick testing |
| `load_simulated_scrnaseq_like` | ~500 | ~200 | Larger simulated with cell types |
| `load_example_with_clusters` | ~200 | ~100 | Data with known cluster labels |

#### Benchmarking (scptensor.benchmark)
| Feature | Description |
|---------|-------------|
| Synthetic data generation | Parameterized synthetic SCP data |
| Competitor comparison | Compare against scanpy, scprep |
| Performance metrics | Runtime, memory, accuracy |
| Visualization | Benchmark result plots |

---

## Completed Tasks Summary

### P0: Critical Blockers (7/7 Complete)
- Module import structure fixed
- Test infrastructure established
- CI/CD pipeline operational
- Core tests written (80+ tests)

### P1: High Priority (12/12 Complete)
- All imputation methods implemented (4/4)
- Integration methods complete (5/5)
- Type hints added to public APIs
- Error handling layer established
- API documentation generated
- Integration tests passing
- Sparse matrix operations optimized
- Numba JIT compiled hot loops
- Dependency management configured
- Feature selection module complete
- Differential expression module complete
- Example datasets module complete

### P2: Medium Priority (8/8 Complete)
- Competitor benchmarking framework
- Tutorial notebooks (4 tutorials)
- Pre-commit hooks configured
- Documentation website
- CHANGELOG.md
- Contributing guidelines
- Advanced QC metrics
- Visualization recipes

**Overall Status: 27/27 tasks complete (100%)**

---

## Quick Start Guide

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/ScpTensor.git
cd ScpTensor

# Create virtual environment and install
uv venv
source .venv/bin/activate
uv pip install -e .
```

### Basic Usage

```python
import numpy as np
import polars as pl
from scptensor import ScpContainer, Assay, ScpMatrix

# Create a container with sample data
n_samples, n_features = 100, 500
X = np.random.rand(n_samples, n_features)

# Create metadata
obs = pl.DataFrame({
    "_index": [f"S{i}" for i in range(n_samples)],
    "batch": ["A"] * 50 + ["B"] * 50,
    "condition": ["control"] * 50 + ["treated"] * 50
})
var = pl.DataFrame({
    "_index": [f"P{i}" for i in range(n_features)]
})

# Initialize container
matrix = ScpMatrix(X=X)
assay = Assay(var=var, layers={"raw": matrix})
container = ScpContainer(obs=obs, assays={"proteins": assay})

# Run analysis pipeline
from scptensor.normalization import log_normalize
from scptensor.impute import knn
from scptensor.integration import combat
from scptensor.dim_reduction import pca, umap
from scptensor.cluster import run_kmeans
from scptensor.feature_selection import select_hvg
from scptensor.diff_expr import diff_expr_ttest

# Feature selection
container = select_hvg(container, n_top_features=200)

# Normalize
container = log_normalize(container, assay_name="proteins", layer_name="raw")

# Impute missing values
container = knn(container, assay_name="proteins", layer_name="log", k=5)

# Batch correction
container = combat(container, assay_name="proteins", batch_key="batch")

# Dimensionality reduction
container = pca(container, assay_name="proteins", layer_name="corrected")
container = umap(container, assay_name="proteins", layer_name="pca")

# Clustering
container = run_kmeans(container, assay_name="proteins", layer_name="umap", n_clusters=2)

# Differential expression
result = diff_expr_ttest(container, group_key="condition")
```

---

## Module Overview

### Core Layer
- **Location:** `scptensor/core/`
- **Purpose:** Data structures and foundational utilities
- **Key Files:** `structures.py`, `matrix_ops.py`, `sparse_utils.py`, `jit_ops.py`, `exceptions.py`, `io.py`

### Normalization
- **Location:** `scptensor/normalization/`
- **Purpose:** Transform data distributions
- **Methods:** log, median centering, TMM, upper quartile, z-score

### Imputation
- **Location:** `scptensor/impute/`
- **Purpose:** Fill missing values
- **Methods:** KNN, PPCA, SVD, MissForest, QRILC, MinProb, MinDet, LLS, BPCA, NMF (all complete)

### Integration
- **Location:** `scptensor/integration/`
- **Purpose:** Batch effect correction
- **Methods:** ComBat, Harmony, MNN, Scanorama, Nonlinear (all complete)

### Dimensionality Reduction
- **Location:** `scptensor/dim_reduction/`
- **Purpose:** Reduce feature dimensions
- **Methods:** PCA, UMAP

### Clustering
- **Location:** `scptensor/cluster/`
- **Purpose:** Group similar samples
- **Methods:** KMeans, graph clustering

### Quality Control
- **Location:** `scptensor/qc/`
- **Purpose:** Data quality assessment
- **Methods:** Basic QC, outlier detection, filtering

### Feature Selection
- **Location:** `scptensor/feature_selection/`
- **Purpose:** Select informative features
- **Methods:** HVG, VST, dropout-based, model-based, dispersion, PCA loadings

### Differential Expression
- **Location:** `scptensor/diff_expr/`
- **Purpose:** Statistical group comparisons
- **Methods:** t-test, Mann-Whitney, ANOVA, Kruskal-Wallis, FDR

### Visualization
- **Location:** `scptensor/viz/`
- **Purpose:** Plotting and visualization
- **Methods:** Static (Matplotlib) and interactive (Plotly)

### Datasets
- **Location:** `scptensor/datasets/`
- **Purpose:** Example data for tutorials and testing
- **Datasets:** toy example, simulated scRNA-seq-like, with clusters

### Benchmarking
- **Location:** `scptensor/benchmark/`
- **Purpose:** Performance testing and competitor comparison
- **Tools:** Synthetic data, metrics, visualization, competitor suite

---

## Performance Characteristics

### Sparse Matrix Support
- Automatic conversion when sparsity > 50%
- 4-7x memory reduction on typical SCP data
- Sparse-safe operations throughout the pipeline

### JIT Compilation
- Numba JIT on hot loops
- 2-5x speedup on critical operations
- Fallback to pure Python if Numba unavailable

### Benchmarks
| Operation | 1000x500 Matrix | 1000x5000 Matrix |
|-----------|-----------------|------------------|
| Log normalize | <0.1s | <1s |
| KNN impute (k=5) | <5s | <30s |
| PPCA impute | <2s | <15s |
| MissForest impute | <10s | <60s |
| PCA (50 components) | <1s | <10s |
| ComBat | <1s | <10s |
| Harmony | <5s | <30s |
| UMAP | <5s | <30s |

---

## Documentation

### User Documentation
- `README.md` - Project overview and installation
- `docs/PROJECT_STATUS.md` - This file
- `docs/PROJECT_STRUCTURE.md` - Directory structure and module details
- `docs/API_QUICK_REFERENCE.md` - Usage examples and code snippets
- `CHANGELOG.md` - Version history and changes

### Developer Documentation
- `CONTRIBUTING.md` - Contributing guidelines
- `CLAUDE.md` - Project-specific instructions for AI assistants
- `docs/design/MIGRATION.md` - Migration guide for version updates

### Design Documentation
- `docs/design/INDEX.md` - Navigation hub for design docs
- `docs/design/MASTER.md` - Strategic overview
- `docs/design/ARCHITECTURE.md` - Technical specifications
- `docs/design/ROADMAP.md` - Execution plan
- `docs/design/API_REFERENCE.md` - Complete API documentation

### Tutorials
- `docs/tutorials/tutorial_01_getting_started.ipynb` - Introduction to ScpTensor
- `docs/tutorials/tutorial_02_qc_normalization.ipynb` - QC and normalization
- `docs/tutorials/tutorial_03_imputation_integration.ipynb` - Imputation and batch correction
- `docs/tutorials/tutorial_04_clustering_visualization.ipynb` - Clustering and visualization

---

## CI/CD Pipeline

### Continuous Integration (GitHub Actions)
- **ci.yml** - Runs on every push/PR
  - Python 3.11, 3.12, 3.13 testing
  - Linting (ruff)
  - Type checking (mypy)
  - Unit tests with coverage
  - Security scanning

### Continuous Deployment
- **cd.yml** - Automatic PyPI release on tags
  - Builds distribution packages
  - Publishes to PyPI

### Additional Workflows
- **dependency-review.yml** - Dependency vulnerability scanning
- **nightly-benchmark.yml** - Daily performance benchmarking

---

## Development Status

### Current Phase: Beta Release
- All core features implemented
- Test infrastructure operational
- CI/CD pipeline active
- Documentation complete
- Tutorial notebooks available

### Next Release: v0.2.0
**Planned:** 2026-Q2
- Advanced visualization options
- Additional integration methods
- Performance optimizations
- Extended tutorial series

---

## Dependencies

### Required
- Python >= 3.11
- numpy >= 1.26
- scipy >= 1.14
- polars >= 1.0
- scikit-learn >= 1.5
- matplotlib >= 3.9
- scienceplots >= 2.0
- plotly >= 6.0
- umap-learn >= 0.5.6
- numba >= 0.60
- psutil >= 6.0

### Optional
- harmonypy - for Harmony integration
- scanorama - for Scanorama integration
- scanpy - for additional functionality
- kneed - for knee detection in HVG

### Development
- pytest >= 9.0
- pytest-cov >= 6.0
- ruff >= 0.9
- mypy >= 1.15
- pre-commit >= 4.0
- sphinx >= 8.0

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest --cov=scptensor --cov-report=html

# Run specific module tests
pytest tests/core/ -v
pytest tests/integration/ -v
pytest tests/test_feature_selection.py -v
pytest tests/test_diff_expr.py -v
```

**Coverage:**
- Core structures: ~85%
- Overall: ~80%
- New modules (feature_selection, diff_expr): ~75%

---

## Contributing

Contributions are welcome! Please see `CONTRIBUTING.md` for guidelines.

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Ensure all tests pass
5. Submit a pull request

---

## License

[Your License Here]

---

## Contact

- GitHub: https://github.com/yourusername/ScpTensor
- Issues: https://github.com/yourusername/ScpTensor/issues

---

**Document Maintainer:** ScpTensor Team
**Last Updated:** 2026-01-14
