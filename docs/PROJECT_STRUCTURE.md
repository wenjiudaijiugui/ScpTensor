# ScpTensor Project Structure

**Version:** v0.1.0
**Last Updated:** 2025-01-09

---

## Directory Tree

```
ScpTensor/
|-- .github/
|   `-- workflows/
|       `-- ci.yml                 # GitHub Actions CI/CD pipeline
|
|-- docs/                          # Documentation
|   |-- design/                    # Design documents
|   |   |-- INDEX.md               # Navigation hub for design docs
|   |   |-- MASTER.md              # Strategic overview
|   |   |-- ARCHITECTURE.md        # Technical specifications
|   |   |-- ROADMAP.md             # Execution plan
|   |   |-- API_REFERENCE.md       # Complete API documentation
|   |   `-- MIGRATION.md           # Migration guide
|   |-- api/                       # Generated API documentation (Sphinx)
|   |-- notebooks/                 # Tutorial Jupyter notebooks
|   |-- _build/                    # Built documentation
|   |-- _static/                   # Static files for Sphinx
|   |-- COMPETITOR_BENCHMARK.md    # Benchmark results
|   |-- COMPLETION_STATUS.md       # Task completion status
|   |-- DEVELOPER_GUIDE.md         # Contributing guidelines
|   |-- DEVELOPMENT.md             # Development workflow
|   |-- ISSUES_AND_LIMITATIONS.md  # Known issues
|   |-- PROJECT_STATUS.md          # Project overview
|   |-- PROJECT_STRUCTURE.md       # This file
|   |-- API_QUICK_REFERENCE.md     # Quick API reference
|   |-- conf.py                    # Sphinx configuration
|   |-- index.rst                  # Sphinx index
|   `-- quickstart.rst             # Quick start guide
|
|-- scripts/                       # Utility scripts
|   |-- README.md                  # Scripts documentation
|   |-- doc_loader.py              # Progressive documentation loader
|   |-- run_competitor_benchmark.py # Run competitor benchmarks
|   `-- verify_sparse_optimizations.py # Verify sparse optimizations
|
|-- scptensor/                     # Main package
|   |-- __init__.py
|   |-- benchmark/                 # Benchmarking module
|   |   |-- __init__.py
|   |   |-- benchmark_suite.py     # Main benchmark orchestration
|   |   |-- competitor_benchmark.py # Competitor comparison
|   |   |-- competitor_suite.py    # Competitor test suite
|   |   |-- competitor_viz.py      # Competitor visualization
|   |   |-- core.py                # Benchmark core utilities
|   |   |-- metrics.py             # Benchmark metrics
|   |   |-- parameter_grid.py      # Parameter grid search
|   |   |-- synthetic_data.py      # Synthetic data generation
|   |   `-- visualization.py       # Benchmark visualization
|   |
|   |-- cluster/                   # Clustering algorithms
|   |   |-- __init__.py
|   |   |-- basic.py               # Basic clustering utilities
|   |   |-- graph.py               # Graph-based clustering (Leiden/Louvain)
|   |   `-- kmeans.py              # K-means clustering
|   |
|   |-- core/                      # Core data structures and utilities
|   |   |-- __init__.py            # Public API exports
|   |   |-- exceptions.py          # Custom exception hierarchy
|   |   |-- io.py                  # I/O utilities (save/load)
|   |   |-- jit_ops.py             # Numba JIT-compiled operations
|   |   |-- matrix_ops.py          # Matrix operation utilities
|   |   |-- reader.py              # Data reader
|   |   |-- sparse_utils.py        # Sparse matrix utilities
|   |   |-- structures.py          # Core data structures
|   |   `-- utils.py               # General utilities
|   |
|   |-- datasets/                  # Example datasets
|   |   |-- __init__.py
|   |   `-- _example.py            # Example dataset loader
|   |
|   |-- diff_expr/                 # Differential expression (in development)
|   |   |-- __init__.py
|   |   `-- core.py                # Differential expression core
|   |
|   |-- dim_reduction/             # Dimensionality reduction
|   |   |-- __init__.py
|   |   |-- pca.py                 # Principal Component Analysis
|   |   |-- umap.py                # UMAP wrapper
|   |   `-- _umap.py               # Internal UMAP implementation
|   |
|   |-- feature_selection/         # Feature selection
|   |   `-- hvg.py                 # Highly Variable Genes/Proteins
|   |
|   |-- impute/                    # Imputation methods
|   |   |-- __init__.py
|   |   |-- knn.py                 # K-nearest neighbors imputation
|   |   |-- missforest.py          # MissForest imputation
|   |   |-- ppca.py                # Probabilistic PCA imputation
|   |   `-- svd.py                 # SVD imputation
|   |
|   |-- integration/               # Batch effect correction
|   |   |-- __init__.py
|   |   |-- combat.py              # ComBat batch correction
|   |   |-- harmony.py             # Harmony integration
|   |   |-- mnn.py                 # Mutual Nearest Neighbors
|   |   |-- nonlinear.py           # Nonlinear integration methods
|   |   `-- scanorama.py           # Scanorama integration
|   |
|   |-- normalization/             # Normalization methods
|   |   |-- __init__.py
|   |   |-- global_median.py       # Global median normalization
|   |   |-- log.py                 # Log transform
|   |   |-- median_centering.py    # Median centering
|   |   |-- median_scaling.py      # Median scaling
|   |   |-- sample_mean.py         # Sample mean normalization
|   |   |-- sample_median.py       # Sample median normalization
|   |   |-- tmm.py                 # TMM normalization
|   |   |-- upper_quartile.py      # Upper quartile normalization
|   |   `-- zscore.py              # Z-score standardization
|   |
|   |-- qc/                        # Quality Control
|   |   |-- __init__.py
|   |   |-- advanced.py            # Advanced QC methods
|   |   |-- basic.py               # Basic QC calculations
|   |   `-- outlier.py             # Outlier detection
|   |
|   |-- referee/                   # Benchmark referee
|   |   |-- benchmark.py           # Referee benchmark logic
|   |   `-- metrics.py             # Referee metrics
|   |
|   |-- standardization/           # Standardization methods
|   |   |-- __init__.py
|   |   `-- zscore.py              # Z-score standardization
|   |
|   |-- utils/                     # Utility functions
|   |   |-- __init__.py
|   |   `-- data_genetator.py      # Data generation utilities
|   |
|   `-- viz/                       # Visualization
|       |-- __init__.py
|       |-- interactive.py         # Interactive Plotly visualizations
|       |-- base/                  # Base visualization components
|       |   |-- __init__.py
|       |   |-- heatmap.py         # Heatmap plots
|       |   |-- scatter.py         # Scatter plots
|       |   |-- style.py           # Plot styling
|       |   `-- violin.py          # Violin plots
|       `-- recipes/               # Pre-configured visualization recipes
|           |-- __init__.py
|           |-- embedding.py       # Embedding visualizations
|           |-- qc.py              # QC visualizations
|           `-- stats.py           # Statistical plots
|
|-- tests/                         # Test suite
|   |-- benchmark_jit.py           # JIT benchmarks
|   |-- benchmark_sparse_operations.py # Sparse operation benchmarks
|   |-- conftest.py                # Pytest configuration
|   |-- test_error_handling.py     # Error handling tests
|   |-- test_filtering.py          # Filtering tests
|   |-- test_matrix_ops_sparse.py  # Sparse matrix tests
|   |-- test_sparse_utils.py       # Sparse utilities tests
|   |-- total_proc.py              # End-to-end pipeline test
|   |-- total_proc.ipynb           # Pipeline notebook
|   |-- core/                      # Core module tests
|   |   |-- __init__.py
|   |   |-- test_assay.py
|   |   |-- test_assay_edge_cases.py
|   |   |-- test_container_basic.py
|   |   |-- test_container_edge_cases.py
|   |   |-- test_helpers.py
|   |   |-- test_mask_codes.py
|   |   |-- test_matrix.py
|   |   `-- test_matrix_edge_cases.py
|   |-- integration/               # Integration tests
|   |   |-- __init__.py
|   |   |-- conftest.py
|   |   |-- test_pipeline.py       # Pipeline integration tests
|   |   |-- test_synthetic.py      # Synthetic data tests
|   |   `-- test_workflows.py      # Workflow tests
|   |-- data/                      # Test data
|   |   `-- PXD061065/             # Example SCP dataset
|   |-- impute/                    # Imputation tests
|   |-- normalization/             # Normalization tests
|   |-- res/                       # Test results
|   |-- pipeline_results/          # Pipeline test outputs
|   `-- viz/                       # Visualization tests
|
|-- .github/                       # GitHub configuration
|   `-- workflows/
|       `-- ci.yml                 # CI/CD pipeline
|
|-- .pre-commit-config.yaml        # Pre-commit hooks configuration
|-- .serena/                       # Serena configuration
|-- CLAUDE.md                      # Project instructions for Claude
|-- CONTRIBUTING.md                # Contributing guidelines
|-- MIGRATION.md                   # Migration guide
|-- P1-7_COMPLETION_SUMMARY.md     # P1-7 task summary
|-- README.md                      # Project README
|-- SPARSE_OPTIMIZATION_REPORT.md  # Sparse optimization report
|-- benchmark_example.py           # Benchmark example script
|-- coverage.json                  # Coverage report data
|-- pyproject.toml                 # Project configuration
`-- uv.lock                       # Dependency lock file
```

---

## Module Descriptions

### Core Layer (`scptensor/core`)

**Purpose:** Foundation for all ScpTensor functionality

| File | Description |
|------|-------------|
| `structures.py` | Core data structures: `ScpContainer`, `Assay`, `ScpMatrix`, `ProvenanceLog`, `MaskCode` |
| `exceptions.py` | Custom exception hierarchy for error handling |
| `matrix_ops.py` | Matrix operation utilities with sparse support |
| `sparse_utils.py` | Sparse matrix utilities (conversion, memory optimization) |
| `jit_ops.py` | Numba JIT-compiled operations for performance |
| `reader.py` | Data ingestion from various formats |
| `io.py` | Save/load utilities |
| `utils.py` | General utility functions |

---

### Normalization (`scptensor/normalization`)

**Purpose:** Transform data distributions for analysis

| File | Method | Description |
|------|--------|-------------|
| `log.py` | `log_normalize` | Log transform (base 2 default) |
| `sample_median.py` | `sample_median_normalization` | Median centering per sample |
| `sample_mean.py` | `sample_mean_normalization` | Mean centering per sample |
| `global_median.py` | `global_median_normalization` | Global median centering |
| `tmm.py` | `tmm_normalization` | TMM scaling for between-sample |
| `upper_quartile.py` | `upper_quartile_normalization` | Upper quartile scaling |
| `median_centering.py` | `median_centering` | Alternative median centering |
| `median_scaling.py` | `median_scaling` | Median-based scaling |
| `zscore.py` | `zscore_standardization` | Z-score normalization |

---

### Imputation (`scptensor/impute`)

**Purpose:** Fill missing values using statistical methods

| File | Method | Description | Complexity |
|------|--------|-------------|------------|
| `knn.py` | `knn` | K-nearest neighbors imputation | O(n^2) |
| `ppca.py` | `ppca` | Probabilistic PCA imputation | O(n^3) |
| `svd.py` | `svd_impute` | SVD-based imputation | O(n^3) |
| `missforest.py` | `missforest` | Random forest imputation | O(n^2 log n) |

---

### Integration (`scptensor/integration`)

**Purpose:** Batch effect correction and data integration

| File | Method | Description | Dependencies |
|------|--------|-------------|--------------|
| `combat.py` | `combat` | ComBat empirical Bayes correction | Built-in |
| `harmony.py` | `harmony` | Harmony iterative integration | harmonypy (optional) |
| `mnn.py` | `mnn_correct` | Mutual Nearest Neighbors correction | Built-in |
| `scanorama.py` | `scanorama_integrate` | Scanorama integration | scanorama (optional) |
| `nonlinear.py` | Nonlinear integration methods | Advanced integration | Various |

---

### Dimensionality Reduction (`scptensor/dim_reduction`)

**Purpose:** Reduce feature dimensions for visualization and analysis

| File | Method | Description |
|------|--------|-------------|
| `pca.py` | `pca` | Principal Component Analysis |
| `umap.py` | `umap` | UMAP wrapper |
| `_umap.py` | Internal UMAP implementation | UMAP algorithm |

---

### Clustering (`scptensor/cluster`)

**Purpose:** Group similar samples based on feature similarity

| File | Method | Description |
|------|--------|-------------|
| `kmeans.py` | `run_kmeans` | K-means clustering |
| `graph.py` | Graph clustering | Leiden/Louvain clustering via scanpy |
| `basic.py` | Basic clustering utilities | Clustering helpers |

---

### Quality Control (`scptensor/qc`)

**Purpose:** Assess and filter data quality

| File | Methods | Description |
|------|---------|-------------|
| `basic.py` | `basic_qc` | Basic QC metrics calculation |
| `outlier.py` | `detect_outliers` | Statistical outlier detection |
| `advanced.py` | `calculate_qc_metrics`, `filter_*` | Advanced filtering and metrics |

---

### Visualization (`scptensor/viz`)

**Purpose:** Plotting and visualization

| Directory/File | Type | Methods |
|----------------|------|---------|
| `base/scatter.py` | Static (Matplotlib) | `scatter` |
| `base/heatmap.py` | Static (Matplotlib) | `heatmap` |
| `base/violin.py` | Static (Matplotlib) | `violin` |
| `base/style.py` | Static (Matplotlib) | SciencePlots style configuration |
| `interactive.py` | Interactive (Plotly) | `scatter_plot`, `violin_plot`, `heatmap_i`, `cluster_viz`, `save_html` |
| `recipes/embedding.py` | Recipe | `embedding` (PCA/UMAP plots) |
| `recipes/qc.py` | Recipe | `qc_completeness`, `qc_matrix_spy` |
| `recipes/stats.py` | Recipe | `volcano` |

---

### Benchmarking (`scptensor/benchmark`)

**Purpose:** Performance testing and competitor comparison

| File | Description |
|------|-------------|
| `benchmark_suite.py` | Main benchmark orchestration |
| `competitor_benchmark.py` | Competitor comparison logic |
| `competitor_suite.py` | Competitor test suite |
| `competitor_viz.py` | Competitor visualization |
| `core.py` | Benchmark core utilities |
| `metrics.py` | Benchmark metrics calculation |
| `parameter_grid.py` | Parameter grid search |
| `synthetic_data.py` | Synthetic SCP data generation |
| `visualization.py` | Benchmark visualization |

---

### Feature Selection (`scptensor/feature_selection`)

**Purpose:** Select informative features for analysis

| File | Method | Description |
|------|--------|-------------|
| `hvg.py` | `select_hvg` | Highly Variable Genes/Proteins selection |

---

### Differential Expression (`scptensor/diff_expr`)

**Purpose:** Find differentially expressed features (in development)

| File | Description |
|------|-------------|
| `core.py` | Differential expression core functions |

---

## Key Files and Their Purposes

### Project Configuration

| File | Purpose |
|------|---------|
| `pyproject.toml` | Project metadata, dependencies, tool configuration |
| `uv.lock` | Locked dependency versions for reproducibility |
| `.pre-commit-config.yaml` | Pre-commit hooks configuration |

### Documentation

| File | Purpose |
|------|---------|
| `README.md` | Project overview and quick start |
| `CLAUDE.md` | Project-specific instructions for AI assistant |
| `CONTRIBUTING.md` | Contribution guidelines |
| `MIGRATION.md` | Version migration guide |

### Status and Tracking

| File | Purpose |
|------|---------|
| `P1-7_COMPLETION_SUMMARY.md` | P1-7 task completion details |
| `SPARSE_OPTIMIZATION_REPORT.md` | Sparse matrix optimization report |
| `docs/COMPLETION_STATUS.md` | Overall task completion status |
| `docs/PROJECT_STATUS.md` | Project overview and capabilities |
| `docs/PROJECT_STRUCTURE.md` | This file |
| `docs/API_QUICK_REFERENCE.md` | Quick API reference and examples |

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/doc_loader.py` | Progressive documentation loader |
| `scripts/run_competitor_benchmark.py` | Run competitor benchmarks |
| `scripts/verify_sparse_optimizations.py` | Verify sparse optimizations |

---

## Testing Structure

### Test Organization

```
tests/
|-- core/              # Core module tests
|-- integration/       # Integration/end-to-end tests
|-- impute/            # Imputation tests
|-- normalization/     # Normalization tests
|-- viz/               # Visualization tests
|-- data/              # Test datasets
|-- res/               # Test results
|-- pipeline_results/  # Pipeline test outputs
```

### Key Test Files

| File | Purpose |
|------|---------|
| `conftest.py` | Shared pytest fixtures |
| `test_error_handling.py` | Error handling validation |
| `test_filtering.py` | Filtering functionality |
| `test_sparse_utils.py` | Sparse utilities tests |
| `test_matrix_ops_sparse.py` | Sparse matrix operation tests |
| `total_proc.py` | End-to-end pipeline test |

---

## Design Documentation Structure

The `docs/design/` directory contains comprehensive design documentation:

| Document | Lines | Purpose |
|----------|-------|---------|
| `INDEX.md` | ~400 | Navigation hub for design docs |
| `MASTER.md` | ~639 | Strategic overview and priorities |
| `ARCHITECTURE.md` | ~1100 | Technical specifications |
| `ROADMAP.md` | ~700 | Execution plan and milestones |
| `API_REFERENCE.md` | ~900 | Complete API documentation |
| `MIGRATION.md` | ~600 | Version migration guide |

---

## Build Artifacts

| Directory | Purpose |
|-----------|---------|
| `dist/` | Built Python packages (wheel, tar.gz) |
| `htmlcov/` | HTML coverage reports |
| `scptensor.egg-info/` | Package metadata |
| `docs/_build/` | Built Sphinx documentation |

---

**Document Maintainer:** ScpTensor Team
**Last Updated:** 2025-01-09
