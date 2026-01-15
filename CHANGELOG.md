# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### CI/CD
- GitHub Actions workflow for continuous integration (`ci.yml`)
- GitHub Actions workflow for continuous deployment (`cd.yml`)
- GitHub Actions workflow for nightly benchmarks (`nightly-benchmark.yml`)
- GitHub Actions workflow for dependency review (`dependency-review.yml`)
- Pre-commit hooks configuration with Ruff, Mypy, and general file checks

#### Documentation
- Sphinx documentation setup with Furo theme
- API documentation generation with autodoc-typehints
- Markdown support via MyST parser
- 4 comprehensive tutorial notebooks:
  - `tutorial_01_getting_started.ipynb` - Basic introduction to ScpTensor
  - `tutorial_02_qc_normalization.ipynb` - Quality control and normalization workflows
  - `tutorial_03_imputation_integration.ipynb` - Missing value imputation and batch correction
  - `tutorial_04_clustering_visualization.ipynb` - Dimensionality reduction, clustering, and visualization
- Quick start guide (`quickstart.rst`)
- Developer guide (`docs/DEVELOPER_GUIDE.md`)
- API quick reference (`docs/API_QUICK_REFERENCE.md`)
- Project status documentation (`docs/PROJECT_STATUS.md`)
- Project structure documentation (`docs/PROJECT_STRUCTURE.md`)
- Dataset registry (`docs/DATASETS.md`)
- Competitor benchmark documentation (`docs/COMPETITOR_BENCHMARK.md`)
- Progressive documentation loader (`scripts/doc_loader.py`)

#### Testing
- pytest infrastructure with coverage reporting
- 46 test files covering core functionality:
  - Core tests: `test_assay.py`, `test_container_basic.py`, `test_container_edge_cases.py`, `test_matrix.py`, `test_matrix_edge_cases.py`, `test_mask_codes.py`, `test_helpers.py`
  - Integration tests: `test_full_pipeline.py`, `test_pipeline.py`, `test_synthetic.py`, `test_workflows.py`
  - Module tests: `test_benchmark.py`, `test_cluster.py`, `test_competitor_benchmark.py`, `test_diff_expr.py`, `test_error_handling.py`, `test_feature_selection.py`, `test_filtering.py`, `test_impute.py`, `test_integration.py`, `test_matrix_ops_sparse.py`, `test_normalization.py`, `test_qc.py`, `test_sparse_utils.py`, `test_viz.py`
  - Utility tests: `test_utils_batch.py`, `test_utils_data_generator.py`, `test_utils_stats.py`, `test_utils_transform.py`
- Real data comparison tests (`tests/real_data_comparison/`)
- Benchmark tests: `benchmark_jit.py`, `benchmark_jit_integration.py`, `benchmark_jit_speedup.py`, `benchmark_log_jit.py`, `benchmark_sparse_operations.py`
- Test fixtures and configuration (`tests/conftest.py`)

#### Features
- **Feature Selection Module** (`scptensor/feature_selection/`):
  - Highly variable gene (HVG) selection (`hvg.py`)
  - Dropout-based feature selection (`dropout.py`)
  - Model-based feature selection (`model.py`)
  - Variance stabilizing transformation (`vst.py`)
  - Shared utilities (`_shared.py`)

- **Competitor Benchmarking** (`scptensor/benchmark/`):
  - Competitor benchmark suite (`competitor_suite.py`)
  - Competitor visualization (`competitor_viz.py`)
  - Competitor benchmark runner (`competitor_benchmark.py`)

- **Dataset Module** (`scptensor/datasets/`):
  - Dataset registry with multiple SCP datasets
  - Support for DIA, PlexDIA, PScope, SCoPE2, cell cycle, clinical, and spatial data
  - External dataset integrations
  - Example dataset loader (`_example.py`)

- **Differential Expression** (`scptensor/diff_expr/`):
  - Core differential expression analysis (`core.py`)

- **Quality Control** (`scptensor/qc/`):
  - Advanced QC metrics (`advanced.py`)
  - Outlier detection (`outlier.py`)

- **Integration Methods** (`scptensor/integration/`):
  - Harmony integration (`harmony.py`)
  - MNN (Mutual Nearest Neighbors) integration (`mnn.py`)
  - Scanorama integration (`scanorama.py`)
  - Nonlinear integration methods (`nonlinear.py`)

#### Performance
- **Sparse Matrix Optimizations** (`scptensor/core/sparse_utils.py`):
  - Sparse-aware matrix operations
  - Memory-efficient storage for sparse SCP data
  - Optimized mask operations for sparse matrices

- **JIT Compilation** (`scptensor/core/jit_ops.py`):
  - Numba JIT-compiled core operations
  - Accelerated logarithmic transformations
  - Fast median computations
  - Optimized statistical operations

- **Batch Processing** (`scptensor/utils/batch.py`):
  - Memory-efficient batch processing utilities
  - Chunked operations for large datasets

#### Utilities
- Statistical utilities (`scptensor/utils/stats.py`)
- Transformation utilities (`scptensor/utils/transform.py`)
- Data generator (`scptensor/utils/data_generator.py`)

#### Scripts
- Progressive documentation loader (`scripts/doc_loader.py`)
- Competitor benchmark runner (`scripts/run_competitor_benchmark.py`)
- Sparse optimization verification (`scripts/verify_sparse_optimizations.py`)

### Changed
- **Core Module**:
  - Enhanced exception handling (`scptensor/core/exceptions.py`)
  - Improved matrix operations (`scptensor/core/matrix_ops.py`)
  - Updated core structures with type annotations (`scptensor/core/structures.py`)
  - New IO operations (`scptensor/core/io.py`)
  - Utility functions (`scptensor/core/utils.py`)

- **Normalization**: Updated all normalization methods with proper type hints and documentation
- **Imputation**: Updated all imputation methods with proper type hints and documentation
- **Dimensionality Reduction**: Updated PCA and UMAP with type annotations
- **Clustering**: Updated KMeans and graph clustering with type annotations
- **Visualization**: Updated all visualization modules with SciencePlots style
- **Benchmark**: Updated benchmark suite with performance metrics and visualization

### Fixed
- Mask code handling across all modules for proper provenance tracking
- Import errors in `integration/` and `qc/` modules (added `__init__.py` files)
- Type annotation inconsistencies across public APIs
- Documentation formatting issues

### Removed
- Old `dim_reduction/_umap.py` (consolidated into `umap.py`)
- Old `standardization/zscore.py` (moved to normalization module)
- Old `utils/data_genetator.py` (typo fixed to `data_generator.py`)

---

## [0.1.0-alpha] - 2025-01-05

### Added
- Initial release of ScpTensor
- Core data structures: `ScpContainer`, `Assay`, `ScpMatrix`
- Mask code system for provenance tracking (0=VALID, 1=MBR, 2=LOD, 3=FILTERED, 5=IMPUTED)
- Quality control module (basic metrics)
- 6 normalization methods: median centering, median scaling, z-score, log, TMM, upper quartile, sample mean/median
- 4 imputation methods: KNN, MissForest, PPCA, SVD
- ComBat batch effect correction
- PCA and UMAP dimensionality reduction
- KMeans and graph clustering (Leiden/Louvain support)
- Visualization module with SciencePlots style
- Basic benchmark suite
- Polars-based data handling for metadata

### Known Limitations
- Integration module incomplete (only ComBat implemented)
- QC module lacks comprehensive outlier detection
- Zero unit test coverage
- Limited documentation
- Performance optimizations not applied
- Feature selection module missing
- Differential expression analysis incomplete

---

[Unreleased]: https://github.com/yourusername/ScpTensor/compare/v0.1.0-alpha...HEAD
[0.1.0-alpha]: https://github.com/yourusername/ScpTensor/releases/tag/v0.1.0-alpha
