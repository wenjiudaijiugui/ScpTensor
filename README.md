# ScpTensor: A High-Performance Single-Cell Proteomics Analysis Framework

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Author:** Shenshang (ScpTensor Team)
**Date:** 2026-01-14
**Version:** 0.1.0-beta
**Status:** Production Ready (~80%)

---

## Abstract

ScpTensor is a cutting-edge Python library specifically designed for the rigorous analysis of single-cell proteomics (SCP) data. It introduces a structured data container, `ScpContainer`, optimized for multi-assay experiments, enabling efficient handling of complex datasets including peptides and proteins. The framework integrates a comprehensive suite of tools for quality control, imputation, normalization, batch effect correction, feature selection, dimensionality reduction, and clustering. By leveraging high-performance libraries such as Polars, NumPy, SciPy, and Numba JIT, ScpTensor ensures scalability and speed, making it an essential tool for modern computational biology research.

---

## Features

### Core Capabilities

| Category | Features |
|----------|----------|
| **Data Structures** | Hierarchical `ScpContainer` → `Assay` → `ScpMatrix` with provenance tracking |
| **Quality Control** | Sample-level filtering, outlier detection, missing value analysis |
| **Normalization** | Median centering, median scaling, z-score, log, TMM, upper quartile, global median |
| **Imputation** | KNN, SVD, PPCA, MissForest with mask-based provenance |
| **Integration** | ComBat, Harmony, MNN, Scanorama for batch correction |
| **Feature Selection** | Highly variable genes (HVG), variance stabilizing transformation (VST), dropout-based selection |
| **Dimensionality Reduction** | PCA, UMAP with sparse matrix support |
| **Clustering** | KMeans, graph-based (Leiden/Louvain) |
| **Visualization** | Publication-ready plots with SciencePlots style |
| **Benchmarking** | Comprehensive benchmarking suite with competitor comparisons |

### Performance Highlights

- **Sparse matrix optimization** for datasets with >50% missing values
- **Numba JIT compilation** for hot loops (up to 100x speedup)
- **Polars backend** for efficient metadata operations
- **774+ unit tests** with comprehensive coverage
- **Competitor benchmarks** against Scanpy and other SCP tools

---

## Installation

### Prerequisites

- Python 3.12 or higher
- `uv` package manager (recommended)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/yourusername/ScpTensor.git
cd ScpTensor

# Create a virtual environment with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install core dependencies
uv pip install -e .

# Optional: Install integration methods (Harmony, Scanorama)
uv pip install -e ".[integration]"

# Optional: Install benchmarking tools (Scanpy for comparison)
uv pip install -e ".[benchmark]"

# Optional: Install IO utilities (AnnData interoperability)
uv pip install -e ".[io]"

# Development installation (includes testing, linting, docs)
uv pip install -e ".[dev]"
```

### Verify Installation

```python
import scptensor as scp
print(f"ScpTensor version: {scp.__version__}")
print(f"Available modules: {', '.join(scp.__all__)}")
```

---

## Quick Start

### Basic Workflow

```python
import numpy as np
import polars as pl
from scptensor import ScpContainer, Assay, ScpMatrix

# 1. Create or load data
n_samples, n_features = 100, 500
X = np.random.rand(n_samples, n_features)

obs = pl.DataFrame({
    "_index": [f"S{i}" for i in range(n_samples)],
    "group": ["A"] * 50 + ["B"] * 50
})

var = pl.DataFrame({
    "_index": [f"P{i}" for i in range(n_features)]
})

# 2. Initialize container
matrix = ScpMatrix(X=X)
assay = Assay(var=var, layers={"raw": matrix})
container = ScpContainer(obs=obs, assays={"proteins": assay})

# 3. Quality control
from scptensor.qc import basic

# Filter samples with too many missing values
container = basic.filter_samples(
    container,
    assay_name="proteins",
    min_non_missing=0.5
)

# 4. Normalization
from scptensor.normalization import log

container = log.log_normalize(
    container,
    assay_name="proteins",
    layer_name="raw",
    new_layer_name="log"
)

# 5. Imputation
from scptensor.impute import knn

container = knn.knn_impute(
    container,
    assay_name="proteins",
    layer_name="log",
    new_layer_name="imputed"
)

# 6. Dimensionality reduction
from scptensor.dim_reduction import pca

result = pca.pca(
    container,
    assay_name="proteins",
    layer_name="imputed"
)

# 7. Clustering
from scptensor.cluster import kmeans

container = kmeans.kmeans_cluster(
    container,
    assay_name="proteins",
    layer_name="imputed",
    n_clusters=4
)

# 8. Visualization
from scptensor.viz.recipes import embedding

embedding.plot_embedding(
    container,
    assay_name="proteins",
    layer_name="imputed",
    color_by="group"
)
```

### Loading Real Data

```python
from scptensor.core.reader import read_sc_dataset

# Load from SDRF format (SCP standard)
container = read_sc_dataset(
    data_path="path/to/data.tsv",
    design_path="path/to/design.tsv"
)

# Or load from existing formats
from scptensor.io import from_anndata

container = from_anndata(adata)
```

---

## Competitor Benchmarking

ScpTensor includes a comprehensive benchmarking suite that compares performance against competing tools:

```python
from scptensor.benchmark import CompetitorBenchmarkSuite

# Define methods to compare
methods = {
    "scptensor": ["knn", "svd"],
    "scanpy": ["knn", "magic"],
}

# Run benchmark
suite = CompetitorBenchmarkSuite()
results = suite.run_benchmark(
    dataset=container,
    methods=methods,
    metrics=["rmse", "runtime", "memory"]
)

# Visualize results
from scptensor.benchmark import CompetitorResultVisualizer

viz = CompetitorResultVisualizer(results)
viz.plot_comparison()
viz.plot_speedup()
```

See [docs/COMPETITOR_BENCHMARK.md](docs/COMPETITOR_BENCHMARK.md) for detailed benchmark results and methodology.

---

## Documentation

### Available Documentation

| Resource | Description |
|----------|-------------|
| [API Quick Reference](docs/API_QUICK_REFERENCE.md) | Compact API documentation |
| [Developer Guide](docs/DEVELOPER_GUIDE.md) | Contributing guidelines |
| [Project Status](docs/PROJECT_STATUS.md) | Current development progress |
| [Datasets Guide](docs/DATASETS.md) | Available test datasets |
| [Competitor Benchmarks](docs/COMPETITOR_BENCHMARK.md) | Performance comparisons |

### Tutorials

Interactive Jupyter notebooks are available in `docs/tutorials/`:

1. **Getting Started** - Basic data structures and operations
2. **QC & Normalization** - Quality control workflows
3. **Imputation & Integration** - Handling missing values and batch effects
4. **Clustering & Visualization** - Downstream analysis

### Building Documentation

```bash
# Install documentation dependencies
uv pip install -e ".[docs]"

# Build HTML docs
cd docs
make html

# Serve locally (for development)
./serve.sh
```

---

## Development

### Running Tests

```bash
# Run all tests
uv run pytest tests/

# Run with coverage report
uv run pytest --cov=scptensor --cov-report=html

# Run specific test module
uv run pytest tests/core/test_container.py -v

# Run only unit tests
uv run pytest -m unit
```

### Code Quality

```bash
# Format code
uv run ruff format scptensor/

# Lint code
uv run ruff check scptensor/

# Type checking
uv run mypy scptensor/
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run manually
uv run pre-commit run --all-files
```

---

## Project Status

**Current Version:** v0.1.0-beta
**Production Readiness:** ~80%

### Completed (P0)

- [x] Core data structures (ScpContainer, Assay, ScpMatrix)
- [x] Mask-based provenance tracking
- [x] Quality control module
- [x] Normalization methods (7 methods)
- [x] Imputation methods (KNN, SVD, PPCA, MissForest)
- [x] Integration methods (ComBat, Harmony, MNN, Scanorama)
- [x] Dimensionality reduction (PCA, UMAP)
- [x] Clustering (KMeans, graph-based)
- [x] Feature selection (HVG, VST, dropout-based)
- [x] Visualization suite
- [x] Benchmarking framework with competitor comparisons
- [x] 774+ unit tests
- [x] CI/CD pipeline

### In Progress (P1)

- [ ] Advanced differential expression methods
- [ ] GPU acceleration support
- [ ] Enhanced Scanpy ecosystem integration
- [ ] Real dataset validation suite

### Project Structure

```
scptensor/
├── core/           # Data structures and IO
├── qc/             # Quality control
├── normalization/  # 7 normalization methods
├── impute/         # 4 imputation methods
├── integration/    # Batch correction (4 methods)
├── feature_selection/  # HVG, VST, dropout
├── dim_reduction/  # PCA, UMAP
├── cluster/        # KMeans, graph clustering
├── viz/            # Visualization tools
├── benchmark/      # Benchmarking + competitor suite
├── datasets/       # Example datasets
└── utils/          # Helper functions
```

---

## Citation

If you use ScpTensor in your research, please cite:

```bibtex
@software{scptensor2026,
  author = {Shenshang and ScpTensor Team},
  title = {ScpTensor: A High-Performance Single-Cell Proteomics Analysis Framework},
  year = {2026},
  version = {0.1.0-beta},
  url = {https://github.com/yourusername/ScpTensor}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome! Please see [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) for guidelines.

---

## Contact

For issues, questions, or contributions, please visit the [GitHub repository](https://github.com/yourusername/ScpTensor).

---

*Generated by ScpTensor Team*
