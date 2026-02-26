# ScpTensor Project Instructions

**Project:** ScpTensor - Single-Cell Proteomics Analysis Framework
**Version:** v0.1.0-beta
**Last Updated:** 2026-01-14

---

## Project-Specific Context

ScpTensor is a Python library for single-cell proteomics (SCP) data analysis with a hierarchical data structure (`ScpContainer` → `Assay` → `ScpMatrix`) and comprehensive analysis tools (QC, normalization, imputation, batch correction, dimensionality reduction, clustering, feature selection, differential expression).

**Current Status:** Functional framework with comprehensive analysis capabilities. Core modules are complete and tested. Includes benchmarking suite with competitor comparison, tutorial notebooks, and CI/CD pipeline.

---

## Code Style and Standards

### Core Principles (from global CLAUDE.md)

- **YAGNI:** Never add code for "future might need" features
- **Avoid premature abstraction:** Less than 3 uses → don't extract
- **Dependency minimalism:** Use stdlib over external packages when possible

### Project-Specific Standards

1. **Type Safety:** All public APIs must have complete type annotations
   ```python
   def log_normalize(
       container: ScpContainer,
       assay_name: str,
       base_layer: str,
       new_layer_name: str = "log",
       base: float = 2.0,
       offset: float = 1.0
   ) -> ScpContainer:
   ```

2. **Immutable Pattern:** Functions create new layers, never modify in-place
   ```python
   # Good: Functional pattern
   container = log_normalize(container, ...)

   # Bad: In-place modification (breaks provenance)
   container.assays['proteins'].layers['log'] = ...
   ```

3. **Documentation:** English-only docstrings (NumPy style)
   ```python
   def log_normalize(...) -> ScpContainer:
       """
       Apply logarithmic transformation to data.

       Parameters
       ----------
       container : ScpContainer
           Input container
       ...
       """
   ```

4. **Testing:** Full pytest infrastructure with 26+ test files

---

## Module Organization

### Current Structure

```
scptensor/
├── core/                 # Data structures (ScpContainer, Assay, ScpMatrix)
│   ├── structures.py     # Main data classes
│   ├── exceptions.py     # Error classes
│   ├── matrix_ops.py     # Matrix operations
│   ├── io.py             # I/O utilities
│   ├── jit_ops.py        # JIT-compiled operations
│   └── sparse_utils.py   # Sparse matrix utilities
├── normalization/        # 8 normalization methods
├── impute/               # 4 imputation methods
├── integration/          # 5 batch correction methods
├── qc/                   # Quality control (basic + advanced)
├── dim_reduction/        # PCA, UMAP
├── cluster/              # KMeans, graph-based clustering
├── feature_selection/    # HVG, VST, dropout rate, model-based
├── diff_expr/            # Differential expression analysis
├── datasets/             # Example datasets and registry
├── utils/                # Batch processing, stats, transforms
├── viz/                  # Visualization (SciencePlots style)
│   ├── base/             # Base plotting functions
│   └── recipes/          # Pre-configured visualizations
├── benchmark/            # Benchmarking suite + competitor comparison
├── referee/              # Evaluation metrics
└── standardization/      # Data standardization methods
```

### Module Completion Status

| Module | Status | Notes |
|--------|--------|-------|
| core/ | Complete | Full data structure implementation |
| normalization/ | Complete | 8 methods implemented |
| impute/ | Complete | KNN, MissForest, PPCA, SVD |
| integration/ | Complete | ComBat, Harmony, MNN, Scanorama, nonlinear |
| qc/ | Complete | Basic QC + advanced outlier detection |
| dim_reduction/ | Complete | PCA, UMAP |
| cluster/ | Complete | KMeans, graph clustering |
| feature_selection/ | Complete | HVG, VST, dropout, model-based |
| diff_expr/ | Complete | Core differential expression |
| viz/ | Complete | Base plots + recipes |
| benchmark/ | Complete | Suite + competitor comparison |
| datasets/ | Complete | Example datasets with registry |
| utils/ | Complete | Batch, stats, transform utilities |

### Testing Infrastructure

- **26+ test files** covering all major modules
- **pytest** configuration with coverage reporting
- **Unit, integration, and slow test markers**
- Tests located in `tests/` directory

Run tests:
```bash
uv run pytest                         # Run all tests
uv run pytest tests/test_core/        # Run specific module tests
uv run pytest -m "not slow"           # Skip slow tests
uv run pytest --cov=scptensor         # With coverage report
```

---

## Development Workflow

### 1. Implementation Phase

Follow project coding standards:
- Add type hints to all functions
- Use functional pattern (return new objects)
- Update ProvenanceLog
- Add NumPy-style docstrings

### 2. Testing Phase

**Testing infrastructure is now available.**

Write tests in `tests/test_<module>.py`:
```python
import pytest
from scptensor import create_test_container

def test_basic_functionality():
    """Test basic functionality."""
    container = create_test_container()
    assert container is not None
    assert container.n_samples > 0
```

Run tests with coverage:
```bash
uv run pytest --cov=scptensor --cov-report=html
```

---

## CI/CD Pipeline

The project uses GitHub Actions for continuous integration:

### Workflows

- **ci.yml**: Main CI pipeline running on every push
  - Runs tests across Python 3.12, 3.13
  - Checks code style with ruff
  - Type checking with mypy
  - Coverage reporting

- **cd.yml**: Continuous deployment
  - Builds and publishes packages
  - Runs on version tags

- **nightly-benchmark.yml**: Nightly benchmark runs
  - Performance regression detection
  - Competitor comparison updates

- **dependency-review.yml**: Security scanning
  - Checks for vulnerable dependencies

### Pre-commit Hooks

Configure pre-commit hooks (recommended):
```bash
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg
```

Hooks run on every commit:
- Ruff linting and formatting
- MyPy type checking
- YAML/TOML validation
- Trailing whitespace fix
- Large file detection

---

## Data Structures Overview

### Hierarchy

```
ScpContainer (top-level)
├── obs: pl.DataFrame           # Sample metadata (n_samples × metadata)
├── assays: Dict[str, Assay]    # Named assay registry
└── history: List[ProvenanceLog]  # Operation audit trail

Assay (feature-space)
├── var: pl.DataFrame            # Feature metadata (n_features × metadata)
└── layers: Dict[str, ScpMatrix] # Named layer registry

ScpMatrix (physical storage)
├── X: Union[np.ndarray, sp.spmatrix]  # Values
├── M: Union[np.ndarray, sp.spmatrix, None]  # Mask codes
└── metadata: MatrixMetadata     # Quality scores
```

### Mask Codes (Provenance Tracking)

- `0`: VALID (detected value)
- `1`: MBR (missing between runs)
- `2`: LOD (below detection limit)
- `3`: FILTERED (QC removed)
- `5`: IMPUTED (filled value)

**Always update mask when modifying values!**

---

## Common Tasks Quick Reference

### Adding a New Normalization Method

1. Check existing module structure in `scptensor/normalization/`
2. Follow the pattern of existing methods (e.g., `log_transform.py`)
3. Update `scptensor/normalization/__init__.py` to export new function
4. Add tests in `tests/test_normalization.py`

### Understanding Data Flow

1. Read core structures in `scptensor/core/structures.py`
2. Check module-specific patterns in relevant subdirectories
3. Review existing tests for usage examples

### Benchmarking

Run the benchmark suite:
```bash
# Run ScpTensor benchmarks
uv run python -m scptensor.benchmark.benchmark_suite

# Run competitor comparison
uv run python -m scptensor.benchmark.run_competitor_benchmark
```

---

## Visualization Standards

### Required Style

```python
import matplotlib.pyplot as plt
import scienceplots

# Apply style
plt.style.use(["science", "no-latex"])

# Set DPI
plt.savefig('output.png', dpi=300)
```

**Requirements:**
- Use SciencePlots style
- DPI = 300 for publication quality
- **NO Chinese characters in figures**
- English-only labels and text

### Available Visualizations

- `viz/base/scatter.py` - Scatter plots
- `viz/base/heatmap.py` - Heatmaps
- `viz/base/violin.py` - Violin plots
- `viz/recipes/qc.py` - QC-specific visualizations
- `viz/recipes/embedding.py` - Dimensionality reduction plots
- `viz/recipes/stats.py` - Statistical visualizations

---

## Performance Considerations

### Sparse Matrices

Use `scipy.sparse.csr_matrix` for data with >50% missing values:
```python
from scipy import sparse

X_sparse = sparse.csr_matrix((data, (rows, cols)), shape=(n, m))
matrix = ScpMatrix(X=X_sparse, M=M_sparse)
```

### Numba JIT

Hot loops use Numba JIT (automatically applied in critical paths):
```python
from numba import jit

@jit(nopython=True, cache=True)
def fast_mask_operation(M: np.ndarray) -> np.ndarray:
    ...
```

JIT-optimized operations are in `core/jit_ops.py`.

---

## Key File Locations

### Source Code
```
scptensor/
├── core/                      # Core structures + JIT + sparse utils
├── normalization/             # 8 normalization methods
├── impute/                    # 4 imputation methods
├── integration/               # 5 batch correction methods
├── qc/                        # Quality control
├── dim_reduction/             # PCA, UMAP
├── cluster/                   # KMeans, graph clustering
├── feature_selection/         # HVG, VST, dropout, model-based
├── diff_expr/                 # Differential expression
├── datasets/                  # Example datasets
├── utils/                     # Batch, stats, transforms
├── viz/                       # Visualization tools
├── benchmark/                 # Benchmarking + competitor comparison
├── referee/                   # Evaluation metrics
└── standardization/           # Standardization methods
```

### Tests
```
tests/
├── conftest.py                # Pytest fixtures
├── core/                      # Core module tests
├── test_*.py                  # Module-specific tests
├── integration/               # Integration tests
└── real_data_comparison/      # Real data validation
```

### Scripts
```
scripts/
├── performance_benchmark.py   # Performance benchmarking
└── README.md                  # Usage guide
```

---

## Integration with Global CLAUDE.md

This project-level CLAUDE.md extends the global CLAUDE.md with ScpTensor-specific instructions. Global principles (YAGNI, dependency minimalism, type safety, English-only docs, SciencePlots) **MUST** be followed.

---

## Quick Command Reference

```bash
# Environment setup
uv sync                                               # Install dependencies
uv pip install -e ".[dev]"                           # Install with dev deps

# Development
uv run pytest tests/                                 # Run tests
uv run pytest --cov=scptensor                        # With coverage
uv run ruff check scptensor/                         # Lint
uv run ruff format scptensor/                        # Format
uv run mypy scptensor/                               # Type check

# Pre-commit
uv run pre-commit install                            # Install hooks
uv run pre-commit run --all-files                    # Run all hooks

# Benchmarking
uv run python -m scptensor.benchmark.benchmark_suite
uv run python -m scptensor.benchmark.run_competitor_benchmark

# Performance scripts
uv run python scripts/performance_benchmark.py
```

---

**Last Updated:** 2026-01-14
**Version:** v0.1.0-beta
**Maintainer:** ScpTensor Team
