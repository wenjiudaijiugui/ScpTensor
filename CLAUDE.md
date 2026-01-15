# ScpTensor Project Instructions

**Project:** ScpTensor - Single-Cell Proteomics Analysis Framework
**Version:** v0.1.0-beta
**Last Updated:** 2026-01-14

---

## Project-Specific Context

ScpTensor is a Python library for single-cell proteomics (SCP) data analysis with a hierarchical data structure (`ScpContainer` → `Assay` → `ScpMatrix`) and comprehensive analysis tools (QC, normalization, imputation, batch correction, dimensionality reduction, clustering, feature selection, differential expression).

**Current Status:** Functional framework with comprehensive analysis capabilities. Core modules are complete and tested. Includes benchmarking suite with competitor comparison, tutorial notebooks, and CI/CD pipeline.

---

## Design Documentation - Progressive Loading System

**CRITICAL:** This project has extensive design documentation (~4000 lines total). **NEVER load all design documents at once.** Always use the progressive loading system.

### Quick Start

```bash
# Step 1: Always start with INDEX (find what you need)
python3 scripts/doc_loader.py INDEX 1-100

# Step 2: Load specific sections only
python3 scripts/doc_loader.py <DOC> <LINE_RANGE>

# Example: Load executive summary (30 lines)
python3 scripts/doc_loader.py MASTER 1-30

# Example: Load normalization module spec (40 lines)
python3 scripts/doc_loader.py ARCHITECTURE 310-350
```

### Available Commands

```bash
# Load line range
python3 scripts/doc_loader.py <DOC> <START>-<END>

# Search for keyword
python3 scripts/doc_loader.py <DOC> search "<keyword>"

# Get document outline
python3 scripts/doc_loader.py <DOC> outline

# Count lines
python3 scripts/doc_loader.py <DOC> count

# Load full document (use sparingly!)
python3 scripts/doc_loader.py <DOC> all
```

### Available Documents

| Document | Lines | Purpose | Command Example |
|----------|-------|---------|-----------------|
| **INDEX** | 400 | Navigation hub - START HERE! | `doc_loader.py INDEX 1-100` |
| **MASTER** | 639 | Strategic overview, priorities | `doc_loader.py MASTER 1-50` |
| **ARCHITECTURE** | 1100 | Module specs, data structures | `doc_loader.py ARCHITECTURE 1-100` |
| **ROADMAP** | 700 | Execution plan, milestones | `doc_loader.py ROADMAP 51-150` |
| **MIGRATION** | 600 | Upgrade guide alpha→beta | `doc_loader.py MIGRATION 1-50` |
| **API_REFERENCE** | 900 | Complete API documentation | `doc_loader.py API_REFERENCE 81-200` |

### Common Workflows

#### When Designing New Features
```bash
# 1. Check current status
python3 scripts/doc_loader.py MASTER 1-50

# 2. Find module responsibility
python3 scripts/doc_loader.py ARCHITECTURE 1-100

# 3. Learn extension points
python3 scripts/doc_loader.py ARCHITECTURE 751-850

# 4. See similar APIs
python3 scripts/doc_loader.py API_REFERENCE 201-350
```

#### When Fixing Bugs
```bash
# 1. Check if known issue
python3 scripts/doc_loader.py INDEX search "bug"

# 2. Understand error handling
python3 scripts/doc_loader.py ARCHITECTURE 551-650

# 3. Check API contract
python3 scripts/doc_loader.py API_REFERENCE search "<function>"
```

#### When Planning Work
```bash
# 1. See priority matrix
python3 scripts/doc_loader.py ROADMAP 51-150

# 2. Check current sprint
python3 scripts/doc_loader.py ROADMAP 360-380

# 3. Review risks
python3 scripts/doc_loader.py ROADMAP 501-600
```

### Detailed Usage Guide

See `scripts/README.md` for complete documentation, examples, and troubleshooting.

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

### 1. Before Starting Work

```bash
# Check project status
python3 scripts/doc_loader.py MASTER 1-50

# Check what's planned
python3 scripts/doc_loader.py ROADMAP 51-150

# Check if task is in roadmap
python3 scripts/doc_loader.py ROADMAP search "<task>"
```

### 2. Design Phase

```bash
# Load architecture specs
python3 scripts/doc_loader.py ARCHITECTURE 1-100

# Check extension points
python3 scripts/doc_loader.py ARCHITECTURE 751-850

# See similar APIs
python3 scripts/doc_loader.py API_REFERENCE 201-350
```

### 3. Implementation Phase

Follow project coding standards:
- Add type hints to all functions
- Use functional pattern (return new objects)
- Update ProvenanceLog
- Add NumPy-style docstrings

### 4. Testing Phase

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

```bash
# 1. Check existing module
python3 scripts/doc_loader.py ARCHITECTURE 310-350

# 2. Check extension points
python3 scripts/doc_loader.py ARCHITECTURE 751-850

# 3. See API examples
python3 scripts/doc_loader.py API_REFERENCE 220-270

# Implementation location: scptensor/normalization/your_method.py
```

### Understanding Data Flow

```bash
# Load architecture overview
python3 scripts/doc_loader.py MASTER 78-150

# Load integration patterns
python3 scripts/doc_loader.py ARCHITECTURE 451-550
```

### Running Tutorials

The project includes 4 tutorial notebooks in `docs/tutorials/`:

1. **tutorial_01_getting_started.ipynb** - Basic data loading and exploration
2. **tutorial_02_qc_normalization.ipynb** - Quality control and normalization
3. **tutorial_03_imputation_integration.ipynb** - Missing value imputation and batch correction
4. **tutorial_04_clustering_visualization.ipynb** - Clustering and visualization

Run with Jupyter:
```bash
uv run jupyter notebook docs/tutorials/
```

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

### Documentation
```
docs/
├── ISSUES_AND_LIMITATIONS.md  # Known problems (READ THIS!)
├── tutorials/                 # 4 Jupyter notebook tutorials
└── design/
    ├── INDEX.md                # Design doc navigation (START HERE!)
    ├── MASTER.md               # Strategic overview
    ├── ARCHITECTURE.md         # Technical specifications
    ├── ROADMAP.md              # Execution plan
    ├── MIGRATION.md            # Upgrade guide
    └── API_REFERENCE.md        # Complete API
```

### Scripts
```
scripts/
├── doc_loader.py              # Progressive doc loader (USE THIS!)
└── README.md                  # Detailed usage guide
```

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

---

## When to Use Progressive Loading

### DO Use Progressive Loading For:

- Understanding module architecture
- Learning API specifications
- Planning implementation
- Debugging by checking design docs
- Reviewing priority matrix
- Understanding data structures
- Finding best practices

### DON'T Load Full Documents For:

- Quick syntax checks (use API_REFERENCE specific sections)
- Simple parameter lookups (use search instead)
- Routine tasks (memorize common patterns)

---

## Context Management Strategy

### When AI Asks About Design

1. **Check INDEX first** to locate relevant section
2. **Load only that section** (typically 50-200 lines)
3. **If more needed**, load adjacent sections incrementally

**Example:**
```
User: "How do I add a new normalization method?"

Step 1: Check INDEX
→ python3 scripts/doc_loader.py INDEX search "normalization"
→ Found: ARCHITECTURE lines 310-350

Step 2: Load that section
→ python3 scripts/doc_loader.py ARCHITECTURE 310-350

Step 3: Load API examples
→ python3 scripts/doc_loader.py API_REFERENCE 220-270

Total loaded: ~100 lines (instead of 4000+)
```

---

## Troubleshooting Design Doc Issues

### Problem: Can't find information

**Solution:**
```bash
# 1. Search INDEX
python3 scripts/doc_loader.py INDEX search "<keyword>"

# 2. Search specific document
python3 scripts/doc_loader.py <DOC> search "<keyword>"

# 3. Get document outline
python3 scripts/doc_loader.py <DOC> outline
```

### Problem: Line numbers outdated

**Solution:**
```bash
# Recount lines
python3 scripts/doc_loader.py <DOC> count

# Update INDEX.md with correct line numbers
```

---

## Integration with Global CLAUDE.md

This project-level CLAUDE.md extends the global CLAUDE.md with ScpTensor-specific instructions. Global principles (YAGNI, dependency minimalism, type safety, English-only docs, SciencePlots) **MUST** be followed.

---

## Quick Command Reference

```bash
# Documentation
python3 scripts/doc_loader.py INDEX 1-100          # Load index
python3 scripts/doc_loader.py <DOC> <START>-<END>    # Load section
python3 scripts/doc_loader.py <DOC> search "<kw>"    # Search

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

# Tutorials
uv run jupyter notebook docs/tutorials/              # Launch notebooks

# Benchmarking
uv run python -m scptensor.benchmark.benchmark_suite
uv run python -m scptensor.benchmark.run_competitor_benchmark
```

---

**Last Updated:** 2026-01-14
**Version:** v0.1.0-beta
**Maintainer:** ScpTensor Team

For detailed doc loader usage, see: `scripts/README.md`
