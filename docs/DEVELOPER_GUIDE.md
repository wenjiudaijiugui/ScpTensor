# ScpTensor Developer Guide

**Version:** 0.1.0
**Last Updated:** 2025-01-09

---

## Table of Contents

1. [Project Setup](#project-setup)
2. [Code Organization](#code-organization)
3. [Development Workflow](#development-workflow)
4. [Code Style Guidelines](#code-style-guidelines)
5. [Testing Conventions](#testing-conventions)
6. [Documentation Standards](#documentation-standards)
7. [Adding New Features](#adding-new-features)
8. [Performance Guidelines](#performance-guidelines)
9. [Release Process](#release-process)
10. [Resources](#resources)

---

## Project Setup

### Prerequisites

- Python 3.12 or 3.13
- `uv` - Fast Python package manager (recommended)
- Git

### First-Time Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/ScpTensor.git
cd ScpTensor

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# Install development dependencies
uv pip install -e ".[dev]"
```

### Installing Pre-commit Hooks

```bash
# Install pre-commit hooks (automatically runs linters/formatters before commits)
uv run pre-commit install
```

See [Pre-commit Hooks Guide](./DEVELOPMENT.md) for detailed information.

### Running Tests

```bash
# Run all tests
uv run pytest tests/

# Run with coverage report
uv run pytest --cov=scptensor --cov-report=html --cov-report=term-missing

# Run specific test file
uv run pytest tests/core/test_structures.py -v

# Run tests with specific marker
uv run pytest -m unit
uv run pytest -m integration
uv run pytest -m "not slow"
```

### Building Documentation

```bash
# Install documentation dependencies
uv pip install -e ".[docs]"

# Build documentation (from docs/ directory)
cd docs
make html
```

---

## Code Organization

### Directory Structure

```
scptensor/
├── core/              # Core data structures and exceptions
│   ├── __init__.py
│   ├── structures.py  # ScpContainer, Assay, ScpMatrix, MaskCode
│   ├── exceptions.py  # Exception hierarchy
│   ├── matrix_ops.py  # Matrix operations
│   ├── reader.py      # Data ingestion
│   └── utils.py       # Utility functions
├── normalization/     # Data normalization methods
├── impute/           # Missing value imputation
├── integration/      # Batch effect correction
├── qc/               # Quality control
├── dim_reduction/    # PCA, UMAP
├── cluster/          # Clustering algorithms
├── diff_expr/        # Differential expression
├── feature_selection/ # Feature selection methods
├── standardization/  # Data standardization
├── utils/            # General utilities
├── viz/              # Visualization tools
├── referee/          # Referee-related methods
├── benchmark/        # Benchmarking tools
└── __init__.py
```

### Module Responsibilities

| Module | Responsibility |
|--------|----------------|
| `core/` | Base data structures (`ScpContainer`, `Assay`, `ScpMatrix`), exceptions, matrix operations |
| `normalization/` | Data distribution transformations (log, scaling, centering) |
| `impute/` | Missing value filling (KNN, PPCA, SVD, MissForest) |
| `integration/` | Batch effect correction (ComBat, Harmony, MNN, Scanorama) |
| `qc/` | Quality control for samples and features |
| `dim_reduction/` | Dimensionality reduction (PCA, UMAP) |
| `cluster/` | Clustering algorithms (KMeans, graph-based) |
| `viz/` | Visualization with SciencePlots style |

### Data Structure Hierarchy

```
ScpContainer (top-level)
|
├── obs: pl.DataFrame           # Sample metadata (n_samples x metadata)
├── assays: Dict[str, Assay]    # Named assay registry
├── links: List[AggregationLink] # Feature aggregation relationships
└── history: List[ProvenanceLog] # Operation audit trail

Assay (feature-space)
|
├── var: pl.DataFrame            # Feature metadata (n_features x metadata)
└── layers: Dict[str, ScpMatrix] # Named layer registry

ScpMatrix (physical storage)
|
├── X: Union[np.ndarray, sp.spmatrix]  # Values (dense or sparse)
├── M: Union[np.ndarray, sp.spmatrix, None]  # Mask codes
└── metadata: MatrixMetadata     # Quality scores
```

### Mask Code System

The mask matrix `M` tracks the provenance of each value using integer codes:

| Code | Name | Description |
|------|------|-------------|
| 0 | VALID | Valid, detected values |
| 1 | MBR | Match Between Runs missing |
| 2 | LOD | Below Limit of Detection |
| 3 | FILTERED | Filtered out (quality control) |
| 4 | OUTLIER | Statistical outlier |
| 5 | IMPUTED | Imputed/filled value |
| 6 | UNCERTAIN | Uncertain data quality |

**Always update the mask when modifying values!**

```python
from scptensor.core.structures import MaskCode

# When imputing values, set mask to IMPUTED
M[missing_indices] = MaskCode.IMPUTED

# When filtering samples, set mask to FILTERED
M[outlier_samples, :] = MaskCode.FILTERED
```

---

## Development Workflow

### Git Workflow

1. Create a feature branch from `main`
2. Make your changes
3. Pre-commit hooks run automatically
4. Push and create a pull request

```bash
# Create feature branch
git checkout -b feature/my-feature

# Make changes and commit
git add .
git commit -m "feat: add my new feature"

# Push to remote
git push origin feature/my-feature
```

### Pre-commit Hooks

Pre-commit hooks automatically run before each commit:
- **Ruff** - Linting and formatting
- **MyPy** - Type checking
- **General hooks** - Trailing whitespace, YAML/TOML validation

If hooks fail:
1. Review the error messages
2. Some issues are auto-fixed (formatting)
3. Commit again after fixing issues

### Manual Code Quality Checks

```bash
# Run ruff linter
uv run ruff check .

# Auto-fix ruff issues
uv run ruff check --fix .

# Format code with ruff
uv run ruff format .

# Run mypy type checker
uv run mypy scptensor

# Run all pre-commit hooks manually
uv run pre-commit run --all-files
```

---

## Code Style Guidelines

### Type Hints

All public APIs must have complete type annotations:

```python
from typing import Any
import numpy as np
import polars as pl
from scptensor.core.structures import ScpContainer

def my_function(
    container: ScpContainer,
    assay_name: str,
    param1: float,
    param2: int | None = None,
) -> ScpContainer:
    """
    Brief description of what the function does.

    Args:
        container: The input container.
        assay_name: Name of the assay to process.
        param1: Description of param1.
        param2: Optional description of param2.

    Returns:
        The processed container.

    Raises:
        AssayNotFoundError: If assay_name doesn't exist.
        ValueError: If param1 is invalid.
    """
    # Implementation
    return container
```

### Import Organization

Imports are organized by Ruff automatically. Manual order:

1. Standard library imports
2. Third-party imports
3. Local imports (scptensor)

```python
# Standard library
from dataclasses import dataclass
from typing import Any

# Third-party
import numpy as np
import polars as pl

# Local
from scptensor.core.structures import ScpContainer
```

### Naming Conventions

- **Functions:** `snake_case`
- **Classes:** `PascalCase`
- **Constants:** `UPPER_SNAKE_CASE`
- **Private:** `_leading_underscore`

### Line Length

- Maximum line length: 100 characters
- Enforced by Ruff formatter

---

## Testing Conventions

### Test Organization

Tests are organized in `tests/` mirroring the `scptensor/` structure:

```
tests/
├── core/              # Tests for scptensor.core
├── normalization/     # Tests for scptensor.normalization
├── impute/           # Tests for scptensor.impute
├── conftest.py       # Shared fixtures
└── ...
```

### Writing Tests

Use pytest for testing:

```python
import pytest
import numpy as np
from scptensor.core.structures import ScpContainer, Assay, ScpMatrix

def test_log_normalize_basic():
    """Test basic log normalization."""
    # Arrange
    container = create_test_container()

    # Act
    result = log_normalize(container, assay_name="proteins")

    # Assert
    assert "log" in result.assays["proteins"].layers
    assert result.assays["proteins"].layers["log"].X is not None

def test_log_normalize_invalid_base():
    """Test that invalid base raises ValueError."""
    container = create_test_container()

    with pytest.raises(ValueError, match="positive"):
        log_normalize(container, assay_name="proteins", base=-1.0)
```

### Test Fixtures

Use `conftest.py` for shared fixtures:

```python
import pytest
import numpy as np
import polars as pl
from scptensor.core.structures import ScpContainer, Assay, ScpMatrix

@pytest.fixture
def sample_container():
    """Create a sample container for testing."""
    n_samples, n_features = 10, 20
    X = np.random.rand(n_samples, n_features)
    obs = pl.DataFrame({
        "_index": [f"S{i}" for i in range(n_samples)],
        "batch": ["A"] * 5 + ["B"] * 5
    })
    var = pl.DataFrame({
        "_index": [f"P{i}" for i in range(n_features)]
    })
    matrix = ScpMatrix(X=X)
    assay = Assay(var=var, layers={"raw": matrix})
    return ScpContainer(obs=obs, assays={"proteins": assay})
```

### Test Markers

Use pytest markers for categorizing tests:

```python
@pytest.mark.unit
def test_function():
    """Unit test - fast, isolated tests."""
    pass

@pytest.mark.integration
def test_pipeline():
    """Integration test - tests module interactions."""
    pass

@pytest.mark.slow
def test_large_dataset():
    """Slow test - mark to skip in quick runs."""
    pass
```

### Module Self-Tests

For quick module verification, add tests in `if __name__ == "__main__"` block:

```python
def my_function(x: int) -> int:
    return x * 2

if __name__ == "__main__":
    # Quick self-test
    assert my_function(5) == 10
    assert my_function(0) == 0
    print("All tests passed!")
```

---

## Documentation Standards

### Docstrings

Use NumPy-style docstrings:

```python
def log_normalize(
    container: ScpContainer,
    assay_name: str = "protein",
    base_layer: str = "raw",
    new_layer_name: str = "log",
    base: float = 2.0,
    offset: float = 1.0,
) -> ScpContainer:
    """
    Apply logarithmic transformation to data.

    This function supports both dense and sparse matrices. For sparse matrices,
    the log transformation is applied only to non-zero elements, preserving
    sparsity for efficiency.

    Parameters
    ----------
    container : ScpContainer
        Input container with the assay to transform.
    assay_name : str, default="protein"
        Name of the assay to transform.
    base_layer : str, default="raw"
        Name of the layer to use as input.
    new_layer_name : str, default="log"
        Name of the new layer to create.
    base : float, default=2.0
        Log base for transformation.
    offset : float, default=1.0
        Offset added before logging to handle zeros.

    Returns
    -------
    ScpContainer
        Container with the new normalized layer added.

    Raises
    ------
    AssayNotFoundError
        If the specified assay does not exist.
    LayerNotFoundError
        If the specified layer does not exist.
    ValueError
        If base or offset parameters are invalid.

    Examples
    --------
    >>> container = log_normalize(container, assay_name="proteins")
    >>> "log" in container.assays["proteins"].layers
    True
    """
    # Implementation
```

### English-Only Documentation

All documentation must be in English:
- Docstrings
- Comments
- README files
- Variable names (transliterated Chinese is discouraged)

### Visualization Style

All plots must use SciencePlots style with English-only labels:

```python
import matplotlib.pyplot as plt
import scienceplots

# Apply style
plt.style.use(["science", "no-latex"])

fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_xlabel("Samples")  # English only
ax.set_ylabel("Expression")
ax.set_title("Gene Expression Distribution")

plt.savefig("output.png", dpi=300)
```

---

## Adding New Features

### Adding a New Normalization Method

1. Create a new file in `scptensor/normalization/`:

```python
# scptensor/normalization/my_method.py

import numpy as np
from scptensor.core.structures import ScpContainer, ScpMatrix
from scptensor.core.exceptions import AssayNotFoundError, LayerNotFoundError
from scptensor.core.sparse_utils import is_sparse_matrix

def my_normalize(
    container: ScpContainer,
    assay_name: str = "protein",
    base_layer: str = "raw",
    new_layer_name: str = "my_normalized",
    param1: float = 1.0,
) -> ScpContainer:
    """
    Apply my normalization method to data.

    Parameters
    ----------
    container : ScpContainer
        Input container.
    assay_name : str, default="protein"
        Name of assay to normalize.
    base_layer : str, default="raw"
        Source layer name.
    new_layer_name : str, default="my_normalized"
        Output layer name.
    param1 : float, default=1.0
        Method-specific parameter.

    Returns
    -------
    ScpContainer
        Container with new layer added.
    """
    # Validate assay and layer exist
    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]
    if base_layer not in assay.layers:
        raise LayerNotFoundError(base_layer, assay_name)

    # Get input data
    input_matrix = assay.layers[base_layer]
    X = input_matrix.X
    M = input_matrix.M

    # Apply normalization (handle sparse/dense)
    if is_sparse_matrix(X):
        # Sparse implementation
        X_normalized = _sparse_normalize(X, param1)
    else:
        # Dense implementation
        X_normalized = _dense_normalize(X, param1)

    # Create new layer
    new_matrix = ScpMatrix(
        X=X_normalized,
        M=M.copy() if M is not None else None
    )
    assay.add_layer(new_layer_name, new_matrix)

    # Log operation for provenance
    container.log_operation(
        action="my_normalize",
        params={
            "assay": assay_name,
            "base_layer": base_layer,
            "new_layer": new_layer_name,
            "param1": param1,
        },
        description=f"My normalization applied to {assay_name}/{base_layer}.",
    )

    return container

def _sparse_normalize(X, param1):
    """Sparse implementation helper."""
    # Implementation
    pass

def _dense_normalize(X, param1):
    """Dense implementation helper."""
    # Implementation
    pass

if __name__ == "__main__":
    # Self-test
    print("Self-test passed!")
```

2. Export in `scptensor/normalization/__init__.py`:

```python
from scptensor.normalization.my_method import my_normalize

__all__ = ["my_normalize", ...]
```

3. Add tests in `tests/normalization/test_my_method.py`

### Adding a New Imputation Method

Follow the same pattern as normalization, but ensure:

1. **Update mask codes:** Set `M[imputed_indices] = MaskCode.IMPUTED`
2. **Preserve original mask:** Keep track of what was originally missing

```python
from scptensor.core.structures import MaskCode

# Get missing mask
missing_mask = M != MaskCode.VALID

# Perform imputation
X_imputed = imputation_function(X, missing_mask)

# Update mask to mark imputed values
if new_M is None:
    new_M = M.copy() if M is not None else np.zeros_like(X, dtype=np.int8)
new_M[missing_mask] = MaskCode.IMPUTED
```

### API Design Patterns

**Functional Pattern (Preferred):**
Functions return new objects, never modify in-place:

```python
# Good: Functional pattern
container = log_normalize(container, assay_name="proteins")

# Bad: In-place modification (breaks provenance)
log_normalize_inplace(container, assay_name="proteins")
```

**Layer Creation:**
Always create new layers rather than modifying existing ones:

```python
# Good: Create new layer
new_matrix = ScpMatrix(X=X_normalized, M=M.copy())
assay.add_layer("normalized", new_matrix)

# Bad: Modify existing layer
assay.layers["raw"].X = X_normalized
```

**Provenance Logging:**
Always log operations for traceability:

```python
container.log_operation(
    action="function_name",
    params={
        "assay": assay_name,
        "layer": base_layer,
        # ... all relevant parameters
    },
    description="Human-readable description.",
)
```

---

## Performance Guidelines

### When to Use Sparse Matrices

Use sparse matrices (`scipy.sparse.csr_matrix`) when:

- Data has >50% missing values
- Matrix has >10,000 features
- Memory usage is a concern

```python
from scipy import sparse
from scptensor.core.structures import ScpMatrix

# Create sparse matrix
X_sparse = sparse.csr_matrix((data, (rows, cols)), shape=(n, m))
matrix = ScpMatrix(X=X_sparse, M=M_sparse)
```

### JIT Compilation with Numba

For hot loops in performance-critical paths:

```python
from numba import jit
import numpy as np

@jit(nopython=True, cache=True)
def fast_mask_operation(M: np.ndarray) -> np.ndarray:
    """
    Fast mask operation with JIT compilation.

    Note: Only use types supported by Numba (no Polars, no scipy.sparse).
    """
    result = np.zeros_like(M)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] == 0:
                result[i, j] = 1
    return result
```

### Profiling and Benchmarking

Profile before optimizing:

```python
import cProfile
import pstats

def profile_my_function():
    """Profile a function to find bottlenecks."""
    profiler = cProfile.Profile()
    profiler.enable()

    # Run function
    my_function(large_dataset)

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats(pstats.SortKey.CUMULATIVE)
    stats.print_stats(20)  # Top 20 functions
```

Use the benchmark module for performance tests:

```python
from scptensor.benchmark import BenchmarkSuite

suite = BenchmarkSuite()
suite.add_case("my_function", my_function, {"size": 1000})
suite.run()
```

### Memory Efficiency

- Use generators for large datasets
- Use sparse matrices for sparse data
- Free memory with `del` when done with large objects

```python
# Generator pattern for processing chunks
def process_in_chunks(data, chunk_size=1000):
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        yield process_chunk(chunk)
```

---

## Release Process

### Version Bumping

Update version in `pyproject.toml`:

```toml
[project]
name = "scptensor"
version = "0.2.0"  # Bump version
```

### Changelog

Maintain `CHANGELOG.md`:

```markdown
## [0.2.0] - 2025-01-15

### Added
- New normalization method: quantile normalization
- Support for sparse matrices in imputation

### Fixed
- Memory leak in PCA computation
- Mask code handling in batch correction

### Changed
- Improved error messages for missing assays
```

### Release Steps

1. Update version
2. Update CHANGELOG
3. Run full test suite
4. Build documentation
5. Create git tag
6. Build and publish to PyPI

```bash
# Run tests
uv run pytest tests/ --cov=scptensor

# Build documentation
cd docs && make html

# Create release commit
git add .
git commit -m "chore: release v0.2.0"
git tag v0.2.0

# Build package
uv build

# Publish (when PyPI access is configured)
uv publish
```

---

## Resources

### Internal Documentation

- [Design Docs Index](./design/INDEX.md) - Navigation hub for design documents
- [Architecture Specification](./design/ARCHITECTURE.md) - Technical specifications
- [API Reference](./design/API_REFERENCE.md) - Complete API documentation
- [Development (Pre-commit Hooks)](./DEVELOPMENT.md) - Pre-commit configuration
- [Issues and Limitations](./ISSUES_AND_LIMITATIONS.md) - Known problems

### External Documentation

- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [MyPy Documentation](https://mypy.readthedocs.io/)
- [Pytest Documentation](https://docs.pytest.org/)
- [SciencePlots](https://scienceplots.readthedocs.io/)
- [Polars Documentation](https://pola.rs/)

### Progress Documentation

- Use the doc loader for accessing design docs:
  ```bash
  python3 scripts/doc_loader.py INDEX 1-100
  python3 scripts/doc_loader.py ARCHITECTURE 310-350
  ```

---

## Quick Reference

### Common Imports

```python
# Core structures
from scptensor.core.structures import (
    ScpContainer,
    Assay,
    ScpMatrix,
    MaskCode,
    ProvenanceLog,
    MatrixMetadata,
    AggregationLink,
)

# Exceptions
from scptensor.core.exceptions import (
    ScpTensorError,
    AssayNotFoundError,
    LayerNotFoundError,
    ValidationError,
    DimensionError,
)

# Utilities
from scptensor.core.sparse_utils import (
    is_sparse_matrix,
    ensure_sparse_format,
    sparse_safe_log1p,
)
```

### Common Patterns

```python
# Create a container
container = ScpContainer(
    obs=obs_df,
    assays={"proteins": protein_assay},
)

# Access data
X = container.assays["proteins"].layers["raw"].X
M = container.assays["proteins"].layers["raw"].M

# Add new layer
new_matrix = ScpMatrix(X=X_new, M=M_new)
container.assays["proteins"].add_layer("processed", new_matrix)

# Log operation
container.log_operation(
    action="my_action",
    params={"param1": value1},
    description="Description of what was done.",
)
```

---

**For questions or issues, please open an issue on GitHub.**
