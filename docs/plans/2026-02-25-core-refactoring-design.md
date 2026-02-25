# ScpTensor Core Refactoring Design Document

**Project:** ScpTensor v0.2.0 Refactoring
**Date:** 2026-02-25
**Status:** Design Phase
**Author:** ScpTensor Team

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Quantitative Objectives](#quantitative-objectives)
3. [Refactoring Principles](#refactoring-principles)
4. [Current Codebase Analysis](#current-codebase-analysis)
5. [Team Division and Responsibilities](#team-division-and-responsibilities)
6. [Refactoring Strategy](#refactoring-strategy)
7. [Module-by-Module Specifications](#module-by-module-specifications)
8. [API Simplification](#api-simplification)
9. [Mathematical Verification](#mathematical-verification)
10. [Implementation Workflow](#implementation-workflow)
11. [Quality Assurance](#quality-assurance)
12. [Timeline and Milestones](#timeline-and-milestones)

---

## Executive Summary

### Overview

This document outlines a comprehensive refactoring initiative for the ScpTensor codebase, targeting a **30% reduction in code volume** while improving code quality, simplifying APIs, and ensuring mathematical correctness.

### Strategic Priorities

We prioritize refactoring goals in the following order:

1. **Code Quality** - Maintainability, readability, testability
2. **API Simplification** - User experience, learning curve
3. **Mathematical Verification** - Algorithm correctness, numerical stability
4. **Performance Optimization** - Speed, memory efficiency

### Scope

- **Full Coverage:** All core modules will be refactored
- **Bold Refactoring:** We are not afraid to make breaking changes
- **Best Practices:** Follow Python community standards and patterns

---

## Quantitative Objectives

### Code Metrics

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| Total Lines of Code | 34,223 | ~24,000 | -30% |
| Cyclomatic Complexity | Unknown | < 10 | - |
| Code Duplication | Unknown | < 3% | - |
| Type Coverage | ~70% | 95%+ | +25% |
| Test Coverage | ~60% | 85%+ | +25% |

### Module-Specific Targets

| Module | Current LOC | Target LOC | Reduction |
|--------|-------------|------------|-----------|
| `core/` | 5,475 | ~3,800 | -30% |
| `normalization/` | 658 | ~450 | -32% |
| `impute/` | 2,381 | ~1,600 | -33% |
| `integration/` | 1,708 | ~1,150 | -33% |
| `qc/` | 3,589 | ~2,400 | -33% |
| `dim_reduction/` | 1,105 | ~750 | -32% |
| `cluster/` | 618 | ~420 | -32% |
| `diff_expr/` | 4,185 | ~2,800 | -33% |
| `utils/` | 1,984 | ~1,350 | -32% |
| `viz/` | 10,094 | ~7,000 | -31% |
| `io/` | 1,914 | ~1,300 | -32% |
| `standardization/` | 218 | ~150 | -31% |

### User Experience Metrics

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| Average API Call Length | ~150 chars | ~90 chars | -40% |
| Required Parameters (avg) | 4.2 | 2.5 | -40% |
| Chaining Support | Partial | Full | ✓ |
| Type Inference | Manual | Auto | ✓ |

---

## Refactoring Principles

### 1. YAGNI (You Aren't Gonna Need It)

**Definition:** Never add code for "future might need" features.

**Implementation:**

- Delete unused functions and parameters
- Remove "just in case" abstractions
- Eliminate speculative features
- Remove commented-out code

**Example:**

```python
# Bad: Speculative abstraction
class DataValidator:
    def validate(self, data, strict=False, future_feature=None):
        # Validate data
        pass

# Good: Simple, focused function
def validate_data(data: np.ndarray) -> bool:
    """Validate data array."""
    return data.size > 0 and np.isfinite(data).all()
```

### 2. Single Responsibility Principle (SRP)

**Definition:** Each function should do one thing well.

**Implementation:**

- Functions < 50 lines
- One clear purpose
- Max 3 parameters (use data classes for more)
- Clear return types

**Example:**

```python
# Bad: Multiple responsibilities
def process_and_save(data, output_path, normalize=True, impute=False):
    if normalize:
        data = (data - data.mean()) / data.std()
    if impute:
        data = np.nan_to_num(data)
    np.save(output_path, data)
    return data

# Good: Single responsibility
def normalize_data(data: np.ndarray) -> np.ndarray:
    """Standardize data to zero mean and unit variance."""
    return (data - data.mean()) / data.std()

def save_data(data: np.ndarray, path: str) -> None:
    """Save array to file."""
    np.save(path, data)
```

### 3. Avoid Premature Abstraction

**Rule:** Extract only when used 3+ times.

**Implementation:**

- Copy-paste is OK for 1-2 uses
- Extract functions at third use
- Use type hints for clarity
- Prefer composition over inheritance

**Example:**

```python
# Bad: Premature abstraction
class MatrixOperationFactory:
    def create_normalizer(self, method):
        if method == 'log':
            return LogNormalizer()
        # ...

# Good: Direct functions
def log_normalize(data: np.ndarray, base: float = 2.0) -> np.ndarray:
    """Apply log transformation."""
    return np.log(data + 1.0) / np.log(base)
```

### 4. Functional Style

**Definition:** Prefer pure functions over mutable state.

**Implementation:**

- Immutable data structures where possible
- Return new objects instead of modifying
- Avoid side effects
- Explicit dependencies

**Example:**

```python
# Bad: In-place modification
def add_layer(container, layer_name, data):
    container.layers[layer_name] = data
    return container

# Good: Return new object
def with_layer(container: ScpContainer, name: str, data: np.ndarray) -> ScpContainer:
    """Return container with additional layer."""
    new_container = container.copy()
    new_container.layers[name] = data
    return new_container
```

---

## Current Codebase Analysis

### Module Breakdown

```
scptensor/
├── core/              # 5,475 lines - Data structures, I/O, JIT
├── normalization/     # 658 lines  - 8 normalization methods
├── impute/            # 2,381 lines - 4 imputation algorithms
├── integration/       # 1,708 lines - 5 batch correction methods
├── qc/                # 3,589 lines - Quality control
├── dim_reduction/     # 1,105 lines - PCA, UMAP
├── cluster/           # 618 lines  - Clustering algorithms
├── diff_expr/         # 4,185 lines - Differential expression
├── utils/             # 1,984 lines - Utilities
├── viz/               # 10,094 lines - Visualization
├── io/                # 1,914 lines - I/O operations
└── standardization/   # 218 lines  - Standardization methods
```

### Key Issues Identified

1. **High Complexity in core/**
   - Mixed responsibilities (data + I/O + JIT)
   - Long functions (>100 lines)
   - Complex inheritance

2. **Large viz/ Module**
   - 10,094 lines (29% of codebase)
   - Potential code duplication
   - Mixed base/recipes structure

3. **Inconsistent APIs**
   - Different parameter naming conventions
   - Inconsistent return types
   - Mixed functional/OO styles

4. **Limited Type Coverage**
   - Many functions lack type hints
   - Optional types not properly defined
   - Generic types underutilized

---

## Team Division and Responsibilities

### Team Structure (8 Members)

```
┌─────────────────────────────────────────────────────────┐
│                    Team Lead (Coordinator)              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Member 1    Member 2    Member 3    Member 4          │
│  core/       norm/       impute/      integration/      │
│  ↓           ↓           ↓            ↓                 │
│  5,475→3,800 658→450    2,381→1,600  1,708→1,150       │
│                                                         │
│  Member 5    Member 6    Member 7    Member 8          │
│  qc/         dr+clust    feat+diff    util+viz          │
│  ↓           ↓           ↓            ↓                 │
│  3,589→2,400 1,723→1,170 4,185→2,800 12,992→8,350      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Member 1: Core Module Specialist

**Responsibilities:**
- Refactor `core/` module (5,475 → 3,800 lines, -30%)
- Data structures: `ScpContainer`, `Assay`, `ScpMatrix`
- Core operations: matrix ops, JIT compilation
- I/O operations

**Key Tasks:**

1. Split core/ into focused submodules:
   ```
   core/
   ├── structures.py      # Data classes
   ├── container.py       # ScpContainer ops
   ├── matrix.py          # ScpMatrix ops
   ├── mask.py            # Mask operations
   └── provenance.py      # Provenance tracking
   ```

2. Simplify data structures:
   - Reduce inheritance depth
   - Use dataclasses for simple structures
   - Implement `__slots__` for memory efficiency

3. Optimize JIT operations:
   - Identify hot paths
   - Apply selective JIT compilation
   - Cache compiled functions

**Success Criteria:**
- All functions < 50 lines
- 100% type coverage
- No circular imports
- Provenance tracking simplified

### Member 2: Normalization Module Specialist

**Responsibilities:**
- Refactor `normalization/` module (658 → 450 lines, -32%)
- 8 normalization methods
- Consistent API across methods

**Key Tasks:**

1. Standardize normalization API:
   ```python
   def normalize(
       container: ScpContainer,
       method: NormalizeMethod,
       assay: str = "proteins",
       layer: str = "data",
       **kwargs
   ) -> ScpContainer:
       """Normalize data using specified method."""
   ```

2. Extract common patterns:
   - Preprocessing utilities
   - Validation functions
   - Post-processing helpers

3. Implement method base class:
   ```python
   @dataclass
   class NormalizeMethod:
       name: str
       validate: Callable
       apply: Callable
   ```

**Success Criteria:**
- All methods follow same API
- Shared code extracted
- Mathematical formulas documented
- Unit tests for edge cases

### Member 3: Imputation Module Specialist

**Responsibilities:**
- Refactor `impute/` module (2,381 → 1,600 lines, -33%)
- 4 imputation algorithms
- Memory efficiency optimization

**Key Tasks:**

1. Simplify imputation algorithms:
   - Remove redundant code paths
   - Use scikit-learn where possible
   - Implement sparse matrix support

2. Create unified imputation interface:
   ```python
   def impute(
       container: ScpContainer,
       method: ImputeMethod,
       assay: str = "proteins",
       **kwargs
   ) -> ScpContainer:
       """Impute missing values."""
   ```

3. Optimize memory usage:
   - Chunked processing for large datasets
   - Sparse matrix optimizations
   - Lazy evaluation where possible

**Success Criteria:**
- 30% code reduction
- Memory usage reduced by 20%
- Support for >1M cells
- Comprehensive benchmark suite

### Member 4: Integration Module Specialist

**Responsibilities:**
- Refactor `integration/` module (1,708 → 1,150 lines, -33%)
- 5 batch correction methods
- External library integration (Harmony, Scanorama)

**Key Tasks:**

1. Standardize integration API:
   ```python
   def integrate(
       container: ScpContainer,
       method: IntegrateMethod,
       batch_key: str = "batch",
       assay: str = "proteins",
       **kwargs
   ) -> ScpContainer:
       """Correct batch effects."""
   ```

2. Improve external library integration:
   - Wrapper classes for Harmony, Scanorama
   - Version compatibility handling
   - Graceful fallbacks

3. Add integration diagnostics:
   - Batch mixing metrics
   - Visualization helpers
   - Before/after comparisons

**Success Criteria:**
- Consistent API across methods
- External dependencies optional
- Diagnostic tools included
- Performance benchmarks

### Member 5: Quality Control Specialist

**Responsibilities:**
- Refactor `qc/` module (3,589 → 2,400 lines, -33%)
- Basic and advanced QC metrics
- Outlier detection
- QC visualization integration

**Key Tasks:**

1. Reorganize QC module:
   ```
   qc/
   ├── metrics.py         # Core QC metrics
   ├── outliers.py        # Outlier detection
   ├── filters.py         # Filtering operations
   └── diagnostics.py     # Diagnostic tools
   ```

2. Simplify QC pipeline:
   ```python
   def qc_pipeline(
       container: ScpContainer,
       filters: List[QcFilter],
       outlier_method: str = "isolation_forest"
   ) -> ScpContainer:
       """Apply complete QC pipeline."""
   ```

3. Integrate with visualization:
   - Move QC-specific plots to viz/recipes/qc.py
   - Keep metrics in qc/
   - Clean separation of concerns

**Success Criteria:**
- Clear module structure
- Reusable filter components
- Comprehensive outlier detection
- Integration with viz/

### Member 6: Dimensionality Reduction & Clustering Specialist

**Responsibilities:**
- Refactor `dim_reduction/` + `cluster/` (1,723 → 1,170 lines, -32%)
- PCA, UMAP implementations
- KMeans, graph-based clustering

**Key Tasks:**

1. Dimensionality reduction:
   ```python
   def reduce_dimensionality(
       container: ScpContainer,
       method: ReduceMethod,
       n_components: int = 50,
       assay: str = "proteins"
   ) -> ScpContainer:
       """Reduce dimensionality."""
   ```

2. Clustering:
   ```python
   def cluster(
       container: ScpContainer,
       method: ClusterMethod,
       n_clusters: int = 10,
       assay: str = "proteins"
   ) -> ScpContainer:
       """Cluster cells."""
   ```

3. Pipeline integration:
   - Seamless dim_red → clustering
   - Shared embedding space
   - Cluster evaluation metrics

**Success Criteria:**
- Unified API for both modules
- Chainable operations
- Support for large datasets
- Evaluation metrics included

### Member 7: Feature Selection & Differential Expression Specialist

**Responsibilities:**
- Refactor `feature_selection/` + `diff_expr/` (4,185 → 2,800 lines, -33%)
- HVG, VST, dropout rate methods
- Differential expression analysis

**Key Tasks:**

1. Feature selection:
   ```python
   def select_features(
       container: ScpContainer,
       method: SelectMethod,
       n_features: int = 2000,
       assay: str = "proteins"
   ) -> ScpContainer:
       """Select highly variable features."""
   ```

2. Differential expression:
   ```python
   def differential_expression(
       container: ScpContainer,
       groupby: str,
       method: str = "wilcoxon",
       assay: str = "proteins"
   ) -> pd.DataFrame:
       """Compute differential expression."""
   ```

3. Statistical validation:
   - Multiple testing correction
   - Effect size calculations
   - Confidence intervals

**Success Criteria:**
- Consistent statistical framework
- Multiple testing support
- Publication-ready outputs
- Comprehensive validation

### Member 8: Utilities, Visualization & I/O Specialist

**Responsibilities:**
- Refactor `utils/` + `viz/` + `io/` (12,992 → 8,350 lines, -36%)
- Largest refactoring task
- Focus on code deduplication

**Key Tasks:**

1. Utilities reorganization:
   ```
   utils/
   ├── stats.py           # Statistical utilities
   ├── transform.py       # Data transforms
   └── batch.py           # Batch processing
   ```

2. Visualization refactoring:
   - Extract common plotting patterns
   - Simplify base/recipes structure
   - Reduce code duplication

3. I/O simplification:
   ```python
   def read(
       path: str,
       format: str = "auto"
   ) -> ScpContainer:
       """Read data from file."""

   def write(
       container: ScpContainer,
       path: str,
       format: str = "auto"
   ) -> None:
       """Write data to file."""
   ```

**Success Criteria:**
- 36% code reduction (largest reduction)
- No code duplication > 2%
- Simple I/O interface
- Comprehensive viz recipes

---

## Refactoring Strategy

### Phase 1: Analysis (Days 1-2)

**Goal:** Understand current codebase and identify refactoring opportunities.

**Tasks:**

1. Code Metrics Collection
   ```bash
   # Cyclomatic complexity
   radon cc scptensor/ -a

   # Code duplication
   lizard scptensor/

   # Type coverage
   mypy --cobertura-xml-report scptensor/
   ```

2. Dependency Analysis
   - Map import dependencies
   - Identify circular imports
   - Find unused imports

3. Complexity Hotspots
   - Functions > 50 lines
   - Classes > 300 lines
   - Cyclomatic complexity > 10

**Deliverables:**
- Module analysis report
- Complexity heatmap
- Dependency graph
- Refactoring priority list

### Phase 2: Design (Day 3)

**Goal:** Create detailed refactoring plan for each module.

**Tasks:**

1. API Design
   - Define consistent interfaces
   - Specify type signatures
   - Document parameter contracts

2. Architecture Design
   - Module structure
   - Import hierarchy
   - Data flow diagrams

3. Test Strategy
   - Test cases to preserve
   - New tests to add
   - Coverage targets

**Deliverables:**
- API specification
- Architecture diagram
- Test plan

### Phase 3: Implementation (Days 4-8)

**Goal:** Execute refactoring while maintaining functionality.

**Tasks:**

1. Create Refactoring Branch
   ```bash
   git checkout -b refactor/2026-02-core-refactoring
   ```

2. Module-by-Module Refactoring
   - Start with leaf modules (no dependencies)
   - Move to core modules
   - End with integration points

3. Continuous Testing
   - Run tests after each module
   - Maintain >85% coverage
   - Fix regressions immediately

**Best Practices:**

```python
# Refactoring checklist for each function
□ Function < 50 lines
□ Single responsibility
□ Type hints complete
□ Docstring (NumPy style)
□ Unit tests written
□ Edge cases handled
□ Performance checked
```

### Phase 4: Validation (Days 9-10)

**Goal:** Ensure refactored code is correct and performant.

**Tasks:**

1. Functional Validation
   - All tests pass
   - Benchmark comparison
   - Real data validation

2. Performance Validation
   - Speed benchmarks
   - Memory profiling
   - Scalability tests

3. Mathematical Validation
   - Formula verification
   - Edge case testing
   - Comparison with reference implementations

**Deliverables:**
- Validation report
- Performance comparison
- Mathematical verification

---

## Module-by-Module Specifications

### Core Module Refactoring

#### Current Issues

1. **Mixed Responsibilities**
   - Data structures mixed with I/O
   - JIT operations intertwined with core logic
   - Provenance tracking scattered

2. **Complex Functions**
   - Several functions > 100 lines
   - High cyclomatic complexity
   - Nested conditionals

3. **Type System**
   - Incomplete type hints
   - Generic types underutilized
   - Optional types unclear

#### Proposed Structure

```python
# structures.py - Data classes with clear hierarchy
from dataclasses import dataclass
from typing import Dict, Optional, Union
import numpy as np
import scipy.sparse as sp

@dataclass
class ScpMatrix:
    """Single-cell proteomics data matrix."""
    X: Union[np.ndarray, sp.spmatrix]
    M: Optional[Union[np.ndarray, sp.spmatrix]] = None
    metadata: Optional[Dict] = None

    def __post_init__(self):
        """Validate matrix dimensions."""
        if self.M is not None and self.M.shape != self.X.shape:
            raise ValueError("X and M must have same shape")

# container.py - Container operations
class ScpContainer:
    """Container for single-cell proteomics data."""

    def __init__(
        self,
        obs: pl.DataFrame,
        assays: Dict[str, Assay],
        metadata: Optional[Dict] = None
    ):
        self.obs = obs
        self.assays = assays
        self.metadata = metadata or {}

    @property
    def n_samples(self) -> int:
        """Number of samples."""
        return len(self.obs)

    def copy(self) -> 'ScpContainer':
        """Create deep copy."""
        return ScpContainer(
            obs=self.obs.clone(),
            assays={k: v.copy() for k, v in self.assays.items()},
            metadata=self.metadata.copy()
        )

# mask.py - Mask operations
class MaskCode(IntEnum):
    """Provenance mask codes."""
    VALID = 0
    MBR = 1      # Missing between runs
    LOD = 2      # Below detection limit
    FILTERED = 3 # QC filtered
    IMPUTED = 5  # Imputed value

def update_mask(
    M: np.ndarray,
    indices: Tuple[np.ndarray, np.ndarray],
    code: MaskCode
) -> np.ndarray:
    """Update mask at specified indices."""
    M_new = M.copy()
    M_new[indices] = code
    return M_new
```

#### Key Improvements

1. **Dataclasses for Simple Structures**
   - Automatic `__init__`, `__repr__`
   - Type validation
   - Immutability where appropriate

2. **Property-Based Accessors**
   - Computed attributes
   - Clear API
   - Lazy evaluation

3. **Type Safety**
   - Complete type hints
   - Generic types for flexibility
   - Runtime validation

#### API Simplification Example

```python
# Before (current API)
container = ScpContainer(obs_df, assays_dict, metadata_dict)
assay = container.assays['proteins']
layer = assay.layers['data']
new_layer = ScpMatrix(data, mask)
assay.add_layer('normalized', new_layer)
container.update_assay('proteins', assay)

# After (refactored API)
container = (
    ScpContainer(obs_df, assays_dict)
    .with_layer('proteins', 'normalized', data, mask)
)
```

### Normalization Module Refactoring

#### Current Issues

1. **Inconsistent APIs**
   - Different parameter names across methods
   - Inconsistent return types
   - Mixed validation approaches

2. **Code Duplication**
   - Repeated validation logic
   - Duplicated preprocessing steps
   - Common postprocessing patterns

#### Proposed API

```python
# normalization/base.py
from abc import ABC, abstractmethod
from typing import Protocol

class NormalizeMethod(Protocol):
    """Protocol for normalization methods."""
    name: str
    validate: Callable[[np.ndarray], bool]
    apply: Callable[[np.ndarray], np.ndarray]

# normalization/normalize.py
@dataclass
class Normalizer:
    """Unified normalization interface."""
    method: NormalizeMethod
    params: Dict[str, Any]

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Apply normalization."""
        if not self.method.validate(data):
            raise ValueError(f"Invalid data for {self.method.name}")
        return self.method.apply(data, **self.params)

# normalization/methods/log.py
def log_normalize(
    data: np.ndarray,
    base: float = 2.0,
    offset: float = 1.0
) -> np.ndarray:
    """
    Apply logarithmic transformation.

    Parameters
    ----------
    data : np.ndarray
        Input data
    base : float, default=2.0
        Logarithm base
    offset : float, default=1.0
        Offset to avoid log(0)

    Returns
    -------
    np.ndarray
        Log-transformed data

    Notes
    -----
    The log transformation is defined as:

    .. math::

        X_{log} = \\frac{\\log(X + c)}{\\log(b)}

    where :math:`X` is the input data, :math:`b` is the base,
    and :math:`c` is the offset.
    """
    return np.log(data + offset) / np.log(base)

# Main API function
def normalize(
    container: ScpContainer,
    method: Union[str, NormalizeMethod],
    assay: str = "proteins",
    layer: str = "data",
    output_layer: Optional[str] = None,
    **kwargs
) -> ScpContainer:
    """
    Normalize data using specified method.

    Parameters
    ----------
    container : ScpContainer
        Input container
    method : Union[str, NormalizeMethod]
        Normalization method or name
    assay : str, default="proteins"
        Assay to normalize
    layer : str, default="data"
        Layer to normalize
    output_layer : Optional[str], default=None
        Output layer name (default: {layer}_{method})
    **kwargs
        Method-specific parameters

    Returns
    -------
    ScpContainer
        Container with normalized data

    Examples
    --------
    >>> container = normalize(container, "log", base=2.0)
    >>> container = normalize(container, "quantile", n_quantiles=100)
    """
    if isinstance(method, str):
        method = get_method(method)

    normalizer = Normalizer(method, kwargs)
    data = container.assays[assay].layers[layer].X
    normalized = normalizer(data)

    output_layer = output_layer or f"{layer}_{method.name}"
    return container.with_layer(assay, output_layer, normalized)
```

#### Key Improvements

1. **Unified API**
   - Single `normalize()` function
   - Method registry
   - Consistent parameters

2. **Type Safety**
   - Protocol-based methods
   - Complete type hints
   - Runtime validation

3. **Mathematical Documentation**
   - LaTeX formulas
   - Algorithm descriptions
   - Reference implementations

### Imputation Module Refactoring

#### Current Issues

1. **High Complexity**
   - 2,381 lines for 4 methods
   - Complex algorithm implementations
   - Mixed optimization strategies

2. **Memory Inefficiency**
   - Full matrix operations
   - No chunking for large datasets
   - Sparse matrix support incomplete

#### Proposed API

```python
# imputation/base.py
class ImputeMethod(Protocol):
    """Protocol for imputation methods."""
    name: str
    supports_sparse: bool
    validate: Callable[[np.ndarray], bool]
    apply: Callable[[np.ndarray], np.ndarray]

# imputation/methods/knn.py
def knn_impute(
    data: np.ndarray,
    n_neighbors: int = 5,
    weights: str = "uniform",
    metric: str = "euclidean"
) -> np.ndarray:
    """
    K-nearest neighbors imputation.

    Parameters
    ----------
    data : np.ndarray
        Input data with missing values
    n_neighbors : int, default=5
        Number of neighbors to use
    weights : str, default="uniform"
        Weight function ("uniform" or "distance")
    metric : str, default="euclidean"
        Distance metric

    Returns
    -------
    np.ndarray
        Data with imputed values

    Notes
    -----
    KNN imputation finds the k nearest neighbors for each sample
    with missing values and imputes using weighted average:

    .. math::

        X_{ij} = \\sum_{k \\in N_i} w_k X_{kj}

    where :math:`N_i` are the nearest neighbors and :math:`w_k`
    are the weights.
    """
    from sklearn.impute import KNNImputer

    imputer = KNNImputer(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric
    )
    return imputer.fit_transform(data)

# Main API function
def impute(
    container: ScpContainer,
    method: Union[str, ImputeMethod],
    assay: str = "proteins",
    layer: str = "data",
    **kwargs
) -> ScpContainer:
    """
    Impute missing values.

    Parameters
    ----------
    container : ScpContainer
        Input container
    method : Union[str, ImputeMethod]
        Imputation method
    assay : str, default="proteins"
        Assay to impute
    layer : str, default="data"
        Layer to impute
    **kwargs
        Method-specific parameters

    Returns
    -------
    ScpContainer
        Container with imputed data

    Examples
    --------
    >>> container = impute(container, "knn", n_neighbors=5)
    >>> container = impute(container, "missforest", max_iter=10)
    """
    if isinstance(method, str):
        method = get_impute_method(method)

    data = container.assays[assay].layers[layer].X
    mask = container.assays[assay].layers[layer].M

    imputed = method.apply(data, **kwargs)

    # Update mask to mark imputed values
    new_mask = update_mask(mask, np.isnan(data), MaskCode.IMPUTED)

    output_layer = f"{layer}_imputed"
    return container.with_layer(assay, output_layer, imputed, new_mask)
```

#### Key Improvements

1. **Leverage External Libraries**
   - Use scikit-learn KNN imputer
   - Use fancyimpute MissForest
   - Reduce custom implementations

2. **Memory Efficiency**
   - Chunked processing
   - Sparse matrix support
   - Lazy evaluation

3. **Mask Updates**
   - Automatic provenance tracking
   - Clear imputation markers
   - Audit trail

### Integration Module Refactoring

#### Current Issues

1. **Inconsistent APIs**
   - Different parameter conventions
   - Incompatible return types
   - Mixed external library integrations

2. **Limited Diagnostics**
   - No batch mixing metrics
   - Missing quality assessment
   - Limited visualization support

#### Proposed API

```python
# integration/base.py
class IntegrateMethod(Protocol):
    """Protocol for integration methods."""
    name: str
    requires_batch: bool
    validate: Callable[[np.ndarray, np.ndarray], bool]
    apply: Callable[[np.ndarray, np.ndarray], np.ndarray]
    diagnostics: Callable[[np.ndarray, np.ndarray], Dict]

# integration/methods/combat.py
def combat_correct(
    data: np.ndarray,
    batch: np.ndarray,
    covariates: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    ComBat batch correction.

    Parameters
    ----------
    data : np.ndarray
        Input data (n_samples × n_features)
    batch : np.ndarray
        Batch labels (n_samples,)
    covariates : Optional[np.ndarray], default=None
        Covariate matrix (n_samples × n_covariates)

    Returns
    -------
    np.ndarray
        Batch-corrected data

    Notes
    -----
    ComBat uses empirical Bayes to correct for batch effects:

    .. math::

        Y_{ij}^{*} = \\frac{Y_{ij} - \\hat{\\alpha}_{\\gamma(i)} - \\hat{\\beta}_{\\gamma(i)} X_{ij}}{\\hat{\\delta}_{\\gamma(i)}}

    where :math:`\\gamma(i)` is the batch for sample i,
    and :math:`\\hat{\\alpha}, \\hat{\\beta}, \\hat{\\delta}` are
    estimated using empirical Bayes.
    """
    from pycombat import ComBat

    model = ComBat(data, batch)
    corrected = model.fit_transform()

    return corrected

# Main API function
def integrate(
    container: ScpContainer,
    method: Union[str, IntegrateMethod],
    batch_key: str = "batch",
    assay: str = "proteins",
    layer: str = "data",
    **kwargs
) -> ScpContainer:
    """
    Correct batch effects.

    Parameters
    ----------
    container : ScpContainer
        Input container
    method : Union[str, IntegrateMethod]
        Integration method
    batch_key : str, default="batch"
        Key for batch labels in obs
    assay : str, default="proteins"
        Assay to integrate
    layer : str, default="data"
        Layer to integrate
    **kwargs
        Method-specific parameters

    Returns
    -------
    ScpContainer
        Container with integrated data

    Examples
    --------
    >>> container = integrate(container, "combat", batch_key="batch")
    >>> container = integrate(container, "harmony", batch_key="sample")
    """
    if isinstance(method, str):
        method = get_integrate_method(method)

    data = container.assays[assay].layers[layer].X
    batch = container.obs[batch_key].to_numpy()

    corrected = method.apply(data, batch, **kwargs)

    output_layer = f"{layer}_integrated"
    return container.with_layer(assay, output_layer, corrected)
```

#### Key Improvements

1. **Consistent Interface**
   - Single `integrate()` function
   - Method registry
   - Standardized parameters

2. **Diagnostic Support**
   - Batch mixing metrics
   - Quality assessment
   - Before/after visualization

3. **External Library Integration**
   - Graceful handling of optional dependencies
   - Version compatibility
   - Fallback strategies

---

## API Simplification

### Design Principles

1. **Fluent Interface**
   - Chainable methods
   - Method chaining
   - Pipeline support

2. **Smart Defaults**
   - Sensible defaults for common cases
   - Auto-detection where possible
   - Minimal required parameters

3. **Type Safety**
   - Complete type hints
   - Generic types
   - Runtime validation

### API Examples

#### Before vs After

**Example 1: Normalization Pipeline**

```python
# Before (current API - verbose)
from scptensor.normalization import log_normalize, quantile_normalize
from scptensor.qc import filter_cells

container = filter_cells(container, min_genes=200)
container = log_normalize(container, assay='proteins', layer='data', base=2.0)
container = quantile_normalize(container, assay='proteins', layer='log', n_quantiles=100)

# After (refactored API - concise)
container = (
    container
    .filter_cells(min_genes=200)
    .normalize("log", base=2.0)
    .normalize("quantile", n_quantiles=100)
)
```

**Example 2: Complete Analysis Pipeline**

```python
# Before (current API)
from scptensor.qc import qc_pipeline
from scptensor.normalization import log_normalize
from scptensor.impute import knn_impute
from scptensor.integration import combat_correct
from scptensor.dim_reduction import pca, umap
from scptensor.cluster import kmeans_clustering

# QC
container = qc_pipeline(container, min_genes=200, min_cells=3, mt_threshold=0.2)

# Normalization
container = log_normalize(container, assay='proteins', layer='data', base=2.0)

# Imputation
container = knn_impute(container, assay='proteins', layer='log', n_neighbors=5)

# Integration
container = combat_correct(container, assay='proteins', layer='log_imputed', batch_key='batch')

# Dimensionality reduction
container = pca(container, assay='proteins', layer='log_imputed_integrated', n_components=50)
container = umap(container, assay='proteins', layer='pca', n_neighbors=15)

# Clustering
container = kmeans_clustering(container, assay='proteins', layer='umap', n_clusters=10)

# After (refactored API)
container = (
    container
    .qc_pipeline(min_genes=200, min_cells=3, mt_threshold=0.2)
    .normalize("log", base=2.0)
    .impute("knn", n_neighbors=5)
    .integrate("combat", batch_key="batch")
    .pca(n_components=50)
    .umap(n_neighbors=15)
    .cluster("kmeans", n_clusters=10)
)
```

**Example 3: Type Inference**

```python
# Before (current API - manual type specification)
def process_data(container: ScpContainer, assay_name: str) -> ScpContainer:
    assay = container.assays[assay_name]
    layer = assay.layers['data']
    result = some_function(layer.X)
    new_layer = ScpMatrix(result, layer.M)
    assay.add_layer('result', new_layer)
    container.update_assay(assay_name, assay)
    return container

# After (refactored API - automatic type inference)
def process_data(container: ScpContainer) -> ScpContainer:
    """Process data with automatic type handling."""
    return container.apply(lambda x: some_function(x))
```

### Fluent Interface Implementation

```python
# core/container.py
class ScpContainer:
    """Enhanced container with fluent interface."""

    def normalize(
        self,
        method: Union[str, NormalizeMethod],
        assay: str = "proteins",
        **kwargs
    ) -> 'ScpContainer':
        """Normalize and return new container."""
        return normalize(self, method, assay, **kwargs)

    def impute(
        self,
        method: Union[str, ImputeMethod],
        assay: str = "proteins",
        **kwargs
    ) -> 'ScpContainer':
        """Impute and return new container."""
        return impute(self, method, assay, **kwargs)

    def integrate(
        self,
        method: Union[str, IntegrateMethod],
        batch_key: str = "batch",
        assay: str = "proteins",
        **kwargs
    ) -> 'ScpContainer':
        """Integrate and return new container."""
        return integrate(self, method, batch_key, assay, **kwargs)

    def pca(
        self,
        n_components: int = 50,
        assay: str = "proteins"
    ) -> 'ScpContainer':
        """Run PCA and return new container."""
        return pca(self, n_components, assay)

    def umap(
        self,
        n_neighbors: int = 15,
        assay: str = "proteins"
    ) -> 'ScpContainer':
        """Run UMAP and return new container."""
        return umap(self, n_neighbors, assay)

    def cluster(
        self,
        method: str = "kmeans",
        n_clusters: int = 10,
        assay: str = "proteins"
    ) -> 'ScpContainer':
        """Cluster and return new container."""
        return cluster(self, method, n_clusters, assay)
```

---

## Mathematical Verification

### Formula Documentation

Every algorithm must include mathematical formulas in LaTeX format.

#### Example: Log Normalization

```python
def log_normalize(
    data: np.ndarray,
    base: float = 2.0,
    offset: float = 1.0
) -> np.ndarray:
    """
    Apply logarithmic transformation.

    Parameters
    ----------
    data : np.ndarray
        Input data matrix :math:`X \\in \\mathbb{R}^{n \\times m}`
    base : float, default=2.0
        Logarithm base :math:`b`
    offset : float, default=1.0
        Offset :math:`c` to avoid :math:`\\log(0)`

    Returns
    -------
    np.ndarray
        Log-transformed data :math:`X_{log}`

    Notes
    -----
    The log transformation is defined as:

    .. math::

        X_{log,ij} = \\frac{\\log(X_{ij} + c)}{\\log(b)}

    For :math:`b = 2` (default), this computes :math:`\\log_2(X + c)`.

    The offset :math:`c` ensures the transformation is defined for
    zero and negative values. Common choices:

    - :math:`c = 1`: Standard pseudo-count
    - :math:`c = \\max(0, 1 - \\min(X))`: Ensures positivity

    Examples
    --------
    >>> data = np.array([[1, 2, 0], [4, 0, 6]])
    >>> log_normalize(data, base=2.0, offset=1.0)
    array([[1.0, 1.58, 0.0],
           [2.32, 0.0, 2.58]])
    """
    return np.log(data + offset) / np.log(base)
```

### Algorithm Validation

#### Test Coverage Requirements

1. **Basic Functionality**
   - Normal case
   - Edge cases (empty, single value)
   - Type handling

2. **Mathematical Correctness**
   - Compare with reference implementation
   - Verify formula implementation
   - Check numerical stability

3. **Edge Cases**
   - Zero values
   - Missing values
   - Extreme values
   - Singular matrices

#### Validation Template

```python
# tests/test_normalization/test_log_normalize.py
import pytest
import numpy as np
from scptensor.normalization import log_normalize

class TestLogNormalize:
    """Test log normalization."""

    def test_basic_functionality(self):
        """Test basic log normalization."""
        data = np.array([[1.0, 2.0, 4.0], [8.0, 16.0, 32.0]])
        result = log_normalize(data, base=2.0, offset=0.0)

        expected = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_with_offset(self):
        """Test log normalization with offset."""
        data = np.array([[0.0, 1.0, 2.0]])
        result = log_normalize(data, base=2.0, offset=1.0)

        expected = np.array([[0.0, 1.0, 1.58]])
        np.testing.assert_array_almost_equal(result, expected, decimal=2)

    def test_formula_implementation(self):
        """Verify formula: log(X + c) / log(b)."""
        data = np.array([[1.0, 2.0, 4.0]])
        base = 2.0
        offset = 1.0

        result = log_normalize(data, base, offset)
        expected = np.log(data + offset) / np.log(base)

        np.testing.assert_array_almost_equal(result, expected)

    def test_edge_cases(self):
        """Test edge cases."""
        # Empty array
        with pytest.raises(ValueError):
            log_normalize(np.array([]))

        # All zeros
        result = log_normalize(np.zeros((3, 3)), offset=1.0)
        assert np.all(result == 0.0)

        # Large values
        large = np.array([[1e10, 1e20]])
        result = log_normalize(large)
        assert np.all(np.isfinite(result))

    def test_comparison_with_sklearn(self):
        """Compare with scikit-learn FunctionTransformer."""
        from sklearn.preprocessing import FunctionTransformer

        data = np.random.rand(100, 50) * 100

        # Our implementation
        result_ours = log_normalize(data, base=2.0, offset=1.0)

        # Scikit-learn
        transformer = FunctionTransformer(
            lambda x: np.log(x + 1.0) / np.log(2.0)
        )
        result_sklearn = transformer.fit_transform(data)

        np.testing.assert_array_almost_equal(result_ours, result_sklearn)

    def test_sparse_matrix_support(self):
        """Test sparse matrix support."""
        from scipy import sparse

        dense = np.random.rand(100, 50)
        dense[dense < 0.8] = 0  # Make sparse
        sparse_mat = sparse.csr_matrix(dense)

        result_dense = log_normalize(dense)
        result_sparse = log_normalize(sparse_mat)

        np.testing.assert_array_almost_equal(
            result_dense, result_sparse.toarray()
        )
```

### Reference Implementation Comparison

```python
# tests/benchmark/test_algorithm_correctness.py
import pytest
import numpy as np
from sklearn.preprocessing import StandardScaler
from scanpy.pp import normalize_total, log1p

from scptensor.normalization import (
    normalize_total as scpn_normalize_total,
    log_normalize,
    scale_normalize
)

class TestAlgorithmCorrectness:
    """Compare with reference implementations."""

    def test_normalize_total_vs_scanpy(self):
        """Compare normalize_total with Scanpy."""
        data = np.random.rand(100, 50) * 100

        # ScpTensor
        result_scptensor = scpn_normalize_total(data, target_sum=1e4)

        # Scanpy (requires AnnData conversion)
        import scanpy as sc
        adata = sc.AnnData(data)
        sc.pp.normalize_total(adata, target_sum=1e4)
        result_scanpy = adata.X

        np.testing.assert_array_almost_equal(
            result_scptensor, result_scanpy, decimal=5
        )

    def test_scale_vs_sklearn(self):
        """Compare scale with sklearn StandardScaler."""
        data = np.random.rand(100, 50) * 100

        # ScpTensor
        result_scptensor = scale_normalize(data)

        # Sklearn
        scaler = StandardScaler()
        result_sklearn = scaler.fit_transform(data)

        np.testing.assert_array_almost_equal(
            result_scptensor, result_sklearn, decimal=5
        )

    def test_log_vs_numpy(self):
        """Compare log with numpy."""
        data = np.random.rand(100, 50) * 100

        # ScpTensor
        result_scptensor = log_normalize(data, base=2.0, offset=1.0)

        # NumPy
        result_numpy = np.log2(data + 1.0)

        np.testing.assert_array_almost_equal(
            result_scptensor, result_numpy, decimal=10
        )
```

---

## Implementation Workflow

### Pre-Refactoring Checklist

```markdown
## Module: [module_name]

### Analysis
- [ ] Current LOC: [count]
- [ ] Target LOC: [target]
- [ ] Complexity hotspots identified
- [ ] Dependencies mapped
- [ ] Test coverage: [current]%

### Design
- [ ] API specification written
- [ ] Type signatures defined
- [ ] Test cases specified
- [ ] Documentation outline

### Resources
- [ ] Team member assigned
- [ ] Time allocated: [days]
- [ ] Dependencies resolved
```

### Refactoring Process

#### Step 1: Create Branch

```bash
# Format: refactor/YYYY-MM-DD-module-name
git checkout -b refactor/2026-02-25-core-refactoring
git checkout -b refactor/2026-02-25-normalization
```

#### Step 2: Write Tests First

```python
# tests/test_normalization_refactored.py
import pytest
import numpy as np
from scptensor.normalization import normalize

def test_normalize_api():
    """Test new unified normalize API."""
    container = create_test_container()

    # Test method name
    result = normalize(container, "log", base=2.0)
    assert result is not None
    assert result != container  # Immutable

    # Test method object
    from scptensor.normalization import LogNormalize
    result = normalize(container, LogNormalize(), base=2.0)
    assert result is not None

def test_normalize_chaining():
    """Test that normalize supports chaining."""
    container = create_test_container()

    result = (
        container
        .normalize("log")
        .normalize("quantile")
    )

    assert "data_log_quantile" in result.assays["proteins"].layers
```

#### Step 3: Refactor Code

```python
# normalization/normalize.py
from typing import Union, Dict, Any
import numpy as np

# Method registry
_METHODS: Dict[str, NormalizeMethod] = {}

def register_method(method: NormalizeMethod) -> None:
    """Register normalization method."""
    _METHODS[method.name] = method

def get_method(name: str) -> NormalizeMethod:
    """Get method by name."""
    if name not in _METHODS:
        raise ValueError(f"Unknown method: {name}")
    return _METHODS[name]

# Register built-in methods
register_method(LogNormalize())
register_method(QuantileNormalize())
register_method(ScaleNormalize())

def normalize(
    container: ScpContainer,
    method: Union[str, NormalizeMethod],
    assay: str = "proteins",
    layer: str = "data",
    **kwargs
) -> ScpContainer:
    """Normalize data using specified method."""
    # Implementation
    pass
```

#### Step 4: Update Documentation

```python
def normalize(...) -> ScpContainer:
    """
    Normalize data using specified method.

    Parameters
    ----------
    container : ScpContainer
        Input container
    method : Union[str, NormalizeMethod]
        Normalization method (name or object)
    assay : str, default="proteins"
        Assay to normalize
    layer : str, default="data"
        Layer to normalize
    **kwargs
        Method-specific parameters

    Returns
    -------
    ScpContainer
        Container with normalized data

    Raises
    ------
    ValueError
        If method is unknown or validation fails

    Examples
    --------
    >>> container = normalize(container, "log", base=2.0)
    >>> container = normalize(container, "quantile", n_quantiles=100)

    See Also
    --------
    scptensor.normalization.methods : Available methods
    """
    pass
```

#### Step 5: Run Tests

```bash
# Run specific module tests
pytest tests/test_normalization/ -v

# Run with coverage
pytest tests/test_normalization/ --cov=scptensor.normalization

# Check type hints
mypy scptensor/normalization/

# Lint
ruff check scptensor/normalization/
ruff format scptensor/normalization/
```

#### Step 6: Validate Performance

```python
# tests/benchmark/test_normalization_performance.py
import pytest
import numpy as np
from time import time

def test_normalization_performance():
    """Test normalization performance."""
    container = create_large_test_container(n_samples=100000, n_features=100)

    start = time()
    result = normalize(container, "log")
    elapsed = time() - start

    # Should complete in reasonable time
    assert elapsed < 10.0, f"Too slow: {elapsed:.2f}s"

    # Check memory usage
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    assert memory_mb < 1000, f"Too much memory: {memory_mb:.0f}MB"
```

#### Step 7: Code Review

```markdown
## Pull Request Checklist

### Code Quality
- [ ] Functions < 50 lines
- [ ] Single responsibility
- [ ] No code duplication
- [ ] Complete type hints

### Testing
- [ ] All tests pass
- [ ] Coverage > 85%
- [ ] Edge cases covered
- [ ] Performance benchmarks

### Documentation
- [ ] NumPy-style docstrings
- [ ] Mathematical formulas
- [ ] Examples included
- [ ] Type hints complete

### Validation
- [ ] Compared with reference implementation
- [ ] Edge cases tested
- [ ] Performance validated
- [ ] Memory usage checked
```

#### Step 8: Merge

```bash
# Update main branch
git checkout main
git merge refactor/2026-02-25-normalization
git push origin main
```

---

## Quality Assurance

### Testing Strategy

#### Test Categories

1. **Unit Tests**
   - Function-level testing
   - Edge cases
   - Error handling
   - Target: 90%+ coverage

2. **Integration Tests**
   - Module interactions
   - Pipeline workflows
   - Data flow validation
   - Target: All major paths

3. **Performance Tests**
   - Speed benchmarks
   - Memory profiling
   - Scalability tests
   - Target: No regressions

4. **Mathematical Tests**
   - Formula verification
   - Reference implementation comparison
   - Numerical stability
   - Target: 100% algorithm coverage

#### Test Structure

```
tests/
├── unit/                      # Unit tests
│   ├── test_core/
│   ├── test_normalization/
│   ├── test_impute/
│   └── ...
├── integration/               # Integration tests
│   ├── test_pipelines.py
│   ├── test_workflows.py
│   └── test_data_flow.py
├── benchmark/                 # Performance tests
│   ├── test_speed.py
│   ├── test_memory.py
│   └── test_scalability.py
├── mathematical/              # Algorithm validation
│   ├── test_formulas.py
│   ├── test_reference_comparison.py
│   └── test_numerical_stability.py
└── real_data/                 # Real-world validation
    ├── test_datasets.py
    └── test_publications.py
```

### Code Review Checklist

```markdown
## Code Review Template

### Functionality
- [ ] Meets requirements
- [ ] Handles edge cases
- [ ] Error handling complete
- [ ] No regressions

### Code Quality
- [ ] Follows style guide
- [ ] Type hints complete
- [ ] Docstrings present
- [ ] No code duplication

### Performance
- [ ] No performance regressions
- [ ] Memory efficient
- [ ] Scalable to large data
- [ ] Benchmarks included

### Testing
- [ ] Tests comprehensive
- [ ] Coverage sufficient
- [ ] Edge cases covered
- [ ] Integration tested

### Documentation
- [ ] API documented
- [ ] Examples included
- [ ] Mathematical formulas
- [ ] Migration guide (if breaking)
```

### Continuous Integration

```yaml
# .github/workflows/refactor-ci.yml
name: Refactor CI

on:
  push:
    branches: [main, refactor/*]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.12', '3.13']

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        pip install -e ".[dev]"

    - name: Run tests
      run: |
        pytest --cov=scptensor --cov-report=xml

    - name: Check coverage
      run: |
        coverage report --fail-under=85

    - name: Type check
      run: |
        mypy scptensor/

    - name: Lint
      run: |
        ruff check scptensor/
        ruff format --check scptensor/

  benchmark:
    runs-on: ubuntu-latest
    needs: test

    steps:
    - uses: actions/checkout@v3

    - name: Run benchmarks
      run: |
        pytest tests/benchmark/ --benchmark-json=output.json

    - name: Compare performance
      run: |
        python scripts/compare_benchmark.py output.json
```

---

## Timeline and Milestones

### Overall Schedule

**Total Duration:** 6 weeks (42 days)
**Team Size:** 8 members
**Parallel Work:** 2-3 modules simultaneously

### Week-by-Week Breakdown

#### Week 1-2: Foundation (Batch 1)

**Modules:** `core/`, `utils/`

**Timeline:**
- Days 1-2: Analysis
- Day 3: Design
- Days 4-8: Implementation
- Days 9-10: Validation

**Deliverables:**
- Refactored core data structures
- Simplified utilities
- API foundations
- Test infrastructure

**Success Criteria:**
- All tests passing
- 30% code reduction
- Performance maintained
- Documentation complete

#### Week 2-3: Core Analysis (Batch 2)

**Modules:** `normalization/`, `qc/`

**Timeline:**
- Days 11-12: Analysis
- Day 13: Design
- Days 14-18: Implementation
- Days 19-20: Validation

**Deliverables:**
- Unified normalization API
- Simplified QC pipeline
- Comprehensive tests
- Performance benchmarks

**Success Criteria:**
- Consistent API
- Mathematical validation
- Integration tests passing
- Documentation complete

#### Week 3-4: Advanced Methods (Batch 3)

**Modules:** `impute/`, `integration/`

**Timeline:**
- Days 21-22: Analysis
- Day 23: Design
- Days 24-28: Implementation
- Days 29-30: Validation

**Deliverables:**
- Simplified imputation API
- Unified integration interface
- External library integration
- Diagnostic tools

**Success Criteria:**
- 33% code reduction
- Memory efficiency improved
- All algorithms validated
- Integration tests passing

#### Week 4-5: Analysis Tools (Batch 4)

**Modules:** `dim_reduction/`, `cluster/`, `feature_selection/`, `diff_expr/`

**Timeline:**
- Days 31-32: Analysis
- Day 33: Design
- Days 34-38: Implementation
- Days 39-40: Validation

**Deliverables:**
- Chainable analysis operations
- Unified clustering API
- Feature selection improvements
- Differential expression enhancements

**Success Criteria:**
- Fluent interface working
- All methods validated
- Performance benchmarks
- Real data validation

#### Week 5-6: Visualization & I/O (Batch 5)

**Modules:** `viz/`, `io/`, `standardization/`

**Timeline:**
- Days 41-42: Analysis
- Day 43: Design
- Days 44-48: Implementation
- Days 49-50: Validation

**Deliverables:**
- Simplified visualization API
- Reduced code duplication
- Unified I/O interface
- Complete documentation

**Success Criteria:**
- 36% code reduction (largest)
- No duplication > 2%
- All visualizations working
- Complete test coverage

#### Week 6: Integration & Testing

**Activities:**
- End-to-end testing
- Performance validation
- Documentation finalization
- Release preparation

**Deliverables:**
- All tests passing
- Performance report
- Migration guide
- Release notes

### Milestones

| Milestone | Date | Deliverable |
|-----------|------|-------------|
| M1: Foundation Complete | Day 10 | Core + Utils refactored |
| M2: Analysis Complete | Day 20 | Normalization + QC refactored |
| M3: Methods Complete | Day 30 | Impute + Integration refactored |
| M4: Tools Complete | Day 40 | Dim reduction + Clustering + Features refactored |
| M5: Visualization Complete | Day 50 | Viz + I/O refactored |
| M6: Release Ready | Day 56 | v0.2.0 release |

### Risk Management

#### Potential Risks

1. **Timeline Overrun**
   - Mitigation: Buffer time in each phase
   - Fallback: Reduce scope of non-critical modules

2. **Performance Regression**
   - Mitigation: Continuous benchmarking
   - Fallback: Performance optimization sprint

3. **Breaking Changes**
   - Mitigation: Migration guide
   - Fallback: Backward compatibility layer

4. **Team Coordination**
   - Mitigation: Daily standups
   - Fallback: Reduce parallel work

#### Contingency Plans

```markdown
## Contingency: Timeline Extension

If refactoring takes longer than expected:

1. **Priority 1 (Must Have)**
   - core/
   - normalization/
   - qc/

2. **Priority 2 (Should Have)**
   - impute/
   - integration/
   - dim_reduction/
   - cluster/

3. **Priority 3 (Nice to Have)**
   - feature_selection/
   - diff_expr/
   - viz/
   - utils/
   - io/

## Contingency: Performance Issues

If performance regresses:

1. Immediate rollback of problematic changes
2. Performance profiling sprint (3 days)
3. Targeted optimization
4. Re-benchmark before merging
```

---

## Appendix

### A. Code Examples

#### Complete Analysis Pipeline

```python
from scptensor import ScpContainer, read_h5ad
from scptensor.viz import qc_plot, umap_plot

# Load data
container = read_h5ad("data.h5ad")

# Complete analysis pipeline
result = (
    container
    # Quality control
    .qc_pipeline(
        min_genes=200,
        min_cells=3,
        mt_threshold=0.2
    )
    # Normalization
    .normalize("total", target_sum=1e4)
    .normalize("log", base=2.0)
    .normalize("scale")
    # Imputation
    .impute("knn", n_neighbors=5)
    # Integration
    .integrate("combat", batch_key="batch")
    # Dimensionality reduction
    .pca(n_components=50)
    .umap(n_neighbors=15)
    # Clustering
    .cluster("kmeans", n_clusters=10)
    # Feature selection
    .select_features("vst", n_features=2000)
)

# Visualization
qc_plot(result, metric="n_genes")
umap_plot(result, color="cluster")
```

#### Custom Method Registration

```python
from scptensor.normalization import register_method, NormalizeMethod
import numpy as np

# Define custom method
class CustomNormalize:
    """Custom normalization method."""

    name = "custom"

    def validate(self, data: np.ndarray) -> bool:
        """Validate input."""
        return data.size > 0 and np.isfinite(data).all()

    def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Apply normalization."""
        factor = kwargs.get("factor", 1.0)
        return data * factor

# Register method
register_method(CustomNormalize())

# Use method
container = container.normalize("custom", factor=2.0)
```

### B. Migration Guide

#### Breaking Changes

```markdown
## Migration Guide: v0.1.0 → v0.2.0

### Normalization API

**Before:**
```python
from scptensor.normalization import log_normalize
result = log_normalize(container, assay='proteins', layer='data', base=2.0)
```

**After:**
```python
result = container.normalize("log", assay="proteins", base=2.0)
```

### Imputation API

**Before:**
```python
from scptensor.impute import knn_impute
result = knn_impute(container, assay='proteins', layer='data', n_neighbors=5)
```

**After:**
```python
result = container.impute("knn", assay="proteins", n_neighbors=5)
```

### Integration API

**Before:**
```python
from scptensor.integration import combat_correct
result = combat_correct(container, assay='proteins', layer='data', batch_key='batch')
```

**After:**
```python
result = container.integrate("combat", batch_key="batch")
```
```

### C. Performance Benchmarks

```python
# Benchmark results (target)

## Normalization
- Log normalize: 100K × 100 in 0.5s
- Quantile normalize: 100K × 100 in 2.0s
- Scale normalize: 100K × 100 in 0.3s

## Imputation
- KNN impute: 10K × 100 in 5.0s
- MissForest: 10K × 100 in 30.0s

## Integration
- ComBat: 100K × 100 in 1.0s
- Harmony: 100K × 100 in 5.0s

## Dimensionality Reduction
- PCA: 100K × 100 in 2.0s
- UMAP: 100K × 50 in 10.0s

## Clustering
- KMeans: 100K × 50 in 1.0s
- Leiden: 100K × 50 in 5.0s
```

### D. References

1. **Python Best Practices**
   - PEP 8: Style Guide
   - PEP 484: Type Hints
   - PEP 257: Docstrings

2. **Scientific Python**
   - NumPy Style Guide
   - SciPy Documentation
   - Matplotlib Style Guide

3. **Single-Cell Analysis**
   - Scanpy Best Practices
   - scikit-learn API Design
   - Bioconductor Guidelines

---

## Conclusion

This refactoring initiative represents a significant investment in code quality, maintainability, and user experience. By following this design document, the ScpTensor team will:

1. **Reduce code complexity** by 30% while maintaining functionality
2. **Improve API usability** through fluent interfaces and smart defaults
3. **Ensure mathematical correctness** through comprehensive validation
4. **Maintain performance** through continuous benchmarking
5. **Build for the future** with clean, maintainable code

The success of this refactoring depends on:
- Strict adherence to principles (YAGNI, SRP, functional style)
- Comprehensive testing at every stage
- Continuous validation of algorithms
- Clear communication and coordination

**Let's build a better ScpTensor together!**

---

**Document Version:** 1.0
**Last Updated:** 2026-02-25
**Status:** Ready for Review
**Next Steps:** Team assignment and Week 1 kickoff
