# ScpTensor Core Refactoring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor ScpTensor codebase to reduce code by 30%, improve maintainability, and simplify API

**Architecture:** Module-by-module refactoring with 8 parallel team members, each owning a complete module from analysis to validation

**Tech Stack:** Python 3.12+, NumPy, SciPy, Polars, Numba, pytest, mypy, ruff

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Phase 0: Setup and Preparation](#phase-0-setup-and-preparation)
3. [Module 1: Core Module Refactoring](#module-1-core-module-refactoring)
4. [Module 2: Normalization Module Refactoring](#module-2-normalization-module-refactoring)
5. [Module 3: Imputation Module Refactoring](#module-3-imputation-module-refactoring)
6. [Module 4: Integration Module Refactoring](#module-4-integration-module-refactoring)
7. [Module 5: Quality Control Module Refactoring](#module-5-quality-control-module-refactoring)
8. [Module 6: Dimensionality Reduction & Clustering Refactoring](#module-6-dimensionality-reduction--clustering-refactoring)
9. [Module 7: Feature Selection & Differential Expression Refactoring](#module-7-feature-selection--differential-expression-refactoring)
10. [Module 8: Utilities, Visualization & I/O Refactoring](#module-8-utilities-visualization--io-refactoring)
11. [Testing Strategy](#testing-strategy)
12. [Validation Checklist](#validation-checklist)
13. [Risk Mitigation](#risk-mitigation)

---

## Project Overview

### Current State

| Module | Current LOC | Target LOC | Reduction | Owner |
|--------|-------------|------------|-----------|-------|
| `core/` | 5,475 | ~3,800 | -30% | Member 1 |
| `normalization/` | 658 | ~450 | -32% | Member 2 |
| `impute/` | 2,381 | ~1,600 | -33% | Member 3 |
| `integration/` | 1,708 | ~1,150 | -33% | Member 4 |
| `qc/` | 3,589 | ~2,400 | -33% | Member 5 |
| `dim_reduction/` + `cluster/` | 1,723 | ~1,170 | -32% | Member 6 |
| `feature_selection/` + `diff_expr/` | 4,185 | ~2,800 | -33% | Member 7 |
| `utils/` + `viz/` + `io/` | 12,992 | ~8,350 | -36% | Member 8 |
| **Total** | **34,223** | **~24,000** | **-30%** | **8 Members** |

### Timeline

- **Phase 0:** Setup (Days 1-2)
- **Phase 1:** Analysis & Design (Days 3-5)
- **Phase 2:** Implementation (Days 6-30)
- **Phase 3:** Validation (Days 31-35)
- **Phase 4:** Documentation & Release (Days 36-42)

### Success Criteria

1. **Code Metrics:**
   - Total LOC reduced by 30%
   - All functions < 50 lines
   - Cyclomatic complexity < 10
   - Type coverage > 95%
   - Test coverage > 85%

2. **Quality Metrics:**
   - All tests passing
   - Zero regressions
   - Performance maintained or improved
   - Mathematical correctness verified

3. **Developer Experience:**
   - Average API call length reduced by 40%
   - Required parameters reduced by 40%
   - Full chaining support
   - Complete type inference

---

## Phase 0: Setup and Preparation

### Task 0.1: Create Refactoring Branch

**Owner:** All Members
**Duration:** 30 minutes

**Step 1: Create branch**

```bash
git checkout main
git pull origin main
git checkout -b refactor/2026-02-core-refactoring
git push -u origin refactor/2026-02-core-refactoring
```

**Step 2: Verify branch**

```bash
git branch --show-current
# Expected output: refactor/2026-02-core-refactoring
```

**Step 3: Setup development environment**

```bash
uv sync
uv pip install -e ".[dev]"
```

**Step 4: Run baseline tests**

```bash
uv run pytest tests/ -v --tb=short
# Expected: All tests pass
```

---

### Task 0.2: Establish Code Quality Tools

**Owner:** All Members
**Duration:** 1 hour

**Step 1: Install analysis tools**

```bash
uv pip install radon lizard
```

**Step 2: Generate baseline metrics**

```bash
# Cyclomatic complexity
radon cc scptensor/ -a > metrics/baseline_complexity.txt

# Code duplication
lizard scptensor/ > metrics/baseline_duplication.txt

# Type coverage
uv run mypy scptensor/ --cobertura-xml-report metrics/

# Line counts
find scptensor/ -name "*.py" -type f | xargs wc -l > metrics/baseline_lines.txt
```

**Step 3: Document baseline**

Create `metrics/baseline_summary.md`:

```markdown
# Baseline Metrics

## Code Volume
- Total Lines: [from baseline_lines.txt]
- Number of Files: [count from baseline_lines.txt]

## Complexity
- Average Cyclomatic Complexity: [from baseline_complexity.txt]
- Functions with complexity > 10: [count]

## Duplication
- Duplicate percentage: [from baseline_duplication.txt]
- Most duplicated code: [top 3]
```

**Step 4: Commit baseline**

```bash
git add metrics/
git commit -m "refactor: establish baseline metrics

- Record current code volume, complexity, duplication
- Set reference for refactoring measurements

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Module 1: Core Module Refactoring

**Owner:** Member 1 (Core Module Specialist)
**Current LOC:** 5,475
**Target LOC:** ~3,800
**Reduction:** -30%
**Duration:** 6 days

---

### Task 1.1: Simplify ScpContainer Initialization

**Goal:** Reduce initialization complexity by using dataclasses and smart defaults

**Files:**
- Modify: `scptensor/core/structures.py:1-200`
- Test: `tests/core/test_structures.py:50-150`

**Step 1: Write the failing test**

Create `tests/core/test_container_initialization.py`:

```python
"""Test simplified container initialization."""
import polars as pl
import pytest
from scptensor.core.structures import ScpContainer, Assay, ScpMatrix
import numpy as np


def test_container_minimal_initialization():
    """Test container creation with minimal parameters."""
    obs = pl.DataFrame({"sample_id": ["s1", "s2", "s3"]})
    assay = Assay(
        var=pl.DataFrame({"feature_id": ["f1", "f2"]}),
        layers={"data": ScpMatrix(X=np.random.rand(3, 2))}
    )

    # Before: Required many parameters
    # After: Only requires obs and assays
    container = ScpContainer(obs=obs, assays={"proteins": assay})

    assert container.n_samples == 3
    assert container.n_assays == 1
    assert len(container.history) == 0  # Auto-initialized
    assert container.metadata == {}  # Default empty dict


def test_container_with_metadata():
    """Test container creation with metadata."""
    obs = pl.DataFrame({"sample_id": ["s1", "s2"]})
    assay = Assay(
        var=pl.DataFrame({"feature_id": ["f1"]}),
        layers={"data": ScpMatrix(X=np.random.rand(2, 1))}
    )

    metadata = {"project": "test", "version": "0.2.0"}
    container = ScpContainer(obs=obs, assays={"proteins": assay}, metadata=metadata)

    assert container.metadata == metadata


def test_container_auto_validation():
    """Test automatic validation on initialization."""
    obs = pl.DataFrame({"sample_id": ["s1", "s2"]})
    assay = Assay(
        var=pl.DataFrame({"feature_id": ["f1"]}),
        layers={"data": ScpMatrix(X=np.random.rand(2, 1))}
    )

    # Should not raise
    container = ScpContainer(obs=obs, assays={"proteins": assay})
    assert container is not None


def test_container_dimension_mismatch():
    """Test that dimension mismatches are caught."""
    obs = pl.DataFrame({"sample_id": ["s1", "s2"]})
    assay = Assay(
        var=pl.DataFrame({"feature_id": ["f1", "f2", "f3"]}),
        layers={"data": ScpMatrix(X=np.random.rand(2, 1))}  # Wrong dims
    )

    with pytest.raises(ValueError, match="dimension"):
        ScpContainer(obs=obs, assays={"proteins": assay})
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/core/test_container_initialization.py -v
# Expected: FAIL (simplified API not implemented yet)
```

**Step 3: Write minimal implementation**

Modify `scptensor/core/structures.py`:

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import polars as pl


@dataclass
class ScpContainer:
    """
    Single-cell proteomics data container.

    Parameters
    ----------
    obs : pl.DataFrame
        Sample metadata (n_samples × metadata_cols)
    assays : Dict[str, Assay]
        Named assay registry
    metadata : Dict[str, Any], optional
        Container-level metadata

    Examples
    --------
    >>> container = ScpContainer(obs_df, {"proteins": assay})
    >>> container.n_samples
    100
    """
    obs: pl.DataFrame
    assays: Dict[str, 'Assay']
    metadata: Dict[str, Any] = field(default_factory=dict)
    history: List['ProvenanceLog'] = field(default_factory=list)

    def __post_init__(self):
        """Validate container after initialization."""
        # Auto-validate dimensions
        self._validate_dimensions()

    def _validate_dimensions(self):
        """Validate dimension consistency across assays."""
        n_samples = len(self.obs)
        for name, assay in self.assays.items():
            for layer_name, layer in assay.layers.items():
                if layer.X.shape[0] != n_samples:
                    raise ValueError(
                        f"Assay '{name}', layer '{layer_name}': "
                        f"Expected {n_samples} samples, got {layer.X.shape[0]}"
                    )

    @property
    def n_samples(self) -> int:
        """Number of samples."""
        return len(self.obs)

    @property
    def n_assays(self) -> int:
        """Number of assays."""
        return len(self.assays)

    def copy(self) -> 'ScpContainer':
        """Create deep copy of container."""
        return ScpContainer(
            obs=self.obs.clone(),
            assays={k: v.copy() for k, v in self.assays.items()},
            metadata=self.metadata.copy(),
            history=self.history.copy()
        )
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/core/test_container_initialization.py -v
# Expected: PASS
```

**Step 5: Commit**

```bash
git add scptensor/core/structures.py tests/core/test_container_initialization.py
git commit -m "refactor(core): simplify ScpContainer initialization

- Use dataclass for automatic __init__ and __repr__
- Add smart defaults for metadata and history
- Auto-validate dimensions in __post_init__
- Reduce API complexity by 40%

Before:
    container = ScpContainer(obs, assays, metadata, history)

After:
    container = ScpContainer(obs, assays, metadata={})  # Simple!

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 1.2: Simplify Assay Data Structure

**Goal:** Use dataclass for Assay with smart defaults

**Files:**
- Modify: `scptensor/core/structures.py:200-400`
- Test: `tests/core/test_assay.py:50-100`

**Step 1: Write the failing test**

```python
def test_assay_minimal_initialization():
    """Test assay creation with minimal parameters."""
    var = pl.DataFrame({"feature_id": ["f1", "f2", "f3"]})
    layer = ScpMatrix(X=np.random.rand(10, 3))

    # Minimal initialization
    assay = Assay(var=var, layers={"data": layer})

    assert assay.n_features == 3
    assert assay.n_layers == 1
    assert assay.metadata == {}  # Auto-initialized


def test_assay_layer_management():
    """Test adding and removing layers."""
    assay = Assay(
        var=pl.DataFrame({"feature_id": ["f1"]}),
        layers={"data": ScpMatrix(X=np.random.rand(10, 1))}
    )

    # Add layer
    assay.layers["normalized"] = ScpMatrix(X=np.random.rand(10, 1))
    assert assay.n_layers == 2

    # Remove layer
    del assay.layers["normalized"]
    assert assay.n_layers == 1
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/core/test_assay.py::test_assay_minimal_initialization -v
```

**Step 3: Write minimal implementation**

```python
@dataclass
class Assay:
    """
    Assay data structure.

    Parameters
    ----------
    var : pl.DataFrame
        Feature metadata (n_features × metadata_cols)
    layers : Dict[str, ScpMatrix]
        Named layer registry
    metadata : Dict[str, Any], optional
        Assay-level metadata
    """
    var: pl.DataFrame
    layers: Dict[str, 'ScpMatrix']
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate assay after initialization."""
        self._validate_dimensions()

    def _validate_dimensions(self):
        """Validate dimension consistency across layers."""
        n_features = len(self.var)
        for name, layer in self.layers.items():
            if layer.X.shape[1] != n_features:
                raise ValueError(
                    f"Layer '{name}': Expected {n_features} features, "
                    f"got {layer.X.shape[1]}"
                )

    @property
    def n_features(self) -> int:
        """Number of features."""
        return len(self.var)

    @property
    def n_layers(self) -> int:
        """Number of layers."""
        return len(self.layers)

    def copy(self) -> 'Assay':
        """Create deep copy of assay."""
        return Assay(
            var=self.var.clone(),
            layers={k: v.copy() for k, v in self.layers.items()},
            metadata=self.metadata.copy()
        )
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/core/test_assay.py::test_assay_minimal_initialization -v
```

**Step 5: Commit**

```bash
git add scptensor/core/structures.py tests/core/test_assay.py
git commit -m "refactor(core): simplify Assay structure

- Convert to dataclass
- Add smart defaults for metadata
- Auto-validate dimensions
- Simplify API surface

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 1.3: Optimize ScpMatrix with Slots

**Goal:** Use __slots__ for memory efficiency

**Files:**
- Modify: `scptensor/core/structures.py:400-600`
- Test: `tests/core/test_matrix.py:50-150`

**Step 1: Write the failing test**

```python
def test_matrix_memory_efficiency():
    """Test that __slots__ reduces memory usage."""
    import sys

    matrix = ScpMatrix(
        X=np.random.rand(1000, 100),
        M=np.zeros((1000, 100), dtype=np.uint8)
    )

    # With __slots__, should have no __dict__
    assert not hasattr(matrix, '__dict__')

    # Memory footprint should be smaller
    size_without_slots = sys.getsizeof(matrix.__dict__) if hasattr(matrix, '__dict__') else 0
    assert size_without_slots == 0


def test_matrix_initialization():
    """Test matrix creation with various configurations."""
    # Dense matrix
    m1 = ScpMatrix(X=np.random.rand(10, 5))
    assert m1.X.shape == (10, 5)
    assert m1.M is None  # Default

    # With mask
    m2 = ScpMatrix(
        X=np.random.rand(10, 5),
        M=np.zeros((10, 5), dtype=np.uint8)
    )
    assert m2.M is not None

    # With metadata
    m3 = ScpMatrix(
        X=np.random.rand(10, 5),
        metadata={"quality": "high"}
    )
    assert m3.metadata == {"quality": "high"}
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/core/test_matrix.py::test_matrix_memory_efficiency -v
```

**Step 3: Write minimal implementation**

```python
from typing import Union, Optional
import numpy as np
import scipy.sparse as sp


@dataclass
class ScpMatrix:
    """
    Single-cell proteomics data matrix.

    Parameters
    ----------
    X : Union[np.ndarray, sp.spmatrix]
        Data matrix (n_samples × n_features)
    M : Union[np.ndarray, sp.spmatrix, None], optional
        Mask matrix (provenance tracking)
    metadata : Optional[Dict], optional
        Matrix metadata

    Notes
    -----
    Mask codes:
    - 0: VALID (detected value)
    - 1: MBR (missing between runs)
    - 2: LOD (below detection limit)
    - 3: FILTERED (QC removed)
    - 5: IMPUTED (filled value)
    """
    X: Union[np.ndarray, sp.spmatrix]
    M: Optional[Union[np.ndarray, sp.spmatrix]] = None
    metadata: Optional[Dict[str, Any]] = None

    # Enable __slots__ for memory efficiency
    __slots__ = ('X', 'M', 'metadata')

    def __post_init__(self):
        """Validate matrix after initialization."""
        if self.M is not None:
            if self.M.shape != self.X.shape:
                raise ValueError(
                    f"X and M must have same shape: "
                    f"{self.X.shape} != {self.M.shape}"
                )

    @property
    def shape(self) -> tuple:
        """Matrix shape."""
        return self.X.shape

    @property
    def n_samples(self) -> int:
        """Number of samples."""
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        """Number of features."""
        return self.X.shape[1]

    def copy(self) -> 'ScpMatrix':
        """Create deep copy of matrix."""
        return ScpMatrix(
            X=self.X.copy() if hasattr(self.X, 'copy') else self.X,
            M=self.M.copy() if (self.M is not None and hasattr(self.M, 'copy')) else self.M,
            metadata=self.metadata.copy() if self.metadata else None
        )
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/core/test_matrix.py::test_matrix_memory_efficiency -v
```

**Step 5: Commit**

```bash
git add scptensor/core/structures.py tests/core/test_matrix.py
git commit -m "refactor(core): add __slots__ to ScpMatrix

- Use __slots__ for memory efficiency
- Reduce memory footprint by ~20%
- Maintain clean API

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 1.4: Simplify Provenance Tracking

**Goal:** Use dataclass for ProvenanceLog

**Files:**
- Modify: `scptensor/core/structures.py:600-800`
- Test: `tests/core/test_provenance.py:50-100`

**Step 1: Write the failing test**

```python
def test_provenance_log_creation():
    """Test provenance log creation."""
    from datetime import datetime

    log = ProvenanceLog(
        operation="normalize",
        params={"method": "log", "base": 2.0},
        timestamp=datetime.now()
    )

    assert log.operation == "normalize"
    assert log.params["base"] == 2.0
    assert log.timestamp is not None


def test_container_history_tracking():
    """Test that operations are tracked in history."""
    container = create_test_container()
    initial_history_len = len(container.history)

    # After some operation, history should grow
    # (This will be tested in actual operation tests)

    assert len(container.history) >= initial_history_len
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/core/test_provenance.py::test_provenance_log_creation -v
```

**Step 3: Write minimal implementation**

```python
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ProvenanceLog:
    """
    Provenance tracking log entry.

    Parameters
    ----------
    operation : str
        Operation name (e.g., "normalize", "impute")
    params : Dict[str, Any]
        Operation parameters
    timestamp : datetime, optional
        Operation timestamp (auto-generated if None)
    """
    operation: str
    params: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

    def __repr__(self) -> str:
        """String representation."""
        return f"ProvenanceLog({self.operation}, {self.timestamp})"


def track_operation(container: ScpContainer, operation: str, **params) -> ScpContainer:
    """
    Track operation in container history.

    Parameters
    ----------
    container : ScpContainer
        Input container
    operation : str
        Operation name
    **params
        Operation parameters

    Returns
    -------
    ScpContainer
        Container with updated history
    """
    new_container = container.copy()
    log = ProvenanceLog(operation=operation, params=params)
    new_container.history.append(log)
    return new_container
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/core/test_provenance.py::test_provenance_log_creation -v
```

**Step 5: Commit**

```bash
git add scptensor/core/structures.py tests/core/test_provenance.py
git commit -m "refactor(core): simplify ProvenanceLog

- Convert to dataclass
- Auto-generate timestamp
- Add helper function for tracking
- Reduce code by 15 lines

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 1.5: Extract Mask Operations to Separate Module

**Goal:** Separate concerns - move mask logic to dedicated module

**Files:**
- Create: `scptensor/core/mask.py`
- Modify: `scptensor/core/__init__.py`
- Test: `tests/core/test_mask.py:1-200`

**Step 1: Write the failing test**

```python
"""Test mask operations."""
import numpy as np
import pytest
from scptensor.core.mask import MaskCode, update_mask, get_valid_mask, combine_masks


def test_mask_codes():
    """Test mask code enum."""
    assert MaskCode.VALID == 0
    assert MaskCode.MBR == 1
    assert MaskCode.LOD == 2
    assert MaskCode.FILTERED == 3
    assert MaskCode.IMPUTED == 5


def test_update_mask():
    """Test updating mask at indices."""
    M = np.zeros((5, 5), dtype=np.uint8)
    indices = (np.array([0, 1, 2]), np.array([0, 1, 2]))

    M_new = update_mask(M, indices, MaskCode.IMPUTED)

    assert M_new[0, 0] == MaskCode.IMPUTED
    assert M_new[1, 1] == MaskCode.IMPUTED
    assert M_new[3, 3] == MaskCode.VALID  # Unchanged


def test_get_valid_mask():
    """Test getting valid data mask."""
    M = np.array([
        [0, 1, 2],
        [5, 0, 3]
    ], dtype=np.uint8)

    valid = get_valid_mask(M)

    assert valid[0, 0] == True  # VALID
    assert valid[0, 1] == False  # MBR
    assert valid[0, 2] == False  # LOD
    assert valid[1, 0] == False  # IMPUTED


def test_combine_masks():
    """Test combining multiple masks."""
    M1 = np.array([[0, 1], [2, 0]], dtype=np.uint8)
    M2 = np.array([[5, 0], [0, 3]], dtype=np.uint8)

    combined = combine_masks(M1, M2, priority="highest")

    # Should take highest priority code
    assert combined[0, 0] == max(M1[0, 0], M2[0, 0])
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/core/test_mask.py -v
```

**Step 3: Write minimal implementation**

Create `scptensor/core/mask.py`:

```python
"""
Mask operations for provenance tracking.

Mask codes:
- 0: VALID (detected value)
- 1: MBR (missing between runs)
- 2: LOD (below detection limit)
- 3: FILTERED (QC removed)
- 5: IMPUTED (filled value)
"""
from enum import IntEnum
from typing import Tuple, Union
import numpy as np
import scipy.sparse as sp


class MaskCode(IntEnum):
    """Provenance mask codes."""
    VALID = 0
    MBR = 1      # Missing between runs
    LOD = 2      # Below detection limit
    FILTERED = 3 # QC filtered
    IMPUTED = 5  # Imputed value


def update_mask(
    M: Union[np.ndarray, sp.spmatrix],
    indices: Tuple[np.ndarray, np.ndarray],
    code: MaskCode
) -> Union[np.ndarray, sp.spmatrix]:
    """
    Update mask at specified indices.

    Parameters
    ----------
    M : Union[np.ndarray, sp.spmatrix]
        Mask matrix
    indices : Tuple[np.ndarray, np.ndarray]
        Row and column indices
    code : MaskCode
        New mask code

    Returns
    -------
    Union[np.ndarray, sp.spmatrix]
        Updated mask (new copy)

    Examples
    --------
    >>> M = np.zeros((5, 5), dtype=np.uint8)
    >>> indices = (np.array([0, 1]), np.array([0, 1]))
    >>> M_new = update_mask(M, indices, MaskCode.IMPUTED)
    >>> M_new[0, 0]
    5
    """
    M_new = M.copy()
    M_new[indices] = code
    return M_new


def get_valid_mask(M: Union[np.ndarray, sp.spmatrix]) -> Union[np.ndarray, sp.spmatrix]:
    """
    Get boolean mask for valid data.

    Parameters
    ----------
    M : Union[np.ndarray, sp.spmatrix]
        Mask matrix

    Returns
    -------
    Union[np.ndarray, sp.spmatrix]
        Boolean mask (True where code == VALID)

    Examples
    --------
    >>> M = np.array([[0, 1], [5, 0]], dtype=np.uint8)
    >>> valid = get_valid_mask(M)
    >>> valid
    array([[ True, False],
           [False,  True]])
    """
    return M == MaskCode.VALID


def combine_masks(
    *masks: Union[np.ndarray, sp.spmatrix],
    priority: str = "highest"
) -> Union[np.ndarray, sp.spmatrix]:
    """
    Combine multiple masks.

    Parameters
    ----------
    *masks : Union[np.ndarray, sp.spmatrix]
        Mask matrices to combine
    priority : str, default="highest"
        Combination strategy:
        - "highest": Take highest priority code
        - "lowest": Take lowest priority code
        - "any": Non-zero takes priority

    Returns
    -------
    Union[np.ndarray, sp.spmatrix]
        Combined mask

    Examples
    --------
    >>> M1 = np.array([[0, 1], [2, 0]], dtype=np.uint8)
    >>> M2 = np.array([[5, 0], [0, 3]], dtype=np.uint8)
    >>> combine_masks(M1, M2, priority="highest")
    array([[5, 1],
           [2, 3]], dtype=uint8)
    """
    if len(masks) == 0:
        raise ValueError("At least one mask required")

    if len(masks) == 1:
        return masks[0].copy()

    result = masks[0].copy()
    for mask in masks[1:]:
        if priority == "highest":
            result = np.maximum(result, mask)
        elif priority == "lowest":
            result = np.minimum(result, mask)
        elif priority == "any":
            result = np.where(result != 0, result, mask)
        else:
            raise ValueError(f"Unknown priority: {priority}")

    return result
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/core/test_mask.py -v
```

**Step 5: Commit**

```bash
git add scptensor/core/mask.py tests/core/test_mask.py
git commit -m "refactor(core): extract mask operations to separate module

- Create dedicated mask.py module
- Implement MaskCode enum
- Add update_mask, get_valid_mask, combine_masks
- Separate concerns for better maintainability
- Reduce core/structures.py by 50 lines

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 1.6: Add Fluent Interface Methods to ScpContainer

**Goal:** Enable method chaining for common operations

**Files:**
- Modify: `scptensor/core/structures.py:800-1000`
- Test: `tests/core/test_fluent_interface.py:1-200`

**Step 1: Write the failing test**

```python
def test_fluent_interface():
    """Test method chaining."""
    container = create_test_container()

    # Chain multiple operations
    result = (
        container
        .copy()
        .with_layer("proteins", "test", np.random.rand(10, 5))
    )

    assert isinstance(result, ScpContainer)
    assert "test" in result.assays["proteins"].layers


def test_with_layer():
    """Test adding layer via fluent interface."""
    container = create_test_container()
    new_data = np.random.rand(10, 5)

    result = container.with_layer("proteins", "new_layer", new_data)

    # Original unchanged (immutable)
    assert "new_layer" not in container.assays["proteins"].layers

    # New container has layer
    assert "new_layer" in result.assays["proteins"].layers
    assert np.array_equal(result.assays["proteins"].layers["new_layer"].X, new_data)
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/core/test_fluent_interface.py -v
```

**Step 3: Write minimal implementation**

Add to `ScpContainer` class:

```python
def with_layer(
    self,
    assay: str,
    layer_name: str,
    data: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> 'ScpContainer':
    """
    Return container with new layer (immutable).

    Parameters
    ----------
    assay : str
        Assay name
    layer_name : str
        Layer name
    data : np.ndarray
        Layer data
    mask : Optional[np.ndarray], default=None
        Layer mask

    Returns
    -------
    ScpContainer
        New container with added layer

    Examples
    --------
    >>> container = container.with_layer("proteins", "normalized", data)
    """
    from scptensor.core.structures import ScpMatrix

    new_container = self.copy()
    new_matrix = ScpMatrix(X=data, M=mask)

    if assay not in new_container.assays:
        raise ValueError(f"Assay '{assay}' not found")

    new_container.assays[assay].layers[layer_name] = new_matrix
    return new_container


def with_obs(
    self,
    **columns: np.ndarray
) -> 'ScpContainer':
    """
    Return container with new obs columns (immutable).

    Parameters
    ----------
    **columns : np.ndarray
        Column name and data pairs

    Returns
    -------
    ScpContainer
        New container with added columns

    Examples
    --------
    >>> container = container.with_obs(cluster=np.array([0, 1, 0]))
    """
    new_container = self.copy()
    for col_name, col_data in columns.items():
        new_container.obs = new_container.obs.with_columns(**{col_name: col_data})
    return new_container
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/core/test_fluent_interface.py -v
```

**Step 5: Commit**

```bash
git add scptensor/core/structures.py tests/core/test_fluent_interface.py
git commit -m "refactor(core): add fluent interface to ScpContainer

- Add with_layer() method for immutable layer addition
- Add with_obs() method for immutable metadata addition
- Enable method chaining
- Improve API usability

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 1.7: Simplify JIT Operations

**Goal:** Consolidate JIT functions and reduce complexity

**Files:**
- Modify: `scptensor/core/jit_ops.py:1-500`
- Test: `tests/core/test_jit_ops.py:1-200`

**Step 1: Write the failing test**

```python
def test_jit_mask_operations():
    """Test JIT-compiled mask operations."""
    from scptensor.core.jit_ops import count_mask_codes, apply_mask_threshold

    M = np.array([
        [0, 1, 2, 3, 5],
        [0, 0, 1, 5, 5]
    ], dtype=np.uint8)

    # Count mask codes
    counts = count_mask_codes(M)
    assert counts[0] == 3  # VALID
    assert counts[5] == 3  # IMPUTED

    # Apply threshold
    valid_mask = apply_mask_threshold(M, max_code=0)
    assert np.sum(valid_mask) == 3


def test_jit_performance():
    """Test that JIT provides performance benefit."""
    import time

    from scptensor.core.jit_ops import count_mask_codes

    M = np.random.randint(0, 6, size=(10000, 1000), dtype=np.uint8)

    # First call (compilation)
    start = time.time()
    count_mask_codes(M)
    first_time = time.time() - start

    # Second call (compiled)
    start = time.time()
    count_mask_codes(M)
    second_time = time.time() - start

    # Compiled should be faster or similar
    assert second_time <= first_time * 1.5
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/core/test_jit_ops.py -v
```

**Step 3: Write minimal implementation**

Modify `scptensor/core/jit_ops.py`:

```python
"""
JIT-compiled operations for performance.

Uses Numba JIT to accelerate hot paths.
"""
from numba import jit
import numpy as np


@jit(nopython=True, cache=True)
def count_mask_codes(M: np.ndarray) -> np.ndarray:
    """
    Count occurrences of each mask code.

    Parameters
    ----------
    M : np.ndarray
        Mask matrix (uint8)

    Returns
    -------
    np.ndarray
        Counts for each code (length 6)

    Examples
    --------
    >>> M = np.array([[0, 1], [5, 0]], dtype=np.uint8)
    >>> count_mask_codes(M)
    array([2, 1, 0, 0, 0, 1])
    """
    counts = np.zeros(6, dtype=np.int64)
    n_rows, n_cols = M.shape

    for i in range(n_rows):
        for j in range(n_cols):
            code = M[i, j]
            if code < 6:
                counts[code] += 1

    return counts


@jit(nopython=True, cache=True)
def apply_mask_threshold(
    M: np.ndarray,
    max_code: int
) -> np.ndarray:
    """
    Create boolean mask for values below threshold.

    Parameters
    ----------
    M : np.ndarray
        Mask matrix
    max_code : int
        Maximum code to include

    Returns
    -------
    np.ndarray
        Boolean mask

    Examples
    --------
    >>> M = np.array([[0, 1, 5]], dtype=np.uint8)
    >>> apply_mask_threshold(M, max_code=1)
    array([[ True,  True, False]])
    """
    n_rows, n_cols = M.shape
    result = np.empty((n_rows, n_cols), dtype=np.bool_)

    for i in range(n_rows):
        for j in range(n_cols):
            result[i, j] = M[i, j] <= max_code

    return result


@jit(nopython=True, cache=True)
def filter_by_mask(
    X: np.ndarray,
    M: np.ndarray,
    valid_codes: np.ndarray
) -> np.ndarray:
    """
    Filter data by mask codes.

    Parameters
    ----------
    X : np.ndarray
        Data matrix
    M : np.ndarray
        Mask matrix
    valid_codes : np.ndarray
        Valid mask codes

    Returns
    -------
    np.ndarray
        Filtered data (invalid values set to NaN)

    Examples
    --------
    >>> X = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> M = np.array([[0, 1]], dtype=np.uint8)
    >>> filter_by_mask(X, M, np.array([0]))
    array([[1., nan],
           [3., 4.]])
    """
    n_rows, n_cols = X.shape
    result = X.copy().astype(np.float64)

    for i in range(n_rows):
        for j in range(n_cols):
            code = M[i, j]
            is_valid = False
            for vc in valid_codes:
                if code == vc:
                    is_valid = True
                    break
            if not is_valid:
                result[i, j] = np.nan

    return result
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/core/test_jit_ops.py -v
```

**Step 5: Commit**

```bash
git add scptensor/core/jit_ops.py tests/core/test_jit_ops.py
git commit -m "refactor(core): simplify JIT operations

- Consolidate JIT functions
- Improve performance with selective JIT
- Add comprehensive tests
- Reduce code by 40 lines

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 1.8-1.15: Continue Pattern...

[Continue similar pattern for remaining tasks]

**Remaining tasks for Member 1:**

1.8. Simplify sparse utilities (reduce by 20%)
1.9. Consolidate matrix operations
1.10. Simplify I/O operations
1.11. Reduce type definitions complexity
1.12. Consolidate exception classes
1.13. Simplify filtering operations
1.14. Add comprehensive docstrings
1.15. Final cleanup and validation

---

## Module 2: Normalization Module Refactoring

**Owner:** Member 2 (Normalization Module Specialist)
**Current LOC:** 658
**Target LOC:** ~450
**Reduction:** -32%
**Duration:** 4 days

---

### Task 2.1: Design Unified Normalization Interface

**Goal:** Create consistent API across all normalization methods

**Files:**
- Create: `scptensor/normalization/base.py`
- Test: `tests/normalization/test_base.py:1-100`

**Step 1: Write the failing test**

```python
"""Test unified normalization interface."""
import numpy as np
import pytest
from scptensor.normalization.base import NormalizeMethod, get_method, normalize


def test_method_registry():
    """Test that all methods are registered."""
    methods = ["log", "sqrt", "quantile", "clr", "vst", "tmm", "upperquartile", "cpm"]

    for method in methods:
        assert get_method(method) is not None


def test_normalize_function():
    """Test unified normalize function."""
    from scptensor.core.structures import ScpContainer, Assay, ScpMatrix

    container = create_test_container()

    # Simple normalization
    result = normalize(container, method="log", assay="proteins", base=2.0)

    assert isinstance(result, ScpContainer)
    assert "data_log" in result.assays["proteins"].layers


def test_normalize_with_custom_output():
    """Test normalization with custom output layer."""
    container = create_test_container()

    result = normalize(
        container,
        method="log",
        assay="proteins",
        output_layer="my_normalized"
    )

    assert "my_normalized" in result.assays["proteins"].layers
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/normalization/test_base.py -v
```

**Step 3: Write minimal implementation**

Create `scptensor/normalization/base.py`:

```python
"""
Base classes and unified interface for normalization.
"""
from typing import Protocol, Callable, Dict, Any, Union, Optional
from dataclasses import dataclass
import numpy as np


class NormalizeMethod(Protocol):
    """Protocol for normalization methods."""
    name: str
    validate: Callable[[np.ndarray], bool]
    apply: Callable[[np.ndarray], np.ndarray]


@dataclass
class Normalizer:
    """Unified normalization wrapper."""
    method: NormalizeMethod
    params: Dict[str, Any]

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Apply normalization."""
        if not self.method.validate(data):
            raise ValueError(f"Invalid data for {self.method.name}")
        return self.method.apply(data, **self.params)


# Method registry
_METHODS: Dict[str, NormalizeMethod] = {}


def register_method(method: NormalizeMethod) -> None:
    """Register a normalization method."""
    _METHODS[method.name] = method


def get_method(name: str) -> NormalizeMethod:
    """Get normalization method by name."""
    if name not in _METHODS:
        raise ValueError(f"Unknown normalization method: {name}")
    return _METHODS[name]


def list_methods() -> list[str]:
    """List available normalization methods."""
    return list(_METHODS.keys())
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/normalization/test_base.py -v
```

**Step 5: Commit**

```bash
git add scptensor/normalization/base.py tests/normalization/test_base.py
git commit -m "refactor(normalization): create unified interface

- Add NormalizeMethod protocol
- Implement method registry
- Create base classes for consistency
- Reduce code duplication

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 2.2: Refactor Log Normalization

**Goal:** Simplify log normalization and add mathematical documentation

**Files:**
- Modify: `scptensor/normalization/log.py:1-100`
- Test: `tests/normalization/test_log.py:1-100`

**Step 1: Write the failing test**

```python
def test_log_normalize_formula():
    """Test that log normalization matches formula."""
    data = np.array([[1.0, 2.0], [4.0, 8.0]])

    # Manual calculation: log2(x + 1)
    expected = np.log2(data + 1.0)

    result = log_normalize(data, base=2.0, offset=1.0)

    np.testing.assert_allclose(result, expected)


def test_log_normalize_edge_cases():
    """Test log normalization with edge cases."""
    # Zeros
    data = np.array([[0.0, 1.0], [2.0, 0.0]])
    result = log_normalize(data, base=2.0, offset=1.0)
    assert np.all(np.isfinite(result))

    # Negative values (if offset handles them)
    data = np.array([[-0.5, 0.0], [0.5, 1.0]])
    result = log_normalize(data, base=2.0, offset=1.0)
    assert np.all(np.isfinite(result))
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/normalization/test_log.py -v
```

**Step 3: Write minimal implementation**

```python
"""
Logarithmic normalization.

.. math::

    X_{log,ij} = \\frac{\\log(X_{ij} + c)}{\\log(b)}

where:
- :math:`X` is the input data
- :math:`b` is the logarithm base
- :math:`c` is the offset (default: 1.0)
"""
import numpy as np
from scptensor.normalization.base import register_method, NormalizeMethod


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
        Log-transformed data

    Notes
    -----
    The log transformation is defined as:

    .. math::

        X_{log,ij} = \\frac{\\log(X_{ij} + c)}{\\log(b)}

    For :math:`b = 2` (default), this computes :math:`\\log_2(X + c)`.

    Examples
    --------
    >>> data = np.array([[1, 2, 0], [4, 0, 6]])
    >>> log_normalize(data, base=2.0, offset=1.0)
    array([[1.0, 1.58, 0.0],
           [2.32, 0.0, 2.58]])
    """
    if base <= 0:
        raise ValueError(f"Base must be positive, got {base}")

    if offset < 0:
        raise ValueError(f"Offset must be non-negative, got {offset}")

    return np.log(data + offset) / np.log(base)


def validate_log(data: np.ndarray) -> bool:
    """Validate data for log normalization."""
    return data.size > 0 and np.all(np.isfinite(data + 1.0))


# Register method
register_method(NormalizeMethod(
    name="log",
    validate=validate_log,
    apply=log_normalize
))
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/normalization/test_log.py -v
```

**Step 5: Commit**

```bash
git add scptensor/normalization/log.py tests/normalization/test_log.py
git commit -m "refactor(normalization): simplify log normalization

- Add comprehensive mathematical documentation
- Simplify implementation
- Add validation function
- Register in method registry

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 2.3-2.12: Refactor Remaining Normalization Methods

**Pattern:** Repeat Task 2.2 for each method:

- Task 2.3: Sqrt normalization
- Task 2.4: Quantile normalization
- Task 2.5: CLR normalization
- Task 2.6: VST normalization
- Task 2.7: TMM normalization
- Task 2.8: Upper quartile normalization
- Task 2.9: CPM normalization
- Task 2.10: Add chain calling support

**Each task follows the same 5-step pattern.**

---

### Task 2.11: Add Main normalize() Function

**Goal:** Create single entry point for all normalization

**Files:**
- Create: `scptensor/normalization/normalize.py`
- Test: `tests/normalization/test_normalize.py:1-150`

**Step 1: Write the failing test**

```python
def test_normalize_function():
    """Test main normalize function."""
    container = create_test_container()

    # Normalize with different methods
    result_log = normalize(container, method="log", base=2.0)
    result_sqrt = normalize(container, method="sqrt")

    assert "data_log" in result_log.assays["proteins"].layers
    assert "data_sqrt" in result_sqrt.assays["proteins"].layers


def test_normalize_with_chaining():
    """Test chaining normalize calls."""
    container = create_test_container()

    result = (
        normalize(container, method="log", base=2.0, output_layer="log")
        .normalize("sqrt", assay="proteins", layer="log", output_layer="log_sqrt")
    )

    assert "log_sqrt" in result.assays["proteins"].layers
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/normalization/test_normalize.py -v
```

**Step 3: Write minimal implementation**

Create `scptensor/normalization/normalize.py`:

```python
"""
Main normalization function.
"""
from typing import Union, Optional
import numpy as np
import polars as pl
from scptensor.core.structures import ScpContainer, ScpMatrix
from scptensor.normalization.base import get_method, NormalizeMethod


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

    Notes
    -----
    Available methods:
    - log: Logarithmic transformation
    - sqrt: Square root transformation
    - quantile: Quantile normalization
    - clr: Centered log-ratio
    - vst: Variance stabilizing transformation
    - tmm: Trimmed mean of M-values
    - upperquartile: Upper quartile normalization
    - cpm: Counts per million
    """
    # Get method
    if isinstance(method, str):
        normalize_method = get_method(method)
        method_name = method
    else:
        normalize_method = method
        method_name = method.name

    # Get data
    if assay not in container.assays:
        raise ValueError(f"Assay '{assay}' not found")

    assay_obj = container.assays[assay]
    if layer not in assay_obj.layers:
        raise ValueError(f"Layer '{layer}' not found in assay '{assay}'")

    data = assay_obj.layers[layer].X

    # Apply normalization
    normalized = normalize_method.apply(data, **kwargs)

    # Determine output layer name
    output_layer = output_layer or f"{layer}_{method_name}"

    # Create new layer
    new_matrix = ScpMatrix(X=normalized, M=assay_obj.layers[layer].M)

    # Return new container with added layer
    return container.with_layer(assay, output_layer, normalized)
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/normalization/test_normalize.py -v
```

**Step 5: Commit**

```bash
git add scptensor/normalization/normalize.py tests/normalization/test_normalize.py
git commit -m "refactor(normalization): add unified normalize() function

- Create single entry point
- Support all normalization methods
- Enable method chaining
- Simplify API by 60%

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 2.12: Update Module Exports

**Goal:** Clean up imports and exports

**Files:**
- Modify: `scptensor/normalization/__init__.py`

**Step 1: Write the failing test**

```python
def test_module_exports():
    """Test that public API is exported."""
    from scptensor.normalization import (
        normalize,
        log_normalize,
        sqrt_normalize,
        list_methods
    )

    assert callable(normalize)
    assert callable(log_normalize)
    assert callable(list_methods)
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/normalization/test_exports.py -v
```

**Step 3: Write minimal implementation**

```python
"""
Normalization methods.

Main API:
    normalize: Unified normalization function
    list_methods: List available methods
"""

from scptensor.normalization.base import list_methods
from scptensor.normalization.normalize import normalize
from scptensor.normalization.log import log_normalize
from scptensor.normalization.sqrt import sqrt_normalize
from scptensor.normalization.quantile import quantile_normalize
# ... other methods

__all__ = [
    "normalize",
    "list_methods",
    "log_normalize",
    "sqrt_normalize",
    "quantile_normalize",
    # ...
]
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/normalization/test_exports.py -v
```

**Step 5: Commit**

```bash
git add scptensor/normalization/__init__.py
git commit -m "refactor(normalization): update module exports

- Clean up imports
- Export public API only
- Improve import ergonomics

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Module 3: Imputation Module Refactoring

**Owner:** Member 3 (Imputation Module Specialist)
**Current LOC:** 2,381
**Target LOC:** ~1,600
**Reduction:** -33%
**Duration:** 5 days

---

### Task 3.1: Design Unified Imputation Interface

**Goal:** Create consistent API across all imputation methods

**Files:**
- Create: `scptensor/impute/base.py`
- Test: `tests/impute/test_base.py:1-100`

**Step 1: Write the failing test**

```python
"""Test unified imputation interface."""
import numpy as np
import pytest
from scptensor.impute.base import ImputeMethod, get_impute_method, impute


def test_impute_method_registry():
    """Test that all methods are registered."""
    methods = ["knn", "missforest", "ppca", "svd"]

    for method in methods:
        assert get_impute_method(method) is not None


def test_impute_function():
    """Test unified impute function."""
    container = create_test_container_with_missing()

    result = impute(container, method="knn", assay="proteins", n_neighbors=5)

    assert isinstance(result, ScpContainer)
    assert "data_imputed" in result.assays["proteins"].layers
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/impute/test_base.py -v
```

**Step 3: Write minimal implementation**

Create `scptensor/impute/base.py`:

```python
"""
Base classes and unified interface for imputation.
"""
from typing import Protocol, Callable, Dict, Any, Union
from dataclasses import dataclass
import numpy as np


class ImputeMethod(Protocol):
    """Protocol for imputation methods."""
    name: str
    supports_sparse: bool
    validate: Callable[[np.ndarray], bool]
    apply: Callable[[np.ndarray], np.ndarray]


@dataclass
class Imputer:
    """Unified imputation wrapper."""
    method: ImputeMethod
    params: Dict[str, Any]

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Apply imputation."""
        if not self.method.validate(data):
            raise ValueError(f"Invalid data for {self.method.name}")
        return self.method.apply(data, **self.params)


# Method registry
_IMPUTE_METHODS: Dict[str, ImputeMethod] = {}


def register_impute_method(method: ImputeMethod) -> None:
    """Register an imputation method."""
    _IMPUTE_METHODS[method.name] = method


def get_impute_method(name: str) -> ImputeMethod:
    """Get imputation method by name."""
    if name not in _IMPUTE_METHODS:
        raise ValueError(f"Unknown imputation method: {name}")
    return _IMPUTE_METHODS[name]


def list_impute_methods() -> list[str]:
    """List available imputation methods."""
    return list(_IMPUTE_METHODS.keys())
```

**Step 4-5:** Run tests and commit (same pattern)

---

### Task 3.2: Refactor KNN Imputation

**Goal:** Use scikit-learn and simplify

**Files:**
- Modify: `scptensor/impute/knn.py:1-200`
- Test: `tests/impute/test_knn.py:1-100`

**Step 1: Write the failing test**

```python
def test_knn_impute():
    """Test KNN imputation."""
    data = np.array([
        [1.0, 2.0, np.nan],
        [4.0, np.nan, 6.0],
        [7.0, 8.0, 9.0]
    ])

    result = knn_impute(data, n_neighbors=2)

    assert result.shape == data.shape
    assert np.all(np.isfinite(result))


def test_knn_with_weights():
    """Test KNN with distance weights."""
    data = np.array([
        [1.0, np.nan],
        [2.0, 5.0],
        [3.0, 6.0]
    ])

    result = knn_impute(data, n_neighbors=2, weights="distance")

    assert np.isfinite(result[0, 1])
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/impute/test_knn.py -v
```

**Step 3: Write minimal implementation**

```python
"""
K-nearest neighbors imputation.

.. math::

    X_{ij} = \\sum_{k \\in N_i} w_k X_{kj}

where:
- :math:`N_i` are the k nearest neighbors
- :math:`w_k` are the weights (uniform or distance-based)
"""
import numpy as np
from sklearn.impute import KNNImputer
from scptensor.impute.base import register_impute_method, ImputeMethod


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
        Input data with missing values (NaN)
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

    Examples
    --------
    >>> data = np.array([[1, 2, np.nan], [4, np.nan, 6]])
    >>> knn_impute(data, n_neighbors=2)
    array([[1.0, 2.0, 4.5],
           [4.0, 3.5, 6.0]])
    """
    imputer = KNNImputer(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric
    )
    return imputer.fit_transform(data)


def validate_knn(data: np.ndarray) -> bool:
    """Validate data for KNN imputation."""
    return data.size > 0 and np.any(np.isnan(data))


register_impute_method(ImputeMethod(
    name="knn",
    supports_sparse=True,
    validate=validate_knn,
    apply=knn_impute
))
```

**Step 4-5:** Run tests and commit

---

### Task 3.3-3.10: Continue Pattern...

**Remaining tasks for Member 3:**

3.3. Refactor MissForest imputation (use fancyimpute)
3.4. Refactor PPCA imputation
3.5. Refactor SVD imputation
3.6. Add sparse matrix optimization
3.7. Implement chunked processing for large datasets
3.8. Add boundary condition testing
3.9. Add main impute() function
3.10. Update module exports

---

## Module 4: Integration Module Refactoring

**Owner:** Member 4 (Integration Module Specialist)
**Current LOC:** 1,708
**Target LOC:** ~1,150
**Reduction:** -33%
**Duration:** 4 days

---

### Task 4.1: Design Unified Integration Interface

**Goal:** Create consistent API across all integration methods

**Files:**
- Create: `scptensor/integration/base.py`
- Test: `tests/integration/test_base.py:1-100`

**Step 1: Write the failing test**

```python
"""Test unified integration interface."""
import numpy as np
import pytest
from scptensor.integration.base import IntegrateMethod, get_integrate_method, integrate


def test_integrate_method_registry():
    """Test that all methods are registered."""
    methods = ["combat", "harmony", "mnn", "scanorama", "nonlinear"]

    for method in methods:
        assert get_integrate_method(method) is not None


def test_integrate_function():
    """Test unified integrate function."""
    container = create_test_container_with_batch()

    result = integrate(container, method="combat", batch_key="batch")

    assert isinstance(result, ScpContainer)
    assert "data_integrated" in result.assays["proteins"].layers
```

**Step 2-5:** Same pattern as previous modules

---

### Task 4.2: Refactor ComBat Integration

**Goal:** Simplify and add mathematical documentation

**Files:**
- Modify: `scptensor/integration/combat.py:1-200`
- Test: `tests/integration/test_combat.py:1-100`

**Step 1: Write the failing test**

```python
def test_combat_correction():
    """Test ComBat batch correction."""
    data = np.random.rand(100, 50)
    batch = np.array([0] * 50 + [1] * 50)

    corrected = combat_correct(data, batch)

    assert corrected.shape == data.shape
    assert np.all(np.isfinite(corrected))


def test_combat_with_covariates():
    """Test ComBat with covariates."""
    data = np.random.rand(100, 50)
    batch = np.array([0] * 50 + [1] * 50)
    covariates = np.random.rand(100, 2)

    corrected = combat_correct(data, batch, covariates=covariates)

    assert corrected.shape == data.shape
```

**Step 2-5:** Same pattern

**Step 3 Implementation:**

```python
"""
ComBat batch correction.

.. math::

    Y_{ij}^{*} = \\frac{Y_{ij} - \\hat{\\alpha}_{\\gamma(i)} - \\hat{\\beta}_{\\gamma(i)} X_{ij}}{\\hat{\\delta}_{\\gamma(i)}}

where:
- :math:`Y_{ij}` is the expression value
- :math:`\\gamma(i)` is the batch for sample i
- :math:`\\hat{\\alpha}, \\hat{\\beta}, \\hat{\\delta}` are estimated using empirical Bayes
"""
import numpy as np
from pycombat import ComBat
from scptensor.integration.base import register_integrate_method, IntegrateMethod


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

    Examples
    --------
    >>> data = np.random.rand(100, 50)
    >>> batch = np.array([0] * 50 + [1] * 50)
    >>> corrected = combat_correct(data, batch)
    """
    model = ComBat(data, batch, covariates)
    corrected = model.fit_transform()
    return corrected
```

---

### Task 4.3-4.10: Continue Pattern...

**Remaining tasks for Member 4:**

4.3. Refactor Harmony integration
4.4. Refactor MNN integration
4.5. Refactor Scanorama integration
4.6. Refactor nonlinear integration
4.7. Add diagnostic tools (batch mixing metrics)
4.8. Add main integrate() function
4.9. Add algorithm validation tests
4.10. Update module exports

---

## Module 5: Quality Control Module Refactoring

**Owner:** Member 5 (Quality Control Specialist)
**Current LOC:** 3,589
**Target LOC:** ~2,400
**Reduction:** -33%
**Duration:** 5 days

---

### Task 5.1: Reorganize QC Module Structure

**Goal:** Split into focused submodules

**Files:**
- Create: `scptensor/qc/metrics.py`, `scptensor/qc/outliers.py`, `scptensor/qc/filters.py`
- Test: `tests/qc/test_reorganization.py:1-100`

**Step 1: Write the failing test**

```python
"""Test QC module reorganization."""
from scptensor.qc.metrics import calculate_qc_metrics
from scptensor.qc.outliers import detect_outliers
from scptensor.qc.filters import filter_cells, filter_features


def test_qc_metrics():
    """Test QC metrics calculation."""
    container = create_test_container()

    metrics = calculate_qc_metrics(container)

    assert "n_genes" in metrics
    assert "n_counts" in metrics
    assert "mt_percentage" in metrics


def test_outlier_detection():
    """Test outlier detection."""
    container = create_test_container()

    outliers = detect_outliers(container, method="isolation_forest")

    assert len(outliers) == container.n_samples
    assert outliers.dtype == bool
```

**Step 2-5:** Same pattern

---

### Task 5.2: Simplify QC Pipeline

**Goal:** Create unified QC pipeline

**Files:**
- Create: `scptensor/qc/pipeline.py`
- Test: `tests/qc/test_pipeline.py:1-150`

**Step 1: Write the failing test**

```python
def test_qc_pipeline():
    """Test complete QC pipeline."""
    container = create_test_container()

    result = qc_pipeline(
        container,
        min_genes=200,
        min_cells=3,
        mt_threshold=0.2
    )

    assert result.n_samples <= container.n_samples  # Some filtered


def test_qc_pipeline_with_custom_filters():
    """Test QC pipeline with custom filters."""
    container = create_test_container()

    filters = [
        QcFilter("min_genes", min_genes=200),
        QcFilter("max_mt", mt_threshold=0.2)
    ]

    result = qc_pipeline(container, filters=filters)

    assert isinstance(result, ScpContainer)
```

**Step 2-5:** Same pattern

---

### Task 5.3-5.9: Continue Pattern...

**Remaining tasks for Member 5:**

5.3. Refactor basic QC metrics
5.4. Refactor advanced QC metrics
5.5. Refactor outlier detection methods
5.6. Refactor filtering operations
5.7. Add diagnostic visualization helpers
5.8. Add statistical validation
5.9. Update module exports

---

## Module 6: Dimensionality Reduction & Clustering Refactoring

**Owner:** Member 6 (DR & Clustering Specialist)
**Current LOC:** 1,723
**Target LOC:** ~1,170
**Reduction:** -32%
**Duration:** 5 days

---

### Task 6.1: Refactor PCA

**Goal:** Simplify PCA implementation

**Files:**
- Modify: `scptensor/dim_reduction/pca.py:1-200`
- Test: `tests/dim_reduction/test_pca.py:1-100`

**Step 1: Write the failing test**

```python
def test_pca():
    """Test PCA dimensionality reduction."""
    container = create_test_container()

    result = pca(container, n_components=50)

    assert "pca" in result.assays["proteins"].layers
    assert result.assays["proteins"].layers["pca"].X.shape[1] == 50


def test_pca_with_explained_variance():
    """Test PCA with variance explained."""
    container = create_test_container()

    result = pca(
        container,
        n_components=0.95,  # Keep 95% variance
        return_variance=True
    )

    assert "pca_variance" in result.assays["proteins"].metadata
```

**Step 2-5:** Same pattern

---

### Task 6.2: Refactor UMAP

**Goal:** Simplify UMAP implementation

**Files:**
- Modify: `scptensor/dim_reduction/umap.py:1-200`
- Test: `tests/dim_reduction/test_umap.py:1-100`

**Step 1: Write the failing test**

```python
def test_umap():
    """Test UMAP dimensionality reduction."""
    container = create_test_container()

    result = umap(container, n_neighbors=15, min_dist=0.1)

    assert "umap" in result.assays["proteins"].layers
    assert result.assays["proteins"].layers["umap"].X.shape[1] == 2


def test_umap_3d():
    """Test UMAP with 3 components."""
    container = create_test_container()

    result = umap(container, n_components=3)

    assert result.assays["proteins"].layers["umap"].X.shape[1] == 3
```

**Step 2-5:** Same pattern

---

### Task 6.3: Refactor KMeans Clustering

**Goal:** Simplify KMeans implementation

**Files:**
- Modify: `scptensor/cluster/kmeans.py:1-150`
- Test: `tests/cluster/test_kmeans.py:1-100`

**Step 1: Write the failing test**

```python
def test_kmeans_clustering():
    """Test KMeans clustering."""
    container = create_test_container()

    result = kmeans(container, n_clusters=10)

    assert "cluster" in container.obs.columns
    assert container.obs["cluster"].dtype == int


def test_kmeans_with_init():
    """Test KMeans with custom initialization."""
    container = create_test_container()

    result = kmeans(
        container,
        n_clusters=10,
        init="k-means++",
        n_init=10
    )

    assert "cluster" in container.obs.columns
```

**Step 2-5:** Same pattern

---

### Task 6.4: Refactor Graph Clustering

**Goal:** Simplify graph-based clustering

**Files:**
- Modify: `scptensor/cluster/graph.py:1-200`
- Test: `tests/cluster/test_graph.py:1-100`

**Step 1-5:** Same pattern

---

### Task 6.5-6.12: Continue Pattern...

**Remaining tasks for Member 6:**

6.5. Add unified dimensionality reduction interface
6.6. Add unified clustering interface
6.7. Add pipeline integration (dim_red → clustering)
6.8. Add cluster evaluation metrics
6.9. Refactor embedding space management
6.10. Add visualization helpers
6.11. Add comprehensive tests
6.12. Update module exports

---

## Module 7: Feature Selection & Differential Expression Refactoring

**Owner:** Member 7 (Feature Selection & DE Specialist)
**Current LOC:** 4,185
**Target LOC:** ~2,800
**Reduction:** -33%
**Duration:** 5 days

---

### Task 7.1: Refactor HVG Selection

**Goal:** Simplify highly variable gene selection

**Files:**
- Modify: `scptensor/feature_selection/hvg.py:1-200`
- Test: `tests/feature_selection/test_hvg.py:1-100`

**Step 1: Write the failing test**

```python
def test_hvg_selection():
    """Test highly variable gene selection."""
    container = create_test_container()

    result = select_hvg(container, n_features=2000)

    assert "highly_variable" in result.assays["proteins"].var.columns
    assert result.assays["proteins"].var["highly_variable"].sum() == 2000


def test_hvg_with_flavor():
    """Test HVG with different flavors."""
    container = create_test_container()

    result = select_hvg(
        container,
        n_features=2000,
        flavor="seurat_v3"
    )

    assert "highly_variable" in result.assays["proteins"].var.columns
```

**Step 2-5:** Same pattern

---

### Task 7.2: Refactor Differential Expression

**Goal:** Simplify DE analysis

**Files:**
- Modify: `scptensor/diff_expr/core.py:1-300`
- Test: `tests/diff_expr/test_core.py:1-150`

**Step 1: Write the failing test**

```python
def test_differential_expression():
    """Test differential expression analysis."""
    container = create_test_container_with_groups()

    result = differential_expression(
        container,
        groupby="cluster",
        group1="0",
        group2="1",
        method="wilcoxon"
    )

    assert isinstance(result, pd.DataFrame)
    assert "pval" in result.columns
    assert "logFC" in result.columns


def test_de_with_multiple_testing():
    """Test DE with multiple testing correction."""
    container = create_test_container_with_groups()

    result = differential_expression(
        container,
        groupby="cluster",
        group1="0",
        group2="1",
        method="wilcoxon",
        corr_method="bonferroni"
    )

    assert "pval_adj" in result.columns
```

**Step 2-5:** Same pattern

---

### Task 7.3-7.10: Continue Pattern...

**Remaining tasks for Member 7:**

7.3. Refactor VST feature selection
7.4. Refactor dropout rate feature selection
7.5. Refactor model-based feature selection
7.6. Refactor nonparametric DE tests
7.7. Refactor count model-based DE
7.8. Add statistical validation
7.9. Add confidence intervals
7.10. Update module exports

---

## Module 8: Utilities, Visualization & I/O Refactoring

**Owner:** Member 8 (Utils/Viz/I/O Specialist)
**Current LOC:** 12,992
**Target LOC:** ~8,350
**Reduction:** -36%
**Duration:** 6 days

---

### Task 8.1: Reorganize Utils Module

**Goal:** Split into focused submodules

**Files:**
- Create: `scptensor/utils/stats.py`, `scptensor/utils/transform.py`, `scptensor/utils/batch.py`
- Test: `tests/utils/test_reorganization.py:1-100`

**Step 1: Write the failing test**

```python
"""Test utils reorganization."""
from scptensor.utils.stats import compute_statistics
from scptensor.utils.transform import apply_transform
from scptensor.utils.batch import batch_apply


def test_stats_utils():
    """Test statistical utilities."""
    data = np.random.rand(100, 50)

    stats = compute_statistics(data)

    assert "mean" in stats
    assert "std" in stats
    assert "median" in stats


def test_transform_utils():
    """Test transform utilities."""
    data = np.random.rand(100, 50)

    result = apply_transform(data, "log")

    assert result.shape == data.shape
```

**Step 2-5:** Same pattern

---

### Task 8.2: Simplify Visualization Base

**Goal:** Extract common plotting patterns

**Files:**
- Modify: `scptensor/viz/base/*.py`
- Test: `tests/viz/test_base.py:1-200`

**Step 1: Write the failing test**

```python
def test_scatter_plot():
    """Test scatter plot."""
    import matplotlib.pyplot as plt
    from scptensor.viz.base import scatter

    fig, ax = scatter(
        x=np.random.rand(100),
        y=np.random.rand(100),
        c=np.random.rand(100)
    )

    assert ax is not None
    plt.close(fig)


def test_heatmap():
    """Test heatmap."""
    from scptensor.viz.base import heatmap

    fig, ax = heatmap(np.random.rand(10, 10))

    assert ax is not None
    plt.close(fig)
```

**Step 2-5:** Same pattern

---

### Task 8.3: Simplify Visualization Recipes

**Goal:** Reduce code duplication in recipes

**Files:**
- Modify: `scptensor/viz/recipes/*.py`
- Test: `tests/viz/test_recipes.py:1-200`

**Step 1: Write the failing test**

```python
def test_qc_recipe():
    """Test QC visualization recipe."""
    container = create_test_container()

    fig = plot_qc_metrics(container)

    assert fig is not None
    plt.close(fig)


def test_embedding_recipe():
    """Test embedding visualization recipe."""
    container = create_test_container()

    fig = plot_embedding(container, method="umap")

    assert fig is not None
    plt.close(fig)
```

**Step 2-5:** Same pattern

---

### Task 8.4: Simplify I/O Operations

**Goal:** Create unified read/write interface

**Files:**
- Modify: `scptensor/io/*.py`
- Test: `tests/io/test_unified.py:1-150`

**Step 1: Write the failing test**

```python
def test_read_function():
    """Test unified read function."""
    import tempfile
    import os

    # Create test file
    container = create_test_container()

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = f.name

    try:
        # Write
        write(container, path, format="npz")

        # Read
        result = read(path, format="npz")

        assert isinstance(result, ScpContainer)
        assert result.n_samples == container.n_samples
    finally:
        os.unlink(path)


def test_auto_format_detection():
    """Test automatic format detection."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".npz") as f:
        # Format should be detected from extension
        write(container, f.name)
        result = read(f.name)

        assert isinstance(result, ScpContainer)
```

**Step 2-5:** Same pattern

---

### Task 8.5-8.12: Continue Pattern...

**Remaining tasks for Member 8:**

8.5. Refactor statistical utilities
8.6. Refactor transform utilities
8.7. Refactor batch processing utilities
8.8. Simplify base plotting functions
8.9. Refactor QC visualization recipes
8.10. Refactor embedding visualization recipes
8.11. Refactor other visualization recipes
8.12. Update module exports

---

## Testing Strategy

### Test Coverage Requirements

1. **Unit Tests:** Each function has dedicated unit test
2. **Integration Tests:** Module interaction tests
3. **Regression Tests:** Ensure no functionality lost
4. **Performance Tests:** Benchmark before/after

### Test Organization

```
tests/
├── core/
│   ├── test_structures.py
│   ├── test_mask.py
│   ├── test_jit_ops.py
│   └── test_fluent_interface.py
├── normalization/
│   ├── test_base.py
│   ├── test_log.py
│   ├── test_sqrt.py
│   └── ...
├── impute/
├── integration/
├── qc/
├── dim_reduction/
├── cluster/
├── feature_selection/
├── diff_expr/
├── utils/
├── viz/
└── io/
```

### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific module
uv run pytest tests/core/ -v

# Run with coverage
uv run pytest --cov=scptensor --cov-report=html

# Run regression tests
uv run pytest tests/ -m regression
```

---

## Validation Checklist

### Phase 1: Code Quality Validation

- [ ] All functions < 50 lines
- [ ] Cyclomatic complexity < 10
- [ ] Code duplication < 3%
- [ ] Type coverage > 95%
- [ ] All type hints correct
- [ ] All docstrings complete (NumPy style)

### Phase 2: Functional Validation

- [ ] All tests pass (>85% coverage)
- [ ] No regressions (compare with baseline)
- [ ] All examples in docstrings work
- [ ] Tutorial notebooks run without errors

### Phase 3: Performance Validation

- [ ] Benchmarks run successfully
- [ ] Performance maintained or improved
- [ ] Memory usage reduced where expected
- [ ] No memory leaks

### Phase 4: Mathematical Validation

- [ ] All algorithms have LaTeX formulas
- [ ] Formulas verified with reference implementations
- [ ] Edge cases tested
- [ ] Numerical stability verified

### Phase 5: API Validation

- [ ] Average API call length reduced by 40%
- [ ] Required parameters reduced by 40%
- [ ] Method chaining works for all operations
- [ ] Type inference automatic
- [ ] Smart defaults work as expected

---

## Risk Mitigation

### Risk 1: Breaking Changes

**Mitigation:**
- Comprehensive test suite before refactoring
- Incremental changes with continuous testing
- Version bump to v0.2.0
- Migration guide in documentation

### Risk 2: Performance Regression

**Mitigation:**
- Benchmark suite run before/after
- Performance tests in CI/CD
- JIT compilation preserved
- Profile hot paths

### Risk 3: Loss of Functionality

**Mitigation:**
- Feature parity checklist
- Integration tests for critical paths
- Real data validation
- User acceptance testing

### Risk 4: Coordination Overhead

**Mitigation:**
- Clear module ownership
- Daily sync meetings
- Shared task tracking
- Code review process

### Risk 5: Timeline Slip

**Mitigation:**
- Buffer time in schedule
- Parallel work streams
- Early risk identification
- Regular progress reviews

---

## Success Metrics

### Quantitative Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Total LOC | 34,223 | ~24,000 | `find . -name "*.py" \| xargs wc -l` |
| Test Coverage | ~60% | >85% | `pytest --cov` |
| Type Coverage | ~70% | >95% | `mypy` |
| Avg Function Length | Unknown | <50 lines | `radon cc` |
| Cyclomatic Complexity | Unknown | <10 | `radon cc` |
| Code Duplication | Unknown | <3% | `lizard` |

### Qualitative Metrics

- [ ] API easier to use (user feedback)
- [ ] Code easier to understand (review feedback)
- [ ] Faster onboarding for new developers
- [ ] Fewer bugs reported
- [ ] Better documentation

---

## Timeline Summary

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1 | Setup & Analysis | Branch, baseline metrics, module analysis |
| 2-3 | Implementation (Modules 1-4) | Core, Normalization, Imputation, Integration |
| 4-5 | Implementation (Modules 5-8) | QC, DR/Cluster, Feat/DE, Utils/Viz/IO |
| 5 | Validation | Test suite, benchmarks, mathematical verification |
| 6 | Documentation & Release | Migration guide, release notes, v0.2.0 |

---

## Appendix: Task Template

**Task X.Y: [Task Name]**

**Goal:** [Brief description]

**Files:**
- Modify/Create: `path/to/file.py:line-range`
- Test: `path/to/test.py:line-range`

**Step 1: Write the failing test**

```python
def test_name():
    """Test description."""
    # Arrange
    input_data = ...

    # Act
    result = function_under_test(input_data)

    # Assert
    assert expected == actual
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest path/to/test.py::test_name -v
# Expected: FAIL
```

**Step 3: Write minimal implementation**

```python
def function_under_test(data):
    """
    Function description.

    Parameters
    ----------
    data : type
        Description

    Returns
    -------
    type
        Description
    """
    # Implementation
    return result
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest path/to/test.py::test_name -v
# Expected: PASS
```

**Step 5: Commit**

```bash
git add [files]
git commit -m "refactor(module): description

- Change 1
- Change 2

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

**Document Version:** 1.0
**Last Updated:** 2026-02-25
**Status:** Ready for Implementation
