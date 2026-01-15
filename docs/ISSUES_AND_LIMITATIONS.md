# ScpTensor: Known Issues and Limitations

**Version:** 0.1.0
**Last Updated:** 2025-01-05
**Status:** Early Development - Not Production Ready

---

## Executive Summary

ScpTensor is currently in **early prototype stage**. While the core data structure design (`ScpContainer`, `Assay`, `ScpMatrix`) demonstrates solid architectural principles, the framework suffers from critical issues that **block production deployment**:

- Multiple core modules cannot be imported
- No unit test coverage
- Incomplete feature implementations
- Missing documentation infrastructure

**Estimated Effort to Production:** 2-3 months of focused development

---

## Critical Issues (Blockers)

### 1. Module Architecture Failures

**Severity:** 🔴 CRITICAL
**Impact:** Core functionality inaccessible via standard imports

#### 1.1 Missing `__init__.py` Files

Several directories contain implementation code but lack `__init__.py`, preventing module imports:

```bash
scptensor/integration/    # ❌ Missing __init__.py
scptensor/qc/             # ❌ Missing __init__.py
```

**Evidence:**
```python
# This fails despite file existing:
from scptensor.integration import combat  # ModuleNotFoundError
from scptensor.qc import basic           # ModuleNotFoundError
```

**Workarounds Currently Used:**
```python
# Direct file imports (brittle):
from scptensor.integration.combat import combat  # Works but fragile
```

**Impact:**
- README documentation examples are non-functional
- Users cannot follow standard import patterns
- Breaks Python packaging conventions

---

#### 1.2 Empty Module Implementations

Multiple modules exist as stubs only:

| File | Status | Lines |
|------|--------|-------|
| `scptensor/integration/harmony.py` | Empty (0 bytes) | 0 |
| `scptensor/integration/mnn.py` | Empty (0 bytes) | 0 |
| `scptensor/integration/scanorama.py` | Empty (0 bytes) | 0 |
| `scptensor/diff_expr/__init__.py` | Empty | 0 |
| `scptensor/diff_expr/*.py` | Non-existent | N/A |

**README Claims vs Reality:**

| Module | README Promise | Actual Status |
|--------|----------------|---------------|
| `integration` | combat, harmony, mnn, scanorama | Only combat implemented |
| `diff_expr` | Differential expression | Completely missing |
| `impute` | knn, missforest, ppca, svd | Only knn complete |

---

### 2. Zero Unit Test Coverage

**Severity:** 🔴 CRITICAL
**Impact:** No correctness guarantees, high regression risk

#### 2.1 Missing Test Files

```bash
find . -name "test_*.py" -o -name "*_test.py"
# Returns: 0 results in project root (only venv results)
```

**Current Testing Strategy:**
- Single integration test: `tests/total_proc.py` (291 lines)
- No unit tests for:
  - Core data structures (`ScpMatrix`, `Assay`, `ScpContainer`)
  - Algorithm implementations (KNN, ComBat, PCA, etc.)
  - Edge cases and error handling
  - Type validation

**Risk Examples:**
```python
# scptensor/core/structures.py - Unvalidated assumptions:
X = X_df.to_numpy().T  # Crashes if data contains non-numeric types
# No try-except, no dtype validation
```

---

#### 2.2 Test Infrastructure Issues

**Missing Components:**
- ❌ No `pytest.ini` or `pyproject.toml` test configuration
- ❌ No test fixtures for synthetic data
- ❌ No coverage tracking (pytest-cov)
- ❌ No continuous integration

**Recommendation:**
```toml
# Required in pyproject.toml:
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "--cov=scptensor --cov-report=html --cov-report=term"
```

---

### 3. Documentation Gaps

**Severity:** 🟡 HIGH
**Impact:** Poor usability, unmaintainable codebase

#### 3.1 Missing Documentation Infrastructure

```bash
ls docs/
# Returns: No such file or directory (before this document)
```

**What's Missing:**
- ❌ API documentation (Sphinx/MkDocs)
- ❌ Installation guide beyond README
- ❌ Tutorial notebooks
- ❌ Developer contribution guide
- ❌ Architecture decision records (ADRs)

---

#### 3.2 Code Documentation Issues

**Example 1: Missing Docstrings**
```python
# scptensor/impute/knn.py
def knn(container, assay_name, base_layer, new_layer_name, k=5):
    # What does this function return?
    # What are the preconditions?
    # What exceptions can be raised?
    pass
```

**Example 2: Mixed Language Comments**
```python
# scptensor/core/structures.py:47
"""
记录对容器执行的操作历史。
Record of operations performed on the container.
"""
```
**Issue:** Bilingual docstrings violate documentation best practices.

---

## High-Priority Issues

### 4. Dependency Management Problems

**Severity:** 🟠 MEDIUM-HIGH
**Location:** `pyproject.toml`

#### 4.1 Duplicate Dependency Declarations

```toml
# Current pyproject.toml:
dependencies = [
    # ... runtime deps ...
]

[dependency-groups]
dev = [
    "combat>=0.3.3",      # ❌ Should be in dependencies if used by core code
    "scanpy>=1.11.5",     # ❌ Same issue
]
```

**Problem:** `combat` is imported in `scptensor/integration/combat.py`, which is part of the core library, not just development.

---

#### 4.2 Misclassified Dependencies

| Package | Current Location | Should Be |
|---------|-----------------|-----------|
| `combat` | `dev` | `dependencies` (used in integration module) |
| `scanpy` | `dev` | `dev` or optional (used in benchmarking) |
| `ipykernel` | `dev` | ✅ Correct |
| `pytest` | `dev` | ✅ Correct |

---

### 5. Code Quality Concerns

**Severity:** 🟠 MEDIUM-HIGH

#### 5.1 Inconsistent Type Annotations

**Good Example:**
```python
# scptensor/core/structures.py
def __init__(
    self,
    obs: pl.DataFrame,
    assays: Optional[Dict[str, Assay]] = None
):
    pass
```

**Problematic Examples:**
```python
# scptensor/integration/combat.py
def combat(container, batch_key, assay_name, base_layer, new_layer_name):
    # No type hints for parameters
    # No return type annotation
    pass

# scptensor/impute/knn.py
def knn(container, assay_name, base_layer, new_layer_name, k=5):
    # Same issue
    pass
```

**Impact:**
- Reduced IDE support
- No static type checking benefits
- Harder to understand function contracts

---

#### 5.2 Hardcoded Paths

**Location:** `tests/total_proc.py:45`

```python
data = pl.read_csv(
    "/home/shenshang/projects/ScpTensor/tests/data/PXD061065/..."
)
```

**Issue:** Non-portable absolute path breaks on other machines.

**Fix Required:**
```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
data_path = PROJECT_ROOT / "tests" / "data" / "PXD061065" / "..."
```

---

#### 5.3 Missing Error Handling

**Example: Data Loading**
```python
# tests/total_proc.py:59
X_df = data.select(quantity_columns)
X = X_df.to_numpy().T  # Crashes if non-numeric data present
# No try-except, no validation
```

**Example: Matrix Operations**
```python
# scptensor/core/structures.py
if matrix.X.shape[1] != self.n_features:
    raise ValueError(...)
# Good! But many other operations lack validation
```

---

### 6. Performance Claims vs Reality

**Severity:** 🟡 MEDIUM
**Issue:** README overstates current performance optimizations

#### 6.1 Numba Not Utilized

**README Claim:**
> "By leveraging high-performance libraries such as Polars, NumPy, SciPy, and **Numba**"

**Actual Usage:**
```bash
grep -r "@jit\|@njit\|numba." scptensor/
# Returns: No results
```

**Impact:** Critical loops (e.g., KNN imputation) run at pure Python speeds.

---

#### 6.2 Sparse Matrix Inefficiency

**Observation:**
```python
# Multiple unnecessary conversions to dense:
X_dense = sparse_matrix.toarray()  # Memory explosion for large datasets
```

**Required:**
- Audit all `.toarray()` calls
- Implement sparse-safe operations where possible
- Benchmark memory usage for 10K+ features

---

## Medium-Priority Issues

### 7. Missing Engineering Infrastructure

**Severity:** 🟠 MEDIUM

| Component | Status | Priority |
|-----------|--------|----------|
| CI/CD (GitHub Actions) | ❌ Missing | HIGH |
| Code formatting (black/ruff) | ❌ Not configured | MEDIUM |
| Linting (pylint/mypy) | ❌ Not configured | MEDIUM |
| Pre-commit hooks | ❌ Not set up | MEDIUM |
| CHANGELOG.md | ❌ Missing | LOW |
| LICENSE file | ❌ Missing | HIGH (legal) |

---

### 8. Feature Completeness Gaps

**Severity:** 🟡 MEDIUM

#### 8.1 Imputation Module

| Method | Status | Completion |
|--------|--------|------------|
| KNN | ✅ Implemented | 100% |
| MissForest | ⚠️ Partial | ~60% |
| PPCA | ⚠️ Partial | ~40% |
| SVD | ⚠️ Partial | ~40% |

---

#### 8.2 Integration Module

| Method | Status | File Size |
|--------|--------|-----------|
| ComBat | ✅ Implemented | 10.6 KB |
| Harmony | ❌ Empty | 0 bytes |
| MNN | ❌ Empty | 0 bytes |
| Scanorama | ❌ Empty | 0 bytes |

---

#### 8.3 Quality Control Module

**Issue:** `qc/` module exists but cannot be imported (no `__init__.py`).

**Current State:**
- Implementation files exist (basic.py, outlier.py)
- Cannot use via `from scptensor.qc import basic`
- Must use direct imports: `from scptensor.qc.basic import basic_qc`

---

## Low-Priority Issues

### 9. Minor Code Smells

1. **Inconsistent Naming:**
   - `sample_median_normalization` vs `median_centering` (which is it?)
   - Mix of `normalize` and `normalization` in function names

2. **Magic Numbers:**
   ```python
   # benchmark_example.py
   n_samples=500  # Why 500? Document rationale.
   n_features=6000
   ```

3. **Commented-Out Code:**
   - Multiple sections in `tests/total_proc.py`
   - Should be removed or documented

---

## Recommended Fix Priorities

### Phase 1: Critical Architecture (Week 1)

1. ✅ Add `__init__.py` to all modules
2. ✅ Export public APIs via `__all__`
3. ✅ Remove or stub empty files with TODO markers
4. ✅ Fix hardcoded paths

**Estimated Effort:** 3-5 days

---

### Phase 2: Testing Foundation (Week 2)

1. ✅ Set up pytest infrastructure
2. ✅ Write tests for core structures (50+ tests)
3. ✅ Add CI/CD pipeline
4. ✅ Configure coverage reporting

**Estimated Effort:** 5-7 days

---

### Phase 3: Feature Completion (Weeks 3-4)

1. ✅ Implement missing integration methods (Harmony, MNN) OR remove stubs
2. ✅ Complete imputation methods (PPCA, SVD)
3. ✅ Add comprehensive error handling
4. ✅ Type hint all public functions

**Estimated Effort:** 10-14 days

---

### Phase 4: Documentation & Polish (Week 5-6)

1. ✅ Generate API docs with Sphinx
2. ✅ Write tutorial notebooks
3. ✅ Add developer guide
4. ✅ Create contribution guidelines

**Estimated Effort:** 7-10 days

---

### Phase 5: Performance Optimization (Optional)

1. ✅ Profile hotspots with cProfile
2. ✅ Add Numba JIT to critical loops
3. ✅ Optimize sparse matrix operations
4. ✅ Benchmark against competitors

**Estimated Effort:** 5-7 days

---

## Version Roadmap Suggestion

### v0.1.0-alpha (Current)
- ❌ Cannot release in current state

### v0.1.0-beta (Target: 6 weeks)
- ✅ All modules importable
- ✅ 80%+ test coverage for core
- ✅ Basic API documentation
- ✅ CI/CD operational

### v0.1.0 (Target: 8 weeks)
- ✅ All advertised features working
- ✅ Complete documentation
- ✅ Performance benchmarks
- ✅ Tutorial notebooks

---

## Conclusion

ScpTensor has a **solid foundation** but requires significant work before production use. The main issues are:

1. **Architectural:** Missing `__init__.py` files block core functionality
2. **Testing:** Zero unit tests is unacceptable for scientific software
3. **Completeness:** Many advertised features are stubs or missing
4. **Engineering:** No CI/CD, formatting, or quality checks

**Recommendation:** Treat as pre-alpha software. Do not use for production analysis until Phase 1-3 are complete.

---

## Document Metadata

- **Maintainer:** ScpTensor Development Team
- **Review Cycle:** Weekly during active development
- **Next Review:** 2025-01-12
- **Related Documents:**
  - `README.md` (User overview)
  - `docs/ROADMAP.md` (Future features - to be created)
  - `docs/CONTRIBUTING.md` (Development guide - to be created)
