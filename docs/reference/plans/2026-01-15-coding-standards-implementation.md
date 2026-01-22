# Coding Standards Implementation Plan

**Date:** 2026-01-15
**Version:** 1.0
**Status:** Ready for Implementation

---

## Overview

This plan implements the ScpTensor coding standards defined in `SCPTENSOR_CODING_STANDARDS.md`. The approach is hybrid: establish standards + demonstrate with core module, then align other modules incrementally.

---

## Strategy

1. **Write Standards** ✅ - Create `SCPTENSOR_CODING_STANDARDS.md`
2. **Audit API** ✅ - Create `SCPTENSOR_API_AUDIT.md`
3. **Demonstrate** - Refactor `scptensor/core/` as reference implementation
4. **Align** - Incrementally update other modules
5. **Enforce** - Update CI/CD to check standards compliance

---

## Tasks

### Task 1: Core Module Refactoring

**Module:** `scptensor/core/`

**Files:**
- `structures.py` - Core data classes
- `exceptions.py` - Exception hierarchy
- `matrix_ops.py` - Matrix operations
- `sparse_utils.py` - Sparse utilities
- `io.py` - I/O utilities

**Changes:**

1.1 **structures.py** - ScpContainer class
- Rename `get_assay(name)` → `get_assay(assay_name)` (add alias for backward compat)
- Add `list_assays()` method
- Add `list_layers(assay_name)` method
- Add `summary()` method
- Update all method docstrings to NumPy style
- Add missing Returns/Raises sections

1.2 **structures.py** - Assay class
- Standardize `layer` → `layer_name` parameters
- Add type annotations to all methods
- Update docstrings

1.3 **structures.py** - ScpMatrix class
- Review parameter naming consistency
- Add missing docstrings

1.4 **exceptions.py**
- Ensure all exceptions have clear docstrings
- Standardize error message format

1.5 **matrix_ops.py** & **sparse_utils.py**
- Review function signatures for consistency
- Add type annotations
- Improve error messages

**Tests:**
- Run `pytest tests/core/` to verify no breaking changes
- Add tests for new convenience methods

**Commit:** `refactor(core): standardize API and add convenience methods`

---

### Task 2: Normalization Module Refactoring

**Module:** `scptensor/normalization/`

**Files:** All normalization method modules

**Changes:**

2.1 Standardize parameter names:
- `base_layer` → `source_layer`
- `new_layer` → `new_layer_name`
- Update all docstrings

2.2 Improve error messages:
- Add available assays/layers to KeyError
- Add parameter validation with helpful errors

2.3 Verify type annotations:
- Complete any missing type hints
- Fix any mypy warnings

**Tests:**
- Run `pytest tests/test_normalization.py`
- Update any tests affected by parameter renames

**Commit:** `refactor(normalization): standardize parameter names and error messages`

---

### Task 3: Imputation Module Refactoring

**Module:** `scptensor/impute/`

**Files:** All imputation method modules

**Changes:**

3.1 Standardize parameter names:
- `layer` → `source_layer`
- `output_layer` → `new_layer_name`

3.2 Improve error messages

3.3 Verify type annotations

**Tests:**
- Run `pytest tests/test_impute.py`

**Commit:** `refactor(impute): standardize parameter names and error messages`

---

### Task 4: QC Module Refactoring

**Module:** `scptensor/qc/`

**Files:** `basic.py`, `batch.py`, `bivariate.py`

**Changes:**

4.1 Standardize parameter names:
- `obs_subset` → `obs_names`

4.2 Improve error messages

4.3 Verify type annotations

**Tests:**
- Run `pytest tests/test_qc.py`

**Commit:** `refactor(qc): standardize parameter names and error messages`

---

### Task 5: Developer Experience Enhancements

**Changes:**

5.1 Add `_repr_html_()` to ScpContainer for Jupyter display

5.2 Add fuzzy matching to KeyError messages:
```python
def _find_closest_match(input_name: str, options: Collection[str]) -> str | None:
    from difflib import get_close_matches
    matches = get_close_matches(input_name, options, n=1, cutoff=0.7)
    return matches[0] if matches else None
```

5.3 Add warnings for deprecated parameters with migration guide

**Tests:**
- Test Jupyter display rendering
- Test fuzzy matching suggestions
- Test deprecation warnings

**Commit:** `feat(core): add developer experience enhancements`

---

### Task 6: CI/CD Updates

**Changes:**

6.1 Update pre-commit hooks:
- Add ruff format check
- Add mypy strict mode (incremental)

6.2 Add CI job for standards compliance check

**Commit:** `ci: add standards compliance checks`

---

### Task 7: Documentation Updates

**Changes:**

7.1 Update ROADMAP.md to mark this work

7.2 Update tutorial notebooks to use new parameter names

7.3 Add migration guide for breaking changes

**Commit:** `docs: update for API standards`

---

## Execution Order

1. **Task 1** (Core) - Establish reference implementation
2. **Task 2** (Normalization) - Apply to high-usage module
3. **Task 3** (Imputation) - Continue pattern
4. **Task 4** (QC) - Continue pattern
5. **Task 5** (DX) - Enhance experience after standardization
6. **Task 6** (CI) - Enforce going forward
7. **Task 7** (Docs) - Document changes

---

## Notes

- Each task should be done in a separate branch for easy review
- Run full test suite after each task
- Update API_AUDIT.md as fixes are completed
- Consider deprecation warnings for parameter renames to avoid breaking user code

---

**End of Implementation Plan**
