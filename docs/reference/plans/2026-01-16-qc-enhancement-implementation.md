# QC Module Enhancement - Implementation Plan

**Design Document:** `docs/design/2026-01-16-qc-enhancement-design.md`
**Date:** 2026-01-16
**Status:** Ready for Implementation

---

## Overview

This plan implements enhanced QC metrics for ScpTensor based on single-cell proteomics best practices. The implementation is divided into 4 phases with clear deliverables.

**Total Estimated Effort:** 6-9 person-days

---

## Phase 1: Sensitivity Metrics (2-3 days)

### Task 1.1: Create `scptensor/qc/sensitivity.py`

**Functions to implement:**

1. **`qc_report_metrics()`** - Main entry point
   - Input: `ScpContainer`, `assay_name`, `layer_name`, `group_by` (optional)
   - Output: New `ScpContainer` with QC metrics in `obs`
   - Metrics to add: `n_detected_features`, `total_features`, `completeness`, `local_sensitivity`

2. **`compute_sensitivity()`** - Sensitivity computation
   - Compute total and local sensitivity
   - Return `QCMetrics` dataclass

3. **`compute_completeness()`** - Data completeness
   - Calculate proportion of non-missing values per sample
   - Handle sparse matrices efficiently

4. **`compute_jaccard_index()`** - Jaccard similarity
   - Compute pairwise Jaccard index between samples
   - Return sparse matrix for memory efficiency

5. **`compute_cumulative_sensitivity()`** - CSC curve
   - Compute cumulative feature count vs sample size
   - Support group-wise computation
   - Estimate total sensitivity at saturation

**Dataclass to add:**
```python
@dataclass
class QCMetrics:
    n_features_per_sample: np.ndarray
    completeness_per_sample: np.ndarray
    total_features: int
    mean_sensitivity: float
    estimated_total_sensitivity: int | None = None
    group_stats: dict[str, dict] | None = None
```

**Acceptance Criteria:**
- All functions have complete type hints
- All functions have NumPy-style docstrings
- Functions follow ScpTensor functional pattern (return new objects)
- Sparse matrices handled efficiently
- Tests pass

### Task 1.2: Update `scptensor/qc/__init__.py`

Add exports for new sensitivity module:
```python
from scptensor.qc.sensitivity import (
    compute_completeness,
    compute_cumulative_sensitivity,
    compute_jaccard_index,
    compute_sensitivity,
    qc_report_metrics,
)
```

### Task 1.3: Create `tests/qc/test_sensitivity.py`

**Test cases:**
- Test `qc_report_metrics()` with synthetic data
- Test `compute_sensitivity()` with known values
- Test `compute_completeness()` edge cases (all missing, all valid)
- Test `compute_jaccard_index()` with identical samples
- Test `compute_cumulative_sensitivity()` monotonicity
- Test with sparse matrices
- Test with group_by parameter

**Target coverage:** >90%

### Task 1.4: Create basic visualizations

**File:** `scptensor/viz/recipes/qc_advanced.py`

**Functions:**
- `plot_sensitivity_summary()` - Violin + scatter plot
- `plot_cumulative_sensitivity()` - Line plot with saturation

**Acceptance:**
- SciencePlots style applied
- DPI = 300
- English labels only
- Group-by support

---

## Phase 2: Missing Value Analysis (2-3 days)

### Task 2.1: Create `scptensor/qc/missing.py`

**Functions to implement:**

1. **`analyze_missing_types()`** - Mask type analysis
   - Read mask matrix from layer
   - Count each mask code (0, 1, 2, 3, 5)
   - Add statistics to `var`

2. **`compute_missing_stats()`** - Missing statistics
   - Missing rate per feature
   - Missing rate per sample
   - Identify structural missing (always missing)
   - Identify high-missing samples

3. **`report_missing_values()`** - scp-style report
   - Input: `container`, `assay_name`, `by` (group column)
   - Output: DataFrame with summary statistics per group
   - Columns: LocalSensitivityMean, LocalSensitivitySd, TotalSensitivity, Completeness, NumberCells

**Dataclass to add:**
```python
@dataclass
class MissingValueReport:
    total_missing_rate: float
    valid_rate: float
    mbr_rate: float
    lod_rate: float
    imputed_rate: float
    feature_missing_rate: np.ndarray
    structural_missing_features: list
    sample_missing_rate: np.ndarray
    samples_with_high_missing: list
```

### Task 2.2: Update `scptensor/qc/__init__.py`

Add exports for missing module:
```python
from scptensor.qc.missing import (
    analyze_missing_types,
    compute_missing_stats,
    report_missing_values,
)
```

### Task 2.3: Create `tests/qc/test_missing.py`

**Test cases:**
- Test `analyze_missing_types()` with synthetic masks
- Test `compute_missing_stats()` edge cases
- Test `report_missing_values()` grouping
- Test with various mask combinations
- Test with sparse matrices

### Task 2.4: Create missing value visualizations

**File:** `scptensor/viz/recipes/qc_advanced.py` (extend)

**Functions to add:**
- `plot_missing_type_heatmap()` - Color-coded mask heatmap
- `plot_missing_summary()` - 4-panel summary

---

## Phase 3: CV Statistics (1-2 days)

### Task 3.1: Create `scptensor/qc/variability.py`

**Functions to implement:**

1. **`compute_cv()`** - Coefficient of variation
   - Compute CV per feature: SD/Mean
   - Handle zeros/low values (add minimum threshold)
   - Support group-wise computation

2. **`compute_technical_replicate_cv()`** - Technical replicate CV
   - Input: `replicate_col` specifying replicate groups
   - Compute CV within each group
   - Return CV per feature averaged across replicates

3. **`compute_batch_cv()`** - Batch CV analysis
   - Input: `batch_col` specifying batches
   - Compute within-batch CV
   - Compute between-batch CV
   - Return both metrics

4. **`filter_by_cv()`** - CV-based filtering
   - Remove features with CV > threshold
   - Return new filtered `ScpContainer`

**Dataclass to add:**
```python
@dataclass
class CVReport:
    feature_cv: np.ndarray
    mean_cv: float
    median_cv: float
    cv_by_group: dict[str, np.ndarray]
    within_batch_cv: dict[str, float] | None = None
    between_batch_cv: float | None = None
    high_cv_features: list = None
    low_quality_samples: list = None
```

### Task 3.2: Update `scptensor/qc/__init__.py`

Add exports for variability module:
```python
from scptensor.qc.variability import (
    compute_batch_cv,
    compute_cv,
    compute_technical_replicate_cv,
    filter_by_cv,
)
```

### Task 3.3: Create `tests/qc/test_variability.py`

**Test cases:**
- Test `compute_cv()` with known values
- Test edge cases (all zeros, constant values)
- Test `compute_technical_replicate_cv()` grouping
- Test `compute_batch_cv()` with synthetic batches
- Test `filter_by_cv()` threshold behavior

### Task 3.4: Create CV visualizations

**File:** `scptensor/viz/recipes/qc_advanced.py` (extend)

**Functions to add:**
- `plot_cv_distribution()` - Histogram with threshold
- `plot_cv_by_feature()` - Mean vs CV scatter
- `plot_cv_comparison()` - Within/between batch CV

---

## Phase 4: Documentation & Integration (1 day)

### Task 4.1: Update `scptensor/viz/recipes/__init__.py`

Add exports for new QC visualizations:
```python
from .qc_advanced import (
    plot_cv_comparison,
    plot_cv_by_feature,
    plot_cv_distribution,
    plot_cumulative_sensitivity,
    plot_jaccard_heatmap,
    plot_missing_summary,
    plot_missing_type_heatmap,
    plot_sensitivity_summary,
)
```

### Task 4.2: Update API documentation

**File:** `docs/API_REFERENCE.md`

Add new API entries for:
- All new `qc_*`, `compute_*`, `analyze_*`, `report_*` functions
- All new `plot_*` functions

### Task 4.3: Create tutorial notebook

**File:** `docs/tutorials/tutorial_06_advanced_qc.ipynb`

**Outline:**
1. Introduction to advanced QC metrics
2. Sensitivity analysis
3. Missing value analysis
4. CV statistics
5. Combined QC workflow
6. Visualization examples

### Task 4.4: Update ROADMAP

**File:** `docs/design/ROADMAP.md`

Add v0.2.1 milestone with:
- Completed tasks
- Test coverage metrics
- Documentation links

---

## Implementation Checklist

### Phase 1: Sensitivity Metrics
- [ ] Create `scptensor/qc/sensitivity.py`
- [ ] Implement `qc_report_metrics()`
- [ ] Implement `compute_sensitivity()`
- [ ] Implement `compute_completeness()`
- [ ] Implement `compute_jaccard_index()`
- [ ] Implement `compute_cumulative_sensitivity()`
- [ ] Add `QCMetrics` dataclass
- [ ] Update `scptensor/qc/__init__.py`
- [ ] Create `tests/qc/test_sensitivity.py`
- [ ] Create basic visualizations in `qc_advanced.py`
- [ ] Run tests and verify >90% coverage

### Phase 2: Missing Value Analysis
- [ ] Create `scptensor/qc/missing.py`
- [ ] Implement `analyze_missing_types()`
- [ ] Implement `compute_missing_stats()`
- [ ] Implement `report_missing_values()`
- [ ] Add `MissingValueReport` dataclass
- [ ] Update `scptensor/qc/__init__.py`
- [ ] Create `tests/qc/test_missing.py`
- [ ] Create missing value visualizations
- [ ] Run tests and verify coverage

### Phase 3: CV Statistics
- [ ] Create `scptensor/qc/variability.py`
- [ ] Implement `compute_cv()`
- [ ] Implement `compute_technical_replicate_cv()`
- [ ] Implement `compute_batch_cv()`
- [ ] Implement `filter_by_cv()`
- [ ] Add `CVReport` dataclass
- [ ] Update `scptensor/qc/__init__.py`
- [ ] Create `tests/qc/test_variability.py`
- [ ] Create CV visualizations
- [ ] Run tests and verify coverage

### Phase 4: Documentation & Integration
- [ ] Update `scptensor/viz/recipes/__init__.py`
- [ ] Update `docs/API_REFERENCE.md`
- [ ] Create `docs/tutorials/tutorial_06_advanced_qc.ipynb`
- [ ] Update `docs/design/ROADMAP.md`
- [ ] Run full test suite
- [ ] Run pre-commit hooks

---

## Testing Strategy

### Unit Tests
- Each function tested with synthetic data
- Edge cases covered (empty data, all missing, all valid)
- Sparse matrix handling verified

### Integration Tests
- End-to-end workflow: container → QC analysis → visualization
- Compatibility with existing modules tested
- Provenance tracking verified

### Visualization Tests
- Figure generation verified (no exceptions)
- Style guidelines checked (SciencePlots, DPI=300)

### Coverage Goals
- New modules: >90% coverage
- Overall project: maintain >65%

---

## Dependencies

**No new dependencies required.** All functionality uses existing stack:
- numpy, polars - data structures
- scipy - sparse matrices, statistics
- scikit-learn - existing dependencies
- matplotlib, scienceplots - visualization

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Sparse matrix performance | Use efficient operations, test with large data |
| Memory usage (Jaccard) | Return sparse matrix, offer chunked option |
| CV edge cases (zero mean) | Add minimum threshold, clear error messages |
| Visualization complexity | Build on existing base functions |

---

## Success Criteria

1. All new functions implemented with type hints and docstrings
2. Test coverage >90% for new modules
3. All existing tests still pass
4. Tutorial notebook runs without errors
5. API documentation updated
6. Code passes pre-commit hooks (ruff, mypy)

---

## References

- Design Document: `docs/design/2026-01-16-qc-enhancement-design.md`
- ScpTensor Coding Standards: `docs/SCPTENSOR_CODING_STANDARDS.md`
- API Naming: Use `qc_*`, `compute_*`, `analyze_*`, `report_*`, `plot_*` prefixes

---

**End of Implementation Plan**
