# QC Module Enhancement Design Document

**Project:** ScpTensor - Single-Cell Proteomics Analysis Framework
**Version:** v0.2.1
**Date:** 2026-01-16
**Status:** Design Phase

---

## Executive Summary

This document describes the enhancement of the Quality Control (QC) module for ScpTensor, based on industry best practices from single-cell proteomics research (Vanderaa & Gatto 2023, scp R package, Nature Methods recommendations).

**Problem:** Current QC module lacks several key metrics recommended by the community:
- No sensitivity metrics (total/local feature detection)
- Limited missing value type analysis
- No coefficient of variation (CV) statistics
- Missing comprehensive QC visualizations

**Solution:** Add three new QC modules with corresponding visualization functions:
1. **Sensitivity metrics** - Total/local sensitivity, completeness, Jaccard index
2. **Missing value analysis** - Type distribution, pattern analysis, reporting
3. **Variability statistics** - CV computation, technical replicate CV, batch CV

---

## Background

### Industry Best Practices

Based on research from the single-cell proteomics community:

**Core QC Metrics (Vanderaa & Gatto 2023):**
1. **Total Sensitivity** - Total number of features found in the dataset
2. **Local Sensitivity** - Number of features per cell
3. **Data Completeness** - Proportion of values that are not missing
4. **Jaccard Index** - Feature consistency between cell pairs

**Advanced Metrics:**
5. **Cumulative Sensitivity Curve** - Assess if feature space is adequately sampled
6. **Missing Value Types** - Distinguish MBR, LOD, FILTERED, IMPUTED
7. **Coefficient of Variation** - Assess technical reproducibility

### Current ScpTensor QC Module

| File | Current Functions | Gap |
|------|-------------------|-----|
| `basic.py` | `qc_basic`, `qc_score`, feature variance/missing rate | No sensitivity metrics |
| `advanced.py` | Filtering, contaminant/doublet detection | Limited missing type analysis |
| `batch.py` | Batch metrics, effect detection | No CV statistics |
| `bivariate.py` | Correlation, similarity networks | No Jaccard index |
| `outlier.py` | Isolation Forest outlier detection | - |

---

## Module 1: Sensitivity Metrics

### File: `scptensor/qc/sensitivity.py`

#### Functions

**`qc_report_metrics(container, assay_name, layer_name, group_by=None)`**

Generate comprehensive QC metrics report. This is the main entry point for QC analysis.

```python
def qc_report_metrics(
    container: ScpContainer,
    assay_name: str = "protein",
    layer_name: str = "raw",
    group_by: str | None = None,
) -> ScpContainer:
    """Compute core QC metrics and add to container.

    Adds the following columns to obs:
    - n_detected_features: Number of features detected per sample
    - total_features: Total features in dataset
    - completeness: Data completeness proportion (0-1)
    - local_sensitivity: Local sensitivity (same as n_detected_features)

    If group_by is provided, also adds group-level statistics to obs.

    Returns
    -------
    ScpContainer
        New container with QC metrics added.
    """
```

**`compute_sensitivity(container, assay_name, layer_name)`**

Compute total and local sensitivity metrics.

**`compute_completeness(container, assay_name, layer_name)`**

Compute data completeness (proportion of non-missing values).

**`compute_jaccard_index(container, assay_name, layer_name)`**

Compute Jaccard index between all sample pairs.

```
Jaccard(A, B) = |A ∩ B| / |A ∪ B|
```

Where A and B are the sets of detected features for two samples.

**`compute_cumulative_sensitivity(container, assay_name, layer_name, batch_col=None, n_steps=20)`**

Compute cumulative sensitivity curve for assessing if feature space is adequately sampled.

#### Data Structures

```python
@dataclass
class QCMetrics:
    """Quality control metrics summary."""

    # Per-sample metrics
    n_features_per_sample: np.ndarray
    completeness_per_sample: np.ndarray

    # Global metrics
    total_features: int
    mean_sensitivity: float
    estimated_total_sensitivity: int | None = None

    # Group statistics (if provided)
    group_stats: dict[str, dict] | None = None
```

---

## Module 2: Missing Value Analysis

### File: `scptensor/qc/missing.py`

#### Functions

**`analyze_missing_types(container, assay_name, layer_name)`**

Analyze the distribution of missing value types using the mask matrix.

Mask codes (as defined in ScpTensor):
- `0`: VALID (detected value)
- `1`: MBR (missing between runs)
- `2`: LOD (below detection limit)
- `3`: FILTERED (QC removed)
- `5`: IMPUTED (filled value)

**`compute_missing_stats(container, assay_name, layer_name)`**

Compute comprehensive missing value statistics:
- Missing rate per feature
- Missing rate per sample
- Structural vs random missing

**`report_missing_values(container, assay_name, by)`**

Generate missing value report following scp R package format.

Returns DataFrame with:
- `LocalSensitivityMean`: Mean features per group
- `LocalSensitivitySd`: Standard deviation
- `TotalSensitivity`: Total features in group
- `Completeness`: Data completeness
- `NumberCells`: Number of cells

#### Data Structures

```python
@dataclass
class MissingValueReport:
    """Missing value analysis report."""

    # Overall statistics
    total_missing_rate: float
    valid_rate: float
    mbr_rate: float
    lod_rate: float
    imputed_rate: float

    # Feature-level
    feature_missing_rate: np.ndarray
    structural_missing_features: list

    # Sample-level
    sample_missing_rate: np.ndarray
    samples_with_high_missing: list
```

---

## Module 3: Variability Statistics (CV)

### File: `scptensor/qc/variability.py`

#### Functions

**`compute_cv(container, assay_name, layer_name, group_by=None)`**

Compute coefficient of variation for each feature.

```
CV = Standard Deviation / Mean
```

**`compute_technical_replicate_cv(container, assay_name, layer_name, replicate_col)`**

Compute CV within technical replicate groups to assess experimental reproducibility.

**`compute_batch_cv(container, assay_name, layer_name, batch_col)`**

Compute within-batch and between-batch CV for batch effect assessment.

**`filter_by_cv(container, assay_name, layer_name, cv_threshold)`**

Filter features with CV above threshold (typically CV < 0.3 for quality data).

#### Data Structures

```python
@dataclass
class CVReport:
    """Coefficient of variation report."""

    # Feature-level CV
    feature_cv: np.ndarray
    mean_cv: float
    median_cv: float

    # Grouped CV
    cv_by_group: dict[str, np.ndarray]

    # Batch CV (if batch info provided)
    within_batch_cv: dict[str, float] | None = None
    between_batch_cv: float | None = None

    # Quality assessment
    high_cv_features: list = None
    low_quality_samples: list = None
```

---

## Module 4: QC Visualizations

### File: `scptensor/viz/recipes/qc_advanced.py`

#### Sensitivity Visualizations

**`plot_sensitivity_summary(container, assay_name, layer_name, group_by=None)`**

Violin + scatter plot showing detected features per sample, grouped by metadata.

**`plot_cumulative_sensitivity(container, assay_name, layer_name, group_by=None)`**

Line plot showing cumulative feature count vs sample size. Used to assess if saturation is reached.

**`plot_jaccard_heatmap(container, assay_name, layer_name)`**

Heatmap showing Jaccard index (feature consistency) between all sample pairs.

#### Missing Value Visualizations

**`plot_missing_type_heatmap(container, assay_name, layer_name)`**

Heatmap showing mask matrix value distribution with color coding:
- Green (0): VALID
- Yellow (1): MBR
- Orange (2): LOD
- Red (3): FILTERED
- Purple (5): IMPUTED

**`plot_missing_summary(container, assay_name, layer_name)`**

4-panel summary:
1. Missing rate per sample
2. Missing rate per feature
3. Missing value type distribution (pie)
4. Missing value patterns (bar)

#### CV Visualizations

**`plot_cv_distribution(container, assay_name, layer_name, group_by=None)`**

Histogram showing CV distribution across all features with threshold line.

**`plot_cv_by_feature(container, assay_name, layer_name)`**

Scatter plot: mean intensity vs CV, highlighting high-CV features.

**`plot_cv_comparison(container, assay_name, layer_name, batch_col)`**

Bar chart comparing within-batch vs between-batch CV.

---

## File Structure

```
scptensor/
├── qc/
│   ├── __init__.py              # Updated exports
│   ├── basic.py                 # Existing: basic QC
│   ├── advanced.py              # Existing: advanced QC
│   ├── batch.py                 # Existing: batch effects
│   ├── bivariate.py             # Existing: bivariate analysis
│   ├── outlier.py               # Existing: outlier detection
│   ├── sensitivity.py           # NEW: sensitivity metrics
│   ├── missing.py               # NEW: missing value analysis
│   └── variability.py           # NEW: CV statistics
│
└── viz/
    └── recipes/
        ├── qc.py                # Existing: basic QC viz
        └── qc_advanced.py       # NEW: advanced QC viz
```

---

## API Naming

Following ScpTensor naming conventions (SCPTENSOR_CODING_STANDARDS.md):

| Function Category | Naming Pattern |
|-------------------|----------------|
| Main QC functions | `qc_*` |
| Computation functions | `compute_*` |
| Analysis functions | `analyze_*` |
| Report functions | `report_*` |
| Filtering functions | `filter_*` |
| Visualization functions | `plot_*` |

---

## Dependencies

**New dependencies:** None (all using existing stack)
- numpy
- polars
- scipy
- scikit-learn (for IsolationForest, already in use)
- matplotlib (existing)
- scienceplots (existing)

---

## Testing Plan

### Test Files

```
tests/qc/
├── test_sensitivity.py      # NEW: sensitivity metrics tests
├── test_missing.py          # NEW: missing value analysis tests
├── test_variability.py      # NEW: CV statistics tests
└── test_qc_advanced_viz.py  # NEW: visualization tests
```

### Test Coverage Goals

- Unit tests for each computation function: >90%
- Integration tests for end-to-end workflows
- Visualization tests: verify figure generation

---

## Implementation Phases

### Phase 1: Sensitivity Metrics (2-3 days)
- `scptensor/qc/sensitivity.py`
- Basic visualizations
- Tests

### Phase 2: Missing Value Analysis (2-3 days)
- `scptensor/qc/missing.py`
- Missing value visualizations
- Tests

### Phase 3: CV Statistics (1-2 days)
- `scptensor/qc/variability.py`
- CV visualizations
- Tests

### Phase 4: Documentation (1 day)
- Update API documentation
- Add tutorial notebook
- Update ROADMAP

**Total Estimated Effort:** 6-9 person-days

---

## References

1. Vanderaa, C., & Gatto, L. (2023). Revisiting the Thorny Issue of Missing Values in Single-Cell Proteomics. arXiv:2304.06654

2. Gatto, L., & Vanderaa, C. (2024). A framework for quality control in quantitative proteomics. Journal of Proteome Research.

3. Slavov Lab. Single-cell Proteomics Computational Analysis. https://scp.slavovlab.net/computational-analysis

4. scp R Package. https://uclouvain-cbio.github.io/scp/

5. Nature Methods. Initial recommendations for performing, benchmarking and reporting single-cell proteomics experiments. PMC10130941.

---

**End of Design Document**
