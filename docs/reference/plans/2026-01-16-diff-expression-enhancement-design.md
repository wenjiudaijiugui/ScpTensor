# Differential Expression Module Enhancement Design

**Date:** 2026-01-16
**Status:** Design
**Author:** ScpTensor Team
**Version:** 1.0

---

## Executive Summary

This document describes the enhancement of ScpTensor's differential expression module with five new statistical methods. The enhancement adds count-based models (VOOM, limma-trend, DESeq2-like) and non-parametric tests (Wilcoxon, Brunner-Munzel) to better handle single-cell proteomics data characteristics.

**Key Goals:**
- Support count-based data with mean-variance dependency
- Handle missing values (MNAR) natively
- Provide non-parametric alternatives for non-normal data
- Maintain zero external dependency addition

---

## Proposed Methods

### Count-Based Models

| Method | Description | Complexity |
|--------|-------------|------------|
| `diff_expr_voom` | VOOM transform + limma, suitable for small samples | O(n) |
| `diff_expr_limma_trend` | Empirical Bayes variance shrinkage with trend | O(n) |
| `diff_expr_deseq2` | Negative binomial GLM, DESeq2-like | O(n²) |

### Non-Parametric Methods

| Method | Description | Complexity |
|--------|-------------|------------|
| `diff_expr_wilcoxon` | Wilcoxon rank-sum test, paired/unpaired | O(n log n) |
| `diff_expr_brunner_munzel` | Location parameter test for heteroscedastic data | O(n log n) |

---

## API Design

All methods follow the existing API pattern and return `DiffExprResult`.

```python
def diff_expr_voom(
    container: ScpContainer,
    assay_name: str,
    layer: str,
    groupby: str,
    group1: str,
    group2: str,
    min_count: int = 10,
    normalize: str = "tmm",
) -> DiffExprResult:
    """VOOM transformation with limma analysis.

    Converts counts to log2-CPM with precision weights, then applies
    limma's empirical Bayes moderation. Designed for count-based data
    with mean-variance dependency.

    Parameters
    ----------
    container : ScpContainer
        Input data container.
    assay_name : str
        Assay containing the data.
    layer : str
        Layer with count data.
    groupby : str
        Column in obs defining groups.
    group1 : str
        First group name.
    group2 : str
        Second group name.
    min_count : int, default=10
        Minimum count threshold for feature filtering.
    normalize : str, default="tmm"
        Normalization method (tmm, upper_quartile, none).

    Returns
    -------
    DiffExprResult
        Test results with log2 fold changes and adjusted p-values.

    References
    ----------
    .. [1] Law et al. (2014) voom: precision weights unlock linear model
           analysis tools for RNA-seq. Genome Biology 15:R29.
    """

def diff_expr_limma_trend(
    container: ScpContainer,
    assay_name: str,
    layer: str,
    groupby: str,
    group1: str,
    group2: str,
    trend: bool = True,
    robust: bool = True,
) -> DiffExprResult:
    """limma-trend analysis for count data.

    Applies empirical Bayes variance shrinkage with trend correction
    for mean-variance dependency in count data.

    Parameters
    ----------
    container : ScpContainer
        Input data container.
    assay_name : str
        Assay containing the data.
    layer : str
        Layer with count data.
    groupby : str
        Column in obs defining groups.
    group1 : str
        First group name.
    group2 : str
        Second group name.
    trend : bool, default=True
        Apply trend correction.
    robust : bool, default=True
        Use robust empirical Bayes.

    Returns
    -------
    DiffExprResult
        Test results with moderated statistics.

    References
    ----------
    .. [1] Phipson et al. (2016) diffSpliceDGE and edgeR. ...
    .. [2] Ritchie et al. (2015) limma powers differential expression
           analyses for RNA-sequencing and microarray studies. Nucleic
           Acids Research 43:e47.
    """

def diff_expr_deseq2(
    container: ScpContainer,
    assay_name: str,
    layer: str,
    groupby: str,
    group1: str,
    group2: str,
    fit_type: str = "parametric",
    test: str = "wald",
) -> DiffExprResult:
    """DESeq2-like negative binomial model analysis.

    Models count data using negative binomial GLM, estimates dispersion,
    and tests for differential expression using Wald or LRT.

    Parameters
    ----------
    container : ScpContainer
        Input data container.
    assay_name : str
        Assay containing the data.
    layer : str
        Layer with count data.
    groupby : str
        Column in obs defining groups.
    group1 : str
        First group name.
    group2 : str
        Second group name.
    fit_type : str, default="parametric"
        Dispersion estimation method (parametric, local, mean).
    test : str, default="wald"
        Test type (wald, lrt).

    Returns
    -------
    DiffExprResult
        Test results with NB-based statistics.

    References
    ----------
    .. [1] Love et al. (2014) Moderated estimation of fold change and
           dispersion for RNA-seq data with DESeq2. Genome Biology
           15:550.
    """

def diff_expr_wilcoxon(
    container: ScpContainer,
    assay_name: str,
    layer: str,
    groupby: str,
    group1: str,
    group2: str,
    paired: bool = False,
    zero_method: str = "pratt",
) -> DiffExprResult:
    """Wilcoxon rank-sum test (paired or unpaired).

    Non-parametric test for comparing two groups. More robust than
    t-test for non-normal data.

    Parameters
    ----------
    container : ScpContainer
        Input data container.
    assay_name : str
        Assay containing the data.
    layer : str
        Layer with expression data.
    groupby : str
        Column in obs defining groups.
    group1 : str
        First group name.
    group2 : str
        Second group name.
    paired : bool, default=False
        Use paired test (requires matching pairs in obs).
    zero_method : str, default="pratt"
        Method for handling zero values (pratt, wilcox, zsplit).

    Returns
    -------
    DiffExprResult
        Test results with rank-based statistics.
    """

def diff_expr_brunner_munzel(
    container: ScpContainer,
    assay_name: str,
    layer: str,
    groupby: str,
    group1: str,
    group2: str,
) -> DiffExprResult:
    """Brunner-Munzel test for heteroscedastic data.

    Tests for difference in stochastic equality between groups.
    More robust than Wilcoxon when variances differ significantly.

    Parameters
    ----------
    container : ScpContainer
        Input data container.
    assay_name : str
        Assay containing the data.
    layer : str
        Layer with expression data.
    groupby : str
        Column in obs defining groups.
    group1 : str
        First group name.
    group2 : str
        Second group name.

    Returns
    -------
    DiffExprResult
        Test results with Brunner-Munzel statistics.

    References
    ----------
    .. [1] Brunner and Munzel (2000) The nonparametric Behrens-Fisher
           problem: Asymptotic theory and a small-sample approximation.
           Biometrical Journal 42:17-25.
    """
```

---

## Architecture

### Module Structure

```
scptensor/diff_expr/
├── __init__.py           # Public API exports
├── core.py               # Existing methods (t-test, ANOVA, etc.)
├── count_models.py       # NEW: VOOM, limma-trend, DESeq2
├── nonparametric.py      # NEW: Wilcoxon, Brunner-Munzel
└── _utils.py             # Shared utility functions
```

### Data Flow

```
ScpContainer
    │
    ├─> Extract Data
    │   └─> X, M, obs, var
    │
    ├─> Preprocess
    │   ├─> Filter low-count features
    │   ├─> Normalize (TMM/upper-quartile)
    │   └─> Handle missing values (M != 0)
    │
    ├─> Statistical Test
    │   ├─> VOOM: log2(CPM) → variance weights → limma
    │   ├─> limma-trend: log2(CPM) → trend → eBayes
    │   ├─> DESeq2: estimate dispersion → NB GLM → test
    │   ├─> Wilcoxon: rank → statistic
    │   └─> Brunner-Munzel: relative variance → statistic
    │
    └─> DiffExprResult
```

### Key Design Decisions

1. **Missing Values**: Exclude values where `M != 0` before testing
2. **Low Count Filter**: `min_count` requires feature to have threshold in N samples
3. **Normalization**: Built-in TMM, upper-quartile, or none
4. **Result Format**: Reuse existing `DiffExprResult` structure

---

## Implementation Details

### Dependencies

No new external dependencies. All methods use existing libraries:
- `numpy` for array operations
- `scipy.stats` for statistical functions
- `polars` for data manipulation

### Core Implementations

#### VOOM Transform

```python
def _voom_transform(
    counts: np.ndarray,
    lib_size: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply VOOM transformation.

    Returns log2-CPM and precision weights.
    """
    # Calculate log2-CPM
    cpm = counts / lib_size[:, None] * 1e6
    log2_cpm = np.log2(cpm + 0.5)

    # Estimate mean-variance trend
    mean_expr = np.mean(log2_cpm, axis=0)
    var_expr = np.var(log2_cpm, axis=0)

    # LOESS fit for trend
    trend = _lowess_fit(mean_expr, var_expr)
    weights = 1 / trend

    return log2_cpm, weights
```

#### Empirical Bayes (limma)

```python
def _limma_ebayes(
    logfc: np.ndarray,
    se: np.ndarray,
    df: int,
    robust: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply limma empirical Bayes moderation.

    Returns moderated logFC and standard errors.
    """
    # Estimate prior degrees of freedom
    # Shrink variance estimates
    # Compute moderated t-statistic
```

#### Negative Binomial Dispersion

```python
def _estimate_dispersion(
    counts: np.ndarray,
    design_matrix: np.ndarray,
) -> np.ndarray:
    """Estimate NB dispersion (DESeq2-style)."""
    # Gene-wise dispersion
    # Trended dispersion
    # MAP estimate
```

---

## Testing Strategy

### Test Files

```
tests/diff_expr/
├── test_count_models.py      # VOOM, limma-trend, DESeq2
├── test_nonparametric.py     # Wilcoxon, Brunner-Munzel
└── fixtures.py               # Shared test data
```

### Test Coverage

| Category | Tests | Goal |
|----------|-------|------|
| Unit tests | 15+ | Function correctness |
| Validation | 5+ | Compare with reference implementations |
| Integration | 3+ | End-to-end pipeline |
| Performance | 2+ | Large dataset benchmarks |
| **Total** | **25+** | **>85% coverage** |

### Key Test Cases

```python
# VOOM tests
test_voom_transform_shape()
test_voom_weights_monotonic()    # Weights decrease with expression
test_voom_edge_cases()           # Zero counts, single sample

# limma-trend tests
test_limma_trend_variance_shrinkage()
test_limma_trend_robust_mode()

# DESeq2 tests
test_deseq2_dispersion_range()   # Dispersion in valid range
test_deseq2_wald_vs_lrt()

# Wilcoxon tests
test_wilcoxon_paired_vs_unpaired()
test_wilcoxon_zero_method()

# Brunner-Munzel tests
test_brunner_munzel_heteroscedastic()
```

---

## Documentation Plan

### API Documentation

Complete NumPy-style docstrings for all functions with:
- Parameter descriptions
- Return value specifications
- Usage examples
- Scientific references

### Tutorial

Create `tutorial_09_differential_expression_advanced.ipynb`:

1. **Method Selection Guide**
   - When to use VOOM vs limma-trend vs DESeq2
   - Sample size and effect size considerations

2. **Count-Based Models**
   - Low-count data handling
   - Batch effects in differential expression

3. **Non-Parametric Methods**
   - Heteroscedastic data
   - Paired sample analysis

4. **Result Visualization**
   - Volcano plots, MA plots
   - P-value distributions

### API Reference

Update `docs/api/diff_expr.rst` with new method entries.

---

## Implementation Plan

### Phases

| Phase | Tasks | Duration |
|-------|-------|----------|
| **1. Infrastructure** | File structure, `_utils.py`, `__init__.py` | 1 day |
| **2. Count Models** | VOOM, limma-trend, DESeq2 | 4.5 days |
| **3. Non-Parametric** | Wilcoxon, Brunner-Munzel | 1.5 days |
| **4. Testing** | Unit, integration, validation tests | 3 days |
| **5. Documentation** | Docstrings, tutorial, API reference | 2 days |

**Total**: 12 days

### Acceptance Criteria

- [ ] All methods pass mypy type checking
- [ ] Test coverage >85%
- [ ] Tutorial notebook runs without errors
- [ ] API documentation complete
- [ ] Performance: 5000 features <30 seconds

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-16 | Initial design |

---

**Related Documents:**
- `docs/design/ARCHITECTURE.md` - Module architecture
- `docs/design/API_REFERENCE.md` - API conventions
- `docs/design/ROADMAP.md` - Project roadmap
