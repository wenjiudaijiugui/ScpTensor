# Imputation Module Enhancement Design

**Project:** ScpTensor - Single-Cell Proteomics Analysis Framework
**Document Version:** 1.0
**Date:** 2026-01-16
**Status:** Design Phase

---

## Executive Summary

This document outlines the design for extending the ScpTensor imputation module with 5 new imputation methods commonly used in single-cell proteomics (SCP) research. The enhancement is based on comprehensive literature review of best practices in proteomics imputation (Jin 2021, Wei 2018, Wang 2020, BigOmics 2024).

**Goal:** Add 6 new imputation functions across 5 algorithm categories to provide comprehensive coverage of MAR and MNAR missing value scenarios.

**Key Changes:**
- 5 new implementation files (qrilc.py, minprob.py, lls.py, bpca.py, nmf.py)
- 6 new public API functions
- 5 new test files with validation against reference libraries
- 4 new visualization functions for imputation assessment
- Estimated effort: 10-12 person-days

---

## Background: Missing Values in SCP

### Missing Value Types

| Type | Description | Cause in SCP |
|------|-------------|--------------|
| **MCAR** | Missing Completely at Random | Random instrument fluctuations |
| **MAR** | Missing at Random | Technical/experimental factors |
| **MNAR** | Missing Not at Random | Low abundance (left-censored) |

**Key Finding:** In proteomics, MNAR is the predominant source due to the "zero gap" - undetected proteins are reported as NA rather than zero, creating a large gap between zero and minimum detected intensity.

### Industry Best Practices

According to multiple studies:

1. **Jin et al. 2021**: RF and LLS are best performing; BPCA outperforms SVD
2. **Wei et al. 2018**: RF best for MAR; QRILC best for left-censored MNAR
3. **Wang et al. 2020**: BPCA and KNN rank among top methods
4. **BigOmics 2024**: SVD provides best balance of accuracy and speed

---

## Current Implementation

### Existing Methods (v0.2.1)

```python
from scptensor.impute import impute_knn, impute_ppca, impute_svd, impute_mf
```

| API | Algorithm | Category | Best For |
|-----|-----------|----------|----------|
| `impute_knn` | K-Nearest Neighbors | Local similarity | MAR |
| `impute_ppca` | Probabilistic PCA | Matrix factorization | MAR/MNAR |
| `impute_svd` | Iterative SVD | Matrix factorization | MAR/MNAR |
| `impute_mf` | Random Forest (MissForest) | Machine learning | MAR |

---

## New Methods Design

### 1. QRILC (Quantile Regression Imputation of Left-Censored Data)

**File:** `scptensor/impute/qrilc.py`

**Algorithm:**
1. For each feature, estimate distribution using quantile regression
2. Sample from estimated left-censored distribution
3. Fill missing values with samples

**API:**
```python
def impute_qrilc(
    container: ScpContainer,
    assay_name: str,
    layer_name: str,
    new_layer_name: str = "qrilc",
    q: float = 0.01,  # Left-censored quantile
    random_state: int | None = None,
) -> ScpContainer:
    """Impute missing values using Quantile Regression Imputation of Left-Censored Data.

    Designed specifically for MNAR data where missingness is due to low abundance
    (left-censored). Uses quantile regression to estimate the distribution and
    samples from the left-censored distribution.

    Parameters
    ----------
    container : ScpContainer
        Input container with missing values
    assay_name : str
        Name of the assay to impute
    layer_name : str
        Name of the layer with missing values
    new_layer_name : str, default "qrilc"
        Name for the new imputed layer
    q : float, default 0.01
        Quantile for defining left-censoring threshold (0-1)
    random_state : int or None, default None
        Random seed for reproducibility

    Returns
    -------
    ScpContainer
        New container with imputed layer added

    Notes
    -----
    QRILC is recommended for MNAR missingness by multiple studies (Wei 2018).
    Preserves the tail distribution better than mean-based methods.

    References
    ----------
    .. [1] Wei R, et al. Sci Rep 2018;8:663.
    """
```

**Validation:** Compare with `impyute` custom implementation

---

### 2. MinProb / MinDet

**File:** `scptensor/impute/minprob.py`

**Algorithm:**
- **MinProb**: Sample from distribution around minimum detected value
- **MinDet**: Use deterministic minimum value

**API:**
```python
def impute_minprob(
    container: ScpContainer,
    assay_name: str,
    layer_name: str,
    new_layer_name: str = "minprob",
    sigma: float = 2,  # Width multiplier for minimum distribution
    random_state: int | None = None,
) -> StContainer:
    """Impute missing values using probabilistic minimum imputation.

    Samples from a distribution centered at the minimum detected value,
    scaled by sigma. Suitable for left-censored MNAR data.

    Parameters
    ----------
    sigma : float, default 2
        Standard deviation multiplier. Larger values create more spread.
        - sigma=1: narrow distribution near minimum
        - sigma=2: moderate spread (recommended)
        - sigma=3: wide spread
    """

def impute_mindet(
    container: ScpContainer,
    assay_name: str,
    layer_name: str,
    new_layer_name: str = "mindet",
    sigma: float = 1,
) -> ScpContainer:
    """Impute missing values using deterministic minimum imputation.

    Uses fixed value (min - sigma * spread) for all missing values.
    Faster than minprob but less accurate.
    """
```

**Validation:** Self-reference simple implementation

---

### 3. LLS (Local Least Squares)

**File:** `scptensor/impute/lls.py`

**Algorithm:**
1. For each sample with missing values, find K nearest neighbors
2. Build local linear regression using neighbor's complete features
3. Predict missing values

**API:**
```python
def impute_lls(
    container: ScpContainer,
    assay_name: str,
    layer_name: str,
    new_layer_name: str = "lls",
    k: int = 10,  # Number of neighbors
    max_iter: int = 100,
    tol: float = 1e-6,
) -> ScpContainer:
    """Impute missing values using Local Least Squares.

    Combines KNN with linear regression for improved accuracy in
    high-dimensional data with correlated features.

    Parameters
    ----------
    k : int, default 10
        Number of nearest neighbors to use
    max_iter : int, default 100
        Maximum iterations for convergence
    tol : float, default 1e-6
        Convergence threshold

    Notes
    -----
    LLS ranks among top methods in multiple benchmarks (Jin 2021).
    Particularly effective for high-dimensional proteomics data.

    References
    ----------
    .. [1] Kim H, et al. BMC Bioinformatics 2008;9:72.
    .. [2] Jin S, et al. Sci Rep 2021;11:16409.
    """
```

**Validation:** Compare with `fancyimpute.KNN` + regression

---

### 4. BPCA (Bayesian PCA)

**File:** `scptensor/impute/bpca.py`

**Algorithm:**
1. Use variational Bayesian inference for PCA model
2. Iteratively update latent variables and model parameters
3. Fill missing with reconstructed values

**API:**
```python
def impute_bpca(
    container: ScpContainer,
    assay_name: str,
    layer_name: str,
    new_layer_name: str = "bpca",
    n_components: int | None = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    random_state: int | None = None,
) -> ScpContainer:
    """Impute missing values using Bayesian PCA.

    Uses expectation-maximization with Bayesian inference for
    parameter estimation. Provides high accuracy at computational cost.

    Parameters
    ----------
    n_components : int or None, default None
        Number of principal components. If None, uses min(n_samples, n_features) // 2
    max_iter : int, default 100
        Maximum EM iterations
    tol : float, default 1e-6
        Convergence threshold for log-likelihood change
    random_state : int or None, default None
        Random seed for reproducibility

    Notes
    -----
    BPCA is among the most accurate methods (Wang 2020) but slower
    than alternatives. Recommended for smaller datasets or when
    accuracy is paramount.

    References
    ----------
    .. [1] Oba S, et al. Bioinformatics 2003;19(15):2088-2096.
    .. [2] Marttinen P, et al. BMC Bioinformatics 2017.
    """
```

**Validation:** Compare with `fancyimpute.BiScaler` / custom Bayesian implementation

---

### 5. NMF (Non-negative Matrix Factorization)

**File:** `scptensor/impute/nmf.py`

**Algorithm:**
1. Factorize matrix into W × H where W, H ≥ 0
2. Train using only non-missing elements
3. Reconstruct missing values

**API:**
```python
def impute_nmf(
    container: ScpContainer,
    assay_name: str,
    layer_name: str,
    new_layer_name: str = "nmf",
    n_components: int | None = None,
    max_iter: int = 200,
    tol: float = 1e-4,
    random_state: int | None = None,
) -> ScpContainer:
    """Impute missing values using Non-negative Matrix Factorization.

    Factorizes the matrix into W × H with non-negativity constraints.
    Particularly suitable for intensity data (protein abundance is always ≥ 0).

    Parameters
    ----------
    n_components : int or None, default None
        Number of latent factors. If None, uses heuristic based on data size
    max_iter : int, default 200
        Maximum iterations for coordinate descent
    tol : float, default 1e-4
        Stopping tolerance for reconstruction error
    random_state : int or None, default None
        Random seed for reproducibility

    Notes
    -----
    NMF optimization: min ||X - WH||², s.t. W ≥ 0, H ≥ 0
    Only non-missing elements contribute to the loss function.

    References
    ----------
    .. [1] Lee D, Seung H. Nature 2001;401:788-791.
    .. [2] Gaujoux R, Seoighe C. BMC Bioinformatics 2010;11:367.
    """
```

**Validation:** Compare with `sklearn.decomposition.NMF`

---

## Validation Testing Strategy

### Reference Library Comparison

Each new method must pass validation against a reference implementation:

| Method | Reference Library | Validation Function |
|--------|------------------|---------------------|
| QRILC | `impyute` custom | `test_qrilc_vs_reference()` |
| MinProb | Custom simple | `test_minprob_vs_reference()` |
| MinDet | Custom simple | `test_mindet_vs_reference()` |
| LLS | `fancyimpute.KNN` | `test_lls_vs_reference()` |
| BPCA | `fancyimpute.BiScaler` | `test_bpca_vs_reference()` |
| NMF | `sklearn.decomposition.NMF` | `test_nmf_vs_sklearn()` |

### Validation Test Template

```python
# tests/impute/test_validation_qrilc.py

def test_qrilc_vs_reference():
    """Validate QRILC against reference implementation."""
    # 1. Generate complete synthetic data
    X_complete = _generate_synthetic_proteomics_data(n_samples=100, n_features=50)

    # 2. Artificially introduce MNAR missingness
    X_missing, mask = _introduce_mnar_missingness(X_complete, missing_rate=0.2)

    # 3. Compute reference imputation
    X_ref = _qrilc_reference(X_missing, q=0.01)

    # 4. Compute ScpTensor imputation
    container = create_test_container_from_matrix(X_missing)
    result = impute_qrilc(container, "proteins", "data")
    X_scptensor = result.assays["proteins"].layers["qrilc"].X

    # 5. Assert: relative error < 5%
    relative_error = np.abs(X_scptensor - X_ref) / (np.abs(X_ref) + 1e-10)
    assert np.mean(relative_error[mask]) < 0.05, "QRILC differs from reference"

def test_qrilc_distribution_preservation():
    """Test that QRILC preserves data distribution."""
    # Compare mean, std, skewness before/after imputation
    # Assert distribution statistics are maintained

def test_qrilc_convergence():
    """Test that QRILC converges with reasonable iterations."""
```

### Test Dependencies

```toml
# pyproject.toml additions for dev dependencies
dev = [
    "fancyimpute>=0.5",  # For BPCA, KNN comparison
    "impyute>=0.0.8",    # For reference implementations
]
```

---

## Visualization Methods

### New File: `scptensor/viz/recipes/impute.py`

| Function | Purpose |
|----------|---------|
| `plot_imputation_comparison()` | Compare multiple methods side-by-side |
| `plot_imputation_scatter()` | Scatter plot: imputed vs true values |
| `plot_imputation_metrics()` | Bar chart of NRMSE, PCC metrics |
| `plot_missing_pattern()` | Heatmap of missing value patterns |

### API Design

```python
def plot_imputation_comparison(
    container: ScpContainer,
    assay_name: str,
    layer_name: str,
    methods: list[str] | None = None,
    metrics: list[str] = ["nrmse", "pcc"],
    figsize: tuple[float, float] = (12, 6),
) -> plt.Axes:
    """Compare multiple imputation methods using validation metrics.

    Parameters
    ----------
    container : ScpContainer
        Container with original (missing) data
    assay_name : str
        Assay name
    layer_name : str
        Layer with missing values
    methods : list of str or None
        Imputation methods to compare. If None, uses all available.
    metrics : list of str, default ["nrmse", "pcc"]
        Metrics to compute. Options: "nrmse", "pcc", "cosine"
    figsize : tuple, default (12, 6)
        Figure size

    Returns
    -------
    matplotlib.axes.Axes
        The plot axes

    Examples
    --------
    >>> plot_imputation_comparison(
    ...     container, "proteins", "data",
    ...     methods=["knn", "qrilc", "bpca", "nmf"]
    ... )
    """

def plot_imputation_scatter(
    container_true: ScpContainer,
    container_imputed: ScpContainer,
    assay_name: str,
    layer_name: str,
    layer_imputed: str,
    figsize: tuple[float, float] = (8, 8),
) -> plt.Axes:
    """Scatter plot comparing imputed values against true values.

    Red points indicate imputed values; gray points indicate observed values.

    Parameters
    ----------
    container_true : ScpContainer
        Container with true (complete) values
    container_imputed : ScpContainer
        Container with imputed values
    assay_name : str
        Assay name
    layer_name : str
        Layer name in true container
    layer_imputed : str
        Imputed layer name
    figsize : tuple, default (8, 8)
        Figure size

    Returns
    -------
    matplotlib.axes.Axes
        The plot axes
    """

def plot_imputation_metrics(
    metrics: dict[str, dict[str, float]],
    metric_names: list[str] = ["NRMSE", "PCC"],
    figsize: tuple[float, float] = (10, 6),
) -> plt.Axes:
    """Bar chart of imputation performance metrics.

    Parameters
    ----------
    metrics : dict
        Nested dict: {method: {metric: value}}
        Example: {"knn": {"nrmse": 0.2, "pcc": 0.95}, ...}
    metric_names : list of str
        Display names for metrics
    figsize : tuple, default (10, 6)
        Figure size

    Returns
    -------
    matplotlib.axes.Axes
        The plot axes
    """

def plot_missing_pattern(
    container: ScpContainer,
    assay_name: str,
    layer_name: str,
    max_features: int = 100,
    max_samples: int = 100,
    figsize: tuple[float, float] = (12, 8),
) -> plt.Axes:
    """Heatmap showing missing value patterns.

    Parameters
    ----------
    container : ScpContainer
        Input container
    assay_name : str
        Assay name
    layer_name : str
        Layer name
    max_features : int, default 100
        Maximum features to display (subsample if larger)
    max_samples : int, default 100
        Maximum samples to display (subsample if larger)
    figsize : tuple, default (12, 8)
        Figure size

    Returns
    -------
    matplotlib.axes.Axes
        The plot axes

    Notes
    -----
    Missing values shown in one color, observed in another.
    Helps identify systematic missingness patterns.
    """
```

---

## File Structure

```
scptensor/impute/
├── __init__.py          # Updated exports
├── knn.py               # Existing
├── missforest.py        # Existing
├── ppca.py              # Existing
├── svd.py               # Existing
├── qrilc.py             # NEW: QRILC imputation
├── minprob.py           # NEW: MinProb + MinDet
├── lls.py               # NEW: Local Least Squares
├── bpca.py              # NEW: Bayesian PCA
└── nmf.py               # NEW: Non-negative Matrix Factorization

scptensor/viz/recipes/
├── __init__.py          # Updated exports
├── impute.py            # NEW: Imputation visualizations
├── qc.py                # Existing
├── qc_advanced.py       # Existing
└── ...

tests/impute/
├── __init__.py
├── test_knn.py          # Existing
├── test_mf.py           # Existing
├── test_ppca.py         # Existing
├── test_svd.py          # Existing
├── test_qrilc.py        # NEW: QRILC tests
├── test_minprob.py      # NEW: MinProb/MinDet tests
├── test_lls.py          # NEW: LLS tests
├── test_bpca.py         # NEW: BPCA tests
├── test_nmf.py          # NEW: NMF tests
├── test_validation_qrilc.py    # NEW: QRILC validation
├── test_validation_minprob.py  # NEW: MinProb validation
├── test_validation_lls.py      # NEW: LLS validation
├── test_validation_bpca.py     # NEW: BPCA validation
└── test_validation_nmf.py      # NEW: NMF validation

tests/viz/
├── test_impute.py       # NEW: Imputation visualization tests
└── ...
```

---

## Updated Exports

### `scptensor/impute/__init__.py`

```python
"""Imputation methods for single-cell proteomics data."""

# Existing
from .knn import impute_knn
from .missforest import impute_mf
from .ppca import impute_ppca
from .svd import impute_svd

# New
from .qrilc import impute_qrilc
from .minprob import impute_minprob, impute_mindet
from .lls import impute_lls
from .bpca import impute_bpca
from .nmf import impute_nmf

__all__ = [
    # Existing
    "impute_knn",
    "impute_ppca",
    "impute_svd",
    "impute_mf",
    # New
    "impute_qrilc",
    "impute_minprob",
    "impute_mindet",
    "impute_lls",
    "impute_bpca",
    "impute_nmf",
]
```

### `scptensor/viz/recipes/__init__.py`

```python
# New imports
from .impute import (
    plot_imputation_comparison,
    plot_imputation_scatter,
    plot_imputation_metrics,
    plot_missing_pattern,
)

__all__ = [
    # ... existing exports ...
    # Imputation visualizations
    "plot_imputation_comparison",
    "plot_imputation_scatter",
    "plot_imputation_metrics",
    "plot_missing_pattern",
]
```

---

## Implementation Phases

### Phase 1: Left-Censored Methods (3 days)
- [ ] `qrilc.py` implementation + tests
- [ ] `minprob.py` implementation + tests
- [ ] Validation tests for both

### Phase 2: Local Similarity (2 days)
- [ ] `lls.py` implementation + tests
- [ ] Validation tests

### Phase 3: Bayesian & Matrix Factorization (3 days)
- [ ] `bpca.py` implementation + tests
- [ ] `nmf.py` implementation + tests
- [ ] Validation tests for both

### Phase 4: Visualization (2 days)
- [ ] `impute.py` visualization recipes
- [ ] Visualization tests
- [ ] Tutorial notebook update

### Phase 5: Documentation & Integration (1-2 days)
- [ ] Update API documentation
- [ ] Update ROADMAP.md
- [ ] Tutorial notebook for imputation methods
- [ ] Final integration tests

---

## Dependencies

### New Runtime Dependencies
None (all implementations use existing numpy/scipy/sklearn)

### New Dev Dependencies
```toml
dev = [
    # ... existing ...
    "fancyimpute>=0.5",  # For BPCA, KNN comparison in validation
    "impyute>=0.0.8",    # For reference implementations
]
```

---

## Success Criteria

1. **API Consistency**: All new functions follow existing `impute_*` naming pattern
2. **Mask Semantics**: All methods correctly update mask (IMPUTED=5)
3. **Validation**: Each method passes validation against reference (relative error < 5%)
4. **Test Coverage**: >90% coverage for all new modules
5. **Documentation**: Complete docstrings with examples
6. **Visualization**: At least 4 visualization functions for assessment

---

## References

1. Jin S, et al. "A comparative study of evaluating missing value imputation methods in label-free proteomics." *Sci Rep* 2021;11:16409.

2. Wei R, et al. "Missing Value Imputation Approach for Mass Spectrometry-based Metabolomics Data." *Sci Rep* 2018;8:663.

3. Wang S, et al. "NAguideR: performing and prioritizing missing value imputations for consistent bottom-up proteomic analyses." *Nucleic Acids Res* 2020;48(14):e83.

4. Oba S, et al. "A Bayesian missing value estimation method for gene expression profile data." *Bioinformatics* 2003;19(15):2088-2096.

5. Kim H, et al. "Missing value estimation for DNA microarray gene expression data: Local least squares imputation." *BMC Bioinformatics* 2008;9:72.

6. Lee D, Seung H. "Learning the parts of objects by non-negative matrix factorization." *Nature* 2001;401:788-791.

---

**End of Design Document**
