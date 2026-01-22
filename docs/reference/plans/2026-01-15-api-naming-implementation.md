# API Naming Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rename all public API functions to use consistent module prefixes (norm_*, impute_*, qc_*, integrate_*, cluster_*, reduce_*)

**Architecture:**
- Add new function names as primary exports
- Keep old names as deprecated aliases with warnings
- Update all internal references
- Update tests to use new names

**Tech Stack:** Python 3.12+, pytest, deprecation warnings

---

## Task 1: Normalization Module (norm_*)

**Files:**
- Modify: `scptensor/normalization/__init__.py`
- Modify: `scptensor/normalization/log.py`
- Modify: `scptensor/normalization/median_scaling.py`
- Modify: `scptensor/normalization/global_median.py`
- Modify: `scptensor/normalization/sample_mean.py`
- Modify: `scptensor/normalization/sample_median.py`
- Modify: `scptensor/normalization/upper_quartile.py`
- Modify: `scptensor/normalization/tmm.py`
- Modify: `scptensor/normalization/zscore.py`
- Modify: `scptensor/normalization/median_centering.py`
- Test: `tests/test_normalization.py`

**Step 1: Update log.py - rename function**

```python
# In scptensor/normalization/log.py, rename:
# log_normalize -> norm_log
# Add alias for backward compatibility

def norm_log(
    container: ScpContainer,
    assay_name: str,
    source_layer: str = "raw",
    new_layer_name: str = "log",
    base: float = 2.0,
) -> ScpContainer:
    """Apply logarithmic transformation to data."""
    # ... existing implementation ...

# Backward compatibility alias
def log_normalize(*args, **kwargs):
    import warnings
    warnings.warn(
        "'log_normalize' is deprecated, use 'norm_log' instead. "
        "This will be removed in version 1.0.0.",
        DeprecationWarning,
        stacklevel=2
    )
    return norm_log(*args, **kwargs)
```

**Step 2: Update __init__.py exports**

```python
# In scptensor/normalization/__init__.py

# New names (primary)
from scptensor.normalization.log import norm_log
from scptensor.normalization.median_scaling import norm_median_scale
from scptensor.normalization.global_median import norm_global_median
from scptensor.normalization.sample_mean import norm_sample_mean
from scptensor.normalization.sample_median import norm_sample_median
from scptensor.normalization.upper_quartile import norm_quartile
from scptensor.normalization.tmm import norm_tmm
from scptensor.normalization.zscore import norm_zscore
from scptensor.normalization.median_centering import norm_median_center

# Old names (deprecated, for backward compat)
from scptensor.normalization.log import log_normalize
# ... add other old names as aliases ...

__all__ = [
    # New names
    "norm_log",
    "norm_zscore",
    "norm_median_center",
    "norm_median_scale",
    "norm_sample_mean",
    "norm_sample_median",
    "norm_global_median",
    "norm_tmm",
    "norm_quartile",
    # Old names (deprecated)
    "log_normalize",
    "zscore",
    # ... etc
]
```

**Step 3: Run normalization tests**

```bash
pytest tests/test_normalization.py -v
```

Expected: All tests pass (they may still use old names, which should work via aliases)

**Step 4: Commit**

```bash
git add scptensor/normalization/
git commit -m "refactor(normalization): add norm_* prefixed function names"
```

---

## Task 2: Imputation Module (impute_*)

**Files:**
- Modify: `scptensor/impute/__init__.py`
- Modify: `scptensor/impute/knn.py`
- Modify: `scptensor/impute/ppca.py`
- Modify: `scptensor/impute/svd.py`
- Modify: `scptensor/impute/missforest.py`
- Test: `tests/test_impute.py`

**Step 1: Update knn.py - rename function**

```python
# In scptensor/impute/knn.py

def impute_knn(
    container: ScpContainer,
    assay_name: str,
    source_layer: str = "raw",
    new_layer_name: str = "imputed",
    k: int = 5,
    **kwargs,
) -> ScpContainer:
    """Impute missing values using k-nearest neighbors."""
    # ... existing implementation from old knn() function ...

# Backward compatibility alias
def knn(*args, **kwargs):
    import warnings
    warnings.warn(
        "'knn' is deprecated, use 'impute_knn' instead. "
        "This will be removed in version 1.0.0.",
        DeprecationWarning,
        stacklevel=2
    )
    return impute_knn(*args, **kwargs)
```

**Step 2: Update ppca.py - rename function**

```python
def impute_ppca(...):
    """Impute using probabilistic PCA."""
    # ... existing implementation ...

def ppca(*args, **kwargs):
    warnings.warn("'ppca' is deprecated, use 'impute_ppca' instead.", DeprecationWarning, stacklevel=2)
    return impute_ppca(*args, **kwargs)
```

**Step 3: Update svd.py - rename function**

```python
def impute_svd(...):
    """Impute using singular value decomposition."""
    # ... existing implementation ...

def svd_impute(*args, **kwargs):
    warnings.warn("'svd_impute' is deprecated, use 'impute_svd' instead.", DeprecationWarning, stacklevel=2)
    return impute_svd(*args, **kwargs)
```

**Step 4: Update missforest.py - rename and shorten**

```python
def impute_mf(...):
    """Impute using MissForest algorithm (mf = missforest)."""
    # ... existing implementation ...

def missforest(*args, **kwargs):
    warnings.warn("'missforest' is deprecated, use 'impute_mf' instead.", DeprecationWarning, stacklevel=2)
    return impute_mf(*args, **kwargs)
```

**Step 5: Update __init__.py**

```python
# New names
from scptensor.impute.knn import impute_knn
from scptensor.impute.ppca import impute_ppca
from scptensor.impute.svd import impute_svd
from scptensor.impute.missforest import impute_mf

# Old names (deprecated)
from scptensor.impute.knn import knn
from scptensor.impute.ppca import ppca
from scptensor.impute.svd import svd_impute
from scptensor.impute.missforest import missforest

__all__ = [
    "impute_knn", "impute_ppca", "impute_svd", "impute_mf",
    "knn", "ppca", "svd_impute", "missforest",  # deprecated
]
```

**Step 6: Run imputation tests**

```bash
pytest tests/test_impute.py -v
```

**Step 7: Commit**

```bash
git add scptensor/impute/
git commit -m "refactor(impute): add impute_* prefixed function names"
```

---

## Task 3: Quality Control Module (qc_*)

**Files:**
- Modify: `scptensor/qc/__init__.py`
- Modify: `scptensor/qc/basic.py`
- Modify: `scptensor/qc/advanced.py`
- Modify: `scptensor/qc/batch.py`
- Test: `tests/test_qc.py`

**Step 1: Update basic.py - rename functions**

```python
# In scptensor/qc/basic.py

def qc_basic(...):
    """Compute basic quality control metrics."""
    # ... existing implementation from basic_qc() ...

def basic_qc(*args, **kwargs):
    warnings.warn("'basic_qc' is deprecated, use 'qc_basic' instead.", DeprecationWarning, stacklevel=2)
    return qc_basic(*args, **kwargs)

def qc_score(...):
    """Compute quality score for each feature."""
    # ... existing from compute_quality_score() ...

def compute_quality_score(*args, **kwargs):
    warnings.warn("Deprecated, use 'qc_score' instead.", DeprecationWarning, stacklevel=2)
    return qc_score(*args, **kwargs)
```

**Step 2: Update advanced.py - rename and simplify filtering**

```python
def filter_features_missing(...):
    """Filter features with high missing rate."""
    # ... from filter_features_by_missing_rate ...

def filter_features_by_missing_rate(*args, **kwargs):
    warnings.warn("Use 'filter_features_missing'", DeprecationWarning, stacklevel=2)
    return filter_features_missing(*args, **kwargs)

# Same pattern for:
# filter_features_variance
# filter_features_prevalence
# filter_samples_count
# filter_samples_missing
```

**Step 3: Update batch.py**

```python
def qc_batch_metrics(...):
    """Compute batch-level QC metrics."""
    # ... from compute_batch_metrics ...

def compute_batch_metrics(*args, **kwargs):
    warnings.warn("Use 'qc_batch_metrics'", DeprecationWarning, stacklevel=2)
    return qc_batch_metrics(*args, **kwargs)
```

**Step 4: Update __init__.py**

```python
# New names
from scptensor.qc.basic import qc_basic, qc_score
from scptensor.qc.advanced import (
    filter_features_missing,
    filter_features_variance,
    filter_features_prevalence,
    filter_samples_count,
    filter_samples_missing,
    detect_contaminants,
)
from scptensor.qc.batch import qc_batch_metrics
# ... other imports

# Old names (deprecated)
from scptensor.qc.basic import basic_qc, compute_quality_score
# ... etc
```

**Step 5: Run QC tests**

```bash
pytest tests/test_qc.py -v
```

**Step 6: Commit**

```bash
git add scptensor/qc/
git commit -m "refactor(qc): add qc_* prefixed names and simplify filtering"
```

---

## Task 4: Integration Module (integrate_*)

**Files:**
- Modify: `scptensor/integration/__init__.py`
- Modify: `scptensor/integration/combat.py`
- Modify: `scptensor/integration/harmony.py`
- Modify: `scptensor/integration/mnn.py`
- Modify: `scptensor/integration/scanorama.py`
- Test: `tests/test_integration.py`

**Step 1: Update combat.py**

```python
def integrate_combat(...):
    """Apply ComBat batch correction."""
    # ... existing implementation ...

def combat(*args, **kwargs):
    warnings.warn("'combat' is deprecated, use 'integrate_combat'", DeprecationWarning, stacklevel=2)
    return integrate_combat(*args, **kwargs)
```

**Step 2: Update harmony.py, mnn.py, scanorama.py**

Same pattern: rename to `integrate_harmony`, `integrate_mnn`, `integrate_scanorama`

**Step 3: Run tests**

```bash
pytest tests/test_integration.py -v
```

**Step 4: Commit**

```bash
git add scptensor/integration/
git commit -m "refactor(integration): add integrate_* prefixed names"
```

---

## Task 5: Clustering Module (cluster_*)

**Files:**
- Modify: `scptensor/cluster/__init__.py`
- Modify: `scptensor/cluster/kmeans.py`
- Modify: `scptensor/cluster/leiden.py`
- Test: `tests/test_cluster.py`

**Step 1: Update kmeans.py**

```python
def cluster_kmeans(...):
    """Perform k-means clustering."""
    # ... existing implementation from kmeans() ...

# Keep run_kmeans as deprecated alias, remove standalone kmeans if duplicate
def kmeans(*args, **kwargs):
    warnings.warn("'kmeans' is deprecated, use 'cluster_kmeans'", DeprecationWarning, stacklevel=2)
    return cluster_kmeans(*args, **kwargs)

# If run_kmeans exists, make it an alias too
def run_kmeans(*args, **kwargs):
    warnings.warn("'run_kmeans' is deprecated, use 'cluster_kmeans'", DeprecationWarning, stacklevel=2)
    return cluster_kmeans(*args, **kwargs)
```

**Step 2: Update leiden.py**

```python
def cluster_leiden(...):
    """Perform Leiden clustering."""
    # ... existing implementation ...

def leiden(*args, **kwargs):
    warnings.warn("'leiden' is deprecated, use 'cluster_leiden'", DeprecationWarning, stacklevel=2)
    return cluster_leiden(*args, **kwargs)
```

**Step 3: Run tests**

```bash
pytest tests/test_cluster.py -v
```

**Step 4: Commit**

```bash
git add scptensor/cluster/
git commit -m "refactor(cluster): add cluster_* prefixed names"
```

---

## Task 6: Dimensionality Reduction Module (reduce_*)

**Files:**
- Modify: `scptensor/dim_reduction/__init__.py`
- Modify: `scptensor/dim_reduction/pca.py`
- Modify: `scptensor/dim_reduction/umap.py`
- Test: `tests/test_dim_reduction.py`

**Step 1: Update pca.py**

```python
def reduce_pca(...):
    """Perform principal component analysis."""
    # ... existing implementation ...

def pca(*args, **kwargs):
    warnings.warn("'pca' is deprecated, use 'reduce_pca'", DeprecationWarning, stacklevel=2)
    return reduce_pca(*args, **kwargs)
```

**Step 2: Update umap.py**

```python
def reduce_umap(...):
    """Perform UMAP dimensionality reduction."""
    # ... existing implementation ...

def umap(*args, **kwargs):
    warnings.warn("'umap' is deprecated, use 'reduce_umap'", DeprecationWarning, stacklevel=2)
    return reduce_umap(*args, **kwargs)
```

**Step 3: Run tests**

```bash
pytest tests/test_dim_reduction.py -v
```

**Step 4: Commit**

```bash
git add scptensor/dim_reduction/
git commit -m "refactor(dim_reduction): add reduce_* prefixed names"
```

---

## Task 7: Update Main Package Exports

**Files:**
- Modify: `scptensor/__init__.py`

**Step 1: Update scptensor/__init__.py**

Add new function names to main package exports:

```python
# Normalization (new names)
from scptensor.normalization import (
    norm_log,
    norm_zscore,
    norm_median_center,
    norm_median_scale,
    norm_sample_mean,
    norm_sample_median,
    norm_global_median,
    norm_tmm,
    norm_quartile,
    # Old names (deprecated)
    log_normalize,
    zscore,
    # ... etc
)

# Imputation (new names)
from scptensor.impute import (
    impute_knn,
    impute_ppca,
    impute_svd,
    impute_mf,
    # Old names
    knn,
    ppca,
    svd_impute,
    missforest,
)

# QC (new names)
from scptensor.qc import (
    qc_basic,
    qc_metrics,
    filter_features_missing,
    # ... etc
)

# Integration (new names)
from scptensor.integration import (
    integrate_combat,
    integrate_harmony,
    integrate_mnn,
    integrate_scanorama,
    # Old names
    combat,
    harmony,
    mnn_correct,
    scanorama_integrate,
)

# Clustering (new names)
from scptensor.cluster import (
    cluster_kmeans,
    cluster_leiden,
    # Old names
    kmeans,
    leiden,
)

# Dimensionality Reduction (new names)
from scptensor.dim_reduction import (
    reduce_pca,
    reduce_umap,
    # Old names
    pca,
    umap,
)
```

**Step 2: Run full test suite**

```bash
pytest tests/ -v --tb=short
```

**Step 3: Commit**

```bash
git add scptensor/__init__.py
git commit -m "feat: export new API names at package level"
```

---

## Task 8: Update Documentation

**Files:**
- Modify: `docs/MIGRATION_GUIDE.md`
- Modify: `docs/SCPTENSOR_CODING_STANDARDS.md`
- Create: `docs/API_REFERENCE.md`

**Step 1: Update MIGRATION_GUIDE.md**

Add new section for API naming changes:

```markdown
## API Naming Changes (v0.2.0)

### Function Name Renaming

| Old Name | New Name |
|----------|----------|
| log_normalize | norm_log |
| knn | impute_knn |
| basic_qc | qc_basic |
| combat | integrate_combat |
| kmeans | cluster_kmeans |
| pca | reduce_pca |
| umap | reduce_umap |
```

**Step 2: Update SCPTENSOR_CODING_STANDARDS.md**

Add naming convention section:

```markdown
## Function Naming Convention

All analysis functions use consistent prefixes:
- Normalization: norm_*
- Imputation: impute_*
- Quality Control: qc_*
- Integration: integrate_*
- Clustering: cluster_*
- Dimensionality Reduction: reduce_*
```

**Step 3: Commit**

```bash
git add docs/
git commit -m "docs: update guides for new API naming"
```

---

## Task 9: Update Tutorial Notebooks

**Files:**
- Modify: `docs/tutorials/tutorial_02_qc_normalization.ipynb`
- Modify: `docs/tutorials/tutorial_03_imputation_integration.ipynb`
- Modify: `docs/tutorials/tutorial_04_clustering_visualization.ipynb`

**Step 1: Update function names in notebooks**

Replace old function names with new ones.

**Step 2: Commit**

```bash
git add docs/tutorials/
git commit -m "docs: update tutorials to use new API names"
```

---

## Verification

After all tasks, run:

```bash
# Full test suite
pytest tests/ -v

# Check for any remaining old-name usage
grep -r "log_normalize" scptensor/ tests/ --include="*.py"
grep -r "def knn(" scptensor/ --include="*.py"
grep -r "def combat(" scptensor/ --include="*.py"
```

Expected: All uses are either deprecated aliases or updated to new names.

---

## Summary of Changes

| Module | Functions Renamed | Status |
|--------|------------------|--------|
| Normalization | 9 functions | norm_* prefix |
| Imputation | 4 functions | impute_* prefix |
| QC | 15+ functions | qc_* prefix, simplified filtering |
| Integration | 4 functions | integrate_* prefix |
| Clustering | 2 functions | cluster_* prefix |
| Dim Reduction | 2 functions | reduce_* prefix |
| **Total** | **36+ functions** | **All renamed** |
