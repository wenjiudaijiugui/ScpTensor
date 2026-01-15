# ScpTensor API Naming Redesign

**Date:** 2026-01-15
**Version:** 1.0
**Status:** Draft

---

## Overview

Redesign ScpTensor API naming to achieve consistency across all modules. All functions within a module category use the same prefix, making the API easier to learn and use.

**Core Principle:** Consistency over brevity.

---

## Naming Convention

Each module category uses a consistent prefix:

| Category | Prefix | Pattern |
|----------|--------|---------|
| Normalization | `norm_*` | `norm_{method}` |
| Imputation | `impute_*` | `impute_{algorithm}` |
| Quality Control | `qc_*` | `qc_{type}` |
| Filtering | `filter_*` | `filter_{target}_{condition}` |
| Detection | `detect_*` | `detect_{target}` |
| Integration | `integrate_*` | `integrate_{algorithm}` |
| Clustering | `cluster_*` | `cluster_{algorithm}` |
| Dimensionality Reduction | `reduce_*` | `reduce_{method}` |
| Feature Selection | `select_*` | `select_{method}` (unchanged) |
| Differential Expression | `diff_*` | `diff_{test}` (unchanged) |

---

## Module Changes

### Normalization Module

| Old Name | New Name |
|----------|----------|
| `log_normalize` | `norm_log` |
| `zscore` | `norm_zscore` |
| `median_centering` | `norm_median_center` |
| `median_scaling` | `norm_median_scale` |
| `sample_mean_normalization` | `norm_sample_mean` |
| `sample_median_normalization` | `norm_sample_median` |
| `global_median_normalization` | `norm_global_median` |
| `tmm_normalization` | `norm_tmm` |
| `upper_quartile_normalization` | `norm_quartile` |

**Rationale:** Shorter names, consistent prefix. Removes redundant `_normalization` suffix.

---

### Imputation Module

| Old Name | New Name |
|----------|----------|
| `knn` | `impute_knn` |
| `ppca` | `impute_ppca` |
| `svd_impute` | `impute_svd` |
| `missforest` | `impute_mf` |

**Rationale:** Adds missing `impute_` prefix. Shortens `missforest` to `mf`.

---

### Quality Control Module

| Old Name | New Name |
|----------|----------|
| `basic_qc` | `qc_basic` |
| `compute_quality_score` | `qc_score` |
| `calculate_qc_metrics` | `qc_metrics` |
| `compute_batch_metrics` | `qc_batch_metrics` |
| `detect_batch_effects` | `detect_batch_effects` (unchanged) |
| `detect_contaminant_proteins` | `detect_contaminants` |
| `detect_outliers` | `detect_outliers` (unchanged) |
| `detect_doublets` | `detect_doublets` (unchanged) |

**Filtering Functions:**

| Old Name | New Name |
|----------|----------|
| `filter_features_by_missing_rate` | `filter_features_missing` |
| `filter_features_by_variance` | `filter_features_variance` |
| `filter_features_by_prevalence` | `filter_features_prevalence` |
| `filter_samples_by_total_count` | `filter_samples_count` |
| `filter_samples_by_missing_rate` | `filter_samples_missing` |

**Rationale:** Consistent `qc_*` prefix. Removes verbose `by_*` pattern in filtering.

---

### Integration Module

| Old Name | New Name |
|----------|----------|
| `combat` | `integrate_combat` |
| `harmony` | `integrate_harmony` |
| `mnn_correct` | `integrate_mnn` |
| `scanorama_integrate` | `integrate_scanorama` |

**Rationale:** Consistent `integrate_*` prefix across all methods.

---

### Clustering Module

| Old Name | New Name |
|----------|----------|
| `kmeans` | `cluster_kmeans` |
| `leiden` | `cluster_leiden` |
| `run_kmeans` | **DELETED** (duplicate of `kmeans`) |

**Rationale:** Consistent `cluster_*` prefix. Removes duplicate function.

---

### Dimensionality Reduction Module

| Old Name | New Name |
|----------|----------|
| `pca` | `reduce_pca` |
| `umap` | `reduce_umap` |

**Rationale:** Consistent `reduce_*` prefix.

---

### Feature Selection Module

No changes. Already uses consistent `select_*` prefix.

---

### Differential Expression Module

No changes. Already uses consistent `diff_expr_*` prefix.

---

## Usage Examples

### Before (Current API)

```python
from scptensor import (
    log_normalize,
    knn,
    basic_qc,
    combat,
    kmeans,
    pca
)

# Normalize data
result = log_normalize(container, "proteins", source_layer="raw")

# Impute missing values
result = knn(container, "proteins", source_layer="log")

# Quality control
result = basic_qc(container, "proteins")

# Batch correction
result = combat(container, "proteins")

# Clustering
result = kmeans(container, n_clusters=5)

# Dimensionality reduction
result = pca(container, n_components=10)
```

### After (New API)

```python
from scptensor import (
    norm_log,
    impute_knn,
    qc_basic,
    integrate_combat,
    cluster_kmeans,
    reduce_pca
)

# Normalize data
result = norm_log(container, "proteins", source_layer="raw")

# Impute missing values
result = impute_knn(container, "proteins", source_layer="log")

# Quality control
result = qc_basic(container, "proteins")

# Batch correction
result = integrate_combat(container, "proteins")

# Clustering
result = cluster_kmeans(container, n_clusters=5)

# Dimensionality reduction
result = reduce_pca(container, n_components=10)
```

---

## Benefits

1. **Predictability:** Users can guess function names by knowing the module prefix.
2. **Discoverability:** IDE autocomplete shows related functions grouped by prefix.
3. **Consistency:** All functions in a category follow the same pattern.
4. **Brevity:** Shorter names without losing meaning.

---

## Migration Strategy

### Phase 1: Add New Names
- Introduce new function names alongside old ones
- Old names remain functional but emit deprecation warnings

### Phase 2: Update Documentation
- Update all examples to use new names
- Update migration guide

### Phase 3: Remove Old Names
- Remove deprecated function names in v1.0.0

---

## Implementation Plan

1. Update function definitions in each module file
2. Update `__init__.py` exports
3. Add deprecation warnings for old names
4. Update all tests
5. Update documentation
6. Update tutorial notebooks

---

## Open Questions

1. Should `missforest` be shortened to `mf` or kept as `missforest`?
2. Should batch-related QC functions keep `batch` in the name?
3. Should filtering functions use `filter_*` or `qc_filter_*` prefix?

---

**End of Design Document**
