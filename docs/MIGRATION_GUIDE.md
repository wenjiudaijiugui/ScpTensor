# API Migration Guide

This guide documents API changes between ScpTensor versions and helps you migrate your code.

## Current Version: v0.2.0 (development)

---

## API Naming Changes (v0.2.0)

### Function Name Renaming

**IMPORTANT:** As of v0.2.0, all analysis functions have been renamed to follow a consistent prefix-based naming convention. Old function names have been **removed** and are no longer supported.

| Old Name | New Name | Module |
|----------|----------|--------|
| `log_normalize` | `norm_log` | Normalization |
| `zscore` | `norm_zscore` | Normalization |
| `sample_median_normalization` | `norm_median_sample` | Normalization |
| `sample_mean_normalization` | `norm_mean_sample` | Normalization |
| `global_median_normalization` | `norm_median_global` | Normalization |
| `median_scaling` | `norm_scale_median` | Normalization |
| `median_centering` | `norm_center_median` | Normalization |
| `upper_quartile_normalization` | `norm_quartile` | Normalization |
| `tmm_normalization` | `norm_tmm` | Normalization |
| `knn` | `impute_knn` | Imputation |
| `ppca` | `impute_ppca` | Imputation |
| `svd_impute` | `impute_svd` | Imputation |
| `missforest` | `impute_mf` | Imputation |
| `basic_qc` | `qc_basic` | Quality Control |
| `compute_quality_score` | `qc_score` | Quality Control |
| `detect_outliers` | `qc_detect_outliers` | Quality Control |
| `combat` | `integrate_combat` | Integration |
| `harmony` | `integrate_harmony` | Integration |
| `mnn_correct` | `integrate_mnn` | Integration |
| `scanorama_integrate` | `integrate_scanorama` | Integration |
| `nonlinear_integration` | `integrate_nonlinear` | Integration |
| `kmeans` | `cluster_kmeans` | Clustering |
| `leiden` | `cluster_leiden` | Clustering |
| `pca` | `reduce_pca` | Dimensionality Reduction |
| `umap` | `reduce_umap` | Dimensionality Reduction |

### Filtering Functions (Simplified)

| Old Name | New Name |
|----------|----------|
| `filter_features_by_missing_rate` | `filter_features_missing` |
| `filter_features_by_variance` | `filter_features_variance` |
| `filter_features_by_prevalence` | `filter_features_prevalence` |
| `filter_samples_by_total_count` | `filter_samples_count` |
| `filter_samples_by_missing_rate` | `filter_samples_missing` |
| `detect_contaminant_proteins` | `detect_contaminants` |

### Feature Selection Functions

| Old Name | New Name |
|----------|----------|
| `highly_variable_features` | `select_hvg` |
| `variance_stabilizing_transform` | `select_vst` |

### Migration Example

**Before (v0.1.x):**
```python
from scptensor import log_normalize, knn, basic_qc, combat

result = log_normalize(container, "proteins")
result = knn(container, "proteins")
result = basic_qc(container, "proteins")
result = combat(container, "proteins")
```

**After (v0.2.0):**
```python
from scptensor import norm_log, impute_knn, qc_basic, integrate_combat

result = norm_log(container, "proteins")
result = impute_knn(container, "proteins")
result = qc_basic(container, "proteins")
result = integrate_combat(container, "proteins")
```

### Auto-Migration Script for Function Names

```bash
# Normalization functions
sed -i 's/log_normalize(/norm_log(/g' your_script.py
sed -i 's/zscore(/norm_zscore(/g' your_script.py
sed -i 's/sample_median_normalization(/norm_median_sample(/g' your_script.py
sed -i 's/sample_mean_normalization(/norm_mean_sample(/g' your_script.py
sed -i 's/global_median_normalization(/norm_median_global(/g' your_script.py
sed -i 's/median_scaling(/norm_scale_median(/g' your_script.py
sed -i 's/median_centering(/norm_center_median(/g' your_script.py
sed -i 's/upper_quartile_normalization(/norm_quartile(/g' your_script.py
sed -i 's/tmm_normalization(/norm_tmm(/g' your_script.py

# Imputation functions
sed -i 's/\bknn(/impute_knn(/g' your_script.py
sed -i 's/ppca(/impute_ppca(/g' your_script.py
sed -i 's/svd_impute(/impute_svd(/g' your_script.py
sed -i 's/missforest(/impute_mf(/g' your_script.py

# Quality control functions
sed -i 's/basic_qc(/qc_basic(/g' your_script.py
sed -i 's/compute_quality_score(/qc_score(/g' your_script.py

# Integration functions
sed -i 's/combat(/integrate_combat(/g' your_script.py
sed -i 's/harmony(/integrate_harmony(/g' your_script.py
sed -i 's/mnn_correct(/integrate_mnn(/g' your_script.py
sed -i 's/scanorama_integrate(/integrate_scanorama(/g' your_script.py

# Clustering functions
sed -i 's/\bkmeans(/cluster_kmeans(/g' your_script.py
sed -i 's/leiden(/cluster_leiden(/g' your_script.py

# Dimensionality reduction functions
sed -i 's/\bpca(/reduce_pca(/g' your_script.py
sed -i 's/umap(/reduce_umap(/g' your_script.py
```

---

## Breaking Change: Old Parameter Names Removed

**⚠️ IMPORTANT:** As of v0.2.0, old parameter names have been **removed** and are no longer supported. You must update your code to use the new parameter names.

---

## Parameter Naming Changes

### Normalization Module

| Old Parameter | New Parameter | Status |
|--------------|---------------|--------|
| `base_layer` | `source_layer` | ❌ Removed - use `source_layer` |
| `base_layer_name` | `source_layer` | ❌ Removed - use `source_layer` |
| `new_layer` | `new_layer_name` | ❌ Removed - use `new_layer_name` |

**Affected Functions:**
- `log_normalize()`
- `zscore()`
- `sample_median_normalization()`
- `sample_mean_normalization()`
- `global_median_normalization()`
- `median_scaling()`
- `median_centering()`
- `upper_quartile_normalization()`
- `tmm_normalization()`

### Imputation Module

| Old Parameter | New Parameter | Status |
|--------------|---------------|--------|
| `base_layer` | `source_layer` | ❌ Removed - use `source_layer` |
| `layer` | `source_layer` | ❌ Removed - use `source_layer` |
| `output_layer` | `new_layer_name` | ❌ Removed - use `new_layer_name` |

**Affected Functions:**
- `knn()`
- `svd_impute()`
- `ppca()`
- `missforest()`

### Integration Module

The integration module uses `base_layer` consistently (no change planned):
- `combat()` - uses `base_layer`
- `mnn_correct()` - uses `base_layer`
- `harmony()` - uses `base_layer`
- `scanorama()` - uses `base_layer`
- `nonlinear_integration()` - uses `base_layer`

### Other Modules

**Differential Expression:**
- Uses `layer_name` for specifying the input layer

**Quality Control:**
- Uses `layer_name` for specifying the input layer

**Clustering:**
- Uses `base_layer` for specifying the input layer

**Dimensionality Reduction:**
- PCA: uses `base_layer_name`
- UMAP: uses `base_layer`

---

## Migration Examples

### Example 1: Log Normalization

**Before (v0.1.x - No longer supported):**
```python
from scptensor import log_normalize

result = log_normalize(
    container,
    "proteins",
    base_layer="raw",      # ❌ Old function name + parameter
    new_layer="log"        # ❌ Removed
)
```

**After (v0.2.0 - Current):**
```python
from scptensor import norm_log

result = norm_log(
    container,
    "proteins",
    source_layer="raw",
    new_layer_name="log"
)
```

### Example 2: KNN Imputation

**Before (v0.1.x - No longer supported):**
```python
from scptensor import knn

result = knn(
    container,
    "proteins",
    layer="log",              # ❌ Old function name + parameter
    output_layer="imputed"    # ❌ Removed
)
```

**After (v0.2.0 - Current):**
```python
from scptensor import impute_knn

result = impute_knn(
    container,
    "proteins",
    source_layer="log",
    new_layer_name="imputed"
)
```

### Example 3: Z-score Standardization

**Before (v0.1.x - No longer supported):**
```python
from scptensor import zscore

result = zscore(
    container,
    "proteins",
    base_layer_name="corrected",  # ❌ Old function name + parameter
    new_layer="zscore"            # ❌ Removed
)
```

**After (v0.2.0 - Current):**
```python
from scptensor import norm_zscore

result = norm_zscore(
    container,
    "proteins",
    source_layer="corrected",
    new_layer_name="zscore"
)
```

### Example 4: TMM Normalization

**Before (v0.1.x - No longer supported):**
```python
from scptensor import tmm_normalization

result = tmm_normalization(
    container,
    "proteins",
    base_layer_name="raw"  # ❌ Old function name + parameter
)
```

**After (v0.2.0 - Current):**
```python
from scptensor import norm_tmm

result = norm_tmm(
    container,
    "proteins",
    source_layer="raw"
)
```

---

## Parameter Naming Convention

### Standard Convention

As of v0.2.0, ScpTensor follows these naming conventions:

| Parameter Type | Name | Usage |
|----------------|------|-------|
| Input layer | `source_layer` | The layer to read data from |
| Output layer | `new_layer_name` | The name for the newly created layer |
| Input layer (legacy) | `base_layer` | Used in integration, clustering, dim_reduction |
| Layer reference | `layer_name` | Used in QC, diff_expr for layer reference |

### Module-wise Reference

| Module | Input Parameter | Output Parameter |
|--------|-----------------|------------------|
| Normalization | `source_layer` | `new_layer_name` |
| Imputation | `source_layer` | `new_layer_name` |
| Integration | `base_layer` | `new_layer_name` |
| QC | `layer_name` | N/A |
| Diff Expr | `layer_name` | N/A |
| Clustering | `base_layer` | N/A |
| Dim Reduction | `base_layer_name` / `base_layer` | Varies |

---

## Timeline

- **v0.1.0-beta:** Old parameters supported with deprecation warnings
- **v0.2.0 (Current):** Old parameters **removed**, only new parameter names supported
- **v1.0.0:** Future release

---

## Auto-Migration Script

For large codebases, you can use this sed command to automatically replace old parameter names:

```bash
# Replace base_layer with source_layer in normalization/imputation functions
sed -i 's/base_layer="raw"/source_layer="raw"/g' your_script.py
sed -i 's/base_layer="log"/source_layer="log"/g' your_script.py

# Replace new_layer with new_layer_name
sed -i 's/new_layer="/new_layer_name="/g' your_script.py

# Replace base_layer_name with source_layer
sed -i 's/base_layer_name=/source_layer=/g' your_script.py
```

**Note:** Always review changes after automated replacement, as `base_layer` is still valid in integration and clustering modules.

---

## Questions?

If you encounter issues during migration:
1. Check the function's docstring for current parameter names
2. Refer to the API Reference: `docs/design/API_REFERENCE.md`
3. Open an issue on GitHub with your migration question
