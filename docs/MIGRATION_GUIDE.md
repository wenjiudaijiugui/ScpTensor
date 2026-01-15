# API Migration Guide

This guide documents API changes between ScpTensor versions and helps you migrate your code.

## Current Version: v0.2.0 (development)

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
    base_layer="raw",      # ❌ Removed
    new_layer="log"        # ❌ Removed
)
```

**After (Current):**
```python
from scptensor import log_normalize

result = log_normalize(
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
    layer="log",              # ❌ Removed
    output_layer="imputed"    # ❌ Removed
)
```

**After (Current):**
```python
from scptensor import knn

result = knn(
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
    base_layer_name="corrected",  # ❌ Removed
    new_layer="zscore"            # ❌ Removed
)
```

**After (Current):**
```python
from scptensor import zscore

result = zscore(
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
    base_layer_name="raw"  # ❌ Removed
)
```

**After (Current):**
```python
from scptensor import tmm_normalization

result = tmm_normalization(
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
