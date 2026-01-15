# ScpTensor API Audit

**Date:** 2026-01-15
**Version:** 1.0
**Status:** Draft

---

## Overview

This document audits the current ScpTensor API for inconsistencies and proposes standardized names. Use this as a reference when refactoring functions.

---

## 1. Assay Selection Parameters

| Module | Current Name | Proposed Name | Status |
|--------|-------------|---------------|--------|
| core/structures.py | `name` (in `get_assay`) | `assay_name` | 🔴 Inconsistent |
| normalization/* | `assay_name` | `assay_name` | ✅ Consistent |
| impute/* | `assay_name` | `assay_name` | ✅ Consistent |
| qc/basic.py | `assay_name` | `assay_name` | ✅ Consistent |
| integration/* | `assay_name` | `assay_name` | ✅ Consistent |

**Action Required:** Update `ScpContainer.get_assay(name)` to `get_assay(assay_name)` or create alias.

---

## 2. Layer Selection Parameters

| Module | Current Name | Proposed Name | Status |
|--------|-------------|---------------|--------|
| core/structures.py | `layer` | `layer_name` | 🔴 Inconsistent |
| normalization/* | `base_layer` | `source_layer` | 🟡 Different convention |
| normalization/* | `new_layer` | `new_layer_name` | 🔴 Missing _name suffix |
| impute/* | `layer` | `source_layer` | 🔴 Inconsistent |
| qc/basic.py | `layer_name` | `layer_name` | ✅ Consistent |

**Proposed Standard:**
- Input layer: `source_layer: str = "raw"`
- Output layer: `new_layer_name: str`
- General layer: `layer_name: str`

---

## 3. Sample/Feature Selection Parameters

| Module | Samples | Features | Status |
|--------|---------|----------|--------|
| core/structures.py | `obs_names` | `var_names` | ✅ Consistent |
| normalization/* | N/A | N/A | N/A |
| qc/* | `obs_subset` | N/A | 🔴 Inconsistent |
| filtering/* | `obs_names` | `var_names` | ✅ Consistent |

**Action Required:** Standardize to `obs_names` and `var_names` everywhere.

---

## 4. In-Place Operation Parameters

| Module | Current | Default | Status |
|--------|---------|---------|--------|
| core/structures.py | `inplace` | N/A | ✅ Consistent |
| normalization/* | N/A (always return new) | N/A | ✅ Consistent |
| qc/* | N/A (always return new) | N/A | ✅ Consistent |

**Standard:** Functions should return new objects by default (functional style). Use `inplace=False` only where modifying in-place makes sense.

---

## 5. Function Naming Audit

### 5.1 Transform Functions

| Current | Consistent? | Notes |
|---------|-------------|-------|
| `log_transform` | ✅ | Good |
| `normalize` | ✅ | Good |
| `scale` | ✅ | Good |
| `impute_knn` | ✅ | Includes method name |

### 5.2 Getter Functions

| Current | Consistent? | Proposed |
|---------|-------------|----------|
| `get_assay` | ✅ | - |
| `list_assays` | ✅ | (new) |
| `list_layers` | ✅ | (new) |

---

## 6. Default Value Consistency

| Parameter Type | Standard Default | Current Usage |
|----------------|------------------|---------------|
| source_layer | `"raw"` | Mixed: `"X"`, `"raw"`, `None` |
| new_layer_name | `"normalized"` | Mixed: varies by function |
| obs_names | `None` (all) | ✅ Consistent |
| var_names | `None` (all) | ✅ Consistent |
| threshold | `0.5` | Varies by context |
| seed | `None` | ✅ Consistent |
| verbose | `False` | ✅ Consistent |

---

## 7. Signature Patterns

### 7.1 Desired Pattern (Data Processing)

```python
def <verb>(
    container: ScpContainer,
    assay_name: str,
    source_layer: str = "raw",
    new_layer_name: str = "<output>",
    # Options
    <param>: <type> = <default>,
    # Advanced
    verbose: bool = False,
) -> ScpContainer:
    """<Summary>."""
```

### 7.2 Desired Pattern (Filtering)

```python
def <filter>_filter(
    container: ScpContainer,
    assay_name: str,
    layer_name: str = "raw",
    obs_names: list[str] | None = None,
    var_names: list[str] | None = None,
    # Filter-specific
    threshold: float = 0.5,
) -> ScpContainer:
    """<Summary>."""
```

---

## 8. Priority Fixes

### High Priority (P0)

1. **Standardize layer parameter names**
   - Rename `base_layer` → `source_layer`
   - Rename `layer` → `layer_name`
   - Rename `new_layer` → `new_layer_name`

2. **Fix default layer name inconsistency**
   - Standardize on `"raw"` as default source layer
   - Audit uses of `"X"` and document if intentionally different

### Medium Priority (P1)

3. **Add convenience methods to ScpContainer**
   ```python
   def list_assays(self) -> list[str]: ...
   def list_layers(self, assay_name: str) -> list[str]: ...
   def summary(self) -> str: ...
   ```

4. **Standardize error messages**
   - Add available options to KeyError messages
   - Include "Did you mean?" suggestions

### Low Priority (P2)

5. **Review function names for consistency**
   - Ensure all transformation functions use `_transform` suffix or consistent pattern
   - Ensure all filtering functions use `_filter` suffix

---

## 9. Refactoring Checklist

When updating a module to meet standards:

- [ ] All assay parameters use `assay_name`
- [ ] All layer parameters use `layer_name`, `source_layer`, or `new_layer_name`
- [ ] All selection parameters use `obs_names` and `var_names`
- [ ] Default source layer is `"raw"`
- [ ] New layer names use `_name` suffix
- [ ] Function signatures match desired pattern
- [ ] Docstrings follow NumPy style
- [ ] Type annotations are complete
- [ ] Error messages include context and suggestions
- [ ] Tests pass after changes

---

## 10. Tracking By Module

| Module | P0 Fixes | P1 Fixes | P2 Fixes | Status |
|--------|----------|----------|----------|--------|
| core/structures.py | layer → layer_name | Add list methods | Review names | 🔄 Pending |
| normalization/* | base_layer → source_layer | Error messages | - | 🔄 Pending |
| impute/* | layer → source_layer | Error messages | - | 🔄 Pending |
| qc/* | obs_subset → obs_names | Error messages | - | 🔄 Pending |
| integration/* | - | Error messages | - | ✅ Clean |
| dim_reduction/* | - | Error messages | - | ✅ Clean |
| cluster/* | - | Error messages | - | ✅ Clean |

---

**End of Audit**
