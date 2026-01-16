# ScpTensor Coding Standards

**Version:** 1.0
**Date:** 2026-01-15
**Status:** Active

---

## Overview

This document defines coding standards for ScpTensor development. Follow these rules when writing or modifying code. Consistency improves maintainability and user experience.

---

## 1. Parameter Naming Conventions

### 1.1 Assay Parameters

Use `assay_name: str` for selecting a single assay.

```python
# Good
def normalize(container, assay_name: str, layer_name: str) -> ScpContainer:
    ...

# Bad
def normalize(container, assay: str, name: str) -> ScpContainer:
    ...
```

### 1.2 Layer Parameters

- Single layer: `layer_name: str`
- Multiple layers: `layer_names: list[str]`
- Source layer: `source_layer: str` or `base_layer: str`
- Destination layer: `new_layer_name: str`

```python
# Good
def log_transform(
    container: ScpContainer,
    assay_name: str,
    source_layer: str = "raw",
    new_layer_name: str = "log",
) -> ScpContainer:
    ...
```

### 1.3 Index Selection Parameters

- Samples: `obs_names: list[str] | None` (None = all)
- Features: `var_names: list[str] | None` (None = all)

```python
# Good
def filter_samples(container, obs_names: list[str] | None = None) -> ScpContainer:
    ...
```

### 1.4 Boolean Flags

Use verb form for boolean flags. Default to functional style (return new object).

```python
# Good
def process(container, inplace: bool = False) -> ScpContainer | None:
    if inplace:
        # Modify and return None
        return None
    return new_container
```

---

## 2. Function Signature Standards

### 2.1 Parameter Order

Group parameters logically:

1. **Required data parameters** (container, data arrays)
2. **Required selection parameters** (assay_name, layer_name)
3. **Optional data parameters** with defaults
4. **Option parameters** (thresholds, flags)
5. **Advanced parameters** (compression, parallel)

```python
def function(
    # Required data
    container: ScpContainer,
    assay_name: str,
    # Required selection
    layer_name: str = "raw",
    # Options
    threshold: float = 0.5,
    method: Literal["auto", "manual"] = "auto",
    # Advanced
    verbose: bool = False,
) -> ScpContainer:
    ...
```

### 2.2 Return Type Annotation

All public functions must declare return types explicitly.

```python
# Good
def calculate_mean(x: np.ndarray) -> float:
    ...

# Bad
def calculate_mean(x: np.ndarray):
    ...
```

---

## 3. Docstring Standards

Use NumPy style for all docstrings.

```python
def log_transform(
    container: ScpContainer,
    assay_name: str,
    source_layer: str = "raw",
    new_layer_name: str = "log",
    base: float = 2.0,
    offset: float = 1.0,
) -> ScpContainer:
    """Apply logarithmic transformation to data.

    Creates a new layer with log-transformed values using the formula:
    log_base(X + offset).

    Parameters
    ----------
    container : ScpContainer
        Input container with data to transform.
    assay_name : str
        Name of assay containing the source layer.
    source_layer : str, default "raw"
        Name of layer to transform.
    new_layer_name : str, default "log"
        Name for the output layer. Must not already exist.
    base : float, default 2.0
        Logarithm base. Use 2.0 for log2, 10.0 for log10, or math.e for ln.
    offset : float, default 1.0
        Value added before taking log to avoid log(0).

    Returns
    -------
    ScpContainer
        Container with new log-transformed layer. Original data unchanged.

    Raises
    ------
    KeyError
        If assay_name or source_layer does not exist.
    ValueError
        If new_layer_name already exists in the assay.

    Examples
    --------
    >>> container = log_transform(container, "proteins", "raw", "log")
    >>> container.assays["proteins"].layers.keys()
    dict_keys(['raw', 'log'])
    """
```

---

## 4. Type Annotation Standards

### 4.1 Use Union Pipe Syntax

Python 3.10+ supports `|` instead of `Union`.

```python
# Good (Python 3.10+)
def process(data: str | None) -> str:
    ...

# Avoid
def process(data: Union[str, None]) -> str:
    ...
```

### 4.2 Avoid Any

Use specific types or protocols instead of `Any`.

```python
# Good
def process(data: Sequence[float]) -> float:
    ...

# Bad
def process(data: Any) -> Any:
    ...
```

### 4.3 Type Checking Imports

Use `TYPE_CHECKING` for imports needed only in type hints.

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scptensor.core.structures import Assay

def process_assay(assay: Assay) -> None:
    ...
```

---

## 5. Code Formatting Standards

### 5.1 Ruff Configuration

- Line width: 100 characters
- Use double quotes for strings
- No trailing whitespace
- One blank line between function definitions

### 5.2 Import Order

```python
# 1. Standard library
import os
from pathlib import Path

# 2. Third-party
import numpy as np
import polars as pl

# 3. Local (scptensor)
from scptensor.core.exceptions import ScpTensorError
from scptensor.core.structures import Assay
```

### 5.3 Module Organization

```python
"""Module docstring."""

# 1. Imports
# 2. Constants (if any)
# 3. Type aliases (if any)
# 4. Helper functions (private)
# 5. Main functions/classes (public)
# 6. __all__ export list
```

---

## 6. Error Handling Standards

### 6.1 Use Defined Exception Types

```python
from scptensor.core.exceptions import (
    InvalidParameterError,
    MissingDataError,
)
```

### 6.2 Error Message Format

Error messages should include:
1. What went wrong
2. What was expected
3. How to fix it

```python
# Good
if assay_name not in container.assays:
    available = ", ".join(f"'{k}'" for k in container.assays.keys())
    raise KeyError(
        f"Assay '{assay_name}' not found. "
        f"Available assays: {available}. "
        f"Use container.list_assays() to see all available assays."
    )

# Bad
if assay_name not in container.assays:
    raise KeyError("Invalid assay")
```

### 6.3 Validation First

Validate inputs at the start of functions.

```python
def process(container, assay_name, layer_name):
    # Validate first
    if assay_name not in container.assays:
        raise KeyError(f"Assay '{assay_name}' not found.")
    if layer_name not in container.assays[assay_name].layers:
        raise KeyError(f"Layer '{layer_name}' not found in assay '{assay_name}'.")

    # Then process
    ...
```

---

## 7. Function Naming Conventions

### 7.1 General Purpose Patterns

| Purpose | Naming Pattern | Example |
|---------|---------------|---------|
| Transform data | `verb_object` | `log_transform`, `normalize` |
| Convert format | `to_<format>`, `as_<type>` | `to_pandas`, `as_dense` |
| Check condition | `is_<condition>` | `is_sparse`, `is_normalized` |
| Check property | `has_<property>` | `has_layer`, `has_mask` |
| Get value | `get_<property>` | `get_layer`, `get_shape` |
| List items | `list_<items>` | `list_assays`, `list_layers` |

### 7.2 Analysis Function Prefix Convention

All analysis functions use consistent prefixes by module category:

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
| Feature Selection | `select_*` | `select_{method}` |
| Differential Expression | `diff_*` | `diff_{test}` |

### 7.3 Examples by Category

**Normalization (`norm_*`):**
- `norm_log` - Logarithmic transformation
- `norm_zscore` - Z-score standardization
- `norm_median_sample` - Sample median normalization
- `norm_tmm` - TMM normalization

**Imputation (`impute_*`):**
- `impute_knn` - K-nearest neighbors imputation
- `impute_ppca` - Probabilistic PCA imputation
- `impute_svd` - SVD-based imputation
- `impute_mf` - MissForest imputation

**Quality Control (`qc_*`):**
- `qc_basic` - Basic quality control metrics
- `qc_score` - Compute quality scores
- `qc_detect_outliers` - Outlier detection

**Filtering (`filter_*`):**
- `filter_features_missing` - Filter features by missing rate
- `filter_features_variance` - Filter features by variance
- `filter_samples_count` - Filter samples by total count

**Integration (`integrate_*`):**
- `integrate_combat` - ComBat batch correction
- `integrate_harmony` - Harmony integration
- `integrate_mnn` - MNN correction
- `integrate_scanorama` - Scanorama integration

**Clustering (`cluster_*`):**
- `cluster_kmeans` - K-means clustering
- `cluster_leiden` - Leiden clustering

**Dimensionality Reduction (`reduce_*`):**
- `reduce_pca` - Principal Component Analysis
- `reduce_umap` - UMAP embedding

**Feature Selection (`select_*`):**
- `select_hvg` - Highly variable genes
- `select_vst` - Variance stabilizing transform

### 7.4 Benefits of Prefix Convention

1. **Predictability:** Users can guess function names by module category
2. **Discoverability:** IDE autocomplete groups related functions together
3. **Consistency:** All functions in a category follow the same pattern
4. **Namespacing:** Reduces naming conflicts (e.g., `pca()` vs `reduce_pca()`)

---

## 8. Testing Standards

### 8.1 Test File Organization

```python
# Tests should be organized by module
tests/
├── core/
│   ├── test_structures.py
│   └── test_matrix_ops.py
├── normalization/
│   └── test_normalize.py
└── test_io_export.py
```

### 8.2 Test Naming

- Test functions: `test_<function>_<scenario>`
- Test classes: `Test<Class>`

```python
def test_log_transform_with_zeros():
    """Test log transform with zero values."""

def test_log_transform_invalid_layer_raises():
    """Test log transform raises for invalid layer."""
```

---

## 9. Documentation Standards

### 9.1 Module Docstrings

Every module should have a docstring describing its purpose.

```python
"""Normalization methods for single-cell proteomics data.

This module provides functions to transform and normalize data for
downstream analysis. Methods include log transformation, scaling,
and variance stabilization.

Typical usage:
    >>> from scptensor import log_transform, quantile_normalize
    >>> container = log_transform(container, "proteins")
    >>> container = quantile_normalize(container, "proteins", "log")
"""
```

### 9.2 Comments

Write code that is self-documenting. Use comments sparingly.

```python
# Good: Code is clear
filtered = data[data > threshold]

# Bad: Comment states the obvious
# Filter data to keep values above threshold
filtered = data[data > threshold]

# Good: Comment explains why, not what
# Using log2 for consistency with SCP convention
transformed = np.log2(data + 1)
```

---

**End of Standards**
