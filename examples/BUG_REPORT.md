# Bug Hunting Report for ScpTensor QC Module

**Date:** 2026-02-28
**Test Script:** `examples/test_qc_bug_hunting.py`
**Total Tests:** 25
**Passed:** 25
**Bugs Found:** 0 (all fixed)

---

## Summary

The bug hunting script systematically tested edge cases in the QC module and related visualization functions. All 25 tests passed successfully. Previously identified bugs have been fixed:

- **Bug #2 (Fixed):** Log transform now clips negative values to 0 with a warning, preventing Inf values.
- **Bug #1 & #3 (Fixed):** `qc_completeness` now handles None mask matrix correctly.

---

## Bug #1: qc_completeness fails when mask matrix is None

**Severity:** Medium
**Location:** `/home/shenshang/projects/ScpTensor/scptensor/viz/recipes/qc.py`, line 85
**Status:** FIXED

### Description
The `qc_completeness` function assumes `matrix.M` (mask matrix) is always not None and tries to perform operations on it directly. When a matrix is created without a mask (e.g., from sparse data or simple initialization), `matrix.M` is None, causing an `AxisError`.

### Reproduction Steps
```python
import numpy as np
import polars as pl
from scptensor import ScpContainer, Assay, ScpMatrix, qc_completeness

obs = pl.DataFrame({'_index': ['S1', 'S2']})
var = pl.DataFrame({'_index': ['P1', 'P2']})
assay = Assay(var=var, layers={'raw': ScpMatrix(X=np.random.rand(2, 2))})
container = ScpContainer(obs=obs, assays={'proteins': assay})

# This will fail with: AxisError: axis 1 is out of bounds for array of dimension 0
ax = qc_completeness(container, assay_name='proteins', layer='raw')
```

### Error Message
```
numpy.exceptions.AxisError: axis 1 is out of bounds for array of dimension 0
```

### Root Cause
In `scptensor/viz/recipes/qc.py:85`:
```python
valid_counts = np.sum(matrix.M == 0, axis=1)
```
This line assumes `matrix.M` is a valid numpy array, but it can be None.

### Fix Applied
Updated `qc_completeness` to handle None mask matrix using the same pattern as `qc_matrix_spy`:
```python
M = matrix.M if matrix.M is not None else np.zeros_like(matrix.X, dtype=np.int8)
valid_counts = np.sum(M == 0, axis=1)
```

### Impact
- Affects all visualizations that use `qc_completeness` when:
  - Data is loaded without masks
  - Sparse matrices are used
  - Data is initialized manually

---

## Bug #2: Log transform produces Inf values with negative input

**Severity:** Low (data validation issue)
**Location:** `/home/shenshang/projects/ScpTensor/scptensor/normalization/log_transform.py`, line 151
**Status:** FIXED

### Description
When the input data contains negative values, the log transform can produce `-inf` values. The default offset of 1.0 means that values of -1.0 become 0.0 after adding the offset, and `log(0) = -inf`.

### Reproduction Steps
```python
import numpy as np
import polars as pl
from scptensor import ScpContainer, Assay, ScpMatrix, log_transform

obs = pl.DataFrame({'_index': ['S1']})
var = pl.DataFrame({'_index': ['P1']})

X = np.array([[-1.0, 0.0, 1.0]])  # Contains negative value
assay = Assay(var=var, layers={'raw': ScpMatrix(X=X)})
container = ScpContainer(obs=obs, assays={'proteins': assay})

container = log_transform(
    container,
    assay_name='proteins',
    source_layer='raw',
    new_layer_name='log',
)

# Result now contains finite values (negatives clipped to 0)
log_X = container.assays['proteins'].layers['log'].X
# [0.0, 0.0, ~0.69] - all finite
```

### Warning Message (after fix)
```
UserWarning: Input contains negative values. These will be clipped to 0 before log transform.
```

### Root Cause
In `scptensor/normalization/log_transform.py:151`:
```python
X_log = np.log(X + offset) / log_scale
```
When `X` contains values <= `-offset`, the result is `log(<=0)` which is `-inf` or `nan`.

### Fix Applied
Added input validation to check for negative values and clip them to 0 with a warning:
```python
if np.any(X < 0):
    warnings.warn(
        "Input contains negative values. These will be clipped to 0 before log transform.",
        UserWarning
    )
    X = np.maximum(X, 0)
```

### Impact
- Low impact for typical proteomics data (should not have negative values)
- After fix: No Inf values produced, all outputs are finite
- Warning is issued to alert users of the clipping

---

## Bug #3: Sparse matrix with NaN values causes qc_completeness to fail

**Severity:** Medium
**Location:** Same as Bug #1 (`scptensor/viz/recipes/qc.py`, line 85)
**Status:** FIXED (same fix as Bug #1)

### Description
When a sparse matrix contains NaN values (which should not happen in well-formed sparse matrices, but can occur due to data corruption or incorrect construction), and the mask matrix is None, `qc_completeness` fails with the same error as Bug #1.

### Reproduction Steps
```python
import numpy as np
import polars as pl
from scipy import sparse
from scptensor import ScpContainer, Assay, ScpMatrix, qc_completeness

X_dense = np.random.rand(10, 20)
X_dense[5, 10] = np.nan
X = sparse.csr_matrix(X_dense)

obs = pl.DataFrame({'_index': [f'S{i}' for i in range(10)]})
var = pl.DataFrame({'_index': [f'P{i}' for i in range(20)]})

assay = Assay(var=var, layers={'raw': ScpMatrix(X=X)})
container = ScpContainer(obs=obs, assays={'proteins': assay})

# Fails with AxisError
ax = qc_completeness(container, assay_name='proteins', layer='raw')
```

### Root Cause
Same as Bug #1 - the mask matrix is None by default for sparse matrices.

### Note
Sparse matrices should not contain NaN values by design. If this occurs, it indicates:
1. Data corruption
2. Incorrect sparse matrix construction
3. NaN values were present before sparse conversion

This bug was fixed by the same change as Bug #1 - the function now handles None mask matrix correctly.

---

## Tests That Passed

The following edge cases were tested and passed:

1. **Single sample handling** - All visualization functions handle single sample correctly
2. **Missing group columns** - `qc_completeness` defaults to "All" group when group_by column doesn't exist
3. **All NaN data** - `qc_matrix_spy` handles all-NaN matrices
4. **All masked data** - `qc_matrix_spy` handles fully masked matrices
5. **Empty violin data** - Correctly raises ValueError
6. **Single point violin** - Handles single data point
7. **Empty scatter coordinates** - Handles empty coordinate arrays
8. **Mixed mask codes** - Correctly renders all mask code types
9. **Type conversion** - X matrix properly converted to float64
10. **Mask dtype** - Mask matrix properly stored as int8
11. **Sparse dtype consistency** - Sparse matrices maintain correct dtypes
12. **Shape mismatch detection** - Correctly validates mask/X shape consistency
13. **Invalid mask code detection** - Catches and reports invalid mask codes
14. **None mask handling** - Properly treats None mask as all-valid
15. **Mask preservation** - Log transform preserves mask matrix in new layers
16. **Non-ASCII characters** - Handles non-ASCII characters in sample/feature names
17. **Negative/zero values in log** - Log transform clips negatives to 0 with warning
18. **Sparse NaN handling** - `qc_completeness` handles sparse matrices with NaN
19. **Minimal features** - Handles single feature correctly
20. **Large dataset memory** - 1000x500 dataset uses only 8.77 MB
21. **Sparse efficiency** - Operations complete in < 0.01s
22. **Memory leak detection** - No significant memory leaks detected
23. **DIA-NN file format** - Creates valid DIA-NN-like format files
24. **ProvenanceLog tracking** - Operations are correctly logged
25. **Immutable pattern** - Original data not modified by operations

---

## Performance Results

- **Memory usage:** 8.77 MB for 1000x500 dataset (well under 500MB threshold)
- **Operation speed:** Sparse operations complete in < 0.01s
- **Memory leak:** 0.00 MB leaked over 10 iterations

---

## Recommendations

1. ~~**Fix Bug #1 (and #3):** Update `qc_completeness` to handle None mask matrix~~ **FIXED**
2. ~~**Address Bug #2:** Add input validation for log transform with configurable handling~~ **FIXED**
3. **Add tests:** Include edge case tests in the main test suite
4. **Documentation:** Document expected behavior with None masks and negative values
5. **Validation:** Consider adding validation in ScpMatrix initialization for NaN in sparse matrices

---

## Files Referenced

- Test script: `/home/shenshang/projects/ScpTensor/examples/test_qc_bug_hunting.py`
- Bug location: `/home/shenshang/projects/ScpTensor/scptensor/viz/recipes/qc.py`
- Log transform: `/home/shenshang/projects/ScpTensor/scptensor/normalization/log_transform.py`
