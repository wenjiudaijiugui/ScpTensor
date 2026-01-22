# BPCA Implementation - Quick Reference

## File Location
```
scptensor/impute/bpca.py
```

## Quick Summary
- **Status:** ✅ Complete and tested
- **Lines:** 720 (complete rewrite)
- **API:** 100% backward compatible
- **Accuracy:** 99.7% correlation on test data
- **Tests:** All passing (37 passed)

## Key Functions

### Public API
```python
def impute_bpca(
    container: ScpContainer,
    assay_name: str,
    source_layer: str,
    new_layer_name: str = "imputed_bpca",
    n_components: int | None = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    random_state: int | None = None,
) -> ScpContainer
```

### Core Algorithm (New)
```python
def _is_missing(x: npt.NDArray, missing_val: Union[float, None]) -> npt.NDArray
def _not_missing(x: npt.NDArray, missing_val: Union[float, None]) -> npt.NDArray
def bpca_init(y: npt.NDArray, q: int, missing_value: Union[float, None] = np.nan) -> dict
def bpca_em_step(M: dict, y: npt.NDArray) -> dict
def bpca_fill(x999: npt.NDArray, k: Optional[int] = None,
              maxepoch: Optional[int] = None,
              missing_value: Union[float, None] = np.nan) -> Tuple[npt.NDArray, dict]
```

## 8 Critical Fixes Applied

1. **Sparse SVD** - Uses `scipy.sparse.linalg.svds` with sorting
2. **Unbiased Covariance** - Removed `bias=1` from `np.cov()`
3. **Exact Tau Init** - Explicit formula with `np.float64` casting
4. **Post-Processing Normalization** - Normalize AFTER all samples processed
5. **Explicit Addition** - `T = T + ...` instead of `T += ...`
6. **Edge Case Handling** - Handles `gnomiss` correctly
7. **Explicit Tau Bounds** - `max(min(...))` instead of `np.clip()`
8. **Correct yest Reconstruction** - Preserves original non-missing values

## Usage

### Basic
```python
from scptensor import impute_bpca

result = impute_bpca(
    container,
    assay_name="proteins",
    source_layer="raw",
    n_components=10
)
```

### Advanced
```python
from scptensor.impute.bpca import bpca_fill
import numpy as np

# Direct algorithm access
X_filled, model = bpca_fill(
    x999=X_missing,
    k=10,
    maxepoch=100,
    missing_value=np.nan
)

# Access model parameters
W = model['W']      # Weight matrix
mu = model['mu']    # Mean vector
tau = model['tau']  # Noise precision
```

## Test Results
```bash
# Module tests
$ uv run python -m scptensor.impute.bpca
✅ All tests passed

# Integration tests
$ uv run pytest tests/test_impute.py
37 passed, 1 xfailed
```

## Documentation
- Comprehensive docstrings (NumPy style)
- Type hints (numpy.typing.NDArray)
- Detailed notes for each fix
- Usage examples in docstrings

## Performance
- **Accuracy:** 0.997 correlation
- **Convergence:** Every 10 epochs
- **Threshold:** |log10(tau) - log10(tau_old)| < 1e-4
- **Bounds:** tau ∈ [1e-10, 1e10]

## Compatibility
✅ Fully backward compatible
✅ No breaking changes
✅ Same API signature
✅ Same return types

---
**For full details:** See `BPCA_UPDATE_SUMMARY.md`
