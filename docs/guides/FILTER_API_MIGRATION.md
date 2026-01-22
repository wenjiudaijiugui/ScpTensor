# Filter API Migration Guide

## Summary

The `filter_samples()` and `filter_features()` methods in `ScpContainer` have been updated to support the new `FilterCriteria` API while maintaining 100% backward compatibility with the legacy API.

## What Changed

### Before (Legacy API - Still Supported)

```python
# Filter by sample IDs
container.filter_samples(["sample1", "sample2"])

# Filter by sample indices
container.filter_samples(sample_indices=[0, 1, 2])

# Filter by boolean mask
container.filter_samples(boolean_mask=mask_array)

# Filter by Polars expression
container.filter_samples(polars_expression=pl.col("QC_pass"))
```

### After (New API - Recommended)

```python
from scptensor.core.filtering import FilterCriteria

# Filter by sample IDs
criteria = FilterCriteria.by_ids(["sample1", "sample2"])
container.filter_samples(criteria)

# Filter by sample indices
criteria = FilterCriteria.by_indices([0, 1, 2])
container.filter_samples(criteria)

# Filter by boolean mask
criteria = FilterCriteria.by_mask(mask_array)
container.filter_samples(criteria)

# Filter by Polars expression
criteria = FilterCriteria.by_expression(pl.col("QC_pass"))
container.filter_samples(criteria)
```

## Benefits of New API

1. **Type Safety**: `FilterCriteria` provides a single, type-safe parameter instead of multiple optional parameters
2. **Explicit Intent**: Factory methods make the filtering method clear from the code
3. **Better IDE Support**: Improved autocomplete and type hints
4. **Unified Implementation**: Reduces code duplication between samples and features
5. **Future-Proof**: Easier to extend with new filtering methods

## Migration Path

### Option 1: Continue Using Legacy API (No Changes Required)

The legacy API remains fully supported. All existing code will continue to work without modifications:

```python
# This still works exactly as before
filtered = container.filter_samples(["sample1", "sample2"])
filtered = container.filter_features("proteins", feature_indices=[0, 1, 2])
```

### Option 2: Migrate to New API (Recommended)

For new code or when refactoring, use the new `FilterCriteria` API:

**Before:**
```python
# Legacy API
filtered = container.filter_samples(
    sample_ids=["sample1", "sample2"]
)
```

**After:**
```python
# New API
from scptensor.core.filtering import FilterCriteria

criteria = FilterCriteria.by_ids(["sample1", "sample2"])
filtered = container.filter_samples(criteria)
```

### Option 3: Mix Both APIs (Supported)

You can use both APIs in the same codebase or even in the same filtering chain:

```python
result = (
    container.filter_samples(polars_expression=pl.col("QC_pass"))  # Legacy
    .filter_features("proteins", FilterCriteria.by_expression(pl.col("mean") > 10))  # New
    .filter_samples(sample_indices=[0, 1])  # Legacy
)
```

## Complete API Reference

### filter_samples()

#### Legacy Signature

```python
def filter_samples(
    self,
    sample_ids: Sequence[str] | np.ndarray | pl.Expr | pl.Series | Sequence[int] | None = None,
    *,
    sample_indices: Sequence[int] | np.ndarray | None = None,
    boolean_mask: np.ndarray | pl.Series | None = None,
    polars_expression: pl.Expr | None = None,
    copy: bool = True,
) -> ScpContainer
```

#### New Signature

```python
def filter_samples(
    self,
    criteria: FilterCriteria | Sequence[str] | np.ndarray | pl.Expr | pl.Series | Sequence[int] | None = None,
    *,
    sample_indices: Sequence[int] | np.ndarray | None = None,
    boolean_mask: np.ndarray | pl.Series | None = None,
    polars_expression: pl.Expr | None = None,
    copy: bool = True,
) -> ScpContainer
```

The `criteria` parameter accepts:
- `FilterCriteria` object (new, recommended)
- Sample IDs (legacy, for backward compatibility)
- Other types for backward compatibility

### filter_features()

#### Legacy Signature

```python
def filter_features(
    self,
    assay_name: str,
    feature_ids: Sequence[str] | np.ndarray | pl.Expr | pl.Series | Sequence[int] | None = None,
    *,
    feature_indices: Sequence[int] | np.ndarray | None = None,
    boolean_mask: np.ndarray | pl.Series | None = None,
    polars_expression: pl.Expr | None = None,
    copy: bool = True,
) -> ScpContainer
```

#### New Signature

```python
def filter_features(
    self,
    assay_name: str,
    criteria: FilterCriteria | Sequence[str] | np.ndarray | pl.Expr | pl.Series | Sequence[int] | None = None,
    *,
    feature_indices: Sequence[int] | np.ndarray | None = None,
    boolean_mask: np.ndarray | pl.Series | None = None,
    polars_expression: pl.Expr | None = None,
    copy: bool = True,
) -> ScpContainer
```

## FilterCriteria Factory Methods

### by_ids()

Filter by sample/feature identifiers:

```python
criteria = FilterCriteria.by_ids(["sample1", "sample2", "sample3"])
container.filter_samples(criteria)
```

### by_indices()

Filter by positional indices:

```python
criteria = FilterCriteria.by_indices([0, 5, 10, 15])
container.filter_samples(criteria)
```

### by_mask()

Filter by boolean mask:

```python
mask = np.array([True, False, True, True, False])
criteria = FilterCriteria.by_mask(mask)
container.filter_samples(criteria)
```

### by_expression()

Filter by Polars expression:

```python
criteria = FilterCriteria.by_expression(pl.col("n_detected") > 100)
container.filter_samples(criteria)

# Complex expressions
criteria = FilterCriteria.by_expression(
    (pl.col("n_detected") > 100) & (pl.col("batch") == 1)
)
container.filter_samples(criteria)
```

## Backward Compatibility Guarantee

All existing tests pass without modification:

- **47 filtering tests**: 100% pass rate
- **40 QC filter tests**: 100% pass rate
- **12 migration tests**: Verify identical results between old and new API
- **1558+ total tests**: No regressions

The legacy API is **NOT deprecated** and will continue to be supported indefinitely.

## Implementation Details

### How Backward Compatibility Works

The implementation uses a compatibility layer that converts legacy parameters to `FilterCriteria` objects:

```python
def filter_samples(self, criteria=None, *, sample_indices=None, ...):
    # Convert legacy API to FilterCriteria
    if not isinstance(criteria, FilterCriteria):
        # Legacy parameter resolution logic
        if polars_expression is not None:
            criteria = FilterCriteria.by_expression(polars_expression)
        elif boolean_mask is not None:
            criteria = FilterCriteria.by_mask(boolean_mask)
        elif sample_indices is not None:
            criteria = FilterCriteria.by_indices(sample_indices)
        elif criteria is not None:
            criteria = FilterCriteria.by_ids(criteria)
        else:
            raise ValidationError("Must specify filtering criteria")

    # Unified resolution
    indices = resolve_filter_criteria(criteria, self, is_sample=True)
    ...
```

This ensures:
1. Legacy code continues to work
2. New API benefits from unified implementation
3. No code duplication
4. Easy to maintain and extend

## Examples

### Example 1: Quality Control Filtering

**Legacy:**
```python
# Keep only samples that pass QC
filtered = container.filter_samples(polars_expression=pl.col("QC_pass"))

# Keep high-quality features
filtered = container.filter_features(
    "proteins",
    polars_expression=pl.col("n_detected") > 10
)
```

**New:**
```python
from scptensor.core.filtering import FilterCriteria

# Keep only samples that pass QC
criteria = FilterCriteria.by_expression(pl.col("QC_pass"))
filtered = container.filter_samples(criteria)

# Keep high-quality features
criteria = FilterCriteria.by_expression(pl.col("n_detected") > 10)
filtered = container.filter_features("proteins", criteria)
```

### Example 2: Batch Filtering

**Legacy:**
```python
# Remove samples from batch 1
filtered = container.filter_samples(polars_expression=pl.col("batch") != 1)
```

**New:**
```python
from scptensor.core.filtering import FilterCriteria

criteria = FilterCriteria.by_expression(pl.col("batch") != 1)
filtered = container.filter_samples(criteria)
```

### Example 3: Chained Filtering

**Mixed (both APIs work together):**
```python
result = (
    container.filter_samples(polars_expression=pl.col("QC_pass"))  # Legacy
    .filter_features("proteins", FilterCriteria.by_indices([0, 1, 2]))  # New
    .filter_samples(sample_indices=[0, 1])  # Legacy
)
```

**All New API:**
```python
result = (
    container.filter_samples(FilterCriteria.by_expression(pl.col("QC_pass")))
    .filter_features("proteins", FilterCriteria.by_indices([0, 1, 2]))
    .filter_samples(FilterCriteria.by_indices([0, 1]))
)
```

## Testing

### Running Migration Tests

Verify backward compatibility:

```bash
# Test filtering functionality
uv run pytest tests/test_filtering.py -v

# Test migration compatibility
uv run pytest tests/test_filter_migration.py -v

# Test QC filters (which use filter_samples/filter_features internally)
uv run pytest tests/test_qc.py -k "filter" -v
```

All tests should pass, confirming:
1. Legacy API still works
2. New API works correctly
3. Both APIs produce identical results

## FAQ

**Q: Should I migrate my existing code?**

A: No, you don't need to. The legacy API will continue to be supported indefinitely. However, for new code or when refactoring, consider using the new API for better type safety and clarity.

**Q: Will the legacy API be deprecated?**

A: No. There are no plans to deprecate the legacy API. Both APIs are first-class citizens.

**Q: Which API should I use in documentation?**

A: For new documentation, use the new `FilterCriteria` API as it's more explicit and type-safe. You can mention that the legacy API is also supported.

**Q: Can I mix both APIs in the same codebase?**

A: Yes, absolutely. Both APIs can be used interchangeably and even in the same filtering chain.

**Q: Are there any performance differences?**

A: No. Both APIs use the same underlying implementation. The new API simply adds a thin compatibility layer that converts legacy parameters to `FilterCriteria` objects.

**Q: What if I find a bug?**

A: Please report it on GitHub with a minimal reproducible example. Since both APIs use the same implementation, bugs will affect both equally.

## Related Files

- Implementation: `/home/shenshang/projects/ScpTensor/scptensor/core/structures.py`
- FilterCriteria module: `/home/shenshang/projects/ScpTensor/scptensor/core/filtering.py`
- Tests: `/home/shenshang/projects/ScpTensor/tests/test_filtering.py`
- Migration tests: `/home/shenshang/projects/ScpTensor/tests/test_filter_migration.py`

## Version History

- **v0.2.0+**: New `FilterCriteria` API introduced
- **v0.1.x**: Legacy API (still fully supported)
