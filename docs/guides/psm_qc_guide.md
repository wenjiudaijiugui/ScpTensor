# PSM-Level QC Module Documentation

## Overview

The PSM (Peptide Spectrum Match) Quality Control module provides comprehensive QC operations for single-cell proteomics data at the PSM level. This module is designed for isobaric labeling experiments (TMT, iTRAQ) and includes specialized functions for FDR control, contaminant removal, and channel-level normalization.

## Installation

The module is part of ScpTensor's QC suite:

```python
from scptensor.qc import (
    filter_psms_by_pif,
    filter_psms_by_contaminants,
    pep_to_qvalue,
    filter_psms_by_qvalue,
    compute_sample_carrier_ratio,
    compute_median_cv,
    divide_by_reference,
)
```

## Functions

### 1. PIF Filtering (`filter_psms_by_pif`)

Filter PSMs by Peptide Identification Frequency (PIF) threshold.

**Purpose**: Remove low-confidence peptide identifications based on PIF scores.

**Parameters**:
- `container`: ScpContainer with PSM data
- `assay_name`: Assay name (default: "protein")
- `layer_name`: Layer name (default: "raw")
- `min_pif`: Minimum PIF threshold, 0-1 (default: 0.8)
- `inplace`: If False, add statistics; if True, filter (default: False)

**Example**:
```python
# Preview filter statistics
container = filter_psms_by_pif(container, min_pif=0.8, inplace=False)
keep_mask = container.assays['protein'].var['keep_pif_0.8']

# Apply filter
container = filter_psms_by_pif(container, min_pif=0.8, inplace=True)
```

**Requirements**:
- PIF scores must be stored in `var['pif']`
- Values must be in range [0, 1]

---

### 2. Contaminant Filtering (`filter_psms_by_contaminants`)

Remove reverse sequences and contaminant proteins.

**Purpose**: Filter out decoy matches and common contaminants (keratins, trypsin, etc.)

**Parameters**:
- `container`: ScpContainer with PSM data
- `assay_name`: Assay name (default: "protein")
- `layer_name`: Layer name (default: "raw")
- `contaminant_columns`: List of columns indicating contaminants
  - Default: ["is_reverse", "is_contaminant", "potential_contaminant"]

**Example**:
```python
# Using default columns
container = filter_psms_by_contaminants(container)

# Using custom columns
container = filter_psms_by_contaminants(
    container,
    contaminant_columns=["decoy", "contaminant_flag"]
)
```

**Note**: Always check that your var DataFrame has the necessary columns.

---

### 3. PEP to Q-Value (`pep_to_qvalue`)

Convert Posterior Error Probability (PEP) to q-values for FDR control.

**Purpose**: Control False Discovery Rate using statistical methods.

**Parameters**:
- `container`: ScpContainer with PEP values
- `assay_name`: Assay name (default: "protein")
- `pep_column`: Column with PEP values (default: "pep")
- `method`: "storey" or "bh" (default: "storey")
  - Storey: Estimates π0 (proportion of true nulls) for more power
  - BH: Benjamini-Hochberg procedure (more conservative)
- `lambda_param`: Lambda for Storey's π0 estimation (default: 0.5)

**Example**:
```python
# Compute q-values using Storey's method
container = pep_to_qvalue(container, method="storey")

# Compute q-values using BH method
container = pep_to_qvalue(container, method="bh", lambda_param=0.5)

# Access q-values
qvals = container.assays['protein'].var['qvalue'].to_numpy()

# Filter by 1% FDR
significant = qvals < 0.01
```

**Mathematical Details**:

**Storey's Method**:
1. Estimate π0: `π0 = (# p-values > λ) / (m × (1 - λ))`
2. Calculate q-values with π0 correction
3. More powerful when many true positives exist

**Benjamini-Hochberg**:
1. Assumes π0 = 1 (all nulls are true)
2. Standard linear step-up procedure
3. More conservative

---

### 4. Q-Value Filtering (`filter_psms_by_qvalue`)

Filter PSMs by q-value threshold (FDR control).

**Purpose**: Apply FDR threshold to control false discoveries.

**Parameters**:
- `container`: ScpContainer with q-values (from `pep_to_qvalue`)
- `assay_name`: Assay name (default: "protein")
- `layer_name`: Layer name (default: "raw")
- `max_qvalue`: Maximum q-value/FDR (default: 0.01)
- `inplace`: If False, add statistics; if True, filter (default: False)

**Example**:
```python
# First compute q-values
container = pep_to_qvalue(container, method="storey")

# Filter at 1% FDR
container = filter_psms_by_qvalue(container, max_qvalue=0.01, inplace=True)

# Filter at 5% FDR
container = filter_psms_by_qvalue(container, max_qvalue=0.05, inplace=True)
```

**Workflow**:
```python
# Complete FDR control workflow
container = pep_to_qvalue(container, method="storey")  # Compute q-values
container = filter_psms_by_qvalue(container, max_qvalue=0.01, inplace=True)  # Apply filter
```

---

### 5. Sample-to-Carrier Ratio (`compute_sample_carrier_ratio`)

Compute SCR metrics for isobaric labeling experiments.

**Purpose**: Assess relative contribution of sample vs carrier channels.

**Parameters**:
- `container`: ScpContainer with sample and carrier channels
- `assay_name`: Assay name (default: "protein")
- `layer_name`: Layer name (default: "raw")
- `carrier_identifier`: Identifier for carrier channels (default: "Carrier")
- `max_scr`: Maximum acceptable SCR (default: 0.1)
- `sample_type_column`: Column with sample types (optional)

**Output Columns** (added to `obs`):
- `scr_median`: Median SCR across all features
- `scr_mean`: Mean SCR across all features
- `scr_high_psm_count`: Number of features with SCR > max_scr

**Example**:
```python
# Compute SCR
container = compute_sample_carrier_ratio(
    container,
    carrier_identifier="Carrier",
    sample_type_column="sample_type"
)

# Identify samples with high SCR
high_scr_samples = container.obs.filter(
    pl.col('scr_median') > 0.1
)

# Get SCR statistics
scr_median = container.obs['scr_median'].to_numpy()
scr_mean = container.obs['scr_mean'].to_numpy()
```

**Interpretation**:
- SCR < 0.1: Good (sample contributes < 10% of signal)
- SCR > 0.1: Potential issue (check loading, labeling efficiency)

---

### 6. Median CV (`compute_median_cv`)

Compute median coefficient of variation for samples.

**Purpose**: Assess technical variability and reproducibility.

**Parameters**:
- `container`: ScpContainer with intensity data
- `assay_name`: Assay name (default: "protein")
- `layer_name`: Layer name (default: "raw")
- `cv_threshold`: Threshold for flagging high CV (default: 0.65)
- `group_by`: Column name for grouping (optional)

**Output Columns** (added to `obs`):
- `median_cv`: Median CV across features
- `is_high_cv`: Boolean flag if median_cv > cv_threshold

**Example**:
```python
# Compute CV per sample
container = compute_median_cv(container, cv_threshold=0.65)

# Identify high CV samples
high_cv_samples = container.obs.filter(
    pl.col('is_high_cv') == True
)

# Get CV values
median_cv = container.obs['median_cv'].to_numpy()
```

**Interpretation**:
- CV < 0.5: Good reproducibility
- CV 0.5-0.65: Moderate variability
- CV > 0.65: High variability (potential quality issue)

---

### 7. Reference Normalization (`divide_by_reference`)

Normalize channels by dividing by reference channel.

**Purpose**: Remove technical variation using reference channel.

**Parameters**:
- `container`: ScpContainer with reference channel
- `assay_name`: Assay name (default: "protein")
- `layer_name`: Source layer name (default: "raw")
- `reference_identifier`: Reference channel identifier (default: "Reference")
- `aggregation`: "mean", "median", or "sum" (default: "median")
- `new_layer_name`: Name for normalized layer (default: "reference_normalized")
- `epsilon`: Small constant to avoid division by zero (default: 1e-6)

**Example**:
```python
# Create normalized layer
container = divide_by_reference(
    container,
    reference_identifier="Reference",
    aggregation="median"
)

# Access normalized data
X_norm = container.assays['protein'].layers['reference_normalized'].X

# Use normalized data for downstream analysis
# ...
```

**Use Cases**:
- Cross-run normalization
- Batch effect correction
- Loading difference correction

---

## Typical Workflow

### Complete PSM QC Pipeline

```python
from scptensor.qc import (
    filter_psms_by_pif,
    filter_psms_by_contaminants,
    pep_to_qvalue,
    filter_psms_by_qvalue,
    compute_sample_carrier_ratio,
    compute_median_cv,
    divide_by_reference,
)

# 1. Filter by PIF (quality threshold)
container = filter_psms_by_pif(container, min_pif=0.8, inplace=True)

# 2. Remove contaminants
container = filter_psms_by_contaminants(container)

# 3. FDR control
container = pep_to_qvalue(container, method="storey")
container = filter_psms_by_qvalue(container, max_qvalue=0.01, inplace=True)

# 4. Compute QC metrics
container = compute_sample_carrier_ratio(container)
container = compute_median_cv(container)

# 5. Apply reference normalization
container = divide_by_reference(container)

# 6. Filter low-quality samples
high_scr = container.obs['scr_median'] > 0.1
high_cv = container.obs['is_high_cv']
keep_samples = ~(high_scr | high_cv)

container = container.filter_samples(
    pl.col('scr_median') <= 0.1
)

print(f"Final dataset: {container.n_samples} samples, "
      f"{container.assays['protein'].n_features} features")
```

---

## Error Handling

The module uses ScpTensor's exception hierarchy for clear error messages:

```python
from scptensor.core.exceptions import (
    AssayNotFoundError,
    LayerNotFoundError,
    ScpValueError,
)

# Common errors:
try:
    container = filter_psms_by_pif(container, min_pif=1.5)
except ScpValueError as e:
    print(f"Invalid parameter: {e}")

try:
    container = pep_to_qvalue(container, pep_column="missing_column")
except ScpValueError as e:
    print(f"Column not found: {e}")
```

---

## Best Practices

### 1. Always validate metadata before filtering
```python
# Check required columns exist
assert "pif" in container.assays['protein'].var.columns
assert "pep" in container.assays['protein'].var.columns
```

### 2. Use inplace=False for preview
```python
# Preview without modifying
container = filter_psms_by_pif(container, min_pif=0.8, inplace=False)
keep_mask = container.assays['protein'].var['keep_pif_0.8'].to_numpy()
print(f"Would keep {keep_mask.sum()} / {len(keep_mask)} PSMs")

# Then apply
container = filter_psms_by_pif(container, min_pif=0.8, inplace=True)
```

### 3. Log operations for provenance
```python
# All PSM QC functions log operations automatically
print(container.history[-1]['description'])
```

### 4. Check FDR levels before filtering
```python
container = pep_to_qvalue(container, method="storey")
qvals = container.assays['protein'].var['qvalue'].to_numpy()

print(f"Q-value statistics:")
print(f"  Min: {qvals.min():.4f}")
print(f"  Median: {np.median(qvals):.4f}")
print(f"  Max: {qvals.max():.4f}")
print(f"  @ 1% FDR: {(qvals < 0.01).sum()} PSMs")
print(f"  @ 5% FDR: {(qvals < 0.05).sum()} PSMs")
```

### 5. Use appropriate FDR method
```python
# Storey's method: more powerful, assumes many true positives
container = pep_to_qvalue(container, method="storey")

# BH method: more conservative, suitable for exploratory analysis
container = pep_to_qvalue(container, method="bh")
```

---

## Performance Considerations

### Vectorized Operations
All functions use Polars vectorized operations for efficiency:
- Filtering: `pl.col('pif') >= min_pif`
- Aggregation: `pl.col('value').median()`

### Sparse Matrix Support
Functions handle both dense and sparse matrices:
```python
# Works with both
X_dense = np.array(...)  # Dense
X_sparse = sp.csr_matrix(...)  # Sparse
```

### Memory Efficiency
- Use `inplace=False` to preview without copying data
- Sparse matrices reduce memory for >50% missing data

---

## Testing

Run the test suite:

```bash
# Quick test
uv run python run_psm_tests.py

# Full pytest suite
uv run pytest tests/test_psm_qc.py -v

# With coverage
uv run pytest tests/test_psm_qc.py --cov=scptensor.qc.psm
```

---

## References

### FDR Control
- Storey, J. D. (2002). "A direct approach to false discovery rates." JRSS-B.
- Benjamini, Y., & Hochberg, Y. (1995). "Controlling the false discovery rate." JRSS-B.

### PSM Quality Metrics
- PIF (Peptide Identification Frequency): MSstats quality metric
- SCR (Sample-to-Carrier Ratio): TMT/isobaric labeling QC metric

### Isobaric Labeling
- TMT: Tandem Mass Tag labeling
- iTRAQ: Isobaric Tags for Relative and Absolute Quantitation

---

## API Reference

See source code for complete API documentation:
- `/home/shenshang/projects/ScpTensor/scptensor/qc/psm.py`

All functions follow ScpTensor conventions:
- Complete type annotations
- NumPy-style docstrings
- Functional programming pattern
- Comprehensive error handling

---

## Troubleshooting

### Issue: "PIF column not found"
**Solution**: Ensure PIF scores are computed and stored in `var['pif']`

### Issue: "No carrier channels found"
**Solution**: Check that sample IDs contain the carrier identifier

### Issue: Division by zero in reference normalization
**Solution**: Increase `epsilon` parameter (default: 1e-6)

### Issue: All features filtered by FDR
**Solution**: Check PEP distribution, consider less stringent FDR (5% instead of 1%)

---

**Version**: v0.1.0-beta
**Last Updated**: 2026-01-20
**Author**: ScpTensor Development Team
