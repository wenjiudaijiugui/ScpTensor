# ScpTensor Architecture Specification

**Version:** v0.1.0-beta Target
**Document Version:** 1.0
**Last Updated:** 2025-01-05
**Status:** Design Specification

---

## 1. Module Responsibility Matrix

### Overview

ScpTensor is organized into specialized modules, each with a clear responsibility and well-defined public API. This section documents the architectural design at the module level.

---

### Core Layer (scptensor.core)

**Responsibility:** Data structures and provenance tracking foundation

**Public API:**
- `ScpContainer` - Global sample management across assays
- `Assay` - Feature-space specific data layer management
- `ScpMatrix` - Physical storage with mask-based provenance tracking
- `ProvenanceLog` - Operation history audit trail
- `MaskCode` - Data status enumeration
- `MatrixOps` - Matrix operation utilities
- `reader` - Data ingestion utilities

**Dependencies:** None (base layer for all other modules)

**Design Principles:**
- **Immutability:** Functions return new objects rather than modifying in-place
- **Type Safety:** Full type annotations on all public APIs
- **Validation:** Strict input validation with clear error messages

---

### Data Processing Modules

#### normalization/

**Responsibility:** Transform data distributions (log transform, scaling, centering)

**Inputs:** `ScpContainer` with source layer
**Outputs:** `ScpContainer` with new normalized layer
**Side Effects:** Updates `container.history` with operation log

**Methods:**
- `log_normalize()` - Log transform (base 2 by default)
- `sample_median_normalization()` - Median centering per sample
- `sample_mean_normalization()` - Mean centering per sample
- `global_median_normalization()` - Global median centering
- `tmm_normalization()` - TMM scaling (for between-sample normalization)
- `upper_quartile_normalization()` - Upper quartile scaling
- `zscore_standardization()` - Z-score normalization

**Data Contract:**
- Preserves sample/feature dimensions exactly
- Creates new layer (does not modify source)
- Returns `ScpContainer` (functional pattern)
- Updates `ProvenanceLog` with operation details

**Design Decision:** Why multiple normalization methods?

Different SCP analysis workflows require different normalization strategies:
- **Log transform:** Standard for proteomics intensity data
- **Median centering:** Removes sample-specific biases
- **TMM:** Effective for between-sample comparisons
- **Z-score:** Required for some downstream algorithms

---

#### impute/

**Responsibility:** Fill missing values using statistical/matrix methods

**Methods:**
- `knn()` - K-nearest neighbors imputation
- `ppca()` - Probabilistic PCA imputation
- `svd()` - Singular value decomposition imputation
- `missforest()` - Random forest-based imputation

**Data Contract:**
- **Input:** Layer with `M` mask (non-zero = missing)
- **Output:** Layer with `M` mask updated (all zeros = filled)
- **Provenance:** Original missing values marked as `IMPUTED` (5) in mask
- **Sparsity:** Preserves sparsity where possible
- **Updates:** `ProvenanceLog` with imputation parameters

**Design Decision:** Why 4 methods?

| Method | Strength | Use Case | Complexity |
|--------|----------|----------|------------|
| KNN | Fast, local similarity | Small datasets, quick prototyping | O(n²) |
| PPCA | Global structure preservation | Large-scale missingness | O(n³) |
| SVD | Low-rank approximation | Matrix completion problems | O(n³) |
| MissForest | Non-parametric | Complex missing patterns | O(n² log n) |

**Implementation Notes:**
- All methods must handle both dense and sparse matrices
- KNN should use efficient nearest neighbor algorithms (ball tree, kd-tree)
- PPCA/SVD should use randomized SVD for large datasets
- MissForest implementation should use scikit-learn's RandomForestRegressor

---

#### integration/

**Responsibility:** Remove batch effects and merge multiple datasets

**Methods:**
- `combat()` - Empirical Bayes (ComBat)
- `harmony()` - Iterative clustering-based correction
- `mnn()` - Mutual nearest neighbors alignment
- `scanorama()` - Data fusion-based integration

**Data Contract:**
- **Requires:** `obs[batch_column]` for batch labels
- **Modifies:** Values in specified layer (creates new layer)
- **Preserves:** Biological variance (goal)
- **Updates:** `ProvenanceLog` with correction parameters

**Design Decision:** Why multiple integration methods?

| Method | Strength | Limitations | Best For |
|--------|----------|-------------|----------|
| ComBat | Proven, effective | Assumes linear batch effects | Simple batch structures |
| Harmony | Preserves biological variance | Requires PCA first | Complex multi-batch data |
| MNN | Non-linear alignment | Slow for large datasets | Aligning diverse datasets |
| Scanorama | Scalable data fusion | Complex hyperparameters | Large multi-batch studies |

**Common Requirements:**
- All methods assume data is already normalized and log-transformed
- All methods create new layer (don't modify source)
- All methods update `ProvenanceLog`

---

#### qc/

**Responsibility:** Identify and flag low-quality samples/features

**Methods:**
- `basic_qc()` - Missing rate, detection rate analysis
- `outlier_detection()` - Statistical outlier identification

**Data Contract:**
- **Input:** `ScpContainer` with data layer
- **Output:** `ScpContainer` with updated `obs` and `var`
- **Adds Columns:**
  - `obs['completeness']` - Fraction of valid values per sample
  - `obs['n_detected']` - Number of detected features per sample
  - `obs['is_outlier']` - Boolean flag for outlier samples
  - `var['completeness']` - Fraction of valid values per feature
  - `var['n_samples_detected']` - Number of samples detecting each feature
- **Side Effects:** None (only adds metadata, doesn't filter)

**Design Decision:** Separation of concerns

QC methods **identify** issues but don't **filter** them. Users make explicit filtering decisions:
```python
# QC identifies outliers
container = basic_qc(container, assay_name='proteins')
container = outlier_detection(container, method='isolation_forest')

# User decides to filter
outliers = container.obs.filter(pl.col('is_outlier') == True)
container = filter_samples(container, sample_ids=outliers['_index'])
```

---

#### dim_reduction/

**Responsibility:** Reduce dimensionality for visualization and analysis

**Methods:**
- `pca()` - Principal Component Analysis
- `umap()` - Uniform Manifold Approximation and Projection

**Data Contract:**
- **Input:** `ScpContainer` with source assay
- **Output:** `ScpContainer` with new assay containing embeddings
- **New Assay Structure:**
  - `assay.layers['scores']` - Embedding coordinates (n_samples × n_components)
  - `assay.var` contains metadata (explained_variance, loadings, etc.)
- **Creates:** New assay (doesn't modify source assay)

**Design Decision:** Why create new assay for embeddings?

Embeddings are first-class objects in ScpTensor:
- Can be further processed (e.g., UMAP on PCA)
- Have their own metadata (variance explained, nearest neighbors)
- Enable multiple reduction strategies on same data
- Clear separation from original feature space

---

#### cluster/

**Responsibility:** Group samples into clusters

**Methods:**
- `run_kmeans()` - K-means clustering
- `run_leiden()` - Leiden graph clustering
- `run_louvain()` - Louvain graph clustering

**Data Contract:**
- **Input:** `ScpContainer` with embedding assay (usually PCA)
- **Output:** `ScpContainer` with updated `obs`
- **Adds Column:** `obs[key_added]` with cluster labels (0, 1, 2, ...)
- **Side Effects:** None (only adds labels to obs)

**Design Decision:** Separate cluster from dim_reduction

Clustering operates on **embeddings**, not raw data:
- More efficient
- Standard workflow (PCA → cluster)
- Enables clustering on any embedding (PCA, UMAP, etc.)

---

#### viz/

**Responsibility:** Generate publication-quality visualizations

**Methods:**
- `qc_completeness()` - Bar plot of data completeness
- `qc_matrix_spy()` - Spy plot of missing value patterns
- `embedding()` - Scatter plot of dimensionality reduction
- `heatmap()` - Heatmap of feature expression
- `volcano()` - Volcano plot for differential expression

**Data Contract:**
- **Input:** `ScpContainer` with relevant data
- **Output:** `matplotlib.pyplot.Axes` object
- **Style:** SciencePlots (["science", "no-latex"])
- **DPI:** 300 (publication quality)
- **Language:** English only (no Chinese characters)

**Design Decision:** Return Axes objects

Returns Axes for flexibility:
```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
embedding(container, 'umap', 'group', ax=axes[0])
embedding(container, 'umap', 'batch', ax=axes[1])
plt.tight_layout()
plt.savefig('comparison.png', dpi=300)
```

---

## 2. Data Structure Specifications

### ScpContainer

**Purpose:** Top-level manager for multi-assay experiments

```python
class ScpContainer:
    def __init__(
        self,
        obs: pl.DataFrame,
        assays: Optional[Dict[str, Assay]] = None,
        links: Optional[List[AggregationLink]] = None,
        history: Optional[List[ProvenanceLog]] = None,
        sample_id_col: str = "_index"
    ) -> None
```

**Attributes:**

| Attribute | Type | Description | Mutability |
|-----------|------|-------------|------------|
| `obs` | `pl.DataFrame` | Sample metadata (n_samples × metadata_cols) | Read-only view |
| `assays` | `Dict[str, Assay]` | Assay registry (name → assay mapping) | Read-only view |
| `links` | `List[AggregationLink]` | Assay relationships | Read-only view |
| `history` | `List[ProvenanceLog]` | Operation audit trail | Read-only view |
| `sample_id_col` | `str` | Column name for unique sample IDs | Immutable |
| `n_samples` | `int` | Number of samples (property) | Computed |
| `sample_ids` | `pl.Series` | Sample ID column (property) | Computed |

**Invariants:**

1. **Sample Dimension Consistency:**
   ```
   For all assays A, all layers L in A.layers:
       L.X.shape[0] == obs.height
   ```

2. **Unique Sample IDs:**
   ```
   obs[sample_id_col].n_unique() == obs.height
   ```

3. **Valid Assay Links:**
   ```
   For all links L:
       L.source_assay in assays.keys()
       L.target_assay in assays.keys()
   ```

**Methods:**

- `add_assay(name: str, assay: Assay) -> None` - Register new assay
- `log_operation(...) -> None` - Add operation to history
- `validate_links() -> None` - Validate AggregationLink integrity
- `copy(deep: bool = True) -> ScpContainer` - Copy container
- `deepcopy() -> ScpContainer` - Deep copy
- `shallow_copy() -> ScpContainer` - Shallow copy (shared assays)

---

### Assay

**Purpose:** Feature-space specific data manager

```python
class Assay:
    def __init__(
        self,
        var: pl.DataFrame,
        layers: Optional[Dict[str, ScpMatrix]] = None,
        feature_id_col: str = "_index"
    ) -> None
```

**Attributes:**

| Attribute | Type | Description | Mutability |
|-----------|------|-------------|------------|
| `var` | `pl.DataFrame` | Feature metadata (n_features × metadata_cols) | Read-only view |
| `layers` | `Dict[str, ScpMatrix]` | Data layers (name → matrix mapping) | Read-only view |
| `feature_id_col` | `str` | Column name for unique feature IDs | Immutable |
| `n_features` | `int` | Number of features (property) | Computed |
| `feature_ids` | `pl.Series` | Feature ID column (property) | Computed |
| `X` | `Union[np.ndarray, sp.spmatrix]` | Shortcut to 'X' layer (property) | Computed |

**Invariants:**

1. **Feature Dimension Consistency:**
   ```
   For all layers L in layers:
       L.X.shape[1] == var.height
   ```

2. **Unique Feature IDs:**
   ```
   var[feature_id_col].n_unique() == var.height
   ```

3. **Layer Naming Convention:**
   ```
   Common layer names: 'raw', 'log', 'imputed', 'corrected', 'scaled'
   ```

**Methods:**

- `add_layer(name: str, matrix: ScpMatrix) -> None` - Add data layer
- `subset(feature_indices, copy_data=True) -> Assay` - Subset features
- `get_layer(layer_name: str) -> ScpMatrix` - Get layer with validation

---

### ScpMatrix

**Purpose:** Physical storage with provenance tracking via mask

```python
@dataclass
class ScpMatrix:
    X: Union[np.ndarray, sp.spmatrix]           # Quantitative values
    M: Union[np.ndarray, sp.spmatrix, None]     # Mask codes
    metadata: Optional[MatrixMetadata]          # Quality scores
```

**Attributes:**

| Attribute | Type | Description | Constraints |
|-----------|------|-------------|-------------|
| `X` | `np.ndarray` or `sp.spmatrix` | Quantitative values | `dtype` must be `float64` or `float32` |
| `M` | `np.ndarray`, `sp.spmatrix`, or `None` | Mask codes | `dtype` must be `int8` if present |
| `metadata` | `MatrixMetadata` or `None` | Quality metadata | Optional |

**Invariants:**

1. **Shape Consistency:**
   ```
   if M is not None:
       X.shape == M.shape
   ```

2. **Valid Mask Codes:**
   ```
   if M is not None:
       all(M values in {0, 1, 2, 3, 4, 5, 6})
   ```

3. **Floating Point X:**
   ```
   np.issubdtype(X.dtype, np.floating) == True
   ```

**MaskCode Enum:**

| Code | Name | Description |
|------|------|-------------|
| 0 | VALID | Valid detected value |
| 1 | MBR | Missing Between Runs |
| 2 | LOD | Below Limit of Detection |
| 3 | FILTERED | Filtered out by QC |
| 4 | OUTLIER | Statistical outlier |
| 5 | IMPUTED | Imputed/filled value |
| 6 | UNCERTAIN | Uncertain data quality |

**Methods:**

- `get_m() -> Union[np.ndarray, sp.spmatrix]` - Get mask (zeros if None)
- `copy() -> ScpMatrix` - Deep copy

---

### AggregationLink

**Purpose:** Define relationships between assays (e.g., peptide → protein)

```python
@dataclass
class AggregationLink:
    source_assay: str                    # Source assay name
    target_assay: str                    # Target assay name
    linkage: pl.DataFrame                # Mapping table
```

**Required Columns in `linkage`:**

| Column | Type | Description |
|--------|------|-------------|
| `source_id` | str | Feature ID in source assay |
| `target_id` | str | Feature ID in target assay |

**Invariants:**

1. **Valid Assays:**
   ```
   source_assay in container.assays.keys()
   target_assay in container.assays.keys()
   ```

2. **Valid Feature IDs:**
   ```
   all(linkage['source_id'] in container.assays[source_assay].feature_ids)
   all(linkage['target_id'] in container.assays[target_assay].feature_ids)
   ```

---

### ProvenanceLog

**Purpose:** Track all operations for reproducibility

```python
@dataclass
class ProvenanceLog:
    timestamp: str                          # ISO format timestamp
    action: str                             # Operation name
    params: Dict[str, Any]                  # Operation parameters
    software_version: Optional[str]         # ScpTensor version
    description: Optional[str]              # Human-readable description
```

**Example:**
```python
ProvenanceLog(
    timestamp="2025-01-05T14:30:00",
    action="log_normalize",
    params={"base": 2.0, "offset": 1.0},
    software_version="0.1.0-beta",
    description="Log2 transform with offset 1.0"
)
```

---

## 3. Design Patterns

### Pattern 1: Immutable Layer Creation

**Problem:** Tracking data transformations through analysis pipeline

**Solution:** Functions create new layers, never modify in-place

```python
# Good: Functional pattern
container = log_normalize(
    container,
    assay_name="proteins",
    base_layer="raw",
    new_layer_name="log"
)
# Creates: container.assays['proteins'].layers['log']
# Preserves: container.assays['proteins'].layers['raw']

# Bad: In-place modification (breaks reproducibility)
container.assays['proteins'].layers['log'] = ...
# No provenance tracking
```

**Rationale:**
- Enables full reproducibility via `ProvenanceLog`
- Users can compare different normalization strategies
- Easy rollback to previous layers
- Clear data lineage

---

### Pattern 2: Mask-Based Provenance

**Problem:** Knowing which values are original vs. imputed vs. filtered

**Solution:** Mask matrix tracks data status independently from values

```python
# Before imputation
X[row, col] = 0.0        # Placeholder
M[row, col] = 1          # MBR (missing)

# After KNN imputation
X[row, col] = 12.3       # Filled value
M[row, col] = 5          # IMPUTED

# Downstream analysis can filter/weight by provenance
valid_mask = (M == 0)           # Only original detected values
imputed_mask = (M == 5)         # Only imputed values
```

**Rationale:**
- Downstream methods can treat imputed values differently
- Enables sensitivity analysis (with/without imputed values)
- Clear audit trail for publications
- Quality control transparency

---

### Pattern 3: Assay-Aggregation Links

**Problem:** Relating peptide-level data to protein-level data

**Solution:** Explicit `AggregationLink` objects with mapping tables

```python
link = AggregationLink(
    source_assay="peptides",
    target_assay="proteins",
    linkage=pl.DataFrame({
        "source_id": ["PEP001", "PEP002", "PEP003", "PEP004"],
        "target_id": ["PROT001", "PROT001", "PROT002", "PROT002"]
    })
)

# Usage: Aggregate peptides to proteins
container.aggregage_proteins(
    link=link,
    method="sum",
    new_assay_name="proteins"
)
```

**Rationale:**
- Flexible many-to-many mappings (isoforms → protein)
- Preserves mapping for reproducibility
- Enables different aggregation strategies (sum, mean, max)
- Clear audit trail

---

### Pattern 4: Functional Pipeline Design

**Problem:** Building complex analysis pipelines

**Solution:** All functions accept `ScpContainer`, return modified `ScpContainer`

```python
# Pipeline: Raw → Normalized → Imputed → Corrected → Clustered
container = (
    container
    |> log_normalize(assay_name="proteins", base_layer="raw", new_layer_name="log")
    |> knn(assay_name="proteins", base_layer="log", new_layer_name="imputed", k=5)
    |> combat(batch_key="batch", assay_name="proteins", base_layer="imputed", new_layer_name="corrected")
    |> pca(assay_name="proteins", base_layer_name="corrected", n_components=50)
    |> run_kmeans(assay_name="pca", n_clusters=5)
)

# Alternative: Standard Python
container = log_normalize(container, assay_name="proteins", base_layer="raw", new_layer_name="log")
container = knn(container, assay_name="proteins", base_layer="log", new_layer_name="imputed", k=5)
container = combat(container, batch_key="batch", assay_name="proteins", base_layer="imputed", new_layer_name="corrected")
container = pca(container, assay_name="proteins", base_layer_name="corrected", n_components=50)
container = run_kmeans(container, assay_name="pca", n_clusters=5)
```

**Rationale:**
- Composable operations
- Easy to insert/remove steps
- Clear data flow
- Testable individual components

---

## 4. Integration Patterns

### Pattern A: Sequential Processing

**Most common workflow:** Data flows through modules sequentially

```
Raw Data
  → QC (identify outliers)
  → Normalization (log transform)
  → Imputation (fill missing)
  → Integration (batch correction)
  → Dimensionality Reduction (PCA)
  → Clustering (KMeans)
  → Visualization
```

**Key Points:**
- Each step creates new layer
- Original data preserved
- Easy to backtrack
- Provenance automatically tracked

---

### Pattern B: Branching Workflows

**Multiple preprocessing strategies compared:**

```python
# Strategy 1: KNN imputation
container_knn = container |> log_normalize() |> knn(k=5)

# Strategy 2: PPCA imputation
container_ppca = container |> log_normalize() |> ppca(n_components=10)

# Compare results
compare_results(container_knn, container_ppca)
```

---

### Pattern C: Multi-Assay Analysis

**Analyze both peptides and proteins:**

```python
# Container with two assays
container = ScpContainer(
    obs=obs,
    assays={
        "peptides": peptide_assay,
        "proteins": protein_assay
    },
    links=[peptide_to_protein_link]
)

# Analyze separately
container = log_normalize(container, assay_name="peptides", ...)
container = log_normalize(container, assay_name="proteins", ...)

# Aggregate peptides to proteins
container = aggregate(
    container,
    link=peptide_to_protein_link,
    method="sum"
)
```

---

## 5. Error Handling Strategy

### Error Hierarchy

```python
class ScpTensorError(Exception):
    """Base exception for all ScpTensor errors"""
    pass

class StructureError(ScpTensorError):
    """Data structure inconsistency"""
    pass

class ValidationError(ScpTensorError):
    """Input validation failure"""
    pass

class LayerNotFoundError(ScpTensorError):
    """Requested layer doesn't exist"""
    pass

class AssayNotFoundError(ScpTensorError):
    """Requested assay doesn't exist"""
    pass
```

### Validation Strategy

**Preconditions:**
```python
def log_normalize(container, assay_name, base_layer, new_layer_name, ...):
    # Validate assay exists
    if assay_name not in container.assays:
        raise AssayNotFoundError(f"Assay '{assay_name}' not found")

    # Validate base layer exists
    if base_layer not in container.assays[assay_name].layers:
        raise LayerNotFoundError(f"Layer '{base_layer}' not found in assay '{assay_name}'")

    # Validate data is positive (for log)
    X = container.assays[assay_name].layers[base_layer].X
    if np.any(X < 0):
        raise ValueError("Log transform requires non-negative values")

    # Proceed with computation
    ...
```

**Postconditions:**
```python
    # After computation, validate output
    if new_layer.X.shape != input_shape:
        raise StructureError("Output shape mismatch")

    if np.any(np.isnan(new_layer.X)):
        raise StructureError("Output contains NaN values")

    # Add to history
    container.log_operation(...)
    return container
```

---

## 6. Performance Considerations

### Sparsity

**ScpTensor supports both dense and sparse matrices:**

```python
# Dense: For small/complete datasets
X_dense = np.random.rand(100, 1000)
matrix = ScpMatrix(X=X_dense, M=None)

# Sparse: For large/sparse datasets (SCP typical)
X_sparse = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(n_samples, n_features))
matrix = ScpMatrix(X=X_sparse, M=M_sparse)
```

**Best Practices:**
- Use sparse matrices when missing rate > 50%
- Preserve sparsity through pipeline (avoid `.toarray()`)
- KNN should support sparse inputs (use sparse distance metrics)

---

### Memory Management

**Layer copying:**
```python
# Immutable pattern creates copies (memory overhead)
container = log_normalize(container, ...)  # Creates new layer

# For large datasets, consider removing old layers
del container.assays['proteins'].layers['raw']  # Free memory
```

**Recommendation:** Implement `cleanup_layers()` utility
```python
def cleanup_layers(container, assay_name, keep_layers):
    """Remove all layers except specified ones"""
    assay = container.assays[assay_name]
    to_remove = [name for name in assay.layers if name not in keep_layers]
    for name in to_remove:
        del assay.layers[name]
    return container
```

---

### Numba JIT

**Target operations for JIT compilation:**

1. **Mask operations:** Fast iteration over mask codes
2. **Distance computations:** KNN imputation distance calculations
3. **Matrix operations:** Custom matrix functions not in NumPy/SciPy

**Example:**
```python
from numba import jit
import numpy as np

@jit(nopython=True, cache=True)
def count_mask_codes(M: np.ndarray) -> np.ndarray:
    """Fast count of each mask code"""
    counts = np.zeros(7, dtype=np.int64)  # 7 mask codes
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            counts[M[i, j]] += 1
    return counts
```

---

## 7. Extension Points

### Adding New Normalization Methods

```python
# 1. Create function in normalization/your_method.py
def your_normalize(
    container: ScpContainer,
    assay_name: str,
    base_layer: str,
    new_layer_name: str = "your_normalized",
    **params
) -> ScpContainer:
    """
    Your normalization method.

    Parameters
    ----------
    container : ScpContainer
        Input container
    assay_name : str
        Target assay name
    base_layer : str
        Source layer name
    new_layer_name : str, optional
        New layer name

    Returns
    -------
    ScpContainer
        Container with new normalized layer
    """
    # Validate inputs
    assay = container.assays[assay_name]
    if base_layer not in assay.layers:
        raise LayerNotFoundError(...)

    # Apply normalization
    X = assay.layers[base_layer].X
    X_norm = your_algorithm(X, **params)

    # Create new layer
    new_layer = ScpMatrix(X=X_norm, M=assay.layers[base_layer].M)
    assay.add_layer(new_layer_name, new_layer)

    # Log operation
    container.log_operation(
        action="your_normalize",
        params=params,
        description="Your normalization method"
    )

    return container

# 2. Export in normalization/__init__.py
from .your_method import your_normalize

__all__ = [..., "your_normalize"]
```

---

### Adding New Imputation Methods

Follow same pattern as normalization:
1. Create function in `impute/your_method.py`
2. Accept `ScpContainer`, return `ScpContainer`
3. Update mask: `M[missing] = MaskCode.IMPUTED.value`
4. Export in `impute/__init__.py`

---

### Adding New Integration Methods

Follow same pattern:
1. Create function in `integration/your_method.py`
2. Require `batch_key` parameter for `obs[batch_column]`
3. Create new layer with corrected values
4. Export in `integration/__init__.py`

---

## 8. Testing Strategy

### Unit Test Structure

```python
# tests/core/test_structures.py
import pytest
import numpy as np
from scptensor.core import ScpContainer, Assay, ScpMatrix

class TestScpMatrix:
    def test_shape_validation(self):
        """M should have same shape as X"""
        X = np.random.rand(10, 20)
        M = np.zeros((10, 20), dtype=np.int8)
        matrix = ScpMatrix(X=X, M=M)
        assert matrix.X.shape == matrix.M.shape

    def test_invalid_mask_code_raises(self):
        """Invalid mask codes should raise ValueError"""
        X = np.random.rand(10, 20)
        M = np.full((10, 20), 99, dtype=np.int8)  # Invalid code
        with pytest.raises(ValueError):
            ScpMatrix(X=X, M=M)

class TestAssay:
    def test_subset_preserves_features(self):
        """Subsetting should return correct number of features"""
        var = pl.DataFrame({"_index": [f"P{i:03d}" for i in range(100)]})
        X = np.random.rand(50, 100)
        assay = Assay(var=var, layers={"X": ScpMatrix(X=X)})

        subset = assay.subset([0, 1, 2])
        assert subset.n_features == 3
```

### Integration Test Structure

```python
# tests/integration/test_pipeline.py
def test_full_pipeline():
    """Test complete analysis pipeline"""
    # Generate synthetic data
    container = generate_synthetic_data(n_samples=100, n_features=500)

    # Run pipeline
    container = log_normalize(container, assay_name="proteins", ...)
    container = knn(container, assay_name="proteins", ...)
    container = combat(container, batch_key="batch", ...)
    container = pca(container, assay_name="proteins", ...)
    container = run_kmeans(container, assay_name="pca", n_clusters=3)

    # Validate results
    assert "log" in container.assays["proteins"].layers
    assert "imputed" in container.assays["proteins"].layers
    assert "corrected" in container.assays["proteins"].layers
    assert "pca" in container.assays
    assert "kmeans_cluster" in container.obs.columns
```

---

## 9. Deprecation Policy

### Versioning

ScpTensor follows Semantic Versioning 2.0.0:
- **MAJOR:** Incompatible API changes
- **MINOR:** Backwards-compatible functionality
- **PATCH:** Backwards-compatible bug fixes

### Deprecation Process

1. **Mark as deprecated:**
```python
def old_function(...):
    """
    .. deprecated:: 0.2.0
        old_function is deprecated and will be removed in v0.3.0.
        Use new_function instead.
    """
    import warnings
    warnings.warn(
        "old_function is deprecated, use new_function",
        DeprecationWarning,
        stacklevel=2
    )
    ...
```

2. **Document in API_REFERENCE.md** under "Deprecation Timeline"

3. **Remove in next MAJOR version**

---

## Appendix

### A. Type Annotation Standards

All public APIs must have complete type annotations:

```python
from typing import Dict, List, Optional, Union
import numpy as np
import polars as pl

def example_function(
    container: ScpContainer,
    assay_name: str,
    layers: Optional[List[str]] = None,
    params: Optional[Dict[str, Any]] = None
) -> ScpContainer:
    ...
```

### B. Naming Conventions

- **Modules:** lowercase_with_underscores (e.g., `dim_reduction`)
- **Classes:** CapitalizedWords (e.g., `ScpContainer`)
- **Functions:** lowercase_with_underscores (e.g., `log_normalize`)
- **Constants:** UPPER_CASE (e.g., `DEFAULT_OFFSET`)
- **Private:** _leading_underscore (e.g., `_internal_helper`)

### C. Documentation Standards

All public functions must have NumPy-style docstrings:

```python
def log_normalize(
    container: ScpContainer,
    assay_name: str,
    base_layer: str,
    new_layer_name: str = "log",
    base: float = 2.0,
    offset: float = 1.0
) -> ScpContainer:
    """
    Apply log transformation to data.

    Computes log_base(X + offset) for all values in the specified layer.

    Parameters
    ----------
    container : ScpContainer
        Input container with data to normalize
    assay_name : str
        Name of assay containing the layer to normalize
    base_layer : str
        Name of layer to transform (e.g., "raw")
    new_layer_name : str, optional
        Name for the new normalized layer (default: "log")
    base : float, optional
        Logarithm base (default: 2.0 for log2)
    offset : float, optional
        Additive offset to avoid log(0) (default: 1.0)

    Returns
    -------
    ScpContainer
        Container with new normalized layer added

    Raises
    ------
    LayerNotFoundError
        If base_layer does not exist in assay
    ValueError
        If data contains negative values

    Examples
    --------
    >>> container = log_normalize(
    ...     container,
    ...     assay_name="proteins",
    ...     base_layer="raw",
    ...     new_layer_name="log",
    ...     base=2.0
    ... )

    Notes
    -----
    Log transformation is standard for proteomics intensity data.
    Base 2 is commonly used (log2 fold changes).

    See Also
    --------
    sample_median_normalization : Median centering
    zscore_standardization : Z-score normalization
    """
```

---

**Document Owner:** ScpTensor Architecture Team
**Review Cycle:** Per release or when architecture changes
**Next Review:** v0.1.0-beta release

**End of ARCHITECTURE.md**
