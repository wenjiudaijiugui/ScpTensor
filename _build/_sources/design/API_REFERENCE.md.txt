# ScpTensor API Reference

**Version:** v0.1.0-beta
**Last Updated:** 2025-01-05
**API Stability:** Public (backwards compatible commitment)

---

## Table of Contents

- [Core Data Structures](#1-core-data-structures)
- [Normalization Module](#2-normalization-module)
- [Imputation Module](#3-imputation-module)
- [Integration Module](#4-integration-module)
- [QC Module](#5-qc-module)
- [Dimensionality Reduction Module](#6-dimensionality-reduction-module)
- [Clustering Module](#7-clustering-module)
- [Visualization Module](#8-visualization-module)
- [Utils Module](#9-utils-module)
- [Exceptions](#10-exceptions)

---

## 1. Core Data Structures

### ScpContainer

Top-level container for multi-assay single-cell proteomics experiments.

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

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `obs` | `pl.DataFrame` | ✅ | Sample metadata (n_samples × metadata_cols). Must contain unique IDs in `sample_id_col` |
| `assays` | `Dict[str, Assay]` | ❌ | Assay registry (name → assay mapping). Default: empty dict |
| `links` | `List[AggregationLink]` | ❌ | Assay relationships for aggregation. Default: empty list |
| `history` | `List[ProvenanceLog]` | ❌ | Operation audit trail. Default: empty list |
| `sample_id_col` | `str` | ❌ | Column in `obs` containing unique sample IDs. Default: "_index" |

**Attributes:**

```python
container.n_samples      # int: Number of samples
container.sample_ids     # pl.Series: Sample ID column
container.obs            # pl.DataFrame: Sample metadata (read-only)
container.assays         # Dict[str, Assay]: Assay registry (read-only)
container.links          # List[AggregationLink]: Assay relationships
container.history        # List[ProvenanceLog]: Operation history
```

**Methods:**

#### `add_assay(name, assay)`

Register a new assay to the container.

```python
def add_assay(
    self,
    name: str,
    assay: Assay
) -> None
```

**Parameters:**
- `name`: Assay name (e.g., "proteins", "peptides")
- `assay`: Assay object

**Raises:**
- `ValueError`: If assay name already exists
- `ValueError`: If sample dimension mismatch

**Example:**
```python
var = pl.DataFrame({"_index": ["P1", "P2", "P3"]})
X = np.random.rand(10, 3)
assay = Assay(var=var, layers={"raw": ScpMatrix(X=X)})
container.add_assay("proteins", assay)
```

---

#### `log_operation(action, params, description, software_version)`

Record an operation to the history log.

```python
def log_operation(
    self,
    action: str,
    params: Dict[str, Any],
    description: Optional[str] = None,
    software_version: Optional[str] = None
) -> None
```

**Parameters:**
- `action`: Operation name (e.g., "normalize", "impute")
- `params`: Dictionary of operation parameters
- `description`: Human-readable description (optional)
- `software_version`: ScpTensor version (optional)

**Example:**
```python
container.log_operation(
    action="log_normalize",
    params={"base": 2.0, "offset": 1.0},
    description="Log2 transform with offset 1.0"
)
```

---

#### `copy(deep=True)`

Copy the container.

```python
def copy(
    self,
    deep: bool = True
) -> ScpContainer
```

**Parameters:**
- `deep`: If True, deep copy all data. If False, shallow copy (shared assays)

**Returns:** New ScpContainer instance

---

### Assay

Feature-space specific data manager (e.g., proteins, peptides).

```python
class Assay:
    def __init__(
        self,
        var: pl.DataFrame,
        layers: Optional[Dict[str, ScpMatrix]] = None,
        feature_id_col: str = "_index"
    ) -> None
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `var` | `pl.DataFrame` | ✅ | Feature metadata (n_features × metadata_cols). Must contain unique IDs in `feature_id_col` |
| `layers` | `Dict[str, ScpMatrix]` | ❌ | Data layer registry. Default: empty dict |
| `feature_id_col` | `str` | ❌ | Column in `var` containing unique feature IDs. Default: "_index" |

**Attributes:**

```python
assay.n_features       # int: Number of features
assay.feature_ids      # pl.Series: Feature ID column
assay.var              # pl.DataFrame: Feature metadata (read-only)
assay.layers           # Dict[str, ScpMatrix]: Layer registry (read-only)
assay.X                # Shortcut to layers['X'].X (if 'X' layer exists)
```

**Methods:**

#### `add_layer(name, matrix)`

Add a new data layer.

```python
def add_layer(
    self,
    name: str,
    matrix: ScpMatrix
) -> None
```

**Parameters:**
- `name`: Layer name (e.g., "raw", "log", "imputed")
- `matrix`: ScpMatrix object

**Raises:**
- `ValueError`: If feature dimension mismatch

---

#### `subset(feature_indices, copy_data=True)`

Return a new assay with a subset of features.

```python
def subset(
    self,
    feature_indices: Union[List[int], np.ndarray],
    copy_data: bool = True
) -> Assay
```

**Parameters:**
- `feature_indices`: Indices of features to keep
- `copy_data`: Whether to copy underlying data

**Returns:** New Assay instance

**Example:**
```python
# Keep first 100 features
subset_assay = assay.subset(np.arange(100))

# Keep specific features
subset_assay = assay.subset([0, 5, 10, 15])
```

---

### ScpMatrix

Physical storage unit with mask-based provenance tracking.

```python
@dataclass
class ScpMatrix:
    X: Union[np.ndarray, sp.spmatrix]           # Quantitative values
    M: Union[np.ndarray, sp.spmatrix, None]     # Mask codes
    metadata: Optional[MatrixMetadata]          # Quality metadata
```

**Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `X` | `np.ndarray` or `sp.spmatrix` | ✅ | Quantitative values (dtype: float64/float32). Shape: (n_samples, n_features) |
| `M` | `np.ndarray`, `sp.spmatrix`, or `None` | ❌ | Mask codes (dtype: int8). Shape must match X. Default: None (all valid) |
| `metadata` | `MatrixMetadata` | ❌ | Additional quality scores. Default: None |

**Mask Codes (MaskCode Enum):**

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

#### `get_m()`

Return the mask matrix. If M is None, returns a zero matrix.

```python
def get_m(self) -> Union[np.ndarray, sp.spmatrix]
```

---

#### `copy()`

Deep copy of the matrix.

```python
def copy(self) -> ScpMatrix
```

---

### ProvenanceLog

Operation history entry for reproducibility tracking.

```python
@dataclass
class ProvenanceLog:
    timestamp: str                    # ISO format timestamp
    action: str                       # Operation name
    params: Dict[str, Any]            # Operation parameters
    software_version: Optional[str]   # ScpTensor version
    description: Optional[str]        # Human-readable description
```

**Example:**
```python
ProvenanceLog(
    timestamp="2025-01-05T14:30:00",
    action="log_normalize",
    params={"base": 2.0, "offset": 1.0},
    software_version="0.1.0-beta",
    description="Log2 transform"
)
```

---

## 2. Normalization Module

Data normalization and transformation methods.

### Import

```python
from scptensor.normalization import (
    log_normalize,
    sample_median_normalization,
    sample_mean_normalization,
    global_median_normalization,
    tmm_normalization,
    upper_quartile_normalization
)
```

---

### log_normalize

Apply logarithmic transformation to data.

```python
def log_normalize(
    container: ScpContainer,
    assay_name: str,
    base_layer: str,
    new_layer_name: str = "log",
    base: float = 2.0,
    offset: float = 1.0
) -> ScpContainer
```

**Parameters:**
- `container`: Input container
- `assay_name`: Target assay name
- `base_layer`: Source layer name
- `new_layer_name`: New layer name. Default: "log"
- `base`: Logarithm base. Default: 2.0 (log2)
- `offset`: Additive offset (to avoid log(0)). Default: 1.0

**Returns:** Container with new normalized layer

**Formula:** `log_base(X + offset)`

**Raises:**
- `LayerNotFoundError`: If base_layer doesn't exist
- `ValueError`: If data contains negative values

**Example:**
```python
container = log_normalize(
    container,
    assay_name="proteins",
    base_layer="raw",
    new_layer_name="log",
    base=2.0,
    offset=1.0
)
```

---

### sample_median_normalization

Median centering per sample (row-wise median subtraction).

```python
def sample_median_normalization(
    container: ScpContainer,
    assay_name: str,
    base_layer: str,
    new_layer_name: str = "median_centered"
) -> ScpContainer
```

**Parameters:**
- `container`: Input container
- `assay_name`: Target assay name
- `base_layer`: Source layer name
- `new_layer_name`: New layer name. Default: "median_centered"

**Returns:** Container with normalized layer

**Formula:** `X - median(X, axis=1, keepdims=True)`

---

### tmm_normalization

TMM (Trimmed Mean of M-values) normalization for between-sample scaling.

```python
def tmm_normalization(
    container: ScpContainer,
    assay_name: str,
    base_layer: str,
    new_layer_name: str = "tmm",
    ref_sample_index: Optional[int] = None
) -> ScpContainer
```

**Parameters:**
- `container`: Input container
- `assay_name`: Target assay name
- `base_layer`: Source layer name
- `new_layer_name`: New layer name. Default: "tmm"
- `ref_sample_index`: Reference sample index. Default: None (auto-select)

**Returns:** Container with TMM-normalized layer

**Reference:** Robinson & Oshlack (2010) Genome Biology 11:R25

---

## 3. Imputation Module

Missing value imputation methods.

### Import

```python
from scptensor.impute import knn, ppca, svd
```

---

### knn

K-nearest neighbors imputation.

```python
def knn(
    container: ScpContainer,
    assay_name: str,
    base_layer: str,
    new_layer_name: str = "imputed",
    k: int = 5,
    weights: str = "uniform",
    metric: str = "euclidean"
) -> ScpContainer
```

**Parameters:**
- `container`: Input container
- `assay_name`: Target assay name
- `base_layer`: Source layer name (must have mask M)
- `new_layer_name`: New layer name. Default: "imputed"
- `k`: Number of nearest neighbors. Default: 5
- `weights`: 'uniform' or 'distance'. Default: 'uniform'
- `metric`: Distance metric ('euclidean', 'manhattan', 'cosine'). Default: 'euclidean'

**Returns:** Container with imputed layer

**Behavior:**
- Only imputes values where mask M is non-zero
- Sets mask M to 0 (VALID) for imputed values
- **Note:** Also updates mask to IMPUTED (5) for imputed values

**Example:**
```python
container = knn(
    container,
    assay_name="proteins",
    base_layer="log",
    new_layer_name="imputed",
    k=5,
    metric="euclidean"
)
```

---

### ppca

Probabilistic PCA imputation.

```python
def ppca(
    container: ScpContainer,
    assay_name: str,
    base_layer: str,
    new_layer_name: str = "imputed",
    n_components: int = 10,
    max_iter: int = 100,
    tol: float = 1e-6
) -> ScpContainer
```

**Parameters:**
- `container`: Input container
- `assay_name`: Target assay name
- `base_layer`: Source layer name
- `new_layer_name`: New layer name. Default: "imputed"
- `n_components`: Number of principal components. Default: 10
- `max_iter`: Maximum EM iterations. Default: 100
- `tol`: Convergence tolerance. Default: 1e-6

**Returns:** Container with imputed layer

**Reference:** Bishop (1999)

---

### svd

SVD-based imputation via low-rank approximation.

```python
def svd(
    container: ScpContainer,
    assay_name: str,
    base_layer: str,
    new_layer_name: str = "imputed",
    rank: int = 10
) -> ScpContainer
```

**Parameters:**
- `container`: Input container
- `assay_name`: Target assay name
- `base_layer`: Source layer name
- `new_layer_name`: New layer name. Default: "imputed"
- `rank`: Rank for SVD (number of singular values to keep). Default: 10

**Returns:** Container with imputed layer

---

## 4. Integration Module

Batch effect correction methods.

### Import

```python
from scptensor.integration import combat, harmony
```

---

### combat

ComBat batch correction (Empirical Bayes).

```python
def combat(
    container: ScpContainer,
    batch_key: str,
    assay_name: str,
    base_layer: str,
    new_layer_name: str = "corrected",
    covariate_keys: Optional[List[str]] = None,
    parametric: bool = True
) -> ScpContainer
```

**Parameters:**
- `container`: Input container
- `batch_key`: Column name in `obs` containing batch labels
- `assay_name`: Target assay name
- `base_layer`: Source layer name
- `new_layer_name`: New layer name. Default: "corrected"
- `covariate_keys`: List of covariate columns to preserve (biological variables). Default: None
- `parametric`: Use parametric adjustment. Default: True

**Returns:** Container with batch-corrected layer

**Requirements:**
- `obs[batch_key]` must exist
- Data should be normalized and log-transformed first

**Reference:** Johnson et al. (2007) Biostatistics 8:118-127

**Example:**
```python
container = combat(
    container,
    batch_key="batch",
    assay_name="proteins",
    base_layer="imputed",
    new_layer_name="corrected"
)
```

---

### harmony

Harmony integration (iterative clustering-based correction).

```python
def harmony(
    container: ScpContainer,
    batch_key: str,
    assay_name: str,
    base_layer: str,
    new_layer_name: str = "harmonized",
    n_pcs: int = 50,
    theta: float = 2.0,
    max_iter: int = 20
) -> ScpContainer
```

**Parameters:**
- `container`: Input container
- `batch_key`: Column name in `obs` containing batch labels
- `assay_name`: Target assay name (usually PCA embeddings)
- `base_layer`: Source layer name
- `new_layer_name`: New layer name. Default: "harmonized"
- `n_pcs`: Number of principal components to use. Default: 50
- `theta`: Clustering penalty (0-3, higher = more correction). Default: 2.0
- `max_iter`: Maximum iterations. Default: 20

**Returns:** Container with harmonized layer

**Reference:** Korsunsky et al. (2019) Nature Methods 16:1289-1300

---

## 5. QC Module

Quality control and outlier detection.

### Import

```python
from scptensor.qc import basic_qc, outlier_detection
```

---

### basic_qc

Basic quality control metrics.

```python
def basic_qc(
    container: ScpContainer,
    assay_name: str,
    layer_name: str = "raw",
    completeness_threshold: float = 0.5
) -> ScpContainer
```

**Parameters:**
- `container`: Input container
- `assay_name`: Target assay name
- `layer_name`: Layer to analyze. Default: "raw"
- `completeness_threshold`: Threshold for flagging low-quality. Default: 0.5

**Returns:** Container with updated `obs` and `var`

**Added Columns:**
- `obs['completeness']`: Fraction of valid values per sample
- `obs['n_detected']`: Number of detected features per sample
- `obs['is_low_quality']`: Boolean flag for samples below threshold
- `var['completeness']`: Fraction of valid values per feature
- `var['n_samples_detected']`: Number of samples detecting each feature

---

### outlier_detection

Statistical outlier detection.

```python
def outlier_detection(
    container: ScpContainer,
    assay_name: str,
    layer_name: str = "raw",
    method: str = "isolation_forest",
    contamination: float = 0.1
) -> ScpContainer
```

**Parameters:**
- `container`: Input container
- `assay_name`: Target assay name
- `layer_name`: Layer to analyze. Default: "raw"
- `method`: Detection method ('isolation_forest', 'local_outlier_factor', 'robust_covariance'). Default: 'isolation_forest'
- `contamination`: Expected outlier proportion. Default: 0.1

**Returns:** Container with updated `obs`

**Added Columns:**
- `obs['is_outlier']`: Boolean flag for outlier samples

---

## 6. Dimensionality Reduction Module

Dimensionality reduction methods.

### Import

```python
from scptensor.dim_reduction import pca, umap
```

---

### pca

Principal Component Analysis.

```python
def pca(
    container: ScpContainer,
    assay_name: str,
    base_layer_name: str,
    new_assay_name: str = "pca",
    n_components: int = 50,
    use_highly_variable: bool = False
) -> ScpContainer
```

**Parameters:**
- `container`: Input container
- `assay_name`: Source assay name
- `base_layer_name`: Source layer name
- `new_assay_name`: New assay name for PCA results. Default: "pca"
- `n_components`: Number of principal components. Default: 50
- `use_highly_variable`: Use only highly variable features. Default: False

**Returns:** Container with new PCA assay

**New Assay Structure:**
- `assay.layers['scores']`: PC coordinates (n_samples × n_components)
- `assay.var['explained_variance']`: Variance explained by each PC
- `assay.var['pc_loadings']`: Feature loadings for each PC

---

### umap

UMAP non-linear dimensionality reduction.

```python
def umap(
    container: ScpContainer,
    assay_name: str,
    base_layer: str,
    new_assay_name: str = "umap",
    n_neighbors: int = 30,
    min_dist: float = 0.3,
    n_components: int = 2
) -> ScpContainer
```

**Parameters:**
- `container`: Input container
- `assay_name`: Source assay name (usually PCA)
- `base_layer`: Source layer name
- `new_assay_name`: New assay name for UMAP results. Default: "umap"
- `n_neighbors`: Neighbors for manifold approximation (controls local/global balance). Default: 30
- `min_dist`: Minimum distance between points (controls clustering). Default: 0.3
- `n_components`: Output dimension. Default: 2

**Returns:** Container with new UMAP assay

**Reference:** McInnes et al. (2018) arXiv:1802.03426

---

## 7. Clustering Module

Clustering algorithms.

### Import

```python
from scptensor.cluster import run_kmeans, run_leiden
```

---

### run_kmeans

K-means clustering.

```python
def run_kmeans(
    container: ScpContainer,
    assay_name: str,
    base_layer: str,
    n_clusters: int,
    key_added: str = "kmeans_cluster",
    random_state: int = 0
) -> ScpContainer
```

**Parameters:**
- `container`: Input container
- `assay_name`: Assay containing embeddings (usually PCA)
- `base_layer`: Layer to cluster on
- `n_clusters`: Number of clusters
- `key_added`: Column name in `obs` for cluster labels. Default: "kmeans_cluster"
- `random_state`: Random seed. Default: 0

**Returns:** Container with updated `obs`

**Added Columns:**
- `obs[key_added]`: Cluster labels (0, 1, 2, ..., n_clusters-1)

---

### run_leiden

Leiden graph clustering.

```python
def run_leiden(
    container: ScpContainer,
    assay_name: str,
    base_layer: str,
    resolution: float = 1.0,
    key_added: str = "leiden_cluster"
) -> ScpContainer
```

**Parameters:**
- `container`: Input container
- `assay_name`: Assay containing embeddings
- `base_layer`: Layer to cluster on
- `resolution`: Resolution parameter (higher = more clusters). Default: 1.0
- `key_added`: Column name in `obs` for cluster labels. Default: "leiden_cluster"

**Returns:** Container with updated `obs`

**Reference:** Traag et al. (2019) Scientific Reports 9:5233

---

## 8. Visualization Module

Publication-quality visualization tools.

### Import

```python
from scptensor.viz import (
    qc_completeness,
    qc_matrix_spy,
    embedding,
    heatmap
)
```

**Style:** All plots use SciencePlots with `plt.style.use(["science", "no-latex"])`

---

### embedding

Plot dimensionality reduction embeddings (PCA/UMAP/t-SNE).

```python
def embedding(
    container: ScpContainer,
    basis: str,
    color: str,
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> plt.Axes
```

**Parameters:**
- `container`: Input container
- `basis`: Embedding name ('pca', 'umap', 'tsne')
- `color`: Column in `obs` to color by
- `ax`: Matplotlib Axes (optional). Default: None (creates new figure)
- `**kwargs`: Additional arguments for `plt.scatter`

**Returns:** Matplotlib Axes object

**Example:**
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot UMAP colored by group
embedding(container, basis='umap', color='group', ax=axes[0])

# Plot UMAP colored by batch
embedding(container, basis='umap', color='batch', ax=axes[1])

plt.tight_layout()
plt.savefig('umap_comparison.png', dpi=300)
```

---

### heatmap

Plot expression heatmap.

```python
def heatmap(
    container: ScpContainer,
    assay_name: str,
    layer_name: str,
    feature_ids: List[str],
    sample_order: Optional[List[str]] = None,
    cmap: str = "viridis",
    **kwargs
) -> plt.Axes
```

**Parameters:**
- `container`: Input container
- `assay_name`: Assay name
- `layer_name`: Layer name
- `feature_ids`: List of feature IDs to plot
- `sample_order`: Optional sample ordering
- `cmap`: Colormap name. Default: "viridis"
- `**kwargs`: Additional arguments for `seaborn.heatmap`

**Returns:** Matplotlib Axes object

---

### qc_completeness

Plot data completeness by group.

```python
def qc_completeness(
    container: ScpContainer,
    layer: str = "raw",
    group_by: Optional[str] = None
) -> plt.Axes
```

**Parameters:**
- `container`: Input container
- `layer`: Layer to analyze. Default: "raw"
- `group_by`: Column in `obs` to group by. Default: None

**Returns:** Matplotlib Axes object

---

### qc_matrix_spy

Spy plot of missing value patterns.

```python
def qc_matrix_spy(
    container: ScpContainer,
    assay_name: str,
    layer_name: str = "raw"
) -> plt.Axes
```

**Parameters:**
- `container`: Input container
- `assay_name`: Assay name
- `layer_name`: Layer to visualize. Default: "raw"

**Returns:** Matplotlib Axes object

---

## 9. Utils Module

Helper utility functions.

### Import

```python
from scptensor.utils import (
    get_highly_variable_features,
    filter_samples,
    filter_features
)
```

---

### get_highly_variable_features

Identify highly variable features.

```python
def get_highly_variable_features(
    container: ScpContainer,
    assay_name: str,
    layer_name: str,
    n_top_features: int = 2000,
    min_mean: float = 0.0125,
    max_mean: float = 3.0,
    min_disp: float = 0.5
) -> List[str]
```

**Parameters:**
- `container`: Input container
- `assay_name`: Assay name
- `layer_name`: Layer name
- `n_top_features`: Number of top features to return. Default: 2000
- `min_mean`: Minimum mean expression. Default: 0.0125
- `max_mean`: Maximum mean expression. Default: 3.0
- `min_disp`: Minimum dispersion. Default: 0.5

**Returns:** List of feature IDs

---

### filter_samples

Filter samples by metadata.

```python
def filter_samples(
    container: ScpContainer,
    sample_ids: List[str]
) -> ScpContainer
```

**Parameters:**
- `container`: Input container
- `sample_ids`: List of sample IDs to keep

**Returns:** Container with subset of samples

---

### filter_features

Filter features in an assay.

```python
def filter_features(
    container: ScpContainer,
    assay_name: str,
    feature_ids: List[str]
) -> ScpContainer
```

**Parameters:**
- `container`: Input container
- `assay_name`: Assay to filter
- `feature_ids`: List of feature IDs to keep

**Returns:** Container with subset of features

---

## 10. Exceptions

Exception hierarchy for error handling.

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

**Usage:**
```python
from scptensor.core import AssayNotFoundError

try:
    result = combat(container, batch_key="batch", assay_name="proteins", ...)
except AssayNotFoundError as e:
    print(f"Error: {e}")
```

---

## Type Annotation Quick Reference

```python
from typing import Dict, List, Optional, Union
import numpy as np
import polars as pl
import scipy.sparse as sp

# Core types
ScpContainer
Assay
ScpMatrix
ProvenanceLog
MaskCode

# Common type aliases
FeatureIDs = List[str]
SampleIDs = List[str]
LayerName = str
AssayName = str
BatchKey = str

# Matrix types
DenseMatrix = np.ndarray
SparseMatrix = sp.spmatrix
AnyMatrix = Union[np.ndarray, sp.spmatrix]

# Metadata types
ObsDataFrame = pl.DataFrame  # Sample metadata
VarDataFrame = pl.DataFrame  # Feature metadata
```

---

## Performance Notes

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| `ScpContainer.__init__` | O(n) | n = number of assays |
| `Assay.subset` | O(k*m) | k = features, m = samples |
| `log_normalize` | O(n*m) | n*m = data size |
| `knn` | O(n²*m) | Use sparse matrices when possible |
| `pca` | O(min(n,m)² * max(n,m)) | Uses randomized SVD for large datasets |
| `umap` | O(n log n) | n = number of samples |

**Optimization Tips:**
1. Use sparse matrices (`scipy.sparse.csr_matrix`) for data with >50% missing values
2. Normalize before imputing (reduces numerical errors)
3. Use PCA before UMAP (faster, better results)
4. Enable Numba JIT (automatically used in hot loops)

---

## Deprecation Policy

Deprecated APIs will be marked with warnings and removed in the next MAJOR version.

**Currently Deprecated:**
- None (v0.1.0-beta)

**Removed in v0.1.0-beta:**
- Direct submodule imports (e.g., `from scptensor.integration.combat import combat`)

---

## Index

- **A**: [Assay](#assay), [AggregationLink](#aggregationlink), [API_REFERENCE.md](#)
- **C**: [copy](#copydeeptrue)
- **E**: [embedding](#embedding)
- **H**: [heatmap](#heatmap)
- **K**: [knn](#knn), [run_kmeans](#run_kmeans)
- **L**: [log_normalize](#log_normalize), [LayerNotFoundError](#10-exceptions)
- **M**: [MaskCode](#maskcode-enum), [ProvenanceLog](#provenancelog)
- **P**: [pca](#pca), [ppca](#ppca), [ProvenanceLog](#provenancelog)
- **Q**: [basic_qc](#basic_qc), [qc_completeness](#qc_completeness), [qc_matrix_spy](#qc_matrix_spy)
- **S**: [ScpContainer](#scpcontainer), [ScpMatrix](#scpmatrix), [svd](#svd)
- **U**: [umap](#umap)

---

**Document Version:** 1.0
**API Version:** v0.1.0-beta
**Last Updated:** 2025-01-05
**Maintainer:** ScpTensor Team

**End of API_REFERENCE.md**
