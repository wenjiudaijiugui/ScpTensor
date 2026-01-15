# ScpTensor API Quick Reference

**Version:** v0.1.0
**Last Updated:** 2025-01-09

---

## Common Import Patterns

### Core Imports
```python
# Data structures
from scptensor import ScpContainer, Assay, ScpMatrix, ProvenanceLog, MaskCode

# I/O utilities
from scptensor import reader, load_csv, save_csv, load_h5ad, save_h5ad, load_npz, save_npz

# Sparse utilities
from scptensor import (
    is_sparse_matrix,
    get_sparsity_ratio,
    to_sparse_if_beneficial,
    sparse_copy,
    cleanup_layers,
    get_memory_usage,
)

# JIT operations (if Numba available)
from scptensor import NUMBA_AVAILABLE, count_mask_codes, find_missing_indices
```

### Module Imports
```python
# Normalization
from scptensor.normalization import (
    log_normalize,
    sample_median_normalization,
    sample_mean_normalization,
    global_median_normalization,
    tmm_normalization,
    upper_quartile_normalization,
)

# Imputation
from scptensor.impute import knn, ppca, svd_impute, missforest

# Integration
from scptensor.integration import combat, harmony, mnn_correct, scanorama_integrate

# Dimensionality Reduction
from scptensor.dim_reduction import pca, umap

# Clustering
from scptensor.cluster import run_kmeans

# Quality Control
from scptensor.qc import basic_qc, detect_outliers, calculate_qc_metrics

# Feature Selection
from scptensor.feature_selection import select_hvg

# Visualization
from scptensor.viz import scatter, heatmap, violin, embedding, qc_completeness
```

---

## Core Data Structures

### Creating a ScpContainer
```python
import numpy as np
import polars as pl
from scptensor import ScpContainer, Assay, ScpMatrix

# Sample data
n_samples, n_features = 100, 500
X = np.random.rand(n_samples, n_features)

# Metadata
obs = pl.DataFrame({
    "_index": [f"S{i}" for i in range(n_samples)],
    "batch": ["A"] * 50 + ["B"] * 50,
    "group": ["Control"] * 50 + ["Treatment"] * 50,
})

var = pl.DataFrame({
    "_index": [f"P{i}" for i in range(n_features)],
})

# Create container
matrix = ScpMatrix(X=X)
assay = Assay(var=var, layers={"raw": matrix})
container = ScpContainer(obs=obs, assays={"proteins": assay})
```

### Working with Layers
```python
# Access layer
X = container.assays["proteins"].layers["raw"].X

# Add new layer (typically done by analysis functions)
new_matrix = ScpMatrix(X=X_normalized)
container.assays["proteins"].layers["normalized"] = new_matrix

# List layers
print(list(container.assays["proteins"].layers.keys()))
```

### Provenance Tracking
```python
# Check operation history
for log_entry in container.history:
    print(f"{log_entry.operation}: {log_entry.timestamp}")

# Access mask codes (0=VALID, 1=MBR, 2=LOD, 3=FILTERED, 5=IMPUTED)
M = container.assays["proteins"].layers["raw"].M
```

---

## Normalization

### Log Transform
```python
from scptensor.normalization import log_normalize

container = log_normalize(
    container,
    assay_name="proteins",
    base_layer="raw",
    new_layer_name="log",
    base=2.0,          # Log base (default: 2)
    offset=1.0,        # Offset to handle zeros
)
```

### Median Centering
```python
from scptensor.normalization import sample_median_normalization

container = sample_median_normalization(
    container,
    assay_name="proteins",
    base_layer="log",
    new_layer_name="median_centered",
)
```

### TMM Normalization
```python
from scptensor.normalization import tmm_normalization

container = tmm_normalization(
    container,
    assay_name="proteins",
    base_layer="log",
    new_layer_name="tmm",
    reference_batch="A",  # Optional: specify reference batch
)
```

---

## Imputation

### KNN Imputation
```python
from scptensor.impute import knn

container = knn(
    container,
    assay_name="proteins",
    base_layer="log",
    new_layer_name="imputed_knn",
    k=5,                      # Number of neighbors
    weights="uniform",        # or "distance"
    batch_size=500,           # For memory control
    oversample_factor=3,      # Search k*3 neighbors, filter to k valid
)
```

### PPCA Imputation
```python
from scptensor.impute import ppca

container = ppca(
    container,
    assay_name="proteins",
    base_layer="log",
    new_layer_name="imputed_ppca",
    n_components=10,          # Number of principal components
    max_iter=100,             # Maximum iterations
    tol=1e-4,                 # Convergence tolerance
)
```

### SVD Imputation
```python
from scptensor.impute import svd_impute

container = svd_impute(
    container,
    assay_name="proteins",
    base_layer="log",
    new_layer_name="imputed_svd",
    rank=10,                  # Rank for SVD approximation
)
```

---

## Batch Correction

### ComBat
```python
from scptensor.integration import combat

container = combat(
    container,
    batch_key="batch",         # Column in obs
    assay_name="proteins",
    base_layer="imputed_knn",
    new_layer_name="corrected",
    covariates=None,          # Optional: list of covariate columns
)
```

### MNN Correction
```python
from scptensor.integration import mnn_correct

container = mnn_correct(
    container,
    batch_key="batch",
    assay_name="proteins",
    base_layer="corrected",
    new_layer_name="mnn_corrected",
    k=20,                     # Number of MNN pairs
    sigma=1.0,                # Bandwidth for kernel
)
```

### Harmony (optional dependency)
```python
from scptensor.integration import harmony

container = harmony(
    container,
    batch_key="batch",
    assay_name="proteins",
    base_layer="corrected",
    new_layer_name="harmony",
    base_layer_key="pca",     # Use PCA results as input
)
```

---

## Dimensionality Reduction

### PCA
```python
from scptensor.dim_reduction import pca

container = pca(
    container,
    assay_name="proteins",
    base_layer="corrected",
    n_components=50,          # Number of PCs
    new_assay_name="pca",
    seed=42,                  # Random seed
)

# Access results
pca_coords = container.assays["pca"].layers["X"].X
variance_ratios = container.assays["pca"].var["variance_ratio"].to_list()
```

### UMAP
```python
from scptensor.dim_reduction import umap

container = umap(
    container,
    assay_name="proteins",
    base_layer="corrected",
    n_components=2,           # Usually 2 for visualization
    new_assay_name="umap",
    n_neighbors=15,           # UMAP parameter
    min_dist=0.1,             # UMAP parameter
    seed=42,
)
```

---

## Clustering

### K-Means
```python
from scptensor.cluster import run_kmeans

container = run_kmeans(
    container,
    assay_name="pca",         # Usually on PCA space
    base_layer="X",
    new_assay_name="cluster_kmeans",
    n_clusters=5,
    key_added="kmeans_cluster",  # Add to obs
    random_state=42,
)

# Access cluster labels
clusters = container.obs["kmeans_cluster"].to_list()
```

---

## Quality Control

### Basic QC
```python
from scptensor.qc import basic_qc

container = basic_qc(
    container,
    assay_name="proteins",
    min_features=200,         # Min features per sample
    min_cells=3,              # Min samples per feature
    detection_threshold=0.0,  # Threshold for detection
)
```

### Outlier Detection
```python
from scptensor.qc import detect_outliers

outliers = detect_outliers(
    container,
    assay_name="proteins",
    layer="log",
    method="isolation",       # or "zscore", "local_outlier_factor"
    contamination=0.1,
)
```

### Calculate QC Metrics
```python
from scptensor.qc import calculate_qc_metrics

container = calculate_qc_metrics(
    container,
    assay_name="proteins",
    layer="log",
)

# Metrics are added to obs
print(container.obs.columns)
```

---

## Feature Selection

### Highly Variable Genes/Proteins
```python
from scptensor.feature_selection import select_hvg

container = select_hvg(
    container,
    assay_name="proteins",
    layer="log",
    n_top_features=2000,
    method="cv",              # or "dispersion"
    subset=True,              # True: filter to HVGs, False: add column
)
```

---

## Visualization

### Static Plots (Matplotlib + SciencePlots)

#### Scatter Plot
```python
from scptensor.viz import scatter
import numpy as np

coords = np.array([[0, 1], [1, 2], [2, 3]])  # N x 2
colors = np.array([0, 1, 0])
ax = scatter(coords, c=colors, title="Scatter Plot")
```

#### Embedding Plot
```python
from scptensor.viz import embedding

ax = embedding(
    container,
    basis="umap",             # or "pca"
    color="batch",            # Column in obs
    show_missing=False,
)
```

#### QC Completeness Plot
```python
from scptensor.viz import qc_completeness

ax = qc_completeness(
    container,
    assay_name="proteins",
    layer="log",
)
```

### Interactive Plots (Plotly)

```python
from scptensor.viz import scatter_plot, save_html

fig = scatter_plot(
    container,
    basis="umap",
    color="batch",
)

save_html(fig, "umap_plot.html")
```

---

## Sparse Matrix Utilities

### Check Sparsity
```python
from scptensor import is_sparse_matrix, get_sparsity_ratio

if is_sparse_matrix(X):
    ratio = get_sparsity_ratio(X)
    print(f"Matrix is {ratio:.1%} sparse")
```

### Auto-Convert to Sparse
```python
from scptensor import to_sparse_if_beneficial

X_sparse = to_sparse_if_beneficial(X, threshold=0.5)
```

### Memory Management
```python
from scptensor import cleanup_layers, get_memory_usage

# Get memory stats
stats = get_memory_usage(X)
print(f"Memory: {stats['nbytes'] / 1024 / 1024:.2f} MB")

# Clean up old layers
cleanup_layers(container, "proteins", keep_layers=["imputed"])
```

---

## Complete Analysis Pipeline

```python
import numpy as np
import polars as pl
from scptensor import ScpContainer, Assay, ScpMatrix

# 1. Load or create data
container = create_container()  # Your data loading function

# 2. Quality Control
from scptensor.qc import basic_qc
container = basic_qc(container, assay_name="proteins")

# 3. Normalization
from scptensor.normalization import log_normalize
container = log_normalize(container, assay_name="proteins")

# 4. Imputation
from scptensor.impute import knn
container = knn(container, assay_name="proteins", layer_name="log")

# 5. Batch Correction
from scptensor.integration import combat
container = combat(container, batch_key="batch", assay_name="proteins")

# 6. Feature Selection
from scptensor.feature_selection import select_hvg
container = select_hvg(container, assay_name="proteins", n_top_features=2000)

# 7. Dimensionality Reduction
from scptensor.dim_reduction import pca, umap
container = pca(container, assay_name="proteins")
container = umap(container, assay_name="proteins")

# 8. Clustering
from scptensor.cluster import run_kmeans
container = run_kmeans(container, assay_name="pca", n_clusters=5)

# 9. Visualization
from scptensor.viz import embedding, qc_completeness
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

embedding(container, basis="umap", color="batch", ax=axes[0])
embedding(container, basis="umap", color="kmeans_cluster", ax=axes[1])

plt.tight_layout()
plt.savefig("analysis_results.png", dpi=300)
```

---

## Error Handling

```python
from scptensor.core.exceptions import (
    ScpTensorError,
    AssayNotFoundError,
    LayerNotFoundError,
    ValidationError,
    DimensionError,
)

try:
    container = log_normalize(container, assay_name="nonexistent")
except AssayNotFoundError as e:
    print(f"Assay not found: {e}")
except ValidationError as e:
    print(f"Validation error: {e}")
```

---

## Working with Masks

```python
from scptensor import MaskCode

# Access mask
M = container.assays["proteins"].layers["raw"].M

# Count mask codes
from scptensor import count_mask_codes
counts = count_mask_codes(M)
print(f"Valid: {counts[0]}, Missing: {counts[1]}, Imputed: {counts[5]}")

# Find missing indices
from scptensor import find_missing_indices
missing_rows, missing_cols = find_missing_indices(M)
```

---

## I/O Operations

```python
from scptensor import load_csv, save_csv, load_h5ad, save_h5ad

# Load from CSV
container = load_csv("data/", assay_name="proteins")

# Save to CSV
save_csv(container, "output/", assay_name="proteins")

# Load AnnData (requires scanpy)
container = load_h5ad("data.h5ad")

# Save to AnnData
save_h5ad(container, "output.h5ad")
```

---

**Document Maintainer:** ScpTensor Team
**Last Updated:** 2025-01-09

For complete API documentation, see `docs/design/API_REFERENCE.md`.
