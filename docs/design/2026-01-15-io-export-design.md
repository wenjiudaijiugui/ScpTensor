# ScpTensor Data I/O Export Design

**Date:** 2026-01-15
**Version:** 1.0
**Status:** Draft
**Target Release:** v0.3.0

---

## Overview

Design a comprehensive data export system for ScpTensor that enables seamless interoperability with other single-cell analysis tools (Scanpy, Seurat) while preserving all ScpTensor-specific data structures.

**Primary Goal:** Export ScpContainer to HDF5 and Parquet formats with complete fidelity.

---

## Architecture

### Module Structure

```
scptensor/io/
├── __init__.py              # Public API exports
├── exporters.py             # Exporter base classes and implementations
├── importers.py             # Importer classes (future expansion)
└── converters.py            # Format conversion utilities
```

### Core Components

**1. ContainerExporter** - Main export orchestrator

```python
class ContainerExporter:
    def save_hdf5(path, **kwargs) -> None
    def save_parquet(path, **kwargs) -> None
    def save(path, format="auto", **kwargs) -> None
```

**2. Data Mapping**

| Source | Destination |
|--------|-------------|
| `container.obs` | `/obs` group (HDF5) or `obs.parquet` |
| `container.assays[name].var` | `/assays/{name}/var` |
| `container.assays[name].layers` | `/assays/{name}/layers/{layer_name}` |
| `container.history` | `/provenance/history` |
| Matrix `M` (mask) | Parallel dataset to `X` |

**Design Principles**

1. **Round-trip fidelity** - Exported files reload to identical containers
2. **Incremental** - HDF5 first (scientific standard), Parquet second (metadata optimization)
3. **Extensible** - Architecture supports future AnnData/Seurat converters

---

## HDF5 Storage Format

### File Structure

```
scptensor_data.h5
├── /                           # Root group
│   ├── format_version          # Attribute: "1.0"
│   ├── scptensor_version       # Attribute: "0.2.0"
│   ├── creation_time           # Attribute: ISO 8601 timestamp
│   │
│   ├── /obs                    # Sample metadata (DataFrame)
│   │   ├── _index              # Sample ID array
│   │   ├── batch               # Column: batch info
│   │   ├── group               # Column: group labels
│   │   └── ...                 # Other columns
│   │
│   ├── /assays                 # Assay group
│   │   └── /{assay_name}       # Per-assay structure
│   │       ├── /var            # Feature metadata (DataFrame)
│   │       │   ├── _index
│   │       │   └── ...
│   │       │
│   │       └── /layers         # Data layer group
│   │           └── /{layer_name}
│   │               ├── X       # Data matrix (CSR/CSC or dense)
│   │               ├── M       # Mask matrix (uint8)
│   │       │
│   │       └── /metadata       # MatrixMetadata
│   │           ├── quality_scores
│   │           └── detection_stats
│   │
│   └── /provenance             # Operation history
│       └── /history
│           ├── operation       # Operation name array
│           ├── params          # JSON-serialized parameters
│           └── timestamp       # Timestamp array
```

### Data Type Mapping

| ScpTensor Type | HDF5 Storage Type | Notes |
|----------------|-------------------|-------|
| `pl.DataFrame` | Group + fixed-length arrays | One dataset per column |
| `np.ndarray` (dense) | Dataset with float32 | Compressed: gzip(level=4) |
| `sp.spmatrix` (CSR) | 3 datasets: data/indices/indptr | scipy.sparse format |
| `str` | Variable-length string | h5py.string_dtype() |
| `ProvenanceLog` | Table structure | Flattened by field |

### Compression Strategy

- **Sparse matrices** (>70% zeros): CSR format, no compression
- **Dense matrices**: gzip level=4 (speed/ratio balance)
- **String columns**: No compression (small data overhead)

---

## API Design

### Main Function

```python
def save_hdf5(
    container: ScpContainer,
    path: str | Path,
    *,
    compression: str | None = "gzip",
    compression_level: int = 4,
    save_obs: bool = True,
    save_assays: list[str] | None = None,
    save_layers: list[str] | None = None,
    save_history: bool = True,
    overwrite: bool = False,
) -> None:
    """
    Export ScpContainer to HDF5 format.

    Parameters
    ----------
    container : ScpContainer
        Container to export
    path : str | Path
        Output file path (.h5)
    compression : str | None, default "gzip"
        Compression algorithm (gzip/lzf/blosc/None)
    compression_level : int, default 4
        Compression level (0-9)
    save_assays : list[str] | None, default None
        Assays to save, None = all
    save_layers : list[str] | None, default None
        Layers to save, None = all
    save_history : bool, default True
        Whether to save operation history
    overwrite : bool, default False
        Whether to overwrite existing file
    """
```

### Convenience API

```python
# Extend ScpContainer class
class ScpContainer:
    def save(self, path: str | Path, **kwargs) -> None:
        """Smart save: auto-detect format from extension (.h5/.parquet)"""

    @classmethod
    def load(cls, path: str | Path) -> "ScpContainer":
        """Smart load: auto-detect format from extension"""
```

### Usage Examples

```python
# Basic: one-click save
container.save("analysis_results.h5")

# Advanced: selective export
save_hdf5(
    container,
    "results.h5",
    save_assays=["proteins"],
    save_layers=["X", "imputed"],
    compression="gzip",
    compression_level=6
)

# Reload
restored = ScpContainer.load("analysis_results.h5")
```

---

## Error Handling

### Exception Hierarchy

```python
class IOPasswordError(ScpTensorError):
    """HDF5 file password protection errors"""

class IOFormatError(ScpTensorError):
    """File format corruption or version incompatibility"""

class IOWriteError(ScpTensorError):
    """Write failures (disk space, permissions, etc.)"""
```

### Edge Cases

| Scenario | Handling |
|----------|----------|
| File exists | Default: error; `overwrite=True`: overwrite |
| Disk full | Pre-check + write-time capture with clear error |
| Assay/layers missing | Skip and warn when `save_assays` specifies non-existent |
| Empty container | Allow save (valid structure), load restores empty |
| Unsupported type | Convert or raise `TypeError` |
| Version mismatch | Check `format_version`, support migration |

### Validation

```python
def _validate_hdf5_structure(file: h5py.File) -> None:
    """Verify HDF5 file is valid ScpTensor format."""
    required_groups = ["obs", "assays"]
    for group in required_groups:
        if group not in file:
            raise IOFormatError(f"Missing required group: {group}")

    version = file.attrs.get("format_version", "0.0")
    if not _is_compatible_version(version):
        raise IOFormatError(f"Version {version} not supported")
```

---

## Implementation Plan

### Phases

| Phase | Tasks | Priority |
|-------|-------|----------|
| **1** | HDF5 exporter basic functionality | P0 |
| | - `save_hdf5()` implementation | P0 |
| | - Sparse/dense matrix serialization | P0 |
| | - obs/var DataFrame serialization | P0 |
| **2** | Complete data support | P1 |
| | - Mask matrix M save/load | P1 |
| | - Multi-layer structure | P1 |
| | - ProvenanceLog serialization | P1 |
| **3** | Loader implementation | P1 |
| | - `load_hdf5()` function | P1 |
| | - Version compatibility check | P1 |
| **4** | Convenience API | P2 |
| | - `ScpContainer.save()` method | P2 |
| | - `ScpContainer.load()` classmethod | P2 |
| **5** | Parquet support | P2 |
| | - `save_parquet()` / `load_parquet()` | P2 |

### Dependencies

| Package | Purpose | Status |
|---------|---------|--------|
| `h5py` | HDF5 file operations | Existing |
| `pyarrow` | Parquet support | Existing (polars dep) |
| `scipy` | Sparse matrix operations | Existing |
| `polars` | DataFrame operations | Existing |

**No new dependencies required.**

### File Manifest

```
New files:
├── scptensor/io/__init__.py
├── scptensor/io/exporters.py
├── scptensor/io/importers.py
├── tests/test_io_export.py
└── tests/test_io_import.py

Modified files:
├── scptensor/core/structures.py  # Add save/load methods
└── scptensor/__init__.py          # Export public API
```

---

## Test Coverage

| Category | Cases |
|----------|-------|
| Unit | Each data type serialization/deserialization |
| Integration | Full container save/load round-trip |
| Boundary | Empty container, large files, special characters |
| Error | Permission errors, corrupted files, version mismatch |
| Compatibility | Cross-version file read/write |

---

## Open Questions

1. Should we support chunked writing for extremely large datasets (>10GB)?
2. Should export include a "compatibility mode" for older ScpTensor versions?
3. Should Parquet export store sparse matrices in a denormalized long format?

---

**End of Design Document**
