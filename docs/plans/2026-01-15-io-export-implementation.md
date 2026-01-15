# HDF5 Data Export Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a comprehensive HDF5 export system for ScpContainer that preserves all data structures (obs, assays, layers, masks, metadata, history) with round-trip fidelity.

**Architecture:** Modular exporter in `scptensor/io/` with separate functions for serialization of each data type (DataFrame, dense array, sparse matrix, ProvenanceLog). Main entry point is `save_hdf5()` function with optional parameters for selective export.

**Tech Stack:** h5py (HDF5 files), scipy (sparse matrices), polars (DataFrames), numpy (arrays) - all existing dependencies.

---

## Task 1: Create IO Module Structure and Exception Classes

**Files:**
- Create: `scptensor/io/__init__.py`
- Create: `scptensor/io/exceptions.py`
- Test: `tests/test_io_exceptions.py`

**Step 1: Write the failing test**

Create `tests/test_io_exceptions.py`:

```python
"""Tests for IO exception hierarchy."""

import pytest
from scptensor.io.exceptions import IOPasswordError, IOFormatError, IOWriteError


def test_io_format_error_message():
    """Test IOFormatError message formatting."""
    err = IOFormatError("Missing required group: obs")
    assert "Missing required group" in str(err)
    assert isinstance(err, Exception)


def test_io_write_error_message():
    """Test IOWriteError message formatting."""
    err = IOWriteError("Disk full", path="/data/output.h5")
    assert "Disk full" in str(err)
    assert "/data/output.h5" in str(err)


def test_io_password_error():
    """Test IOPasswordError for encrypted files."""
    err = IOPasswordError("File is password protected")
    assert "password" in str(err).lower()


def test_exception_inheritance():
    """Test all IO exceptions inherit from ScpTensorError."""
    from scptensor.core.exceptions import ScpTensorError

    assert issubclass(IOFormatError, ScpTensorError)
    assert issubclass(IOWriteError, ScpTensorError)
    assert issubclass(IOPasswordError, ScpTensorError)
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_io_exceptions.py -v
```

Expected: `ImportError: cannot import name 'IOFormatError'`

**Step 3: Write minimal implementation**

Create `scptensor/io/exceptions.py`:

```python
"""I/O module exception hierarchy."""

from scptensor.core.exceptions import ScpTensorError


class IOPasswordError(ScpTensorError):
    """HDF5 file password protection errors."""

    pass


class IOFormatError(ScpTensorError):
    """File format corruption or version incompatibility."""

    pass


class IOWriteError(ScpTensorError):
    """Write failures (disk space, permissions, etc.)."""

    def __init__(self, message: str, path: str | None = None) -> None:
        self.path = path
        if path:
            message = f"{message}: {path}"
        super().__init__(message)
```

Create `scptensor/io/__init__.py`:

```python
"""Data I/O module for ScpTensor.

Provides export and import functionality for ScpContainer,
supporting HDF5 and Parquet formats with complete data fidelity.
"""

from scptensor.io.exceptions import IOFormatError, IOPasswordError, IOWriteError

__all__ = ["IOFormatError", "IOPasswordError", "IOWriteError"]
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_io_exceptions.py -v
```

Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add scptensor/io/ tests/test_io_exceptions.py
git commit -m "feat(io): add IO module structure and exception hierarchy"
```

---

## Task 2: Implement DataFrame Serialization (obs/var)

**Files:**
- Create: `scptensor/io/serializers.py`
- Modify: `scptensor/io/__init__.py`
- Test: `tests/test_io_serializers.py`

**Step 1: Write the failing test**

Create `tests/test_io_serializers.py`:

```python
"""Tests for data serialization utilities."""

import numpy as np
import polars as pl
import pytest
import tempfile
from pathlib import Path


def test_serialize_dataframe_to_hdf5(sample_obs):
    """Test serializing polars DataFrame to HDF5 group."""
    from scptensor.io.serializers import serialize_dataframe

    import h5py

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.h5"
        with h5py.File(path, "w") as f:
            group = f.create_group("obs")
            serialize_dataframe(sample_obs, group)

        # Verify structure
        with h5py.File(path, "r") as f:
            assert "obs" in f
            assert "_index" in f["obs"]
            assert "batch" in f["obs"]
            assert "group" in f["obs"]
            assert list(f["obs"]["_index"][:]) == ["S1", "S2", "S3", "S4", "S5"]


def test_deserialize_dataframe_from_hdf5():
    """Test deserializing HDF5 group to polars DataFrame."""
    from scptensor.io.serializers import deserialize_dataframe

    import h5py

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.h5"

        # Create test data
        with h5py.File(path, "w") as f:
            group = f.create_group("obs")
            group.create_dataset("_index", data=["S1", "S2"])
            group.create_dataset("batch", data=["A", "B"])
            group.create_dataset("value", data=[1.0, 2.0])

        # Deserialize
        with h5py.File(path, "r") as f:
            df = deserialize_dataframe(f["obs"])

        assert isinstance(df, pl.DataFrame)
        assert df.shape == (2, 3)
        assert df.columns == ["_index", "batch", "value"]
        assert df["_index"].to_list() == ["S1", "S2"]
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_io_serializers.py::test_serialize_dataframe_to_hdf5 -v
```

Expected: `ImportError: cannot import name 'serialize_dataframe'`

**Step 3: Write minimal implementation**

Create `scptensor/io/serializers.py`:

```python
"""Serialization utilities for ScpTensor data structures."""

from __future__ import annotations

import h5py
import numpy as np
import polars as pl
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scptensor.core.structures import ScpMatrix


def serialize_dataframe(df: pl.DataFrame, group: h5py.Group) -> None:
    """Serialize polars DataFrame to HDF5 group.

    Each column becomes a dataset. String columns use variable-length dtype.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to serialize
    group : h5py.Group
        Target HDF5 group
    """
    string_dtype = h5py.string_dtype(encoding="utf-8")

    for col_name in df.columns:
        col_data = df[col_name].to_numpy()

        # Determine dtype
        if col_data.dtype == object or col_data.dtype == str:
            # String column
            dtype = string_dtype
        elif np.issubdtype(col_data.dtype, np.integer):
            dtype = np.int64
        elif np.issubdtype(col_data.dtype, np.floating):
            dtype = np.float64
        elif col_data.dtype == bool:
            dtype = bool
        else:
            dtype = None  # Let h5py infer

        group.create_dataset(col_name, data=col_data, dtype=dtype)


def deserialize_dataframe(group: h5py.Group) -> pl.DataFrame:
    """Deserialize HDF5 group to polars DataFrame.

    Parameters
    ----------
    group : h5py.Group
        Source HDF5 group containing column datasets

    Returns
    -------
    pl.DataFrame
        Reconstructed DataFrame
    """
    data = {}
    for col_name in group.keys():
        dataset = group[col_name]
        data[col_name] = dataset[:]

    return pl.DataFrame(data)


def serialize_dense_matrix(
    X: np.ndarray,
    group: h5py.Group,
    name: str = "X",
    compression: str | None = "gzip",
    compression_level: int = 4,
) -> None:
    """Serialize dense numpy array to HDF5 dataset.

    Parameters
    ----------
    X : np.ndarray
        Dense array to serialize
    group : h5py.Group
        Target HDF5 group
    name : str, default "X"
        Dataset name
    compression : str | None, default "gzip"
        Compression algorithm
    compression_level : int, default 4
        Compression level (0-9)
    """
    group.create_dataset(
        name,
        data=X.astype(np.float32),
        compression=compression,
        compression_opts=compression_level if compression else None,
    )


def serialize_sparse_matrix(
    X: sp.spmatrix,
    group: h5py.Group,
    name: str = "X",
) -> None:
    """Serialize scipy sparse matrix to HDF5 group.

    Stores as three datasets: data, indices, indptr (CSR format).

    Parameters
    ----------
    X : sp.spmatrix
        Sparse matrix to serialize (will be converted to CSR)
    group : h5py.Group
        Target HDF5 group
    name : str, default "X"
        Base name for datasets
    """
    csr = X.tocsr()
    group.create_dataset(f"{name}_data", data=csr.data)
    group.create_dataset(f"{name}_indices", data=csr.indices)
    group.create_dataset(f"{name}_indptr", data=csr.indptr)
    group.attrs[f"{name}_format"] = "csr"
    group.attrs[f"{name}_shape"] = csr.shape
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_io_serializers.py -v
```

Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add scptensor/io/serializers.py scptensor/io/__init__.py tests/test_io_serializers.py
git commit -m "feat(io): implement DataFrame and matrix serialization utilities"
```

---

## Task 3: Implement ProvenanceLog Serialization

**Files:**
- Modify: `scptensor/io/serializers.py`
- Test: `tests/test_io_serializers.py`

**Step 1: Write the failing test**

Add to `tests/test_io_serializers.py`:

```python
def test_serialize_provenance_log(sample_container):
    """Test serializing operation history."""
    from scptensor.io.serializers import serialize_provenance, deserialize_provenance

    import h5py

    # Add some history
    sample_container.log_operation("test_op", {"n": 5}, "Test operation")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.h5"
        with h5py.File(path, "w") as f:
            group = f.create_group("provenance")
            serialize_provenance(sample_container.history, group)

        # Deserialize and verify
        with h5py.File(path, "r") as f:
            history = deserialize_provenance(f["provenance"])

        assert len(history) == 1
        assert history[0].action == "test_op"
        assert history[0].params == {"n": 5}
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_io_serializers.py::test_serialize_provenance_log -v
```

Expected: `ImportError: cannot import name 'serialize_provenance'`

**Step 3: Write implementation**

Add to `scptensor/io/serializers.py`:

```python
import json
from scptensor.core.structures import ProvenanceLog


def serialize_provenance(
    history: list[ProvenanceLog],
    group: h5py.Group,
) -> None:
    """Serialize operation history to HDF5 group.

    Parameters
    ----------
    history : list[ProvenanceLog]
        List of provenance logs
    group : h5py.Group
        Target HDF5 group
    """
    if not history:
        return

    actions = [log.action for log in history]
    timestamps = [log.timestamp for log in history]
    params_json = [json.dumps(log.params) for log in history]
    versions = [log.software_version or "" for log in history]
    descriptions = [log.description or "" for log in history]

    string_dtype = h5py.string_dtype(encoding="utf-8")

    group.create_dataset("action", data=actions, dtype=string_dtype)
    group.create_dataset("timestamp", data=timestamps, dtype=string_dtype)
    group.create_dataset("params", data=params_json, dtype=string_dtype)
    group.create_dataset("software_version", data=versions, dtype=string_dtype)
    group.create_dataset("description", data=descriptions, dtype=string_dtype)


def deserialize_provenance(group: h5py.Group) -> list[ProvenanceLog]:
    """Deserialize operation history from HDF5 group.

    Parameters
    ----------
    group : h5py.Group
        Source HDF5 group

    Returns
    -------
    list[ProvenanceLog]
        Reconstructed history
    """
    if "action" not in group:
        return []

    n = len(group["action"])
    history = []

    for i in range(n):
        log = ProvenanceLog(
            timestamp=group["timestamp"][i],
            action=group["action"][i],
            params=json.loads(group["params"][i]),
            software_version=group["software_version"][i] or None,
            description=group["description"][i] or None,
        )
        history.append(log)

    return history
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_io_serializers.py::test_serialize_provenance_log -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add scptensor/io/serializers.py tests/test_io_serializers.py
git commit -m "feat(io): implement ProvenanceLog serialization"
```

---

## Task 4: Implement Main save_hdf5() Function (Basic)

**Files:**
- Create: `scptensor/io/exporters.py`
- Modify: `scptensor/io/__init__.py`
- Test: `tests/test_io_export.py`

**Step 1: Write the failing test**

Create `tests/test_io_export.py`:

```python
"""Tests for HDF5 export functionality."""

import numpy as np
import polars as pl
import pytest
import tempfile
from pathlib import Path


def test_save_hdf5_basic(sample_container):
    """Test basic HDF5 save functionality."""
    from scptensor.io.exporters import save_hdf5

    import h5py

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "output.h5"
        save_hdf5(sample_container, path)

        # Verify file was created
        assert path.exists()

        # Verify structure
        with h5py.File(path, "r") as f:
            assert "format_version" in f.attrs
            assert "obs" in f
            assert "assays" in f
            assert "proteins" in f["assays"]


def test_save_hdf5_with_overwrite(sample_container):
    """Test overwrite parameter."""
    from scptensor.io.exporters import save_hdf5
    from scptensor.io.exceptions import IOWriteError

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "output.h5"
        save_hdf5(sample_container, path)

        # Should fail without overwrite
        with pytest.raises(IOWriteError):
            save_hdf5(sample_container, path, overwrite=False)

        # Should succeed with overwrite
        save_hdf5(sample_container, path, overwrite=True)


def test_save_hdf5_preserves_obs(sample_container):
    """Test that obs metadata is preserved."""
    from scptensor.io.exporters import save_hdf5
    from scptensor.io.serializers import deserialize_dataframe

    import h5py

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "output.h5"
        save_hdf5(sample_container, path)

        with h5py.File(path, "r") as f:
            obs_loaded = deserialize_dataframe(f["obs"])

        assert obs_loaded.shape == sample_container.obs.shape
        assert set(obs_loaded.columns) == set(sample_container.obs.columns)
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_io_export.py::test_save_hdf5_basic -v
```

Expected: `ImportError: cannot import name 'save_hdf5'`

**Step 3: Write minimal implementation**

Create `scptensor/io/exporters.py`:

```python
"""HDF5 export functionality for ScpContainer."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import h5py

from scptensor.core.structures import ScpContainer
from scptensor.io.exceptions import IOWriteError
from scptensor.io.serializers import (
    deserialize_dataframe,
    serialize_dataframe,
    serialize_dense_matrix,
    serialize_provenance,
    serialize_sparse_matrix,
)

if TYPE_CHECKING:
    from typing import TYPE_CHECKING

FORMAT_VERSION = "1.0"


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
    save_obs : bool, default True
        Whether to save sample metadata
    save_assays : list[str] | None, default None
        Assays to save, None = all
    save_layers : list[str] | None, default None
        Layers to save per assay, None = all
    save_history : bool, default True
        Whether to save operation history
    overwrite : bool, default False
        Whether to overwrite existing file

    Raises
    ------
    IOWriteError
        If file exists and overwrite=False
    """
    path = Path(path)

    # Check file existence
    if path.exists() and not overwrite:
        raise IOWriteError("File already exists", path=str(path))

    # Determine which assays to save
    assays_to_save = save_assays if save_assays is not None else list(container.assays.keys())

    # Create file
    with h5py.File(path, "w") as f:
        # Write attributes
        f.attrs["format_version"] = FORMAT_VERSION
        f.attrs["scptensor_version"] = "0.2.0"
        f.attrs["creation_time"] = datetime.now().isoformat()

        # Save obs
        if save_obs:
            obs_group = f.create_group("obs")
            serialize_dataframe(container.obs, obs_group)

        # Save assays
        assays_group = f.create_group("assays")
        for assay_name in assays_to_save:
            if assay_name not in container.assays:
                continue
            _save_assay(
                container.assays[assay_name],
                assays_group.create_group(assay_name),
                save_layers=save_layers,
                compression=compression,
                compression_level=compression_level,
            )

        # Save history
        if save_history and container.history:
            prov_group = f.create_group("provenance")
            history_group = prov_group.create_group("history")
            serialize_provenance(container.history, history_group)


def _save_assay(
    assay,
    group: h5py.Group,
    save_layers: list[str] | None = None,
    compression: str | None = "gzip",
    compression_level: int = 4,
) -> None:
    """Save a single assay to HDF5 group.

    Parameters
    ----------
    assay : Assay
        Assay object to save
    group : h5py.Group
        Target HDF5 group
    save_layers : list[str] | None
        Layers to save, None = all
    compression : str | None
        Compression algorithm
    compression_level : int
        Compression level
    """
    # Save var (feature metadata)
    var_group = group.create_group("var")
    serialize_dataframe(assay.var, var_group)

    # Save layers
    layers_to_save = save_layers if save_layers is not None else list(assay.layers.keys())
    layers_group = group.create_group("layers")

    for layer_name in layers_to_save:
        if layer_name not in assay.layers:
            continue

        layer_group = layers_group.create_group(layer_name)
        matrix = assay.layers[layer_name]

        # Save X
        from scipy import sparse

        if sparse.issparse(matrix.X):
            serialize_sparse_matrix(matrix.X, layer_group, "X")
        else:
            serialize_dense_matrix(matrix.X, layer_group, "X", compression, compression_level)

        # Save M if present
        if matrix.M is not None:
            if sparse.issparse(matrix.M):
                serialize_sparse_matrix(matrix.M, layer_group, "M")
            else:
                serialize_dense_matrix(matrix.M, layer_group, "M", compression=None)
```

Update `scptensor/io/__init__.py`:

```python
"""Data I/O module for ScpTensor."""

from scptensor.io.exceptions import IOFormatError, IOPasswordError, IOWriteError
from scptensor.io.exporters import save_hdf5

__all__ = [
    "IOFormatError",
    "IOPasswordError",
    "IOWriteError",
    "save_hdf5",
]
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_io_export.py -v
```

Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add scptensor/io/exporters.py scptensor/io/__init__.py tests/test_io_export.py
git commit -m "feat(io): implement main save_hdf5() function with assay export"
```

---

## Task 5: Implement load_hdf5() Function

**Files:**
- Modify: `scptensor/io/exporters.py` (or create importers.py)
- Modify: `scptensor/io/__init__.py`
- Test: `tests/test_io_import.py`

**Step 1: Write the failing test**

Create `tests/test_io_import.py`:

```python
"""Tests for HDF5 import functionality."""

import pytest
import tempfile
from pathlib import Path


def test_round_trip_container(sample_container):
    """Test complete save/load round trip."""
    from scptensor.io.exporters import save_hdf5
    from scptensor.io.importers import load_hdf5

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "round_trip.h5"
        save_hdf5(sample_container, path)

        loaded = load_hdf5(path)

        # Verify basic properties
        assert loaded.n_samples == sample_container.n_samples
        assert loaded.obs.shape == sample_container.obs.shape
        assert "proteins" in loaded.assays


def test_load_hdf5_invalid_format():
    """Test loading invalid HDF5 file."""
    from scptensor.io.importers import load_hdf5
    from scptensor.io.exceptions import IOFormatError

    import h5py

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "invalid.h5"
        with h5py.File(path, "w") as f:
            f.attrs["format_version"] = "0.0"  # Invalid version
            f.create_group("obs")

        with pytest.raises(IOFormatError):
            load_hdf5(path)
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_io_import.py::test_round_trip_container -v
```

Expected: `ImportError: cannot import name 'load_hdf5'`

**Step 3: Write implementation**

Create `scptensor/io/importers.py`:

```python
"""HDF5 import functionality for ScpContainer."""

from __future__ import annotations

from pathlib import Path

import h5py
from scipy import sparse

from scptensor.core.structures import Assay, ScpContainer, ScpMatrix
from scptensor.io.exceptions import IOFormatError
from scptensor.io.serializers import deserialize_dataframe, deserialize_provenance

FORMAT_VERSION = "1.0"
SUPPORTED_VERSIONS = {"1.0"}


def _is_compatible_version(version: str) -> bool:
    """Check if format version is supported."""
    return version in SUPPORTED_VERSIONS


def _validate_hdf5_structure(file: h5py.File) -> None:
    """Verify HDF5 file is valid ScpTensor format."""
    required_groups = ["obs", "assays"]
    for group in required_groups:
        if group not in file:
            raise IOFormatError(f"Missing required group: {group}")

    version = file.attrs.get("format_version", "0.0")
    if not _is_compatible_version(version):
        raise IOFormatError(f"Version {version} not supported")


def load_hdf5(path: str | Path) -> ScpContainer:
    """
    Load ScpContainer from HDF5 file.

    Parameters
    ----------
    path : str | Path
        Path to HDF5 file

    Returns
    -------
    ScpContainer
        Loaded container

    Raises
    ------
    IOFormatError
        If file format is invalid or incompatible
    """
    path = Path(path)

    with h5py.File(path, "r") as f:
        _validate_hdf5_structure(f)

        # Load obs
        obs = deserialize_dataframe(f["obs"])

        # Load assays
        assays = {}
        for assay_name in f["assays"].keys():
            assay_group = f["assays"][assay_name]
            assays[assay_name] = _load_assay(assay_group)

        # Load history
        history = []
        if "provenance" in f and "history" in f["provenance"]:
            history = deserialize_provenance(f["provenance"]["history"])

    return ScpContainer(obs=obs, assays=assays, history=history)


def _load_assay(group: h5py.Group) -> Assay:
    """Load assay from HDF5 group."""
    # Load var
    var = deserialize_dataframe(group["var"])

    # Load layers
    layers = {}
    if "layers" in group:
        for layer_name in group["layers"].keys():
            layer_group = group["layers"][layer_name]
            layers[layer_name] = _load_matrix(layer_group)

    return Assay(var=var, layers=layers)


def _load_matrix(group: h5py.Group) -> ScpMatrix:
    """Load ScpMatrix from HDF5 group."""
    # Load X
    X = _load_matrix_data(group, "X")

    # Load M (optional)
    M = None
    if "M" in group or "M_data" in group:
        M = _load_matrix_data(group, "M")

    return ScpMatrix(X=X, M=M)


def _load_matrix_data(group: h5py.Group, name: str):
    """Load matrix data (dense or sparse) from HDF5."""
    # Check for sparse format
    if f"{name}_data" in group:
        # CSR sparse format
        data = group[f"{name}_data"][:]
        indices = group[f"{name}_indices"][:]
        indptr = group[f"{name}_indptr"][:]
        shape = tuple(group.attrs.get(f"{name}_shape", (len(indptr) - 1, len(indices))))
        return sparse.csr_matrix((data, indices, indptr), shape=shape)
    elif name in group:
        # Dense format
        return group[name][:]
    else:
        raise ValueError(f"Matrix data '{name}' not found in group")
```

Update `scptensor/io/__init__.py`:

```python
"""Data I/O module for ScpTensor."""

from scptensor.io.exceptions import IOFormatError, IOPasswordError, IOWriteError
from scptensor.io.exporters import save_hdf5
from scptensor.io.importers import load_hdf5

__all__ = [
    "IOFormatError",
    "IOPasswordError",
    "IOWriteError",
    "save_hdf5",
    "load_hdf5",
]
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_io_import.py -v
```

Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add scptensor/io/importers.py scptensor/io/__init__.py tests/test_io_import.py
git commit -m "feat(io): implement load_hdf5() function with format validation"
```

---

## Task 6: Add ScpContainer.save() and .load() Methods

**Files:**
- Modify: `scptensor/core/structures.py`
- Test: `tests/test_io_container_methods.py`

**Step 1: Write the failing test**

Create `tests/test_io_container_methods.py`:

```python
"""Tests for ScpContainer save/load convenience methods."""

import pytest
import tempfile
from pathlib import Path


def test_container_save_method(sample_container):
    """Test ScpContainer.save() method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.h5"
        sample_container.save(path)
        assert path.exists()


def test_container_load_classmethod(sample_container):
    """Test ScpContainer.load() classmethod."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.h5"
        sample_container.save(path)
        loaded = ScpContainer.load(path)

        assert loaded.n_samples == sample_container.n_samples


def test_container_auto_detect_format(sample_container):
    """Test automatic format detection from extension."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # HDF5
        h5_path = Path(tmpdir) / "test.h5"
        sample_container.save(h5_path)
        assert h5_path.exists()

        # Wrong extension should raise error
        bad_path = Path(tmpdir) / "test.unknown"
        with pytest.raises(ValueError):
            sample_container.save(bad_path)
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_io_container_methods.py::test_container_save_method -v
```

Expected: `AttributeError: 'ScpContainer' object has no attribute 'save'`

**Step 3: Write implementation**

Add to `scptensor/core/structures.py` at the end of ScpContainer class (after `log_operation` method):

```python
    def save(
        self,
        path: str | Path,
        *,
        compression: str | None = "gzip",
        compression_level: int = 4,
        overwrite: bool = False,
    ) -> None:
        """Save container to file, auto-detecting format from extension.

        Parameters
        ----------
        path : str | Path
            Output file path. Extension determines format (.h5, .parquet)
        compression : str | None, default "gzip"
            Compression algorithm for HDF5
        compression_level : int, default 4
            Compression level (0-9)
        overwrite : bool, default False
            Whether to overwrite existing file

        Raises
        ------
        ValueError
            If file extension is not recognized
        """
        from scptensor.io import save_hdf5

        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == ".h5" or suffix == ".hdf5":
            save_hdf5(
                self,
                path,
                compression=compression,
                compression_level=compression_level,
                overwrite=overwrite,
            )
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Use .h5")

    @classmethod
    def load(cls, path: str | Path) -> "ScpContainer":
        """Load container from file, auto-detecting format from extension.

        Parameters
        ----------
        path : str | Path
            Input file path. Extension determines format

        Returns
        -------
        ScpContainer
            Loaded container

        Raises
        ------
        ValueError
            If file extension is not recognized
        """
        from scptensor.io import load_hdf5

        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == ".h5" or suffix == ".hdf5":
            return load_hdf5(path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Use .h5")
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_io_container_methods.py -v
```

Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add scptensor/core/structures.py tests/test_io_container_methods.py
git commit -m "feat(io): add ScpContainer.save() and .load() convenience methods"
```

---

## Task 7: Export to scptensor package namespace

**Files:**
- Modify: `scptensor/__init__.py`

**Step 1: Update exports**

Modify `scptensor/__init__.py` to include I/O functions:

```python
# ... existing imports ...

from scptensor.io import load_hdf5, save_hdf5

# ... existing __all__ ...

__all__ = [
    # ... existing exports ...
    "save_hdf5",
    "load_hdf5",
]
```

**Step 2: Verify imports work**

```bash
uv run python -c "from scptensor import save_hdf5, load_hdf5; print('✅ Imports work')"
```

Expected: `✅ Imports work`

**Step 3: Commit**

```bash
git add scptensor/__init__.py
git commit -m "feat(io): export save_hdf5 and load_hdf5 to package namespace"
```

---

## Task 8: Integration and Edge Case Tests

**Files:**
- Test: `tests/test_io_integration.py`

**Step 1: Write comprehensive integration tests**

Create `tests/test_io_integration.py`:

```python
"""Integration tests for I/O functionality."""

import numpy as np
import pytest
import tempfile
from pathlib import Path


def test_full_round_trip_with_masks(sample_container_with_mask):
    """Test round trip preserves mask matrix."""
    from scptensor.io import save_hdf5, load_hdf5

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.h5"
        save_hdf5(sample_container_with_mask, path)

        loaded = load_hdf5(path)

        # Verify mask was preserved
        original_matrix = sample_container_with_mask.assays["proteins"].layers["raw"]
        loaded_matrix = loaded.assays["proteins"].layers["raw"]

        assert loaded_matrix.M is not None
        assert np.array_equal(loaded_matrix.M, original_matrix.M)


def test_round_trip_multi_assay(sample_container_multi_assay):
    """Test round trip with multiple assays."""
    from scptensor.io import save_hdf5, load_hdf5

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.h5"
        save_hdf5(sample_container_multi_assay, path)

        loaded = load_hdf5(path)

        assert set(loaded.assays.keys()) == {"proteins", "peptides"}


def test_selective_assay_export(sample_container_multi_assay):
    """Test exporting only specific assays."""
    from scptensor.io import save_hdf5, load_hdf5

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.h5"
        save_hdf5(sample_container_multi_assay, path, save_assays=["proteins"])

        loaded = load_hdf5(path)

        assert "proteins" in loaded.assays
        assert "peptides" not in loaded.assays


def test_round_trip_with_history(sample_container):
    """Test operation history is preserved."""
    from scptensor.io import save_hdf5, load_hdf5

    sample_container.log_operation("normalize", {"method": "log"}, "Log normalize")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.h5"
        save_hdf5(sample_container, path)

        loaded = load_hdf5(path)

        assert len(loaded.history) == 1
        assert loaded.history[0].action == "normalize"


def test_empty_container_export():
    """Test exporting container with no assays."""
    from scptensor.core import ScpContainer
    from scptensor.io import save_hdf5, load_hdf5

    obs = pl.DataFrame({"_index": ["S1", "S2"]})
    container = ScpContainer(obs=obs)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.h5"
        save_hdf5(container, path)

        loaded = load_hdf5(path)

        assert loaded.n_samples == 2
        assert len(loaded.assays) == 0
```

**Step 2: Run all I/O tests**

```bash
uv run pytest tests/test_io*.py -v
```

Expected: All tests pass

**Step 3: Commit**

```bash
git add tests/test_io_integration.py
git commit -m "test(io): add integration and edge case tests"
```

---

## Task 9: Update Documentation

**Files:**
- Modify: `docs/design/ROADMAP.md`
- Modify: `scptensor/io/__init__.py` (add module docstring)

**Step 1: Update module docstring**

Update `scptensor/io/__init__.py`:

```python
"""Data I/O module for ScpTensor.

This module provides export and import functionality for ScpContainer,
supporting HDF5 format with complete data fidelity.

Main Functions:
    save_hdf5: Export container to HDF5 file
    load_hdf5: Import container from HDF5 file

Example:
    >>> from scptensor import ScpContainer
    >>> container = ScpContainer(obs)
    >>> container.save("analysis.h5")  # One-click export
    >>> loaded = ScpContainer.load("analysis.h5")  # Import
"""
```

**Step 2: Update ROADMAP**

Add to `docs/design/ROADMAP.md`:

```markdown
### 2026-01-15
- **Data I/O Export feature COMPLETED** - HDF5 export with full fidelity
  - save_hdf5()/load_hdf5() functions
  - ScpContainer.save()/.load() convenience methods
  - Complete round-trip preservation (obs, assays, layers, masks, history)
```

**Step 3: Commit**

```bash
git add docs/design/ROADMAP.md scptensor/io/__init__.py
git commit -m "docs(io): update documentation for I/O export feature"
```

---

## Summary

**Total Tasks:** 9
**Estimated Time:** 4-6 person-days
**Files Created:**
- `scptensor/io/__init__.py`
- `scptensor/io/exceptions.py`
- `scptensor/io/serializers.py`
- `scptensor/io/exporters.py`
- `scptensor/io/importers.py`
- `tests/test_io_exceptions.py`
- `tests/test_io_serializers.py`
- `tests/test_io_export.py`
- `tests/test_io_import.py`
- `tests/test_io_container_methods.py`
- `tests/test_io_integration.py`

**Files Modified:**
- `scptensor/core/structures.py` - Added save/load methods
- `scptensor/__init__.py` - Package namespace exports
- `docs/design/ROADMAP.md` - Feature tracking

**Dependencies:** None (all existing)

**Next Phase (Parquet support):** Deferred to v0.3.1
