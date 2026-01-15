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
