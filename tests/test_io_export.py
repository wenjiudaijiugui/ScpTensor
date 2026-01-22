"""Tests for HDF5 export functionality."""

import tempfile
from pathlib import Path

import pytest


def test_save_hdf5_basic(sample_container):
    """Test basic HDF5 save functionality."""
    import h5py

    from scptensor.io.hdf5 import save_hdf5

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
    from scptensor.io.exceptions import IOWriteError
    from scptensor.io.hdf5 import save_hdf5

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
    import h5py

    from scptensor.io.hdf5 import deserialize_dataframe, save_hdf5

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "output.h5"
        save_hdf5(sample_container, path)

        with h5py.File(path, "r") as f:
            obs_loaded = deserialize_dataframe(f["obs"])

        assert obs_loaded.shape == sample_container.obs.shape
        assert set(obs_loaded.columns) == set(sample_container.obs.columns)
