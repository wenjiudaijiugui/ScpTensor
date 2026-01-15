"""Tests for HDF5 import functionality."""

import tempfile
from pathlib import Path

import pytest


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
    import h5py

    from scptensor.io.exceptions import IOFormatError
    from scptensor.io.importers import load_hdf5

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "invalid.h5"
        with h5py.File(path, "w") as f:
            f.attrs["format_version"] = "0.0"  # Invalid version
            f.create_group("obs")

        with pytest.raises(IOFormatError):
            load_hdf5(path)
