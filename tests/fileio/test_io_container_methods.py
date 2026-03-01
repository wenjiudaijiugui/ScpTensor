"""Tests for ScpContainer save/load convenience methods."""

import tempfile
from pathlib import Path

import pytest

from scptensor import ScpContainer


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
