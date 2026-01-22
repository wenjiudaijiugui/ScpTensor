"""Tests for data serialization utilities."""

import tempfile
from pathlib import Path

import polars as pl


def test_serialize_dataframe_to_hdf5(sample_obs):
    """Test serializing polars DataFrame to HDF5 group."""
    import h5py

    from scptensor.io.hdf5 import deserialize_dataframe, serialize_dataframe

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.h5"
        with h5py.File(path, "w") as f:
            group = f.create_group("obs")
            serialize_dataframe(sample_obs, group)

        # Verify structure - check that datasets exist
        with h5py.File(path, "r") as f:
            assert "obs" in f
            assert "_index" in f["obs"]
            assert "batch" in f["obs"]
            assert "group" in f["obs"]
            # Raw HDF5 returns bytes for strings
            raw_index = [x.decode("utf-8") for x in f["obs"]["_index"][:]]
            assert raw_index == ["S1", "S2", "S3", "S4", "S5"]

        # Verify round-trip through deserialize
        with h5py.File(path, "r") as f:
            df_restored = deserialize_dataframe(f["obs"])

        assert df_restored.equals(sample_obs)


def test_deserialize_dataframe_from_hdf5():
    """Test deserializing HDF5 group to polars DataFrame."""
    import h5py

    from scptensor.io.hdf5 import deserialize_dataframe

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


def test_serialize_provenance_log(sample_container):
    """Test serializing operation history."""
    import h5py

    from scptensor.io.hdf5 import deserialize_provenance, serialize_provenance

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
