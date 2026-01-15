"""Integration tests for HDF5 I/O functionality.

Tests comprehensive round-trip scenarios including masks, multi-assay,
selective export, history preservation, and edge cases.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from scipy import sparse

from scptensor import Assay, load_hdf5, save_hdf5, ScpContainer, ProvenanceLog, ScpMatrix


class TestRoundTripWithMasks:
    """Test round-trip export preserves mask matrices."""

    def test_round_trip_with_sparse_mask(self, tmp_path):
        """Test export and import with sparse mask matrix."""
        # Create container with sparse data and mask
        n_samples, n_features = 50, 100
        X = sparse.random(n_samples, n_features, density=0.3, format="csr")
        M = sparse.csr_matrix(X.shape)
        M.data = np.where(np.random.rand(len(M.data)) > 0.5, 1, 0)

        matrix = ScpMatrix(X=X, M=M)
        var = pl.DataFrame({
            "_index": [f"P{i}" for i in range(n_features)],
            "feature": [f"P{i}" for i in range(n_features)],
            "mean": np.random.rand(n_features)
        })
        assay = Assay(var=var, layers={"X": matrix})

        obs = pl.DataFrame({
            "_index": [f"S{i}" for i in range(n_samples)],
            "sample": [f"S{i}" for i in range(n_samples)],
            "group": np.random.choice(["A", "B"], n_samples)
        })
        container = ScpContainer(obs=obs, assays={"proteins": assay})

        # Save and load
        path = tmp_path / "test_masks.h5"
        save_hdf5(container, path)

        loaded = load_hdf5(path)

        # Verify container structure
        assert loaded.n_samples == n_samples
        assert "proteins" in loaded.assays
        assert "X" in loaded.assays["proteins"].layers

        # Verify mask was preserved
        loaded_matrix = loaded.assays["proteins"].layers["X"]
        assert loaded_matrix.M is not None
        assert sparse.issparse(loaded_matrix.M)
        np.testing.assert_array_equal(loaded_matrix.M.toarray(), M.toarray())

    def test_round_trip_with_dense_mask(self, tmp_path):
        """Test export and import with dense mask matrix."""
        n_samples, n_features = 20, 30
        X = np.random.randn(n_samples, n_features)
        M = np.random.choice([0, 1, 2, 3], size=(n_samples, n_features))

        matrix = ScpMatrix(X=X, M=M)
        var = pl.DataFrame({
            "_index": [f"P{i}" for i in range(n_features)],
            "feature": [f"P{i}" for i in range(n_features)],
            "mean": np.random.rand(n_features)
        })
        assay = Assay(var=var, layers={"X": matrix})

        obs = pl.DataFrame({
            "_index": [f"S{i}" for i in range(n_samples)],
            "sample": [f"S{i}" for i in range(n_samples)],
            "group": np.random.choice(["A", "B"], n_samples)
        })
        container = ScpContainer(obs=obs, assays={"proteins": assay})

        # Save and load
        path = tmp_path / "test_dense_mask.h5"
        save_hdf5(container, path)

        loaded = load_hdf5(path)

        # Verify mask preserved
        loaded_matrix = loaded.assays["proteins"].layers["X"]
        assert loaded_matrix.M is not None
        assert not sparse.issparse(loaded_matrix.M)
        np.testing.assert_array_equal(loaded_matrix.M, M)

    def test_round_trip_with_no_mask(self, tmp_path):
        """Test export and import when mask is None."""
        n_samples, n_features = 20, 30
        X = np.random.randn(n_samples, n_features)

        matrix = ScpMatrix(X=X, M=None)
        var = pl.DataFrame({
            "_index": [f"P{i}" for i in range(n_features)],
            "feature": [f"P{i}" for i in range(n_features)],
            "mean": np.random.rand(n_features)
        })
        assay = Assay(var=var, layers={"X": matrix})

        obs = pl.DataFrame({
            "_index": [f"S{i}" for i in range(n_samples)],
            "sample": [f"S{i}" for i in range(n_samples)],
            "group": ["A"] * n_samples
        })
        container = ScpContainer(obs=obs, assays={"proteins": assay})

        # Save and load
        path = tmp_path / "test_no_mask.h5"
        save_hdf5(container, path)

        loaded = load_hdf5(path)

        # Verify mask is None
        loaded_matrix = loaded.assays["proteins"].layers["X"]
        assert loaded_matrix.M is None


class TestMultiAssayRoundTrip:
    """Test round-trip with multiple assays."""

    def test_multi_assay_export_import(self, tmp_path):
        """Test container with multiple assays preserves all data."""
        n_samples = 30
        obs = pl.DataFrame({
            "_index": [f"S{i}" for i in range(n_samples)],
            "sample": [f"S{i}" for i in range(n_samples)],
            "batch": np.random.choice(["B1", "B2"], n_samples)
        })

        # Create two assays
        assay1 = Assay(
            var=pl.DataFrame({
                "_index": [f"P{i}" for i in range(50)],
                "feature": [f"P{i}" for i in range(50)],
                "mean": np.random.rand(50)
            }),
            layers={
                "X": ScpMatrix(X=np.random.randn(n_samples, 50))
            }
        )

        assay2 = Assay(
            var=pl.DataFrame({
                "_index": [f"M{i}" for i in range(30)],
                "feature": [f"M{i}" for i in range(30)],
                "mean": np.random.rand(30)
            }),
            layers={
                "X": ScpMatrix(X=np.random.randn(n_samples, 30))
            }
        )

        container = ScpContainer(
            obs=obs,
            assays={"proteins": assay1, "metabolites": assay2}
        )

        # Save and load
        path = tmp_path / "test_multi_assay.h5"
        save_hdf5(container, path)

        loaded = load_hdf5(path)

        # Verify both assays present
        assert "proteins" in loaded.assays
        assert "metabolites" in loaded.assays
        assert loaded.assays["proteins"].var.shape == (50, 3)
        assert loaded.assays["metabolites"].var.shape == (30, 3)


class TestSelectiveExport:
    """Test selective export of specific assays and layers."""

    def test_selective_assay_export(self, tmp_path):
        """Test exporting only specific assays."""
        n_samples = 20
        obs = pl.DataFrame({
            "_index": [f"S{i}" for i in range(n_samples)],
            "sample": [f"S{i}" for i in range(n_samples)],
            "batch": ["B1"] * n_samples
        })

        container = ScpContainer(
            obs=obs,
            assays={
                "assay1": Assay(
                    var=pl.DataFrame({
                        "_index": [f"A{i}" for i in range(10)],
                        "feature": [f"A{i}" for i in range(10)]
                    }),
                    layers={"X": ScpMatrix(X=np.random.randn(n_samples, 10))}
                ),
                "assay2": Assay(
                    var=pl.DataFrame({
                        "_index": [f"B{i}" for i in range(10)],
                        "feature": [f"B{i}" for i in range(10)]
                    }),
                    layers={"X": ScpMatrix(X=np.random.randn(n_samples, 10))}
                ),
            }
        )

        # Export only assay1
        path = tmp_path / "test_selective_assay.h5"
        save_hdf5(container, path, save_assays=["assay1"])

        loaded = load_hdf5(path)

        # Verify only assay1 present
        assert "assay1" in loaded.assays
        assert "assay2" not in loaded.assays

    def test_selective_layer_export(self, tmp_path):
        """Test exporting only specific layers."""
        n_samples, n_features = 20, 30
        obs = pl.DataFrame({
            "_index": [f"S{i}" for i in range(n_samples)],
            "sample": [f"S{i}" for i in range(n_samples)],
            "batch": ["B1"] * n_samples
        })

        assay = Assay(
            var=pl.DataFrame({
                "_index": [f"P{i}" for i in range(n_features)],
                "feature": [f"P{i}" for i in range(n_features)],
                "mean": np.random.rand(n_features)
            }),
            layers={
                "X": ScpMatrix(X=np.random.randn(n_samples, n_features)),
                "log": ScpMatrix(X=np.random.randn(n_samples, n_features)),
                "scaled": ScpMatrix(X=np.random.randn(n_samples, n_features)),
            }
        )

        container = ScpContainer(obs=obs, assays={"proteins": assay})

        # Export only X and log layers
        path = tmp_path / "test_selective_layer.h5"
        save_hdf5(container, path, save_layers=["X", "log"])

        loaded = load_hdf5(path)

        # Verify only selected layers present
        assert "X" in loaded.assays["proteins"].layers
        assert "log" in loaded.assays["proteins"].layers
        assert "scaled" not in loaded.assays["proteins"].layers


class TestHistoryPreservation:
    """Test operation history preservation during export/import."""

    def test_history_round_trip(self, tmp_path):
        """Test that operation history is preserved."""
        n_samples, n_features = 20, 30
        container = ScpContainer(
            obs=pl.DataFrame({
                "_index": [f"S{i}" for i in range(n_samples)],
                "sample": [f"S{i}" for i in range(n_samples)],
                "batch": ["B1"] * n_samples
            }),
            assays={
                "proteins": Assay(
                    var=pl.DataFrame({
                        "_index": [f"P{i}" for i in range(n_features)],
                        "feature": [f"P{i}" for i in range(n_features)]
                    }),
                    layers={"X": ScpMatrix(X=np.random.randn(n_samples, n_features))}
                )
            },
            history=[
                ProvenanceLog(
                    timestamp=datetime.now().isoformat(),
                    action="normalize",
                    params={"method": "log"},
                    software_version="0.1.0",
                    description="Log normalization"
                ),
                ProvenanceLog(
                    timestamp=datetime.now().isoformat(),
                    action="impute",
                    params={"method": "knn", "k": 5},
                    software_version="0.1.0",
                    description="KNN imputation"
                ),
            ]
        )

        # Save and load
        path = tmp_path / "test_history.h5"
        save_hdf5(container, path, save_history=True)

        loaded = load_hdf5(path)

        # Verify history preserved
        assert len(loaded.history) == 2
        assert loaded.history[0].action == "normalize"
        assert loaded.history[1].action == "impute"
        assert loaded.history[1].params["k"] == 5

    def test_no_history_export(self, tmp_path):
        """Test export with save_history=False."""
        n_samples, n_features = 20, 30
        container = ScpContainer(
            obs=pl.DataFrame({
                "_index": [f"S{i}" for i in range(n_samples)],
                "sample": [f"S{i}" for i in range(n_samples)],
                "batch": ["B1"] * n_samples
            }),
            assays={
                "proteins": Assay(
                    var=pl.DataFrame({
                        "_index": [f"P{i}" for i in range(n_features)],
                        "feature": [f"P{i}" for i in range(n_features)]
                    }),
                    layers={"X": ScpMatrix(X=np.random.randn(n_samples, n_features))}
                )
            },
            history=[
                ProvenanceLog(
                    timestamp=datetime.now().isoformat(),
                    action="test_action",
                    params={},
                    software_version="0.1.0"
                )
            ]
        )

        # Save without history
        path = tmp_path / "test_no_history.h5"
        save_hdf5(container, path, save_history=False)

        loaded = load_hdf5(path)

        # Verify history is empty
        assert len(loaded.history) == 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_container_export(self, tmp_path):
        """Test export of minimal empty container."""
        container = ScpContainer(
            obs=pl.DataFrame({"_index": [], "sample": []}),
            assays={}
        )

        path = tmp_path / "test_empty.h5"
        save_hdf5(container, path)

        loaded = load_hdf5(path)

        # Verify empty structure preserved
        assert loaded.n_samples == 0
        assert len(loaded.assays) == 0

    def test_large_dataset_export(self, tmp_path):
        """Test export of larger dataset."""
        n_samples, n_features = 500, 1000
        container = ScpContainer(
            obs=pl.DataFrame({
                "_index": [f"S{i}" for i in range(n_samples)],
                "sample": [f"S{i}" for i in range(n_samples)],
                "batch": np.random.choice(["B1", "B2", "B3"], n_samples)
            }),
            assays={
                "proteins": Assay(
                    var=pl.DataFrame({
                        "_index": [f"P{i}" for i in range(n_features)],
                        "feature": [f"P{i}" for i in range(n_features)],
                        "mean": np.random.rand(n_features)
                    }),
                    layers={
                        "X": ScpMatrix(X=sparse.random(
                            n_samples, n_features, density=0.1, format="csr"
                        ))
                    }
                )
            }
        )

        path = tmp_path / "test_large.h5"
        save_hdf5(container, path)

        loaded = load_hdf5(path)

        # Verify dimensions preserved
        assert loaded.n_samples == n_samples
        assert loaded.assays["proteins"].var.shape[0] == n_features

    def test_special_characters_in_metadata(self, tmp_path):
        """Test handling of special characters in string columns."""
        n_samples, n_features = 10, 15
        container = ScpContainer(
            obs=pl.DataFrame({
                "_index": [f"S{i}_{j}" for i, j in enumerate(range(n_samples))],
                "sample": [f"S{i}_{j}" for i, j in enumerate(range(n_samples))],
                "group": ["group-A", "group/B", "group\\C"] + ["D"] * (n_samples - 3),
                "unicode": ["样本", "données", "Данные"] + ["E"] * (n_samples - 3)
            }),
            assays={
                "proteins": Assay(
                    var=pl.DataFrame({
                        "_index": [f"P{i};value" for i in range(n_features)],
                        "feature": [f"P{i};value" for i in range(n_features)],
                        "desc": ["protein|test"] * n_features
                    }),
                    layers={"X": ScpMatrix(X=np.random.randn(n_samples, n_features))}
                )
            }
        )

        path = tmp_path / "test_special_chars.h5"
        save_hdf5(container, path)

        loaded = load_hdf5(path)

        # Verify special characters preserved
        assert "样本" in loaded.obs["unicode"].to_list()
        assert "group-A" in loaded.obs["group"].to_list()

    def test_compression_levels(self, tmp_path):
        """Test different compression levels."""
        n_samples, n_features = 100, 200
        container = ScpContainer(
            obs=pl.DataFrame({
                "_index": [f"S{i}" for i in range(n_samples)],
                "sample": [f"S{i}" for i in range(n_samples)],
                "batch": ["B1"] * n_samples
            }),
            assays={
                "proteins": Assay(
                    var=pl.DataFrame({
                        "_index": [f"P{i}" for i in range(n_features)],
                        "feature": [f"P{i}" for i in range(n_features)],
                        "mean": np.random.rand(n_features)
                    }),
                    layers={"X": ScpMatrix(X=np.random.randn(n_samples, n_features))}
                )
            }
        )

        # Test different compression levels
        for level in [0, 4, 9]:
            path = tmp_path / f"test_comp_{level}.h5"
            save_hdf5(container, path, compression_level=level)

            loaded = load_hdf5(path)
            np.testing.assert_array_almost_equal(
                loaded.assays["proteins"].layers["X"].X,
                container.assays["proteins"].layers["X"].X
            )
