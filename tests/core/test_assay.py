"""
Tests for Assay core structure.

This module contains tests for Assay functionality.
"""

import pytest
import numpy as np
import polars as pl
from scipy import sparse


class TestAssayBasic:
    """Test basic Assay functionality."""

    def test_assay_creation(self):
        """Test creating a minimal Assay."""
        from scptensor.core import Assay, ScpMatrix

        var = pl.DataFrame({"_index": ["P1", "P2", "P3"]})
        X = np.random.rand(10, 3)
        matrix = ScpMatrix(X=X)
        assay = Assay(var=var, layers={"raw": matrix})

        assert assay.n_features == 3
        assert "raw" in assay.layers

    def test_assay_with_feature_id_col(self):
        """Test Assay with custom feature_id_col."""
        from scptensor.core import Assay, ScpMatrix

        var = pl.DataFrame({
            "_index": ["P1", "P2"],
            "protein_id": ["Prot1", "Prot2"]
        })
        X = np.random.rand(5, 2)
        matrix = ScpMatrix(X=X)
        assay = Assay(var=var, layers={"raw": matrix}, feature_id_col="protein_id")

        assert assay.feature_id_col == "protein_id"
        assert list(assay.feature_ids) == ["Prot1", "Prot2"]

    def test_n_features_property(self):
        """Test n_features property returns correct count."""
        from scptensor.core import Assay, ScpMatrix

        var = pl.DataFrame({"_index": [f"P{i}" for i in range(100)]})
        X = np.random.rand(50, 100)
        matrix = ScpMatrix(X=X)
        assay = Assay(var=var, layers={"raw": matrix})

        assert assay.n_features == 100

    def test_feature_ids_property(self):
        """Test feature_ids property returns correct IDs."""
        from scptensor.core import Assay, ScpMatrix

        var = pl.DataFrame({"_index": ["GeneA", "GeneB", "GeneC"]})
        X = np.random.rand(5, 3)
        matrix = ScpMatrix(X=X)
        assay = Assay(var=var, layers={"raw": matrix})

        assert list(assay.feature_ids) == ["GeneA", "GeneB", "GeneC"]

    def test_x_property_default_layer(self):
        """Test X property returns data from 'X' layer if exists."""
        from scptensor.core import Assay, ScpMatrix

        var = pl.DataFrame({"_index": ["P1"]})
        X = np.array([[1.0, 2.0, 3.0]]).T
        matrix = ScpMatrix(X=X)
        assay = Assay(var=var, layers={"X": matrix})

        result = assay.X
        assert result is not None
        assert result.shape == (3, 1)
        np.testing.assert_array_equal(result, X)

    def test_x_property_returns_none_when_no_x_layer(self):
        """Test X property returns None when no 'X' layer exists."""
        from scptensor.core import Assay, ScpMatrix

        var = pl.DataFrame({"_index": ["P1"]})
        X = np.random.rand(3, 1)
        matrix = ScpMatrix(X=X)
        assay = Assay(var=var, layers={"raw": matrix})  # Not "X"

        result = assay.X
        assert result is None

    def test_access_layer_directly(self):
        """Test accessing layers directly via layers dict."""
        from scptensor.core import Assay, ScpMatrix

        var = pl.DataFrame({"_index": ["P1"]})
        X1 = np.random.rand(5, 1)
        X2 = np.random.rand(5, 1) * 2
        matrix1 = ScpMatrix(X=X1)
        matrix2 = ScpMatrix(X=X2)
        assay = Assay(var=var, layers={"layer1": matrix1, "layer2": matrix2})

        result1 = assay.layers["layer1"].X
        result2 = assay.layers["layer2"].X

        assert result1.shape == (5, 1)
        assert result2.shape == (5, 1)
        # Verify different data
        assert not np.allclose(result1, result2)

    def test_add_layer(self):
        """Test adding a new layer to assay."""
        from scptensor.core import Assay, ScpMatrix

        var = pl.DataFrame({"_index": ["P1", "P2"]})
        X1 = np.random.rand(3, 2)
        matrix1 = ScpMatrix(X=X1)
        assay = Assay(var=var, layers={"raw": matrix1})

        # Add new layer
        X2 = np.random.rand(3, 2) * 10
        matrix2 = ScpMatrix(X=X2)
        assay.add_layer("normalized", matrix2)

        assert "normalized" in assay.layers
        assert len(assay.layers) == 2
        np.testing.assert_array_equal(assay.layers["normalized"].X, X2)

    def test_assay_repr(self):
        """Test __repr__ method returns string representation."""
        from scptensor.core import Assay, ScpMatrix

        var = pl.DataFrame({"_index": ["P1", "P2", "P3"]})
        X = np.random.rand(5, 3)
        matrix = ScpMatrix(X=X)
        assay = Assay(var=var, layers={"raw": matrix})

        repr_str = repr(assay)
        assert isinstance(repr_str, str)
        assert "Assay" in repr_str
        assert "3" in repr_str  # n_features

    def test_assay_multiple_layers(self):
        """Test assay with multiple layers."""
        from scptensor.core import Assay, ScpMatrix

        var = pl.DataFrame({"_index": ["P1"]})
        X_raw = np.random.rand(5, 1)
        X_log = np.log1p(X_raw)
        X_norm = (X_log - X_log.mean()) / X_log.std()

        assay = Assay(
            var=var,
            layers={
                "raw": ScpMatrix(X=X_raw),
                "log": ScpMatrix(X=X_log),
                "normalized": ScpMatrix(X=X_norm)
            }
        )

        assert len(assay.layers) == 3
        assert "raw" in assay.layers
        assert "log" in assay.layers
        assert "normalized" in assay.layers

    def test_assay_with_sparse_matrix(self):
        """Test assay with sparse matrix storage."""
        from scptensor.core import Assay, ScpMatrix

        var = pl.DataFrame({"_index": [f"P{i}" for i in range(10)]})
        X_dense = np.random.rand(5, 10)
        # Make 80% sparse
        X_dense[X_dense < 0.8] = 0
        X_sparse = sparse.csr_matrix(X_dense)

        matrix = ScpMatrix(X=X_sparse)
        assay = Assay(var=var, layers={"raw": matrix})

        assert assay.n_features == 10
        assert sparse.issparse(assay.layers["raw"].X)

    def test_assay_with_metadata(self):
        """Test assay with additional metadata in var."""
        from scptensor.core import Assay, ScpMatrix

        var = pl.DataFrame({
            "_index": ["P1", "P2", "P3"],
            "protein_name": ["Protein1", "Protein2", "Protein3"],
            "chromosome": ["chr1", "chr2", "chr3"],
            "is_significant": [True, False, True]
        })
        X = np.random.rand(10, 3)
        matrix = ScpMatrix(X=X)
        assay = Assay(var=var, layers={"raw": matrix})

        assert "protein_name" in assay.var.columns
        assert "chromosome" in assay.var.columns
        assert assay.var["is_significant"].to_list() == [True, False, True]

    def test_assay_subset_by_indices(self):
        """Test subsetting assay by feature indices."""
        from scptensor.core import Assay, ScpMatrix

        var = pl.DataFrame({"_index": ["P1", "P2", "P3", "P4", "P5"]})
        X = np.arange(20).reshape(4, 5)  # 4 samples, 5 features
        matrix = ScpMatrix(X=X)
        assay = Assay(var=var, layers={"raw": matrix})

        # Subset to features [0, 2, 4] -> P1, P3, P5
        subset_assay = assay.subset(feature_indices=[0, 2, 4])

        assert subset_assay.n_features == 3
        assert list(subset_assay.feature_ids) == ["P1", "P3", "P5"]
        assert subset_assay.layers["raw"].X.shape == (4, 3)

    def test_assay_subset_preserves_layers(self):
        """Test that subsetting preserves all layers."""
        from scptensor.core import Assay, ScpMatrix

        var = pl.DataFrame({"_index": ["P1", "P2"]})
        X1 = np.random.rand(3, 2)
        X2 = np.random.rand(3, 2) * 2

        assay = Assay(
            var=var,
            layers={
                "layer1": ScpMatrix(X=X1),
                "layer2": ScpMatrix(X=X2)
            }
        )

        subset_assay = assay.subset(feature_indices=[0])

        assert len(subset_assay.layers) == 2
        assert "layer1" in subset_assay.layers
        assert "layer2" in subset_assay.layers
        assert subset_assay.layers["layer1"].X.shape == (3, 1)

    def test_assay_subset_with_invalid_indices(self):
        """Test subsetting with invalid feature indices."""
        from scptensor.core import Assay, ScpMatrix

        var = pl.DataFrame({"_index": ["P1", "P2"]})
        X = np.random.rand(3, 2)
        matrix = ScpMatrix(X=X)
        assay = Assay(var=var, layers={"raw": matrix})

        # Try to subset with out-of-bounds index
        with pytest.raises(Exception):
            assay.subset(feature_indices=[0, 5])
