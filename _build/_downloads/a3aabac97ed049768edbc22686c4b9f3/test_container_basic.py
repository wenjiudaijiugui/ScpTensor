"""
Basic tests for ScpContainer core structure.

This module contains fundamental tests for ScpContainer functionality.
"""

import pytest
import numpy as np
import polars as pl
from copy import deepcopy


class TestScpContainerBasic:
    """Test basic ScpContainer functionality."""

    def test_container_creation(self):
        """Test creating a minimal ScpContainer."""
        from scptensor.core import ScpContainer, Assay, ScpMatrix

        # Create sample data
        obs = pl.DataFrame({"_index": ["S1", "S2", "S3"]})
        var = pl.DataFrame({"_index": ["P1", "P2"]})
        X = np.random.rand(3, 2)
        matrix = ScpMatrix(X=X)
        assay = Assay(var=var, layers={"X": matrix})

        # Create container
        container = ScpContainer(obs=obs, assays={"test": assay})

        # Assertions
        assert container.n_samples == 3
        assert "test" in container.assays
        assert len(container.assays) == 1

    def test_container_add_assay(self):
        """Test adding an assay to container."""
        from scptensor.core import ScpContainer, Assay, ScpMatrix

        # Create container
        obs = pl.DataFrame({"_index": ["S1", "S2"]})
        container = ScpContainer(obs=obs, assays={})

        # Add assay
        var = pl.DataFrame({"_index": ["P1"]})
        X = np.random.rand(2, 1)
        assay = Assay(var=var, layers={"X": ScpMatrix(X=X)})
        container = container.add_assay("protein", assay)

        # Assertions
        assert "protein" in container.assays
        assert container.assays["protein"].n_features == 1

    def test_import_integration_module(self):
        """Test that integration module can be imported."""
        from scptensor.integration import combat, harmony

        # Verify functions exist
        assert callable(combat)
        assert callable(harmony)

    def test_import_qc_module(self):
        """Test that qc module can be imported."""
        from scptensor.qc import basic_qc, detect_outliers

        # Verify functions exist
        assert callable(basic_qc)
        assert callable(detect_outliers)

    def test_n_samples_property(self):
        """Test n_samples property returns correct sample count."""
        from scptensor.core import ScpContainer, Assay, ScpMatrix

        obs = pl.DataFrame({"_index": ["S1", "S2", "S3", "S4"]})
        var = pl.DataFrame({"_index": ["P1", "P2"]})
        X = np.random.rand(4, 2)
        matrix = ScpMatrix(X=X)
        assay = Assay(var=var, layers={"X": matrix})

        container = ScpContainer(obs=obs, assays={"test": assay})

        assert container.n_samples == 4

    def test_sample_ids_property(self):
        """Test sample_ids property returns correct sample IDs."""
        from scptensor.core import ScpContainer, Assay, ScpMatrix

        obs = pl.DataFrame({"_index": ["S1", "S2", "S3"]})
        var = pl.DataFrame({"_index": ["P1"]})
        X = np.random.rand(3, 1)
        matrix = ScpMatrix(X=X)
        assay = Assay(var=var, layers={"X": matrix})

        container = ScpContainer(obs=obs, assays={"test": assay})

        assert list(container.sample_ids) == ["S1", "S2", "S3"]

    def test_validate_with_valid_data(self):
        """Test _validate with valid obs and assays."""
        from scptensor.core import ScpContainer, Assay, ScpMatrix

        obs = pl.DataFrame({"_index": ["S1", "S2"]})
        var = pl.DataFrame({"_index": ["P1"]})
        X = np.random.rand(2, 1)
        matrix = ScpMatrix(X=X)
        assay = Assay(var=var, layers={"X": matrix})

        # Should not raise any exception
        container = ScpContainer(obs=obs, assays={"test": assay})
        container._validate()

    def test_validate_with_invalid_obs(self):
        """Test _validate with invalid obs (missing _index)."""
        from scptensor.core import ScpContainer, Assay, ScpMatrix

        obs = pl.DataFrame({"sample_id": ["S1", "S2"]})  # Missing _index
        var = pl.DataFrame({"_index": ["P1"]})
        X = np.random.rand(2, 1)
        matrix = ScpMatrix(X=X)
        assay = Assay(var=var, layers={"X": matrix})

        # Validation happens during initialization
        with pytest.raises(ValueError, match="Sample ID column"):
            ScpContainer(obs=obs, assays={"test": assay})

    def test_validate_with_mismatched_assay(self):
        """Test _validate with assay having different sample count."""
        from scptensor.core import ScpContainer, Assay, ScpMatrix

        obs = pl.DataFrame({"_index": ["S1", "S2"]})
        var = pl.DataFrame({"_index": ["P1"]})
        X = np.random.rand(3, 1)  # 3 samples instead of 2
        matrix = ScpMatrix(X=X)
        assay = Assay(var=var, layers={"X": matrix})

        # Validation happens during initialization
        with pytest.raises(ValueError, match="Sample dimension mismatch"):
            ScpContainer(obs=obs, assays={"test": assay})

    def test_log_operation(self):
        """Test log_operation method records operation in history."""
        from scptensor.core import ScpContainer, Assay, ScpMatrix

        obs = pl.DataFrame({"_index": ["S1"]})
        var = pl.DataFrame({"_index": ["P1"]})
        X = np.random.rand(1, 1)
        matrix = ScpMatrix(X=X)
        assay = Assay(var=var, layers={"X": matrix})

        container = ScpContainer(obs=obs, assays={"test": assay})
        initial_history_len = len(container.history)

        container.log_operation(
            action="test_operation",
            params={"param1": "value1"},
            description="Test operation"
        )

        assert len(container.history) == initial_history_len + 1
        assert container.history[-1].action == "test_operation"
        assert container.history[-1].params == {"param1": "value1"}

    def test_repr(self):
        """Test __repr__ method returns string representation."""
        from scptensor.core import ScpContainer, Assay, ScpMatrix

        obs = pl.DataFrame({"_index": ["S1", "S2"]})
        var = pl.DataFrame({"_index": ["P1"]})
        X = np.random.rand(2, 1)
        matrix = ScpMatrix(X=X)
        assay = Assay(var=var, layers={"X": matrix})

        container = ScpContainer(obs=obs, assays={"test": assay})

        repr_str = repr(container)
        assert isinstance(repr_str, str)
        assert "ScpContainer" in repr_str
        assert "2" in repr_str  # n_samples

    def test_copy(self):
        """Test copy method creates shallow copy."""
        from scptensor.core import ScpContainer, Assay, ScpMatrix

        obs = pl.DataFrame({"_index": ["S1", "S2"]})
        var = pl.DataFrame({"_index": ["P1"]})
        X = np.random.rand(2, 1)
        matrix = ScpMatrix(X=X)
        assay = Assay(var=var, layers={"X": matrix})

        container = ScpContainer(obs=obs, assays={"test": assay})
        container_copy = container.copy()

        assert container_copy is not container
        assert container_copy.n_samples == container.n_samples
        assert container_copy.sample_id_col == container.sample_id_col

    def test_shallow_copy(self):
        """Test shallow_copy method."""
        from scptensor.core import ScpContainer, Assay, ScpMatrix

        obs = pl.DataFrame({"_index": ["S1"]})
        var = pl.DataFrame({"_index": ["P1"]})
        X = np.random.rand(1, 1)
        matrix = ScpMatrix(X=X)
        assay = Assay(var=var, layers={"X": matrix})

        container = ScpContainer(obs=obs, assays={"test": assay})
        shallow = container.shallow_copy()

        assert shallow is not container
        assert shallow.n_samples == container.n_samples

    def test_deepcopy(self):
        """Test deepcopy method creates fully independent copy."""
        from scptensor.core import ScpContainer, Assay, ScpMatrix

        obs = pl.DataFrame({"_index": ["S1"]})
        var = pl.DataFrame({"_index": ["P1"]})
        X = np.random.rand(1, 1)
        matrix = ScpMatrix(X=X)
        assay = Assay(var=var, layers={"X": matrix})

        container = ScpContainer(obs=obs, assays={"test": assay})
        deep = container.deepcopy()

        assert deep is not container
        assert deep.n_samples == container.n_samples

        # Modify original
        container.log_operation("test", {}, "Test")

        # Deep copy should not have the new operation
        assert len(deep.history) != len(container.history)

    def test_container_with_metadata(self):
        """Test container with additional metadata columns in obs."""
        from scptensor.core import ScpContainer, Assay, ScpMatrix

        obs = pl.DataFrame({
            "_index": ["S1", "S2"],
            "group": ["A", "B"],
            "batch": ["batch1", "batch2"]
        })
        var = pl.DataFrame({"_index": ["P1"]})
        X = np.random.rand(2, 1)
        matrix = ScpMatrix(X=X)
        assay = Assay(var=var, layers={"X": matrix})

        container = ScpContainer(obs=obs, assays={"test": assay})

        assert "group" in container.obs.columns
        assert "batch" in container.obs.columns
        assert container.obs["group"].to_list() == ["A", "B"]

    def test_container_multiple_assays(self):
        """Test container with multiple assays."""
        from scptensor.core import ScpContainer, Assay, ScpMatrix

        obs = pl.DataFrame({"_index": ["S1", "S2"]})

        # Create first assay
        var1 = pl.DataFrame({"_index": ["P1", "P2"]})
        X1 = np.random.rand(2, 2)
        matrix1 = ScpMatrix(X=X1)
        assay1 = Assay(var=var1, layers={"X": matrix1})

        # Create second assay
        var2 = pl.DataFrame({"_index": ["Peptide1", "Peptide2", "Peptide3"]})
        X2 = np.random.rand(2, 3)
        matrix2 = ScpMatrix(X=X2)
        assay2 = Assay(var=var2, layers={"X": matrix2})

        container = ScpContainer(obs=obs, assays={"protein": assay1, "peptide": assay2})

        assert len(container.assays) == 2
        assert "protein" in container.assays
        assert "peptide" in container.assays
        assert container.assays["protein"].n_features == 2
        assert container.assays["peptide"].n_features == 3

    def test_history_tracking(self):
        """Test that operation history is properly tracked."""
        from scptensor.core import ScpContainer, Assay, ScpMatrix

        obs = pl.DataFrame({"_index": ["S1"]})
        var = pl.DataFrame({"_index": ["P1"]})
        X = np.random.rand(1, 1)
        matrix = ScpMatrix(X=X)
        assay = Assay(var=var, layers={"X": matrix})

        container = ScpContainer(obs=obs, assays={"test": assay})

        # Log multiple operations
        container.log_operation("normalize", {"method": "log"}, "Log normalization")
        container.log_operation("impute", {"method": "knn", "k": 5}, "KNN imputation")

        assert len(container.history) >= 2
        assert container.history[-2].action == "normalize"
        assert container.history[-1].action == "impute"
        assert container.history[-1].params == {"method": "knn", "k": 5}

