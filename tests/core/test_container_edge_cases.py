"""
Edge case tests for ScpContainer.

This module tests edge cases, boundary conditions, and error handling for ScpContainer.
"""

import numpy as np
import polars as pl
import pytest

from scptensor.core import AggregationLink, Assay, MaskCode, ProvenanceLog, ScpContainer, ScpMatrix


class TestScpContainerEdgeCases:
    """Test edge cases for ScpContainer."""

    def test_container_with_single_sample(self):
        """Test ScpContainer with single sample."""
        obs = pl.DataFrame({"_index": ["S1"]})
        var = pl.DataFrame({"_index": ["P1", "P2"]})
        X = np.random.rand(1, 2)
        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X)})
        container = ScpContainer(obs=obs, assays={"test": assay})
        assert container.n_samples == 1
        assert list(container.sample_ids) == ["S1"]

    def test_container_with_many_samples(self):
        """Test ScpContainer with many samples."""
        n_samples = 10000
        obs = pl.DataFrame({"_index": [f"S{i}" for i in range(n_samples)]})
        var = pl.DataFrame({"_index": ["P1"]})
        X = np.random.rand(n_samples, 1)
        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X)})
        container = ScpContainer(obs=obs, assays={"test": assay})
        assert container.n_samples == n_samples

    def test_container_with_no_assays(self):
        """Test ScpContainer with no assays."""
        obs = pl.DataFrame({"_index": ["S1", "S2"]})
        container = ScpContainer(obs=obs, assays={})
        assert container.n_samples == 2
        assert len(container.assays) == 0

    def test_container_with_empty_obs(self):
        """Test ScpContainer with empty obs (0 samples)."""
        obs = pl.DataFrame({"_index": []})
        container = ScpContainer(obs=obs, assays={})
        assert container.n_samples == 0


class TestScpContainerValidation:
    """Test ScpContainer validation."""

    def test_container_missing_sample_id_col_raises_error(self):
        """Test that missing sample_id_col raises ValueError."""
        obs = pl.DataFrame({"sample_id": ["S1", "S2"]})  # No _index
        var = pl.DataFrame({"_index": ["P1"]})
        X = np.random.rand(2, 1)
        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X)})
        with pytest.raises(ValueError, match="Sample ID column '_index' not found"):
            ScpContainer(obs=obs, assays={"test": assay})

    def test_container_custom_missing_sample_id_col_raises_error(self):
        """Test that missing custom sample_id_col raises ValueError."""
        obs = pl.DataFrame({"_index": ["S1", "S2"]})  # No sample_id
        var = pl.DataFrame({"_index": ["P1"]})
        X = np.random.rand(2, 1)
        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X)})
        with pytest.raises(ValueError, match="Sample ID column 'sample_id' not found"):
            ScpContainer(obs=obs, assays={"test": assay}, sample_id_col="sample_id")

    def test_container_duplicate_sample_ids_raises_error(self):
        """Test that duplicate sample IDs raise ValueError."""
        obs = pl.DataFrame({"_index": ["S1", "S1", "S2"]})  # Duplicate S1
        var = pl.DataFrame({"_index": ["P1"]})
        X = np.random.rand(3, 1)
        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X)})
        with pytest.raises(ValueError, match="not unique"):
            ScpContainer(obs=obs, assays={"test": assay})

    def test_container_validate_assay_sample_dimension_mismatch(self):
        """Test that assay sample dimension mismatch raises ValueError."""
        obs = pl.DataFrame({"_index": ["S1", "S2"]})
        var = pl.DataFrame({"_index": ["P1"]})
        X = np.random.rand(3, 1)  # 3 samples instead of 2
        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X)})
        with pytest.raises(ValueError, match="Sample dimension mismatch"):
            ScpContainer(obs=obs, assays={"test": assay})

    def test_container_validate_multiple_assays(self):
        """Test that validation checks all assays."""
        obs = pl.DataFrame({"_index": ["S1", "S2"]})
        var = pl.DataFrame({"_index": ["P1"]})
        X1 = np.random.rand(2, 1)
        X2 = np.random.rand(3, 1)  # Wrong dimension
        assay1 = Assay(var=var, layers={"raw": ScpMatrix(X=X1)})
        assay2 = Assay(var=var, layers={"raw": ScpMatrix(X=X2)})
        with pytest.raises(ValueError, match="Sample dimension mismatch"):
            ScpContainer(obs=obs, assays={"assay1": assay1, "assay2": assay2})


class TestScpContainerSampleIdCol:
    """Test custom sample_id_col functionality."""

    def test_container_custom_sample_id_col(self):
        """Test ScpContainer with custom sample_id_col."""
        obs = pl.DataFrame({"_index": ["x1", "x2"], "sample_id": ["S1", "S2"]})
        var = pl.DataFrame({"_index": ["P1"]})
        X = np.random.rand(2, 1)
        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X)})
        container = ScpContainer(obs=obs, assays={"test": assay}, sample_id_col="sample_id")
        assert container.sample_id_col == "sample_id"
        assert list(container.sample_ids) == ["S1", "S2"]

    def test_container_sample_ids_property_uses_correct_col(self):
        """Test that sample_ids uses the correct column."""
        obs = pl.DataFrame({"_index": ["x1", "x2"], "cell_id": ["C1", "C2"]})
        var = pl.DataFrame({"_index": ["P1"]})
        X = np.random.rand(2, 1)
        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X)})
        container = ScpContainer(obs=obs, assays={"test": assay}, sample_id_col="cell_id")
        assert list(container.sample_ids) == ["C1", "C2"]


class TestScpContainerAddAssay:
    """Test add_assay method."""

    def test_add_assay_to_empty_container(self):
        """Test adding assay to container with no assays."""
        obs = pl.DataFrame({"_index": ["S1", "S2"]})
        container = ScpContainer(obs=obs, assays={})
        var = pl.DataFrame({"_index": ["P1"]})
        X = np.random.rand(2, 1)
        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X)})
        result = container.add_assay("test", assay)
        assert "test" in result.assays

    def test_add_assay_validates_dimension(self):
        """Test that add_assay validates sample dimension."""
        obs = pl.DataFrame({"_index": ["S1", "S2"]})
        container = ScpContainer(obs=obs, assays={})
        var = pl.DataFrame({"_index": ["P1"]})
        X = np.random.rand(3, 1)  # Wrong dimension
        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X)})
        with pytest.raises(ValueError, match="Sample dimension mismatch"):
            container.add_assay("test", assay)

    def test_add_assay_duplicate_name_raises_error(self):
        """Test that adding duplicate assay name raises ValueError."""
        obs = pl.DataFrame({"_index": ["S1"]})
        var = pl.DataFrame({"_index": ["P1"]})
        X = np.random.rand(1, 1)
        assay1 = Assay(var=var, layers={"raw": ScpMatrix(X=X)})
        assay2 = Assay(var=var, layers={"raw": ScpMatrix(X=X)})
        container = ScpContainer(obs=obs, assays={"test": assay1})
        with pytest.raises(ValueError, match="already exists"):
            container.add_assay("test", assay2)

    def test_add_assay_returns_self(self):
        """Test that add_assay returns container for chaining."""
        obs = pl.DataFrame({"_index": ["S1"]})
        container = ScpContainer(obs=obs, assays={})
        var = pl.DataFrame({"_index": ["P1"]})
        X = np.random.rand(1, 1)
        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X)})
        result = container.add_assay("test", assay)
        assert result is container


class TestScpContainerHistory:
    """Test operation history tracking."""

    def test_history_initialized_empty(self):
        """Test that history is initialized as empty list."""
        obs = pl.DataFrame({"_index": ["S1"]})
        container = ScpContainer(obs=obs, assays={})
        assert len(container.history) == 0

    def test_history_with_initial_logs(self):
        """Test creating container with initial history."""
        obs = pl.DataFrame({"_index": ["S1"]})
        history = [
            ProvenanceLog(
                timestamp="2025-01-01T00:00:00",
                action="import",
                params={},
                description="Initial import",
            )
        ]
        container = ScpContainer(obs=obs, assays={}, history=history)
        assert len(container.history) == 1
        assert container.history[0].action == "import"

    def test_log_operation_adds_to_history(self):
        """Test that log_operation adds entry to history."""
        obs = pl.DataFrame({"_index": ["S1"]})
        container = ScpContainer(obs=obs, assays={})
        container.log_operation("test", {"key": "value"}, "Test description")
        assert len(container.history) == 1
        assert container.history[0].action == "test"
        assert container.history[0].params == {"key": "value"}
        assert container.history[0].description == "Test description"

    def test_log_operation_with_software_version(self):
        """Test log_operation with software version."""
        obs = pl.DataFrame({"_index": ["S1"]})
        container = ScpContainer(obs=obs, assays={})
        container.log_operation("normalize", {"method": "log"}, software_version="0.1.0")
        assert container.history[0].software_version == "0.1.0"

    def test_log_operation_without_description(self):
        """Test log_operation without optional description."""
        obs = pl.DataFrame({"_index": ["S1"]})
        container = ScpContainer(obs=obs, assays={})
        container.log_operation("test", {"key": "value"})
        assert container.history[0].description is None

    def test_log_operation_timestamp_format(self):
        """Test that log_operation creates ISO format timestamp."""
        obs = pl.DataFrame({"_index": ["S1"]})
        container = ScpContainer(obs=obs, assays={})
        container.log_operation("test", {})
        timestamp = container.history[0].timestamp
        assert "T" in timestamp  # ISO format


class TestScpContainerCopy:
    """Test copy methods."""

    def test_shallow_copy_shares_obs(self):
        """Test that shallow copy shares obs DataFrame."""
        obs = pl.DataFrame({"_index": ["S1", "S2"], "value": [1, 2]})
        var = pl.DataFrame({"_index": ["P1"]})
        X = np.array([[1.0], [2.0]])
        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X)})
        container = ScpContainer(obs=obs, assays={"test": assay})
        shallow = container.shallow_copy()
        # They should share the same obs reference
        assert shallow.obs is container.obs

    def test_deepcopy_clones_obs(self):
        """Test that deepcopy clones obs DataFrame."""
        obs = pl.DataFrame({"_index": ["S1", "S2"], "value": [1, 2]})
        var = pl.DataFrame({"_index": ["P1"]})
        X = np.array([[1.0], [2.0]])
        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X)})
        container = ScpContainer(obs=obs, assays={"test": assay})
        deep = container.deepcopy()
        # They should have same values but different objects
        assert deep.obs is not container.obs
        assert deep.obs["value"].to_list() == [1, 2]

    def test_deepcopy_independent_history(self):
        """Test that deepcopy creates independent history."""
        obs = pl.DataFrame({"_index": ["S1"]})
        container = ScpContainer(obs=obs, assays={})
        container.log_operation("test1", {}, "Test 1")
        deep = container.deepcopy()
        container.log_operation("test2", {}, "Test 2")
        assert len(container.history) == 2
        assert len(deep.history) == 1

    def test_copy_with_deep_true(self):
        """Test copy with deep=True uses deepcopy."""
        obs = pl.DataFrame({"_index": ["S1"]})
        container = ScpContainer(obs=obs, assays={})
        container.log_operation("test", {}, "Test")
        copy = container.copy(deep=True)
        # Modify original
        container.log_operation("test2", {}, "Test 2")
        # Copy should not have new entry
        assert len(copy.history) == 1

    def test_copy_with_deep_false(self):
        """Test copy with deep=False uses shallow_copy."""
        obs = pl.DataFrame({"_index": ["S1"]})
        container = ScpContainer(obs=obs, assays={})
        shallow = container.copy(deep=False)
        # Should share obs reference
        assert shallow.obs is container.obs


class TestScpContainerRepr:
    """Test __repr__ method."""

    def test_repr_empty_assays(self):
        """Test __repr__ with no assays."""
        obs = pl.DataFrame({"_index": ["S1", "S2"]})
        container = ScpContainer(obs=obs, assays={})
        repr_str = repr(container)
        assert "ScpContainer" in repr_str
        assert "n_samples=2" in repr_str

    def test_repr_with_single_assay(self):
        """Test __repr__ with single assay."""
        obs = pl.DataFrame({"_index": ["S1"]})
        var = pl.DataFrame({"_index": ["P1"]})
        X = np.random.rand(1, 1)
        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X)})
        container = ScpContainer(obs=obs, assays={"test": assay})
        repr_str = repr(container)
        assert "test" in repr_str
        assert "1" in repr_str  # n_features

    def test_repr_with_multiple_assays(self):
        """Test __repr__ with multiple assays."""
        obs = pl.DataFrame({"_index": ["S1"]})
        var1 = pl.DataFrame({"_index": ["P1", "P2"]})
        var2 = pl.DataFrame({"_index": ["P3", "P4", "P5"]})
        assay1 = Assay(var=var1, layers={"X": ScpMatrix(X=np.random.rand(1, 2))})
        assay2 = Assay(var=var2, layers={"X": ScpMatrix(X=np.random.rand(1, 3))})
        container = ScpContainer(obs=obs, assays={"assay1": assay1, "assay2": assay2})
        repr_str = repr(container)
        assert "assay1" in repr_str
        assert "assay2" in repr_str


class TestScpContainerLinks:
    """Test AggregationLink functionality."""

    def test_container_with_links(self):
        """Test creating container with aggregation links."""
        obs = pl.DataFrame({"_index": ["S1"]})
        var1 = pl.DataFrame({"_index": ["P1", "P2"]})
        var2 = pl.DataFrame({"_index": ["PEP1", "PEP2", "PEP3"]})
        assay1 = Assay(var=var1, layers={"X": ScpMatrix(X=np.random.rand(1, 2))})
        assay2 = Assay(var=var2, layers={"X": ScpMatrix(X=np.random.rand(1, 3))})

        linkage = pl.DataFrame({"source_id": ["PEP1", "PEP2"], "target_id": ["P1", "P2"]})
        link = AggregationLink(source_assay="peptides", target_assay="proteins", linkage=linkage)

        container = ScpContainer(
            obs=obs, assays={"proteins": assay1, "peptides": assay2}, links=[link]
        )
        assert len(container.links) == 1
        assert container.links[0].source_assay == "peptides"

    def test_validate_links_with_missing_source_assay(self):
        """Test that validate_links detects missing source assay."""
        obs = pl.DataFrame({"_index": ["S1"]})
        var = pl.DataFrame({"_index": ["P1"]})
        assay = Assay(var=var, layers={"X": ScpMatrix(X=np.random.rand(1, 1))})

        linkage = pl.DataFrame({"source_id": ["PEP1"], "target_id": ["P1"]})
        link = AggregationLink(
            source_assay="missing_peptides", target_assay="proteins", linkage=linkage
        )

        with pytest.raises(ValueError, match="source assay 'missing_peptides' not found"):
            ScpContainer(obs=obs, assays={"proteins": assay}, links=[link])

    def test_validate_links_with_missing_target_assay(self):
        """Test that validate_links detects missing target assay."""
        obs = pl.DataFrame({"_index": ["S1"]})
        var = pl.DataFrame({"_index": ["PEP1"]})
        assay = Assay(var=var, layers={"X": ScpMatrix(X=np.random.rand(1, 1))})

        linkage = pl.DataFrame({"source_id": ["PEP1"], "target_id": ["P1"]})
        link = AggregationLink(
            source_assay="peptides", target_assay="missing_proteins", linkage=linkage
        )

        with pytest.raises(ValueError, match="target assay 'missing_proteins' not found"):
            ScpContainer(obs=obs, assays={"peptides": assay}, links=[link])


class TestScpContainerWithMetadata:
    """Test ScpContainer with various obs metadata."""

    def test_container_with_numeric_metadata(self):
        """Test container with numeric metadata columns."""
        obs = pl.DataFrame({"_index": ["S1", "S2"], "age": [25, 30], "score": [0.5, 0.8]})
        var = pl.DataFrame({"_index": ["P1"]})
        X = np.random.rand(2, 1)
        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X)})
        container = ScpContainer(obs=obs, assays={"test": assay})
        assert container.obs["age"].to_list() == [25, 30]

    def test_container_with_boolean_metadata(self):
        """Test container with boolean metadata columns."""
        obs = pl.DataFrame(
            {"_index": ["S1", "S2"], "is_control": [True, False], "passed_qc": [True, True]}
        )
        var = pl.DataFrame({"_index": ["P1"]})
        X = np.random.rand(2, 1)
        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X)})
        container = ScpContainer(obs=obs, assays={"test": assay})
        assert container.obs["is_control"].to_list() == [True, False]

    def test_container_with_string_metadata(self):
        """Test container with string metadata columns."""
        obs = pl.DataFrame(
            {
                "_index": ["S1", "S2"],
                "batch": ["batch1", "batch2"],
                "condition": ["control", "treatment"],
            }
        )
        var = pl.DataFrame({"_index": ["P1"]})
        X = np.random.rand(2, 1)
        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X)})
        container = ScpContainer(obs=obs, assays={"test": assay})
        assert container.obs["condition"].to_list() == ["control", "treatment"]

    def test_container_with_mixed_metadata_types(self):
        """Test container with mixed types in obs."""
        obs = pl.DataFrame(
            {
                "_index": ["S1", "S2"],
                "batch": ["B1", "B2"],
                "n_proteins": [100, 150],
                "is_complete": [True, False],
                "quality_score": [0.85, 0.92],
            }
        )
        var = pl.DataFrame({"_index": ["P1"]})
        X = np.random.rand(2, 1)
        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X)})
        container = ScpContainer(obs=obs, assays={"test": assay})
        assert len(container.obs.columns) == 5


class TestScpContainerMultipleAssays:
    """Test ScpContainer with multiple assays."""

    def test_container_with_different_feature_counts(self):
        """Test container with assays having different feature counts."""
        obs = pl.DataFrame({"_index": ["S1", "S2"]})
        var1 = pl.DataFrame({"_index": ["P1", "P2"]})
        var2 = pl.DataFrame({"_index": ["PEP1", "PEP2", "PEP3", "PEP4"]})
        assay1 = Assay(var=var1, layers={"X": ScpMatrix(X=np.random.rand(2, 2))})
        assay2 = Assay(var=var2, layers={"X": ScpMatrix(X=np.random.rand(2, 4))})
        container = ScpContainer(obs=obs, assays={"proteins": assay1, "peptides": assay2})
        assert container.assays["proteins"].n_features == 2
        assert container.assays["peptides"].n_features == 4

    def test_container_with_sparse_and_dense_assays(self):
        """Test container with mix of sparse and dense assays."""
        from scipy import sparse

        obs = pl.DataFrame({"_index": ["S1", "S2"]})
        var1 = pl.DataFrame({"_index": ["P1"]})
        var2 = pl.DataFrame({"_index": ["P2"]})
        X_dense = np.random.rand(2, 1)
        X_sparse = sparse.csr_matrix(np.random.rand(2, 1))
        assay1 = Assay(var=var1, layers={"X": ScpMatrix(X=X_dense)})
        assay2 = Assay(var=var2, layers={"X": ScpMatrix(X=X_sparse)})
        container = ScpContainer(obs=obs, assays={"dense": assay1, "sparse": assay2})
        assert not sparse.issparse(container.assays["dense"].layers["X"].X)
        assert sparse.issparse(container.assays["sparse"].layers["X"].X)

    def test_container_access_assay_by_name(self):
        """Test accessing assays by name."""
        obs = pl.DataFrame({"_index": ["S1"]})
        var = pl.DataFrame({"_index": ["P1"]})
        assay = Assay(var=var, layers={"X": ScpMatrix(X=np.random.rand(1, 1))})
        container = ScpContainer(obs=obs, assays={"my_assay": assay})
        assert "my_assay" in container.assays
        assert container.assays["my_assay"].n_features == 1


class TestScpContainerMaskPropagation:
    """Test mask code handling in container context."""

    def test_container_with_masked_matrices(self):
        """Test container with assays containing masked matrices."""
        obs = pl.DataFrame({"_index": ["S1", "S2"]})
        var = pl.DataFrame({"_index": ["P1", "P2", "P3"]})
        X = np.random.rand(2, 3)
        M = np.zeros((2, 3), dtype=np.int8)
        M[0, 0] = MaskCode.MBR
        M[1, 1] = MaskCode.IMPUTED
        assay = Assay(var=var, layers={"raw": ScpMatrix(X=X, M=M)})
        container = ScpContainer(obs=obs, assays={"test": assay})
        assert container.assays["test"].layers["raw"].M is not None
        assert np.sum(container.assays["test"].layers["raw"].M == MaskCode.IMPUTED) == 1
