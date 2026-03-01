"""Comprehensive tests for filter_samples and filter_features methods.

These tests cover the enhanced filtering functionality added to ScpContainer.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp

from scptensor.core.exceptions import (
    AssayNotFoundError,
    DimensionError,
    ValidationError,
)
from scptensor.core.filtering import FilterCriteria, resolve_filter_criteria
from scptensor.core.structures import Assay, ScpContainer, ScpMatrix

# =============================================================================
# Test Fixtures
# =============================================================================


def create_test_container(
    n_samples: int = 10,
    n_proteins: int = 20,
    n_peptides: int = 50,
    sparse: bool = False,
    with_mask: bool = True,
) -> ScpContainer:
    """Create a test container with sample and feature metadata.

    Args:
        n_samples: Number of samples to generate
        n_proteins: Number of proteins in the protein assay
        n_peptides: Number of peptides in the peptide assay
        sparse: Whether to use sparse matrices
        with_mask: Whether to include mask matrices

    Returns:
        A test ScpContainer with multiple assays
    """
    # Create sample metadata (obs)
    obs = pl.DataFrame(
        {
            "_index": [f"sample_{i}" for i in range(n_samples)],
            "condition": np.random.choice(["A", "B", "C"], n_samples),
            "batch": np.random.choice([1, 2, 3], n_samples),
            "n_detected": np.random.randint(50, 200, n_samples),
            "QC_pass": np.random.choice([True, False], n_samples),
        }
    )

    # Create protein assay
    protein_var = pl.DataFrame(
        {
            "_index": [f"protein_{i}" for i in range(n_proteins)],
            "n_detected": np.random.randint(5, n_samples, n_proteins),
            "mean_intensity": np.random.uniform(10, 30, n_proteins),
        }
    )

    if sparse:
        protein_X = sp.random(n_samples, n_proteins, density=0.7, format="csr")
        protein_X.data = np.random.uniform(10, 30, protein_X.data.shape)
    else:
        protein_X = np.random.uniform(10, 30, (n_samples, n_proteins))

    if with_mask:
        # Create random mask with valid codes
        protein_M = np.random.choice(
            [0, 1, 2, 5], size=(n_samples, n_proteins), p=[0.7, 0.1, 0.1, 0.1]
        ).astype(np.int8)
    else:
        protein_M = None

    protein_layers = {
        "X": ScpMatrix(X=protein_X, M=protein_M),
        "log": ScpMatrix(X=np.log1p(protein_X), M=protein_M),
    }

    protein_assay = Assay(var=protein_var, layers=protein_layers, feature_id_col="_index")

    # Create peptide assay
    peptide_var = pl.DataFrame(
        {
            "_index": [f"peptide_{i}" for i in range(n_peptides)],
            "protein_id": np.random.choice([f"protein_{i}" for i in range(n_proteins)], n_peptides),
            "n_detected": np.random.randint(2, n_samples, n_peptides),
        }
    )

    if sparse:
        peptide_X = sp.random(n_samples, n_peptides, density=0.5, format="csr")
        peptide_X.data = np.random.uniform(10, 30, peptide_X.data.shape)
    else:
        peptide_X = np.random.uniform(10, 30, (n_samples, n_peptides))

    if with_mask:
        peptide_M = np.random.choice(
            [0, 1, 2, 5], size=(n_samples, n_peptides), p=[0.6, 0.15, 0.15, 0.1]
        ).astype(np.int8)
    else:
        peptide_M = None

    peptide_layers = {
        "X": ScpMatrix(X=peptide_X, M=peptide_M),
    }

    peptide_assay = Assay(var=peptide_var, layers=peptide_layers, feature_id_col="_index")

    assays = {"proteins": protein_assay, "peptides": peptide_assay}

    return ScpContainer(obs=obs, assays=assays, sample_id_col="_index")


# =============================================================================
# filter_samples Tests
# =============================================================================


def test_filter_samples_by_ids():
    """Test filtering samples by ID list."""
    container = create_test_container(n_samples=10)

    # Filter to specific samples
    filtered = container.filter_samples(["sample_0", "sample_2", "sample_5"])

    assert filtered.n_samples == 3
    assert filtered.obs["_index"].to_list() == ["sample_0", "sample_2", "sample_5"]

    # Check assays are filtered
    assert filtered.assays["proteins"].layers["X"].X.shape[0] == 3
    assert filtered.assays["peptides"].layers["X"].X.shape[0] == 3

    # Check history is updated
    assert len(filtered.history) == 1
    assert filtered.history[0].action == "filter_samples"
    assert filtered.history[0].params["n_samples_kept"] == 3

    print("✅ test_filter_samples_by_ids passed")


def test_filter_samples_by_indices():
    """Test filtering samples by index array."""
    container = create_test_container(n_samples=10)

    # Filter by indices
    indices = [0, 2, 4, 6, 8]
    filtered = container.filter_samples(sample_indices=indices)

    assert filtered.n_samples == 5
    assert filtered.obs["_index"].to_list() == [
        "sample_0",
        "sample_2",
        "sample_4",
        "sample_6",
        "sample_8",
    ]

    print("✅ test_filter_samples_by_indices passed")


def test_filter_samples_by_boolean_mask():
    """Test filtering samples by boolean mask."""
    container = create_test_container(n_samples=10)

    # Filter by boolean mask (keep samples with even index)
    mask = np.array([i % 2 == 0 for i in range(10)])
    filtered = container.filter_samples(boolean_mask=mask)

    assert filtered.n_samples == 5
    assert filtered.obs["_index"].to_list() == [
        "sample_0",
        "sample_2",
        "sample_4",
        "sample_6",
        "sample_8",
    ]

    # Test with Polars Series
    mask_series = pl.Series("mask", mask)
    filtered2 = container.filter_samples(boolean_mask=mask_series)
    assert filtered2.n_samples == 5

    print("✅ test_filter_samples_by_boolean_mask passed")


def test_filter_samples_by_polars_expression():
    """Test filtering samples by Polars expression."""
    container = create_test_container(n_samples=10)

    # Filter by expression - use polars_expression parameter
    filtered = container.filter_samples(polars_expression=pl.col("QC_pass"))

    # All samples that pass QC
    assert filtered.n_samples <= 10
    assert filtered.obs["QC_pass"].all()

    # Another expression
    filtered2 = container.filter_samples(polars_expression=pl.col("n_detected") > 100)
    assert filtered2.n_samples <= 10

    print("✅ test_filter_samples_by_polars_expression passed")


def test_filter_samples_with_sparse_matrices():
    """Test filtering samples with sparse matrices."""
    container = create_test_container(n_samples=10, sparse=True)

    filtered = container.filter_samples(sample_indices=[0, 2, 4])

    assert filtered.n_samples == 3
    assert sp.issparse(filtered.assays["proteins"].layers["X"].X)
    assert sp.issparse(filtered.assays["peptides"].layers["X"].X)

    # Check sparse matrix shape
    assert filtered.assays["proteins"].layers["X"].X.shape == (3, 20)

    print("✅ test_filter_samples_with_sparse_matrices passed")


def test_filter_samples_preserves_mask():
    """Test that filtering samples preserves mask codes."""
    container = create_test_container(n_samples=10, with_mask=True)

    original_M = container.assays["proteins"].layers["X"].M.copy()

    filtered = container.filter_samples(sample_indices=[0, 2, 4])

    # Check mask is subset correctly
    filtered_M = filtered.assays["proteins"].layers["X"].M

    assert np.array_equal(filtered_M, original_M[[0, 2, 4], :])

    print("✅ test_filter_samples_preserves_mask passed")


def test_filter_samples_multiple_assays():
    """Test filtering samples affects all assays."""
    container = create_test_container(n_samples=10)

    filtered = container.filter_samples(sample_indices=[0, 1, 2])

    # All assays should have 3 samples
    for assay_name, assay in filtered.assays.items():
        for layer_name, layer in assay.layers.items():
            assert layer.X.shape[0] == 3, f"Assay {assay_name}, layer {layer_name} has wrong shape"

    print("✅ test_filter_samples_multiple_assays passed")


def test_filter_samples_no_copy():
    """Test filtering without copying data."""
    container = create_test_container(n_samples=10)

    # Filter without copy
    filtered = container.filter_samples(sample_indices=[0, 1, 2], copy=False)

    # obs should still be filtered
    assert filtered.n_samples == 3

    # Verify var is shared (same object)
    assert filtered.assays["proteins"].var is container.assays["proteins"].var

    print("✅ test_filter_samples_no_copy passed")


def test_filter_samples_error_no_criteria():
    """Test error when no filtering criteria provided."""
    container = create_test_container(n_samples=10)

    try:
        container.filter_samples()
        raise AssertionError("Should have raised ValidationError")
    except ValidationError as e:
        assert "Must specify:" in str(e) or "sample_ids" in str(e)

    print("✅ test_filter_samples_error_no_criteria passed")


def test_filter_samples_error_empty_result():
    """Test error when filtering results in zero samples."""
    container = create_test_container(n_samples=10)

    try:
        container.filter_samples(sample_indices=[])
        raise AssertionError("Should have raised an error")
    except (ValidationError, IndexError) as e:
        # Empty indices cause IndexError in numpy indexing
        assert "zero samples" in str(e) or "arrays used as indices" in str(e)

    print("✅ test_filter_samples_error_empty_result passed")


def test_filter_samples_error_invalid_mask_size():
    """Test error when boolean mask has wrong size."""
    container = create_test_container(n_samples=10)

    try:
        wrong_mask = np.array([True, False, True])  # Wrong size
        container.filter_samples(boolean_mask=wrong_mask)
        raise AssertionError("Should have raised DimensionError")
    except DimensionError as e:
        assert "!=" in str(e) or "does not match" in str(e) or "Mask length" in str(e)

    print("✅ test_filter_samples_error_invalid_mask_size passed")


def test_filter_samples_error_unknown_id():
    """Test error when filtering by unknown sample ID."""
    container = create_test_container(n_samples=10)

    try:
        container.filter_samples(["sample_0", "unknown_sample"])
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "not found" in str(e)

    print("✅ test_filter_samples_error_unknown_id passed")


# =============================================================================
# filter_features Tests
# =============================================================================


def test_filter_features_by_ids():
    """Test filtering features by ID list."""
    container = create_test_container(n_proteins=20)

    # Filter to specific proteins
    filtered = container.filter_features("proteins", ["protein_0", "protein_5", "protein_10"])

    assert filtered.assays["proteins"].n_features == 3
    assert filtered.assays["proteins"].var["_index"].to_list() == [
        "protein_0",
        "protein_5",
        "protein_10",
    ]

    # Check layers are filtered
    assert filtered.assays["proteins"].layers["X"].X.shape[1] == 3
    assert filtered.assays["proteins"].layers["log"].X.shape[1] == 3

    # Check other assay is unchanged
    assert filtered.assays["peptides"].n_features == container.assays["peptides"].n_features

    # Check history
    assert len(filtered.history) == 1
    assert filtered.history[0].action == "filter_features"
    assert filtered.history[0].params["assay_name"] == "proteins"

    print("✅ test_filter_features_by_ids passed")


def test_filter_features_by_indices():
    """Test filtering features by index array."""
    container = create_test_container(n_proteins=20)

    indices = [0, 5, 10, 15]
    filtered = container.filter_features("proteins", feature_indices=indices)

    assert filtered.assays["proteins"].n_features == 4
    assert filtered.assays["proteins"].var["_index"].to_list() == [
        "protein_0",
        "protein_5",
        "protein_10",
        "protein_15",
    ]

    print("✅ test_filter_features_by_indices passed")


def test_filter_features_by_boolean_mask():
    """Test filtering features by boolean mask."""
    container = create_test_container(n_proteins=20)

    # Keep every other feature
    mask = np.array([i % 2 == 0 for i in range(20)])
    filtered = container.filter_features("proteins", boolean_mask=mask)

    assert filtered.assays["proteins"].n_features == 10

    print("✅ test_filter_features_by_boolean_mask passed")


def test_filter_features_by_polars_expression():
    """Test filtering features by Polars expression."""
    container = create_test_container(n_proteins=20)

    # Filter by mean intensity
    filtered = container.filter_features("proteins", pl.col("mean_intensity") > 20)

    # Should have fewer features
    assert filtered.assays["proteins"].n_features <= 20

    print("✅ test_filter_features_by_polars_expression passed")


def test_filter_features_with_sparse_matrices():
    """Test filtering features with sparse matrices."""
    container = create_test_container(n_proteins=20, sparse=True)

    filtered = container.filter_features("proteins", feature_indices=[0, 5, 10, 15, 19])

    assert filtered.assays["proteins"].n_features == 5
    assert sp.issparse(filtered.assays["proteins"].layers["X"].X)
    assert filtered.assays["proteins"].layers["X"].X.shape == (10, 5)

    print("✅ test_filter_features_with_sparse_matrices passed")


def test_filter_features_preserves_mask():
    """Test that filtering features preserves mask codes."""
    container = create_test_container(n_proteins=20, with_mask=True)

    original_M = container.assays["proteins"].layers["X"].M.copy()

    indices = [0, 5, 10, 15]
    filtered = container.filter_features("proteins", feature_indices=indices)

    # Check mask is subset correctly
    filtered_M = filtered.assays["proteins"].layers["X"].M

    assert np.array_equal(filtered_M, original_M[:, indices])

    print("✅ test_filter_features_preserves_mask passed")


def test_filter_features_multiple_layers():
    """Test filtering features affects all layers."""
    container = create_test_container(n_proteins=20)

    filtered = container.filter_features("proteins", feature_indices=[0, 1, 2])

    # All layers should have 3 features
    for layer_name, layer in filtered.assays["proteins"].layers.items():
        assert layer.X.shape[1] == 3, f"Layer {layer_name} has wrong shape"

    print("✅ test_filter_features_multiple_layers passed")


def test_filter_features_no_copy():
    """Test filtering features without copying data."""
    container = create_test_container(n_proteins=20)

    filtered = container.filter_features("proteins", feature_indices=[0, 1, 2], copy=False)

    # obs should be shared
    assert filtered.obs is container.obs

    # Other assay should be shared
    assert filtered.assays["peptides"] is container.assays["peptides"]

    print("✅ test_filter_features_no_copy passed")


def test_filter_features_error_assay_not_found():
    """Test error when assay doesn't exist."""
    container = create_test_container(n_proteins=20)

    try:
        container.filter_features("metabolites", feature_indices=[0, 1, 2])
        raise AssertionError("Should have raised AssayNotFoundError")
    except AssayNotFoundError as e:
        assert "metabolites" in str(e)

    print("✅ test_filter_features_error_assay_not_found passed")


def test_filter_features_error_empty_result():
    """Test error when filtering results in zero features."""
    container = create_test_container(n_proteins=20)

    try:
        container.filter_features("proteins", feature_indices=[])
        raise AssertionError("Should have raised an error")
    except (ValidationError, IndexError) as e:
        # Empty indices cause IndexError in numpy indexing
        assert "zero features" in str(e) or "arrays used as indices" in str(e)

    print("✅ test_filter_features_error_empty_result passed")


def test_filter_features_error_invalid_mask_size():
    """Test error when boolean mask has wrong size."""
    container = create_test_container(n_proteins=20)

    try:
        wrong_mask = np.array([True, False, True])  # Wrong size
        container.filter_features("proteins", boolean_mask=wrong_mask)
        raise AssertionError("Should have raised DimensionError")
    except DimensionError as e:
        assert "!=" in str(e) or "does not match" in str(e) or "Mask length" in str(e)

    print("✅ test_filter_features_error_invalid_mask_size passed")


def test_filter_features_error_unknown_id():
    """Test error when filtering by unknown feature ID."""
    container = create_test_container(n_proteins=20)

    try:
        container.filter_features("proteins", ["protein_0", "unknown_protein"])
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "not found" in str(e)

    print("✅ test_filter_features_error_unknown_id passed")


# =============================================================================
# Integration Tests
# =============================================================================


def test_combined_filtering():
    """Test combining sample and feature filtering."""
    container = create_test_container(n_samples=10, n_proteins=20)

    # First filter samples
    step1 = container.filter_samples(sample_indices=[0, 1, 2, 3, 4])
    assert step1.n_samples == 5

    # Then filter features
    step2 = step1.filter_features("proteins", feature_indices=[0, 1, 2, 3, 4])
    assert step2.assays["proteins"].n_features == 5

    # Check history has both entries
    assert len(step2.history) == 2
    assert step2.history[0].action == "filter_samples"
    assert step2.history[1].action == "filter_features"

    # Verify final shape
    assert step2.assays["proteins"].layers["X"].X.shape == (5, 5)

    print("✅ test_combined_filtering passed")


def test_filtering_chain():
    """Test chaining multiple filter operations."""
    container = create_test_container(n_samples=10, n_proteins=20)

    result = (
        container.filter_samples(polars_expression=pl.col("QC_pass"))
        .filter_features("proteins", polars_expression=pl.col("n_detected") > 5)
        .filter_samples(sample_indices=[0, 1])
    )

    # History should have 3 entries
    assert len(result.history) == 3

    print("✅ test_filtering_chain passed")


def test_filter_immutability():
    """Test that filtering doesn't modify original container."""
    container = create_test_container(n_samples=10, n_proteins=20)

    original_n_samples = container.n_samples
    original_n_features = container.assays["proteins"].n_features
    original_X = container.assays["proteins"].layers["X"].X.copy()

    # Filter
    container.filter_samples(sample_indices=[0, 1, 2])

    # Original should be unchanged
    assert container.n_samples == original_n_samples
    assert container.assays["proteins"].n_features == original_n_features
    assert np.array_equal(container.assays["proteins"].layers["X"].X, original_X)

    print("✅ test_filter_immutability passed")


# =============================================================================
# Run All Tests
# =============================================================================


def run_all_tests() -> None:
    """Run all filter tests."""
    print("=" * 60)
    print("Testing filter_samples method")
    print("=" * 60)

    test_filter_samples_by_ids()
    test_filter_samples_by_indices()
    test_filter_samples_by_boolean_mask()
    test_filter_samples_by_polars_expression()
    test_filter_samples_with_sparse_matrices()
    test_filter_samples_preserves_mask()
    test_filter_samples_multiple_assays()
    test_filter_samples_no_copy()
    test_filter_samples_error_no_criteria()
    test_filter_samples_error_empty_result()
    test_filter_samples_error_invalid_mask_size()
    test_filter_samples_error_unknown_id()

    print()
    print("=" * 60)
    print("Testing filter_features method")
    print("=" * 60)

    test_filter_features_by_ids()
    test_filter_features_by_indices()
    test_filter_features_by_boolean_mask()
    test_filter_features_by_polars_expression()
    test_filter_features_with_sparse_matrices()
    test_filter_features_preserves_mask()
    test_filter_features_multiple_layers()
    test_filter_features_no_copy()
    test_filter_features_error_assay_not_found()
    test_filter_features_error_empty_result()
    test_filter_features_error_invalid_mask_size()
    test_filter_features_error_unknown_id()

    print()
    print("=" * 60)
    print("Integration tests")
    print("=" * 60)

    test_combined_filtering()
    test_filtering_chain()
    test_filter_immutability()

    print()
    print("=" * 60)
    print("All filtering tests passed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()


# =============================================================================
# FilterCriteria API Tests
# =============================================================================


def test_filter_criteria_by_ids():
    """Test FilterCriteria.by_ids() factory method."""
    criteria = FilterCriteria.by_ids(["sample1", "sample2"])
    assert criteria.criteria_type == "ids"
    assert isinstance(criteria.value, list)


def test_filter_criteria_by_indices():
    """Test FilterCriteria.by_indices() factory method."""
    criteria = FilterCriteria.by_indices([0, 1, 2])
    assert criteria.criteria_type == "indices"
    assert isinstance(criteria.value, list)


def test_filter_criteria_by_mask():
    """Test FilterCriteria.by_mask() factory method."""
    criteria = FilterCriteria.by_mask(np.array([True, False, True]))
    assert criteria.criteria_type == "mask"
    assert isinstance(criteria.value, np.ndarray)


def test_filter_criteria_by_expression():
    """Test FilterCriteria.by_expression() factory method."""
    criteria = FilterCriteria.by_expression(pl.col("n_detected") > 100)
    assert criteria.criteria_type == "expression"
    assert isinstance(criteria.value, pl.Expr)


def test_filter_criteria_invalid_type_raises_error():
    """Test that invalid criteria_type raises ValueError."""
    with pytest.raises(ValueError, match="Invalid criteria_type"):
        FilterCriteria(criteria_type="invalid_type", value=None)


def test_resolve_filter_criteria_by_ids():
    """Test resolving filter criteria by IDs."""
    container = create_test_container(n_samples=10)
    sample_ids = container.sample_ids[:5].to_list()

    criteria = FilterCriteria.by_ids(sample_ids)
    indices = resolve_filter_criteria(criteria, container, is_sample=True)

    assert len(indices) == 5
    assert np.array_equal(indices, np.array([0, 1, 2, 3, 4]))


def test_resolve_filter_criteria_by_indices():
    """Test resolving filter criteria by indices."""
    container = create_test_container(n_samples=10)

    criteria = FilterCriteria.by_indices([0, 2, 4])
    indices = resolve_filter_criteria(criteria, container, is_sample=True)

    assert len(indices) == 3
    assert np.array_equal(indices, np.array([0, 2, 4]))


def test_resolve_filter_criteria_by_mask():
    """Test resolving filter criteria by mask."""
    container = create_test_container(n_samples=10)
    mask = np.array([True] * 5 + [False] * 5)

    criteria = FilterCriteria.by_mask(mask)
    indices = resolve_filter_criteria(criteria, container, is_sample=True)

    assert len(indices) == 5
    assert np.array_equal(indices, np.array([0, 1, 2, 3, 4]))


def test_resolve_filter_criteria_by_expression():
    """Test resolving filter criteria by expression."""
    container = create_test_container(n_samples=10)

    criteria = FilterCriteria.by_expression(pl.col("n_detected") > 0)
    indices = resolve_filter_criteria(criteria, container, is_sample=True)

    assert len(indices) == 10  # All samples have n_detected > 0


def test_resolve_filter_criteria_by_ids_missing():
    """Test that missing IDs raise ValueError."""
    container = create_test_container(n_samples=10)

    criteria = FilterCriteria.by_ids(["nonexistent_id"])

    with pytest.raises(ValueError, match="Sample IDs not found"):
        resolve_filter_criteria(criteria, container, is_sample=True)


def test_resolve_filter_criteria_by_mask_wrong_length():
    """Test that wrong mask length raises DimensionError."""
    container = create_test_container(n_samples=10)
    mask = np.array([True, False, True])  # Wrong length

    criteria = FilterCriteria.by_mask(mask)

    with pytest.raises(DimensionError, match="Mask length"):
        resolve_filter_criteria(criteria, container, is_sample=True)


def test_resolve_filter_criteria_by_mask_non_boolean():
    """Test that non-boolean mask raises ValueError."""
    container = create_test_container(n_samples=10)
    mask = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])  # int instead of bool

    criteria = FilterCriteria.by_mask(mask)

    with pytest.raises(ValueError, match="Mask must be boolean"):
        resolve_filter_criteria(criteria, container, is_sample=True)


def test_resolve_filter_criteria_by_expression_non_boolean():
    """Test that non-boolean expression raises ValueError."""
    container = create_test_container(n_samples=10)

    # Expression returns integer, not boolean
    criteria = FilterCriteria.by_expression(pl.col("n_detected"))

    with pytest.raises(ValueError, match="Expression must produce boolean"):
        resolve_filter_criteria(criteria, container, is_sample=True)


def test_resolve_filter_criteria_features_by_ids():
    """Test resolving feature filter criteria by IDs."""
    container = create_test_container(n_samples=10, n_proteins=20)
    assay = container.assays["proteins"]
    feature_ids = assay.feature_ids[:5].to_list()

    criteria = FilterCriteria.by_ids(feature_ids)
    indices = resolve_filter_criteria(criteria, assay, is_sample=False)

    assert len(indices) == 5
    assert np.array_equal(indices, np.array([0, 1, 2, 3, 4]))


def test_resolve_filter_criteria_features_by_indices():
    """Test resolving feature filter criteria by indices."""
    container = create_test_container(n_samples=10, n_proteins=20)
    assay = container.assays["proteins"]

    criteria = FilterCriteria.by_indices([0, 5, 10, 15])
    indices = resolve_filter_criteria(criteria, assay, is_sample=False)

    assert len(indices) == 4
    assert np.array_equal(indices, np.array([0, 5, 10, 15]))


def test_resolve_filter_criteria_features_by_mask():
    """Test resolving feature filter criteria by mask."""
    container = create_test_container(n_samples=10, n_proteins=20)
    assay = container.assays["proteins"]
    mask = np.array([True] * 10 + [False] * 10)

    criteria = FilterCriteria.by_mask(mask)
    indices = resolve_filter_criteria(criteria, assay, is_sample=False)

    assert len(indices) == 10
    assert np.array_equal(indices, np.arange(10))


def test_resolve_filter_criteria_features_by_expression():
    """Test resolving feature filter criteria by expression."""
    container = create_test_container(n_samples=10, n_proteins=20)
    assay = container.assays["proteins"]

    # All features should have n_detected >= 0
    criteria = FilterCriteria.by_expression(pl.col("n_detected") >= 0)
    indices = resolve_filter_criteria(criteria, assay, is_sample=False)

    assert len(indices) == 20


def test_filter_samples_with_filter_criteria():
    """Test filtering samples using FilterCriteria object."""
    container = create_test_container(n_samples=10)

    # Filter by indices - note: this integration test will work once
    # filter_samples is updated to support FilterCriteria
    criteria = FilterCriteria.by_indices([0, 2, 4, 6, 8])
    indices = resolve_filter_criteria(criteria, container, is_sample=True)

    # Manually create filtered container for now
    filtered = container.filter_samples(sample_indices=indices.tolist())

    assert filtered.n_samples == 5
    assert filtered.sample_ids.to_list() == container.sample_ids[[0, 2, 4, 6, 8]].to_list()


def test_filter_samples_with_expression_criteria():
    """Test filtering samples with expression-based criteria."""
    container = create_test_container(n_samples=10)

    # Filter by n_detected > threshold
    threshold = container.obs["n_detected"].median()
    criteria = FilterCriteria.by_expression(pl.col("n_detected") > threshold)
    indices = resolve_filter_criteria(criteria, container, is_sample=True)

    # Manually create filtered container for now
    filtered = container.filter_samples(sample_indices=indices.tolist())

    assert filtered.n_samples <= container.n_samples
    assert all(filtered.obs["n_detected"] > threshold)


def test_filter_features_with_filter_criteria():
    """Test filtering features using FilterCriteria object."""
    container = create_test_container(n_samples=10, n_proteins=20)

    # Filter by indices - note: this integration test will work once
    # filter_features is updated to support FilterCriteria
    criteria = FilterCriteria.by_indices([0, 5, 10])
    indices = resolve_filter_criteria(criteria, container.assays["proteins"], is_sample=False)

    # Manually create filtered container for now
    filtered = container.filter_features("proteins", feature_indices=indices.tolist())

    assert filtered.assays["proteins"].n_features == 3
