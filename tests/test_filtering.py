"""Comprehensive tests for filter_samples and filter_features methods.

These tests cover the enhanced filtering functionality added to ScpContainer.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import scipy.sparse as sp

from scptensor.core.structures import Assay, ProvenanceLog, ScpContainer, ScpMatrix
from scptensor.core.exceptions import (
    AssayNotFoundError,
    DimensionError,
    ValidationError,
)

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
            "protein_id": np.random.choice(
                [f"protein_{i}" for i in range(n_proteins)], n_peptides
            ),
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
    filtered = container.filter_samples(polars_expression=pl.col("QC_pass") == True)

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
            assert layer.X.shape[0] == 3, (
                f"Assay {assay_name}, layer {layer_name} has wrong shape"
            )

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
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "Must specify one of" in str(e)

    print("✅ test_filter_samples_error_no_criteria passed")


def test_filter_samples_error_empty_result():
    """Test error when filtering results in zero samples."""
    container = create_test_container(n_samples=10)

    try:
        container.filter_samples(sample_indices=[])
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "zero samples" in str(e)

    print("✅ test_filter_samples_error_empty_result passed")


def test_filter_samples_error_invalid_mask_size():
    """Test error when boolean mask has wrong size."""
    container = create_test_container(n_samples=10)

    try:
        wrong_mask = np.array([True, False, True])  # Wrong size
        container.filter_samples(boolean_mask=wrong_mask)
        assert False, "Should have raised DimensionError"
    except DimensionError as e:
        assert "does not match" in str(e)

    print("✅ test_filter_samples_error_invalid_mask_size passed")


def test_filter_samples_error_unknown_id():
    """Test error when filtering by unknown sample ID."""
    container = create_test_container(n_samples=10)

    try:
        container.filter_samples(["sample_0", "unknown_sample"])
        assert False, "Should have raised ValueError"
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
    filtered = container.filter_features(
        "proteins", ["protein_0", "protein_5", "protein_10"]
    )

    assert filtered.assays["proteins"].n_features == 3
    assert (
        filtered.assays["proteins"].var["_index"].to_list()
        == ["protein_0", "protein_5", "protein_10"]
    )

    # Check layers are filtered
    assert filtered.assays["proteins"].layers["X"].X.shape[1] == 3
    assert filtered.assays["proteins"].layers["log"].X.shape[1] == 3

    # Check other assay is unchanged
    assert filtered.assays["peptides"].n_features == container.assays[
        "peptides"
    ].n_features

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
    assert (
        filtered.assays["proteins"].var["_index"].to_list()
        == ["protein_0", "protein_5", "protein_10", "protein_15"]
    )

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
    filtered = container.filter_features(
        "proteins", pl.col("mean_intensity") > 20
    )

    # Should have fewer features
    assert filtered.assays["proteins"].n_features <= 20

    print("✅ test_filter_features_by_polars_expression passed")


def test_filter_features_with_sparse_matrices():
    """Test filtering features with sparse matrices."""
    container = create_test_container(n_proteins=20, sparse=True)

    filtered = container.filter_features(
        "proteins", feature_indices=[0, 5, 10, 15, 19]
    )

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

    filtered = container.filter_features(
        "proteins", feature_indices=[0, 1, 2], copy=False
    )

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
        assert False, "Should have raised AssayNotFoundError"
    except AssayNotFoundError as e:
        assert "metabolites" in str(e)

    print("✅ test_filter_features_error_assay_not_found passed")


def test_filter_features_error_empty_result():
    """Test error when filtering results in zero features."""
    container = create_test_container(n_proteins=20)

    try:
        container.filter_features("proteins", feature_indices=[])
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "zero features" in str(e)

    print("✅ test_filter_features_error_empty_result passed")


def test_filter_features_error_invalid_mask_size():
    """Test error when boolean mask has wrong size."""
    container = create_test_container(n_proteins=20)

    try:
        wrong_mask = np.array([True, False, True])  # Wrong size
        container.filter_features("proteins", boolean_mask=wrong_mask)
        assert False, "Should have raised DimensionError"
    except DimensionError as e:
        assert "does not match" in str(e)

    print("✅ test_filter_features_error_invalid_mask_size passed")


def test_filter_features_error_unknown_id():
    """Test error when filtering by unknown feature ID."""
    container = create_test_container(n_proteins=20)

    try:
        container.filter_features("proteins", ["protein_0", "unknown_protein"])
        assert False, "Should have raised ValueError"
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
        container.filter_samples(polars_expression=pl.col("QC_pass") == True)
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
    filtered = container.filter_samples(sample_indices=[0, 1, 2])

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
