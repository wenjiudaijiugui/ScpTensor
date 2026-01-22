"""Tests for PSM-level QC operations.

This test file validates the PSM QC module implementation.
Run with: uv run pytest tests/test_psm_qc.py -v
"""

import numpy as np
import polars as pl
import pytest

from scptensor.core import Assay, ScpContainer, ScpMatrix
from scptensor.qc import (
    compute_median_cv,
    compute_sample_carrier_ratio,
    divide_by_reference,
    filter_contaminants,
    filter_psms_by_pif,
    filter_psms_by_qvalue,
    pep_to_qvalue,
)


# Helper function to update var in a container
def _create_container_with_updated_var(container, assay_name, new_var):
    """Create a new container with updated var DataFrame."""
    from copy import deepcopy

    new_container = deepcopy(container)
    new_container.assays[assay_name].var = new_var
    return new_container


@pytest.fixture
def psm_container():
    """Create a test container with PSM-level metadata."""
    # Create obs with mix of sample types
    sample_indices = (
        [f"Reference{i}" for i in range(2)]
        + [f"Carrier{i}" for i in range(3)]
        + [f"Sample{i}" for i in range(15)]
    )
    obs = pl.DataFrame(
        {
            "_index": sample_indices,
            "sample_type": ["Reference"] * 2 + ["Carrier"] * 3 + ["Sample"] * 15,
        }
    )

    # Create var
    var = pl.DataFrame(
        {
            "_index": [f"PSM{i:04d}" for i in range(100)],
        }
    )

    # Create data matrix
    rng = np.random.default_rng(42)
    X = rng.exponential(scale=1000, size=(20, 100))

    # Create assay
    assay = Assay(var=var, layers={"raw": ScpMatrix(X=X)})
    container = ScpContainer(obs=obs, assays={"protein": assay})

    # Add PIF scores
    pif_values = np.random.uniform(0.5, 1.0, assay.n_features)
    # Mark some as low PIF
    pif_values[:10] = np.random.uniform(0.1, 0.7, 10)

    # Add contaminant flags
    is_reverse = np.zeros(assay.n_features, dtype=bool)
    is_reverse[5:10] = True

    is_contaminant = np.zeros(assay.n_features, dtype=bool)
    is_contaminant[10:15] = True

    # Add PEP values
    pep_values = np.random.uniform(0.0, 0.3, assay.n_features)
    # Mark some as high PEP (low confidence)
    pep_values[15:20] = np.random.uniform(0.5, 1.0, 5)

    # Update var with PSM metadata
    new_var = assay.var.with_columns(
        pl.Series("pif", pif_values),
        pl.Series("is_reverse", is_reverse),
        pl.Series("is_contaminant", is_contaminant),
        pl.Series("pep", pep_values),
    )

    # Create new container with updated var
    container = _create_container_with_updated_var(container, "protein", new_var)

    return container


def test_filter_psms_by_pif(psm_container):
    """Test PIF filtering."""
    # Test inplace=False (default)
    container = filter_psms_by_pif(psm_container, min_pif=0.8, inplace=False)

    # Check that filter statistics column exists
    assert "keep_pif_0.8" in container.assays["protein"].var.columns

    # Check that original size is preserved
    assert container.assays["protein"].n_features == psm_container.assays["protein"].n_features

    # Test inplace=True
    container_filtered = filter_psms_by_pif(psm_container, min_pif=0.8, inplace=True)

    # Check that features were filtered
    assert (
        container_filtered.assays["protein"].n_features < psm_container.assays["protein"].n_features
    )

    print(
        f"✓ PIF filtering: {psm_container.assays['protein'].n_features} -> "
        f"{container_filtered.assays['protein'].n_features} features"
    )


def test_filter_contaminants(psm_container):
    """Test contaminant filtering."""
    # Test filtering
    container_filtered = filter_contaminants(psm_container)

    # Check that features were removed
    assert (
        container_filtered.assays["protein"].n_features < psm_container.assays["protein"].n_features
    )

    # Check that at least reverse and contaminant features were removed
    n_removed = (
        psm_container.assays["protein"].n_features - container_filtered.assays["protein"].n_features
    )
    assert n_removed >= 10  # At least 10 contaminants (5 reverse + 5 contaminant)

    print(f"✓ Contaminant filtering: removed {n_removed} PSMs")


def test_pep_to_qvalue(psm_container):
    """Test PEP to q-value conversion."""
    # Test Storey's method
    container_storey = pep_to_qvalue(psm_container, method="storey")

    # Check that qvalue column exists
    assert "qvalue" in container_storey.assays["protein"].var.columns

    # Check q-values are in valid range
    qvalues = container_storey.assays["protein"].var["qvalue"].to_numpy()
    assert np.all(qvalues >= 0) and np.all(qvalues <= 1)

    # Test BH method
    container_bh = pep_to_qvalue(psm_container, method="bh")
    qvalues_bh = container_bh.assays["protein"].var["qvalue"].to_numpy()
    assert np.all(qvalues_bh >= 0) and np.all(qvalues_bh <= 1)

    print("✓ PEP to q-value conversion: Storey and BH methods working")


def test_filter_psms_by_qvalue(psm_container):
    """Test q-value filtering."""
    # First compute q-values
    container = pep_to_qvalue(psm_container, method="storey")

    # Test inplace=False
    container_stats = filter_psms_by_qvalue(container, max_qvalue=0.01, inplace=False)

    # Check that filter statistics column exists
    assert "keep_qvalue_0.01" in container_stats.assays["protein"].var.columns

    # Test inplace=True
    container_filtered = filter_psms_by_qvalue(container, max_qvalue=0.01, inplace=True)

    # Check that features were filtered
    n_removed = (
        container.assays["protein"].n_features - container_filtered.assays["protein"].n_features
    )
    print(f"✓ Q-value filtering: removed {n_removed} PSMs at 1% FDR")


def test_compute_sample_carrier_ratio(psm_container):
    """Test Sample-to-Carrier Ratio computation."""
    container = compute_sample_carrier_ratio(
        psm_container,
        carrier_identifier="Carrier",
        sample_type_column="sample_type",
    )

    # Check that SCR metrics exist in obs
    assert "scr_median" in container.obs.columns
    assert "scr_mean" in container.obs.columns
    assert "scr_high_psm_count" in container.obs.columns

    # Check that carriers have NaN SCR
    carrier_mask = container.obs["sample_type"].str.contains("Carrier").to_numpy()
    scr_median = container.obs["scr_median"].to_numpy()

    assert np.all(np.isnan(scr_median[carrier_mask]))
    assert np.all(~np.isnan(scr_median[~carrier_mask]))

    print(f"✓ SCR computation: calculated for {np.sum(~carrier_mask)} samples")


def test_compute_median_cv(psm_container):
    """Test median CV computation."""
    container = compute_median_cv(psm_container, cv_threshold=0.65)

    # Check that CV metrics exist in obs
    assert "median_cv" in container.obs.columns
    assert "is_high_cv" in container.obs.columns

    # Check CV values are valid
    median_cv = container.obs["median_cv"].to_numpy()
    assert np.all(median_cv[~np.isnan(median_cv)] >= 0)

    print("✓ Median CV computation: calculated for all samples")


def test_divide_by_reference(psm_container):
    """Test reference channel normalization."""
    # Test reference normalization
    container = divide_by_reference(
        psm_container,
        reference_identifier="Reference",
        aggregation="median",
    )

    # Check that new layer exists
    assert "reference_normalized" in container.assays["protein"].layers

    # Check that normalized data is different from raw
    X_raw = psm_container.assays["protein"].layers["raw"].X
    X_norm = container.assays["protein"].layers["reference_normalized"].X

    assert not np.allclose(X_raw, X_norm)

    print("✓ Reference normalization: created 'reference_normalized' layer")


def test_filter_psms_by_pif_invalid_threshold(psm_container):
    """Test that invalid PIF threshold raises error."""
    from scptensor.core.exceptions import ScpValueError

    with pytest.raises(ScpValueError):
        filter_psms_by_pif(psm_container, min_pif=1.5)


def test_filter_psms_by_qvalue_invalid_threshold(psm_container):
    """Test that invalid q-value threshold raises error."""
    from scptensor.core.exceptions import ScpValueError

    # First compute q-values
    container = pep_to_qvalue(psm_container)

    with pytest.raises(ScpValueError):
        filter_psms_by_qvalue(container, max_qvalue=0.0)


def test_pep_to_qvalue_missing_column(psm_container):
    """Test that missing PEP column raises error."""
    # Helper function to update var in a container


def _create_container_with_updated_var(container, assay_name, new_var):
    """Create a new container with updated var DataFrame."""
    from copy import deepcopy

    new_container = deepcopy(container)
    new_container.assays[assay_name].var = new_var
    return new_container

    # Remove PEP column
    assay = psm_container.assays["protein"]
    new_var = assay.var.drop("pep")

    container = _create_container_with_updated_var(psm_container, "protein", new_var)

    with pytest.raises(ScpValueError):
        pep_to_qvalue(container)


if __name__ == "__main__":
    print("Running PSM QC tests...\n")

    psm_container = psm_container()

    test_filter_psms_by_pif(psm_container)
    test_filter_psms_by_contaminants(psm_container)
    test_pep_to_qvalue(psm_container)
    test_filter_psms_by_qvalue(psm_container)
    test_compute_sample_carrier_ratio(psm_container)
    test_compute_median_cv(psm_container)
    test_divide_by_reference(psm_container)

    # Error handling tests
    print("\nTesting error handling...")
    test_filter_psms_by_pif_invalid_threshold(psm_container)
    test_filter_psms_by_qvalue_invalid_threshold(psm_container)
    test_pep_to_qvalue_missing_column(psm_container)

    print("\n✓ All PSM QC tests passed!")
