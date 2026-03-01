"""Migration tests verifying new FilterCriteria API produces same results as legacy API.

These tests ensure 100% backward compatibility by comparing outputs from the new
FilterCriteria API with the legacy API.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from scptensor.core.filtering import FilterCriteria
from scptensor.core.structures import ScpContainer


def create_test_container(
    n_samples: int = 20,
    n_proteins: int = 30,
    seed: int = 42,
) -> ScpContainer:
    """Create a reproducible test container."""
    rng = np.random.default_rng(seed)

    obs = pl.DataFrame(
        {
            "_index": [f"sample_{i}" for i in range(n_samples)],
            "condition": rng.choice(["A", "B", "C"], n_samples),
            "batch": rng.choice([1, 2, 3], n_samples),
            "n_detected": rng.integers(50, 200, n_samples),
            "QC_pass": rng.choice([True, False], n_samples),
        }
    )

    protein_var = pl.DataFrame(
        {
            "_index": [f"protein_{i}" for i in range(n_proteins)],
            "n_detected": rng.integers(5, n_samples, n_proteins),
            "mean_intensity": rng.uniform(10, 30, n_proteins),
        }
    )

    from scptensor.core.structures import Assay, ScpMatrix

    protein_X = rng.uniform(10, 30, (n_samples, n_proteins))
    protein_M = rng.choice(
        [0, 1, 2, 5], size=(n_samples, n_proteins), p=[0.7, 0.1, 0.1, 0.1]
    ).astype(np.int8)

    protein_layers = {"X": ScpMatrix(X=protein_X, M=protein_M)}
    protein_assay = Assay(var=protein_var, layers=protein_layers, feature_id_col="_index")

    assays = {"proteins": protein_assay}
    return ScpContainer(obs=obs, assays=assays, sample_id_col="_index")


def assert_containers_equal(
    container1: ScpContainer,
    container2: ScpContainer,
    msg: str = "Containers should be equal",
) -> None:
    """Assert two containers have equal data."""
    assert container1.n_samples == container2.n_samples, f"{msg}: n_samples differ"
    assert len(container1.assays) == len(container2.assays), f"{msg}: number of assays differ"

    # Compare sample IDs
    assert container1.sample_ids.to_list() == container2.sample_ids.to_list(), (
        f"{msg}: sample_ids differ"
    )

    # Compare obs DataFrame
    assert container1.obs.equals(container2.obs), f"{msg}: obs DataFrames differ"

    # Compare assays
    for assay_name in container1.assays:
        assay1 = container1.assays[assay_name]
        assay2 = container2.assays[assay_name]

        assert assay1.n_features == assay2.n_features, f"{msg}: {assay_name} n_features differ"
        assert assay1.feature_ids.to_list() == assay2.feature_ids.to_list(), (
            f"{msg}: {assay_name} feature_ids differ"
        )
        assert assay1.var.equals(assay2.var), f"{msg}: {assay_name} var DataFrames differ"

        # Compare data layers
        for layer_name in assay1.layers:
            layer1 = assay1.layers[layer_name]
            layer2 = assay2.layers[layer_name]

            assert np.array_equal(layer1.X, layer2.X), f"{msg}: {assay_name}.{layer_name} X differs"
            if layer1.M is not None and layer2.M is not None:
                assert np.array_equal(layer1.M, layer2.M), (
                    f"{msg}: {assay_name}.{layer_name} M differs"
                )


class TestFilterSamplesMigration:
    """Test migration from legacy to new filter_samples API."""

    def test_filter_samples_by_ids_migration(self):
        """Test that FilterCriteria.by_ids produces same result as legacy ID list."""
        container = create_test_container(n_samples=20)
        sample_ids = ["sample_0", "sample_2", "sample_5", "sample_10", "sample_15"]

        # Legacy API
        legacy_result = container.filter_samples(sample_ids)

        # New API
        criteria = FilterCriteria.by_ids(sample_ids)
        new_result = container.filter_samples(criteria)

        assert_containers_equal(legacy_result, new_result, "filter_samples by IDs")

    def test_filter_samples_by_indices_migration(self):
        """Test that FilterCriteria.by_indices produces same result as legacy sample_indices."""
        container = create_test_container(n_samples=20)
        indices = [0, 2, 5, 10, 15]

        # Legacy API
        legacy_result = container.filter_samples(sample_indices=indices)

        # New API
        criteria = FilterCriteria.by_indices(indices)
        new_result = container.filter_samples(criteria)

        assert_containers_equal(legacy_result, new_result, "filter_samples by indices")

    def test_filter_samples_by_mask_migration(self):
        """Test that FilterCriteria.by_mask produces same result as legacy boolean_mask."""
        container = create_test_container(n_samples=20)
        mask = np.array([i % 2 == 0 for i in range(20)])

        # Legacy API
        legacy_result = container.filter_samples(boolean_mask=mask)

        # New API
        criteria = FilterCriteria.by_mask(mask)
        new_result = container.filter_samples(criteria)

        assert_containers_equal(legacy_result, new_result, "filter_samples by mask")

    def test_filter_samples_by_expression_migration(self):
        """Test that FilterCriteria.by_expression produces same result as legacy polars_expression."""
        container = create_test_container(n_samples=20)

        # Legacy API
        legacy_result = container.filter_samples(polars_expression=pl.col("QC_pass"))

        # New API
        criteria = FilterCriteria.by_expression(pl.col("QC_pass"))
        new_result = container.filter_samples(criteria)

        assert_containers_equal(legacy_result, new_result, "filter_samples by expression")

    def test_filter_samples_complex_expression_migration(self):
        """Test complex Polars expression migration."""
        container = create_test_container(n_samples=20)
        expr = (pl.col("n_detected") > 100) & (pl.col("batch") == 1)

        # Legacy API
        legacy_result = container.filter_samples(polars_expression=expr)

        # New API
        criteria = FilterCriteria.by_expression(expr)
        new_result = container.filter_samples(criteria)

        assert_containers_equal(legacy_result, new_result, "filter_samples by complex expression")


class TestFilterFeaturesMigration:
    """Test migration from legacy to new filter_features API."""

    def test_filter_features_by_ids_migration(self):
        """Test that FilterCriteria.by_ids produces same result as legacy ID list."""
        container = create_test_container(n_proteins=30)
        feature_ids = ["protein_0", "protein_5", "protein_10", "protein_20", "protein_25"]

        # Legacy API
        legacy_result = container.filter_features("proteins", feature_ids)

        # New API
        criteria = FilterCriteria.by_ids(feature_ids)
        new_result = container.filter_features("proteins", criteria)

        assert_containers_equal(legacy_result, new_result, "filter_features by IDs")

    def test_filter_features_by_indices_migration(self):
        """Test that FilterCriteria.by_indices produces same result as legacy feature_indices."""
        container = create_test_container(n_proteins=30)
        indices = [0, 5, 10, 20, 25]

        # Legacy API
        legacy_result = container.filter_features("proteins", feature_indices=indices)

        # New API
        criteria = FilterCriteria.by_indices(indices)
        new_result = container.filter_features("proteins", criteria)

        assert_containers_equal(legacy_result, new_result, "filter_features by indices")

    def test_filter_features_by_mask_migration(self):
        """Test that FilterCriteria.by_mask produces same result as legacy boolean_mask."""
        container = create_test_container(n_proteins=30)
        mask = np.array([i % 3 == 0 for i in range(30)])

        # Legacy API
        legacy_result = container.filter_features("proteins", boolean_mask=mask)

        # New API
        criteria = FilterCriteria.by_mask(mask)
        new_result = container.filter_features("proteins", criteria)

        assert_containers_equal(legacy_result, new_result, "filter_features by mask")

    def test_filter_features_by_expression_migration(self):
        """Test that FilterCriteria.by_expression produces same result as legacy polars_expression."""
        container = create_test_container(n_proteins=30)

        # Legacy API
        legacy_result = container.filter_features(
            "proteins", polars_expression=pl.col("mean_intensity") > 20
        )

        # New API
        criteria = FilterCriteria.by_expression(pl.col("mean_intensity") > 20)
        new_result = container.filter_features("proteins", criteria)

        assert_containers_equal(legacy_result, new_result, "filter_features by expression")

    def test_filter_features_complex_expression_migration(self):
        """Test complex Polars expression migration."""
        container = create_test_container(n_proteins=30)
        expr = (pl.col("mean_intensity") > 15) & (pl.col("n_detected") > 10)

        # Legacy API
        legacy_result = container.filter_features("proteins", polars_expression=expr)

        # New API
        criteria = FilterCriteria.by_expression(expr)
        new_result = container.filter_features("proteins", criteria)

        assert_containers_equal(legacy_result, new_result, "filter_features by complex expression")


class TestChainedFilteringMigration:
    """Test that chained filtering works with both APIs."""

    def test_chained_filtering_mixed_api(self):
        """Test that we can mix legacy and new API in filtering chain."""
        container = create_test_container(n_samples=20, n_proteins=30)

        # All legacy
        result1 = (
            container.filter_samples(polars_expression=pl.col("QC_pass"))
            .filter_features("proteins", pl.col("mean_intensity") > 20)
            .filter_samples(sample_indices=[0, 1])
        )

        # All new
        result2 = (
            container.filter_samples(FilterCriteria.by_expression(pl.col("QC_pass")))
            .filter_features(
                "proteins", FilterCriteria.by_expression(pl.col("mean_intensity") > 20)
            )
            .filter_samples(FilterCriteria.by_indices([0, 1]))
        )

        # Mixed
        result3 = (
            container.filter_samples(polars_expression=pl.col("QC_pass"))
            .filter_features(
                "proteins", FilterCriteria.by_expression(pl.col("mean_intensity") > 20)
            )
            .filter_samples(sample_indices=[0, 1])
        )

        assert_containers_equal(result1, result2, "All legacy vs all new")
        assert_containers_equal(result2, result3, "All new vs mixed")

    def test_copy_parameter_preserved(self):
        """Test that copy parameter works with new API."""
        container = create_test_container(n_samples=20)

        # Legacy API with copy=False
        legacy_no_copy = container.filter_samples(sample_indices=[0, 1, 2], copy=False)

        # New API with copy=False
        criteria = FilterCriteria.by_indices([0, 1, 2])
        new_no_copy = container.filter_samples(criteria, copy=False)

        # Both should share the same var DataFrame
        assert (
            legacy_no_copy.assays["proteins"].var
            is new_no_copy.assays["proteins"].var
            is container.assays["proteins"].var
        ), "copy=False should share var DataFrame"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
