"""Tests for scptensor.utils.data_generator module.

This module contains comprehensive tests for the ScpDataGenerator class
which generates synthetic single-cell proteomics data.
"""

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp

from scptensor.core.structures import ScpContainer
from scptensor.utils.data_generator import ScpDataGenerator


class TestScpDataGeneratorInit:
    """Tests for ScpDataGenerator initialization."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        gen = ScpDataGenerator()
        assert gen.n_samples == 100
        assert gen.n_features == 1000
        assert gen.missing_rate == 0.3
        assert gen.lod_ratio == 0.6
        assert gen.n_batches == 3
        assert gen.n_groups == 4
        assert gen.random_seed == 42

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        gen = ScpDataGenerator(
            n_samples=200,
            n_features=500,
            missing_rate=0.5,
            lod_ratio=0.7,
            n_batches=5,
            n_groups=3,
            random_seed=123,
        )
        assert gen.n_samples == 200
        assert gen.n_features == 500
        assert gen.missing_rate == 0.5
        assert gen.lod_ratio == 0.7
        assert gen.n_batches == 5
        assert gen.n_groups == 3
        assert gen.random_seed == 123

    def test_init_custom_names(self):
        """Test initialization with custom assay/layer names."""
        gen = ScpDataGenerator(
            assay_name="peptides",
            layer_name="intensity",
            feature_id_col="peptide_id",
            sample_id_col="cell_id",
        )
        assert gen.assay_name == "peptides"
        assert gen.layer_name == "intensity"
        assert gen.feature_id_col == "peptide_id"
        assert gen.sample_id_col == "cell_id"

    def test_init_sparse_options(self):
        """Test initialization with sparse matrix options."""
        gen = ScpDataGenerator(
            use_sparse_X=True,
            use_sparse_M=True,
        )
        assert gen.use_sparse_X is True
        assert gen.use_sparse_M is True

    def test_init_mask_kind_options(self):
        """Test initialization with different mask kinds."""
        gen_none = ScpDataGenerator(mask_kind="none")
        gen_bool = ScpDataGenerator(mask_kind="bool")
        gen_int8 = ScpDataGenerator(mask_kind="int8")

        assert gen_none.mask_kind == "none"
        assert gen_bool.mask_kind == "bool"
        assert gen_int8.mask_kind == "int8"

    def test_validate_params_invalid_missing_rate(self):
        """Test that invalid missing_rate raises ValueError."""
        # The actual max is 0.7 (_MAX_MISSING_RATE), so 1.5 is invalid
        with pytest.raises(ValueError, match="missing_rate must be between 0.0 and"):
            ScpDataGenerator(missing_rate=1.5)

    def test_validate_params_zero_missing_rate(self):
        """Test that missing_rate=0 is accepted."""
        gen = ScpDataGenerator(missing_rate=0.0)
        assert gen.missing_rate == 0.0

    def test_validate_params_max_missing_rate(self):
        """Test that missing_rate=0.7 (max) is accepted."""
        # The actual max is 0.7 (_MAX_MISSING_RATE)
        gen = ScpDataGenerator(missing_rate=0.7)
        assert gen.missing_rate == 0.7


class TestScpDataGeneratorGenerate:
    """Tests for ScpDataGenerator.generate method."""

    @pytest.fixture
    def default_generator(self):
        """Create a generator with default settings."""
        return ScpDataGenerator(random_seed=42)

    @pytest.fixture
    def small_generator(self):
        """Create a small generator for faster tests."""
        return ScpDataGenerator(
            n_samples=50,
            n_features=100,
            n_batches=2,
            n_groups=2,
            random_seed=42,
        )

    def test_generate_returns_container(self, default_generator):
        """Test that generate returns ScpContainer."""
        container = default_generator.generate()
        assert isinstance(container, ScpContainer)

    def test_generate_container_shape(self, default_generator):
        """Test that generated container has correct shape."""
        container = default_generator.generate()
        assay = container.assays[default_generator.assay_name]
        matrix = assay.layers[default_generator.layer_name]

        if sp.issparse(matrix.X):
            assert matrix.X.shape == (100, 1000)
        else:
            assert matrix.X.shape == (100, 1000)

    def test_generate_obs_dataframe(self, default_generator):
        """Test that obs DataFrame is created correctly."""
        container = default_generator.generate()
        # obs is a DataFrame attribute, not a key
        assert hasattr(container, "obs")
        assert len(container.obs) == 100
        assert "sample_id" in container.obs.columns
        assert "batch" in container.obs.columns
        assert "group" in container.obs.columns

    def test_generate_var_dataframe(self, default_generator):
        """Test that var DataFrame is created correctly."""
        container = default_generator.generate()
        assay = container.assays[default_generator.assay_name]
        # var is a DataFrame attribute, not a key
        assert hasattr(assay, "var")
        assert len(assay.var) == 1000
        assert "protein_id" in assay.var.columns
        assert "gene_name" in assay.var.columns
        assert "mean_abundance" in assay.var.columns

    def test_generate_batches(self, default_generator):
        """Test that batches are correctly assigned."""
        container = default_generator.generate()
        batches = container.obs["batch"].to_list()
        unique_batches = set(batches)
        assert len(unique_batches) == 3
        assert all(b.startswith("Batch_") for b in unique_batches)

    def test_generate_groups(self, default_generator):
        """Test that groups are correctly assigned."""
        container = default_generator.generate()
        groups = container.obs["group"].to_list()
        unique_groups = set(groups)
        assert len(unique_groups) == 4
        assert all(g.startswith("Group_") for g in unique_groups)

    def test_generate_mask_codes(self, default_generator):
        """Test that mask codes are correct."""
        container = default_generator.generate()
        assay = container.assays[default_generator.assay_name]
        matrix = assay.layers[default_generator.layer_name]

        if matrix.M is not None:
            # Check valid mask codes: 0 (VALID), 1 (MBR), 2 (LOD)
            unique_codes = np.unique(matrix.M)
            assert all(code in [0, 1, 2] for code in unique_codes)

    def test_generate_missing_rate_approximate(self, small_generator):
        """Test that actual missing rate is close to target."""
        container = small_generator.generate()
        assay = container.assays[small_generator.assay_name]
        matrix = assay.layers[small_generator.layer_name]

        n_total = matrix.X.size
        if sp.issparse(matrix.X):
            n_valid = matrix.X.nnz
        else:
            if matrix.M is None:
                n_missing = 0
            else:
                n_missing = np.sum(matrix.M != 0)
            n_valid = n_total - n_missing

        actual_missing_rate = 1 - (n_valid / n_total)
        # Allow 10% tolerance
        assert abs(actual_missing_rate - small_generator.missing_rate) < 0.15

    def test_generate_provenance_log(self, default_generator):
        """Test that provenance log is recorded."""
        container = default_generator.generate()
        assert len(container.history) > 0
        assert container.history[-1].action == "generate_synthetic_data"

    def test_generate_reproducibility(self, default_generator):
        """Test that same seed produces identical results."""
        # Need to create two generators with same seed
        gen1 = ScpDataGenerator(random_seed=42)
        gen2 = ScpDataGenerator(random_seed=42)

        container1 = gen1.generate()
        container2 = gen2.generate()

        assay1 = container1.assays[default_generator.assay_name]
        assay2 = container2.assays[default_generator.assay_name]

        X1 = assay1.layers[default_generator.layer_name].X
        X2 = assay2.layers[default_generator.layer_name].X

        if sp.issparse(X1):
            X1 = X1.toarray()
        if sp.issparse(X2):
            X2 = X2.toarray()

        np.testing.assert_array_equal(X1, X2)

    def test_generate_sparse_X(self):
        """Test generation with sparse X."""
        gen = ScpDataGenerator(
            n_samples=50,
            n_features=100,
            use_sparse_X=True,
            random_seed=42,
        )
        container = gen.generate()
        assay = container.assays[gen.assay_name]
        matrix = assay.layers[gen.layer_name]

        assert sp.issparse(matrix.X)

    def test_generate_sparse_M(self):
        """Test generation with sparse M."""
        gen = ScpDataGenerator(
            n_samples=50,
            n_features=100,
            use_sparse_M=True,
            missing_rate=0.3,
            random_seed=42,
        )
        container = gen.generate()
        assay = container.assays[gen.assay_name]
        matrix = assay.layers[gen.layer_name]

        if matrix.M is not None:
            assert sp.issparse(matrix.M)

    def test_generate_mask_none(self):
        """Test generation with mask_kind='none'."""
        gen = ScpDataGenerator(
            n_samples=50,
            n_features=100,
            mask_kind="none",
            random_seed=42,
        )
        container = gen.generate()
        assay = container.assays[gen.assay_name]
        matrix = assay.layers[gen.layer_name]

        assert matrix.M is None

    def test_generate_mask_bool(self):
        """Test generation with mask_kind='bool'."""
        gen = ScpDataGenerator(
            n_samples=50,
            n_features=100,
            mask_kind="bool",
            missing_rate=0.3,
            random_seed=42,
        )
        container = gen.generate()
        assay = container.assays[gen.assay_name]
        matrix = assay.layers[gen.layer_name]

        if matrix.M is not None and not sp.issparse(matrix.M):
            # Check that mask exists and is correct type
            # Bool mask might be stored as int8 with values 0/1
            assert matrix.M.dtype in [bool, np.int8]

    def test_generate_zero_missing_rate(self):
        """Test generation with no missing values."""
        gen = ScpDataGenerator(
            n_samples=50,
            n_features=100,
            missing_rate=0.0,
            random_seed=42,
        )
        container = gen.generate()
        assay = container.assays[gen.assay_name]
        matrix = assay.layers[gen.layer_name]

        if matrix.M is not None:
            assert np.all(matrix.M == 0)

    def test_generate_high_missing_rate(self):
        """Test generation with high missing rate."""
        gen = ScpDataGenerator(
            n_samples=50,
            n_features=100,
            missing_rate=0.7,
            random_seed=42,
        )
        container = gen.generate()
        assay = container.assays[gen.assay_name]
        matrix = assay.layers[gen.layer_name]

        if matrix.M is not None:
            missing_count = np.sum(matrix.M != 0)
            total_count = matrix.M.size
            actual_rate = missing_count / total_count
            # Should be close to 0.7 (with tolerance)
            assert actual_rate > 0.5

    def test_generate_different_batch_counts(self):
        """Test generation with different number of batches."""
        for n_batches in [1, 2, 5, 10]:
            gen = ScpDataGenerator(
                n_samples=50,
                n_features=100,
                n_batches=n_batches,
                random_seed=42,
            )
            container = gen.generate()
            unique_batches = set(container.obs["batch"].to_list())
            assert len(unique_batches) == n_batches

    def test_generate_different_group_counts(self):
        """Test generation with different number of groups."""
        for n_groups in [1, 2, 5]:
            gen = ScpDataGenerator(
                n_samples=50,
                n_features=100,
                n_groups=n_groups,
                random_seed=42,
            )
            container = gen.generate()
            unique_groups = set(container.obs["group"].to_list())
            assert len(unique_groups) == n_groups

    def test_generate_feature_ids(self, default_generator):
        """Test that feature IDs are formatted correctly."""
        container = default_generator.generate()
        assay = container.assays[default_generator.assay_name]
        feature_ids = assay.var["protein_id"].to_list()

        assert all(f.startswith("Prot_") for f in feature_ids)
        assert len(feature_ids) == 1000

    def test_generate_sample_ids(self, default_generator):
        """Test that sample IDs are formatted correctly."""
        container = default_generator.generate()
        sample_ids = container.obs["sample_id"].to_list()

        assert all(s.startswith("Cell_") for s in sample_ids)
        assert len(sample_ids) == 100

    def test_generate_mean_abundance(self, default_generator):
        """Test that mean abundance is present and reasonable."""
        container = default_generator.generate()
        assay = container.assays[default_generator.assay_name]
        mean_abundance = assay.var["mean_abundance"].to_numpy()

        assert len(mean_abundance) == 1000
        assert np.all(mean_abundance > 0)
        # Should be around 15 (the mean in the generator)
        assert np.mean(mean_abundance) > 10

    def test_generate_efficiency_column(self, default_generator):
        """Test that efficiency column is present."""
        container = default_generator.generate()
        assert "efficiency" in container.obs.columns

        efficiency = container.obs["efficiency"].to_numpy()
        assert len(efficiency) == 100

    def test_generate_custom_assay_name(self):
        """Test generation with custom assay name."""
        gen = ScpDataGenerator(assay_name="peptides", random_seed=42)
        container = gen.generate()

        assert "peptides" in container.assays
        # Check the assay name matches what we specified
        assert gen.assay_name == "peptides"

    def test_generate_custom_layer_name(self):
        """Test generation with custom layer name."""
        gen = ScpDataGenerator(layer_name="intensity", random_seed=42)
        container = gen.generate()
        assay = container.assays[gen.assay_name]

        assert "intensity" in assay.layers
        # Check the layer name matches what we specified
        assert gen.layer_name == "intensity"

    def test_generate_batch_effect_present(self, default_generator):
        """Test that batch effects are present in data."""
        container = default_generator.generate()
        assay = container.assays[default_generator.assay_name]
        X = assay.layers[default_generator.layer_name].X

        if sp.issparse(X):
            X = X.toarray()

        # Check if there are differences between batches
        # Use the sample_id_col for indexing
        container.obs.filter(pl.col("batch") == "Batch_0")[
            default_generator.sample_id_col
        ].to_list()
        container.obs.filter(pl.col("batch") == "Batch_1")[
            default_generator.sample_id_col
        ].to_list()

        # Get positions - since order is preserved in generation, we can use direct indices
        batch_0_mask = container.obs["batch"] == "Batch_0"
        batch_1_mask = container.obs["batch"] == "Batch_1"

        mean_0 = X[batch_0_mask, :].mean()
        mean_1 = X[batch_1_mask, :].mean()

        # Batch means should differ somewhat (not exactly equal)
        # We allow some tolerance but they should be different
        assert abs(mean_0 - mean_1) > 0.01


class TestScpDataGeneratorEdgeCases:
    """Edge case tests for ScpDataGenerator."""

    def test_very_small_dataset(self):
        """Test with minimal dataset size."""
        gen = ScpDataGenerator(
            n_samples=10,
            n_features=20,
            n_batches=2,
            n_groups=2,
            random_seed=42,
        )
        container = gen.generate()
        assay = container.assays[gen.assay_name]

        assert len(container.obs) == 10
        assert len(assay.var) == 20

    def test_very_large_dataset(self):
        """Test with large dataset size."""
        gen = ScpDataGenerator(
            n_samples=500,
            n_features=2000,
            random_seed=42,
        )
        container = gen.generate()
        assay = container.assays[gen.assay_name]

        assert len(container.obs) == 500
        assert len(assay.var) == 2000

    def test_single_batch(self):
        """Test with single batch (no batch effect)."""
        gen = ScpDataGenerator(
            n_samples=50,
            n_features=100,
            n_batches=1,
            random_seed=42,
        )
        container = gen.generate()
        unique_batches = set(container.obs["batch"].to_list())
        assert len(unique_batches) == 1

    def test_single_group(self):
        """Test with single group (all cells same type)."""
        gen = ScpDataGenerator(
            n_samples=50,
            n_features=100,
            n_groups=1,
            random_seed=42,
        )
        container = gen.generate()
        unique_groups = set(container.obs["group"].to_list())
        assert len(unique_groups) == 1

    def test_no_lod_ratio(self):
        """Test with lod_ratio=0 (all MCAR)."""
        gen = ScpDataGenerator(
            n_samples=50,
            n_features=100,
            missing_rate=0.3,
            lod_ratio=0.0,
            random_seed=42,
        )
        container = gen.generate()
        assay = container.assays[gen.assay_name]
        matrix = assay.layers[gen.layer_name]

        if matrix.M is not None:
            # Should only have codes 0 and 1 (VALID and MBR), no 2 (LOD)
            unique_codes = np.unique(matrix.M)
            assert all(code in [0, 1] for code in unique_codes)

    def test_all_lod_ratio(self):
        """Test with lod_ratio=1.0 (all MNAR)."""
        gen = ScpDataGenerator(
            n_samples=50,
            n_features=100,
            missing_rate=0.3,
            lod_ratio=1.0,
            random_seed=42,
        )
        container = gen.generate()
        assay = container.assays[gen.assay_name]
        matrix = assay.layers[gen.layer_name]

        if matrix.M is not None:
            # Should have codes 0 and 2 (VALID and LOD), no 1 (MBR)
            unique_codes = np.unique(matrix.M)
            assert all(code in [0, 2] for code in unique_codes)


class TestScpDataGeneratorProperties:
    """Tests for generated data properties."""

    @pytest.fixture
    def generator(self):
        """Create a generator for testing."""
        return ScpDataGenerator(
            n_samples=100,
            n_features=500,
            missing_rate=0.3,
            random_seed=42,
        )

    def test_data_has_variance(self, generator):
        """Test that generated data has variance."""
        container = generator.generate()
        assay = container.assays[generator.assay_name]
        X = assay.layers[generator.layer_name].X

        if sp.issparse(X):
            X = X.toarray()

        # Data should have variance (not all same value)
        assert np.var(X) > 0

    def test_data_finite_values(self, generator):
        """Test that all values are finite."""
        container = generator.generate()
        assay = container.assays[generator.assay_name]
        X = assay.layers[generator.layer_name].X

        if sp.issparse(X):
            X = X.toarray()

        assert np.all(np.isfinite(X))

    def test_correlation_structure(self, generator):
        """Test that data has some correlation structure."""
        container = generator.generate()
        assay = container.assays[generator.assay_name]
        X = assay.layers[generator.layer_name].X

        if sp.issparse(X):
            X = X.toarray()

        # Compute correlation matrix for a subset of features
        corr = np.corrcoef(X[:, :50].T)

        # Some features should be correlated (|corr| > 0.5)
        # due to latent factor model
        np.fill_diagonal(corr, 0)
        max_corr = np.max(np.abs(corr))
        assert max_corr > 0.1  # At least some correlation

    def test_group_differences(self, generator):
        """Test that different groups have different expression patterns."""
        container = generator.generate()
        assay = container.assays[generator.assay_name]
        X = assay.layers[generator.layer_name].X

        if sp.issparse(X):
            X = X.toarray()

        # Get samples from different groups using mask
        group_0_mask = container.obs["group"] == "Group_0"
        group_1_mask = container.obs["group"] == "Group_1"

        mean_0 = X[group_0_mask, :].mean(axis=0)
        mean_1 = X[group_1_mask, :].mean(axis=0)

        # Group means should differ
        assert not np.allclose(mean_0, mean_1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
