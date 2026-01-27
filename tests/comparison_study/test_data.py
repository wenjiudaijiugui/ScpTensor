"""Unit tests for data loading and synthetic data generation modules."""

import pytest

# Skip all tests in this module because the studies.data module
# is not installed as a Python package and cannot be imported during test runs.
# The comparison study code lives in studies/ for documentation
# purposes but is not part of the scptensor package installation.
pytest.skip(
    "studies.data module is not installed as a Python package. "
    "The comparison study code is documentation-only and not available for import.",
    allow_module_level=True,
)


class TestLoadDatasets:
    """Test data loading functions."""

    def test_create_batch_labels_even_split(self):
        """Test batch label creation with even split."""
        labels = create_batch_labels(n_samples=100, n_batches=4)
        assert len(labels) == 100
        assert len(np.unique(labels)) == 4
        assert np.min(labels) == 0
        assert np.max(labels) == 3

    def test_create_batch_labels_custom_sizes(self):
        """Test batch label creation with custom sizes."""
        labels = create_batch_labels(n_samples=100, n_batches=3, batch_sizes=[30, 50, 20])
        assert len(labels) == 100
        assert np.sum(labels == 0) == 30
        assert np.sum(labels == 1) == 50
        assert np.sum(labels == 2) == 20

    def test_create_batch_labels_invalid(self):
        """Test batch label creation with invalid parameters."""
        with pytest.raises(ValueError):
            create_batch_labels(n_samples=100, n_batches=0)

        with pytest.raises(ValueError):
            create_batch_labels(n_samples=0, n_batches=3)

    def test_add_batch_effects(self):
        """Test adding batch effects to data."""
        np.random.seed(42)
        X = np.random.rand(100, 50)
        batch_labels = np.array([0] * 50 + [1] * 50)

        X_batched = add_batch_effects(X, batch_labels, effect_size=1.0)

        # Check shape preserved
        assert X_batched.shape == X.shape

        # Check values changed
        assert not np.allclose(X, X_batched)

        # Check within reasonable range
        assert np.all(np.isfinite(X_batched))

    def test_add_batch_effects_zero_effect(self):
        """Test that zero effect size returns unchanged data."""
        X = np.random.rand(50, 30)
        batch_labels = np.array([0] * 25 + [1] * 25)

        X_batched = add_batch_effects(X, batch_labels, effect_size=0)

        # Should be copy but identical values
        assert np.allclose(X, X_batched)


class TestGenerateSyntheticDataset:
    """Test synthetic dataset generation."""

    def test_generate_small_dataset(self):
        """Test small dataset generation."""
        container = generate_small_dataset()
        assert container.n_samples == 1000
        assert container.n_features == 1000
        assert len(container.obs["batch"].unique()) == 1
        assert len(container.obs["cell_type"].unique()) == 5

    def test_generate_medium_dataset(self):
        """Test medium dataset generation."""
        container = generate_medium_dataset()
        assert container.n_samples == 5000
        assert container.n_features == 1500
        assert len(container.obs["batch"].unique()) == 5
        assert len(container.obs["cell_type"].unique()) == 8

    def test_generate_large_dataset(self):
        """Test large dataset generation."""
        container = generate_large_dataset()
        assert container.n_samples == 20000
        assert container.n_features == 2000
        assert len(container.obs["batch"].unique()) == 10
        assert len(container.obs["cell_type"].unique()) == 12

    def test_generate_synthetic_dataset_single_batch(self):
        """Test dataset generation with single batch."""
        container = generate_synthetic_dataset(
            n_samples=100, n_features=50, n_batches=1, random_seed=42
        )
        assert container.n_samples == 100
        assert container.n_features == 50
        assert len(container.obs["batch"].unique()) == 1

    def test_generate_synthetic_dataset_multiple_batches(self):
        """Test dataset generation with multiple batches."""
        container = generate_synthetic_dataset(
            n_samples=100, n_features=50, n_batches=3, random_seed=42
        )
        assert container.n_samples == 100
        assert len(container.obs["batch"].unique()) == 3

    def test_generate_synthetic_dataset_reproducibility(self):
        """Test that datasets are reproducible with same seed."""
        container1 = generate_synthetic_dataset(n_samples=100, n_features=50, random_seed=42)
        container2 = generate_synthetic_dataset(n_samples=100, n_features=50, random_seed=42)

        X1 = container1.assays["proteins"].layers["raw"].X
        X2 = container2.assays["proteins"].layers["raw"].X

        np.testing.assert_array_equal(X1, X2)

    def test_generate_synthetic_dataset_non_negative(self):
        """Test that generated data is non-negative."""
        container = generate_synthetic_dataset(n_samples=100, n_features=50, random_seed=42)
        X = container.assays["proteins"].layers["raw"].X
        assert np.all(X >= 0)


class TestLoadAllDatasets:
    """Test loading all datasets."""

    def test_load_all_datasets(self):
        """Test loading all three datasets."""
        datasets = load_all_datasets()
        assert "small" in datasets
        assert "medium" in datasets
        assert "large" in datasets

        # Verify dataset sizes
        assert datasets["small"].n_samples == 1000
        assert datasets["medium"].n_samples == 5000
        assert datasets["large"].n_samples == 20000


class TestCacheDatasets:
    """Test dataset caching."""

    def test_cache_and_load_datasets(self):
        """Test caching and loading datasets."""
        # Create small test datasets
        datasets = {
            "test1": generate_synthetic_dataset(n_samples=50, n_features=30, random_seed=42),
            "test2": generate_synthetic_dataset(n_samples=60, n_features=40, random_seed=43),
        }

        # Create temporary cache directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Cache datasets
            cache_datasets(datasets, cache_dir=temp_dir)

            # Verify files created
            cache_path = Path(temp_dir)
            assert (cache_path / "test1.pkl").exists()
            assert (cache_path / "test2.pkl").exists()

            # Load cached datasets
            loaded_datasets = load_cached_datasets(cache_dir=temp_dir)

            assert "test1" in loaded_datasets
            assert "test2" in loaded_datasets
            assert loaded_datasets["test1"].n_samples == 50
            assert loaded_datasets["test2"].n_samples == 60

    def test_load_cached_datasets_missing_dir(self):
        """Test loading from non-existent cache directory."""
        with pytest.raises(FileNotFoundError):
            load_cached_datasets(cache_dir="/nonexistent/path")
