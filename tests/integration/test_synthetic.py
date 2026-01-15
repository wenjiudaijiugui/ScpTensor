"""
Tests for synthetic data generation and validation.

This module tests the synthetic data generator functions and validates
that generated data has expected properties.
"""

import numpy as np
import polars as pl

from scptensor.core.structures import Assay, ScpContainer, ScpMatrix


class TestSyntheticDataGenerator:
    """Test synthetic data generation functionality."""

    def test_generate_synthetic_data_basic(self):
        """Test basic synthetic data generation."""
        np.random.seed(42)

        n_samples = 50
        n_features = 100

        # Create metadata
        groups = np.array(["GroupA"] * 25 + ["GroupB"] * 25)
        batches = np.random.choice(["Batch1", "Batch2"], size=n_samples)

        obs = pl.DataFrame(
            {
                "sample_id": [f"S{i + 1:03d}" for i in range(n_samples)],
                "group": groups,
                "batch": batches,
                "_index": [f"S{i + 1:03d}" for i in range(n_samples)],
            }
        )

        # Create expression data
        X_true = np.random.lognormal(mean=2, sigma=0.5, size=(n_samples, n_features))
        X_observed = X_true.copy()
        M = np.zeros((n_samples, n_features), dtype=int)

        # Add some missing values
        threshold = np.percentile(X_true, 20)
        lod_mask = X_true < threshold
        X_observed[lod_mask] = 0
        M[lod_mask] = 2

        # Create container
        var = pl.DataFrame(
            {
                "protein_id": [f"P{i + 1:04d}" for i in range(n_features)],
                "_index": [f"P{i + 1:04d}" for i in range(n_features)],
            }
        )

        matrix = ScpMatrix(X=X_observed, M=M)
        assay = Assay(var=var, layers={"raw": matrix}, feature_id_col="protein_id")

        container = ScpContainer(assays={"protein": assay}, obs=obs, sample_id_col="sample_id")

        # Verify container structure
        assert container.n_samples == n_samples
        assert "protein" in container.assays
        assert container.assays["protein"].n_features == n_features
        assert "raw" in container.assays["protein"].layers

        # Verify missing values were created
        raw_layer = container.assays["protein"].layers["raw"]
        assert np.sum(raw_layer.M != 0) > 0

    def test_synthetic_data_with_batch_effect(self):
        """Test synthetic data with known batch effect."""
        np.random.seed(42)

        n_samples = 40
        n_features = 50

        groups = np.array(["GroupA"] * 20 + ["GroupB"] * 20)
        batches = np.array(["Batch1"] * 20 + ["Batch2"] * 20)

        obs = pl.DataFrame(
            {
                "sample_id": [f"S{i + 1:03d}" for i in range(n_samples)],
                "group": groups,
                "batch": batches,
                "_index": [f"S{i + 1:03d}" for i in range(n_samples)],
            }
        )

        # Create data with batch effect
        X = np.random.lognormal(mean=2, sigma=0.5, size=(n_samples, n_features))
        X[batches == "Batch2", :] *= 1.5  # Add batch effect

        M = np.zeros((n_samples, n_features), dtype=int)

        var = pl.DataFrame(
            {
                "protein_id": [f"P{i + 1:04d}" for i in range(n_features)],
                "_index": [f"P{i + 1:04d}" for i in range(n_features)],
            }
        )

        matrix = ScpMatrix(X=X, M=M)
        assay = Assay(var=var, layers={"raw": matrix}, feature_id_col="protein_id")

        ScpContainer(assays={"protein": assay}, obs=obs, sample_id_col="sample_id")

        # Verify batch effect exists in data
        batch1_mean = np.mean(X[batches == "Batch1", :])
        batch2_mean = np.mean(X[batches == "Batch2", :])

        # Batch2 should have higher mean due to effect
        assert batch2_mean > batch1_mean

    def test_synthetic_data_with_group_effect(self):
        """Test synthetic data with known group effect."""
        np.random.seed(42)

        n_samples = 40
        n_features = 50

        groups = np.array(["GroupA"] * 20 + ["GroupB"] * 20)
        batches = np.random.choice(["Batch1", "Batch2"], size=n_samples)

        obs = pl.DataFrame(
            {
                "sample_id": [f"S{i + 1:03d}" for i in range(n_samples)],
                "group": groups,
                "batch": batches,
                "_index": [f"S{i + 1:03d}" for i in range(n_samples)],
            }
        )

        # Create data with group effect in first 20 features
        X = np.random.lognormal(mean=2, sigma=0.5, size=(n_samples, n_features))
        X[groups == "GroupB", :20] *= 2.0  # Add group effect

        M = np.zeros((n_samples, n_features), dtype=int)

        var = pl.DataFrame(
            {
                "protein_id": [f"P{i + 1:04d}" for i in range(n_features)],
                "_index": [f"P{i + 1:04d}" for i in range(n_features)],
            }
        )

        matrix = ScpMatrix(X=X, M=M)
        assay = Assay(var=var, layers={"raw": matrix}, feature_id_col="protein_id")

        ScpContainer(assays={"protein": assay}, obs=obs, sample_id_col="sample_id")

        # Verify group effect exists in affected features
        groupa_mean = np.mean(X[groups == "GroupA", :20])
        groupb_mean = np.mean(X[groups == "GroupB", :20])

        # GroupB should have higher mean in affected features
        assert groupb_mean > groupa_mean

    def test_synthetic_data_missing_value_patterns(self):
        """Test synthetic data with different missing value patterns."""
        np.random.seed(42)

        n_samples = 30
        n_features = 40

        obs = pl.DataFrame(
            {
                "sample_id": [f"S{i + 1:03d}" for i in range(n_samples)],
                "_index": [f"S{i + 1:03d}" for i in range(n_samples)],
            }
        )

        X = np.random.lognormal(mean=2, sigma=0.5, size=(n_samples, n_features))
        X_observed = X.copy()
        M = np.zeros((n_samples, n_features), dtype=int)

        # Pattern 1: LOD (mask code 2) - low abundance values
        threshold = np.percentile(X, 20)
        lod_mask = threshold > X
        X_observed[lod_mask] = 0
        M[lod_mask] = 2

        # Pattern 2: MBR (mask code 1) - random missing
        valid_mask = M == 0
        valid_indices = np.argwhere(valid_mask)
        n_random_missing = 50
        random_indices_idx = np.random.choice(
            len(valid_indices), size=n_random_missing, replace=False
        )
        random_indices = valid_indices[random_indices_idx]
        X_observed[random_indices[:, 0], random_indices[:, 1]] = 0
        M[random_indices[:, 0], random_indices[:, 1]] = 1

        var = pl.DataFrame(
            {
                "protein_id": [f"P{i + 1:04d}" for i in range(n_features)],
                "_index": [f"P{i + 1:04d}" for i in range(n_features)],
            }
        )

        matrix = ScpMatrix(X=X_observed, M=M)
        assay = Assay(var=var, layers={"raw": matrix}, feature_id_col="protein_id")

        ScpContainer(assays={"protein": assay}, obs=obs, sample_id_col="sample_id")

        # Verify missing value patterns
        assert np.sum(M == 1) > 0  # MBR values
        assert np.sum(M == 2) > 0  # LOD values
        assert np.sum(M == 0) > 0  # Valid values

    def test_synthetic_data_reproducibility(self):
        """Test that synthetic data generation is reproducible with same seed."""
        n_samples = 20
        n_features = 30

        # Generate first dataset
        np.random.seed(123)
        X1 = np.random.lognormal(mean=2, sigma=0.5, size=(n_samples, n_features))

        # Generate second dataset with same seed
        np.random.seed(123)
        X2 = np.random.lognormal(mean=2, sigma=0.5, size=(n_samples, n_features))

        # Should be identical
        assert np.array_equal(X1, X2)

    def test_synthetic_data_missing_rate(self):
        """Test control over missing value rate."""
        np.random.seed(42)

        n_samples = 50
        n_features = 100
        target_missing_rate = 0.5

        X = np.random.lognormal(mean=2, sigma=0.5, size=(n_samples, n_features))
        X_observed = X.copy()
        M = np.zeros((n_samples, n_features), dtype=int)

        # Add missing values to achieve target rate
        n_missing = int(n_samples * n_features * target_missing_rate)
        all_indices = np.argwhere(M == 0)
        missing_indices_idx = np.random.choice(len(all_indices), size=n_missing, replace=False)
        missing_indices = all_indices[missing_indices_idx]

        X_observed[missing_indices[:, 0], missing_indices[:, 1]] = 0
        M[missing_indices[:, 0], missing_indices[:, 1]] = 1

        # Calculate actual missing rate
        actual_missing_rate = np.sum(M != 0) / (n_samples * n_features)

        # Should be close to target (within 5% tolerance)
        assert abs(actual_missing_rate - target_missing_rate) < 0.05


class TestSyntheticDataValidation:
    """Test validation of synthetic data properties."""

    def test_validate_container_structure(self):
        """Test that synthetic container has valid structure."""
        np.random.seed(42)

        n_samples = 30
        n_features = 50

        obs = pl.DataFrame(
            {
                "sample_id": [f"S{i + 1:03d}" for i in range(n_samples)],
                "group": np.random.choice(["A", "B"], size=n_samples),
                "batch": np.random.choice(["B1", "B2"], size=n_samples),
                "_index": [f"S{i + 1:03d}" for i in range(n_samples)],
            }
        )

        X = np.random.rand(n_samples, n_features)
        M = np.zeros((n_samples, n_features), dtype=int)

        var = pl.DataFrame(
            {
                "protein_id": [f"P{i + 1:04d}" for i in range(n_features)],
                "_index": [f"P{i + 1:04d}" for i in range(n_features)],
            }
        )

        matrix = ScpMatrix(X=X, M=M)
        assay = Assay(var=var, layers={"raw": matrix}, feature_id_col="protein_id")

        container = ScpContainer(assays={"protein": assay}, obs=obs, sample_id_col="sample_id")

        # Validate structure
        assert container.n_samples == n_samples
        assert container.sample_id_col == "sample_id"
        assert len(container.sample_ids) == n_samples
        assert "protein" in container.assays
        assert container.assays["protein"].feature_id_col == "protein_id"

    def test_validate_metadata_columns(self):
        """Test that synthetic data has expected metadata columns."""
        np.random.seed(42)

        obs = pl.DataFrame(
            {
                "sample_id": ["S1", "S2", "S3"],
                "group": ["A", "B", "A"],
                "batch": ["B1", "B2", "B1"],
                "_index": ["S1", "S2", "S3"],
            }
        )

        X = np.random.rand(3, 10)
        M = np.zeros((3, 10), dtype=int)

        var = pl.DataFrame(
            {
                "protein_id": [f"P{i + 1:04d}" for i in range(10)],
                "_index": [f"P{i + 1:04d}" for i in range(10)],
            }
        )

        matrix = ScpMatrix(X=X, M=M)
        assay = Assay(var=var, layers={"raw": matrix}, feature_id_col="protein_id")

        container = ScpContainer(assays={"protein": assay}, obs=obs, sample_id_col="sample_id")

        # Check metadata columns exist
        assert "sample_id" in container.obs.columns
        assert "group" in container.obs.columns
        assert "batch" in container.obs.columns
        assert "_index" in container.obs.columns

        # Check values
        assert set(container.obs["group"].unique().to_list()) == {"A", "B"}
        assert set(container.obs["batch"].unique().to_list()) == {"B1", "B2"}

    def test_validate_mask_codes(self):
        """Test that synthetic data uses valid mask codes."""
        np.random.seed(42)

        n_samples = 20
        n_features = 30

        obs = pl.DataFrame(
            {
                "sample_id": [f"S{i + 1:03d}" for i in range(n_samples)],
                "_index": [f"S{i + 1:03d}" for i in range(n_samples)],
            }
        )

        X = np.random.rand(n_samples, n_features)
        M = np.zeros((n_samples, n_features), dtype=int)

        # Add various mask codes
        M[0:5, 0:5] = 1  # MBR
        M[5:10, 5:10] = 2  # LOD
        M[10:15, 10:15] = 3  # FILTERED

        var = pl.DataFrame(
            {
                "protein_id": [f"P{i + 1:04d}" for i in range(n_features)],
                "_index": [f"P{i + 1:04d}" for i in range(n_features)],
            }
        )

        matrix = ScpMatrix(X=X, M=M)
        assay = Assay(var=var, layers={"raw": matrix}, feature_id_col="protein_id")

        ScpContainer(assays={"protein": assay}, obs=obs, sample_id_col="sample_id")

        # Verify valid mask codes (0, 1, 2, 3, 5 are valid)
        unique_codes = np.unique(M)
        for code in unique_codes:
            assert code in [0, 1, 2, 3, 5]

    def test_validate_data_types(self):
        """Test that synthetic data has correct data types."""
        np.random.seed(42)

        obs = pl.DataFrame({"sample_id": ["S1", "S2"], "group": ["A", "B"], "_index": ["S1", "S2"]})

        X = np.random.rand(2, 10).astype(np.float64)
        M = np.zeros((2, 10), dtype=np.int8)

        var = pl.DataFrame(
            {
                "protein_id": [f"P{i + 1:04d}" for i in range(10)],
                "_index": [f"P{i + 1:04d}" for i in range(10)],
            }
        )

        matrix = ScpMatrix(X=X, M=M)
        assay = Assay(var=var, layers={"raw": matrix}, feature_id_col="protein_id")

        container = ScpContainer(assays={"protein": assay}, obs=obs, sample_id_col="sample_id")

        # Check data types
        assert container.assays["protein"].layers["raw"].X.dtype == np.float64
        assert container.assays["protein"].layers["raw"].M.dtype == np.int8
        assert isinstance(container.obs, pl.DataFrame)
        assert isinstance(container.assays["protein"].var, pl.DataFrame)
