"""Tests for lazy validation functionality in Assay and ScpContainer.

This module tests the optional lazy validation feature that allows faster
loading of large datasets by deferring integrity checks until explicitly requested.
"""

import numpy as np
import polars as pl
import pytest

from scptensor.core import Assay, ScpContainer, ScpMatrix


class TestAssayLazyValidation:
    """Test lazy validation for Assay class."""

    def test_assay_lazy_validation(self, sample_var: pl.DataFrame) -> None:
        """Test Assay creation with validate_on_init=False."""
        # Create assay but skip validation
        X = np.random.randn(5, 5)
        assay = Assay(
            var=sample_var,
            layers={"raw": ScpMatrix(X=X)},
            validate_on_init=False,  # Skip validation
        )

        # Assay should be created without validation
        assert assay.n_features == 5
        assert "raw" in assay.layers

    def test_assay_explicit_validate(self, sample_var: pl.DataFrame) -> None:
        """Test manual validation of Assay after creation."""
        # Create assay without validation
        X = np.random.randn(5, 5)
        assay = Assay(
            var=sample_var,
            layers={"raw": ScpMatrix(X=X)},
            validate_on_init=False,
        )

        # Manually validate
        assay.validate()

        # Should pass without errors
        assert assay.n_features == 5

    def test_assay_invalid_data_lazy(self, sample_var: pl.DataFrame) -> None:
        """Test that invalid data is caught when validate() is called."""
        # Create assay with wrong feature dimension (10 instead of 5)
        X_wrong = np.random.randn(5, 10)  # Wrong: 10 features

        assay = Assay(
            var=sample_var,
            layers={"raw": ScpMatrix(X=X_wrong)},
            validate_on_init=False,  # Skip validation during init
        )

        # Should not raise during init
        assert assay.n_features == 5

        # But should raise when manually validated
        with pytest.raises(ValueError, match="Layer 'raw': Features 10 != Assay 5"):
            assay.validate()

    def test_assay_default_validates(self, sample_var: pl.DataFrame) -> None:
        """Test that default behavior validates on init (backward compatibility)."""
        # Create assay with wrong dimensions
        X_wrong = np.random.randn(5, 10)  # Wrong: 10 features

        # Should raise immediately with default validate_on_init=True
        with pytest.raises(ValueError, match="Layer 'raw': Features 10 != Assay 5"):
            Assay(var=sample_var, layers={"raw": ScpMatrix(X=X_wrong)})

    def test_assay_explicit_true_validates(self, sample_var: pl.DataFrame) -> None:
        """Test that validate_on_init=True validates immediately."""
        X_wrong = np.random.randn(5, 10)  # Wrong: 10 features

        with pytest.raises(ValueError, match="Layer 'raw': Features 10 != Assay 5"):
            Assay(
                var=sample_var,
                layers={"raw": ScpMatrix(X=X_wrong)},
                validate_on_init=True,
            )

    def test_assay_multiple_layers_lazy(
        self, sample_var: pl.DataFrame, sample_dense_X: np.ndarray
    ) -> None:
        """Test lazy validation with multiple layers."""
        X_log = np.log1p(sample_dense_X)

        assay = Assay(
            var=sample_var,
            layers={
                "raw": ScpMatrix(X=sample_dense_X),
                "log": ScpMatrix(X=X_log),
            },
            validate_on_init=False,
        )

        # Should not validate during init
        assert len(assay.layers) == 2

        # Manual validation should succeed
        assay.validate()
        assert len(assay.layers) == 2


class TestScpContainerLazyValidation:
    """Test lazy validation for ScpContainer class."""

    def test_container_lazy_validation(self, sample_obs: pl.DataFrame, sample_assay: Assay) -> None:
        """Test ScpContainer creation with validate_on_init=False."""
        container = ScpContainer(
            obs=sample_obs,
            assays={"proteins": sample_assay},
            validate_on_init=False,  # Skip validation
        )

        # Container should be created without validation
        assert container.n_samples == 5
        assert "proteins" in container.assays

    def test_container_explicit_validate(
        self, sample_obs: pl.DataFrame, sample_assay: Assay
    ) -> None:
        """Test manual validation of ScpContainer after creation."""
        container = ScpContainer(
            obs=sample_obs,
            assays={"proteins": sample_assay},
            validate_on_init=False,
        )

        # Manually validate
        container.validate()

        # Should pass without errors
        assert container.n_samples == 5

    def test_container_invalid_data_lazy(
        self, sample_obs: pl.DataFrame, sample_var: pl.DataFrame
    ) -> None:
        """Test that invalid data is caught when validate() is called."""
        # Create assay with wrong sample dimension (10 instead of 5)
        X_wrong = np.random.randn(10, 5)  # Wrong: 10 samples
        assay = Assay(
            var=sample_var,
            layers={"raw": ScpMatrix(X=X_wrong)},
            validate_on_init=False,
        )

        container = ScpContainer(
            obs=sample_obs,
            assays={"proteins": assay},
            validate_on_init=False,  # Skip validation during init
        )

        # Should not raise during init
        assert container.n_samples == 5

        # But should raise when manually validated
        with pytest.raises(ValueError, match="Assay 'proteins', Layer 'raw': Samples 10 != 5"):
            container.validate()

    def test_container_default_validates(
        self, sample_obs: pl.DataFrame, sample_var: pl.DataFrame
    ) -> None:
        """Test that default behavior validates on init (backward compatibility)."""
        # Create container with wrong dimensions
        X_wrong = np.random.randn(10, 5)  # Wrong: 10 samples
        assay = Assay(
            var=sample_var,
            layers={"raw": ScpMatrix(X=X_wrong)},
            validate_on_init=False,
        )

        # Should raise immediately with default validate_on_init=True
        with pytest.raises(ValueError, match="Assay 'proteins', Layer 'raw': Samples 10 != 5"):
            ScpContainer(obs=sample_obs, assays={"proteins": assay})

    def test_container_explicit_true_validates(
        self, sample_obs: pl.DataFrame, sample_var: pl.DataFrame
    ) -> None:
        """Test that validate_on_init=True validates immediately."""
        X_wrong = np.random.randn(10, 5)  # Wrong: 10 samples
        assay = Assay(
            var=sample_var,
            layers={"raw": ScpMatrix(X=X_wrong)},
            validate_on_init=False,
        )

        with pytest.raises(ValueError, match="Assay 'proteins', Layer 'raw': Samples 10 != 5"):
            ScpContainer(
                obs=sample_obs,
                assays={"proteins": assay},
                validate_on_init=True,
            )

    def test_container_with_links_lazy_validation(
        self,
        sample_obs: pl.DataFrame,
        sample_var: pl.DataFrame,
        sample_aggregation_link,
    ) -> None:
        """Test lazy validation with links."""
        # Create peptide assay
        var_peptide = pl.DataFrame({"_index": [f"PEP{i}" for i in range(10)]})
        X_peptide = np.random.randn(5, 10)
        peptide_assay = Assay(
            var=var_peptide,
            layers={"X": ScpMatrix(X=X_peptide)},
            validate_on_init=False,
        )

        # Create protein assay
        X_protein = np.random.randn(5, 5)
        protein_assay = Assay(
            var=sample_var,
            layers={"X": ScpMatrix(X=X_protein)},
            validate_on_init=False,
        )

        container = ScpContainer(
            obs=sample_obs,
            assays={"peptides": peptide_assay, "proteins": protein_assay},
            links=[sample_aggregation_link],
            validate_on_init=False,
        )

        # Should not validate during init
        assert len(container.assays) == 2
        assert len(container.links) == 1

        # Manual validation should succeed
        container.validate()
        assert len(container.assays) == 2

    def test_container_invalid_links_lazy(
        self,
        sample_obs: pl.DataFrame,
        sample_assay: Assay,
    ) -> None:
        """Test that invalid links are caught when validate() is called."""
        from scptensor.core import AggregationLink

        # Create link to non-existent assay
        linkage = pl.DataFrame(
            {
                "source_id": ["PEP1", "PEP2"],
                "target_id": ["PROT1", "PROT1"],
            }
        )
        bad_link = AggregationLink(
            source_assay="nonexistent",
            target_assay="proteins",
            linkage=linkage,
        )

        container = ScpContainer(
            obs=sample_obs,
            assays={"proteins": sample_assay},
            links=[bad_link],
            validate_on_init=False,
        )

        # Should not raise during init
        assert len(container.links) == 1

        # But should raise when manually validated
        with pytest.raises(ValueError, match="Link source assay 'nonexistent' not found"):
            container.validate()


class TestBackwardCompatibility:
    """Test backward compatibility with default validation behavior."""

    def test_assay_backward_compatibility(
        self, sample_var: pl.DataFrame, sample_dense_X: np.ndarray
    ) -> None:
        """Test that default behavior hasn't changed."""
        # This should work exactly as before
        assay = Assay(var=sample_var, layers={"raw": ScpMatrix(X=sample_dense_X)})

        assert assay.n_features == 5
        assert "raw" in assay.layers

    def test_container_backward_compatibility(
        self, sample_obs: pl.DataFrame, sample_assay: Assay
    ) -> None:
        """Test that default behavior hasn't changed."""
        # This should work exactly as before
        container = ScpContainer(obs=sample_obs, assays={"proteins": sample_assay})

        assert container.n_samples == 5
        assert "proteins" in container.assays

    def test_assay_validation_catches_errors(self, sample_var: pl.DataFrame) -> None:
        """Test that validation still catches dimension mismatches."""
        X_wrong = np.random.randn(5, 10)  # Wrong: 10 features

        with pytest.raises(ValueError, match="Features 10 != Assay 5"):
            Assay(var=sample_var, layers={"raw": ScpMatrix(X=X_wrong)})

    def test_container_validation_catches_errors(
        self, sample_obs: pl.DataFrame, sample_var: pl.DataFrame
    ) -> None:
        """Test that validation still catches dimension mismatches."""
        X_wrong = np.random.randn(10, 5)  # Wrong: 10 samples
        assay = Assay(var=sample_var, layers={"raw": ScpMatrix(X=X_wrong)}, validate_on_init=False)

        with pytest.raises(ValueError, match="Samples 10 != 5"):
            ScpContainer(obs=sample_obs, assays={"proteins": assay})


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_assay_empty_layers_lazy(self, sample_var: pl.DataFrame) -> None:
        """Test lazy validation with empty layers."""
        assay = Assay(var=sample_var, layers=None, validate_on_init=False)

        assert len(assay.layers) == 0
        assay.validate()  # Should pass with empty layers

    def test_container_empty_assays_lazy(self, sample_obs: pl.DataFrame) -> None:
        """Test lazy validation with empty assays."""
        container = ScpContainer(obs=sample_obs, assays=None, validate_on_init=False)

        assert len(container.assays) == 0
        container.validate()  # Should pass with empty assays

    def test_assay_subset_preserves_lazy_pattern(self, sample_assay_multi_layer: Assay) -> None:
        """Test that subsetting an assay preserves validation behavior."""
        # Subset with lazy validation
        indices = [0, 2, 4]
        subsetted = sample_assay_multi_layer.subset(indices, copy_data=True)

        # Should be valid by default (backward compatibility)
        assert subsetted.n_features == 3
        subsetted.validate()

    def test_container_copy_preserves_lazy_pattern(self, sample_container: ScpContainer) -> None:
        """Test that copying a container preserves validation behavior."""
        # Shallow copy
        shallow = sample_container.shallow_copy()
        assert shallow.n_samples == sample_container.n_samples

        # Deep copy
        deep = sample_container.deepcopy()
        assert deep.n_samples == sample_container.n_samples

    def test_multiple_validate_calls(self, sample_assay: Assay) -> None:
        """Test that validate() can be called multiple times."""
        assay = Assay(
            var=sample_assay.var,
            layers=sample_assay.layers,
            validate_on_init=False,
        )

        # Should be able to call validate() multiple times
        assay.validate()
        assay.validate()  # Second call should also work

    def test_container_multiple_validate_calls(self, sample_container: ScpContainer) -> None:
        """Test that validate() can be called multiple times."""
        # Recreate container with lazy validation
        container = ScpContainer(
            obs=sample_container.obs,
            assays=sample_container.assays,
            validate_on_init=False,
        )

        # Should be able to call validate() multiple times
        container.validate()
        container.validate()  # Second call should also work
