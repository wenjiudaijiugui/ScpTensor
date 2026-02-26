"""Test unified imputation interface."""

import numpy as np
import polars as pl
import pytest

from scptensor.core import Assay, ScpContainer, ScpMatrix
from scptensor.impute.base import (
    ImputeMethod,
    get_impute_method,
    impute,
    list_impute_methods,
    register_impute_method,
)


@pytest.fixture
def create_test_container_with_missing():
    """Factory function to create containers with missing data."""

    def _create(n_samples=50, n_features=20, missing_rate=0.2, random_state=42):
        np.random.seed(random_state)

        # Create correlated data
        U_true = np.random.randn(n_samples, 5)
        V_true = np.random.randn(n_features, 5)
        X_true = U_true @ V_true.T + np.random.randn(n_samples, n_features) * 0.1

        # Add missing values
        X_missing = X_true.copy()
        missing_mask = np.random.rand(n_samples, n_features) < missing_rate
        X_missing[missing_mask] = np.nan

        # Create container
        var = pl.DataFrame({"_index": [f"prot_{i}" for i in range(n_features)]})
        obs = pl.DataFrame({"_index": [f"cell_{i}" for i in range(n_samples)]})

        assay = Assay(var=var)
        assay.add_layer("raw", ScpMatrix(X=X_missing, M=None))

        container = ScpContainer(obs=obs, assays={"proteins": assay})

        return container, X_true, missing_mask

    return _create


def test_impute_method_registry(create_test_container_with_missing):
    """Test that all methods are registered."""
    methods = ["knn", "missforest", "bpca", "lls", "qrilc", "minprob"]

    for method in methods:
        assert get_impute_method(method) is not None, f"Method {method} not registered"


def test_list_impute_methods():
    """Test listing available imputation methods."""
    methods = list_impute_methods()
    assert isinstance(methods, list)
    assert len(methods) >= 6
    assert "knn" in methods
    assert "missforest" in methods
    assert "bpca" in methods


def test_get_impute_method_unknown():
    """Test that unknown method raises error."""
    with pytest.raises(ValueError, match="Unknown imputation method"):
        get_impute_method("unknown_method")


def test_register_impute_method_custom():
    """Test registering a custom imputation method."""

    def custom_validate(data: np.ndarray) -> bool:
        return data.size > 0

    def custom_apply(data: np.ndarray, **kwargs) -> np.ndarray:
        return np.nan_to_num(data, nan=0.0)

    custom_method = ImputeMethod(
        name="custom",
        supports_sparse=False,
        validate=custom_validate,
        apply=custom_apply,
    )

    register_impute_method(custom_method)

    retrieved = get_impute_method("custom")
    assert retrieved.name == "custom"
    assert retrieved.supports_sparse is False


def test_impute_function_knn(create_test_container_with_missing):
    """Test unified impute function with knn method."""
    container, _, _ = create_test_container_with_missing()

    result = impute(
        container,
        method="knn",
        assay="proteins",
        source_layer="raw",
        n_neighbors=5,
    )

    assert isinstance(result, ScpContainer)
    assert "imputed_knn" in result.assays["proteins"].layers

    # Check no NaNs
    X_imputed = result.assays["proteins"].layers["imputed_knn"].X
    assert not np.any(np.isnan(X_imputed))


def test_impute_function_missforest(create_test_container_with_missing):
    """Test unified impute function with missforest method."""
    container, _, _ = create_test_container_with_missing(n_samples=30, n_features=15)

    result = impute(
        container,
        method="missforest",
        assay="proteins",
        source_layer="raw",
        max_iter=5,
        n_estimators=20,
    )

    assert isinstance(result, ScpContainer)
    assert "imputed_missforest" in result.assays["proteins"].layers

    X_imputed = result.assays["proteins"].layers["imputed_missforest"].X
    assert not np.any(np.isnan(X_imputed))


def test_impute_function_bpca(create_test_container_with_missing):
    """Test unified impute function with bpca method."""
    container, _, _ = create_test_container_with_missing()

    result = impute(
        container,
        method="bpca",
        assay="proteins",
        source_layer="raw",
        n_components=5,
    )

    assert isinstance(result, ScpContainer)
    assert "imputed_bpca" in result.assays["proteins"].layers

    X_imputed = result.assays["proteins"].layers["imputed_bpca"].X
    assert not np.any(np.isnan(X_imputed))


def test_impute_function_custom_layer_name(create_test_container_with_missing):
    """Test impute function with custom layer name."""
    container, _, _ = create_test_container_with_missing()

    result = impute(
        container,
        method="knn",
        assay="proteins",
        source_layer="raw",
        new_layer_name="my_imputed",
        n_neighbors=5,
    )

    assert "my_imputed" in result.assays["proteins"].layers
    assert "imputed_knn" not in result.assays["proteins"].layers
