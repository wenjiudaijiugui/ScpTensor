"""Tests for viz.base.validation module."""

import numpy as np
import polars as pl
import pytest

from scptensor import Assay, ScpContainer, ScpMatrix
from scptensor.core.exceptions import LayerNotFoundError, VisualizationError
from scptensor.viz.base.validation import (
    validate_container,
    validate_features,
    validate_groupby,
    validate_layer,
    validate_plot_data,
)


@pytest.fixture
def sample_container():
    """Create a sample container for testing."""
    obs = pl.DataFrame({"_index": [f"S{i}" for i in range(10)], "condition": ["A"] * 5 + ["B"] * 5})
    var = pl.DataFrame({"_index": ["Protein0", "Protein1", "Protein2", "Protein3", "Protein4"]})
    X = np.random.rand(10, 5)
    assay = Assay(var=var, layers={"raw": ScpMatrix(X=X)})
    return ScpContainer(obs=obs, assays={"proteins": assay})


def test_validate_container_pass(sample_container):
    """Test valid container passes."""
    validate_container(sample_container)  # Should not raise


def test_validate_container_none():
    """Test None container raises error."""
    with pytest.raises(VisualizationError, match="Container cannot be None"):
        validate_container(None)


def test_validate_container_wrong_type():
    """Test wrong type raises error."""
    with pytest.raises(VisualizationError, match="Expected ScpContainer"):
        validate_container("not_a_container")


def test_validate_layer_exists(sample_container):
    """Test existing layer passes."""
    validate_layer(sample_container, "proteins", "raw")  # Should not raise


def test_validate_layer_missing_assay(sample_container):
    """Test missing assay raises error."""
    with pytest.raises(VisualizationError, match="Assay 'missing' not found"):
        validate_layer(sample_container, "missing", "raw")


def test_validate_layer_missing_layer(sample_container):
    """Test missing layer raises error."""
    with pytest.raises(LayerNotFoundError, match="Layer 'nonexistent' not found"):
        validate_layer(sample_container, "proteins", "nonexistent")


def test_validate_features_pass(sample_container):
    """Test valid features pass."""
    validate_features(sample_container, "proteins", ["Protein0", "Protein2"])  # Should not raise


def test_validate_features_missing(sample_container):
    """Test missing features raise error."""
    with pytest.raises(VisualizationError, match="Features not found"):
        validate_features(sample_container, "proteins", ["Protein0", "MISSING"])


def test_validate_groupby_pass(sample_container):
    """Test valid groupby column passes."""
    validate_groupby(sample_container, "condition")  # Should not raise


def test_validate_groupby_missing(sample_container):
    """Test missing groupby column raises error."""
    with pytest.raises(VisualizationError, match="Column 'missing' not found in obs"):
        validate_groupby(sample_container, "missing")


def test_validate_plot_data_pass():
    """Test sufficient data passes."""
    validate_plot_data(np.array([1, 2, 3]), n_min=1)  # Should not raise


def test_validate_plot_data_insufficient():
    """Test insufficient data raises error."""
    with pytest.raises(VisualizationError, match="Insufficient data"):
        validate_plot_data(np.array([]), n_min=1)


def test_validate_plot_data_empty():
    """Test empty array raises error."""
    with pytest.raises(VisualizationError, match="Insufficient data"):
        validate_plot_data(np.array([]), n_min=5)


def test_validate_plot_data_exactly_min():
    """Test data exactly at minimum passes."""
    validate_plot_data(np.array([1, 2, 3]), n_min=3)  # Should not raise
