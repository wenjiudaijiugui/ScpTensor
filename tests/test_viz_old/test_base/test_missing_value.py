"""Tests for MissingValueHandler visualization utility."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from scptensor.viz.base.missing_value import MissingValueHandler


def test_missing_colors_defined():
    """Test missing value colors are defined."""
    assert "missing" in MissingValueHandler.MISSING_COLORS
    assert 1 in MissingValueHandler.MISSING_COLORS  # MBR
    assert 2 in MissingValueHandler.MISSING_COLORS  # LOD
    assert 3 in MissingValueHandler.MISSING_COLORS  # FILTERED


def test_separate_mask():
    """Test separating valid and missing values."""
    X = np.array([[1, 2, 3], [4, 5, 6]])
    M = np.array([[0, 1, 0], [2, 0, 3]])

    X_valid, X_missing, M_missing = MissingValueHandler.separate_mask(X, M)
    assert len(X_valid) == 3
    assert len(X_missing) == 3


def test_separate_mask_all_valid():
    """Test separating when all values are valid."""
    X = np.array([[1, 2, 3], [4, 5, 6]])
    M = np.array([[0, 0, 0], [0, 0, 0]])

    X_valid, X_missing, M_missing = MissingValueHandler.separate_mask(X, M)
    assert len(X_valid) == 6
    assert len(X_missing) == 0
    assert len(M_missing) == 0


def test_separate_mask_all_missing():
    """Test separating when all values are missing."""
    X = np.array([[1, 2, 3], [4, 5, 6]])
    M = np.array([[1, 1, 1], [2, 2, 2]])

    X_valid, X_missing, M_missing = MissingValueHandler.separate_mask(X, M)
    assert len(X_valid) == 0
    assert len(X_missing) == 6
    assert len(M_missing) == 6


def test_create_overlay_scatter_basic():
    """Test basic overlay scatter creation."""
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([1, 2, 3, 4, 5])
    M = np.array([0, 1, 0, 2, 3])

    fig, ax = plt.subplots()
    handles = MissingValueHandler.create_overlay_scatter(x, y, M, ax, size=10.0, alpha=0.8)

    assert "Detected" in handles
    assert len(ax.collections) > 0
    plt.close(fig)


def test_create_overlay_scatter_all_valid():
    """Test overlay scatter with only valid values."""
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([1, 2, 3, 4, 5])
    M = np.array([0, 0, 0, 0, 0])

    fig, ax = plt.subplots()
    handles = MissingValueHandler.create_overlay_scatter(x, y, M, ax)

    assert "Detected" in handles
    plt.close(fig)


def test_create_overlay_scatter_all_missing():
    """Test overlay scatter with only missing values."""
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([1, 2, 3, 4, 5])
    M = np.array([1, 1, 2, 2, 3])

    fig, ax = plt.subplots()
    handles = MissingValueHandler.create_overlay_scatter(x, y, M, ax)

    # Should have entries for each missing type
    assert "Missing (1)" in handles or "Missing (2)" in handles or "Missing (3)" in handles
    plt.close(fig)


def test_create_overlay_scatter_custom_color():
    """Test overlay scatter with custom color for valid points."""
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([1, 2, 3, 4, 5])
    M = np.array([0, 1, 0, 2, 3])

    fig, ax = plt.subplots()
    handles = MissingValueHandler.create_overlay_scatter(x, y, M, ax, color="red")

    assert "Detected" in handles
    plt.close(fig)


def test_mask_values_match():
    """Test that separated mask values match original."""
    X = np.array([[1, 2, 3], [4, 5, 6]])
    M = np.array([[0, 1, 0], [2, 0, 3]])

    X_valid, X_missing, M_missing = MissingValueHandler.separate_mask(X, M)

    # Check that valid values come from correct positions
    expected_valid = np.array([1, 3, 5])
    np.testing.assert_array_equal(X_valid, expected_valid)

    # Check that missing values come from correct positions
    expected_missing = np.array([2, 4, 6])
    np.testing.assert_array_equal(X_missing, expected_missing)

    # Check mask types
    expected_types = np.array([1, 2, 3])
    np.testing.assert_array_equal(M_missing, expected_types)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
