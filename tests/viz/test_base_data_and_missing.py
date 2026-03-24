"""Tests for base visualization data and mask helpers."""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest
from scipy import sparse

from scptensor.core import Assay, ScpContainer, ScpMatrix
from scptensor.viz.base import DataExtractor, MissingValueHandler


def test_get_expression_matrix_filters_in_requested_order(sample_container: ScpContainer) -> None:
    """Requested sample/feature order should be preserved in extracted matrix."""
    sample_ids = sample_container.sample_ids.to_list()
    feature_ids = sample_container.assays["proteins"].feature_ids.to_list()
    selected_samples = [sample_ids[3], sample_ids[1]]
    selected_features = [feature_ids[2], feature_ids[0]]

    x, obs, var = DataExtractor.get_expression_matrix(
        sample_container,
        assay_name="proteins",
        layer="raw",
        var_names=selected_features,
        samples=selected_samples,
    )

    raw = sample_container.assays["proteins"].layers["raw"].X
    np.testing.assert_allclose(x, raw[[3, 1]][:, [2, 0]])

    obs_id_col = sample_container.obs.columns.index(sample_container.sample_id_col)
    var_id_col = sample_container.assays["proteins"].var.columns.index(
        sample_container.assays["proteins"].feature_id_col,
    )
    assert obs[:, obs_id_col].tolist() == selected_samples
    assert var[:, var_id_col].tolist() == selected_features


def test_get_expression_matrix_missing_sample_raises(sample_container: ScpContainer) -> None:
    """Unknown sample IDs should raise explicit ValueError."""
    with pytest.raises(ValueError, match="Samples not found"):
        DataExtractor.get_expression_matrix(
            sample_container,
            assay_name="proteins",
            layer="raw",
            samples=["S_DOES_NOT_EXIST"],
        )


def test_get_expression_matrix_missing_feature_raises(sample_container: ScpContainer) -> None:
    """Unknown feature IDs should raise explicit ValueError."""
    with pytest.raises(ValueError, match="Features not found"):
        DataExtractor.get_expression_matrix(
            sample_container,
            assay_name="proteins",
            layer="raw",
            var_names=["P_DOES_NOT_EXIST"],
        )


def test_get_expression_matrix_sparse_inputs_are_converted_to_dense() -> None:
    """Sparse X/M layers should be returned as dense arrays."""
    obs = pl.DataFrame({"_index": ["S1", "S2"]})
    var = pl.DataFrame({"_index": ["P1", "P2"]})

    x_sparse = sparse.csr_matrix(np.array([[1.0, 2.0], [3.0, 4.0]]))
    m_sparse = sparse.csr_matrix(np.array([[0, 5], [0, 0]], dtype=np.int8))
    container = ScpContainer(
        obs=obs,
        assays={"proteins": Assay(var=var, layers={"raw": ScpMatrix(X=x_sparse, M=m_sparse)})},
    )

    x, _, _ = DataExtractor.get_expression_matrix(container, "proteins", "raw")
    assert isinstance(x, np.ndarray)
    assert x.shape == (2, 2)
    assert np.isclose(x[1, 1], 4.0)


def test_get_group_data_missing_column_raises(sample_container: ScpContainer) -> None:
    """Grouping by unknown obs column should fail fast."""
    with pytest.raises(ValueError, match="not found in obs"):
        DataExtractor.get_group_data(sample_container, "nonexistent_group")


def test_handle_missing_values_modes_and_validation() -> None:
    """Missing-value handling should support all modes and validate inputs."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    m = np.array([[0, 1], [5, 0]], dtype=np.int8)

    x_valid, x_missing, m_types = DataExtractor.handle_missing_values(x, m, method="separate")
    np.testing.assert_allclose(np.sort(x_valid), np.array([1.0, 4.0]))
    np.testing.assert_allclose(np.sort(x_missing), np.array([2.0, 3.0]))
    np.testing.assert_array_equal(np.sort(m_types), np.array([1, 5], dtype=np.int8))

    x_transparent, _, m_transparent = DataExtractor.handle_missing_values(
        x,
        m,
        method="transparent",
    )
    assert np.isnan(x_transparent[0, 1])
    assert np.isnan(x_transparent[1, 0])
    np.testing.assert_array_equal(np.sort(m_transparent), np.array([1, 5], dtype=np.int8))

    x_imputed, x_missing_imputed, m_imputed = DataExtractor.handle_missing_values(
        x,
        m,
        method="imputed",
    )
    np.testing.assert_allclose(x_imputed, x)
    assert x_missing_imputed.size == 0
    assert m_imputed.size == 0

    x_no_mask_valid, x_no_mask_missing, m_no_mask_types = DataExtractor.handle_missing_values(
        x,
        None,
        method="separate",
    )
    np.testing.assert_allclose(np.sort(x_no_mask_valid), np.sort(x.ravel()))
    assert x_no_mask_missing.size == 0
    assert m_no_mask_types.size == 0

    with pytest.raises(ValueError, match="Unsupported method"):
        DataExtractor.handle_missing_values(x, m, method="unknown")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="same shape"):
        DataExtractor.handle_missing_values(x, np.array([0, 1], dtype=np.int8), method="separate")


def test_missing_value_handler_overlay_handles_all_nonzero_codes() -> None:
    """Overlay scatter should render known and unknown non-zero mask codes."""
    plt.close("all")
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([3.0, 2.0, 1.0, 0.0])
    m = np.array([0, 5, 6, 7], dtype=np.int8)
    fig, ax = plt.subplots()

    handles = MissingValueHandler.create_overlay_scatter(x, y, m, ax=ax, size=10.0, alpha=0.9)
    assert "Detected" in handles
    assert "Missing (5)" in handles
    assert "Missing (6)" in handles
    assert "Missing (7)" in handles
    assert len(ax.collections) == 4  # valid + three distinct mask codes
    plt.close("all")


def test_missing_value_handler_overlay_shape_mismatch_raises() -> None:
    """Overlay scatter should reject length-mismatched x/y/M arrays."""
    plt.close("all")
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="same number of elements"):
        MissingValueHandler.create_overlay_scatter(
            np.array([0.0, 1.0]),
            np.array([0.0]),
            np.array([0, 1], dtype=np.int8),
            ax=ax,
        )
    plt.close("all")
