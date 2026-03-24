"""Tests for sparse utilities module."""

import numpy as np
import polars as pl
import scipy.sparse as sp

import scptensor.core.sparse_utils as sparse_utils_mod
from scptensor.core.sparse_utils import (
    auto_convert_for_operation,
    cleanup_layers,
    ensure_sparse_format,
    get_format_recommendation,
    get_memory_usage,
    get_sparsity_ratio,
    is_sparse_matrix,
    sparse_center_rows,
    sparse_col_operation,
    sparse_copy,
    sparse_row_operation,
    sparse_safe_log1p,
    sparse_safe_log1p_with_scale,
    to_sparse_if_beneficial,
)
from scptensor.core.structures import Assay, ScpContainer, ScpMatrix


def test_is_sparse_matrix():
    """Test sparse matrix detection."""
    dense = np.array([[1, 2], [3, 4]])
    sparse = sp.csr_matrix([[1, 0], [0, 4]])

    assert not is_sparse_matrix(dense), "Dense array should not be detected as sparse"
    assert is_sparse_matrix(sparse), "CSR matrix should be detected as sparse"


def test_get_sparsity_ratio():
    """Test sparsity ratio calculation."""
    # Test with dense array (7 zeros out of 9 elements = 7/9)
    X_test = np.array([[1, 0, 0], [0, 0, 2], [0, 0, 0]])
    expected = 7 / 9  # 7 zeros out of 9 elements
    result = get_sparsity_ratio(X_test)
    assert abs(result - expected) < 1e-10, f"Expected {expected}, got {result}"

    # Test with sparse matrix
    X_sparse = sp.csr_matrix(X_test)
    result_sparse = get_sparsity_ratio(X_sparse)
    assert abs(result_sparse - expected) < 1e-10, f"Expected {expected}, got {result_sparse}"


def test_to_sparse_if_beneficial():
    """Test conditional sparse conversion."""
    # Should convert (66% sparse > 50% threshold)
    X_sparse_input = np.array([[1, 0, 0], [0, 0, 2]])
    result = to_sparse_if_beneficial(X_sparse_input, threshold=0.5)
    assert is_sparse_matrix(result), "Should convert to sparse above threshold"

    # Should NOT convert (20% sparse < 50% threshold)
    X_dense_input = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
    result2 = to_sparse_if_beneficial(X_dense_input, threshold=0.5)
    assert not is_sparse_matrix(result2), "Should NOT convert to sparse below threshold"


def test_ensure_sparse_format():
    """Test sparse format conversion."""
    # CSC to CSR
    X_csc = sp.csc_matrix([[1, 0], [0, 4]])
    X_csr = ensure_sparse_format(X_csc, format="csr")
    assert isinstance(X_csr, sp.csr_matrix), "Should convert CSC to CSR"

    # Dense to CSR
    X_dense = np.array([[1, 0], [0, 4]])
    X_csr2 = ensure_sparse_format(X_dense, format="csr")
    assert isinstance(X_csr2, sp.csr_matrix), "Should convert dense to CSR"


def test_sparse_copy():
    """Test sparse copy."""
    X = sp.csr_matrix([[1, 0], [0, 4]])
    X_copy = sparse_copy(X)

    # Verify it's a copy
    X_copy[0, 0] = 99
    assert X[0, 0] == 1, "Original should be unchanged after copy modification"


def test_get_memory_usage():
    """Test memory usage calculation."""
    # Test with sparse matrix
    X_sparse = sp.csr_matrix([[1, 0], [0, 4]])
    stats = get_memory_usage(X_sparse)
    assert stats["is_sparse"], "Should detect as sparse"
    assert stats["shape"] == (2, 2), "Shape should be (2, 2)"
    assert "nbytes" in stats, "Should include nbytes"

    # Test with dense matrix
    X_dense = np.array([[1, 2], [3, 4]])
    stats_dense = get_memory_usage(X_dense)
    assert not stats_dense["is_sparse"], "Should detect as dense"
    assert stats_dense["shape"] == (2, 2), "Shape should be (2, 2)"


def test_sparse_vs_dense_memory():
    """Compare memory usage between sparse and dense."""
    # Create a sparse matrix (90% zeros)
    n_rows, n_cols = 1000, 1000
    n_elements = n_rows * n_cols
    n_nonzero = int(0.1 * n_elements)  # 10% non-zero

    # Create sparse data
    rng = np.random.default_rng(42)
    rows = rng.integers(0, n_rows, n_nonzero)
    cols = rng.integers(0, n_cols, n_nonzero)
    data = rng.standard_normal(n_nonzero)

    X_sparse = sp.csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))
    X_dense = X_sparse.toarray()

    stats_sparse = get_memory_usage(X_sparse)
    stats_dense = get_memory_usage(X_dense)

    compression_ratio = stats_dense["nbytes"] / stats_sparse["nbytes"]

    assert compression_ratio > 2, "Sparse should use significantly less memory"


def test_sparse_col_operation_sum_and_mean():
    """Column reductions should use explicit stored values only."""
    X = sp.csr_matrix([[1.0, 0.0, 2.0], [0.0, 0.0, 3.0], [4.0, 5.0, 0.0]])

    result_sum = sparse_col_operation(X, np.sum)
    result_mean = sparse_col_operation(X, np.mean)

    np.testing.assert_allclose(result_sum, np.array([5.0, 5.0, 5.0]))
    np.testing.assert_allclose(result_mean, np.array([2.5, 5.0, 2.5]))


def test_sparse_col_operation_empty_columns_return_zero():
    """Empty sparse columns should follow the stable 0.0 fallback convention."""
    X = sp.csr_matrix([[0.0, 0.0, 2.0], [0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])

    result_sum = sparse_col_operation(X, np.sum)
    result_mean = sparse_col_operation(X, np.mean)

    np.testing.assert_allclose(result_sum, np.array([4.0, 0.0, 2.0]))
    np.testing.assert_allclose(result_mean, np.array([4.0, 0.0, 2.0]))


def test_sparse_row_and_col_custom_functions_still_use_fallback_semantics():
    """Custom reducers should still see only explicit stored values."""
    X = sp.csr_matrix([[1.0, 0.0, 2.0], [0.0, 7.0, 0.0], [4.0, 5.0, 0.0]])

    row_max = sparse_row_operation(X, np.max)
    col_max = sparse_col_operation(X, np.max)

    np.testing.assert_allclose(row_max, np.array([2.0, 7.0, 5.0]))
    np.testing.assert_allclose(col_max, np.array([4.0, 7.0, 2.0]))


def test_sparse_center_rows_returns_csr_and_explicit_nonzero_means():
    """sparse_center_rows should keep CSR storage and use explicit non-zero means."""
    X = sp.csc_matrix([[1.0, 0.0, 3.0], [0.0, 0.0, 0.0], [4.0, 8.0, 0.0]])

    centered, means = sparse_center_rows(X)

    assert isinstance(centered, sp.csr_matrix)
    np.testing.assert_allclose(centered.toarray(), X.toarray())
    np.testing.assert_allclose(means, np.array([2.0, 0.0, 6.0]))


def test_sparse_center_rows_uses_supplied_row_means_without_recomputing():
    """Providing row_means should preserve the supplied parameter contract."""
    X = sp.csr_matrix([[1.0, 0.0], [0.0, 2.0]])
    supplied = np.array([10.0, 20.0], dtype=np.float64)

    centered, means = sparse_center_rows(X, row_means=supplied)

    assert isinstance(centered, sp.csr_matrix)
    np.testing.assert_allclose(centered.toarray(), X.toarray())
    assert means is supplied


def test_sparse_safe_log1p_preserves_structural_zeros_and_input():
    """Sparse log1p should only transform explicit data and leave input untouched."""
    X = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 4.0]]))
    X_before = X.copy()

    out = sparse_safe_log1p(X, offset=1.0, use_jit=False)

    assert sp.isspmatrix_csr(out)
    np.testing.assert_allclose(
        out.toarray(),
        np.array([[np.log1p(1.0), 0.0], [0.0, np.log1p(4.0)]]),
    )
    np.testing.assert_array_equal(X.toarray(), X_before.toarray())


def test_sparse_safe_log1p_with_scale_matches_dense_formula_on_all_entries():
    """Non-unit offsets must densify because structural zeros become non-zero."""
    X = sp.csr_matrix(np.array([[2.0, 0.0, 3.0], [0.0, 5.0, 0.0]]))
    out = sparse_safe_log1p_with_scale(X, offset=2.0, scale=np.log(2), use_jit=False)

    expected = np.log(np.array([[2.0, 0.0, 3.0], [0.0, 5.0, 0.0]]) + 2.0) / np.log(2)
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, expected)


def test_sparse_safe_log1p_with_scale_casts_integer_sparse_data_to_float():
    """Integer sparse inputs should still produce floating transformed outputs."""
    X = sp.csr_matrix(np.array([[1, 0], [0, 3]], dtype=np.int32))

    out = sparse_safe_log1p_with_scale(X, offset=1.0, scale=2.0, use_jit=False)

    assert out.dtype.kind == "f"
    np.testing.assert_allclose(
        out.toarray(),
        np.array([[np.log1p(1.0) / 2.0, 0.0], [0.0, np.log1p(3.0) / 2.0]]),
    )


def test_sparse_safe_log1p_skips_kernel_below_threshold(monkeypatch):
    """use_jit=True should still stay on NumPy path below the auto-JIT threshold."""
    X = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 4.0]]))
    calls = {"count": 0}

    def fake_kernel_factory():
        calls["count"] += 1
        raise AssertionError("JIT kernel should not be requested below threshold")

    monkeypatch.setattr(sparse_utils_mod, "_JIT_THRESHOLD", X.nnz + 10)
    monkeypatch.setattr(sparse_utils_mod, "_get_jit_log_with_scale_kernel", fake_kernel_factory)

    out = sparse_safe_log1p(X, offset=1.0, use_jit=True)

    assert calls["count"] == 0
    np.testing.assert_allclose(
        out.toarray(),
        np.array([[np.log1p(1.0), 0.0], [0.0, np.log1p(4.0)]]),
    )


def test_sparse_safe_log1p_with_scale_uses_kernel_above_threshold(monkeypatch):
    """Above threshold, sparse log helpers should honor the available JIT kernel."""
    X = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 4.0]]))
    calls = {"count": 0}

    def fake_kernel_factory():
        calls["count"] += 1

        def fake_kernel(data: np.ndarray, offset: float, inv_scale: float) -> np.ndarray:
            return np.full(data.shape, 10.0 + offset + inv_scale, dtype=np.float64)

        return fake_kernel

    monkeypatch.setattr(sparse_utils_mod, "_JIT_THRESHOLD", 0)
    monkeypatch.setattr(sparse_utils_mod, "_get_jit_log_with_scale_kernel", fake_kernel_factory)

    out = sparse_safe_log1p_with_scale(X, offset=1.0, scale=2.0, use_jit=True)

    assert calls["count"] == 1
    np.testing.assert_allclose(out.toarray(), np.array([[10.5, 0.0], [0.0, 10.5]]))


def test_sparse_safe_log1p_with_scale_falls_back_when_kernel_unavailable(monkeypatch):
    """Even above threshold, sparse log helpers must fall back when no JIT kernel exists."""
    X = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 4.0]]))

    monkeypatch.setattr(sparse_utils_mod, "_JIT_THRESHOLD", 0)
    monkeypatch.setattr(sparse_utils_mod, "_get_jit_log_with_scale_kernel", lambda: None)

    out = sparse_safe_log1p_with_scale(X, offset=1.0, scale=2.0, use_jit=True)

    assert sp.isspmatrix_csr(out)
    np.testing.assert_allclose(
        out.toarray(),
        np.array([[np.log1p(1.0) / 2.0, 0.0], [0.0, np.log1p(4.0) / 2.0]]),
    )


def test_cleanup_layers_removes_only_unlisted_layers() -> None:
    """cleanup_layers should mutate only the target assay's layer mapping."""
    obs = pl.DataFrame({"_index": ["s1", "s2"]})
    var = pl.DataFrame({"_index": ["p1", "p2"]})
    assay = Assay(
        var=var,
        layers={
            "raw": ScpMatrix(X=np.ones((2, 2))),
            "norm": ScpMatrix(X=np.full((2, 2), 2.0)),
            "imputed": ScpMatrix(X=np.full((2, 2), 3.0)),
        },
    )
    container = ScpContainer(obs=obs, assays={"proteins": assay})

    cleanup_layers(container, assay_name="proteins", keep_layers=["raw", "imputed"])

    assert list(container.assays["proteins"].layers.keys()) == ["raw", "imputed"]


def test_auto_convert_for_operation_keeps_dense_inputs_under_current_contract() -> None:
    """Dense inputs should pass through unchanged instead of being sparsified implicitly."""
    X = np.array([[1.0, 0.0], [0.0, 2.0]])

    out = auto_convert_for_operation(X, operation="row_wise")

    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, X)


def test_get_format_recommendation_prefers_dense_for_low_sparsity() -> None:
    """Low-sparsity matrices should stay dense under the current recommendation contract."""
    assert get_format_recommendation(10, 10, nnz=90, operations=["row_wise"]) == "dense"
