"""Performance regression tests using pytest-benchmark.

This module contains performance tests that track execution time over commits.
Run with:
    uv run pytest tests/test_performance.py --benchmark-only

For comparison between commits:
    uv run pytest tests/test_performance.py --benchmark-compare=<git_commit>

To save baseline for future comparison:
    uv run pytest tests/test_performance.py --benchmark-autosave

To generate histogram:
    uv run pytest tests/test_performance.py --benchmark-histogram
"""

import numpy as np
import pytest
import scipy.sparse as sp

from scptensor.core import jit_ops, sparse_utils
from scptensor.core.structures import Assay, MaskCode, ScpContainer, ScpMatrix

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(params=[(100, 50), (500, 100), (1000, 200)])
def small_data(request):
    """Generate small test data for fast benchmarking."""
    n_samples, n_features = request.param
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    # Add some NaN values
    mask = np.random.rand(n_samples, n_features) < 0.2
    X[mask] = np.nan
    return X


@pytest.fixture(params=[0.5, 0.7, 0.9])
def sparse_matrix(request):
    """Generate sparse matrix with varying sparsity."""
    sparsity = request.param
    np.random.seed(42)
    n_samples, n_features = 500, 100
    X_dense = np.random.randn(n_samples, n_features)
    mask = np.random.rand(n_samples, n_features) < sparsity
    X_dense[mask] = 0
    return sp.csr_matrix(X_dense), sparsity


@pytest.fixture
def test_container():
    """Create a test ScpContainer for benchmarking."""
    np.random.seed(42)
    n_samples, n_features = 100, 50
    X = np.abs(np.random.randn(n_samples, n_features)) * 100 + 1

    import polars as pl

    obs = pl.DataFrame({"_index": [f"s{i}" for i in range(n_samples)]})
    var = pl.DataFrame({"_index": [f"f{i}" for i in range(n_features)]})

    assay = Assay(var=var)
    assay.add_layer("raw", ScpMatrix(X=X))
    return ScpContainer(obs=obs, assays={"test": assay})


# ============================================================================
# JIT Operations Benchmarks
# ============================================================================


class TestJITOperations:
    """Benchmark JIT-compiled operations."""

    def test_euclidean_distance_no_nan(self, benchmark, small_data):
        """Benchmark euclidean distance computation ignoring NaNs."""
        X = small_data

        def bench():
            return jit_ops.euclidean_distance_no_nan(X[0], X[1])

        result = benchmark(bench)
        assert isinstance(result, float)

    def test_mean_no_nan(self, benchmark, small_data):
        """Benchmark mean computation ignoring NaNs."""
        X = small_data
        x = X[0]

        def bench():
            return jit_ops.mean_no_nan(x)

        result = benchmark(bench)
        assert isinstance(result, float)

    def test_var_no_nan(self, benchmark, small_data):
        """Benchmark variance computation ignoring NaNs."""
        X = small_data
        x = X[0]

        def bench():
            return jit_ops.var_no_nan(x)

        result = benchmark(bench)
        assert isinstance(result, float)

    def test_mean_axis_no_nan(self, benchmark, small_data):
        """Benchmark column-wise mean computation ignoring NaNs."""
        X = small_data

        def bench():
            return jit_ops.mean_axis_no_nan(X, 0)

        result = benchmark(bench)
        assert len(result) == X.shape[1]

    def test_var_axis_no_nan(self, benchmark, small_data):
        """Benchmark column-wise variance computation ignoring NaNs."""
        X = small_data

        def bench():
            return jit_ops.var_axis_no_nan(X, 0)

        result = benchmark(bench)
        assert len(result) == X.shape[1]

    def test_count_mask_codes(self, benchmark):
        """Benchmark mask code counting."""
        np.random.seed(42)
        M = np.random.randint(0, 6, (500, 100), dtype=np.int8)

        def bench():
            return jit_ops.count_mask_codes(M)

        result = benchmark(bench)
        assert len(result) == 7

    def test_find_missing_indices(self, benchmark):
        """Benchmark finding missing value indices."""
        np.random.seed(42)
        M = np.random.randint(0, 6, (500, 100), dtype=np.int8)

        def bench():
            return jit_ops.find_missing_indices(M, (1, 2, 5))

        rows, cols = benchmark(bench)
        assert isinstance(rows, np.ndarray)

    def test_fill_nan_with_value(self, benchmark):
        """Benchmark filling NaN values."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        X[np.random.rand(*X.shape) < 0.2] = np.nan

        def bench():
            X_copy = X.copy()
            jit_ops.fill_nan_with_value(X_copy, 0.0)
            return X_copy

        result = benchmark(bench)
        assert not np.any(np.isnan(result))

    def test_knn_weighted_impute(self, benchmark):
        """Benchmark KNN weighted imputation."""
        np.random.seed(42)
        neighbor_values = np.random.randn(20)
        distances = np.abs(np.random.randn(20)) + 0.1

        def bench():
            return jit_ops.knn_weighted_impute(
                neighbor_values, distances, k=5, use_distance_weights=True
            )

        result = benchmark(bench)
        assert isinstance(result, float)


# ============================================================================
# Sparse Operations Benchmarks
# ============================================================================


class TestSparseOperations:
    """Benchmark sparse matrix operations."""

    def test_sparse_safe_log1p(self, benchmark, sparse_matrix):
        """Benchmark sparse log1p transformation."""
        X_sparse, sparsity = sparse_matrix

        def bench():
            result = sparse_utils.sparse_safe_log1p(X_sparse.copy())
            # Force computation
            _ = result.data
            return result

        result = benchmark(bench)
        assert sp.issparse(result)

    def test_sparse_safe_log1p_with_scale(self, benchmark, sparse_matrix):
        """Benchmark sparse log1p with scaling."""
        X_sparse, sparsity = sparse_matrix
        scale = np.log(2.0)

        def bench():
            result = sparse_utils.sparse_safe_log1p_with_scale(
                X_sparse.copy(), offset=1.0, scale=scale
            )
            _ = result.data
            return result

        result = benchmark(bench)
        assert sp.issparse(result)

    def test_sparse_multiply_rowwise(self, benchmark, sparse_matrix):
        """Benchmark row-wise multiplication of sparse matrix."""
        X_sparse, sparsity = sparse_matrix
        factors = np.random.randn(X_sparse.shape[0])

        def bench():
            return sparse_utils.sparse_multiply_rowwise(X_sparse, factors)

        result = benchmark(bench)
        assert sp.issparse(result)

    def test_sparse_multiply_colwise(self, benchmark, sparse_matrix):
        """Benchmark column-wise multiplication of sparse matrix."""
        X_sparse, sparsity = sparse_matrix
        factors = np.random.randn(X_sparse.shape[1])

        def bench():
            return sparse_utils.sparse_multiply_colwise(X_sparse, factors)

        result = benchmark(bench)
        assert sp.issparse(result)

    def test_to_sparse_if_beneficial(self, benchmark):
        """Benchmark sparse conversion decision."""
        np.random.seed(42)
        X_dense = np.random.randn(100, 50)
        X_dense[np.random.rand(*X_dense.shape) < 0.6] = 0

        def bench():
            return sparse_utils.to_sparse_if_beneficial(X_dense, threshold=0.5)

        result = benchmark(bench)
        # Should be sparse since 60% zeros
        assert sp.issparse(result)

    def test_ensure_sparse_format(self, benchmark, sparse_matrix):
        """Benchmark sparse format conversion."""
        X_sparse, sparsity = sparse_matrix

        def bench():
            return sparse_utils.ensure_sparse_format(X_sparse, "csr")

        result = benchmark(bench)
        assert isinstance(result, sp.csr_matrix)

    def test_get_sparsity_ratio(self, benchmark, sparse_matrix):
        """Benchmark sparsity ratio calculation."""
        X_sparse, sparsity = sparse_matrix

        def bench():
            return sparse_utils.get_sparsity_ratio(X_sparse)

        result = benchmark(bench)
        assert 0 <= result <= 1


# ============================================================================
# Matrix Operations Benchmarks
# ============================================================================


class TestMatrixOperations:
    """Benchmark ScpMatrix operations."""

    def test_matrix_copy(self, benchmark):
        """Benchmark ScpMatrix copying."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        M = np.zeros((100, 50), dtype=np.int8)
        matrix = ScpMatrix(X=X, M=M)

        def bench():
            return matrix.copy()

        result = benchmark(bench)
        assert result.X.shape == X.shape

    def test_get_valid_mask(self, benchmark):
        """Benchmark valid mask extraction."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        M = np.random.randint(0, 6, (100, 50), dtype=np.int8)
        matrix = ScpMatrix(X=X, M=M)

        def bench():
            from scptensor.core.matrix_ops import MatrixOps

            return MatrixOps.get_valid_mask(matrix)

        result = benchmark(bench)
        assert result.shape == (100, 50)

    def test_get_mask_statistics(self, benchmark):
        """Benchmark mask statistics computation."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        M = np.random.randint(0, 6, (100, 50), dtype=np.int8)
        matrix = ScpMatrix(X=X, M=M)

        def bench():
            from scptensor.core.matrix_ops import MatrixOps

            return MatrixOps.get_mask_statistics(matrix)

        result = benchmark(bench)
        assert "VALID" in result

    def test_mark_values(self, benchmark):
        """Benchmark marking values with mask code."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        M = np.zeros((100, 50), dtype=np.int8)
        matrix = ScpMatrix(X=X, M=M)
        indices = (np.array([0, 1, 2]), np.array([0, 1, 2]))

        def bench():
            from scptensor.core.matrix_ops import MatrixOps

            return MatrixOps.mark_values(matrix, indices, MaskCode.IMPUTED)

        result = benchmark(bench)
        assert result.M[0, 0] == MaskCode.IMPUTED.value


# ============================================================================
# Normalization Benchmarks
# ============================================================================


class TestNormalization:
    """Benchmark normalization operations."""

    def test_log_normalize_dense(self, benchmark, test_container):
        """Benchmark log normalization on dense data."""
        from scptensor.normalization.log import norm_log as log_normalize

        def bench():
            container = test_container
            # Reset layer
            container.assays["test"].layers = {"raw": container.assays["test"].layers["raw"]}
            return log_normalize(container, "test", "raw", "log")

        result = benchmark(bench)
        assert "log" in result.assays["test"].layers

    def test_log_normalize_sparse(self, benchmark):
        """Benchmark log normalization on sparse data."""
        import polars as pl

        from scptensor.normalization.log import norm_log as log_normalize

        np.random.seed(42)
        n_samples, n_features = 100, 50
        X_dense = np.abs(np.random.randn(n_samples, n_features)) * 100 + 1
        X_sparse = sp.csr_matrix(X_dense)

        obs = pl.DataFrame({"_index": [f"s{i}" for i in range(n_samples)]})
        var = pl.DataFrame({"_index": [f"f{i}" for i in range(n_features)]})

        assay = Assay(var=var)
        assay.add_layer("raw", ScpMatrix(X=X_sparse))
        container = ScpContainer(obs=obs, assays={"test": assay})

        def bench():
            c = container
            c.assays["test"].layers = {"raw": c.assays["test"].layers["raw"]}
            return log_normalize(c, "test", "raw", "log")

        result = benchmark(bench)
        assert "log" in result.assays["test"].layers


# ============================================================================
# Memory Usage Tests
# ============================================================================


class TestMemoryUsage:
    """Test memory usage of key operations."""

    @pytest.mark.parametrize("n_samples,n_features", [(100, 50), (500, 100)])
    def test_memory_matrix_copy(self, benchmark, n_samples, n_features):
        """Benchmark memory usage of matrix copying."""
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features)
        M = np.zeros((n_samples, n_features), dtype=np.int8)
        matrix = ScpMatrix(X=X, M=M)

        def bench():
            return sparse_utils.sparse_copy(matrix.X)

        result = benchmark(bench)
        assert result.shape == X.shape

    @pytest.mark.parametrize("sparsity", [0.5, 0.7, 0.9])
    def test_memory_sparse_operations(self, benchmark, sparsity):
        """Benchmark memory usage of sparse operations."""
        np.random.seed(42)
        n_samples, n_features = 500, 100
        X_dense = np.random.randn(n_samples, n_features)
        mask = np.random.rand(n_samples, n_features) < sparsity
        X_dense[mask] = 0
        X_sparse = sp.csr_matrix(X_dense)

        def bench():
            return sparse_utils.get_memory_usage(X_sparse)

        result = benchmark(bench)
        assert "nbytes" in result


# ============================================================================
# Performance Thresholds
# ============================================================================


@pytest.mark.slow
class TestPerformanceThresholds:
    """Ensure operations complete within acceptable time limits."""

    @pytest.mark.timeout(5)
    def test_jit_operations_timeout(self):
        """Verify JIT operations complete quickly."""
        np.random.seed(42)
        X = np.random.randn(1000, 200)
        X[np.random.rand(*X.shape) < 0.2] = np.nan

        # These should all complete within 5 seconds
        jit_ops.mean_axis_no_nan(X, 0)
        jit_ops.var_axis_no_nan(X, 0)
        jit_ops.euclidean_distance_no_nan(X[0], X[1])

    @pytest.mark.timeout(10)
    def test_sparse_operations_timeout(self):
        """Verify sparse operations complete quickly."""
        np.random.seed(42)
        X = sp.random(1000, 200, density=0.1, format="csr")

        # These should all complete within 10 seconds
        sparse_utils.sparse_safe_log1p(X)
        sparse_utils.sparse_multiply_rowwise(X, np.random.randn(1000))
        sparse_utils.sparse_multiply_colwise(X, np.random.randn(200))
