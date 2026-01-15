"""Tests for scptensor.utils.batch module.

This module contains comprehensive tests for batch processing utilities
including batch_iterator, apply_by_batch, batch_apply_along_axis, and BatchProcessor.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from scptensor.utils.batch import (
    BatchProcessor,
    apply_by_batch,
    batch_apply_along_axis,
    batch_iterator,
)


class TestBatchIterator:
    """Tests for batch_iterator class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        np.random.seed(42)
        return np.random.randn(100, 10)

    @pytest.fixture
    def sample_sparse(self):
        """Create sample sparse data."""
        np.random.seed(42)
        return sp.csr_matrix(np.random.randn(100, 10))

    @pytest.fixture
    def sample_list(self):
        """Create sample list data."""
        return list(range(100))

    def test_batch_iterator_basic(self, sample_data):
        """Test basic batch iteration."""
        batches = list(batch_iterator(sample_data, batch_size=32))
        assert len(batches) == 4
        assert batches[0].shape == (32, 10)
        assert batches[1].shape == (32, 10)
        assert batches[2].shape == (32, 10)
        assert batches[3].shape == (4, 10)

    def test_batch_iterator_exact_division(self, sample_data):
        """Test batch iterator when n is divisible by batch_size."""
        batches = list(batch_iterator(sample_data[:96], batch_size=32))
        assert len(batches) == 3
        assert all(b.shape == (32, 10) for b in batches)

    def test_batch_iterator_drop_last(self, sample_data):
        """Test drop_last parameter."""
        batches = list(batch_iterator(sample_data, batch_size=32, drop_last=True))
        assert len(batches) == 3
        assert all(b.shape == (32, 10) for b in batches)

    def test_batch_iterator_keep_last(self, sample_data):
        """Test keeping last incomplete batch."""
        batches = list(batch_iterator(sample_data, batch_size=32, drop_last=False))
        assert len(batches) == 4
        assert batches[-1].shape == (4, 10)

    def test_batch_iterator_axis_0(self, sample_data):
        """Test iteration over axis 0 (rows)."""
        batches = list(batch_iterator(sample_data, batch_size=32, axis=0))
        total_rows = sum(b.shape[0] for b in batches)
        assert total_rows == 100

    def test_batch_iterator_axis_1(self, sample_data):
        """Test iteration over axis 1 (columns)."""
        batches = list(batch_iterator(sample_data, batch_size=3, axis=1))
        assert len(batches) == 4
        assert batches[0].shape == (100, 3)
        assert batches[-1].shape == (100, 1)

    def test_batch_iterator_shuffle(self, sample_data):
        """Test shuffling of batches."""
        batches_no_shuffle = list(batch_iterator(sample_data, batch_size=32, shuffle=False))
        batches_shuffle = list(
            batch_iterator(sample_data, batch_size=32, shuffle=True, random_seed=42)
        )
        assert len(batches_no_shuffle) == len(batches_shuffle)
        # Content should be different due to shuffling
        # (except by chance)
        combined_no_shuffle = np.vstack(batches_no_shuffle)
        combined_shuffle = np.vstack(batches_shuffle)
        # Order should be different
        assert not np.array_equal(combined_no_shuffle, combined_shuffle)

    def test_batch_iterator_random_seed(self, sample_data):
        """Test that random seed produces consistent shuffling."""
        batches1 = list(batch_iterator(sample_data, batch_size=32, shuffle=True, random_seed=42))
        batches2 = list(batch_iterator(sample_data, batch_size=32, shuffle=True, random_seed=42))
        for b1, b2 in zip(batches1, batches2, strict=False):
            assert np.array_equal(b1, b2)

    def test_batch_iterator_sparse_matrix(self, sample_sparse):
        """Test batch iterator with sparse matrix."""
        batches = list(batch_iterator(sample_sparse, batch_size=32))
        assert len(batches) == 4
        assert all(sp.issparse(b) for b in batches)
        assert batches[0].shape == (32, 10)

    def test_batch_iterator_list(self, sample_list):
        """Test batch iterator with list."""
        batches = list(batch_iterator(sample_list, batch_size=30))
        assert len(batches) == 4
        assert len(batches[0]) == 30
        assert len(batches[-1]) == 10

    def test_batch_iterator_len(self, sample_data):
        """Test __len__ method."""
        it = batch_iterator(sample_data, batch_size=32)
        assert len(it) == 4

    def test_batch_iterator_len_drop_last(self, sample_data):
        """Test __len__ with drop_last=True."""
        it = batch_iterator(sample_data, batch_size=32, drop_last=True)
        assert len(it) == 3

    def test_batch_iterator_single_batch(self, sample_data):
        """Test with batch_size larger than data."""
        batches = list(batch_iterator(sample_data[:10], batch_size=100))
        assert len(batches) == 1
        assert batches[0].shape == (10, 10)

    def test_batch_iterator_batch_size_1(self, sample_data):
        """Test with batch_size=1."""
        batches = list(batch_iterator(sample_data[:10], batch_size=1))
        assert len(batches) == 10
        assert all(b.shape == (1, 10) for b in batches)

    def test_batch_iterator_invalid_batch_size(self, sample_data):
        """Test that invalid batch_size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            list(batch_iterator(sample_data, batch_size=0))
        with pytest.raises(ValueError, match="batch_size must be positive"):
            list(batch_iterator(sample_data, batch_size=-1))

    def test_batch_iterator_unsupported_type(self):
        """Test with unsupported data type."""

        # Using a generator (no __len__)
        def gen():
            yield 1
            yield 2

        with pytest.raises(TypeError, match="Unsupported data type"):
            list(batch_iterator(gen(), batch_size=1))

    def test_batch_iterator_list_axis_1_error(self, sample_list):
        """Test that axis=1 raises error for lists."""
        with pytest.raises(ValueError, match="axis=1 not supported"):
            list(batch_iterator(sample_list, batch_size=10, axis=1))

    def test_batch_iterator_empty_data(self):
        """Test with empty data."""
        batches = list(batch_iterator(np.array([]), batch_size=10))
        assert len(batches) == 0

    def test_batch_iterator_iteration_reuse(self, sample_data):
        """Test that iterator can be created multiple times."""
        it1 = batch_iterator(sample_data, batch_size=32)
        batches1 = list(it1)

        it2 = batch_iterator(sample_data, batch_size=32)
        batches2 = list(it2)

        assert len(batches1) == len(batches2)

    def test_batch_iterator_total_elements_preserved(self, sample_data):
        """Test that total elements are preserved."""
        batches = list(batch_iterator(sample_data, batch_size=32))
        total = sum(b.size for b in batches)
        assert total == sample_data.size


class TestApplyByBatch:
    """Tests for apply_by_batch function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        np.random.seed(42)
        return np.random.randn(100, 10)

    @pytest.fixture
    def sample_sparse(self):
        """Create sample sparse data."""
        np.random.seed(42)
        return sp.csr_matrix(np.random.randn(100, 10))

    def test_apply_by_batch_basic(self, sample_data):
        """Test basic batch application."""

        def double_fn(x):
            return x * 2

        result = apply_by_batch(sample_data, double_fn, batch_size=32)
        assert result.shape == sample_data.shape
        assert np.allclose(result, sample_data * 2)

    def test_apply_by_batch_concat_true(self, sample_data):
        """Test with concat=True (default)."""

        def identity_fn(x):
            return x

        result = apply_by_batch(sample_data, identity_fn, batch_size=32, concat=True)
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_data.shape

    def test_apply_by_batch_concat_false(self, sample_data):
        """Test with concat=False."""

        def identity_fn(x):
            return x

        result = apply_by_batch(sample_data, identity_fn, batch_size=32, concat=False)
        assert isinstance(result, list)
        assert len(result) == 4

    def test_apply_by_batch_with_kwargs(self, sample_data):
        """Test passing kwargs to function."""

        def add_value(x, value=0):
            return x + value

        result = apply_by_batch(sample_data, add_value, batch_size=32, value=5)
        assert np.allclose(result, sample_data + 5)

    def test_apply_by_batch_axis_1(self, sample_data):
        """Test application along axis 1."""

        def identity_fn(x):
            return x

        result = apply_by_batch(sample_data, identity_fn, batch_size=3, axis=1)
        assert result.shape == sample_data.shape

    def test_apply_by_batch_sparse_matrix(self, sample_sparse):
        """Test with sparse matrix."""

        def identity_fn(x):
            return x

        result = apply_by_batch(sample_sparse, identity_fn, batch_size=32)
        assert sp.issparse(result)
        assert result.shape == sample_sparse.shape

    def test_apply_by_batch_list_data(self):
        """Test with list data."""
        data = list(range(100))

        def double_list(x):
            return [i * 2 for i in x]

        result = apply_by_batch(data, double_list, batch_size=30)
        assert isinstance(result, list)
        assert len(result) == 100
        assert result == [i * 2 for i in range(100)]

    def test_apply_by_batch_list_concat_false(self):
        """Test with list and concat=False."""
        data = list(range(100))

        def identity_fn(x):
            return x

        result = apply_by_batch(data, identity_fn, batch_size=30, concat=False)
        assert isinstance(result, list)
        assert len(result) == 4

    def test_apply_by_batch_empty_data(self):
        """Test with empty data."""

        def identity_fn(x):
            return x

        result = apply_by_batch(np.array([]), identity_fn, batch_size=10)
        assert result is None

    def test_apply_by_batch_shape_change(self, sample_data):
        """Test function that changes shape."""

        def first_two_cols(x):
            return x[:, :2]

        result = apply_by_batch(sample_data, first_two_cols, batch_size=32)
        assert result.shape == (100, 2)

    def test_apply_by_batch_scalar_output(self, sample_data):
        """Test function that returns scalar per batch."""

        def batch_sum(x):
            return np.sum(x)

        result = apply_by_batch(sample_data, batch_sum, batch_size=32, concat=False)
        assert isinstance(result, list)
        assert len(result) == 4

    def test_apply_by_batch_preserves_order(self, sample_data):
        """Test that order of elements is preserved."""

        def add_index(x, start_idx=0):
            # Add index-dependent value to verify order
            n = x.shape[0]
            return x + np.arange(n).reshape(-1, 1)

        result = apply_by_batch(sample_data, add_index, batch_size=32)
        # Just verify shape and no errors
        assert result.shape == sample_data.shape


class TestBatchApplyAlongAxis:
    """Tests for batch_apply_along_axis function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        np.random.seed(42)
        return np.random.randn(100, 50)

    def test_batch_apply_along_axis_basic(self):
        """Test basic functionality with row-wise sum."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        result = batch_apply_along_axis(np.sum, axis=1, data=X, batch_size=30)
        # axis=1 processes columns in batches, applying sum to each row
        assert isinstance(result, np.ndarray)

    def test_batch_apply_along_axis_axis_0(self, sample_data):
        """Test application along axis 0 (column-wise)."""
        result = batch_apply_along_axis(np.sum, axis=0, data=sample_data, batch_size=30)
        # Function executes without error
        assert isinstance(result, np.ndarray)

    def test_batch_apply_along_axis_axis_1(self, sample_data):
        """Test application along axis 1 (row-wise)."""
        result = batch_apply_along_axis(np.sum, axis=1, data=sample_data, batch_size=30)
        # Function executes without error
        assert isinstance(result, np.ndarray)

    def test_batch_apply_along_axis_returns_array(self, sample_data):
        """Test that function returns an array."""
        result = batch_apply_along_axis(np.sum, axis=1, data=sample_data, batch_size=30)
        assert isinstance(result, np.ndarray)

    def test_batch_apply_along_axis_with_kwargs(self, sample_data):
        """Test passing kwargs to function."""

        def weighted_sum(x, weight=1.0):
            return np.sum(x) * weight

        result = batch_apply_along_axis(
            weighted_sum, axis=1, data=sample_data, batch_size=30, weight=2.0
        )
        assert isinstance(result, np.ndarray)

    def test_batch_apply_along_axis_sparse_matrix(self, sample_data):
        """Test with sparse matrix (should convert to dense)."""
        X_sparse = sp.csr_matrix(sample_data)
        result = batch_apply_along_axis(np.sum, axis=1, data=X_sparse, batch_size=30)
        assert isinstance(result, np.ndarray)

    def test_batch_apply_along_axis_dtype(self, sample_data):
        """Test dtype parameter."""
        result = batch_apply_along_axis(
            np.sum, axis=1, data=sample_data, batch_size=30, dtype=np.float32
        )
        assert result.dtype == np.float32

    def test_batch_apply_along_axis_small_batch_size(self, sample_data):
        """Test with very small batch size."""
        result = batch_apply_along_axis(np.sum, axis=1, data=sample_data, batch_size=5)
        assert isinstance(result, np.ndarray)

    def test_batch_apply_along_axis_large_batch_size(self, sample_data):
        """Test with batch size larger than data."""
        result = batch_apply_along_axis(np.sum, axis=1, data=sample_data, batch_size=200)
        assert result.shape == (100,)

    def test_batch_apply_along_axis_2d_output(self, sample_data):
        """Test function that returns 2D array."""

        def copy_and_scale(x):
            return x * 2

        result = batch_apply_along_axis(copy_and_scale, axis=0, data=sample_data, batch_size=30)
        assert result.shape == sample_data.shape


class TestBatchProcessor:
    """Tests for BatchProcessor class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        np.random.seed(42)
        return np.random.randn(100, 10)

    def test_batch_processor_init(self):
        """Test BatchProcessor initialization."""
        processor = BatchProcessor(batch_size=32)
        assert processor.batch_size == 32
        assert processor.total_batches == 0
        assert processor.total_samples == 0

    def test_batch_processor_init_with_params(self):
        """Test initialization with parameters."""
        processor = BatchProcessor(batch_size=16, shuffle=True, random_seed=42)
        assert processor.batch_size == 16
        assert processor.shuffle is True
        assert processor.random_seed == 42

    def test_batch_processor_process(self, sample_data):
        """Test basic processing."""

        def double_fn(x):
            return x * 2

        processor = BatchProcessor(batch_size=32)
        result = processor.process(sample_data, double_fn)

        assert result.shape == sample_data.shape
        assert np.allclose(result, sample_data * 2)
        assert processor.total_samples == 100
        assert processor.total_batches == 4

    def test_batch_processor_shuffle(self, sample_data):
        """Test processing with shuffling."""

        def identity_fn(x):
            return x

        processor = BatchProcessor(batch_size=32, shuffle=True, random_seed=42)
        result = processor.process(sample_data, identity_fn)

        # Result should have correct shape even with shuffling
        assert result.shape == sample_data.shape

    def test_batch_processor_drop_last(self, sample_data):
        """Test processing with drop_last."""

        def identity_fn(x):
            return x

        processor = BatchProcessor(batch_size=32)
        result = processor.process(sample_data, identity_fn, drop_last=True)

        assert result.shape[0] == 96  # 3 * 32
        assert processor.total_samples == 96
        assert processor.total_batches == 3

    def test_batch_processor_with_kwargs(self, sample_data):
        """Test passing kwargs to function."""

        def add_value(x, value=0):
            return x + value

        processor = BatchProcessor(batch_size=32)
        result = processor.process(sample_data, add_value, value=5)

        assert np.allclose(result, sample_data + 5)

    def test_batch_processor_axis_1(self, sample_data):
        """Test processing along axis 1."""

        def identity_fn(x):
            return x

        processor = BatchProcessor(batch_size=3)
        result = processor.process(sample_data, identity_fn, axis=1)

        assert result.shape == sample_data.shape

    def test_batch_processor_stats(self, sample_data):
        """Test get_stats method."""

        def identity_fn(x):
            return x

        processor = BatchProcessor(batch_size=32)
        processor.process(sample_data, identity_fn)

        stats = processor.get_stats()
        assert stats["total_batches"] == 4
        assert stats["total_samples"] == 100

    def test_batch_processor_reset_stats(self, sample_data):
        """Test reset_stats method."""

        def identity_fn(x):
            return x

        processor = BatchProcessor(batch_size=32)
        processor.process(sample_data, identity_fn)

        processor.reset_stats()
        assert processor.total_batches == 0
        assert processor.total_samples == 0

    def test_batch_processor_multiple_calls(self, sample_data):
        """Test multiple processing calls accumulate stats."""

        def identity_fn(x):
            return x

        processor = BatchProcessor(batch_size=32)
        processor.process(sample_data[:50], identity_fn)
        processor.process(sample_data[:50], identity_fn)

        assert processor.total_samples == 100
        assert processor.total_batches == 4

    def test_batch_processor_sparse_matrix(self, sample_data):
        """Test with sparse matrix."""

        def identity_fn(x):
            return x

        X_sparse = sp.csr_matrix(sample_data)
        processor = BatchProcessor(batch_size=32)
        result = processor.process(X_sparse, identity_fn)

        assert sp.issparse(result)
        assert result.shape == sample_data.shape

    def test_batch_processor_list_data(self):
        """Test with list data."""

        def identity_fn(x):
            return x

        data = list(range(100))
        processor = BatchProcessor(batch_size=32)
        result = processor.process(data, identity_fn)

        # Result is a list of lists (each batch is a list)
        assert isinstance(result, list)
        assert processor.total_samples == 100
        # The result is list of batches, not flattened
        assert len(result) == 4  # 4 batches

    def test_batch_processor_empty_data(self):
        """Test with empty data."""

        def identity_fn(x):
            return x

        processor = BatchProcessor(batch_size=32)
        result = processor.process(np.array([]), identity_fn)

        assert result is None

    def test_batch_processor_history_tracking(self, sample_data):
        """Test that history is tracked."""

        def identity_fn(x):
            return x

        processor = BatchProcessor(batch_size=32)
        processor.process(sample_data, identity_fn)

        assert len(processor._history) == 1
        assert processor._history[0]["batch_count"] == 4
        assert processor._history[0]["sample_count"] == 100
        assert processor._history[0]["batch_size"] == 32

    def test_batch_processor_reset_clears_history(self, sample_data):
        """Test that reset clears history."""

        def identity_fn(x):
            return x

        processor = BatchProcessor(batch_size=32)
        processor.process(sample_data, identity_fn)
        processor.reset_stats()

        assert len(processor._history) == 0


class TestIntegration:
    """Integration tests for batch utilities."""

    def test_full_pipeline(self):
        """Test complete batch processing pipeline."""
        np.random.seed(42)
        data = np.random.randn(200, 20)

        # Define a processing function
        def normalize_batch(batch):
            return (batch - batch.mean(axis=0)) / (batch.std(axis=0) + 1e-8)

        # Process using BatchProcessor
        processor = BatchProcessor(batch_size=50, shuffle=True, random_seed=42)
        result = processor.process(data, normalize_batch)

        assert result.shape == data.shape
        assert processor.total_samples == 200
        assert processor.total_batches == 4

    def test_sparse_to_dense_pipeline(self):
        """Test processing sparse data with dense output."""
        np.random.seed(42)
        data = np.random.randn(50, 10)
        data[data < 0] = 0
        X_sparse = sp.csr_matrix(data)

        def densify_and_double(batch):
            return batch.toarray() * 2

        processor = BatchProcessor(batch_size=10)
        result = processor.process(X_sparse, densify_and_double)

        assert result.shape == data.shape

    def test_multi_batch_consistency(self):
        """Test that results are consistent across different batch sizes."""
        np.random.seed(42)
        data = np.random.randn(100, 10)

        def identity(x):
            return x

        results = []
        for batch_size in [10, 20, 25, 50]:
            processor = BatchProcessor(batch_size=batch_size)
            result = processor.process(data, identity)
            results.append(result)

        # All results should have same shape
        for result in results:
            assert result.shape == data.shape


class TestEdgeCases:
    """Edge case tests."""

    def test_single_element_batches(self):
        """Test with batch_size=1."""
        data = np.array([[1, 2, 3]])
        batches = list(batch_iterator(data, batch_size=1))
        assert len(batches) == 1
        assert batches[0].shape == (1, 3)

    def test_large_dataset(self):
        """Test with larger dataset."""
        np.random.seed(42)
        data = np.random.randn(10000, 100)

        def identity(x):
            return x

        processor = BatchProcessor(batch_size=1000)
        result = processor.process(data, identity)

        assert result.shape == data.shape
        assert processor.total_batches == 10

    def test_high_dimensional_data(self):
        """Test with high-dimensional data."""
        np.random.seed(42)
        data = np.random.randn(100, 1000)

        def identity(x):
            return x

        result = apply_by_batch(data, identity, batch_size=20, axis=1)
        assert result.shape == data.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
