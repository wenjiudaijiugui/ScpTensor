"""Batch processing utilities for handling large datasets.

This module provides tools for iterating over and applying functions to
large datasets in batches, which helps manage memory usage for big data.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from numpy.typing import NDArray


class batch_iterator:
    """Iterator for processing data in batches.

    This iterator divides a dataset into smaller batches for memory-efficient
    processing. It supports both dense arrays, sparse matrices, and any
    sequence-like objects.

    Parameters
    ----------
    data : Union[NDArray, sp.spmatrix, Sequence]
        Input data to iterate over.
    batch_size : int, default=32
        Number of samples per batch.
    axis : int, default=0
        Axis along which to batch. For 2D arrays:
        - 0: Iterate over rows (samples)
        - 1: Iterate over columns (features)
    drop_last : bool, default=False
        Whether to drop the last batch if it's smaller than batch_size.
    shuffle : bool, default=False
        Whether to shuffle data before batching.
    random_seed : int, optional
        Random seed for shuffling.

    Yields
    ------
    Union[NDArray, sp.spmatrix, Sequence]
        Batch of data.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randn(100, 10)
    >>> for batch in batch_iterator(X, batch_size=32):
    ...     print(batch.shape)
    (32, 10)
    (32, 10)
    (32, 10)
    (4, 10)
    """

    __slots__ = ("data", "batch_size", "axis", "drop_last", "indices", "n_items")

    def __init__(
        self,
        data: NDArray[np.float64] | sp.spmatrix | Sequence,
        batch_size: int = 32,
        axis: int = 0,
        drop_last: bool = False,
        shuffle: bool = False,
        random_seed: int | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self.data = data
        self.batch_size = batch_size
        self.axis = axis
        self.drop_last = drop_last

        # Determine size and create indices
        if sp.issparse(data) or isinstance(data, np.ndarray):
            self.n_items = data.shape[axis]  # type: ignore[union-attr]
        elif hasattr(data, "__len__"):
            self.n_items = len(data)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        # Create shuffled indices if needed
        if shuffle:
            rng = np.random.default_rng(random_seed)
            self.indices = rng.permutation(self.n_items)
        else:
            self.indices = np.arange(self.n_items)

    def __iter__(self) -> Iterator[NDArray[np.float64] | sp.spmatrix | Any]:
        """Iterate over batches."""
        n_batches, remainder = divmod(self.n_items, self.batch_size)

        for i in range(n_batches):
            yield self._get_batch(self.indices[i * self.batch_size : (i + 1) * self.batch_size])

        if not self.drop_last and remainder:
            yield self._get_batch(self.indices[n_batches * self.batch_size :])

    def _get_batch(self, indices: NDArray[np.int64]) -> NDArray[np.float64] | sp.spmatrix | Any:
        """Extract a batch by indices."""
        if sp.issparse(self.data):
            return self.data[indices, :] if self.axis == 0 else self.data[:, indices]  # type: ignore[call-overload]
        if isinstance(self.data, np.ndarray):
            return self.data[indices, ...] if self.axis == 0 else self.data[..., indices]
        if hasattr(self.data, "__getitem__"):
            if self.axis == 0:
                return [self.data[i] for i in indices]
            raise ValueError("axis=1 not supported for generic sequences")
        raise TypeError(f"Cannot extract batch from type: {type(self.data)}")

    def __len__(self) -> int:
        """Return number of batches."""
        n_batches, remainder = divmod(self.n_items, self.batch_size)
        return n_batches + (0 if self.drop_last and remainder > 0 else bool(remainder))


def apply_by_batch(
    data: NDArray[np.float64] | sp.spmatrix | Sequence,
    func: Callable,
    batch_size: int = 32,
    axis: int = 0,
    concat: bool = True,
    **kwargs: Any,
) -> Any:
    """Apply a function to data in batches.

    This function processes data in batches, applying the provided function
    to each batch. Results can be concatenated or returned as a list.

    Parameters
    ----------
    data : Union[NDArray, sp.spmatrix, Sequence]
        Input data to process.
    func : Callable
        Function to apply to each batch. Should accept a single batch as
        its first argument.
    batch_size : int, default=32
        Number of samples per batch.
    axis : int, default=0
        Axis along which to batch.
    concat : bool, default=True
        Whether to concatenate results. If False, returns a list of results.
    **kwargs : Any
        Additional keyword arguments passed to func.

    Returns
    -------
    Any
        Concatenated result or list of batch results, depending on `concat`.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randn(100, 10)
    >>> def normalize_batch(batch):
    ...     return (batch - batch.mean(axis=0)) / batch.std(axis=0)
    >>> result = apply_by_batch(X, normalize_batch, batch_size=32)
    >>> result.shape
    (100, 10)
    """
    results = [func(batch, **kwargs) for batch in batch_iterator(data, batch_size, axis)]

    if not concat or not results:
        return results if not concat else None

    # Concatenate results based on type
    first = results[0]
    if isinstance(first, np.ndarray):
        return np.concatenate(results, axis=axis)
    if sp.issparse(first):
        return sp.vstack(results)
    if isinstance(first, list):
        return [item for r in results for item in r]
    return results


def batch_apply_along_axis(
    func1d: Callable,
    axis: int,
    data: NDArray[np.float64] | sp.spmatrix,
    batch_size: int = 1000,
    dtype: type[np.float64] | None = None,
    **kwargs: Any,
) -> NDArray[np.float64]:
    """Apply a function to 1-D slices along a given axis in batches.

    This is a batched version of numpy.apply_along_axis, useful for
    processing very large arrays that don't fit in memory.

    Parameters
    ----------
    func1d : Callable
        Function to apply to 1-D slices. Should accept a 1-D array.
    axis : int
        Axis along which to apply the function.
    data : Union[NDArray, sp.spmatrix]
        Input data.
    batch_size : int, default=1000
        Number of slices to process per batch.
    dtype : type, optional
        Output data type.
    **kwargs : Any
        Additional keyword arguments passed to func1d.

    Returns
    -------
    NDArray[np.float64]
        Result array with func1d applied along the specified axis.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randn(100, 50)
    >>> def row_sum(row):
    ...     return row.sum()
    >>> result = batch_apply_along_axis(row_sum, axis=1, data=X, batch_size=32)
    >>> result.shape
    (100,)
    """
    X_dense = data.toarray() if sp.issparse(data) else data  # type: ignore[union-attr]
    n_slices = X_dense.shape[axis]

    results = []
    for i in range(0, n_slices, batch_size):
        end_idx = min(i + batch_size, n_slices)
        batch = X_dense[i:end_idx, :] if axis == 0 else X_dense[:, i:end_idx]
        results.append(np.apply_along_axis(func1d, axis, batch, **kwargs))

    # Combine results based on output dimensionality
    if isinstance(results[0], np.ndarray):
        if results[0].ndim == X_dense.ndim - 1:
            combined = np.concatenate(results)
        elif axis == 0:
            combined = np.concatenate(results, axis=0)
        else:
            combined = np.concatenate(results, axis=1)
    else:
        combined = np.array(results)

    return combined.astype(dtype) if dtype else combined


class BatchProcessor:
    """A class for managing batch processing operations.

    This class provides a reusable interface for batch processing with
    state tracking and progress reporting.

    Parameters
    ----------
    batch_size : int, default=32
        Number of samples per batch.
    shuffle : bool, default=False
        Whether to shuffle data before processing.
    random_seed : int, optional
        Random seed for shuffling.

    Attributes
    ----------
    batch_size : int
        Current batch size.
    total_batches : int
        Total number of batches processed.
    total_samples : int
        Total number of samples processed.

    Examples
    --------
    >>> import numpy as np
    >>> processor = BatchProcessor(batch_size=32)
    >>> X = np.random.randn(100, 10)
    >>> def process_fn(batch):
    ...     return batch * 2
    >>> result = processor.process(X, process_fn)
    >>> processor.total_samples
    100
    """

    __slots__ = (
        "batch_size",
        "shuffle",
        "random_seed",
        "total_batches",
        "total_samples",
        "_history",
    )

    def __init__(
        self,
        batch_size: int = 32,
        shuffle: bool = False,
        random_seed: int | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.total_batches = 0
        self.total_samples = 0
        self._history: list[dict[str, Any]] = []

    def process(
        self,
        data: NDArray[np.float64] | sp.spmatrix | Sequence,
        func: Callable,
        axis: int = 0,
        drop_last: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Process data in batches using the provided function.

        Parameters
        ----------
        data : Union[NDArray, sp.spmatrix, Sequence]
            Input data to process.
        func : Callable
            Function to apply to each batch.
        axis : int, default=0
            Axis along which to batch.
        drop_last : bool, default=False
            Whether to drop the last incomplete batch.
        **kwargs : Any
            Additional keyword arguments passed to func.

        Returns
        -------
        Any
            Processed result (concatenated or as list).
        """
        results = []
        batch_count = sample_count = 0

        for batch in batch_iterator(
            data,
            self.batch_size,
            axis,
            drop_last,
            self.shuffle,
            self.random_seed,
        ):
            results.append(func(batch, **kwargs))

            # Track statistics
            n_items = (
                batch.shape[0]
                if sp.issparse(batch)
                else batch.shape[axis]
                if isinstance(batch, np.ndarray)
                else len(batch)
            )
            batch_count += 1
            sample_count += n_items

        # Update state
        self.total_batches += batch_count
        self.total_samples += sample_count
        self._history.append(
            {
                "batch_count": batch_count,
                "sample_count": sample_count,
                "batch_size": self.batch_size,
            }
        )

        if not results:
            return None
        if isinstance(results[0], np.ndarray):
            return np.concatenate(results, axis=axis)
        if sp.issparse(results[0]):
            return sp.vstack(results)
        return results

    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self.total_batches = 0
        self.total_samples = 0
        self._history.clear()

    def get_stats(self) -> dict[str, int]:
        """Get current processing statistics.

        Returns
        -------
        Dict[str, int]
            Dictionary with total_batches and total_samples.
        """
        return {"total_batches": self.total_batches, "total_samples": self.total_samples}


if __name__ == "__main__":
    """Run tests to verify functionality."""
    print("Running tests for batch module...")

    # Test data
    np.random.seed(42)
    X_dense = np.random.randn(100, 10)
    X_sparse = sp.csr_matrix(X_dense)
    X_list = list(range(100))

    # Test 1: batch_iterator with dense array
    print("\n1. Testing batch_iterator (dense array)...")
    batches = list(batch_iterator(X_dense, batch_size=32, axis=0))
    assert len(batches) == 4
    assert batches[0].shape == (32, 10)
    assert batches[-1].shape == (4, 10)
    print(f"   Number of batches: {len(batches)}")
    print(f"   First batch shape: {batches[0].shape}")
    print(f"   Last batch shape: {batches[-1].shape}")

    # Test 2: batch_iterator with drop_last
    print("\n2. Testing batch_iterator (drop_last=True)...")
    batches = list(batch_iterator(X_dense, batch_size=32, drop_last=True))
    assert len(batches) == 3
    print(f"   Number of batches: {len(batches)}")

    # Test 3: batch_iterator with sparse matrix
    print("\n3. Testing batch_iterator (sparse)...")
    batches = list(batch_iterator(X_sparse, batch_size=32))
    assert len(batches) == 4
    assert sp.issparse(batches[0])
    print(f"   All batches sparse: {all(sp.issparse(b) for b in batches)}")

    # Test 4: batch_iterator with shuffle
    print("\n4. Testing batch_iterator (shuffle)...")
    batches = list(batch_iterator(X_dense, batch_size=32, shuffle=True, random_seed=42))
    assert len(batches) == 4
    print(f"   Shuffled batches created: {len(batches)}")

    # Test 5: batch_iterator with list
    print("\n5. Testing batch_iterator (list)...")
    batches = list(batch_iterator(X_list, batch_size=30))
    assert len(batches) == 4
    assert len(batches[-1]) == 10
    print(f"   Number of batches: {len(batches)}")

    # Test 6: apply_by_batch
    print("\n6. Testing apply_by_batch...")
    result = apply_by_batch(X_dense, lambda batch: batch * 2, batch_size=32)
    assert result.shape == X_dense.shape
    assert np.allclose(result, X_dense * 2)
    print(f"   Result shape: {result.shape}")
    print(f"   Values correctly doubled: {np.allclose(result, X_dense * 2)}")

    # Test 7: apply_by_batch with concat=False
    print("\n7. Testing apply_by_batch (concat=False)...")
    results = apply_by_batch(X_dense, lambda batch: batch * 2, batch_size=32, concat=False)
    assert isinstance(results, list)
    assert len(results) == 4
    print(f"   Number of results: {len(results)}")

    # Test 8: batch_apply_along_axis
    print("\n8. Testing batch_apply_along_axis...")
    result = batch_apply_along_axis(np.sum, axis=1, data=X_dense, batch_size=30)
    assert result.shape == (100,)
    print(f"   Result shape: {result.shape}")
    print(f"   Correct sums: {np.allclose(result, X_dense.sum(axis=1))}")

    # Test 9: BatchProcessor
    print("\n9. Testing BatchProcessor...")
    processor = BatchProcessor(batch_size=32)
    result = processor.process(X_dense, lambda batch: batch * 2)
    assert processor.total_samples == 100
    assert processor.total_batches == 4
    stats = processor.get_stats()
    print(f"   Total samples: {stats['total_samples']}")
    print(f"   Total batches: {stats['total_batches']}")

    # Test 10: BatchProcessor reset
    print("\n10. Testing BatchProcessor.reset_stats()...")
    processor.reset_stats()
    assert processor.total_samples == 0
    assert processor.total_batches == 0
    print("   Stats reset successfully")

    # Test 11: Error handling
    print("\n11. Testing error handling...")
    try:
        list(batch_iterator(X_dense, batch_size=-1))
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        print(f"   Correct error raised: {e}")

    print("\n All tests passed for batch module!")
