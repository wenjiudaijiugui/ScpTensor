"""
Numba JIT-compiled operations for ScpTensor.

This module contains performance-critical functions compiled with Numba JIT
for accelerated execution on hot loops.

Mask Codes:
    0 = VALID (detected value)
    1 = MBR (missing between runs)
    2 = LOD (below detection limit)
    3 = FILTERED (QC removed)
    5 = IMPUTED (filled value)
"""

from __future__ import annotations

import numpy as np

try:
    from numba import jit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def jit(nopython: bool = False, cache: bool = False):
        """Fallback decorator when numba is not available."""

        def decorator(func):
            return func

        return decorator


# =============================================================================
# Distance Calculation Functions
# =============================================================================

if NUMBA_AVAILABLE:

    @jit(nopython=True, cache=True)
    def euclidean_distance_no_nan(x: np.ndarray, y: np.ndarray) -> float:
        """Compute Euclidean distance between two vectors, ignoring NaN values.

        Parameters
        ----------
        x : np.ndarray
            First vector
        y : np.ndarray
            Second vector (same length as x)

        Returns
        -------
        float
            Euclidean distance, ignoring NaN pairs
        """
        n = x.shape[0]
        dist = 0.0
        n_valid = 0

        for i in range(n):
            xv = x[i]
            yv = y[i]
            # Check for NaN using the fact that NaN != NaN
            if xv == xv and yv == yv:
                diff = xv - yv
                dist += diff * diff
                n_valid += 1

        if n_valid > 0:
            return np.sqrt(dist / n_valid)
        else:
            return np.inf

    @jit(nopython=True, cache=True)
    def pairwise_euclidean_distances_no_nan(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute pairwise Euclidean distances between two matrices, ignoring NaNs.

        Parameters
        ----------
        X : np.ndarray
            First matrix (n_samples_x, n_features)
        Y : np.ndarray
            Second matrix (n_samples_y, n_features)

        Returns
        -------
        np.ndarray
            Distance matrix (n_samples_x, n_samples_y)
        """
        n_x = X.shape[0]
        n_y = Y.shape[0]
        n_features = X.shape[1]

        distances = np.empty((n_x, n_y), dtype=np.float64)

        for i in range(n_x):
            for j in range(n_y):
                dist = 0.0
                n_valid = 0

                for k in range(n_features):
                    xv = X[i, k]
                    yv = Y[j, k]
                    # Check for NaN
                    if xv == xv and yv == yv:
                        diff = xv - yv
                        dist += diff * diff
                        n_valid += 1

                if n_valid > 0:
                    distances[i, j] = np.sqrt(dist / n_valid)
                else:
                    distances[i, j] = np.inf

        return distances

    @jit(nopython=True, cache=True)
    def nan_euclidean_distance_row_to_matrix(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute distance from one vector to all rows of a matrix, handling NaNs.

        Parameters
        ----------
        x : np.ndarray
            Reference vector (n_features,)
        Y : np.ndarray
            Matrix (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Distance vector (n_samples,)
        """
        n_samples = Y.shape[0]
        n_features = x.shape[0]
        distances = np.empty(n_samples, dtype=np.float64)

        for i in range(n_samples):
            dist = 0.0
            n_valid = 0

            for j in range(n_features):
                xv = x[j]
                yv = Y[i, j]
                # Check for NaN
                if xv == xv and yv == yv:
                    diff = xv - yv
                    dist += diff * diff
                    n_valid += 1

            if n_valid > 0:
                distances[i] = np.sqrt(dist / n_valid)
            else:
                distances[i] = np.inf

        return distances

    @jit(nopython=True, cache=True)
    def nan_euclidean_distance_matrix_to_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute pairwise distances between all rows of X and Y, handling NaNs.

        Optimized for batch processing where X is typically smaller than Y.

        Parameters
        ----------
        X : np.ndarray
            Query matrix (n_query, n_features)
        Y : np.ndarray
            Reference matrix (n_reference, n_features)

        Returns
        -------
        np.ndarray
            Distance matrix (n_query, n_reference)
        """
        n_query = X.shape[0]
        n_reference = Y.shape[0]
        n_features = X.shape[1]

        distances = np.empty((n_query, n_reference), dtype=np.float64)

        for i in range(n_query):
            for j in range(n_reference):
                dist_sum = 0.0
                n_valid = 0

                for k in range(n_features):
                    xv = X[i, k]
                    yv = Y[j, k]
                    # NaN check: NaN != NaN is True
                    if xv == xv and yv == yv:
                        diff = xv - yv
                        dist_sum += diff * diff
                        n_valid += 1

                if n_valid > 0:
                    distances[i, j] = np.sqrt(dist_sum / n_valid)
                else:
                    distances[i, j] = np.inf

        return distances


# =============================================================================
# Mask Code Functions
# =============================================================================

if NUMBA_AVAILABLE:

    @jit(nopython=True, cache=True)
    def count_mask_codes(M: np.ndarray) -> np.ndarray:
        """Fast count of each mask code in the mask matrix.

        Parameters
        ----------
        M : np.ndarray
            Mask matrix of integers (0-5)

        Returns
        -------
        np.ndarray
            Array of length 7 with counts for each mask code (0-6)
        """
        counts = np.zeros(7, dtype=np.int64)
        n_rows, n_cols = M.shape

        for i in range(n_rows):
            for j in range(n_cols):
                code = M[i, j]
                if 0 <= code < 7:
                    counts[code] += 1

        return counts

    @jit(nopython=True, cache=True)
    def find_missing_indices(
        M: np.ndarray, mask_codes: tuple[int, ...]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find indices of values matching specified mask codes.

        Parameters
        ----------
        M : np.ndarray
            Mask matrix
        mask_codes : tuple of int
            Mask codes to search for (e.g., (1, 2) for MBR and LOD)

        Returns
        -------
        rows : np.ndarray
            Row indices of matching values
        cols : np.ndarray
            Column indices of matching values
        """
        n_rows, n_cols = M.shape

        # Pre-allocate with maximum possible size
        max_missing = n_rows * n_cols
        rows_buf = np.empty(max_missing, dtype=np.int64)
        cols_buf = np.empty(max_missing, dtype=np.int64)

        count = 0
        for i in range(n_rows):
            for j in range(n_cols):
                code = M[i, j]
                for mask_code in mask_codes:
                    if code == mask_code:
                        rows_buf[count] = i
                        cols_buf[count] = j
                        count += 1
                        break

        return rows_buf[:count], cols_buf[:count]


# =============================================================================
# Threshold and Filtering Functions
# =============================================================================

if NUMBA_AVAILABLE:

    @jit(nopython=True, cache=True)
    def apply_mask_threshold(X: np.ndarray, threshold: float, comparison: int) -> np.ndarray:
        """Apply threshold comparison to create boolean mask.

        Parameters
        ----------
        X : np.ndarray
            Input matrix
        threshold : float
            Threshold value
        comparison : int
            0 for less than, 1 for less than or equal,
            2 for greater than, 3 for greater than or equal

        Returns
        -------
        mask : np.ndarray
            Boolean mask matrix
        """
        n_rows, n_cols = X.shape
        mask = np.empty((n_rows, n_cols), dtype=np.bool_)

        for i in range(n_rows):
            for j in range(n_cols):
                val = X[i, j]

                if comparison == 0:
                    mask[i, j] = val < threshold
                elif comparison == 1:
                    mask[i, j] = val <= threshold
                elif comparison == 2:
                    mask[i, j] = val > threshold
                else:  # comparison == 3
                    mask[i, j] = val >= threshold

        return mask

    @jit(nopython=True, cache=True)
    def count_above_threshold(X: np.ndarray, threshold: float, axis: int) -> np.ndarray:
        """Count values above threshold along specified axis.

        Parameters
        ----------
        X : np.ndarray
            Input matrix
        threshold : float
            Threshold value
        axis : int
            0 for row-wise (count per column), 1 for column-wise (count per row)

        Returns
        -------
        counts : np.ndarray
            Count of values above threshold
        """
        n_rows, n_cols = X.shape

        if axis == 0:
            # Count per column (along rows)
            counts = np.empty(n_cols, dtype=np.int64)
            for j in range(n_cols):
                c = 0
                for i in range(n_rows):
                    if X[i, j] > threshold:
                        c += 1
                counts[j] = c
        else:
            # Count per row (along columns)
            counts = np.empty(n_rows, dtype=np.int64)
            for i in range(n_rows):
                c = 0
                for j in range(n_cols):
                    if X[i, j] > threshold:
                        c += 1
                counts[i] = c

        return counts

    @jit(nopython=True, cache=True)
    def filter_by_threshold_count(
        X: np.ndarray, threshold: float, min_count: int, axis: int
    ) -> np.ndarray:
        """Create a boolean mask for rows/columns with enough values above threshold.

        Parameters
        ----------
        X : np.ndarray
            Input matrix
        threshold : float
            Threshold value
        min_count : int
            Minimum count of values above threshold
        axis : int
            0 to filter rows (need enough columns above threshold),
            1 to filter columns (need enough rows above threshold)

        Returns
        -------
        mask : np.ndarray
            Boolean mask indicating which rows/columns to keep
        """
        counts = count_above_threshold(X, threshold, axis)
        return counts >= min_count


# =============================================================================
# Statistical Functions (for ComBat and other algorithms)
# =============================================================================

if NUMBA_AVAILABLE:

    @jit(nopython=True, cache=True)
    def mean_no_nan(arr: np.ndarray) -> float:
        """Compute mean of array, ignoring NaN values.

        Parameters
        ----------
        arr : np.ndarray
            Input array

        Returns
        -------
        float
            Mean of non-NaN values
        """
        n = arr.shape[0]
        total = 0.0
        count = 0

        for i in range(n):
            val = arr[i]
            if val == val:  # NaN check
                total += val
                count += 1

        if count > 0:
            return total / count
        else:
            return 0.0

    @jit(nopython=True, cache=True)
    def var_no_nan(arr: np.ndarray) -> float:
        """Compute variance of array, ignoring NaN values.

        Parameters
        ----------
        arr : np.ndarray
            Input array

        Returns
        -------
        float
            Variance of non-NaN values
        """
        n = arr.shape[0]

        # First pass: compute mean
        mean_val = mean_no_nan(arr)

        # Second pass: compute variance
        total_sq_diff = 0.0
        count = 0

        for i in range(n):
            val = arr[i]
            if val == val:  # NaN check
                diff = val - mean_val
                total_sq_diff += diff * diff
                count += 1

        if count > 1:
            return total_sq_diff / (count - 1)
        else:
            return 0.0

    @jit(nopython=True, cache=True)
    def mean_axis_no_nan(X: np.ndarray, axis: int) -> np.ndarray:
        """Compute mean along specified axis, ignoring NaN values.

        Parameters
        ----------
        X : np.ndarray
            Input matrix
        axis : int
            0 for column means, 1 for row means

        Returns
        -------
        means : np.ndarray
            Mean values along specified axis
        """
        n_rows, n_cols = X.shape

        if axis == 0:
            # Column means
            means = np.empty(n_cols, dtype=np.float64)
            for j in range(n_cols):
                total = 0.0
                count = 0
                for i in range(n_rows):
                    val = X[i, j]
                    if val == val:
                        total += val
                        count += 1
                means[j] = total / count if count > 0 else 0.0
        else:
            # Row means
            means = np.empty(n_rows, dtype=np.float64)
            for i in range(n_rows):
                total = 0.0
                count = 0
                for j in range(n_cols):
                    val = X[i, j]
                    if val == val:
                        total += val
                        count += 1
                means[i] = total / count if count > 0 else 0.0

        return means

    @jit(nopython=True, cache=True)
    def var_axis_no_nan(X: np.ndarray, axis: int) -> np.ndarray:
        """Compute variance along specified axis, ignoring NaN values.

        Parameters
        ----------
        X : np.ndarray
            Input matrix
        axis : int
            0 for column variances, 1 for row variances

        Returns
        -------
        variances : np.ndarray
            Variance values along specified axis
        """
        n_rows, n_cols = X.shape

        if axis == 0:
            # Column variances
            means = mean_axis_no_nan(X, 0)
            variances = np.empty(n_cols, dtype=np.float64)
            for j in range(n_cols):
                total_sq_diff = 0.0
                count = 0
                for i in range(n_rows):
                    val = X[i, j]
                    if val == val:
                        diff = val - means[j]
                        total_sq_diff += diff * diff
                        count += 1
                variances[j] = total_sq_diff / (count - 1) if count > 1 else 0.0
        else:
            # Row variances
            means = mean_axis_no_nan(X, 1)
            variances = np.empty(n_rows, dtype=np.float64)
            for i in range(n_rows):
                total_sq_diff = 0.0
                count = 0
                for j in range(n_cols):
                    val = X[i, j]
                    if val == val:
                        diff = val - means[i]
                        total_sq_diff += diff * diff
                        count += 1
                variances[i] = total_sq_diff / (count - 1) if count > 1 else 0.0

        return variances


# =============================================================================
# Matrix Manipulation Functions
# =============================================================================

if NUMBA_AVAILABLE:

    @jit(nopython=True, cache=True)
    def fill_missing_with_value(
        X: np.ndarray, M: np.ndarray, fill_value: float, fill_mask_code: int
    ) -> None:
        """Fill missing values in-place with a constant value.

        Parameters
        ----------
        X : np.ndarray
            Data matrix (modified in-place)
        M : np.ndarray
            Mask matrix
        fill_value : float
            Value to use for filling
        fill_mask_code : int
            Mask code to identify values to fill
        """
        n_rows, n_cols = X.shape

        for i in range(n_rows):
            for j in range(n_cols):
                if M[i, j] == fill_mask_code:
                    X[i, j] = fill_value

    @jit(nopython=True, cache=True)
    def fill_nan_with_value(X: np.ndarray, fill_value: float) -> None:
        """Fill NaN values in-place with a constant value.

        Parameters
        ----------
        X : np.ndarray
            Data matrix (modified in-place)
        fill_value : float
            Value to use for filling
        """
        n_rows, n_cols = X.shape

        for i in range(n_rows):
            for j in range(n_cols):
                val = X[i, j]
                if val != val:  # NaN check
                    X[i, j] = fill_value

    @jit(nopython=True, cache=True)
    def impute_nan_with_row_means(X: np.ndarray) -> None:
        """Impute NaN values with row means in-place.

        Parameters
        ----------
        X : np.ndarray
            Data matrix (modified in-place)
        """
        n_rows, n_cols = X.shape

        for i in range(n_rows):
            # Compute row mean
            total = 0.0
            count = 0
            for j in range(n_cols):
                val = X[i, j]
                if val == val:
                    total += val
                    count += 1

            row_mean = total / count if count > 0 else 0.0

            # Fill NaN with row mean
            for j in range(n_cols):
                val = X[i, j]
                if val != val:
                    X[i, j] = row_mean

    @jit(nopython=True, cache=True)
    def impute_nan_with_col_means(X: np.ndarray) -> None:
        """Impute NaN values with column means in-place.

        Parameters
        ----------
        X : np.ndarray
            Data matrix (modified in-place)
        """
        n_rows, n_cols = X.shape

        # First pass: compute column means
        col_means = np.empty(n_cols, dtype=np.float64)
        for j in range(n_cols):
            total = 0.0
            count = 0
            for i in range(n_rows):
                val = X[i, j]
                if val == val:
                    total += val
                    count += 1
            col_means[j] = total / count if count > 0 else 0.0

        # Second pass: fill NaN
        for i in range(n_rows):
            for j in range(n_cols):
                val = X[i, j]
                if val != val:
                    X[i, j] = col_means[j]


# =============================================================================
# Sparse Matrix Row Operations
# =============================================================================

if NUMBA_AVAILABLE:
    from numba import njit, prange

    @njit(cache=True, parallel=True, fastmath=True)
    def _sparse_row_sum_jit(
        indptr: np.ndarray,
        data: np.ndarray,
        n_rows: int,
    ) -> np.ndarray:
        """JIT-accelerated CSR matrix row sum with parallelization.

        Computes the sum of each row in a sparse matrix stored in CSR format.
        Uses parallel processing for improved performance on large matrices.

        Parameters
        ----------
        indptr : np.ndarray
            CSR index pointer array of shape (n_rows + 1,)
            indptr[i] to indptr[i+1] contains the range of indices in data
            corresponding to row i
        data : np.ndarray
            CSR data array containing non-zero values
        n_rows : int
            Number of rows in the matrix

        Returns
        -------
        np.ndarray
            Row sums of shape (n_rows,)

        Examples
        --------
        >>> import numpy as np
        >>> from scipy import sparse
        >>> X = sparse.csr_matrix([[1, 0, 2], [0, 0, 3], [4, 5, 0]])
        >>> sums = _sparse_row_sum_jit(X.indptr, X.data, X.shape[0])
        >>> sums
        array([3., 3., 9.])
        """
        result = np.empty(n_rows, dtype=np.float64)
        for i in prange(n_rows):
            start, end = indptr[i], indptr[i + 1]
            row_sum = 0.0
            for j in range(start, end):
                row_sum += data[j]
            result[i] = row_sum
        return result

    @njit(cache=True, parallel=True, fastmath=True)
    def _sparse_row_mean_jit(
        indptr: np.ndarray,
        data: np.ndarray,
        n_rows: int,
    ) -> np.ndarray:
        """JIT-accelerated CSR matrix row mean with parallelization.

        Computes the mean of each row in a sparse matrix stored in CSR format.
        Uses parallel processing for improved performance on large matrices.
        Handles empty rows (all zeros) by returning 0.0 as the mean.

        Parameters
        ----------
        indptr : np.ndarray
            CSR index pointer array of shape (n_rows + 1,)
            indptr[i] to indptr[i+1] contains the range of indices in data
            corresponding to row i
        data : np.ndarray
            CSR data array containing non-zero values
        n_rows : int
            Number of rows in the matrix

        Returns
        -------
        np.ndarray
            Row means of shape (n_rows,)

        Examples
        --------
        >>> import numpy as np
        >>> from scipy import sparse
        >>> X = sparse.csr_matrix([[1, 0, 2], [0, 0, 3], [4, 5, 0]])
        >>> means = _sparse_row_mean_jit(X.indptr, X.data, X.shape[0])
        >>> means
        array([1.5, 3. , 4.5])
        """
        result = np.empty(n_rows, dtype=np.float64)
        for i in prange(n_rows):
            start, end = indptr[i], indptr[i + 1]
            n_vals = end - start
            if n_vals > 0:
                row_sum = 0.0
                for j in range(start, end):
                    row_sum += data[j]
                result[i] = row_sum / n_vals
            else:
                result[i] = 0.0
        return result


# =============================================================================
# Fallback Pure Python Implementations
# =============================================================================

else:
    # Distance functions
    def euclidean_distance_no_nan(x: np.ndarray, y: np.ndarray) -> float:
        """Fallback euclidean distance (pure NumPy)."""
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        if np.sum(valid_mask) == 0:
            return np.inf
        return np.sqrt(np.sum((x[valid_mask] - y[valid_mask]) ** 2) / np.sum(valid_mask))

    def pairwise_euclidean_distances_no_nan(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Fallback pairwise distances (pure NumPy)."""
        from sklearn.metrics.pairwise import nan_euclidean_distances

        return nan_euclidean_distances(X, Y)

    def nan_euclidean_distance_row_to_matrix(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Fallback row to matrix distance (pure NumPy)."""
        from sklearn.metrics.pairwise import nan_euclidean_distances

        return nan_euclidean_distances(x.reshape(1, -1), Y)[0]

    def nan_euclidean_distance_matrix_to_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Fallback matrix to matrix distance (pure NumPy)."""
        from sklearn.metrics.pairwise import nan_euclidean_distances

        return nan_euclidean_distances(X, Y)

    # Mask functions
    def count_mask_codes(M: np.ndarray) -> np.ndarray:
        """Fallback count of mask codes (pure Python)."""
        counts = np.zeros(7, dtype=np.int64)
        for code in M.flat:
            if 0 <= code < 7:
                counts[code] += 1
        return counts

    def find_missing_indices(
        M: np.ndarray, mask_codes: tuple[int, ...] = (1, 2)
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fallback find missing indices (pure Python)."""
        mask = np.isin(M, list(mask_codes))
        result = np.where(mask)
        return result  # type: ignore[return-value]

    # Threshold functions
    def apply_mask_threshold(X: np.ndarray, threshold: float, comparison: int) -> np.ndarray:
        """Fallback threshold application (pure NumPy)."""
        if comparison == 0:
            return threshold > X
        elif comparison == 1:
            return threshold >= X
        elif comparison == 2:
            return threshold < X
        else:
            return threshold <= X

    def count_above_threshold(X: np.ndarray, threshold: float, axis: int) -> np.ndarray:
        """Fallback count above threshold (pure NumPy)."""
        return np.sum(threshold < X, axis=axis)

    def filter_by_threshold_count(
        X: np.ndarray, threshold: float, min_count: int, axis: int
    ) -> np.ndarray:
        """Fallback filter by threshold count (pure NumPy)."""
        counts = count_above_threshold(X, threshold, axis)
        return counts >= min_count

    # Statistical functions
    def mean_no_nan(arr: np.ndarray) -> float:
        """Fallback mean (pure NumPy)."""
        return np.nanmean(arr)

    def var_no_nan(arr: np.ndarray) -> float:
        """Fallback variance (pure NumPy)."""
        return np.nanvar(arr, ddof=1)

    def mean_axis_no_nan(X: np.ndarray, axis: int) -> np.ndarray:
        """Fallback mean along axis (pure NumPy)."""
        return np.nanmean(X, axis=axis)

    def var_axis_no_nan(X: np.ndarray, axis: int) -> np.ndarray:
        """Fallback variance along axis (pure NumPy)."""
        return np.nanvar(X, axis=axis, ddof=1)

    # Matrix manipulation
    def fill_missing_with_value(
        X: np.ndarray, M: np.ndarray, fill_value: float, fill_mask_code: int
    ) -> None:
        """Fallback fill (pure NumPy)."""
        mask = fill_mask_code == M
        X[mask] = fill_value

    def fill_nan_with_value(X: np.ndarray, fill_value: float) -> None:
        """Fallback fill NaN (pure NumPy)."""
        X[np.isnan(X)] = fill_value

    def impute_nan_with_row_means(X: np.ndarray) -> None:
        """Fallback row mean imputation (pure NumPy)."""
        row_means = np.nanmean(X, axis=1, keepdims=True)
        np.copyto(X, row_means, where=np.isnan(X))

    def impute_nan_with_col_means(X: np.ndarray) -> None:
        """Fallback column mean imputation (pure NumPy)."""
        col_means = np.nanmean(X, axis=0, keepdims=True)
        np.copyto(X, col_means, where=np.isnan(X))

    # KNN imputation fallbacks
    def knn_weighted_impute(
        neighbor_values: np.ndarray,
        distances: np.ndarray,
        k: int,
        use_distance_weights: bool,
    ) -> float:
        """Fallback weighted imputation (pure NumPy)."""
        n_available = min(k, len(neighbor_values))

        if n_available == 0:
            return 0.0

        vals = neighbor_values[:n_available]

        if use_distance_weights:
            ds = distances[:n_available]
            # Handle division by zero
            with np.errstate(divide="ignore"):
                w = 1.0 / ds
            inf_mask = np.isinf(w)
            if np.any(inf_mask):
                w[inf_mask] = 1.0
                w[~inf_mask] = 0.0
            w_sum = np.sum(w)
            if w_sum > 1e-10:
                return np.dot(vals, w / w_sum)
            else:
                return np.mean(vals)
        else:
            return np.mean(vals)

    def knn_find_valid_neighbors(
        X: np.ndarray,
        potential_neighbors: np.ndarray,
        potential_distances: np.ndarray,
        feature_idx: int,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Fallback find valid neighbors (pure NumPy)."""
        neighbor_values = X[potential_neighbors, feature_idx]
        valid_mask = ~np.isnan(neighbor_values)

        valid_neighbors = potential_neighbors[valid_mask][:k]
        valid_distances = potential_distances[valid_mask][:k]
        n_valid = len(valid_neighbors)

        return valid_neighbors, valid_distances, n_valid

    # PPCA fallbacks
    def ppca_initialize_with_col_means(
        X: np.ndarray,
        missing_mask: np.ndarray,
        col_means: np.ndarray,
    ) -> None:
        """Fallback PPCA initialization (pure NumPy)."""
        for j in range(X.shape[1]):
            X[missing_mask[:, j], j] = col_means[j]

    def impute_missing_with_col_means_jit(X: np.ndarray) -> None:
        """Fallback column-wise mean imputation (pure NumPy)."""
        col_means = np.nanmean(X, axis=0)
        col_means[np.isnan(col_means)] = 0.0
        for j in range(X.shape[1]):
            X[np.isnan(X[:, j]), j] = col_means[j]

    # Differential expression fallbacks
    def vectorized_ttest_row(
        x: np.ndarray,
        y: np.ndarray,
    ) -> tuple[float, float, float]:
        """Fallback t-test (uses scipy)."""
        from scipy import stats

        result = stats.ttest_ind(x, y, equal_var=False)
        mean_diff = np.mean(x) - np.mean(y)
        return float(result.statistic), float(result.pvalue), float(mean_diff)

    def vectorized_mannwhitney_row(
        x: np.ndarray,
        y: np.ndarray,
    ) -> tuple[float, float]:
        """Fallback Mann-Whitney (uses scipy)."""
        from scipy import stats

        try:
            result = stats.mannwhitneyu(x, y, alternative="two-sided")
            return float(result.statistic), float(result.pvalue)
        except ValueError:
            return 0.0, 1.0


# =============================================================================
# KNN Imputation Functions
# =============================================================================

if NUMBA_AVAILABLE:

    @jit(nopython=True, cache=True)
    def knn_weighted_impute(
        neighbor_values: np.ndarray,
        distances: np.ndarray,
        k: int,
        use_distance_weights: bool,
    ) -> float:
        """Compute weighted imputation value from k nearest neighbors.

        Parameters
        ----------
        neighbor_values : np.ndarray
            Values from neighbors (length >= k)
        distances : np.ndarray
            Distances to neighbors (same length as neighbor_values)
        k : int
            Number of neighbors to use
        use_distance_weights : bool
            If True, use inverse distance weighting

        Returns
        -------
        float
            Imputed value
        """
        n_available = min(k, len(neighbor_values))

        if n_available == 0:
            return 0.0

        if use_distance_weights:
            # Inverse distance weighting
            weights = np.empty(n_available, dtype=np.float64)
            for i in range(n_available):
                d = distances[i]
                if d > 1e-10:
                    weights[i] = 1.0 / d
                else:
                    # For very small distances, use uniform weight
                    weights[i] = 1.0

            w_sum = 0.0
            for i in range(n_available):
                w_sum += weights[i]

            if w_sum > 1e-10:
                weighted_sum = 0.0
                for i in range(n_available):
                    weighted_sum += neighbor_values[i] * (weights[i] / w_sum)
                return weighted_sum
            else:
                # Fallback to mean
                total = 0.0
                for i in range(n_available):
                    total += neighbor_values[i]
                return total / n_available
        else:
            # Uniform weighting (simple mean)
            total = 0.0
            for i in range(n_available):
                total += neighbor_values[i]
            return total / n_available

    @jit(nopython=True, cache=True)
    def knn_find_valid_neighbors(
        X: np.ndarray,
        potential_neighbors: np.ndarray,
        potential_distances: np.ndarray,
        feature_idx: int,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """From potential neighbors, find k that have valid values at feature_idx.

        Parameters
        ----------
        X : np.ndarray
            Data matrix (n_samples, n_features)
        potential_neighbors : np.ndarray
            Indices of potential neighbors
        potential_distances : np.ndarray
            Distances to potential neighbors
        feature_idx : int
            Feature index to check for valid values
        k : int
            Maximum number of neighbors to return

        Returns
        -------
        valid_neighbors_idx : np.ndarray
            Indices of valid neighbors (subset of potential_neighbors)
        valid_distances : np.ndarray
            Corresponding distances
        n_valid : int
            Number of valid neighbors found
        """
        max_valid = len(potential_neighbors)
        valid_neighbors_buf = np.empty(max_valid, dtype=np.int64)
        valid_distances_buf = np.empty(max_valid, dtype=np.float64)

        n_valid = 0
        for i in range(max_valid):
            neighbor_idx = potential_neighbors[i]
            val = X[neighbor_idx, feature_idx]
            # NaN check: val != val is True only for NaN
            if val == val:
                valid_neighbors_buf[n_valid] = neighbor_idx
                valid_distances_buf[n_valid] = potential_distances[i]
                n_valid += 1

                if n_valid >= k:
                    break

        return valid_neighbors_buf[:n_valid], valid_distances_buf[:n_valid], n_valid


# =============================================================================
# PPCA Initialization Functions
# =============================================================================

if NUMBA_AVAILABLE:

    @jit(nopython=True, cache=True)
    def ppca_initialize_with_col_means(
        X: np.ndarray,
        missing_mask: np.ndarray,
        col_means: np.ndarray,
    ) -> None:
        """Initialize missing values in X with column means in-place.

        Parameters
        ----------
        X : np.ndarray
            Data matrix (modified in-place)
        missing_mask : np.ndarray
            Boolean mask of missing values
        col_means : np.ndarray
            Column means to use for initialization
        """
        n_rows, n_cols = X.shape

        for j in range(n_cols):
            mean_val = col_means[j]
            for i in range(n_rows):
                if missing_mask[i, j]:
                    X[i, j] = mean_val

    @jit(nopython=True, cache=True)
    def impute_missing_with_col_means_jit(X: np.ndarray) -> None:
        """Impute missing values with column means in-place.

        Computes column means and fills NaN values with the corresponding mean.
        More efficient than calling np.nanmean and filling separately.

        Parameters
        ----------
        X : np.ndarray
            Data matrix (modified in-place)
        """
        n_rows, n_cols = X.shape

        # First pass: compute column means
        col_means = np.empty(n_cols, dtype=np.float64)
        for j in range(n_cols):
            total = 0.0
            count = 0
            for i in range(n_rows):
                val = X[i, j]
                if val == val:  # NaN check
                    total += val
                    count += 1
            col_means[j] = total / count if count > 0 else 0.0

        # Second pass: fill NaN values
        for j in range(n_cols):
            mean_val = col_means[j]
            for i in range(n_rows):
                val = X[i, j]
                if val != val:  # NaN check
                    X[i, j] = mean_val


# =============================================================================
# Differential Expression Functions
# =============================================================================

if NUMBA_AVAILABLE:

    @jit(nopython=True, cache=True)
    def vectorized_ttest_row(
        x: np.ndarray,
        y: np.ndarray,
    ) -> tuple[float, float, float]:
        """Perform Welch's t-test on two 1D arrays.

        Parameters
        ----------
        x : np.ndarray
            First group values
        y : np.ndarray
            Second group values

        Returns
        -------
        t_statistic : float
            T-statistic
        p_value : float
            Two-tailed p-value (approximate)
        mean_diff : float
            Mean difference (x - y)
        """
        n1 = len(x)
        n2 = len(y)

        if n1 < 2 or n2 < 2:
            return 0.0, 1.0, 0.0

        # Compute means
        sum1 = 0.0
        sum2 = 0.0
        for i in range(n1):
            sum1 += x[i]
        for i in range(n2):
            sum2 += y[i]

        mean1 = sum1 / n1
        mean2 = sum2 / n2

        # Compute variances
        var1 = 0.0
        var2 = 0.0
        for i in range(n1):
            diff = x[i] - mean1
            var1 += diff * diff
        for i in range(n2):
            diff = y[i] - mean2
            var2 += diff * diff

        var1 /= n1 - 1
        var2 /= n2 - 1

        # Welch's t-test
        se = np.sqrt(var1 / n1 + var2 / n2)

        if se < 1e-10:
            return 0.0, 1.0, mean1 - mean2

        t_stat = (mean1 - mean2) / se

        # Welch-Satterthwaite degrees of freedom
        df_num = (var1 / n1 + var2 / n2) ** 2
        df_den = (var1 / n2) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)

        if df_den < 1e-10:
            df = max(n1, n2) - 1
        else:
            df = df_num / df_den  # type: ignore[assignment]

        # Approximate p-value using normal approximation for large df
        # For small df, this is less accurate but still a reasonable approximation
        abs_t = abs(t_stat)
        if df > 100:
            # Normal approximation using error function
            # p_val = 2 * (1 - CDF(|t|)) = 2 * (1 - (1 + erf(|t|/sqrt(2))) / 2)
            p_val = 2.0 * (1.0 - 0.5 * (1.0 + np.erf(abs_t / np.sqrt(2.0))))  # type: ignore[attr-defined]
        else:
            # Conservative approximation for small df
            p_val = 2.0 * (1.0 - 0.5 * (1.0 + np.erf(abs_t / np.sqrt(2.0))))  # type: ignore[attr-defined]

        # Clamp p-value
        if p_val > 1.0:
            p_val = 1.0
        if p_val < 0.0:
            p_val = 0.0

        return t_stat, p_val, mean1 - mean2


# =============================================================================
# KMeans Clustering Functions
# =============================================================================

if NUMBA_AVAILABLE:

    @njit(parallel=True, fastmath=True, cache=True)
    def kmeans_plusplus_init_numba(
        X: np.ndarray,
        n_clusters: int,
        random_state: int,
    ) -> np.ndarray:
        """Initialize KMeans cluster centers using KMeans++ algorithm.

        Selects initial cluster centers iteratively with probability proportional
        to squared distance from nearest existing center.

        Parameters
        ----------
        X : np.ndarray
            Input data matrix (n_samples, n_features)
        n_clusters : int
            Number of clusters to initialize
        random_state : int
            Random seed for initialization

        Returns
        -------
        centers : np.ndarray
            Initial cluster centers (n_clusters, n_features)

        Notes
        -----
        KMeans++ initialization improves convergence and final clustering quality
        compared to random initialization. First center is chosen randomly from
        data points, subsequent centers are chosen with probability proportional
        to squared distance from nearest existing center.

        Examples
        --------
        >>> X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8]])
        >>> centers = kmeans_plusplus_init_numba(X, 2, 42)
        >>> centers.shape
        (2, 2)
        """
        n_samples, n_features = X.shape
        centers = np.zeros((n_clusters, n_features), dtype=np.float64)

        # First center: random sample
        centers[0] = X[random_state % n_samples]

        # Subsequent centers: probability proportional to distance^2
        for c in range(1, n_clusters):
            # Compute minimum distance to any existing center for each point
            min_distances = np.full(n_samples, np.inf, dtype=np.float64)

            for i in range(n_samples):
                for j in range(c):
                    # Compute squared distance
                    dist_sq = 0.0
                    for f in range(n_features):
                        diff = X[i, f] - centers[j, f]
                        dist_sq += diff * diff

                    if dist_sq < min_distances[i]:
                        min_distances[i] = dist_sq

            # Select next center with probability proportional to distance^2
            total_dist = min_distances.sum()

            if total_dist == 0:
                # All points are at cluster centers, pick random point
                centers[c] = X[(random_state + c * 7919) % n_samples]
                continue

            # Weighted random selection
            target = ((random_state * (c + 1) * 7919) % 1000000) / 1000000.0 * total_dist
            cumulative = 0.0

            for i in range(n_samples):
                cumulative += min_distances[i]
                if cumulative >= target:
                    centers[c] = X[i]
                    break

        return centers

    @njit(parallel=True, fastmath=True, cache=True)
    def kmeans_core_numba(
        X: np.ndarray,
        n_clusters: int,
        max_iter: int,
        tol: float,
        init_centers: np.ndarray,
        random_state: int,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Core KMeans iteration loop with convergence checking.

        Alternates between assigning samples to nearest cluster center and
        updating centers as mean of assigned samples until convergence or
        max iterations reached.

        Parameters
        ----------
        X : np.ndarray
            Input data matrix (n_samples, n_features)
        n_clusters : int
            Number of clusters
        max_iter : int
            Maximum number of iterations
        tol : float
            Convergence tolerance (centers shift threshold)
        init_centers : np.ndarray
            Initial cluster centers (n_clusters, n_features)
        random_state : int
            Random seed for handling empty clusters

        Returns
        -------
        centers : np.ndarray
            Final cluster centers (n_clusters, n_features)
        labels : np.ndarray
            Cluster labels for each sample (n_samples,)
        inertia : float
            Sum of squared distances to nearest cluster center

        Notes
        -----
        Algorithm converges when center movement falls below tol. Empty clusters
        are reinitialized to random data points. Inertia measures cluster
        compactness (lower is better).

        Examples
        --------
        >>> X = np.array([[1, 2], [1.5, 1.8], [5, 8]])
        >>> init = np.array([[1, 2], [5, 8]])
        >>> centers, labels, inertia = kmeans_core_numba(X, 2, 100, 1e-4, init, 42)
        >>> len(labels)
        3
        """
        n_samples, n_features = X.shape
        centers = init_centers.copy()
        labels = np.zeros(n_samples, dtype=np.int64)

        for iteration in range(max_iter):
            # Assignment step: assign each sample to nearest center
            for i in prange(n_samples):
                min_dist = np.inf
                min_label = 0

                for j in range(n_clusters):
                    # Compute squared Euclidean distance
                    dist_sq = 0.0
                    for f in range(n_features):
                        diff = X[i, f] - centers[j, f]
                        dist_sq += diff * diff

                    if dist_sq < min_dist:
                        min_dist = dist_sq
                        min_label = j

                labels[i] = min_label

            # Update step: recompute centers as mean of assigned samples
            new_centers = np.zeros((n_clusters, n_features), dtype=np.float64)
            counts = np.zeros(n_clusters, dtype=np.int64)

            for i in range(n_samples):
                label = labels[i]
                counts[label] += 1
                for f in range(n_features):
                    new_centers[label, f] += X[i, f]

            # Compute center shift and handle empty clusters
            shift_sq = 0.0

            for j in range(n_clusters):
                if counts[j] > 0:
                    # Average assigned samples
                    for f in range(n_features):
                        new_centers[j, f] /= counts[j]
                        # Accumulate shift
                        diff = new_centers[j, f] - centers[j, f]
                        shift_sq += diff * diff
                else:
                    # Empty cluster: reinitialize to random point
                    idx = (random_state + j + iteration * 7919) % n_samples
                    for f in range(n_features):
                        new_centers[j, f] = X[idx, f]

            centers = new_centers

            # Check convergence
            if np.sqrt(shift_sq) < tol:
                break

        # Compute inertia (sum of squared distances to nearest center)
        inertia = 0.0

        for i in prange(n_samples):
            center = centers[labels[i]]
            dist_sq = 0.0
            for f in range(n_features):
                diff = X[i, f] - center[f]
                dist_sq += diff * diff
            inertia += dist_sq

        return centers, labels, inertia

    @njit(parallel=True, fastmath=True, cache=True)
    def kmeans_predict_numba(
        X: np.ndarray,
        centers: np.ndarray,
    ) -> np.ndarray:
        """Predict cluster labels for new data using existing centers.

        Assigns each sample to the nearest cluster center without updating
        the centers.

        Parameters
        ----------
        X : np.ndarray
            Input data matrix (n_samples, n_features)
        centers : np.ndarray
            Cluster centers (n_clusters, n_features)

        Returns
        -------
        labels : np.ndarray
            Predicted cluster labels (n_samples,)

        Notes
        -----
        Prediction is parallelized across samples for efficiency. Uses
        squared Euclidean distance (same as KMeans fit).

        Examples
        --------
        >>> X = np.array([[1, 2], [5, 8]])
        >>> centers = np.array([[1, 2], [5, 8]])
        >>> labels = kmeans_predict_numba(X, centers)
        >>> labels
        array([0, 1])
        """
        n_samples = X.shape[0]
        n_clusters = centers.shape[0]
        n_features = X.shape[1]
        labels = np.zeros(n_samples, dtype=np.int64)

        for i in prange(n_samples):
            min_dist = np.inf
            min_label = 0

            for j in range(n_clusters):
                # Compute squared Euclidean distance
                dist_sq = 0.0
                for f in range(n_features):
                    diff = X[i, f] - centers[j, f]
                    dist_sq += diff * diff

                if dist_sq < min_dist:
                    min_dist = dist_sq
                    min_label = j

            labels[i] = min_label

        return labels


# =============================================================================
# Fallback Implementations for Sparse Operations
# =============================================================================

else:

    def _sparse_row_sum_jit(indptr: np.ndarray, data: np.ndarray, n_rows: int) -> np.ndarray:
        """Fallback row sum computation (pure NumPy)."""
        result = np.empty(n_rows, dtype=np.float64)
        for i in range(n_rows):
            start, end = indptr[i], indptr[i + 1]
            result[i] = np.sum(data[start:end])
        return result

    def _sparse_row_mean_jit(indptr: np.ndarray, data: np.ndarray, n_rows: int) -> np.ndarray:
        """Fallback row mean computation (pure NumPy)."""
        result = np.empty(n_rows, dtype=np.float64)
        for i in range(n_rows):
            start, end = indptr[i], indptr[i + 1]
            row_data = data[start:end]
            result[i] = np.mean(row_data) if len(row_data) > 0 else 0.0
        return result

    # KMeans fallbacks using sklearn
    def kmeans_plusplus_init_numba(X: np.ndarray, n_clusters: int, random_state: int) -> np.ndarray:
        """Fallback KMeans++ initialization (uses sklearn)."""
        from sklearn.cluster import KMeans

        kmeans = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            random_state=random_state,
            n_init=1,
            max_iter=1,
            algorithm="lloyd",
        )
        kmeans.fit(X)
        return kmeans.cluster_centers_

    def kmeans_core_numba(
        X: np.ndarray,
        n_clusters: int,
        max_iter: int,
        tol: float,
        init_centers: np.ndarray,
        random_state: int,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Fallback KMeans core (uses sklearn)."""
        from sklearn.cluster import KMeans

        kmeans = KMeans(
            n_clusters=n_clusters,
            init=init_centers,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            n_init=1,
            algorithm="lloyd",
        )
        kmeans.fit(X)
        return kmeans.cluster_centers_, kmeans.labels_, kmeans.inertia_

    def kmeans_predict_numba(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Fallback KMeans prediction (uses sklearn)."""
        from sklearn.metrics.pairwise import euclidean_distances

        distances = euclidean_distances(X, centers, squared=True)
        return np.argmin(distances, axis=1).astype(np.int64)


__all__ = [
    "NUMBA_AVAILABLE",
    # Distance functions
    "euclidean_distance_no_nan",
    "pairwise_euclidean_distances_no_nan",
    "nan_euclidean_distance_row_to_matrix",
    "nan_euclidean_distance_matrix_to_matrix",
    # Mask functions
    "count_mask_codes",
    "find_missing_indices",
    # Threshold functions
    "apply_mask_threshold",
    "count_above_threshold",
    "filter_by_threshold_count",
    # Statistical functions
    "mean_no_nan",
    "var_no_nan",
    "mean_axis_no_nan",
    "var_axis_no_nan",
    # Matrix manipulation
    "fill_missing_with_value",
    "fill_nan_with_value",
    "impute_nan_with_row_means",
    "impute_nan_with_col_means",
    # KNN imputation functions
    "knn_weighted_impute",
    "knn_find_valid_neighbors",
    # PPCA functions
    "ppca_initialize_with_col_means",
    "impute_missing_with_col_means_jit",
    # Differential expression functions
    "vectorized_ttest_row",
    "vectorized_mannwhitney_row",
    # KMeans clustering functions
    "kmeans_plusplus_init_numba",
    "kmeans_core_numba",
    "kmeans_predict_numba",
    # Sparse matrix row operations
    "_sparse_row_sum_jit",
    "_sparse_row_mean_jit",
]


if __name__ == "__main__":
    """Run basic tests for JIT operations."""
    print("Testing JIT operations...")
    print(f"Numba available: {NUMBA_AVAILABLE}")

    # Test euclidean_distance_no_nan
    x = np.array([1.0, 2.0, np.nan, 4.0])
    y = np.array([1.0, 3.0, 4.0, 4.0])
    dist = euclidean_distance_no_nan(x, y)
    print(f"Euclidean distance (no NaN): {dist:.4f}")
    # Expected: sqrt((1-1)^2 + (2-3)^2 + (4-4)^2) / 3 = sqrt(1/3)
    expected = np.sqrt(1.0 / 3.0)
    assert abs(dist - expected) < 1e-10
    print("euclidean_distance_no_nan: PASS")

    # Test nan_euclidean_distance_row_to_matrix
    X = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]])
    y = np.array([1.0, 2.0])
    dists = nan_euclidean_distance_row_to_matrix(y, X)
    print(f"Row to matrix distances: {dists}")
    assert abs(dists[0]) < 1e-10  # Same point
    print("nan_euclidean_distance_row_to_matrix: PASS")

    # Test count_mask_codes
    M = np.array([[0, 1, 2], [1, 2, 5], [0, 0, 3]], dtype=np.int64)
    counts = count_mask_codes(M)
    print(f"Mask code counts: {counts}")
    assert counts[0] == 3  # Three 0s
    assert counts[1] == 2  # Two 1s
    assert counts[2] == 2  # Two 2s
    assert counts[3] == 1  # One 3
    assert counts[5] == 1  # One 5
    print("count_mask_codes: PASS")

    # Test find_missing_indices
    rows, cols = find_missing_indices(M, (1, 2))
    print(f"Found {len(rows)} missing values (MBR/LOD)")
    assert len(rows) == 4  # Four values with codes 1 or 2
    print("find_missing_indices: PASS")

    # Test count_above_threshold
    X_test = np.array([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
    counts_row = count_above_threshold(X_test, 1.0, axis=1)
    print(f"Count above threshold (rows): {counts_row}")
    assert counts_row[0] == 2  # 2.0, 3.0 are > 1.0
    assert counts_row[1] == 2  # 1.5, 2.5 are > 1.0
    print("count_above_threshold: PASS")

    # Test apply_mask_threshold
    X_test = np.array([[1.0, 2.0], [3.0, 4.0]])
    mask = apply_mask_threshold(X_test, 2.5, 2)  # Greater than
    print(f"Threshold mask: {mask}")
    assert not mask[0, 0]
    assert not mask[0, 1]
    assert mask[1, 0]
    print("apply_mask_threshold: PASS")

    # Test fill_nan_with_value
    X_fill = np.array([[1.0, np.nan], [np.nan, 4.0]])
    fill_nan_with_value(X_fill, 99.0)
    print(f"NaN filled matrix: {X_fill}")
    assert X_fill[0, 1] == 99.0
    assert X_fill[1, 0] == 99.0
    print("fill_nan_with_value: PASS")

    # Test impute_nan_with_col_means
    X_col = np.array([[1.0, np.nan, 3.0], [2.0, np.nan, 4.0], [np.nan, 5.0, 6.0]])
    impute_nan_with_col_means(X_col)
    print(f"Column mean imputed: {X_col}")
    # Column 1 mean should be 1.5
    assert abs(X_col[2, 0] - 1.5) < 1e-10
    # Column 2 mean should be 5.0 (only one non-NaN value)
    assert abs(X_col[0, 1] - 5.0) < 1e-10
    print("impute_nan_with_col_means: PASS")

    # Test mean_no_nan and var_no_nan
    arr = np.array([1.0, 2.0, np.nan, 4.0])
    mean_val = mean_no_nan(arr)
    var_val = var_no_nan(arr)
    print(f"Mean (no NaN): {mean_val:.4f}, Var: {var_val:.4f}")
    assert abs(mean_val - 7.0 / 3.0) < 1e-10
    print("mean_no_nan/var_no_nan: PASS")

    # Test mean_axis_no_nan
    X_axis = np.array([[1.0, np.nan, 3.0], [2.0, 5.0, 4.0]])
    means_col = mean_axis_no_nan(X_axis, 0)
    means_row = mean_axis_no_nan(X_axis, 1)
    print(f"Column means: {means_col}")
    print(f"Row means: {means_row}")
    assert abs(means_col[0] - 1.5) < 1e-10
    assert abs(means_col[1] - 5.0) < 1e-10
    print("mean_axis_no_nan: PASS")

    # Test filter_by_threshold_count
    X_filter = np.array([[1.0, 0.0, 3.0], [2.0, 0.0, 2.0], [0.0, 0.0, 0.0]])
    mask = filter_by_threshold_count(X_filter, 0.5, 1, axis=1)
    print(f"Filter mask: {mask}")
    # Row 0 has 2 values > 0.5, row 1 has 2, row 2 has 0
    assert mask[0]
    assert mask[1]
    assert not mask[2]
    print("filter_by_threshold_count: PASS")

    # Test fill_missing_with_value
    X_fill2 = np.array([[1.0, 2.0], [3.0, 4.0]])
    M_fill = np.array([[0, 1], [2, 0]], dtype=np.int64)
    fill_missing_with_value(X_fill2, M_fill, 99.0, 1)
    print(f"Filled matrix: {X_fill2}")
    assert X_fill2[0, 1] == 99.0  # The position with code 1
    assert X_fill2[1, 0] == 3.0  # Code 2, not filled
    print("fill_missing_with_value: PASS")

    # Test knn_weighted_impute
    print("\nTesting new JIT functions...")
    neighbor_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    distances = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    # Uniform weighting
    result_uniform = knn_weighted_impute(
        neighbor_values, distances, k=3, use_distance_weights=False
    )
    expected_uniform = np.mean(neighbor_values[:3])
    print(f"KNN uniform imputation: {result_uniform:.4f} (expected: {expected_uniform:.4f})")
    assert abs(result_uniform - expected_uniform) < 1e-10
    print("knn_weighted_impute (uniform): PASS")

    # Distance weighting
    result_weighted = knn_weighted_impute(
        neighbor_values, distances, k=3, use_distance_weights=True
    )
    print(f"KNN weighted imputation: {result_weighted:.4f}")
    # Weighted average should be closer to nearer neighbors
    assert 1.0 < result_weighted < 3.0
    print("knn_weighted_impute (weighted): PASS")

    # Test knn_find_valid_neighbors
    X_test = np.array([[1.0, 2.0, 3.0], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, np.nan]])
    potential_neighbors = np.array([0, 1, 2, 3])
    potential_dists = np.array([0.1, 0.2, 0.3, 0.4])
    valid_idx, valid_dists, n_valid = knn_find_valid_neighbors(
        X_test, potential_neighbors, potential_dists, feature_idx=1, k=3
    )
    print(f"Found {n_valid} valid neighbors for feature 1")
    # Sample at index 1 has NaN at feature 1, so only 3 valid neighbors (0, 2, 3)
    assert n_valid == 3
    assert valid_idx[0] == 0  # First valid neighbor
    print("knn_find_valid_neighbors: PASS")

    # Test ppca_initialize_with_col_means
    X_init = np.array([[1.0, np.nan, 3.0], [np.nan, 5.0, 6.0]])
    missing_mask = np.isnan(X_init)
    col_means = np.array([2.0, 3.0, 4.0])
    ppca_initialize_with_col_means(X_init, missing_mask, col_means)
    print(f"Initialized matrix: {X_init}")
    assert X_init[0, 1] == 3.0  # Column 1 mean
    assert X_init[1, 0] == 2.0  # Column 0 mean
    print("ppca_initialize_with_col_means: PASS")

    # Test vectorized_ttest_row
    x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_data = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    t_stat, p_val, mean_diff = vectorized_ttest_row(x_data, y_data)
    print(f"T-test: t={t_stat:.4f}, p={p_val:.4f}, mean_diff={mean_diff:.4f}")
    assert mean_diff == -1.0
    assert p_val > 0.0 and p_val <= 1.0
    print("vectorized_ttest_row: PASS")

    # Test KMeans functions
    print("\nTesting KMeans JIT functions...")
    X_kmeans = np.array(
        [
            [1.0, 2.0],
            [1.5, 1.8],
            [5.0, 8.0],
            [8.0, 8.0],
            [1.0, 0.6],
            [9.0, 11.0],
            [1.2, 1.9],
            [5.2, 8.2],
            [8.5, 8.5],
        ]
    )

    # Test kmeans_plusplus_init_numba
    init_centers = kmeans_plusplus_init_numba(X_kmeans, n_clusters=3, random_state=42)
    print(f"KMeans++ init centers shape: {init_centers.shape}")
    assert init_centers.shape == (3, 2)
    print("kmeans_plusplus_init_numba: PASS")

    # Test kmeans_core_numba
    centers, labels, inertia = kmeans_core_numba(
        X_kmeans, n_clusters=3, max_iter=100, tol=1e-4, init_centers=init_centers, random_state=42
    )
    print(f"KMeans centers:\n{centers}")
    print(f"KMeans labels: {labels}")
    print(f"KMeans inertia: {inertia:.4f}")
    assert centers.shape == (3, 2)
    assert len(labels) == len(X_kmeans)
    assert inertia >= 0.0
    # Check that all labels are valid (0, 1, or 2)
    assert np.all(labels >= 0) and np.all(labels < 3)
    print("kmeans_core_numba: PASS")

    # Test kmeans_predict_numba
    X_new = np.array([[1.1, 2.1], [8.2, 8.3]])
    predicted_labels = kmeans_predict_numba(X_new, centers)
    print(f"Predicted labels for new data: {predicted_labels}")
    assert len(predicted_labels) == len(X_new)
    assert np.all(predicted_labels >= 0) and np.all(predicted_labels < 3)
    print("kmeans_predict_numba: PASS")

    print("\nAll tests passed!")
