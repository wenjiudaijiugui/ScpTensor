"""Statistical utilities for DIA-based single-cell proteomics data analysis.

This module provides functions for computing various correlation and similarity
metrics with support for both dense and sparse matrices.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr

if TYPE_CHECKING:
    from numpy.typing import NDArray


# Constants
_REGULARIZATION_EPS = 1e-8


def _ensure_dense(X: NDArray[np.float64] | sp.spmatrix) -> NDArray[np.float64]:
    """Convert sparse matrix to dense if necessary.

    Parameters
    ----------
    X : Union[NDArray[np.float64], sp.spmatrix]
        Input matrix.

    Returns
    -------
    NDArray[np.float64]
        Dense array.
    """
    return X.toarray() if sp.issparse(X) else X  # type: ignore[union-attr]


def correlation_matrix(
    X: NDArray[np.float64] | sp.spmatrix,
    method: str = "pearson",
) -> NDArray[np.float64]:
    """Compute correlation matrix between features (columns).

    This function calculates pairwise correlations between columns of the input
    matrix. Supports Pearson and Spearman correlation methods.

    Parameters
    ----------
    X : Union[NDArray[np.float64], sp.spmatrix]
        Input matrix of shape (n_samples, n_features).
    method : str, default="pearson"
        Correlation method: "pearson" or "spearman".

    Returns
    -------
    NDArray[np.float64]
        Correlation matrix of shape (n_features, n_features).
        Diagonal elements are 1.0.

    Raises
    ------
    ValueError
        If method is not "pearson" or "spearman".

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
    >>> corr = correlation_matrix(X)
    >>> corr.shape
    (3, 3)
    """
    if method not in ("pearson", "spearman"):
        raise ValueError(f"Unsupported method: {method}. Use 'pearson' or 'spearman'.")

    X_dense = _ensure_dense(X)
    X_centered = X_dense - X_dense.mean(axis=0, keepdims=True)

    if method == "pearson":
        # Pearson: cov(X,Y) / (std(X) * std(Y))
        norm = np.sqrt((X_centered**2).sum(axis=0))
        denom = np.outer(norm, norm)
        numer = X_centered.T @ X_centered
        corr = np.divide(
            numer,
            denom,
            out=np.zeros_like(denom, dtype=np.float64),
            where=denom > 0,
        )
    else:
        # Spearman: rank-based correlation
        from scipy.stats import rankdata

        X_ranks = np.apply_along_axis(rankdata, 0, X_dense)
        X_ranks_centered = X_ranks - X_ranks.mean(axis=0, keepdims=True)
        norm = np.sqrt((X_ranks_centered**2).sum(axis=0))
        denom = np.outer(norm, norm)
        numer = X_ranks_centered.T @ X_ranks_centered
        corr = np.divide(
            numer,
            denom,
            out=np.zeros_like(denom, dtype=np.float64),
            where=denom > 0,
        )

    # Handle numerical precision
    np.clip(corr, -1.0, 1.0, out=corr)
    np.fill_diagonal(corr, 1.0)

    return corr.astype(np.float64)


def partial_correlation(
    X: NDArray[np.float64] | sp.spmatrix,
    i: int,
    j: int,
    conditioning_set: set[int] | None = None,
) -> float:
    """Compute partial correlation coefficient between two variables.

    Partial correlation measures the correlation between variables i and j
    while controlling for the effect of variables in the conditioning set.

    Parameters
    ----------
    X : Union[NDArray[np.float64], sp.spmatrix]
        Input matrix of shape (n_samples, n_features).
    i : int
        Index of first variable (column).
    j : int
        Index of second variable (column).
    conditioning_set : Set[int], optional
        Set of variable indices to control for. If None, returns simple
        correlation between i and j.

    Returns
    -------
    float
        Partial correlation coefficient between variable i and j.

    Raises
    ------
    ValueError
        If indices are out of bounds or conditioning set is invalid.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randn(100, 5)
    >>> pc = partial_correlation(X, 0, 1, conditioning_set={2, 3})
    >>> -1.0 <= pc <= 1.0
    True
    """
    X_dense = _ensure_dense(X)
    n_samples, n_features = X_dense.shape

    # Validate indices
    if not (0 <= i < n_features) or not (0 <= j < n_features):
        raise ValueError(f"Indices i={i}, j={j} out of bounds for {n_features} features.")

    # Simple correlation if no conditioning set
    if not conditioning_set:
        corr_matrix = correlation_matrix(X_dense[:, [i, j]], method="pearson")
        return float(corr_matrix[0, 1])

    # Validate conditioning set
    invalid_indices = conditioning_set - set(range(n_features))
    if invalid_indices:
        raise ValueError(f"Invalid indices in conditioning set: {invalid_indices}")
    if i in conditioning_set or j in conditioning_set:
        raise ValueError("Variables i and j cannot be in the conditioning set.")

    # Collect all relevant variables
    all_indices = [i, j] + sorted(conditioning_set)
    X_subset = X_dense[:, all_indices]

    # Compute precision matrix (inverse of covariance)
    cov = np.cov(X_subset.T)

    try:
        precision = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        # Regularize singular matrix
        precision = np.linalg.inv(cov + _REGULARIZATION_EPS * np.eye(cov.shape[0]))

    # Partial correlation: -P_ij / sqrt(P_ii * P_jj)
    p_ij, p_ii, p_jj = precision[0, 1], precision[0, 0], precision[1, 1]
    partial_corr = -p_ij / np.sqrt(p_ii * p_jj)

    return float(np.clip(partial_corr, -1.0, 1.0))


def spearman_correlation(
    X: NDArray[np.float64] | sp.spmatrix,
    Y: NDArray[np.float64] | sp.spmatrix | None = None,
) -> NDArray[np.float64] | float:
    """Compute Spearman rank correlation coefficient.

    Spearman correlation assesses monotonic relationships using ranked values.
    This is a non-parametric measure robust to outliers.

    Parameters
    ----------
    X : Union[NDArray[np.float64], sp.spmatrix]
        First input matrix of shape (n_samples,) or (n_samples, n_features).
    Y : Union[NDArray[np.float64], sp.spmatrix], optional
        Second input matrix. If None, computes correlation between columns of X.
        Shape must match X.

    Returns
    -------
    Union[NDArray[np.float64], float]
        If Y is None: correlation matrix of shape (n_features, n_features).
        If Y is provided: correlation coefficient between X and Y.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 4], [3, 6]])
    >>> corr = spearman_correlation(X)
    >>> corr.shape
    (2, 2)
    """
    X_dense = _ensure_dense(X)

    if Y is None:
        # Correlation between columns of X
        if X_dense.ndim == 1:
            X_dense = X_dense.reshape(-1, 1)
        corr, _ = spearmanr(X_dense, axis=0)
        return corr.astype(np.float64)

    # Correlation between X and Y
    Y_dense = _ensure_dense(Y)
    x_flat, y_flat = X_dense.ravel(), Y_dense.ravel()

    if x_flat.shape != y_flat.shape:
        raise ValueError(f"Shape mismatch: X {x_flat.shape} vs Y {y_flat.shape}")

    corr, _ = spearmanr(x_flat, y_flat)
    return float(corr)


def cosine_similarity(
    X: NDArray[np.float64] | sp.spmatrix,
    Y: NDArray[np.float64] | sp.spmatrix | None = None,
) -> NDArray[np.float64] | float:
    """Compute cosine similarity between vectors.

    Cosine similarity measures the cosine of the angle between vectors,
    ranging from -1 (opposite) to 1 (identical direction).

    Parameters
    ----------
    X : Union[NDArray[np.float64], sp.spmatrix]
        First input matrix of shape (n_samples, n_features).
    Y : Union[NDArray[np.float64], sp.spmatrix], optional
        Second input matrix. If None, computes pairwise similarity between
        rows of X. Shape must match X if provided.

    Returns
    -------
    Union[NDArray[np.float64], float]
        If Y is None: similarity matrix of shape (n_samples, n_samples).
        If Y is provided: similarity coefficient between X and Y.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 0, 0], [0, 1, 0]])
    >>> sim = cosine_similarity(X)
    >>> sim.shape
    (2, 2)
    """

    def _compute_norm(matrix: NDArray[np.float64] | sp.spmatrix) -> NDArray[np.float64]:
        """Compute L2 norm for each row."""
        if sp.issparse(matrix):
            return sp.linalg.norm(matrix, axis=1).reshape(-1, 1)
        if matrix.ndim == 1:
            return np.array([[np.sqrt(np.sum(matrix**2))]])
        return np.sqrt(np.sum(matrix**2, axis=1, keepdims=True))

    X_norm = _compute_norm(X)

    if Y is None:
        # Pairwise similarity between rows of X
        sim = X @ X.T if not sp.issparse(X) else X @ X.T
        norms = X_norm @ X_norm.T

        if sp.issparse(norms):
            norms.data[norms.data == 0] = 1.0  # type: ignore[call-overload]
            sim = sim / norms
            result = sim.toarray()  # type: ignore[union-attr]
        else:
            norms[norms == 0] = 1.0
            result = sim / norms

        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)

    # Similarity between X and Y
    Y_norm = _compute_norm(Y)
    sim = X @ Y.T if not (sp.issparse(X) or sp.issparse(Y)) else X @ Y.T
    norms = X_norm @ Y_norm.T

    if sp.issparse(norms):
        norms.data[norms.data == 0] = 1.0  # type: ignore[call-overload]
        sim = sim / norms
        result = sim.toarray()  # type: ignore[union-attr]
    else:
        norms[norms == 0] = 1.0
        result = sim / norms

    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
