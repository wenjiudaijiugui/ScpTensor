"""Core utility functions for ScpTensor."""

from collections.abc import Callable, Collection
from difflib import get_close_matches
from functools import wraps
from typing import Any

import numpy as np

from scptensor.core.exceptions import MissingDependencyError


def _find_closest_match(input_name: str, options: Collection[str]) -> str | None:
    """Find the closest matching option for a given input.

    Uses difflib to suggest corrections when a user provides
    an incorrect assay/layer name.

    Parameters
    ----------
    input_name : str
        The user-provided name.
    options : Collection[str]
        Available valid options.

    Returns
    -------
    str | None
        The closest match if found (similarity > 0.6), None otherwise.

    Examples
    --------
    >>> _find_closest_match("ra", ["raw", "log", "normalized"])
    'raw'
    >>> _find_closest_match("prote", ["proteins", "peptides"])
    'proteins'
    >>> _find_closest_match("xyz", ["raw", "log"]) is None
    True
    """
    if not options:
        return None

    matches = get_close_matches(input_name, options, n=1, cutoff=0.6)
    return matches[0] if matches else None


def compute_pca(
    X: np.ndarray,
    n_components: int = 2,
    center: bool = True,
    random_state: int | None = None,
) -> np.ndarray:
    """Compute Principal Component Analysis on input matrix.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    n_components : int, default=2
        Number of principal components to compute.
    center : bool, default=True
        Whether to center the data before PCA.
    random_state : int, optional
        Random seed for reproducibility (not used in SVD).

    Returns
    -------
    np.ndarray
        PCA scores of shape (n_samples, n_components).
    """
    # Handle NaN: vectorized column median imputation
    if np.any(np.isnan(X)):
        X = X.copy()
        col_medians = np.nanmedian(X, axis=0, keepdims=True)
        nan_mask = np.isnan(X)
        X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

    # Center
    if center:
        X = X - np.mean(X, axis=0, keepdims=True)

    # SVD
    U, S, _ = np.linalg.svd(X, full_matrices=False)

    # Return scores
    return U[:, :n_components] * S[:n_components]


def compute_umap(
    X: np.ndarray,
    n_components: int = 2,
    random_state: int | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """Compute UMAP embedding on input matrix.

    Falls back to random projection if umap-learn is not available.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    n_components : int, default=2
        Number of embedding dimensions.
    random_state : int, optional
        Random seed for reproducibility.
    **kwargs : Any
        Additional parameters passed to UMAP.

    Returns
    -------
    np.ndarray
        Embedding of shape (n_samples, n_components).
    """
    # Handle NaN: vectorized column median imputation
    if np.any(np.isnan(X)):
        X = X.copy()
        col_medians = np.nanmedian(X, axis=0, keepdims=True)
        nan_mask = np.isnan(X)
        X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

    try:
        from umap import UMAP

        return UMAP(n_components=n_components, random_state=random_state, **kwargs).fit_transform(X)
    except ImportError:
        # Fallback: normalized random projection
        if random_state is not None:
            rng = np.random.default_rng(random_state)
        else:
            rng = np.random

        projection = rng.standard_normal((X.shape[1], n_components))
        projection = projection / np.linalg.norm(projection, axis=0, keepdims=True)

        embedding = X @ projection
        return (embedding - np.mean(embedding, axis=0)) / (np.std(embedding, axis=0) + 1e-8)


def requires_dependency(package_name: str, install_hint: str) -> Callable:
    """Decorator to ensure a dependency is installed before executing a function.

    Args:
        package_name: Name of the package to import.
        install_hint: Installation instruction.

    Returns:
        Decorator function that checks for dependency before execution.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                __import__(package_name)
            except ImportError:
                raise MissingDependencyError(package_name)
            return func(*args, **kwargs)

        return wrapper

    return decorator


__all__ = [
    "_find_closest_match",
    "compute_pca",
    "compute_umap",
    "requires_dependency",
]
