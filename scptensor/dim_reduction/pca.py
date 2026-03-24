"""Principal Component Analysis (PCA) for dimensionality reduction.

This module implements PCA with automatic solver selection, following
scikit-learn and scanpy best practices.

Reference:
    Jolliffe, I.T., & Cadima, J. (2016). Principal component analysis:
    a review and recent developments. Philosophical Transactions
    of the Royal Society A, 374(2065), 20150202.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl
import scipy.sparse as sp

from scptensor.core._structure_assay import Assay
from scptensor.core._structure_container import ScpContainer
from scptensor.core._structure_matrix import ScpMatrix
from scptensor.core.assay_alias import resolve_assay_name
from scptensor.dim_reduction.base import (
    _check_no_nan_inf,
    _prepare_matrix,
    _validate_assay_layer,
)

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

# Solver types
SolverType = Literal["auto", "full", "arpack", "randomized", "covariance_eigh"]


def reduce_pca(
    container: ScpContainer,
    assay_name: str,
    base_layer: str,
    new_assay_name: str = "pca",
    n_components: int = 50,
    center: bool = True,
    scale: bool = False,
    svd_solver: SolverType = "auto",
    random_state: int | np.random.RandomState | None = 42,
    dtype: DTypeLike = np.float64,
    return_info: bool = False,
) -> ScpContainer:
    """Perform Principal Component Analysis (PCA).

    This function automatically selects the optimal SVD solver based on
    data characteristics, following scikit-learn's best practices.

    Parameters
    ----------
    container : ScpContainer
        The data container.
    assay_name : str
        Name of the assay to use.
    base_layer : str
        Name of the layer to analyze.
    new_assay_name : str, optional
        Name for the new PCA assay. Default is "pca".
    n_components : int, optional
        Number of principal components. Default is 50.
    center : bool, optional
        Whether to center the data. Default is True.
    scale : bool, optional
        Whether to scale to unit variance. Default is False.
    svd_solver : {"auto", "full", "arpack", "randomized", "covariance_eigh"}, optional
        SVD solver to use. Default is "auto" for automatic selection.
    random_state : int or RandomState or None, optional
        Random seed for reproducibility. Default is 42.
    dtype : dtype or type, optional
        Data type for computation. Default is np.float64.
    return_info : bool, optional
        Whether to return variance info in var. Default is False.

    Returns
    -------
    ScpContainer
        New container with PCA results added as a new assay.

    Raises
    ------
    AssayNotFoundError
        If the specified assay does not exist.
    LayerNotFoundError
        If the specified layer does not exist.
    ValueError
        If invalid parameters or data contains NaN/Inf.

    Examples
    --------
    >>> from scptensor import create_test_container
    >>> from scptensor.dim_reduction import reduce_pca
    >>> container = create_test_container()
    >>> result = reduce_pca(container, "proteins", "imputed", n_components=10)

    Notes
    -----
    The solver selection logic:
    - Sparse + center: arpack
    - Sparse + no center: randomized
    - Tall-skinny (n_features <= 1000): covariance_eigh
    - Small matrices: full
    - Large + few components: randomized
    - Otherwise: arpack

    """
    # Validation
    resolved_assay_name = resolve_assay_name(container, assay_name)
    assay, X = _validate_assay_layer(container, resolved_assay_name, base_layer)
    _check_no_nan_inf(X)

    n_samples, n_features = X.shape
    min_dim = min(n_samples, n_features)
    valid_solvers = {"auto", "full", "arpack", "randomized", "covariance_eigh"}

    if n_samples < 2:
        raise ValueError(f"PCA requires at least 2 samples, got {n_samples}")
    if n_components <= 0:
        raise ValueError(f"n_components must be positive, got {n_components}")
    if svd_solver not in valid_solvers:
        raise ValueError(
            f"Invalid svd_solver '{svd_solver}'. Must be one of {sorted(valid_solvers)}.",
        )

    if n_components > min_dim:
        raise ValueError(
            f"n_components ({n_components}) cannot exceed min(n_samples, n_features) ({min_dim}).",
        )
    if svd_solver == "arpack" and n_components >= min_dim:
        raise ValueError(
            "svd_solver='arpack' requires n_components < min(n_samples, n_features). "
            f"Got n_components={n_components}, min_dim={min_dim}.",
        )

    # Solver selection
    if svd_solver == "auto":
        svd_solver = _select_solver(X, n_components, center)

    # Prepare data
    X_dense = _prepare_matrix(X, dtype=np.dtype(dtype))

    # Center and scale
    if center:
        mean = X_dense.mean(axis=0)
        X_dense -= mean
    else:
        mean = None

    if scale:
        std = X_dense.std(axis=0, ddof=1)
        eps = np.finfo(np.float64).eps
        std = np.where(std < eps, 1.0, std)
        X_dense /= std
    else:
        std = None

    # Compute SVD
    scores, loadings, eigenvalues = _compute_svd(X_dense, n_components, svd_solver, random_state)

    # Compute explained variance
    total_variance = _compute_total_variance(X_dense, center)
    explained_variance = eigenvalues
    eps = np.finfo(np.float64).eps
    if total_variance <= eps:
        explained_variance_ratio = np.zeros_like(explained_variance)
    else:
        explained_variance_ratio = explained_variance / total_variance

    # Create output assay
    pc_names = [f"PC{i + 1}" for i in range(n_components)]

    var_data = {
        "pc_name": pc_names,
        "explained_variance": explained_variance,
        "explained_variance_ratio": explained_variance_ratio,
    }

    if return_info:
        var_data["cumulative_variance_ratio"] = np.cumsum(explained_variance_ratio)

    pca_var = pl.DataFrame(var_data)

    scores_M = np.zeros_like(scores, dtype=np.int8)
    scores_matrix = ScpMatrix(X=scores, M=scores_M)

    pca_assay = Assay(
        var=pca_var,
        layers={"X": scores_matrix},
        feature_id_col="pc_name",
    )

    # Store loadings in original assay.
    loading_cols = {
        f"{new_assay_name}_PC{i + 1}_loading": loadings[:, i] for i in range(n_components)
    }
    loadings_df = pl.DataFrame(loading_cols)

    # Clean old loadings
    prefix = f"{new_assay_name}_PC"
    existing_cols = assay.var.columns
    cols_to_drop = [c for c in existing_cols if c.startswith(prefix) and "_loading" in c]

    var_clean = assay.var.drop(cols_to_drop) if cols_to_drop else assay.var
    new_var = pl.concat([var_clean, loadings_df], how="horizontal")

    original_assay = Assay(
        var=new_var,
        layers=assay.layers.copy(),
        feature_id_col=assay.feature_id_col,
    )

    # Create new container
    new_assays = {
        **container.assays,
        resolved_assay_name: original_assay,
        new_assay_name: pca_assay,
    }

    new_container = ScpContainer(
        obs=container.obs,
        assays=new_assays,
        history=list(container.history),
        sample_id_col=container.sample_id_col,
    )

    # Log operation
    new_container.log_operation(
        action="reduce_pca",
        params={
            "source_assay": resolved_assay_name,
            "source_layer": base_layer,
            "target_assay": new_assay_name,
            "n_components": n_components,
            "center": center,
            "scale": scale,
            "svd_solver": svd_solver,
        },
        description=(
            f"PCA on {resolved_assay_name}/{base_layer} ({n_components} components, {svd_solver})."
        ),
    )

    return new_container


def _select_solver(
    X: np.ndarray | sp.spmatrix,
    n_components: int,
    center: bool,
) -> SolverType:
    """Select optimal SVD solver based on data characteristics.

    Parameters
    ----------
    X : ndarray or spmatrix
        Input matrix.
    n_components : int
        Number of components to compute.
    center : bool
        Whether data will be centered.

    Returns
    -------
    str
        Selected solver name.

    """
    n_samples, n_features = X.shape

    if sp.issparse(X):
        return "arpack" if center else "randomized"

    # Tall and skinny: use covariance
    if n_features <= 1000 and n_samples >= 10 * n_features:
        return "covariance_eigh"

    if n_components >= min(n_samples, n_features):
        return "full"

    # Large with few components
    min_dim = min(n_samples, n_features)
    if n_components < 0.8 * min_dim:
        return "randomized"

    return "arpack"


def _compute_svd(
    X: np.ndarray,
    n_components: int,
    solver: SolverType,
    random_state: int | np.random.RandomState | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute SVD using selected solver.

    Parameters
    ----------
    X : ndarray
        Input matrix (already centered/scaled).
    n_components : int
        Number of components.
    solver : str
        Solver name.
    random_state : int or RandomState or None
        Random seed.

    Returns
    -------
    tuple of ndarray
        (scores, loadings, eigenvalues)

    """
    n_samples, n_features = X.shape
    min_dim = min(n_samples, n_features)

    if solver == "full":
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        U = U[:, :n_components]
        S = S[:n_components]
        Vt = Vt[:n_components, :]

    elif solver == "covariance_eigh":
        # Compute covariance matrix
        C = (X.T @ X) / (n_samples - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]

        eigenvalues = np.maximum(eigenvalues, 0.0)
        S = np.sqrt(eigenvalues[:n_components] * (n_samples - 1))
        basis = eigenvectors[:, :n_components]
        Vt = basis.T
        projected = X @ basis
        U = np.zeros_like(projected)
        nonzero = np.finfo(np.float64).eps < S
        np.divide(projected, S, out=U, where=nonzero)

    elif solver == "randomized":
        from sklearn.utils.extmath import randomized_svd

        U, S, Vt = randomized_svd(
            X,
            n_components=n_components,
            random_state=random_state,
        )

    else:  # arpack
        from scipy.sparse.linalg import svds

        k = n_components
        rng = np.random.default_rng(random_state)
        v0 = rng.standard_normal(min_dim)
        U, S, Vt = svds(X, k=k, which="LM", v0=v0)
        U = U[:, ::-1]
        S = S[::-1]
        Vt = Vt[::-1, :]

    # Enforce sign convention
    for k in range(Vt.shape[0]):
        idx = np.argmax(np.abs(Vt[k, :]))
        if Vt[k, idx] < 0:
            Vt[k, :] *= -1
            U[:, k] *= -1

    # Scores = U * S
    scores = U * S
    loadings = Vt.T
    eigenvalues = S**2 / (n_samples - 1)

    return scores, loadings, eigenvalues


def _compute_total_variance(X: np.ndarray, center: bool) -> float:
    """Compute total variance of data.

    Parameters
    ----------
    X : ndarray
        Input data.
    center : bool
        Whether data was centered.

    Returns
    -------
    float
        Total variance.

    """
    if center:
        return np.sum(np.var(X, axis=0, ddof=1))
    return np.sum(np.square(X)) / (X.shape[0] - 1)


__all__ = ["reduce_pca", "SolverType"]
