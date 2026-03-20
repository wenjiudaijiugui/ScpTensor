"""
Bayesian PCA imputation.

Reference:
    Oba S, Sato MA, Takemasa I, et al. A Bayesian missing value estimation
    method for gene expression profile data. Bioinformatics (2003).

BPCA extends Probabilistic PCA (PPCA) by using Bayesian inference to
automatically determine the effective number of components.
"""

import numpy as np
import numpy.typing as npt
from scipy.sparse.linalg import svds

from scptensor.core.exceptions import (
    AssayNotFoundError,
    DimensionError,
    LayerNotFoundError,
    ScpValueError,
)
from scptensor.core.structures import ScpContainer
from scptensor.impute._utils import (
    add_imputed_layer,
    log_imputation_operation,
    to_dense_float_copy,
)
from scptensor.impute.base import ImputeMethod, register_impute_method

# =============================================================================
# Internal utility functions
# =============================================================================


def _is_missing(x: npt.NDArray, missing_val: float | None = np.nan) -> npt.NDArray:
    """Vectorized missing value detection."""
    if missing_val is None or np.isnan(missing_val):
        return np.isnan(x)
    return x == missing_val


def _not_missing(x: npt.NDArray, missing_val: float | None = np.nan) -> npt.NDArray:
    """Vectorized non-missing value detection."""
    if missing_val is None or np.isnan(missing_val):
        return ~np.isnan(x)
    return x != missing_val


# =============================================================================
# Core BPCA algorithm
# =============================================================================


def _bpca_init(y: npt.NDArray, q: int, missing_value: float | None = np.nan) -> dict:
    """Initialize BPCA model from data matrix."""
    N, d = y.shape
    miss_mask = _is_missing(y, missing_value)
    yest = y.copy()
    yest[miss_mask] = 0

    covy = np.cov(yest, rowvar=False)
    covy = np.nan_to_num(covy, nan=0.0)

    # Robust fallback for near-singular small matrices where ARPACK can fail.
    try:
        U, S, _ = svds(covy, k=q)
        idx = np.argsort(S)[::-1]
        U = U[:, idx]
        S = S[idx]
        W = U * np.sqrt(S)
    except Exception:
        U, S, _ = np.linalg.svd(covy, full_matrices=False)
        keep = min(q, U.shape[1])
        W = U[:, :keep] * np.sqrt(np.clip(S[:keep], 0.0, None))

    mu = np.zeros(d)
    for j in range(d):
        valid = _not_missing(y[:, j], missing_value)
        if valid.any():
            mu[j] = y[valid, j].mean()

    denom = float(np.sum(np.diag(covy)) - np.sum(S))
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        denom = 1e-12
    tau = max(min(1.0 / denom, 1e10), 1e-10)
    alpha = (2e-10 + d) / (tau * np.diag(W.T @ W) + 2e-10)

    return {
        "N": N,
        "q": q,
        "d": d,
        "W": W,
        "mu": mu,
        "tau": tau,
        "alpha": alpha,
        "SigW": np.eye(q),
        "yest": yest,
        "miss_mask": miss_mask,
    }


def _bpca_em_step(M: dict, y: npt.NDArray) -> dict:
    """Perform one EM step of Bayesian PCA."""
    q, N, d = M["q"], M["N"], M["d"]

    Rx = np.eye(q) + M["tau"] * (M["W"].T @ M["W"]) + M["SigW"]
    Rxinv = np.linalg.inv(Rx)

    gnomiss = ~M["miss_mask"].any(axis=1)
    gmiss = M["miss_mask"].any(axis=1)

    idx = gnomiss
    dy = y[idx] - M["mu"]
    Rxinv_Wt = Rxinv @ M["W"].T
    x = M["tau"] * (Rxinv_Wt @ dy.T)

    T = dy.T @ x.T
    trS = np.sum(dy * dy)

    miss_idx = np.where(gmiss)[0]
    for i in miss_idx:
        nomiss = np.where(~M["miss_mask"][i])[0]
        miss = np.where(M["miss_mask"][i])[0]

        if len(nomiss) == 0 or len(miss) == 0:
            continue

        dyo = y[i, nomiss] - M["mu"][nomiss]
        Wo = M["W"][nomiss, :]
        Wm = M["W"][miss, :]

        Rxinv_miss = np.linalg.inv(Rx - M["tau"] * (Wm.T @ Wm))
        x = Rxinv_miss @ (M["tau"] * Wo.T @ dyo)

        dy = y[i, :].copy()
        dy[nomiss] = dyo
        dy[miss] = Wm @ x

        M["yest"][i] = dy + M["mu"]
        T = T + np.outer(dy, x)

    T = T / N
    trS = trS / N

    Dw = Rxinv + M["tau"] * (T.T @ M["W"] @ Rxinv) + np.diag(M["alpha"]) / N
    M["W"] = T @ np.linalg.inv(Dw)
    if not np.isfinite(trS) or abs(trS) < 1e-12:
        trS = 1e-12
    M["tau"] = max(min(d / trS, 1e10), 1e-10)
    M["SigW"] = np.linalg.inv(Dw) * (d / N)
    M["alpha"] = (2 * 1e-10 + d) / (
        M["tau"] * np.diag(M["W"].T @ M["W"]) + np.diag(M["SigW"]) + 2 * 1e-10
    )

    return M


def bpca_impute(
    data: np.ndarray,
    n_components: int = 10,
    max_iter: int = 100,
    random_state: int | None = None,
) -> np.ndarray:
    """Bayesian PCA imputation (core algorithm).

    Parameters
    ----------
    data : np.ndarray
        Input data with missing values (NaN).
    n_components : int, default=10
        Number of principal components.
    max_iter : int, default=100
        Maximum EM iterations.
    random_state : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        Data with imputed values.
    """
    X = data.copy()
    N, d = X.shape

    if random_state is not None:
        np.random.seed(random_state)

    # Exclude samples where ALL values are missing
    all_missing = np.isnan(X).all(axis=1)
    valid = ~all_missing

    if not np.any(valid):
        return np.nan_to_num(X, nan=0.0)

    n_valid = int(np.sum(valid))
    if n_valid < 2:
        # Degenerate case: BPCA covariance initialization is not well-defined.
        # Fall back to simple column-mean fill from available rows.
        col_means = np.nanmean(X[valid], axis=0)
        col_means = np.nan_to_num(col_means, nan=0.0)
        y = X.copy().astype(np.float64)
        miss = np.isnan(y)
        if np.any(miss):
            y[miss] = np.take(col_means, np.where(miss)[1])
        return np.nan_to_num(y, nan=0.0)

    k = min(n_components, d - 1, n_valid - 1)
    if k < 1:
        k = 1

    M = _bpca_init(X[valid], k)

    tauold = 1000.0
    for epoch in range(max_iter):
        M = _bpca_em_step(M, X[valid])
        if (epoch + 1) % 10 == 0:
            dtau = abs(np.log10(M["tau"]) - np.log10(tauold))
            if dtau < 1e-4:
                break
            tauold = M["tau"]

    y = X.copy().astype(np.float64)
    y[valid] = M["yest"]
    return np.nan_to_num(y, nan=0.0)


def validate_bpca(data: np.ndarray) -> bool:
    """Validate data for BPCA imputation."""
    return data.size > 0 and data.shape[0] > 2 and data.shape[1] > 2


# =============================================================================
# ScpTensor wrapper function
# =============================================================================


def impute_bpca(
    container: ScpContainer,
    assay_name: str,
    source_layer: str,
    new_layer_name: str = "imputed_bpca",
    n_components: int | None = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    random_state: int | None = None,
) -> ScpContainer:
    """
    Impute missing values using Bayesian PCA (BPCA).

    Parameters
    ----------
    container : ScpContainer
        Input container with missing values.
    assay_name : str
        Name of the assay to use.
    source_layer : str
        Name of the layer containing data with missing values.
    new_layer_name : str, default "imputed_bpca"
        Name for the new layer with imputed data.
    n_components : int, optional
        Maximum number of principal components.
    max_iter : int, default=100
        Maximum EM iterations.
    tol : float, default=1e-6
        Convergence tolerance (kept for API compatibility).
    random_state : int, optional
        Random seed.

    Returns
    -------
    ScpContainer
        Container with imputed data in new layer.

    Raises
    ------
    ScpValueError
        If parameters are invalid.
    AssayNotFoundError
        If the assay does not exist.
    LayerNotFoundError
        If the layer does not exist.
    DimensionError
        If n_components is too large.

    Examples
    --------
    >>> from scptensor import impute_bpca
    >>> result = impute_bpca(container, "proteins", "raw", n_components=10)
    >>> "imputed_bpca" in result.assays["proteins"].layers
    True
    """
    # Validate parameters
    if max_iter <= 0:
        raise ScpValueError(
            f"max_iter must be positive, got {max_iter}.",
            parameter="max_iter",
            value=max_iter,
        )

    # Validate assay and layer
    if assay_name not in container.assays:
        available = ", ".join(f"'{k}'" for k in container.assays)
        raise AssayNotFoundError(
            assay_name,
            hint=f"Available assays: {available}.",
        )

    assay = container.assays[assay_name]
    if source_layer not in assay.layers:
        available = ", ".join(f"'{k}'" for k in assay.layers)
        raise LayerNotFoundError(
            source_layer,
            assay_name,
            hint=f"Available layers: {available}.",
        )

    input_matrix = assay.layers[source_layer]
    X_original = input_matrix.X.copy()
    n_samples, n_features = X_original.shape

    if n_components is None:
        n_components = min(n_samples, n_features) - 1

    if n_components <= 0:
        raise ScpValueError(
            f"n_components must be positive, got {n_components}.",
            parameter="n_components",
            value=n_components,
        )

    if tol <= 0:
        raise ScpValueError(
            f"tol must be positive, got {tol}.",
            parameter="tol",
            value=tol,
        )
    if n_components >= min(n_samples, n_features):
        raise DimensionError(
            f"n_components ({n_components}) must be less than "
            f"min(n_samples, n_features) = {min(n_samples, n_features)}",
            expected_shape=(n_samples, n_features),
            actual_shape=(n_samples, n_components),
        )

    X_dense = to_dense_float_copy(X_original)

    missing_mask = np.isnan(X_dense)
    layer_name = new_layer_name or "imputed_bpca"
    if not np.any(missing_mask):
        add_imputed_layer(assay, layer_name, X_dense, input_matrix, missing_mask)
        return log_imputation_operation(
            container,
            action="impute_bpca",
            params={
                "assay": assay_name,
                "source_layer": source_layer,
                "new_layer_name": layer_name,
                "n_components": n_components,
            },
            description=f"BPCA imputation on assay '{assay_name}': no missing values found.",
        )

    # Apply BPCA imputation
    X_imputed = bpca_impute(
        X_dense,
        n_components=n_components,
        max_iter=max_iter,
        random_state=random_state,
    )

    # Create new layer
    add_imputed_layer(assay, layer_name, X_imputed, input_matrix, missing_mask)

    # Log operation
    singular_values = np.linalg.svd(X_imputed, full_matrices=False, compute_uv=False)
    scale = singular_values[0] if singular_values.size > 0 else 0.0
    threshold = max(scale * 1e-6, 1e-12)
    effective_components = int(np.sum(singular_values > threshold))
    effective_components = min(effective_components, int(n_components))

    return log_imputation_operation(
        container,
        action="impute_bpca",
        params={
            "assay": assay_name,
            "source_layer": source_layer,
            "new_layer_name": layer_name,
            "n_components": n_components,
            "effective_components": effective_components,
        },
        description=f"BPCA imputation (n_components={n_components}) on assay '{assay_name}'.",
    )


# Register with base interface
register_impute_method(
    ImputeMethod(
        name="bpca",
        supports_sparse=False,
        validate=validate_bpca,
        apply=impute_bpca,
    )
)
