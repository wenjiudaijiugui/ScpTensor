"""Bayesian PCA imputation for single-cell proteomics data.

Reference:
    Oba S, Sato MA, Takemasa I, et al. A Bayesian missing value estimation
    method for gene expression profile data. Bioinformatics (2003).

BPCA extends Probabilistic PCA (PPCA) by using Bayesian inference to
automatically determine the effective number of components and avoid
overfitting through regularization.
"""

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from scipy.sparse.linalg import svds

from scptensor.core.exceptions import (
    AssayNotFoundError,
    DimensionError,
    LayerNotFoundError,
    ScpValueError,
)
from scptensor.core.structures import MaskCode, ScpContainer, ScpMatrix
from scptensor.impute._utils import _update_imputed_mask

# ============================================================================
# Internal utility functions
# ============================================================================


def _is_missing(x: npt.NDArray, missing_val: float | None) -> npt.NDArray:
    """Vectorized missing value detection.

    Parameters
    ----------
    x : NDArray
        Input array to check for missing values
    missing_val : float or None
        Missing value marker. If None or np.nan, uses np.isnan()

    Returns
    -------
    NDArray
        Boolean mask where True indicates missing values
    """
    if missing_val is None or np.isnan(missing_val):
        return np.isnan(x)
    return x == missing_val


def _not_missing(x: npt.NDArray, missing_val: float | None) -> npt.NDArray:
    """Vectorized non-missing value detection.

    Parameters
    ----------
    x : NDArray
        Input array to check for non-missing values
    missing_val : float or None
        Missing value marker. If None or np.nan, uses ~np.isnan()

    Returns
    -------
    NDArray
        Boolean mask where True indicates valid (non-missing) values
    """
    if missing_val is None or np.isnan(missing_val):
        return ~np.isnan(x)
    return x != missing_val


# ============================================================================
# Core BPCA algorithm functions
# ============================================================================


def bpca_init(y: npt.NDArray, q: int, missing_value: float | None = np.nan) -> dict:
    """Initialize BPCA model from data matrix.

    Parameters
    ----------
    y : NDArray
        Data matrix with missing values. Shape: (N, d)
    q : int
        Number of principal components (latent dimensions)
    missing_value : float or None, default np.nan
        Missing value marker

    Returns
    -------
    dict
        Dictionary containing initialized model parameters
    """
    N, d = y.shape
    miss_mask = _is_missing(y, missing_value)
    yest = y.copy()
    yest[miss_mask] = 0

    covy = np.cov(yest, rowvar=False)
    covy = np.nan_to_num(covy, nan=0.0)

    U, S, Vt = svds(covy, k=q)
    idx = np.argsort(S)[::-1]  # Sort singular values
    U = U[:, idx]
    S = S[idx]

    W = U * np.sqrt(S)

    mu = np.zeros(d)
    for j in range(d):
        valid = _not_missing(y[:, j], missing_value)
        if valid.any():
            mu[j] = y[valid, j].mean()

    tau = np.float64(1.0 / (np.sum(np.diag(covy)) - np.sum(S)))
    tau = np.float64(max(min(tau, 1e10), 1e-10))

    alpha = (2e-10 + d) / (tau * np.diag(W.T @ W) + 2e-10)

    gnomiss = ~miss_mask.any(axis=1)
    gmiss = miss_mask.any(axis=1)

    missidx = [np.where(miss_mask[i])[0] for i in range(N)]
    nomissidx = [np.where(~miss_mask[i])[0] for i in range(N)]

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
        "gnomiss": gnomiss,
        "gmiss": gmiss,
        "missidx": missidx,
        "nomissidx": nomissidx,
        "galpha0": 1e-10,
        "balpha0": 1.0,
        "gmu0": 0.001,
        "gtau0": 1e-10,
        "btau0": 1.0,
    }


def bpca_em_step(M: dict, y: npt.NDArray) -> dict:
    """Perform one EM step of Bayesian PCA.

    Parameters
    ----------
    M : dict
        Current model dictionary
    y : NDArray
        Data matrix with missing values

    Returns
    -------
    dict
        Updated model dictionary
    """
    q, N, d = M["q"], M["N"], M["d"]

    Rx = np.eye(q) + M["tau"] * (M["W"].T @ M["W"]) + M["SigW"]
    Rxinv = np.linalg.inv(Rx)
    Rxinv_Wt = Rxinv @ M["W"].T

    idx = M["gnomiss"]
    dy = y[idx] - M["mu"]
    x = M["tau"] * (Rxinv_Wt @ dy.T)

    T = dy.T @ x.T
    trS = np.sum(dy * dy)

    miss_idx = np.where(M["gmiss"])[0]
    for i in miss_idx:
        nomiss = M["nomissidx"][i]
        miss = M["missidx"][i]

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
        T[miss, :] = T[miss, :] + (Wm @ Rxinv_miss)
        trS = trS + dy @ dy + len(miss) / M["tau"] + np.trace(Wm @ Rxinv_miss @ Wm.T)

    T = T / N
    trS = trS / N

    Dw = Rxinv + M["tau"] * (T.T @ M["W"] @ Rxinv) + np.diag(M["alpha"]) / N
    Dwinv = np.linalg.inv(Dw)
    M["W"] = T @ Dwinv

    tau_numerator = d + 2 * M["gtau0"] / N
    tau_denominator = (
        trS
        - np.trace(T.T @ M["W"])
        + ((M["gmu0"] * np.outer(M["mu"], M["mu"]) + 2 * M["gtau0"] / M["btau0"]).sum()) / N
    )
    M["tau"] = max(min(tau_numerator / tau_denominator, 1e10), 1e-10)
    M["SigW"] = Dwinv * (d / N)
    M["alpha"] = (2 * M["galpha0"] + d) / (
        M["tau"] * np.diag(M["W"].T @ M["W"]) + np.diag(M["SigW"]) + 2 * M["galpha0"] / M["balpha0"]
    )

    return M


def bpca_fill(
    x999: npt.NDArray,
    k: int | None = None,
    maxepoch: int | None = None,
    missing_value: float | None = np.nan,
) -> tuple[npt.NDArray, dict]:
    """Fill missing values using Bayesian PCA.

    Parameters
    ----------
    x999 : NDArray
        Data matrix with missing values. Shape: (N, d)
    k : int, optional
        Number of principal components. Defaults to d-1
    maxepoch : int, optional
        Maximum number of EM iterations. Defaults to 200
    missing_value : float or None, default np.nan
        Missing value marker

    Returns
    -------
    y : NDArray
        Filled matrix with estimated values
    M : dict
        Dictionary containing learned parameters

    Notes
    -----
    Samples where ALL values are missing are excluded from training
    but remain in the output with their original values.
    """
    N, d = x999.shape
    k = d - 1 if k is None else k
    maxepoch = 200 if maxepoch is None else maxepoch

    if missing_value is None:
        if np.isnan(x999).any():
            missing_value = np.nan
        elif (x999 == 999.0).any() or (x999 > 900).any():
            missing_value = 999.0

    all_missing = _is_missing(x999, missing_value).all(axis=1)
    valid = ~all_missing
    M = bpca_init(x999[valid], k, missing_value=missing_value)

    tauold = 1000.0
    for epoch in range(maxepoch):
        M = bpca_em_step(M, x999[valid])
        if (epoch + 1) % 10 == 0:
            dtau = abs(np.log10(M["tau"]) - np.log10(tauold))
            if dtau < 1e-4:
                break
            tauold = M["tau"]

    y = x999.copy().astype(np.float64)
    y[valid] = M["yest"]
    return y, M


# ============================================================================
# ScpTensor integration function
# ============================================================================


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
    """Impute missing values using Bayesian PCA (BPCA).

    BPCA models the data as x = Wz + mu + epsilon using an EM algorithm
    with Bayesian regularization. Unlike standard PPCA, BPCA automatically
    determines the effective number of components through Bayesian inference,
    avoiding overfitting.

    Parameters
    ----------
    container : ScpContainer
        Input container with missing values
    assay_name : str
        Name of the assay to use
    source_layer : str
        Name of the layer containing data with missing values
    new_layer_name : str, default "imputed_bpca"
        Name for the new layer with imputed data
    n_components : int, optional
        Maximum number of principal components. If None, uses
        min(n_samples, n_features) - 1
    max_iter : int, default 100
        Maximum EM iterations
    tol : float, default 1e-6
        Convergence tolerance (kept for API compatibility)
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    ScpContainer
        Container with imputed data in new layer

    Raises
    ------
    AssayNotFoundError
        If the assay does not exist
    LayerNotFoundError
        If the layer does not exist
    ScpValueError
        If parameters are invalid
    DimensionError
        If n_components is too large

    Examples
    --------
    >>> from scptensor import impute_bpca
    >>> result = impute_bpca(container, "proteins", "raw", n_components=10)
    >>> "imputed_bpca" in result.assays["proteins"].layers
    True
    """
    if max_iter <= 0:
        raise ScpValueError(
            f"max_iter must be positive, got {max_iter}. "
            "Use max_iter >= 1 for EM algorithm iterations.",
            parameter="max_iter",
            value=max_iter,
        )
    if tol <= 0:
        raise ScpValueError(
            f"tol must be positive, got {tol}. Use tol > 0 for convergence tolerance.",
            parameter="tol",
            value=tol,
        )

    if assay_name not in container.assays:
        available = ", ".join(f"'{k}'" for k in container.assays)
        raise AssayNotFoundError(
            assay_name,
            hint=f"Available assays: {available}. Use container.list_assays() to see all assays.",
        )

    assay = container.assays[assay_name]
    if source_layer not in assay.layers:
        available = ", ".join(f"'{k}'" for k in assay.layers)
        raise LayerNotFoundError(
            source_layer,
            assay_name,
            hint=f"Available layers in assay '{assay_name}': {available}. "
            f"Use assay.list_layers() to see all layers.",
        )

    input_matrix = assay.layers[source_layer]
    X_original = input_matrix.X.copy()
    n_samples, n_features = X_original.shape

    if n_components is None:
        n_components = min(n_samples, n_features) - 1

    if n_components <= 0:
        raise ScpValueError(
            f"n_components must be positive, got {n_components}. "
            "Use n_components >= 1 for BPCA imputation.",
            parameter="n_components",
            value=n_components,
        )

    if n_components >= min(n_samples, n_features):
        raise DimensionError(
            f"n_components ({n_components}) must be less than "
            f"min(n_samples, n_features) = {min(n_samples, n_features)}",
            expected_shape=(n_samples, n_features),
            actual_shape=(n_samples, n_components),
        )

    if sp.issparse(X_original):
        X_dense = X_original.toarray()
    else:
        X_dense = np.asarray(X_original)

    missing_mask = np.isnan(X_dense)
    if not np.any(missing_mask):
        new_matrix = ScpMatrix(X=X_dense, M=_update_imputed_mask(input_matrix.M, missing_mask))
        layer_name = new_layer_name or "imputed_bpca"
        assay.add_layer(layer_name, new_matrix)
        container.log_operation(
            action="impute_bpca",
            params={
                "assay": assay_name,
                "source_layer": source_layer,
                "new_layer_name": layer_name,
                "n_components": n_components,
                "n_iterations": 0,
            },
            description=f"BPCA imputation on assay '{assay_name}': no missing values found.",
        )
        return container

    n_valid = np.sum(~missing_mask)
    if n_valid == 0:
        X_imputed = np.zeros_like(X_dense)
        new_matrix = ScpMatrix(X=X_imputed, M=_update_imputed_mask(input_matrix.M, missing_mask))
        layer_name = new_layer_name or "imputed_bpca"
        assay.add_layer(layer_name, new_matrix)
        container.log_operation(
            action="impute_bpca",
            params={
                "assay": assay_name,
                "source_layer": source_layer,
                "new_layer_name": layer_name,
                "n_components": n_components,
                "n_iterations": 0,
            },
            description=f"BPCA imputation on assay '{assay_name}': all values missing, filled with zeros.",
        )
        return container

    if random_state is not None:
        np.random.seed(random_state)

    X_imputed, model = bpca_fill(
        x999=X_dense, k=n_components, maxepoch=max_iter, missing_value=np.nan
    )

    new_matrix = ScpMatrix(X=X_imputed, M=_update_imputed_mask(input_matrix.M, missing_mask))
    layer_name = new_layer_name or "imputed_bpca"
    assay.add_layer(layer_name, new_matrix)

    W = model["W"]
    tau = model["tau"]
    effective_components = int(np.sum(np.sum(W**2, axis=0) > 1e-3))
    n_iterations = max_iter

    container.log_operation(
        action="impute_bpca",
        params={
            "assay": assay_name,
            "source_layer": source_layer,
            "new_layer_name": layer_name,
            "n_components": n_components,
            "effective_components": effective_components,
            "n_iterations": n_iterations,
            "final_tau": float(tau),
        },
        description=(
            f"BPCA imputation (n_components={n_components}, "
            f"effective={effective_components}) on assay '{assay_name}'."
        ),
    )

    return container


if __name__ == "__main__":
    print("Testing BPCA imputation...")

    np.random.seed(42)
    n_samples = 100
    n_features = 50

    W_true = np.random.randn(n_features, 5)
    Z_true = np.random.randn(5, n_samples)
    X_true = (W_true @ Z_true).T + np.random.randn(n_samples, n_features) * 0.1

    X_missing = X_true.copy()
    missing_mask = np.random.rand(n_samples, n_features) < 0.2
    X_missing[missing_mask] = np.nan

    import polars as pl

    from scptensor.core.structures import Assay, ScpContainer

    var = pl.DataFrame({"_index": [f"prot_{i}" for i in range(n_features)]})
    obs = pl.DataFrame({"_index": [f"cell_{i}" for i in range(n_samples)]})

    assay = Assay(var=var)
    assay.add_layer("raw", ScpMatrix(X=X_missing, M=None))

    container = ScpContainer(obs=obs, assays={"protein": assay})

    result = impute_bpca(
        container,
        assay_name="protein",
        source_layer="raw",
        n_components=10,
        max_iter=50,
        random_state=42,
    )

    assert "imputed_bpca" in result.assays["protein"].layers
    result_matrix = result.assays["protein"].layers["imputed_bpca"]
    X_imputed = result_matrix.X
    M_imputed = result_matrix.M

    assert not np.any(np.isnan(X_imputed))
    assert M_imputed is not None
    assert np.all(M_imputed[missing_mask] == MaskCode.IMPUTED)
    assert np.all(M_imputed[~missing_mask] == MaskCode.VALID)

    imputed_values = X_imputed[missing_mask]
    true_values = X_true[missing_mask]
    correlation = np.corrcoef(imputed_values, true_values)[0, 1]

    print(f"  Imputation correlation: {correlation:.3f}")
    print(f"  Shape: {X_imputed.shape}")
    print(f"  NaN count: {np.sum(np.isnan(X_imputed))}")
    print(f"  Mask code check: {np.sum(M_imputed == MaskCode.IMPUTED)} imputed values")
    print(f"  History log: {len(result.history)} entries")

    print("\nTesting BPCA imputation with existing mask...")

    M_initial = np.zeros(X_missing.shape, dtype=np.int8)
    n_missing = int(np.sum(missing_mask))
    M_initial[missing_mask] = np.where(np.random.rand(n_missing) < 0.5, MaskCode.MBR, MaskCode.LOD)

    assay2 = Assay(var=var)
    assay2.add_layer("raw", ScpMatrix(X=X_missing, M=M_initial))
    container2 = ScpContainer(obs=obs, assays={"protein": assay2})

    result2 = impute_bpca(
        container2,
        assay_name="protein",
        source_layer="raw",
        n_components=10,
        max_iter=50,
        random_state=42,
    )

    result_matrix2 = result2.assays["protein"].layers["imputed_bpca"]
    M_imputed2 = result_matrix2.M

    assert M_imputed2 is not None
    assert np.all(M_imputed2[missing_mask] == MaskCode.IMPUTED)
    assert np.all(M_imputed2[~missing_mask] == MaskCode.VALID)

    print("  Existing mask correctly updated to IMPUTED code")
    print(f"  Mask code check: {np.sum(M_imputed2 == MaskCode.IMPUTED)} imputed values")
    print("All tests passed")
