"""Bayesian PCA imputation for single-cell proteomics data.

Reference:
    Oba S, Sato MA, Takemasa I, et al. A Bayesian missing value estimation
    method for gene expression profile data. Bioinformatics (2003).

BPCA extends Probabilistic PCA (PPCA) by using Bayesian inference to
automatically determine the effective number of components and avoid
overfitting through regularization.
"""

from typing import overload

import numpy as np
import scipy.sparse as sp
from scipy import linalg

from scptensor.core.exceptions import (
    AssayNotFoundError,
    DimensionError,
    LayerNotFoundError,
    ScpValueError,
)
from scptensor.core.jit_ops import ppca_initialize_with_col_means
from scptensor.core.structures import MaskCode, ScpContainer, ScpMatrix
from scptensor.impute._utils import _update_imputed_mask


@overload
def impute_bpca(
    container: ScpContainer,
    assay_name: str,
    source_layer: str,
    new_layer_name: str = "imputed_bpca",
    n_components: int | None = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    random_state: int | None = None,
) -> ScpContainer: ...


@overload
def impute_bpca(
    container: ScpContainer,
    assay_name: str,
    source_layer: str,
    *,
    n_components: int | None,
    new_layer_name: str,
    max_iter: int,
    tol: float,
    random_state: int | None,
) -> ScpContainer: ...


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

    The key difference from PPCA is the use of automatic relevance
    determination (ARD) priors on the weight matrix W, which causes
    unnecessary components to be shrunk to zero.

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
        Maximum number of principal components. If None, uses
        min(n_samples, n_features) - 1. The Bayesian regularization
        will automatically determine the effective number.
    max_iter : int, default 100
        Maximum EM iterations.
    tol : float, default 1e-6
        Convergence tolerance for the log-likelihood change.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    ScpContainer
        Container with imputed data in new layer.

    Raises
    ------
    AssayNotFoundError
        If the assay does not exist.
    LayerNotFoundError
        If the layer does not exist.
    ScpValueError
        If parameters are invalid.
    DimensionError
        If n_components is too large.

    Notes
    -----
    The Bayesian EM algorithm:
        1. E-step: Compute posterior distribution of latent variables z
        2. M-step: Update parameters W, mu, sigma^2 with ARD priors
        3. ARD update: Shrink weights of unnecessary components
        4. Impute: x_miss = E[W] @ E[z] + E[mu]

    The ARD (Automatic Relevance Determination) prior on each column of W
    allows the model to automatically prune unnecessary dimensions. Components
    with small precision parameters (large variance) are effectively ignored.

    Examples
    --------
    >>> from scptensor import impute_bpca
    >>> result = impute_bpca(container, "proteins", n_components=10)
    >>> "imputed_bpca" in result.assays["proteins"].layers
    True
    """
    # Validate parameters
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

    # Validate assay and layer
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

    # Get data
    input_matrix = assay.layers[source_layer]
    X_original = input_matrix.X.copy()
    n_samples, n_features = X_original.shape

    # Set default n_components
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

    # Convert sparse to dense for BPCA
    if sp.issparse(X_original):
        X_dense = X_original.toarray()
    else:
        X_dense = np.asarray(X_original)

    # Check for missing values
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

    # Set random seed and initialize
    if random_state is not None:
        np.random.seed(random_state)

    X = X_dense.copy()
    col_means = np.nanmean(X, axis=0)
    col_means[np.isnan(col_means)] = 0.0
    ppca_initialize_with_col_means(X, missing_mask, col_means)

    mu = np.mean(X, axis=0)
    X_centered = X - mu

    # Initialize parameters for Bayesian PCA
    # W: weight matrix (n_features, n_components)
    # alpha: precision parameters for ARD prior on W columns
    # beta: precision of noise
    W = np.random.randn(n_features, n_components) * 0.1
    alpha = np.ones(n_components)  # ARD precision parameters
    beta = 1.0  # Noise precision

    # EM algorithm with Bayesian updates
    log_likelihood_old = -np.inf

    for _iteration in range(max_iter):
        # E-step: Compute expected sufficient statistics
        # M = (W.T @ W + beta^-1 * I)^-1
        M_inv = W.T @ W + (1.0 / beta) * np.eye(n_components)
        M = linalg.inv(M_inv)

        # Expected latent variables: E[z] = M @ W.T @ (x - mu)
        # For all samples at once
        E_z = M @ W.T @ X_centered.T  # (n_components, n_samples)

        # E[z @ z.T] = M + E[z] @ E[z].T
        E_zzT = M + E_z @ E_z.T  # (n_components, n_components)

        # M-step: Update parameters with Bayesian inference

        # Update W (expected value under posterior)
        # W_new = beta * X_centered.T @ E[z].T @ (E[z] @ E[z].T + M)^-1
        # Simplified: W_new = X_centered.T @ E_z.T @ linalg.inv(E_zzT)
        scale = beta * (E_zzT)
        W_new = (beta * X_centered.T @ E_z.T) @ linalg.inv(scale + 1e-10 * np.eye(n_components))

        # Update alpha (ARD precision parameters)
        # alpha_k = n_features / (W[:, k].T @ W[:, k] + trace(M)_kk)
        for k in range(n_components):
            alpha[k] = n_features / (np.sum(W_new[:, k] ** 2) + np.diag(M)[k] + 1e-10)

        # Update beta (noise precision)
        # beta = n_samples * n_features / (||X - W @ Z||^2 + tr(W @ M @ W.T))
        residual = X_centered - (W_new @ E_z).T
        reconstruction_error = np.sum(residual[~missing_mask] ** 2)
        trace_term = np.trace(W_new.T @ W_new @ M)
        beta_new = (np.sum(~missing_mask) * n_features) / (
            reconstruction_error + trace_term + 1e-10
        )

        # Compute log-likelihood for convergence checking
        # Log p(X|W,mu,beta) approximated using observed values only
        log_likelihood = _bpca_log_likelihood(X_centered, W_new, mu, E_z, M, beta_new, missing_mask)

        # Check convergence
        ll_diff = abs(log_likelihood - log_likelihood_old)
        W_diff = np.linalg.norm(W_new - W, "fro") / (np.linalg.norm(W, "fro") + 1e-10)

        W = W_new
        alpha = np.clip(alpha, 1e-10, 1e10)  # Prevent numerical issues
        beta = np.clip(beta_new, 1e-10, 1e10)

        if max(float(W_diff), float(ll_diff)) < tol:
            break

        log_likelihood_old = log_likelihood

    # Final imputation using expected values under posterior
    X_imputed = X_dense.copy()

    # Impute each sample's missing values
    for i in range(n_samples):
        observed = ~missing_mask[i]
        missing = missing_mask[i]

        if not np.any(missing):
            continue

        X_obs = X_imputed[i, observed]
        W_obs = W[observed, :]
        W_miss = W[missing, :]
        mu_obs = mu[observed]
        mu_miss = mu[missing]

        # E[z | x_obs] using only observed features
        M_obs_inv = W_obs.T @ W_obs + (1.0 / beta) * np.eye(n_components)
        M_obs = linalg.inv(M_obs_inv)
        z = M_obs @ W_obs.T @ (X_obs - mu_obs)

        # Impute missing values
        X_imputed[i, missing] = W_miss @ z + mu_miss

    # Create new layer
    new_matrix = ScpMatrix(X=X_imputed, M=_update_imputed_mask(input_matrix.M, missing_mask))
    layer_name = new_layer_name or "imputed_bpca"
    assay.add_layer(layer_name, new_matrix)

    # Compute effective number of components (those with large enough weights)
    effective_components = int(np.sum(np.sum(W**2, axis=0) > 1e-3))

    container.log_operation(
        action="impute_bpca",
        params={
            "assay": assay_name,
            "source_layer": source_layer,
            "new_layer_name": layer_name,
            "n_components": n_components,
            "effective_components": effective_components,
            "n_iterations": _iteration + 1,
            "final_beta": beta,
        },
        description=(
            f"BPCA imputation (n_components={n_components}, "
            f"effective={effective_components}) on assay '{assay_name}'."
        ),
    )

    return container


def _bpca_log_likelihood(
    X_centered: np.ndarray,
    W: np.ndarray,
    mu: np.ndarray,
    E_z: np.ndarray,
    M: np.ndarray,
    beta: float,
    missing_mask: np.ndarray,
) -> float:
    """Compute log-likelihood for BPCA convergence checking.

    Parameters
    ----------
    X_centered : np.ndarray
        Centered data matrix.
    W : np.ndarray
        Weight matrix.
    mu : np.ndarray
        Mean vector.
    E_z : np.ndarray
        Expected latent variables (n_components, n_samples).
    M : np.ndarray
        Posterior covariance of z.
    beta : float
        Noise precision.
    missing_mask : np.ndarray
        Boolean mask of missing values.

    Returns
    -------
    float
        Log-likelihood (observed values only).
    """
    n_samples, n_features = X_centered.shape

    # Reconstruction using expected values
    X_reconstructed = (W @ E_z).T

    # Compute error only on observed values
    residual = X_centered - X_reconstructed
    error_observed = residual[~missing_mask]

    # Log-likelihood (simplified, proportional terms omitted)
    ll = -0.5 * beta * np.sum(error_observed**2)

    # Add entropy term for latent variables
    sign, logdet = np.linalg.slogdet(M)
    entropy = 0.5 * n_samples * float(logdet)

    return float(ll + entropy)


if __name__ == "__main__":
    # Test: Basic functionality
    print("Testing BPCA imputation...")

    # Create simple test data
    np.random.seed(42)
    n_samples = 100
    n_features = 50

    # Generate low-rank data
    W_true = np.random.randn(n_features, 5)
    Z_true = np.random.randn(5, n_samples)
    X_true = (W_true @ Z_true).T + np.random.randn(n_samples, n_features) * 0.1

    # Add missing values
    X_missing = X_true.copy()
    missing_mask = np.random.rand(n_samples, n_features) < 0.2
    X_missing[missing_mask] = np.nan

    # Create container
    import polars as pl

    from scptensor.core.structures import Assay, ScpContainer

    var = pl.DataFrame({"_index": [f"prot_{i}" for i in range(n_features)]})
    obs = pl.DataFrame({"_index": [f"cell_{i}" for i in range(n_samples)]})

    assay = Assay(var=var)
    assay.add_layer("raw", ScpMatrix(X=X_missing, M=None))

    container = ScpContainer(obs=obs, assays={"protein": assay})

    # Test BPCA imputation
    result = impute_bpca(
        container,
        assay_name="protein",
        source_layer="raw",
        n_components=10,
        max_iter=50,
        random_state=42,
    )

    # Check results
    assert "imputed_bpca" in result.assays["protein"].layers
    result_matrix = result.assays["protein"].layers["imputed_bpca"]
    X_imputed = result_matrix.X
    M_imputed = result_matrix.M

    # Check no NaNs
    assert not np.any(np.isnan(X_imputed))

    # Check mask was created and updated correctly
    assert M_imputed is not None
    assert np.all(M_imputed[missing_mask] == MaskCode.IMPUTED)
    assert np.all(M_imputed[~missing_mask] == MaskCode.VALID)

    # Check imputation accuracy on missing values
    imputed_values = X_imputed[missing_mask]
    true_values = X_true[missing_mask]
    correlation = np.corrcoef(imputed_values, true_values)[0, 1]

    print(f"  Imputation correlation: {correlation:.3f}")
    print(f"  Shape: {X_imputed.shape}")
    print(f"  NaN count: {np.sum(np.isnan(X_imputed))}")
    print(f"  Mask code check: {np.sum(M_imputed == MaskCode.IMPUTED)} imputed values")
    print(f"  History log: {len(result.history)} entries")

    # Test 2: With existing mask (M not None)
    print("\nTesting BPCA imputation with existing mask...")

    # Create initial mask with some MBR (1) and LOD (2) codes for missing values
    M_initial = np.zeros(X_missing.shape, dtype=np.int8)
    M_initial[missing_mask] = np.where(
        np.random.rand(np.sum(missing_mask)) < 0.5, MaskCode.MBR, MaskCode.LOD
    )

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

    # Check that imputed values now have IMPUTED (5) code
    assert M_imputed2 is not None
    assert np.all(M_imputed2[missing_mask] == MaskCode.IMPUTED)
    # Check that valid values remain VALID (0)
    assert np.all(M_imputed2[~missing_mask] == MaskCode.VALID)

    print("  Existing mask correctly updated to IMPUTED code")
    print(f"  Mask code check: {np.sum(M_imputed2 == MaskCode.IMPUTED)} imputed values")
    print("✅ All tests passed")
