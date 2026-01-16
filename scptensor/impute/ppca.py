"""Probabilistic PCA imputation for single-cell proteomics data.

Reference:
    Tipping ME, Bishop CM. Probabilistic principal component analysis.
    Journal of the Royal Statistical Society (1999).
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
def impute_ppca(
    container: ScpContainer,
    assay_name: str,
    source_layer: str,
    new_layer_name: str = "imputed_ppca",
    n_components: int = 10,
    max_iter: int = 100,
    tol: float = 1e-6,
    random_state: int | None = None,
) -> ScpContainer: ...


def impute_ppca(
    container: ScpContainer,
    assay_name: str,
    source_layer: str,
    new_layer_name: str = "imputed_ppca",
    n_components: int = 10,
    max_iter: int = 100,
    tol: float = 1e-6,
    random_state: int | None = None,
) -> ScpContainer:
    """
    Impute missing values using Probabilistic PCA (PPCA).

    PPCA models the data as x = Wz + mu + epsilon where z ~ N(0, I),
    using the EM algorithm to estimate parameters.

    Parameters
    ----------
    container : ScpContainer
        Input container with missing values.
    assay_name : str
        Name of the assay to use.
    source_layer : str, default "raw"
        Name of the layer containing data with missing values.
    new_layer_name : str, default "imputed_ppca"
        Name for the new layer with imputed data.
    n_components : int, default 10
        Number of principal components.
    max_iter : int, default 100
        Maximum EM iterations.
    tol : float, default 1e-6
        Convergence tolerance.
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
    The EM algorithm:
        1. E-step: Compute E[z] = Mx @ W.T @ (x - mu)
        2. M-step: Update W and sigma^2
        3. Impute: x_miss = W_miss @ E[z] + mu_miss

    Examples
    --------
    >>> from scptensor import impute_ppca
    >>> result = impute_ppca(container, "proteins", n_components=10)
    >>> "imputed_ppca" in result.assays["proteins"].layers
    True
    """
    # Validate parameters
    if n_components <= 0:
        raise ScpValueError(
            f"n_components must be positive, got {n_components}. "
            "Use n_components >= 1 for PPCA imputation.",
            parameter="n_components",
            value=n_components,
        )
    if max_iter <= 0:
        raise ScpValueError(
            f"max_iter must be positive, got {max_iter}. "
            "Use max_iter >= 1 for EM algorithm iterations.",
            parameter="max_iter",
            value=max_iter,
        )
    if tol <= 0:
        raise ScpValueError(
            f"tol must be positive, got {tol}. "
            "Use tol > 0 for convergence tolerance.",
            parameter="tol",
            value=tol,
        )

    # Validate assay and layer
    if assay_name not in container.assays:
        available = ", ".join(f"'{k}'" for k in container.assays.keys())
        raise AssayNotFoundError(
            assay_name,
            hint=f"Available assays: {available}. "
            f"Use container.list_assays() to see all assays.",
        )

    assay = container.assays[assay_name]
    if source_layer not in assay.layers:
        available = ", ".join(f"'{k}'" for k in assay.layers.keys())
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

    if n_components >= min(n_samples, n_features):
        raise DimensionError(
            f"n_components ({n_components}) must be less than "
            f"min(n_samples, n_features) = {min(n_samples, n_features)}",
            expected_shape=(n_samples, n_features),
            actual_shape=(n_samples, n_components),
        )

    # Convert sparse to dense for PPCA
    X_dense = X_original.toarray() if sp.issparse(X_original) else X_original

    # Check for missing values
    missing_mask = np.isnan(X_dense)
    if not np.any(missing_mask):
        new_matrix = ScpMatrix(X=X_dense, M=_update_imputed_mask(input_matrix.M, missing_mask))
        layer_name = new_layer_name or "imputed_ppca"
        assay.add_layer(layer_name, new_matrix)
        container.log_operation(
            action="impute_ppca",
            params={
                "assay": assay_name,
                "source_layer": source_layer,
                "new_layer_name": layer_name,
                "n_components": n_components,
                "n_iterations": 0,
            },
            description=f"PPCA imputation on assay '{assay_name}': no missing values found.",
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

    # Initialize parameters
    W = np.random.randn(n_features, n_components) * 0.1
    sigma2 = 1.0

    # EM algorithm
    for iteration in range(max_iter):
        # E-step
        Mx = linalg.inv(W.T @ W + sigma2 * np.eye(n_components))
        Z = Mx @ W.T @ X_centered.T
        S = sigma2 * Mx + Z @ Z.T

        # M-step
        W_new = X_centered.T @ Z.T @ linalg.inv(S)
        X_reconstructed = (W_new @ Z).T
        diff = X_centered - X_reconstructed
        sigma2_new = (np.sum(diff**2) + n_samples * sigma2 * np.trace(Mx)) / (
            n_samples * n_features
        )

        # Check convergence
        W_diff = np.linalg.norm(W_new - W, "fro") / (np.linalg.norm(W, "fro") + 1e-10)
        sigma2_diff = abs(sigma2_new - sigma2) / (sigma2 + 1e-10)

        W = W_new
        sigma2 = sigma2_new

        if max(W_diff, sigma2_diff) < tol:
            break

    # Final imputation
    Mx = linalg.inv(W.T @ W + sigma2 * np.eye(n_components))
    X_imputed = X_dense.copy()

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

        z = Mx @ W_obs.T @ (X_obs - mu_obs)
        X_imputed[i, missing] = W_miss @ z + mu_miss

    # Create new layer
    new_matrix = ScpMatrix(X=X_imputed, M=_update_imputed_mask(input_matrix.M, missing_mask))
    layer_name = new_layer_name or "imputed_ppca"
    assay.add_layer(layer_name, new_matrix)

    container.log_operation(
        action="impute_ppca",
        params={
            "assay": assay_name,
            "source_layer": source_layer,
            "new_layer_name": layer_name,
            "n_components": n_components,
            "n_iterations": iteration + 1,
            "final_sigma2": sigma2,
        },
        description=f"PPCA imputation (n_components={n_components}) on assay '{assay_name}'.",
    )

    return container



if __name__ == "__main__":
    # Test: Basic functionality
    print("Testing PPCA imputation...")

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

    from scptensor.core.structures import Assay, MaskCode, ScpContainer

    var = pl.DataFrame({"_index": [f"prot_{i}" for i in range(n_features)]})
    obs = pl.DataFrame({"_index": [f"cell_{i}" for i in range(n_samples)]})

    assay = Assay(var=var)
    assay.add_layer("raw", ScpMatrix(X=X_missing, M=None))

    container = ScpContainer(obs=obs, assays={"protein": assay})

    # Test PPCA imputation
    result = impute_ppca(
        container,
        assay_name="protein",
        source_layer="raw",
        n_components=5,
        max_iter=50,
        random_state=42,
    )

    # Check results
    assert "imputed_ppca" in result.assays["protein"].layers
    result_matrix = result.assays["protein"].layers["imputed_ppca"]
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
    print("\nTesting PPCA imputation with existing mask...")

    # Create initial mask with some MBR (1) and LOD (2) codes for missing values
    M_initial = np.zeros(X_missing.shape, dtype=np.int8)
    M_initial[missing_mask] = np.where(
        np.random.rand(np.sum(missing_mask)) < 0.5, MaskCode.MBR, MaskCode.LOD
    )

    assay2 = Assay(var=var)
    assay2.add_layer("raw", ScpMatrix(X=X_missing, M=M_initial))
    container2 = ScpContainer(obs=obs, assays={"protein": assay2})

    result2 = impute_ppca(
        container2,
        assay_name="protein",
        source_layer="raw",
        n_components=5,
        max_iter=50,
        random_state=42,
    )

    result_matrix2 = result2.assays["protein"].layers["imputed_ppca"]
    M_imputed2 = result_matrix2.M

    # Check that imputed values now have IMPUTED (5) code
    assert M_imputed2 is not None
    assert np.all(M_imputed2[missing_mask] == MaskCode.IMPUTED)
    # Check that valid values remain VALID (0)
    assert np.all(M_imputed2[~missing_mask] == MaskCode.VALID)

    print("  Existing mask correctly updated to IMPUTED code")
    print(f"  Mask code check: {np.sum(M_imputed2 == MaskCode.IMPUTED)} imputed values")
    print("✅ All tests passed")
