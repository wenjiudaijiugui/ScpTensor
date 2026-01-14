"""SVD-based imputation for single-cell proteomics data.

Reference:
    Troyanskaya O, et al. Missing value estimation methods for DNA microarrays.
    Bioinformatics (2001).
"""

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


def svd_impute(
    container: ScpContainer,
    assay_name: str,
    base_layer: str,
    new_layer_name: str | None = "imputed_svd",
    n_components: int = 10,
    max_iter: int = 100,
    tol: float = 1e-6,
    init_method: str = "mean",
) -> ScpContainer:
    """
    Impute missing values using iterative SVD.

    Iteratively computes SVD and reconstructs low-rank approximation
    to fill missing values.

    Parameters
    ----------
    container : ScpContainer
        Input container with missing values
    assay_name : str
        Name of the assay to use
    base_layer : str
        Name of the layer containing data with missing values
    new_layer_name : str, optional
        Name for the new layer with imputed data (default: 'imputed_svd')
    n_components : int, optional
        Number of singular components (default: 10)
    max_iter : int, optional
        Maximum iterations (default: 100)
    tol : float, optional
        Convergence tolerance (default: 1e-6)
    init_method : str, optional
        Initialization: 'mean' or 'median' (default: 'mean')

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

    Notes
    -----
    Algorithm:
        1. Initialize missing values (mean/median)
        2. Repeat: SVD -> low-rank reconstruction -> fill missing
        3. Stop when convergence or max_iter reached
    """
    # Validate parameters
    if n_components <= 0:
        raise ScpValueError(
            f"n_components must be positive, got {n_components}.",
            parameter="n_components",
            value=n_components,
        )
    if max_iter <= 0:
        raise ScpValueError(
            f"max_iter must be positive, got {max_iter}.",
            parameter="max_iter",
            value=max_iter,
        )
    if tol <= 0:
        raise ScpValueError(
            f"tol must be positive, got {tol}.",
            parameter="tol",
            value=tol,
        )
    if init_method not in ("mean", "median"):
        raise ScpValueError(
            f"init_method must be 'mean' or 'median', got '{init_method}'.",
            parameter="init_method",
            value=init_method,
        )

    # Validate assay and layer
    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]
    if base_layer not in assay.layers:
        raise LayerNotFoundError(base_layer, assay_name)

    # Get data
    input_matrix = assay.layers[base_layer]
    X_original = input_matrix.X.copy()
    n_samples, n_features = X_original.shape

    if n_components >= min(n_samples, n_features):
        raise DimensionError(
            f"n_components ({n_components}) must be less than "
            f"min(n_samples, n_features) = {min(n_samples, n_features)}",
            expected_shape=(n_samples, n_features),
            actual_shape=(n_samples, n_components),
        )

    # Convert sparse to dense
    X_dense = X_original.toarray() if sp.issparse(X_original) else X_original

    # Check for missing values
    missing_mask = np.isnan(X_dense)
    if not np.any(missing_mask):
        new_matrix = ScpMatrix(
            X=X_dense, M=_update_imputed_mask(input_matrix.M, missing_mask)
        )
        layer_name = new_layer_name or "imputed_svd"
        assay.add_layer(layer_name, new_matrix)
        container.log_operation(
            action="impute_svd",
            params={
                "assay": assay_name,
                "base_layer": base_layer,
                "new_layer": layer_name,
                "n_components": n_components,
                "n_iterations": 0,
            },
            description=f"SVD imputation on assay '{assay_name}': no missing values found.",
        )
        return container

    # Initialize missing values
    X = X_dense.copy()

    if init_method == "mean":
        col_means = np.nanmean(X, axis=0)
        col_means[np.isnan(col_means)] = 0.0
        ppca_initialize_with_col_means(X, missing_mask, col_means)
    else:  # median
        col_medians = np.nanmedian(X, axis=0)
        col_medians[np.isnan(col_medians)] = 0.0
        for j in range(n_features):
            X[missing_mask[:, j], j] = col_medians[j]

    # Iterative SVD
    X_old = X.copy()

    for iteration in range(max_iter):
        U, s, Vt = linalg.svd(X, full_matrices=False)

        # Keep top k components
        U_k = U[:, :n_components]
        s_k = s[:n_components]
        Vt_k = Vt[:n_components, :]

        X_reconstructed = U_k @ np.diag(s_k) @ Vt_k
        X[missing_mask] = X_reconstructed[missing_mask]

        # Check convergence
        diff_norm = np.linalg.norm(X[missing_mask] - X_old[missing_mask]) / (
            np.linalg.norm(X_old[missing_mask]) + 1e-10
        )

        if diff_norm < tol:
            break

        X_old = X.copy()

    # Create new layer
    new_matrix = ScpMatrix(
        X=X, M=_update_imputed_mask(input_matrix.M, missing_mask)
    )
    layer_name = new_layer_name or "imputed_svd"
    assay.add_layer(layer_name, new_matrix)

    container.log_operation(
        action="impute_svd",
        params={
            "assay": assay_name,
            "base_layer": base_layer,
            "new_layer": layer_name,
            "n_components": n_components,
            "n_iterations": iteration + 1,
            "init_method": init_method,
        },
        description=f"SVD imputation (n_components={n_components}, init={init_method}) on assay '{assay_name}'.",
    )

    return container


if __name__ == "__main__":
    # Test: Basic functionality
    print("Testing SVD imputation...")

    # Create simple test data
    np.random.seed(42)
    n_samples = 100
    n_features = 50

    # Generate low-rank data
    U_true = np.random.randn(n_samples, 5)
    V_true = np.random.randn(n_features, 5)
    X_true = U_true @ V_true.T + np.random.randn(n_samples, n_features) * 0.1

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

    # Test SVD imputation
    result = svd_impute(
        container, assay_name="protein", base_layer="raw", n_components=5, max_iter=50
    )

    # Check results
    assert "imputed_svd" in result.assays["protein"].layers
    result_matrix = result.assays["protein"].layers["imputed_svd"]
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
    mae = np.mean(np.abs(imputed_values - true_values))
    correlation = np.corrcoef(imputed_values, true_values)[0, 1]

    print(f"  Imputation MAE: {mae:.3f}")
    print(f"  Imputation correlation: {correlation:.3f}")
    print(f"  Shape: {X_imputed.shape}")
    print(f"  NaN count: {np.sum(np.isnan(X_imputed))}")
    print(f"  Mask code check: {np.sum(M_imputed == MaskCode.IMPUTED)} imputed values")
    print(f"  History log: {len(result.history)} entries")

    # Test 2: With existing mask (M not None)
    print("\nTesting SVD imputation with existing mask...")

    # Create initial mask with some MBR (1) and LOD (2) codes for missing values
    M_initial = np.zeros(X_missing.shape, dtype=np.int8)
    M_initial[missing_mask] = np.where(
        np.random.rand(np.sum(missing_mask)) < 0.5, MaskCode.MBR, MaskCode.LOD
    )

    assay2 = Assay(var=var)
    assay2.add_layer("raw", ScpMatrix(X=X_missing, M=M_initial))
    container2 = ScpContainer(obs=obs, assays={"protein": assay2})

    result2 = svd_impute(
        container2, assay_name="protein", base_layer="raw", n_components=5, max_iter=50
    )

    result_matrix2 = result2.assays["protein"].layers["imputed_svd"]
    M_imputed2 = result_matrix2.M

    # Check that imputed values now have IMPUTED (5) code
    assert M_imputed2 is not None
    assert np.all(M_imputed2[missing_mask] == MaskCode.IMPUTED)
    # Check that valid values remain VALID (0)
    assert np.all(M_imputed2[~missing_mask] == MaskCode.VALID)

    print("  Existing mask correctly updated to IMPUTED code")
    print(f"  Mask code check: {np.sum(M_imputed2 == MaskCode.IMPUTED)} imputed values")

    # Test 3: Test with different init method
    print("\nTesting SVD imputation with median initialization...")

    assay3 = Assay(var=var)
    assay3.add_layer("raw", ScpMatrix(X=X_missing, M=None))
    container3 = ScpContainer(obs=obs, assays={"protein": assay3})

    result3 = svd_impute(
        container3,
        assay_name="protein",
        base_layer="raw",
        new_layer_name="imputed_median",
        n_components=5,
        init_method="median",
    )

    assert "imputed_median" in result3.assays["protein"].layers
    X_imputed3 = result3.assays["protein"].layers["imputed_median"].X
    assert not np.any(np.isnan(X_imputed3))

    print("  Median initialization works correctly")

    # Test 4: Test sparse matrix input
    print("\nTesting SVD imputation with sparse input...")

    from scipy import sparse as sp

    X_sparse = sp.csr_matrix(X_missing)

    assay4 = Assay(var=var)
    assay4.add_layer("raw", ScpMatrix(X=X_sparse, M=None))
    container4 = ScpContainer(obs=obs, assays={"protein": assay4})

    result4 = svd_impute(
        container4,
        assay_name="protein",
        base_layer="raw",
        new_layer_name="imputed_from_sparse",
        n_components=5,
    )

    assert "imputed_from_sparse" in result4.assays["protein"].layers
    X_imputed4 = result4.assays["protein"].layers["imputed_from_sparse"].X
    assert not np.any(np.isnan(X_imputed4))

    print("  Sparse input handled correctly")

    # Test 5: Test error handling
    print("\nTesting error handling...")

    try:
        # Invalid assay
        svd_impute(container, assay_name="nonexistent", base_layer="raw")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "not found" in str(e)
        print("  Invalid assay error: OK")

    try:
        # Invalid layer
        svd_impute(container, assay_name="protein", base_layer="nonexistent")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "not found" in str(e)
        print("  Invalid layer error: OK")

    try:
        # Invalid n_components
        svd_impute(
            container,
            assay_name="protein",
            base_layer="raw",
            n_components=1000,  # Too large
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "n_components" in str(e)
        print("  Invalid n_components error: OK")

    try:
        # Invalid init_method
        svd_impute(container, assay_name="protein", base_layer="raw", init_method="invalid")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "init_method" in str(e)
        print("  Invalid init_method error: OK")

    print("\n✅ All tests passed")
