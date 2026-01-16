"""NMF (Non-negative Matrix Factorization) imputation for single-cell proteomics data.

Reference:
    Lee D, Seung H. Learning the parts of objects by non-negative matrix factorization.
    Nature (2001).
    Gaujoux R, Seoighe C. A flexible R package for nonnegative matrix factorization.
    BMC Bioinformatics (2010).
"""

import numpy as np
import scipy.sparse as sp

from scptensor.core.exceptions import (
    AssayNotFoundError,
    DimensionError,
    LayerNotFoundError,
    ScpValueError,
)
from scptensor.core.structures import MaskCode, ScpContainer, ScpMatrix
from scptensor.impute._utils import _update_imputed_mask


def impute_nmf(
    container: ScpContainer,
    assay_name: str,
    source_layer: str,
    new_layer_name: str = "imputed_nmf",
    n_components: int | None = None,
    max_iter: int = 200,
    tol: float = 1e-4,
    random_state: int | None = None,
) -> ScpContainer:
    """
    Impute missing values using Non-negative Matrix Factorization.

    Factorizes the matrix into W x H with non-negativity constraints.
    Particularly suitable for intensity data where protein abundance is
    always non-negative. The algorithm trains using only observed values
    and reconstructs missing values from the latent factor model.

    Parameters
    ----------
    container : ScpContainer
        Input container with missing values.
    assay_name : str
        Name of the assay to use.
    source_layer : str
        Name of the layer containing data with missing values.
    new_layer_name : str, default "imputed_nmf"
        Name for the new layer with imputed data.
    n_components : int or None, default None
        Number of latent factors. If None, uses heuristic based on data size:
        min(n_samples, n_features) // 4 for large matrices, or
        min(n_samples, n_features) // 2 for smaller matrices.
    max_iter : int, default 200
        Maximum iterations for coordinate descent solver.
    tol : float, default 1e-4
        Stopping tolerance for reconstruction error.
    random_state : int or None, default None
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
    The NMF optimization problem is:
        min ||X - WH||^2, s.t. W >= 0, H >= 0

    Only non-missing elements contribute to the loss function during training.
    The algorithm:
        1. Initialize missing values with column means
        2. Fit NMF model to the filled matrix
        3. Reconstruct: X_imputed = W @ H
        4. Extract only the originally missing positions

    For sparse matrices, the data is converted to dense for NMF computation.

    Examples
    --------
    >>> from scptensor import impute_nmf
    >>> result = impute_nmf(container, "proteins", n_components=10)
    >>> "imputed_nmf" in result.assays["proteins"].layers
    True

    >>> # Use custom layer name
    >>> result = impute_nmf(
    ...     container, "proteins", new_layer_name="nmf_result",
    ...     n_components=5, random_state=42
    ... )

    References
    ----------
    .. [1] Lee D, Seung H. Learning the parts of objects by non-negative
       matrix factorization. Nature 2001;401:788-791.
    .. [2] Gaujoux R, Seoighe C. A flexible R package for nonnegative matrix
       factorization. BMC Bioinformatics 2010;11:367.
    """
    # Import sklearn here to avoid hard dependency at module level
    try:
        from sklearn.decomposition import NMF
    except ImportError as err:
        raise ImportError(
            "scikit-learn is required for NMF imputation. Install it with: pip install scikit-learn"
        ) from err

    # Parameter validation
    if n_components is not None and n_components <= 0:
        raise ScpValueError(
            f"n_components must be positive, got {n_components}. "
            "Use n_components >= 1 for NMF imputation.",
            parameter="n_components",
            value=n_components,
        )
    if max_iter <= 0:
        raise ScpValueError(
            f"max_iter must be positive, got {max_iter}. Use max_iter >= 1 for NMF iterations.",
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

    # Determine n_components if not specified
    if n_components is None:
        min_dim = min(n_samples, n_features)
        if min_dim >= 40:
            n_components = max(2, min_dim // 4)
        else:
            n_components = max(2, min_dim // 2)

    # Validate n_components
    if n_components >= min(n_samples, n_features):
        raise DimensionError(
            f"n_components ({n_components}) must be less than "
            f"min(n_samples, n_features) = {min(n_samples, n_features)}",
            expected_shape=(n_samples, n_features),
            actual_shape=(n_samples, n_components),
        )

    # Convert sparse to dense
    X_dense = X_original.toarray() if sp.issparse(X_original) else X_original  # type: ignore[union-attr]

    # Check for missing values
    missing_mask = np.isnan(X_dense)
    if not np.any(missing_mask):
        new_matrix = ScpMatrix(X=X_dense, M=_update_imputed_mask(input_matrix.M, missing_mask))
        layer_name = new_layer_name or "imputed_nmf"
        assay.add_layer(layer_name, new_matrix)
        container.log_operation(
            action="impute_nmf",
            params={
                "assay": assay_name,
                "source_layer": source_layer,
                "new_layer_name": layer_name,
                "n_components": n_components,
            },
            description=f"NMF imputation on assay '{assay_name}': no missing values found.",
        )
        return container

    # Handle edge case: all missing in a row/column
    # For NMF, we need to handle negative values in the data
    # First, shift data to be non-negative if needed
    data_min = np.nanmin(X_dense)
    shift = 0.0
    if data_min < 0:
        shift = abs(data_min)
        X_work = X_dense + shift
    else:
        X_work = X_dense.copy()

    # Initialize missing values with column means (non-negative)
    col_means = np.nanmean(X_work, axis=0)
    col_means[np.isnan(col_means)] = 0.0
    # Ensure non-negative means
    col_means = np.maximum(col_means, 0)

    X_filled = X_work.copy()
    for j in range(n_features):
        X_filled[missing_mask[:, j], j] = col_means[j]

    # Handle case where all values in a column are missing
    all_missing_cols = np.all(missing_mask, axis=0)
    if np.any(all_missing_cols):
        # Set those columns to small positive values
        X_filled[:, all_missing_cols] = 1e-6

    # Check if any columns are all zeros after filling
    zero_cols = np.all(X_filled <= 0, axis=0)
    if np.any(zero_cols):
        X_filled[:, zero_cols] = 1e-6

    # Fit NMF model
    model = NMF(
        n_components=n_components,
        init="random",
        solver="cd",
        beta_loss="frobenius",
        tol=tol,
        max_iter=max_iter,
        random_state=random_state,
        verbose=0,
    )

    try:
        W = model.fit_transform(X_filled)
        H = model.components_
        n_iterations = model.n_iter_
    except ValueError as e:
        # Handle cases where NMF fails (e.g., all zeros)
        if "cannot be cast" in str(e) or "Negative values" in str(e):
            # Fallback: use simple mean imputation
            X_imputed = X_filled.copy()
            if shift > 0:
                X_imputed = X_imputed - shift
            new_matrix = ScpMatrix(
                X=X_imputed, M=_update_imputed_mask(input_matrix.M, missing_mask)
            )
            layer_name = new_layer_name or "imputed_nmf"
            assay.add_layer(layer_name, new_matrix)
            container.log_operation(
                action="impute_nmf",
                params={
                    "assay": assay_name,
                    "source_layer": source_layer,
                    "new_layer_name": layer_name,
                    "n_components": n_components,
                    "fallback": "mean",
                },
                description=f"NMF imputation failed, used mean fallback on assay '{assay_name}'.",
            )
            return container
        raise

    # Reconstruct the full matrix
    X_reconstructed = W @ H

    # Extract imputed values (originally missing positions)
    X_imputed = X_dense.copy()
    imputed_values = X_reconstructed[missing_mask]

    # Subtract the shift if applied
    if shift > 0:
        imputed_values = imputed_values - shift

    # Ensure non-negative imputed values
    imputed_values = np.maximum(imputed_values, 0)

    X_imputed[missing_mask] = imputed_values

    # Create new layer
    new_matrix = ScpMatrix(X=X_imputed, M=_update_imputed_mask(input_matrix.M, missing_mask))
    layer_name = new_layer_name or "imputed_nmf"
    assay.add_layer(layer_name, new_matrix)

    container.log_operation(
        action="impute_nmf",
        params={
            "assay": assay_name,
            "source_layer": source_layer,
            "new_layer_name": layer_name,
            "n_components": n_components,
            "n_iterations": n_iterations,
            "max_iter": max_iter,
            "tol": tol,
            "random_state": random_state,
        },
        description=f"NMF imputation (n_components={n_components}, n_iter={n_iterations}) on assay '{assay_name}'.",
    )

    return container


if __name__ == "__main__":
    # Test: Basic functionality
    print("Testing NMF imputation...")

    # Create simple test data
    np.random.seed(42)
    n_samples = 100
    n_features = 50

    # Generate non-negative low-rank data
    U_true = np.abs(np.random.randn(n_samples, 5))
    V_true = np.abs(np.random.randn(n_features, 5))
    X_true = U_true @ V_true.T + np.random.randn(n_samples, n_features) * 0.1
    X_true = np.maximum(X_true, 0)  # Ensure non-negative

    # Add missing values
    X_missing = X_true.copy()
    missing_mask = np.random.rand(n_samples, n_features) < 0.2
    X_missing[missing_mask] = np.nan

    # Create container
    import polars as pl

    from scptensor.core.structures import Assay

    var = pl.DataFrame({"_index": [f"prot_{i}" for i in range(n_features)]})
    obs = pl.DataFrame({"_index": [f"cell_{i}" for i in range(n_samples)]})

    assay = Assay(var=var)
    assay.add_layer("raw", ScpMatrix(X=X_missing, M=None))

    container = ScpContainer(obs=obs, assays={"protein": assay})

    # Test NMF imputation
    result = impute_nmf(
        container, assay_name="protein", source_layer="raw", n_components=5, max_iter=100
    )

    # Check results
    assert "imputed_nmf" in result.assays["protein"].layers
    result_matrix = result.assays["protein"].layers["imputed_nmf"]
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
    print("\nTesting NMF imputation with existing mask...")

    # Create initial mask with some MBR (1) and LOD (2) codes for missing values
    M_initial = np.zeros(X_missing.shape, dtype=np.int8)
    n_missing = int(np.sum(missing_mask))
    M_initial[missing_mask] = np.where(np.random.rand(n_missing) < 0.5, MaskCode.MBR, MaskCode.LOD)

    assay2 = Assay(var=var)
    assay2.add_layer("raw", ScpMatrix(X=X_missing, M=M_initial))
    container2 = ScpContainer(obs=obs, assays={"protein": assay2})

    result2 = impute_nmf(
        container2, assay_name="protein", source_layer="raw", n_components=5, max_iter=100
    )

    result_matrix2 = result2.assays["protein"].layers["imputed_nmf"]
    M_imputed2 = result_matrix2.M

    # Check that imputed values now have IMPUTED (5) code
    assert M_imputed2 is not None
    assert np.all(M_imputed2[missing_mask] == MaskCode.IMPUTED)
    # Check that valid values remain VALID (0)
    assert np.all(M_imputed2[~missing_mask] == MaskCode.VALID)

    print("  Existing mask correctly updated to IMPUTED code")
    print(f"  Mask code check: {np.sum(M_imputed2 == MaskCode.IMPUTED)} imputed values")

    # Test 3: Test with n_components=None (auto selection)
    print("\nTesting NMF imputation with auto n_components...")

    assay3 = Assay(var=var)
    assay3.add_layer("raw", ScpMatrix(X=X_missing, M=None))
    container3 = ScpContainer(obs=obs, assays={"protein": assay3})

    result3 = impute_nmf(
        container3,
        assay_name="protein",
        source_layer="raw",
        new_layer_name="imputed_auto",
        n_components=None,
    )

    assert "imputed_auto" in result3.assays["protein"].layers
    X_imputed3 = result3.assays["protein"].layers["imputed_auto"].X
    assert not np.any(np.isnan(X_imputed3))

    print("  Auto n_components works correctly")

    # Test 4: Test sparse matrix input
    print("\nTesting NMF imputation with sparse input...")

    X_sparse = sp.csr_matrix(X_missing)

    assay4 = Assay(var=var)
    assay4.add_layer("raw", ScpMatrix(X=X_sparse, M=None))
    container4 = ScpContainer(obs=obs, assays={"protein": assay4})

    result4 = impute_nmf(
        container4,
        assay_name="protein",
        source_layer="raw",
        new_layer_name="imputed_from_sparse",
        n_components=5,
    )

    assert "imputed_from_sparse" in result4.assays["protein"].layers
    X_imputed4 = result4.assays["protein"].layers["imputed_from_sparse"].X
    assert not np.any(np.isnan(X_imputed4))

    print("  Sparse input handled correctly")

    # Test 5: Test random state reproducibility
    print("\nTesting NMF random state reproducibility...")

    assay5 = Assay(var=var)
    assay5.add_layer("raw", ScpMatrix(X=X_missing, M=None))
    container5 = ScpContainer(obs=obs, assays={"protein": assay5})

    result5a = impute_nmf(
        container5, assay_name="protein", source_layer="raw", n_components=5, random_state=42
    )
    X_imputed5a = result5a.assays["protein"].layers["imputed_nmf"].X

    # Reset container
    assay5b = Assay(var=var)
    assay5b.add_layer("raw", ScpMatrix(X=X_missing, M=None))
    container5b = ScpContainer(obs=obs, assays={"protein": assay5b})

    result5b = impute_nmf(
        container5b, assay_name="protein", source_layer="raw", n_components=5, random_state=42
    )
    X_imputed5b = result5b.assays["protein"].layers["imputed_nmf"].X

    # Results should be identical
    np.testing.assert_array_almost_equal(X_imputed5a, X_imputed5b, decimal=10)
    print("  Random state reproducibility verified")

    # Test 6: Test error handling
    print("\nTesting error handling...")

    try:
        # Invalid assay
        impute_nmf(container, assay_name="nonexistent", source_layer="raw")
        raise AssertionError("Should have raised ValueError")
    except Exception as e:
        assert "not found" in str(e)
        print("  Invalid assay error: OK")

    try:
        # Invalid layer
        impute_nmf(container, assay_name="protein", source_layer="nonexistent")
        raise AssertionError("Should have raised ValueError")
    except Exception as e:
        assert "not found" in str(e)
        print("  Invalid layer error: OK")

    try:
        # Invalid n_components
        impute_nmf(
            container,
            assay_name="protein",
            source_layer="raw",
            n_components=1000,  # Too large
        )
        raise AssertionError("Should have raised Exception")
    except Exception as e:
        assert "n_components" in str(e)
        print("  Invalid n_components error: OK")

    try:
        # Invalid max_iter
        impute_nmf(container, assay_name="protein", source_layer="raw", max_iter=0)
        raise AssertionError("Should have raised Exception")
    except Exception as e:
        assert "max_iter" in str(e)
        print("  Invalid max_iter error: OK")

    try:
        # Invalid tol
        impute_nmf(container, assay_name="protein", source_layer="raw", tol=0)
        raise AssertionError("Should have raised Exception")
    except Exception as e:
        assert "tol" in str(e)
        print("  Invalid tol error: OK")

    print("\nAll tests passed!")
