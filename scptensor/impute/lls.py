"""Local Least Squares imputation for single-cell proteomics data.

Reference:
    Kim H, et al. BMC Bioinformatics 2008;9:72.
    Missing value estimation for DNA microarray gene expression data:
    Local least squares imputation.
"""

from typing import overload

import numpy as np
from sklearn.neighbors import NearestNeighbors

from scptensor.core.exceptions import (
    AssayNotFoundError,
    LayerNotFoundError,
    ScpValueError,
)
from scptensor.core.structures import ScpContainer, ScpMatrix
from scptensor.impute._utils import _update_imputed_mask


@overload
def impute_lls(
    container: ScpContainer,
    assay_name: str,
    source_layer: str,
    new_layer_name: str = "imputed_lls",
    k: int = 10,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> ScpContainer: ...


def impute_lls(
    container: ScpContainer,
    assay_name: str,
    source_layer: str,
    new_layer_name: str = "imputed_lls",
    k: int = 10,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> ScpContainer:
    """
    Impute missing values using Local Least Squares.

    This method combines K-nearest neighbors with local linear regression
    for improved accuracy in high-dimensional data with correlated features.
    For each sample with missing values, it finds K nearest neighbors using
    complete features, then builds a local linear model to predict missing
    values. The process iterates until convergence or max_iter is reached.

    Parameters
    ----------
    container : ScpContainer
        Input container with missing values.
    assay_name : str
        Name of the assay to use.
    source_layer : str
        Name of the layer containing data with missing values.
    new_layer_name : str, default "imputed_lls"
        Name for the new layer with imputed data.
    k : int, default 10
        Number of nearest neighbors to use for local regression.
    max_iter : int, default 100
        Maximum iterations for convergence.
    tol : float, default 1e-6
        Convergence threshold for relative change in imputed values.

    Returns
    -------
    ScpContainer
        Container with imputed data in new layer.

    Raises
    ------
    AssayNotFoundError
        If the specified assay does not exist.
    LayerNotFoundError
        If the specified layer does not exist.
    ScpValueError
        If parameters are invalid.

    Notes
    -----
    The LLS algorithm works as follows:
        1. For each sample with missing values, find K nearest neighbors
           using only features that are observed in the target sample.
        2. Build a local linear regression model using the neighbor's
           complete feature values to predict the target's missing values.
        3. Iteratively re-impute until convergence or max_iter.

    Time complexity: O(max_iter * N * k * D) where N is number of samples,
    k is number of neighbors, and D is number of features.

    LLS ranks among top methods in multiple benchmarks for proteomics
    data imputation (Jin 2021). Particularly effective for high-dimensional
    correlated data.

    Examples
    --------
    >>> from scptensor import impute_lls
    >>> result = impute_lls(container, "proteins", k=10)
    >>> "imputed_lls" in result.assays["proteins"].layers
    True

    References
    ----------
    .. [1] Kim H, et al. "Missing value estimation for DNA microarray
       gene expression data: Local least squares imputation."
       BMC Bioinformatics 2008;9:72.
    .. [2] Jin S, et al. "A comparative study of evaluating missing value
       imputation methods in label-free proteomics." Sci Rep 2021;11:16409.
    """
    # Parameter validation
    if k <= 0:
        raise ScpValueError(
            f"Number of neighbors (k) must be positive, got {k}. "
            "Use k >= 1 for nearest neighbor imputation.",
            parameter="k",
            value=k,
        )
    if max_iter <= 0:
        raise ScpValueError(
            f"max_iter must be positive, got {max_iter}. "
            "Use max_iter >= 1 for iterative imputation.",
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
    if hasattr(X_original, "toarray"):
        X_original = X_original.toarray()

    n_samples, n_features = X_original.shape

    # Check for missing values
    missing_mask = np.isnan(X_original)
    if not np.any(missing_mask):
        new_matrix = ScpMatrix(X=X_original, M=_update_imputed_mask(input_matrix.M, missing_mask))
        layer_name = new_layer_name or "imputed_lls"
        assay.add_layer(layer_name, new_matrix)
        container.log_operation(
            action="impute_lls",
            params={
                "assay": assay_name,
                "source_layer": source_layer,
                "new_layer_name": layer_name,
                "k": k,
                "n_iterations": 0,
            },
            description=f"LLS imputation on assay '{assay_name}': no missing values found.",
        )
        return container

    # Initialize with column means
    X = X_original.copy()
    col_means = np.nanmean(X_original, axis=0)
    col_means[np.isnan(col_means)] = 0.0

    # Initial fill for algorithm to work
    for j in range(n_features):
        X[missing_mask[:, j], j] = col_means[j]

    # Iterative imputation
    prev_X_imputed = None
    iterations = 0

    for iteration in range(max_iter):
        prev_X_imputed = X.copy()
        samples_with_missing = np.where(missing_mask.any(axis=1))[0]

        for i in samples_with_missing:
            sample_missing = missing_mask[i]
            sample_observed = ~sample_missing

            if not sample_missing.any():
                continue

            # Find features that are observed in this sample
            observed_idx = np.where(sample_observed)[0]
            missing_idx = np.where(sample_missing)[0]

            if len(observed_idx) == 0:
                # All features missing - use column means
                X[i, missing_idx] = col_means[missing_idx]
                continue

            if len(missing_idx) == 0:
                continue

            # Find neighbors using observed features only
            k_search = min(k + 1, n_samples)  # +1 to exclude self

            # Use only observed features for neighbor search
            X_obs_features = X[:, observed_idx]

            # Handle case where observed features have NaN in other samples
            # Fill temporarily with column means for distance computation
            X_obs_features_clean = X_obs_features.copy()
            for feat_idx in range(X_obs_features_clean.shape[1]):
                nan_mask = np.isnan(X_obs_features_clean[:, feat_idx])
                if nan_mask.any():
                    X_obs_features_clean[nan_mask, feat_idx] = col_means[observed_idx[feat_idx]]

            nbrs = NearestNeighbors(n_neighbors=k_search, algorithm="auto").fit(
                X_obs_features_clean
            )
            distances, neighbor_indices = nbrs.kneighbors([X_obs_features_clean[i]])

            # Remove self from neighbors
            mask_not_self = neighbor_indices[0] != i
            neighbor_indices = neighbor_indices[0][mask_not_self]
            distances = distances[0][mask_not_self]

            # Limit to k neighbors
            if len(neighbor_indices) > k:
                neighbor_indices = neighbor_indices[:k]
                distances = distances[:k]

            if len(neighbor_indices) == 0:
                # No neighbors found - use column means
                X[i, missing_idx] = col_means[missing_idx]
                continue

            # Build local linear model for each missing feature
            for j in missing_idx:
                # For predicting feature j, use neighbors that have this feature
                neighbor_values = X[neighbor_indices, j]

                # Check which neighbors have valid values for this feature
                valid_neighbor_mask = np.isfinite(neighbor_values)

                if not valid_neighbor_mask.any():
                    # No valid neighbors - use column mean
                    X[i, j] = col_means[j]
                    continue

                valid_neighbors = neighbor_indices[valid_neighbor_mask]
                valid_neighbor_values = neighbor_values[valid_neighbor_mask]

                # Use observed features as predictors
                # For the target sample, get its observed feature values
                target_obs_values = X[i, observed_idx]

                # Build regression matrix Z: neighbors x observed_features
                # Add intercept column
                n_valid_neighbors = len(valid_neighbors)
                n_obs_features = len(observed_idx)

                if n_valid_neighbors <= n_obs_features:
                    # Not enough neighbors for regression - use weighted average
                    weights = 1.0 / (distances[valid_neighbor_mask] + 1e-10)
                    weights /= weights.sum()
                    X[i, j] = np.dot(valid_neighbor_values, weights)
                    continue

                # Build design matrix for regression
                # Z: observed feature values for valid neighbors
                Z = np.zeros((n_valid_neighbors, n_obs_features + 1))
                Z[:, 0] = 1.0  # Intercept
                Z[:, 1:] = X[valid_neighbors][:, observed_idx]

                # Response: target feature values for valid neighbors
                y = valid_neighbor_values

                try:
                    # Solve least squares: Z @ beta = y
                    beta, _, _, _ = np.linalg.lstsq(Z, y, rcond=None)

                    # Predict for target sample
                    target_design = np.concatenate([[1.0], target_obs_values])
                    X[i, j] = np.dot(target_design, beta)

                except np.linalg.LinAlgError:
                    # Fallback to weighted average if regression fails
                    weights = 1.0 / (distances[valid_neighbor_mask] + 1e-10)
                    weights /= weights.sum()
                    X[i, j] = np.dot(valid_neighbor_values, weights)

        # Check convergence
        if prev_X_imputed is not None:
            # Compute relative change only for originally missing values
            change = np.abs(X[missing_mask] - prev_X_imputed[missing_mask])
            max_change = np.max(change)
            mean_val = np.mean(np.abs(X[missing_mask])) + 1e-10
            relative_change = max_change / mean_val

            if relative_change < tol:
                iterations = iteration + 1
                break

        iterations = iteration + 1

    # Create new layer with updated mask
    new_matrix = ScpMatrix(X=X, M=_update_imputed_mask(input_matrix.M, missing_mask))
    layer_name = new_layer_name or "imputed_lls"
    assay.add_layer(layer_name, new_matrix)

    container.log_operation(
        action="impute_lls",
        params={
            "assay": assay_name,
            "source_layer": source_layer,
            "new_layer_name": layer_name,
            "k": k,
            "n_iterations": iterations,
        },
        description=f"LLS imputation (k={k}, iterations={iterations}) on assay '{assay_name}'.",
    )

    return container


if __name__ == "__main__":
    # Test: Basic functionality
    print("Testing LLS imputation...")

    # Create simple test data
    np.random.seed(42)
    n_samples = 100
    n_features = 50

    # Generate low-rank data (correlated features)
    U_true = np.random.randn(n_samples, 5)
    V_true = np.random.randn(n_features, 5)
    X_true = U_true @ V_true.T + np.random.randn(n_samples, n_features) * 0.1

    # Add missing values
    X_missing = X_true.copy()
    missing_mask = np.random.rand(n_samples, n_features) < 0.2
    X_missing[missing_mask] = np.nan

    # Create container
    import polars as pl

    from scptensor.core.structures import Assay, MaskCode

    var = pl.DataFrame({"_index": [f"prot_{i}" for i in range(n_features)]})
    obs = pl.DataFrame({"_index": [f"cell_{i}" for i in range(n_samples)]})

    assay = Assay(var=var)
    assay.add_layer("raw", ScpMatrix(X=X_missing, M=None))

    container = ScpContainer(obs=obs, assays={"protein": assay})

    # Test LLS imputation
    result = impute_lls(
        container,
        assay_name="protein",
        source_layer="raw",
        k=10,
        max_iter=10,
    )

    # Check results
    assert "imputed_lls" in result.assays["protein"].layers
    result_matrix = result.assays["protein"].layers["imputed_lls"]
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
    print("\nTesting LLS imputation with existing mask...")

    # Create initial mask with some MBR (1) and LOD (2) codes for missing values
    M_initial = np.zeros(X_missing.shape, dtype=np.int8)
    M_initial[missing_mask] = np.where(
        np.random.rand(np.sum(missing_mask)) < 0.5, MaskCode.MBR, MaskCode.LOD
    )

    assay2 = Assay(var=var)
    assay2.add_layer("raw", ScpMatrix(X=X_missing, M=M_initial))
    container2 = ScpContainer(obs=obs, assays={"protein": assay2})

    result2 = impute_lls(
        container2,
        assay_name="protein",
        source_layer="raw",
        k=10,
        max_iter=10,
    )

    result_matrix2 = result2.assays["protein"].layers["imputed_lls"]
    M_imputed2 = result_matrix2.M

    # Check that imputed values now have IMPUTED (5) code
    assert M_imputed2 is not None
    assert np.all(M_imputed2[missing_mask] == MaskCode.IMPUTED)
    # Check that valid values remain VALID (0)
    assert np.all(M_imputed2[~missing_mask] == MaskCode.VALID)

    print("  Existing mask correctly updated to IMPUTED code")
    print(f"  Mask code check: {np.sum(M_imputed2 == MaskCode.IMPUTED)} imputed values")
    print("All tests passed")
