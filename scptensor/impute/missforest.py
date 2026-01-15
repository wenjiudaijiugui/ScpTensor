"""MissForest imputation for single-cell proteomics data."""

from typing import overload

import numpy as np
import sklearn.ensemble

from scptensor.core.exceptions import (
    AssayNotFoundError,
    LayerNotFoundError,
    ScpValueError,
)
from scptensor.core.jit_ops import impute_missing_with_col_means_jit
from scptensor.core.structures import ScpContainer, ScpMatrix
from scptensor.impute._utils import _update_imputed_mask


@overload
def missforest(
    container: ScpContainer,
    assay_name: str,
    source_layer: str,
    new_layer_name: str = "imputed_missforest",
    max_iter: int = 10,
    n_estimators: int = 100,
    max_depth: int | None = None,
    n_jobs: int = -1,
    random_state: int = 42,
    verbose: int = 0,
) -> ScpContainer: ...


def missforest(
    container: ScpContainer,
    assay_name: str,
    source_layer: str,
    new_layer_name: str = "imputed_missforest",
    max_iter: int = 10,
    n_estimators: int = 100,
    max_depth: int | None = None,
    n_jobs: int = -1,
    random_state: int = 42,
    verbose: int = 0,
) -> ScpContainer:
    """
    Impute missing values using MissForest (Random Forest imputation).

    Iteratively trains Random Forest on observed values to predict
    missing values for each feature.

    Parameters
    ----------
    container : ScpContainer
        Input container with missing values.
    assay_name : str
        Name of the assay to use.
    source_layer : str, default "raw"
        Name of the layer containing data with missing values.
    new_layer_name : str, default "imputed_missforest"
        Name for the new layer with imputed data.
    max_iter : int, default 10
        Maximum iterations.
    n_estimators : int, default 100
        Number of trees in forest.
    max_depth : int, optional
        Maximum tree depth.
    n_jobs : int, default -1
        Parallel jobs.
    random_state : int, default 42
        Random seed.
    verbose : int, default 0
        Verbosity level.

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

    Examples
    --------
    >>> from scptensor import missforest
    >>> result = missforest(container, "proteins", n_estimators=50)
    >>> "imputed_missforest" in result.assays["proteins"].layers
    True
    """
    # Validate parameters
    if max_iter <= 0:
        raise ScpValueError(
            f"max_iter must be positive, got {max_iter}. "
            "Use max_iter >= 1 for MissForest iterations.",
            parameter="max_iter",
            value=max_iter,
        )
    if n_estimators <= 0:
        raise ScpValueError(
            f"n_estimators must be positive, got {n_estimators}. "
            "Use n_estimators >= 1 for number of trees in the forest.",
            parameter="n_estimators",
            value=n_estimators,
        )
    if max_depth is not None and max_depth <= 0:
        raise ScpValueError(
            f"max_depth must be positive or None, got {max_depth}. "
            "Use max_depth >= 1 or None for unlimited tree depth.",
            parameter="max_depth",
            value=max_depth,
        )
    if verbose not in (0, 1, 2):
        raise ScpValueError(
            f"verbose must be 0, 1, or 2, got {verbose}. "
            "Use verbose=0 for silent, verbose=1 for progress updates, verbose=2 for detailed output.",
            parameter="verbose",
            value=verbose,
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

    input_matrix = assay.layers[source_layer]
    X_original = input_matrix.X

    # Check for missing values
    missing_mask_original = np.isnan(X_original)
    if not np.any(missing_mask_original):
        if verbose > 0:
            print("No missing values found. Returning original data.")
        new_matrix = ScpMatrix(
            X=X_original.copy(),
            M=_update_imputed_mask(input_matrix.M, missing_mask_original),
        )
        assay.add_layer(new_layer_name or "imputed_missforest", new_matrix)
        return container

    # Initialize with mean imputation
    X_in = X_original.copy()
    impute_missing_with_col_means_jit(X_in)
    missing_mask = np.isnan(X_in)

    # Sort columns by missing count (fewest missing first)
    missing_counts = missing_mask.sum(axis=0)
    sorted_indices = [idx for idx in np.argsort(missing_counts) if missing_counts[idx] > 0]

    X_old = X_in.copy()
    gamma = 0.0

    # Iterative imputation
    for iteration in range(max_iter):
        if verbose > 0:
            print(f"MissForest iteration {iteration + 1}/{max_iter}")

        for col_idx in sorted_indices:
            obs_rows = ~missing_mask[:, col_idx]
            mis_rows = missing_mask[:, col_idx]

            if not np.any(mis_rows):
                continue

            # Prepare training data (excluding current column)
            X_train = np.delete(X_in[obs_rows, :], col_idx, axis=1)
            y_train = X_in[obs_rows, col_idx]
            X_test = np.delete(X_in[mis_rows, :], col_idx, axis=1)

            # Train and predict
            rf = sklearn.ensemble.RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                n_jobs=n_jobs,
                random_state=random_state,
            )
            rf.fit(X_train, y_train)
            X_in[mis_rows, col_idx] = rf.predict(X_test)

        # Check convergence
        diff = np.sum((X_in[missing_mask] - X_old[missing_mask]) ** 2)
        norm = np.sum(X_in[missing_mask] ** 2)
        gamma = diff / (norm + 1e-9)

        if verbose > 0:
            print(f"  Difference (gamma): {gamma:.6f}")

        if gamma < 1e-4:
            if verbose > 0:
                print("Converged.")
            break

        X_old = X_in.copy()

    # Create new layer
    new_matrix = ScpMatrix(X=X_in, M=_update_imputed_mask(input_matrix.M, missing_mask_original))
    layer_name = new_layer_name or "imputed_missforest"
    assay.add_layer(layer_name, new_matrix)

    container.log_operation(
        action="impute_missforest",
        params={
            "assay": assay_name,
            "source_layer": source_layer,
            "new_layer_name": layer_name,
            "max_iter": max_iter,
            "n_estimators": n_estimators,
            "gamma": gamma,
        },
        description=f"MissForest imputation on assay '{assay_name}'.",
    )

    return container


if __name__ == "__main__":
    # Test: Basic functionality
    print("Testing MissForest imputation...")

    # Create simple test data
    np.random.seed(42)
    n_samples = 100
    n_features = 20

    # Generate correlated data
    X_true = np.random.randn(n_samples, n_features)

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

    # Test MissForest imputation
    result = missforest(
        container, assay_name="protein", source_layer="raw", max_iter=5, n_estimators=50, verbose=1
    )

    # Check results
    assert "imputed_missforest" in result.assays["protein"].layers
    X_imputed = result.assays["protein"].layers["imputed_missforest"].X

    # Check no NaNs
    assert not np.any(np.isnan(X_imputed))

    # Check imputation accuracy on missing values
    imputed_values = X_imputed[missing_mask]
    true_values = X_true[missing_mask]
    mae = np.mean(np.abs(imputed_values - true_values))
    correlation = np.corrcoef(imputed_values, true_values)[0, 1]

    print(f"  Imputation MAE: {mae:.3f}")
    print(f"  Imputation correlation: {correlation:.3f}")
    print(f"  Shape: {X_imputed.shape}")
    print(f"  NaN count: {np.sum(np.isnan(X_imputed))}")
    print("✅ All tests passed")
