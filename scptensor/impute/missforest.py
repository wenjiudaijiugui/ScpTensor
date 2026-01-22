"""MissForest imputation for single-cell proteomics data."""

import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import RandomForestRegressor

# Enable experimental IterativeImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from scptensor.core.exceptions import (
    AssayNotFoundError,
    LayerNotFoundError,
    ScpValueError,
)
from scptensor.core.jit_ops import impute_missing_with_col_means_jit
from scptensor.core.structures import ScpContainer, ScpMatrix
from scptensor.impute._utils import _update_imputed_mask


def impute_mf(
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

    Uses sklearn's IterativeImputer with RandomForestRegressor to iteratively
    train on observed values and predict missing values.

    Parameters
    ----------
    container : ScpContainer
        Input container with missing values.
    assay_name : str
        Name of the assay to use.
    source_layer : str
        Name of the layer containing data with missing values.
    new_layer_name : str, default "imputed_missforest"
        Name for the new layer with imputed data.
    max_iter : int, default 10
        Maximum iterations for iterative imputation.
    n_estimators : int, default 100
        Number of trees in the random forest.
    max_depth : int, optional
        Maximum tree depth. None for unlimited depth.
    n_jobs : int, default -1
        Number of parallel jobs. -1 uses all available cores.
    random_state : int, default 42
        Random seed for reproducibility.
    verbose : int, default 0
        Verbosity level. 0=silent, 1=progress, 2=detailed.

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
    >>> from scptensor import impute_mf
    >>> result = impute_mf(container, "proteins", n_estimators=50)
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

    # Convert sparse to dense for IterativeImputer
    if sp.issparse(X_original):
        X_in = X_original.toarray().copy()
    else:
        X_in = X_original.copy()

    # Initialize with mean imputation (for initial strategy consistency)
    impute_missing_with_col_means_jit(X_in)
    missing_mask = np.isnan(X_in)

    # Check if all values are missing (edge case)
    if np.all(missing_mask):
        if verbose > 0:
            print("Warning: All values are missing. Returning mean-imputed data.")
        X_imputed = X_in
    else:
        # Create IterativeImputer with RandomForestRegressor
        imputer = IterativeImputer(
            estimator=RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                n_jobs=n_jobs,
                random_state=random_state,
            ),
            max_iter=max_iter,
            initial_strategy="mean",
            verbose=bool(verbose > 0),
            random_state=random_state,
        )

        # Perform imputation
        try:
            if verbose > 0:
                print(f"Running MissForest imputation with max_iter={max_iter}...")
            X_imputed = imputer.fit_transform(X_in)
            if verbose > 0:
                print("Imputation completed.")
        except Exception as e:
            raise ScpValueError(
                f"IterativeImputer failed: {e}. "
                "This may occur if there are insufficient observed values for imputation. "
                "Check your data quality and consider using a simpler imputation method.",
                parameter="iterative_imputer",
                value=None,
            ) from e

    # Create new layer with updated mask
    new_matrix = ScpMatrix(
        X=X_imputed, M=_update_imputed_mask(input_matrix.M, missing_mask_original)
    )
    layer_name = new_layer_name or "imputed_missforest"
    assay.add_layer(layer_name, new_matrix)

    # Log operation
    container.log_operation(
        action="impute_missforest",
        params={
            "assay": assay_name,
            "source_layer": source_layer,
            "new_layer_name": layer_name,
            "max_iter": max_iter,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
        },
        description=f"MissForest imputation on assay '{assay_name}' using sklearn.impute.IterativeImputer.",
    )

    return container


if __name__ == "__main__":
    # Test: Basic functionality
    print("Testing MissForest imputation with sklearn.impute.IterativeImputer...")

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
    result = impute_mf(
        container,
        assay_name="protein",
        source_layer="raw",
        max_iter=5,
        n_estimators=50,
        verbose=1,
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
