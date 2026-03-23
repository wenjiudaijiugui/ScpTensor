"""
MissForest imputation (Random Forest based imputation).

Uses sklearn's IterativeImputer with RandomForestRegressor to iteratively
train on observed values and predict missing values.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from scptensor.core.exceptions import ScpValueError
from scptensor.core.jit_ops import impute_missing_with_col_means_jit
from scptensor.core.structures import ScpContainer
from scptensor.impute._utils import (
    add_imputed_layer,
    log_imputation_operation,
    preserve_observed_values,
    to_dense_float_copy,
)
from scptensor.impute.base import ImputeMethod, register_impute_method, validate_layer_context

# =============================================================================
# Core MissForest algorithm (pure function for registry)
# =============================================================================


def missforest_impute(
    data: np.ndarray,
    max_iter: int = 10,
    n_estimators: int = 100,
    max_depth: int | None = None,
    n_jobs: int = -1,
    random_state: int = 42,
) -> np.ndarray:
    """MissForest imputation (core algorithm).

    Parameters
    ----------
    data : np.ndarray
        Input data with missing values (NaN).
    max_iter : int, default=10
        Maximum iterations for iterative imputation.
    n_estimators : int, default=100
        Number of trees in the random forest.
    max_depth : int, optional
        Maximum tree depth.
    n_jobs : int, default=-1
        Number of parallel jobs.
    random_state : int, default=42
        Random seed.

    Returns
    -------
    np.ndarray
        Data with imputed values.
    """
    X = data.copy()
    missing_mask = np.isnan(X)

    if not np.any(missing_mask):
        return X
    if np.all(missing_mask):
        return np.zeros_like(X, dtype=np.float64)

    # Initialize with mean imputation
    impute_missing_with_col_means_jit(X)

    # Create IterativeImputer
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            random_state=random_state,
        ),
        max_iter=max_iter,
        initial_strategy="mean",
        random_state=random_state,
    )

    return imputer.fit_transform(X)


def validate_missforest(data: np.ndarray) -> bool:
    """Validate data for MissForest imputation."""
    return data.size > 0 and data.shape[0] > 1


# =============================================================================
# ScpTensor wrapper function
# =============================================================================


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
    max_iter : int, default=10
        Maximum iterations for iterative imputation.
    n_estimators : int, default=100
        Number of trees in the random forest.
    max_depth : int, optional
        Maximum tree depth.
    n_jobs : int, default=-1
        Number of parallel jobs.
    random_state : int, default=42
        Random seed.
    verbose : int, default=0
        Verbosity level.

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

    Examples
    --------
    >>> from scptensor import impute_mf
    >>> result = impute_mf(container, "proteins", "raw", n_estimators=50)
    >>> "imputed_missforest" in result.assays["proteins"].layers
    True
    """
    # Validate parameters
    if max_iter <= 0:
        raise ScpValueError(
            f"max_iter must be positive, got {max_iter}.",
            parameter="max_iter",
            value=max_iter,
        )
    if n_estimators <= 0:
        raise ScpValueError(
            f"n_estimators must be positive, got {n_estimators}.",
            parameter="n_estimators",
            value=n_estimators,
        )
    if verbose not in (0, 1, 2):
        raise ScpValueError(
            f"verbose must be 0, 1, or 2, got {verbose}.",
            parameter="verbose",
            value=verbose,
        )
    if max_depth is not None and max_depth <= 0:
        raise ScpValueError(
            f"max_depth must be positive, got {max_depth}.",
            parameter="max_depth",
            value=max_depth,
        )

    ctx = validate_layer_context(container, assay_name, source_layer)
    assay = ctx.assay
    input_matrix = ctx.layer
    X_in = to_dense_float_copy(input_matrix.X)
    missing_mask_original = np.isnan(X_in)
    layer_name = new_layer_name or "imputed_missforest"

    if not np.any(missing_mask_original):
        add_imputed_layer(assay, layer_name, X_in.copy(), input_matrix, missing_mask_original)
        return log_imputation_operation(
            container,
            action="impute_missforest",
            params={
                "assay": ctx.resolved_assay_name,
                "source_layer": source_layer,
                "new_layer_name": layer_name,
                "max_iter": max_iter,
                "n_estimators": n_estimators,
            },
            description=(
                f"MissForest imputation on assay '{ctx.resolved_assay_name}': "
                "no missing values found."
            ),
        )

    # Apply MissForest imputation
    try:
        if verbose > 0:
            print(f"Running MissForest imputation with max_iter={max_iter}...")
        X_imputed = missforest_impute(
            X_in,
            max_iter=max_iter,
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        if verbose > 0:
            print("Imputation completed.")
    except Exception as e:
        raise ScpValueError(
            f"IterativeImputer failed: {e}",
            parameter="iterative_imputer",
        ) from e
    preserve_observed_values(X_imputed, X_in, missing_mask_original)

    # Create new layer
    add_imputed_layer(assay, layer_name, X_imputed, input_matrix, missing_mask_original)

    # Log operation
    return log_imputation_operation(
        container,
        action="impute_missforest",
        params={
            "assay": ctx.resolved_assay_name,
            "source_layer": source_layer,
            "new_layer_name": layer_name,
            "max_iter": max_iter,
            "n_estimators": n_estimators,
        },
        description=f"MissForest imputation on assay '{ctx.resolved_assay_name}'.",
    )


register_impute_method(
    ImputeMethod(
        name="missforest",
        supports_sparse=False,
        validate=validate_missforest,
        apply=impute_mf,
    )
)
