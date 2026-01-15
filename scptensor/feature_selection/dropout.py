"""Dropout-based feature selection for single-cell proteomics data.

This module filters features based on their dropout rate (proportion of
missing/zero values), which is particularly useful for single-cell data
where many features may have high rates of missing values.
"""

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from scipy import sparse

from scptensor.core.structures import Assay, ScpContainer
from scptensor.feature_selection._shared import (
    _validate_assay_layer,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


def select_by_dropout(
    container: ScpContainer,
    assay_name: str = "protein",
    layer: str = "raw",
    max_dropout_rate: float = 0.5,
    min_detected: int = 3,
    subset: bool = False,
) -> ScpContainer:
    """Select features based on dropout rate.

    This function filters features by their proportion of missing/zero values.
    Features with dropout rates exceeding the threshold are flagged or removed.

    Parameters
    ----------
    container : ScpContainer
        The ScpContainer object containing the data.
    assay_name : str, default="protein"
        Name of the assay to process.
    layer : str, default="raw"
        Layer to use for dropout calculation.
    max_dropout_rate : float, default=0.5
        Maximum allowable dropout rate for a feature. Features with dropout
        rate above this threshold will be excluded. Must be in [0, 1].
    min_detected : int, default=3
        Minimum number of samples where the feature must be detected (non-zero).
    subset : bool, default=False
        If True, returns a container with only features passing the filter.
        If False, adds 'pass_dropout_filter' column to var.

    Returns
    -------
    ScpContainer
        Container with dropout-filtered features.

    Raises
    ------
    ValueError
        If assay or layer not found, max_dropout_rate is not in [0, 1],
        or no features pass the filter.

    Examples
    --------
    >>> # Keep only features with dropout rate < 50%
    >>> container_filtered = select_by_dropout(container, max_dropout_rate=0.5)
    >>> # Keep features detected in at least 10 samples
    >>> container_filtered = select_by_dropout(container, min_detected=10)
    >>> # Annotate features without subsetting
    >>> container = select_by_dropout(container, subset=False)
    >>> passing = container.assays['protein'].var['pass_dropout_filter'].sum()
    """
    if not 0 <= max_dropout_rate <= 1:
        raise ValueError(f"max_dropout_rate must be in [0, 1], got {max_dropout_rate}")

    assay = _validate_assay_layer(container, assay_name, layer)
    X = assay.layers[layer].X
    M = assay.layers[layer].M

    # Convert to dense if sparse
    if sparse.issparse(X):
        X = X.toarray()
    if M is not None and sparse.issparse(M):
        M = M.toarray()

    # Compute dropout statistics
    n_detected, dropout_rate = _compute_dropout_with_intensity(X, M)

    # Determine which features pass
    pass_filter = (dropout_rate <= max_dropout_rate) & (n_detected >= min_detected)
    passing_indices = np.where(pass_filter)[0]

    if subset:
        if len(passing_indices) == 0:
            raise ValueError(
                f"No features pass the dropout filter. "
                f"Consider relaxing max_dropout_rate ({max_dropout_rate}) "
                f"or min_detected ({min_detected})."
            )
        top_indices = passing_indices
        bool_mask = None  # Not used in subset mode
    else:
        # For annotation mode, mark all features with pass/fail
        top_indices = None  # Not used
        bool_mask = pass_filter

    return _subset_or_annotate_dropout(
        container=container,
        assay_name=assay_name,
        assay=assay,
        top_indices=top_indices,
        bool_mask=bool_mask,
        subset=subset,
        action="select_by_dropout",
        params={
            "assay_name": assay_name,
            "layer": layer,
            "max_dropout_rate": max_dropout_rate,
            "min_detected": min_detected,
            "subset": subset,
        },
    )


def _compute_dropout_with_intensity(
    X: "NDArray[np.float64]",
    M: "NDArray[np.float64] | None",
) -> tuple["NDArray[np.int64]", "NDArray[np.float64]"]:
    """Compute detection counts and dropout rate per feature.

    Parameters
    ----------
    X : ndarray
        Data matrix (n_samples, n_features).
    M : ndarray or None
        Mask matrix. Values with mask != 0 are invalid.

    Returns
    -------
    n_detected : ndarray
        Number of detections per feature.
    dropout_rate : ndarray
        Dropout rate per feature.
    """
    n_samples = X.shape[0]

    # Detected: non-zero AND (mask valid OR no mask) AND not NaN
    detected_mask = (X != 0) & ~np.isnan(X)
    if M is not None:
        detected_mask &= M == 0

    n_detected = np.sum(detected_mask, axis=0)
    dropout_rate = 1 - (n_detected / n_samples)
    return n_detected.astype(np.int64), dropout_rate


def _subset_or_annotate_dropout(
    container: "ScpContainer",
    assay_name: str,
    assay: "Assay",
    top_indices: "NDArray[np.int64] | None",
    bool_mask: "NDArray[np.bool_] | None",
    subset: bool,
    action: str,
    params: dict,
) -> "ScpContainer":
    """Either subset features or annotate var for dropout filtering.

    Parameters
    ----------
    container : ScpContainer
        Input container.
    assay_name : str
        Name of the assay.
    assay : Assay
        The assay object.
    top_indices : ndarray or None
        Indices of selected features (for subset mode).
    bool_mask : ndarray or None
        Boolean mask of passing features (for annotation mode).
    subset : bool
        If True, subset features; else annotate.
    action : str
        Action name for logging.
    params : dict
        Parameters for logging.

    Returns
    -------
    ScpContainer
        Modified container.
    """
    if subset:
        assert top_indices is not None
        new_container = container.filter_features(assay_name, feature_indices=top_indices)
        n_pass = len(top_indices)
        description = f"Selected {n_pass}/{assay.n_features} features."
    else:
        assert bool_mask is not None
        # Add annotation to var with additional columns
        new_var = assay.var.with_columns(
            pl.Series("pass_dropout_filter", bool_mask),
        )

        new_assay = assay.__class__(var=new_var, layers=assay.layers)
        new_assays = dict(container.assays)
        new_assays[assay_name] = new_assay

        new_container = container.__class__(
            obs=container.obs, assays=new_assays, history=list(container.history)
        )
        n_pass = np.sum(bool_mask)
        description = f"Annotated {n_pass}/{assay.n_features} features passing filter."

    new_container.log_operation(
        action=action,
        params=params,
        description=description,
    )
    return new_container


def get_dropout_stats(
    container: ScpContainer,
    assay_name: str = "protein",
    layer: str = "raw",
) -> pl.DataFrame:
    """Calculate dropout statistics for all features.

    This function computes summary statistics about the dropout rate
    across all features in an assay layer.

    Parameters
    ----------
    container : ScpContainer
        The ScpContainer object.
    assay_name : str, default="protein"
        Name of the assay.
    layer : str, default="raw"
        Layer to analyze.

    Returns
    -------
    pl.DataFrame
        DataFrame with dropout statistics for each feature:
        - dropout_rate: Proportion of samples with missing/zero values
        - n_detected: Number of samples with detected values
        - n_missing: Number of samples with missing values
        - mean_intensity: Mean intensity (only detected values)

    Raises
    ------
    ValueError
        If assay or layer not found.

    Examples
    --------
    >>> stats = get_dropout_stats(container)
    >>> print(stats.filter(pl.col("dropout_rate") < 0.5))
    """
    assay = _validate_assay_layer(container, assay_name, layer)
    X = assay.layers[layer].X
    M = assay.layers[layer].M

    # Convert to dense if sparse
    if sparse.issparse(X):
        X = X.toarray()
    if M is not None and sparse.issparse(M):
        M = M.toarray()

    n_samples = X.shape[0]
    X.shape[1]

    # Compute detection mask
    detected_mask = (X != 0) & ~np.isnan(X)
    if M is not None:
        detected_mask &= M == 0

    # Vectorized mean intensity calculation
    # Replace non-detected with NaN for mean calculation
    X_for_mean = X.copy()
    X_for_mean[~detected_mask] = np.nan
    mean_intensity = np.nanmean(X_for_mean, axis=0)

    n_detected = np.sum(detected_mask, axis=0)
    dropout_rate = 1 - (n_detected / n_samples)

    return pl.DataFrame(
        {
            "dropout_rate": dropout_rate,
            "n_detected": n_detected,
            "n_missing": n_samples - n_detected,
            "mean_intensity": mean_intensity,
        }
    )


if __name__ == "__main__":
    print("Running dropout-based feature selection tests...")

    # Create test data with varying dropout rates
    n_samples, n_features = 100, 50
    np.random.seed(42)
    X_test = np.random.randn(n_samples, n_features)

    # Introduce dropouts: first 10 features high (>50%), next 10 medium (~30%)
    X_test[:60, :10] = 0
    X_test[:30, 10:20] = 0
    X_test[:5, 20:] = 0
    X_test[10:15, 5:10] = np.nan

    from scptensor.core.structures import Assay, ScpContainer, ScpMatrix

    var_test = pl.DataFrame({"_index": [f"feature_{i}" for i in range(n_features)]})
    obs_test = pl.DataFrame({"_index": [f"sample_{i}" for i in range(n_samples)]})

    assay_test = Assay(
        var=var_test,
        layers={"raw": ScpMatrix(X=X_test, M=None)},
        feature_id_col="_index",
    )
    container_test = ScpContainer(obs=obs_test, assays={"protein": assay_test})

    # Test 1: Basic dropout filtering
    print("\nTest 1: Dropout filtering (max_rate=0.4)")
    result = select_by_dropout(container_test, max_dropout_rate=0.4, subset=True)
    assert result.assays["protein"].n_features < n_features
    print(f"  Passed: {result.assays['protein'].n_features} features remaining")

    # Test 2: min_detected constraint
    print("\nTest 2: min_detected=50")
    result2 = select_by_dropout(container_test, min_detected=50, subset=True)
    assert result2.assays["protein"].n_features < n_features
    print(f"  Passed: {result2.assays['protein'].n_features} features remaining")

    # Test 3: Annotation mode
    print("\nTest 3: Annotation mode")
    result3 = select_by_dropout(container_test, max_dropout_rate=0.2, subset=False)
    assert "pass_dropout_filter" in result3.assays["protein"].var.columns
    n_pass = result3.assays["protein"].var["pass_dropout_filter"].sum()
    assert n_pass < n_features, f"Expected some features to fail, got {n_pass}/{n_features}"
    print(f"  Passed: {n_pass} features annotated")

    # Test 4: Dropout stats
    print("\nTest 4: Dropout statistics")
    stats = get_dropout_stats(container_test)
    assert stats.shape[0] == n_features
    assert "dropout_rate" in stats.columns
    print(f"  Mean dropout rate: {stats['dropout_rate'].mean():.3f}")

    # Test 5: Edge case - no features pass
    print("\nTest 5: Strict filter (should error)")
    try:
        select_by_dropout(container_test, max_dropout_rate=0.01, min_detected=200, subset=True)
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        print("  Passed: Correctly raised ValueError")

    # Test 6: Invalid max_dropout_rate
    print("\nTest 6: Invalid parameter")
    try:
        select_by_dropout(container_test, max_dropout_rate=1.5)
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        print("  Passed: Correctly raised ValueError")

    print("\n" + "=" * 50)
    print("All dropout tests passed!")
    print("=" * 50)
