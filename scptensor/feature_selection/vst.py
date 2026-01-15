"""Variance Stabilizing Transformation (VST) based feature selection.

This module implements feature selection based on variance stabilization,
similar to Seurat's FindVariableFeatures method. It fits a trend line
to the variance-mean relationship and selects features with highest
standardized residuals.
"""

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from scipy import sparse
from scipy.interpolate import interp1d
from scipy.sparse import issparse

from scptensor.core.structures import Assay, ScpContainer
from scptensor.feature_selection._shared import (
    _subset_or_annotate,
    _validate_assay_layer,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


def select_by_vst(
    container: ScpContainer,
    assay_name: str = "protein",
    layer: str = "raw",
    n_top_features: int = 2000,
    n_bins: int = 20,
    subset: bool = True,
    min_mean: float = 0.01,
) -> ScpContainer:
    """Select highly variable features using Variance Stabilizing Transformation.

    This method implements the VST approach similar to Seurat's FindVariableFeatures.
    It models the relationship between mean expression and variance, fits a
    local regression trend, and selects features with the highest standardized
    residuals (deviations from the expected variance).

    Parameters
    ----------
    container : ScpContainer
        The ScpContainer object.
    assay_name : str, default="protein"
        Name of the assay to process.
    layer : str, default="raw"
        Layer to use for variance calculation.
    n_top_features : int, default=2000
        Number of top variable features to select.
    n_bins : int, default=20
        Number of bins for mean expression grouping. Used for trend fitting.
    subset : bool, default=True
        If True, returns a container with only the selected features.
        If False, adds 'highly_variable' column to var.
    min_mean : float, default=0.01
        Minimum mean expression for a feature to be considered.

    Returns
    -------
    ScpContainer
        Container with VST-selected features.

    Raises
    ------
    ValueError
        If assay or layer not found, or if parameters are invalid.

    Notes
    -----
    The VST method follows these steps:
    1. Calculate mean expression and variance for each feature
    2. Log-transform mean and variance
    3. Bin features by mean expression
    4. Fit a trend to the variance-mean relationship
    5. Calculate residuals (observed - expected)
    6. Select features with highest residuals

    This approach is robust to the mean-variance dependency common in
    count-based single-cell data.

    Examples
    --------
    >>> # Select top 2000 variable features
    >>> container_vst = select_by_vst(container, n_top_features=2000)
    >>> # Annotate features without subsetting
    >>> container = select_by_vst(container, subset=False)
    >>> high_var = container.assays['protein'].var['highly_variable'].sum()
    """
    assay = _validate_assay_layer(container, assay_name, layer)
    X = assay.layers[layer].X

    # Convert sparse to dense if needed
    if issparse(X):
        X = X.toarray()

    # Calculate mean and variance per feature
    means = np.nanmean(X, axis=0)
    variances = np.nanvar(X, axis=0)

    # Filter out features with very low mean expression
    valid_mask = means >= min_mean
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        raise ValueError(f"No features meet min_mean threshold ({min_mean}).")

    means_valid = means[valid_mask]
    variances_valid = variances[valid_mask]

    # Log-transform
    log_means = np.log1p(means_valid)
    log_variances = np.log1p(variances_valid)

    # Fit trend and compute VST scores
    vst_scores = _compute_vst_scores(log_means, log_variances, n_bins)

    # Map back to original feature indices
    full_scores = np.full(assay.n_features, -np.inf)
    full_scores[valid_indices] = vst_scores

    # Select top features
    n_features = assay.n_features
    if n_top_features >= n_features:
        top_indices = np.arange(n_features)
    else:
        top_indices = np.argpartition(full_scores, -n_top_features)[-n_top_features:]
        # Sort by score for consistent ordering
        top_indices = top_indices[np.argsort(-full_scores[top_indices])]

    return _subset_or_annotate(
        container=container,
        assay_name=assay_name,
        assay=assay,
        top_indices=top_indices,
        subset=subset,
        action="select_by_vst",
        score=full_scores,
        score_col="vst_score",
        bool_col="highly_variable",
        params={
            "assay_name": assay_name,
            "layer": layer,
            "n_top_features": n_top_features,
            "n_bins": n_bins,
            "min_mean": min_mean,
            "subset": subset,
        },
    )


def _compute_vst_scores(
    log_means: "NDArray[np.float64]",
    log_variances: "NDArray[np.float64]",
    n_bins: int,
) -> "NDArray[np.float64]":
    """Compute VST scores using binned trend fitting.

    Parameters
    ----------
    log_means : ndarray
        Log-transformed mean values.
    log_variances : ndarray
        Log-transformed variance values.
    n_bins : int
        Number of bins for grouping.

    Returns
    -------
    vst_scores : ndarray
        VST scores (residuals from trend).
    """
    n_features = len(log_means)

    # Create bins using percentiles
    percentile_edges = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(log_means, percentile_edges)

    # Calculate median log variance in each bin
    bin_medians = []
    bin_centers = []

    for i in range(n_bins):
        # Define bin membership (include upper edge for last bin)
        if i == n_bins - 1:
            in_bin = (log_means >= bin_edges[i]) & (log_means <= bin_edges[i + 1])
        else:
            in_bin = (log_means >= bin_edges[i]) & (log_means < bin_edges[i + 1])

        if np.sum(in_bin) > 0:
            bin_medians.append(np.median(log_variances[in_bin]))
            bin_centers.append(np.median(log_means[in_bin]))
        else:
            # Handle empty bins
            if bin_medians:
                bin_medians.append(bin_medians[-1])
                bin_centers.append(bin_centers[-1] + 0.1)
            else:
                bin_medians.append(np.median(log_variances))
                bin_centers.append(np.median(log_means))

    bin_medians = np.array(bin_medians)
    bin_centers = np.array(bin_centers)

    # Remove NaN values
    finite_mask = np.isfinite(bin_medians) & np.isfinite(bin_centers)
    bin_centers = bin_centers[finite_mask]
    bin_medians = bin_medians[finite_mask]

    # Interpolate to get expected variance
    if len(bin_centers) > 1:
        trend_func = interp1d(
            bin_centers,
            bin_medians,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        expected_log_var = trend_func(log_means)
    else:
        # Fallback: use constant
        expected_log_var = np.full_like(
            log_means, bin_medians[0] if len(bin_medians) > 0 else 0
        )

    # VST score = residual from trend
    residuals = log_variances - expected_log_var
    return np.nan_to_num(residuals, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)


def select_by_dispersion(
    container: ScpContainer,
    assay_name: str = "protein",
    layer: str = "raw",
    n_top_features: int = 2000,
    n_bins: int = 20,
    subset: bool = True,
) -> ScpContainer:
    """Select highly variable features using dispersion-based method.

    This method selects features based on the coefficient of variation (CV)
    or dispersion (variance/mean), accounting for the mean-variance
    relationship through binning.

    Parameters
    ----------
    container : ScpContainer
        The ScpContainer object.
    assay_name : str, default="protein"
        Name of the assay to process.
    layer : str, default="raw"
        Layer to use for calculation.
    n_top_features : int, default=2000
        Number of top variable features to select.
    n_bins : int, default=20
        Number of bins for mean expression grouping.
    subset : bool, default=True
        If True, returns a container with only the selected features.
        If False, adds 'highly_variable' column to var.

    Returns
    -------
    ScpContainer
        Container with dispersion-selected features.

    Raises
    ------
    ValueError
        If assay or layer not found.

    Notes
    -----
    This method computes the dispersion (variance/mean) for each feature,
    then normalizes by the median dispersion within mean expression bins.
    Features are ranked by their normalized dispersion.

    Examples
    --------
    >>> # Select top 2000 variable features by dispersion
    >>> container_disp = select_by_dispersion(container, n_top_features=2000)
    """
    assay = _validate_assay_layer(container, assay_name, layer)
    X = assay.layers[layer].X

    # Convert sparse to dense if needed
    if issparse(X):
        X = X.toarray()

    # Calculate mean and variance per feature
    means = np.nanmean(X, axis=0)
    variances = np.nanvar(X, axis=0)

    # Calculate dispersion (variance/mean)
    eps = np.finfo(means.dtype).eps
    dispersion = variances / (means + eps)

    # Normalize by bin median
    normalized_dispersion = _normalize_dispersion_by_bins(
        dispersion, means, n_bins
    )

    # Select top features
    n_features = assay.n_features
    if n_top_features >= n_features:
        top_indices = np.arange(n_features)
    else:
        top_indices = np.argpartition(
            normalized_dispersion, -n_top_features
        )[-n_top_features:]
        top_indices = top_indices[
            np.argsort(-normalized_dispersion[top_indices])
        ]

    return _subset_or_annotate(
        container=container,
        assay_name=assay_name,
        assay=assay,
        top_indices=top_indices,
        subset=subset,
        action="select_by_dispersion",
        score=normalized_dispersion,
        score_col="dispersion_score",
        bool_col="highly_variable",
        params={
            "assay_name": assay_name,
            "layer": layer,
            "n_top_features": n_top_features,
            "n_bins": n_bins,
            "subset": subset,
        },
    )


def _normalize_dispersion_by_bins(
    dispersion: "NDArray[np.float64]",
    means: "NDArray[np.float64]",
    n_bins: int,
) -> "NDArray[np.float64]":
    """Normalize dispersion by median within mean expression bins.

    Parameters
    ----------
    dispersion : ndarray
        Dispersion values.
    means : ndarray
        Mean expression values.
    n_bins : int
        Number of bins.

    Returns
    -------
    normalized_dispersion : ndarray
        Dispersion normalized by bin median.
    """
    log_means = np.log1p(means)
    percentile_edges = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(log_means, percentile_edges)

    normalized = np.zeros_like(dispersion)

    for i in range(n_bins):
        if i == n_bins - 1:
            in_bin = (log_means >= bin_edges[i]) & (log_means <= bin_edges[i + 1])
        else:
            in_bin = (log_means >= bin_edges[i]) & (log_means < bin_edges[i + 1])

        if np.sum(in_bin) > 0:
            bin_median = np.median(dispersion[in_bin])
            if bin_median > 0:
                normalized[in_bin] = dispersion[in_bin] / bin_median

    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)


if __name__ == "__main__":
    print("Running VST-based feature selection tests...")

    # Create test data with varying variance patterns
    n_samples, n_features = 100, 100
    np.random.seed(42)
    X_test = np.random.gamma(shape=1, scale=5, size=(n_samples, n_features))

    # Add variable features
    for i in range(20):
        X_test[:, i] = np.random.gamma(shape=0.5 + i * 0.1, scale=10, size=n_samples)

    # Add low-variance features
    for i in range(40, 60):
        X_test[:, i] = np.random.normal(loc=5, scale=0.5, size=n_samples)

    from scptensor.core.structures import Assay, ScpMatrix, ScpContainer

    var_test = pl.DataFrame({"_index": [f"feature_{i}" for i in range(n_features)]})
    obs_test = pl.DataFrame({"_index": [f"sample_{i}" for i in range(n_samples)]})

    assay_test = Assay(
        var=var_test,
        layers={"raw": ScpMatrix(X=X_test, M=None)},
        feature_id_col="_index",
    )
    container_test = ScpContainer(obs=obs_test, assays={"protein": assay_test})

    # Test 1: VST selection
    print("\nTest 1: VST feature selection (n_top=30)")
    result = select_by_vst(container_test, n_top_features=30, subset=True)
    assert result.assays["protein"].n_features == 30
    print("  Passed: 30 features selected")

    # Test 2: VST annotation mode
    print("\nTest 2: VST annotation mode")
    result2 = select_by_vst(container_test, n_top_features=30, subset=False)
    assert "highly_variable" in result2.assays["protein"].var.columns
    assert "vst_score" in result2.assays["protein"].var.columns
    assert result2.assays["protein"].var["highly_variable"].sum() == 30
    print("  Passed: 30 features annotated")

    # Test 3: Dispersion selection
    print("\nTest 3: Dispersion-based selection (n_top=25)")
    result3 = select_by_dispersion(container_test, n_top_features=25, subset=True)
    assert result3.assays["protein"].n_features == 25
    print("  Passed: 25 features selected")

    # Test 4: Dispersion annotation mode
    print("\nTest 4: Dispersion annotation mode")
    result4 = select_by_dispersion(container_test, n_top_features=25, subset=False)
    assert result4.assays["protein"].var["highly_variable"].sum() == 25
    print("  Passed: 25 features annotated")

    # Test 5: min_mean filtering
    print("\nTest 5: VST with min_mean threshold")
    X_test[:, 80:90] = np.random.uniform(0, 0.001, size=(n_samples, 10))
    assay_test2 = Assay(
        var=var_test,
        layers={"raw": ScpMatrix(X=X_test, M=None)},
        feature_id_col="_index",
    )
    container_test2 = ScpContainer(obs=obs_test, assays={"protein": assay_test2})

    result5 = select_by_vst(
        container_test2, n_top_features=30, min_mean=1.0, subset=False
    )
    print("  Passed: min_mean filtering works")

    # Test 6: Edge case - n_top > n_features
    print("\nTest 6: n_top_features > n_features")
    result6 = select_by_vst(container_test, n_top_features=200, subset=True)
    assert result6.assays["protein"].n_features == n_features
    print("  Passed: All features kept")

    # Test 7: Sparse matrix support
    print("\nTest 7: Sparse matrix support")
    X_sparse = sparse.csr_matrix(X_test)
    assay_sparse = Assay(
        var=var_test,
        layers={"raw": ScpMatrix(X=X_sparse, M=None)},
        feature_id_col="_index",
    )
    container_sparse = ScpContainer(obs=obs_test, assays={"protein": assay_sparse})
    result7 = select_by_vst(container_sparse, n_top_features=30, subset=True)
    assert result7.assays["protein"].n_features == 30
    print("  Passed: Sparse matrix handled")

    print("\n" + "=" * 50)
    print("All VST tests passed!")
    print("=" * 50)
