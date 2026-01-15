"""Highly Variable Genes/Proteins (HVG) selection.

This module implements coefficient-of-variation based feature selection
for identifying the most variable proteins/peptides in single-cell data.
"""

from typing import Literal

import numpy as np
import polars as pl
from scipy import sparse

from scptensor.core.structures import Assay, ScpContainer
from scptensor.feature_selection._shared import (
    _compute_mean_var,
    _subset_or_annotate,
    _validate_assay_layer,
)


def select_hvg(
    container: ScpContainer,
    assay_name: str = "protein",
    layer: str = "raw",
    n_top_features: int = 2000,
    method: Literal["cv", "dispersion"] = "cv",
    subset: bool = True,
) -> ScpContainer:
    """Select Highly Variable Genes/Proteins (HVG).

    This function identifies features with high variability across samples,
    using either coefficient of variation (CV) or dispersion as the metric.

    Parameters
    ----------
    container : ScpContainer
        Input container.
    assay_name : str, default="protein"
        Name of the assay.
    layer : str, default="raw"
        Layer to use for calculation.
    n_top_features : int, default=2000
        Number of top features to select.
    method : {'cv', 'dispersion'}, default="cv"
        Variability metric:
        - 'cv': Coefficient of Variation (std / mean)
        - 'dispersion': Variance-to-mean ratio (var / mean)
    subset : bool, default=True
        If True, returns a container with only HVGs.
        If False, adds 'highly_variable' column to var.

    Returns
    -------
    ScpContainer
        Container with HVG-filtered features or annotated var.

    Raises
    ------
    ValueError
        If assay or layer not found.

    Examples
    --------
    >>> # Select top 2000 highly variable proteins
    >>> container_hvg = select_hvg(container, n_top_features=2000)
    >>> # Annotate without subsetting
    >>> container = select_hvg(container, subset=False)
    >>> hvg_mask = container.assays['protein'].var['highly_variable']
    """
    assay = _validate_assay_layer(container, assay_name, layer)
    X = assay.layers[layer].X

    # Efficient mean/var calculation (sparse-aware)
    mean, var = _compute_mean_var(X, axis=0)

    # Compute variability score
    eps = np.finfo(mean.dtype).eps
    if method == "cv":
        score = np.sqrt(var) / (mean + eps)
    else:
        score = var / (mean + eps)

    # Handle NaN/Inf
    score = np.nan_to_num(score, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)

    # Select top features
    n_features = assay.n_features
    if n_top_features >= n_features:
        top_indices = np.arange(n_features)
    else:
        top_indices = np.argpartition(score, -n_top_features)[-n_top_features:]

    return _subset_or_annotate(
        container=container,
        assay_name=assay_name,
        assay=assay,
        top_indices=top_indices,
        subset=subset,
        action="select_hvg",
        score=score,
        score_col="variability_score",
        bool_col="highly_variable",
        params={"n_top": n_top_features, "method": method, "subset": subset},
    )


if __name__ == "__main__":
    print("Running HVG selection tests...")

    # Create test data
    np.random.seed(42)
    n_samples, n_features = 100, 50
    X_test = np.random.gamma(shape=1, scale=5, size=(n_samples, n_features))

    # Add variable features
    for i in range(10):
        X_test[:, i] = np.random.gamma(shape=0.5 + i * 0.1, scale=10, size=n_samples)

    from scptensor.core.structures import Assay, ScpContainer, ScpMatrix

    var_test = pl.DataFrame({"_index": [f"feature_{i}" for i in range(n_features)]})
    obs_test = pl.DataFrame({"_index": [f"sample_{i}" for i in range(n_samples)]})

    assay_test = Assay(
        var=var_test,
        layers={"raw": ScpMatrix(X=X_test, M=None)},
        feature_id_col="_index",
    )
    container_test = ScpContainer(obs=obs_test, assays={"protein": assay_test})

    # Test 1: CV-based selection
    print("\nTest 1: CV-based HVG selection (n_top=20)")
    result = select_hvg(container_test, method="cv", n_top_features=20, subset=True)
    assert result.assays["protein"].n_features == 20
    print("  Passed: 20 features selected")

    # Test 2: Dispersion-based selection
    print("\nTest 2: Dispersion-based HVG selection")
    result2 = select_hvg(container_test, method="dispersion", n_top_features=15, subset=True)
    assert result2.assays["protein"].n_features == 15
    print("  Passed: 15 features selected")

    # Test 3: Annotation mode
    print("\nTest 3: Annotation mode (subset=False)")
    result3 = select_hvg(container_test, n_top_features=20, subset=False)
    assert "highly_variable" in result3.assays["protein"].var.columns
    assert "variability_score" in result3.assays["protein"].var.columns
    n_hvg = result3.assays["protein"].var["highly_variable"].sum()
    assert n_hvg == 20
    print("  Passed: 20 features annotated")

    # Test 4: Sparse matrix support
    print("\nTest 4: Sparse matrix support")
    X_sparse = sparse.csr_matrix(X_test)
    assay_sparse = Assay(
        var=var_test,
        layers={"raw": ScpMatrix(X=X_sparse, M=None)},
        feature_id_col="_index",
    )
    container_sparse = ScpContainer(obs=obs_test, assays={"protein": assay_sparse})
    result4 = select_hvg(container_sparse, n_top_features=20, subset=True)
    assert result4.assays["protein"].n_features == 20
    print("  Passed: Sparse matrix handled")

    print("\n" + "=" * 50)
    print("All HVG tests passed!")
    print("=" * 50)
