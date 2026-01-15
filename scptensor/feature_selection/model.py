"""Model-based feature selection for single-cell proteomics data.

This module provides feature selection based on statistical models,
including random forest importance and PCA loading magnitudes.
"""

from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl
from scipy import sparse
from scipy.sparse import issparse

from scptensor.core.structures import Assay, ScpContainer
from scptensor.feature_selection._shared import (
    _subset_or_annotate,
    _validate_assay_layer,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


def select_by_model_importance(
    container: ScpContainer,
    assay_name: str = "protein",
    layer: str = "raw",
    n_top_features: int = 2000,
    method: Literal["random_forest", "variance_threshold"] = "random_forest",
    n_estimators: int = 50,
    max_depth: int | None = None,
    random_state: int = 42,
    subset: bool = True,
    variance_threshold: float = 0.0,
) -> ScpContainer:
    """Select features based on model-derived importance scores.

    This function uses machine learning models to rank features by importance.
    The available methods are:
    - 'random_forest': Uses random forest feature importance (mean decrease in impurity)
    - 'variance_threshold': Selects features with variance above threshold

    Parameters
    ----------
    container : ScpContainer
        The ScpContainer object.
    assay_name : str, default="protein"
        Name of the assay to process.
    layer : str, default="raw"
        Layer to use for importance calculation.
    n_top_features : int, default=2000
        Number of top features to select based on importance scores.
    method : {'random_forest', 'variance_threshold'}, default='random_forest'
        Method for calculating feature importance.
    n_estimators : int, default=50
        Number of trees in random forest (only for 'random_forest' method).
    max_depth : int | None, default=None
        Maximum depth of random forest trees. None means unlimited.
    random_state : int, default=42
        Random seed for reproducibility.
    subset : bool, default=True
        If True, returns a container with only selected features.
        If False, adds 'selected_by_model' column to var.
    variance_threshold : float, default=0.0
        Minimum variance for a feature to be selected (for 'variance_threshold' method).

    Returns
    -------
    ScpContainer
        Container with model-selected features.

    Raises
    ------
    ValueError
        If assay or layer not found, or if method is invalid.
    ImportError
        If scikit-learn is required but not installed.

    Notes
    -----
    For unsupervised feature selection, the random forest method creates
    pseudo-labels using k-means clustering on the samples. The forest then
    learns to predict these clusters, and feature importances reflect
    discriminative power.

    Examples
    --------
    >>> # Select features using random forest importance
    >>> container_rf = select_by_model_importance(container, method='random_forest')
    >>> # Select high-variance features
    >>> container_var = select_by_model_importance(
    ...     container, method='variance_threshold', variance_threshold=1.0
    ... )
    >>> # Annotate features without subsetting
    >>> container = select_by_model_importance(container, subset=False)
    """
    VALID_METHODS = ("random_forest", "variance_threshold")
    if method not in VALID_METHODS:
        raise ValueError(
            f"method must be one of {VALID_METHODS}, got '{method}'"
        )

    assay = _validate_assay_layer(container, assay_name, layer)
    X = assay.layers[layer].X

    # Convert sparse to dense if needed
    if issparse(X):
        X = X.toarray()

    # Handle NaN values
    X = np.nan_to_num(X, nan=0.0)

    # Compute importance scores
    if method == "variance_threshold":
        importance_scores = _variance_importance(X, variance_threshold, n_top_features)
    else:  # random_forest
        importance_scores = _random_forest_importance(
            X, n_top_features, n_estimators, max_depth, random_state, container.n_samples
        )

    # Select top features
    n_features = assay.n_features
    if n_top_features >= n_features:
        top_indices = np.arange(n_features)
    else:
        top_indices = np.argpartition(importance_scores, -n_top_features)[-n_top_features:]
        top_indices = top_indices[np.argsort(-importance_scores[top_indices])]

    return _subset_or_annotate(
        container=container,
        assay_name=assay_name,
        assay=assay,
        top_indices=top_indices,
        subset=subset,
        action="select_by_model_importance",
        score=importance_scores,
        score_col="model_importance",
        bool_col="selected_by_model",
        params={
            "assay_name": assay_name,
            "layer": layer,
            "method": method,
            "n_top_features": n_top_features,
            "n_estimators": n_estimators,
            "random_state": random_state,
            "subset": subset,
        },
    )


def _variance_importance(
    X: "NDArray[np.float64]",
    threshold: float,
    n_top: int,
) -> "NDArray[np.float64]":
    """Compute variance-based importance scores.

    Parameters
    ----------
    X : ndarray
        Data matrix (n_samples, n_features).
    threshold : float
        Variance threshold for filtering.
    n_top : int
        Maximum number of features to select.

    Returns
    -------
    importance : ndarray
        Variance scores (0 for features below threshold).
    """
    variances = np.var(X, axis=0)

    # Zero out features below threshold
    importance = variances * (variances > threshold).astype(float)

    return importance


def _random_forest_importance(
    X: "NDArray[np.float64]",
    n_top: int,
    n_estimators: int,
    max_depth: int | None,
    random_state: int,
    n_samples: int,
) -> "NDArray[np.float64]":
    """Compute random forest feature importance.

    Parameters
    ----------
    X : ndarray
        Data matrix.
    n_top : int
        Number of top features to select.
    n_estimators : int
        Number of trees.
    max_depth : int or None
        Maximum tree depth.
    random_state : int
        Random seed.
    n_samples : int
        Number of samples in original data.

    Returns
    -------
    importance : ndarray
        Feature importance scores.
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.cluster import KMeans
    except ImportError as e:
        raise ImportError(
            "scikit-learn is required for random_forest method. "
            "Install it with: pip install scikit-learn"
        ) from e

    # Determine reasonable number of clusters
    n_clusters = min(10, max(2, n_samples // 10))

    # Perform k-means to get pseudo-labels
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    pseudo_labels = kmeans.fit_predict(X)

    # Train random forest
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        bootstrap=True,
    )
    rf.fit(X, pseudo_labels)

    return rf.feature_importances_


def select_by_pca_loadings(
    container: ScpContainer,
    assay_name: str = "protein",
    layer: str = "raw",
    n_top_features: int = 2000,
    n_components: int = 50,
    subset: bool = True,
) -> ScpContainer:
    """Select features based on PCA loading magnitudes.

    This method performs PCA and ranks features by their absolute loadings
    on the top principal components. Features with high loadings contribute
    more to the major axes of variation in the data.

    Parameters
    ----------
    container : ScpContainer
        The ScpContainer object.
    assay_name : str, default="protein"
        Name of the assay to process.
    layer : str, default="raw"
        Layer to use for PCA.
    n_top_features : int, default=2000
        Number of top features to select.
    n_components : int, default=50
        Number of principal components to consider for loading calculation.
    subset : bool, default=True
        If True, returns a container with only selected features.
        If False, adds 'selected_by_pca' column to var.

    Returns
    -------
    ScpContainer
        Container with PCA-selected features.

    Raises
    ------
    ValueError
        If assay or layer not found.
    ImportError
        If scikit-learn is not installed.

    Notes
    -----
    Features are scored by the sum of absolute loadings across the top
    n_components principal components, weighted by explained variance.

    Examples
    --------
    >>> # Select features based on PCA loadings
    >>> container_pca = select_by_pca_loadings(container, n_components=50)
    """
    assay = _validate_assay_layer(container, assay_name, layer)
    X = assay.layers[layer].X

    # Convert sparse to dense if needed
    if issparse(X):
        X = X.toarray()

    # Handle NaN values
    X = np.nan_to_num(X, nan=0.0)

    # Center the data
    X_centered = X - np.mean(X, axis=0)

    # Limit n_components
    n_components = min(n_components, X.shape[0], X.shape[1])

    try:
        from sklearn.decomposition import PCA
    except ImportError as e:
        raise ImportError(
            "scikit-learn is required for PCA-based selection. "
            "Install it with: pip install scikit-learn"
        ) from e

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_centered)

    # Weighted sum of absolute loadings
    loadings = np.abs(pca.components_.T)  # (n_features, n_components)
    weights = pca.explained_variance_ratio_
    importance_scores = np.sum(loadings * weights, axis=1)

    # Select top features
    n_features = assay.n_features
    if n_top_features >= n_features:
        top_indices = np.arange(n_features)
    else:
        top_indices = np.argpartition(importance_scores, -n_top_features)[-n_top_features:]
        top_indices = top_indices[np.argsort(-importance_scores[top_indices])]

    return _subset_or_annotate(
        container=container,
        assay_name=assay_name,
        assay=assay,
        top_indices=top_indices,
        subset=subset,
        action="select_by_pca_loadings",
        score=importance_scores,
        score_col="pca_importance",
        bool_col="selected_by_pca",
        params={
            "assay_name": assay_name,
            "layer": layer,
            "n_top_features": n_top_features,
            "n_components": n_components,
            "subset": subset,
        },
    )


if __name__ == "__main__":
    print("Running model-based feature selection tests...")

    # Create test data
    n_samples, n_features = 100, 50
    np.random.seed(42)
    X_test = np.random.randn(n_samples, n_features)

    # Make some features more important (higher variance)
    X_test[:, :10] *= 3
    X_test[:, 10:20] *= 2

    from scptensor.core.structures import Assay, ScpMatrix, ScpContainer

    var_test = pl.DataFrame({"_index": [f"feature_{i}" for i in range(n_features)]})
    obs_test = pl.DataFrame({"_index": [f"sample_{i}" for i in range(n_samples)]})

    assay_test = Assay(
        var=var_test,
        layers={"raw": ScpMatrix(X=X_test, M=None)},
        feature_id_col="_index",
    )
    container_test = ScpContainer(obs=obs_test, assays={"protein": assay_test})

    # Test 1: Variance threshold method
    print("\nTest 1: Variance threshold selection")
    result = select_by_model_importance(
        container_test,
        method="variance_threshold",
        variance_threshold=0.5,
        n_top_features=20,
        subset=True,
    )
    assert result.assays["protein"].n_features <= n_features
    print(f"  Passed: {result.assays['protein'].n_features} features selected")

    # Test 2: Annotation mode
    print("\nTest 2: Annotation mode")
    result2 = select_by_model_importance(
        container_test,
        method="variance_threshold",
        n_top_features=15,
        subset=False,
    )
    assert "selected_by_model" in result2.assays["protein"].var.columns
    assert result2.assays["protein"].var["selected_by_model"].sum() == 15
    print("  Passed: 15 features annotated")

    # Test 3: Random forest method
    print("\nTest 3: Random forest importance")
    try:
        result3 = select_by_model_importance(
            container_test,
            method="random_forest",
            n_top_features=20,
            n_estimators=20,
            subset=True,
        )
        assert result3.assays["protein"].n_features == 20
        print("  Passed: 20 features selected by RF")
    except ImportError:
        print("  Skipped: scikit-learn not available")

    # Test 4: PCA loading selection
    print("\nTest 4: PCA loading selection")
    try:
        result4 = select_by_pca_loadings(
            container_test,
            n_top_features=20,
            n_components=10,
            subset=True,
        )
        assert result4.assays["protein"].n_features == 20
        print("  Passed: 20 features selected by PCA")
    except ImportError:
        print("  Skipped: scikit-learn not available")

    # Test 5: PCA annotation mode
    print("\nTest 5: PCA annotation mode")
    try:
        result5 = select_by_pca_loadings(
            container_test,
            n_top_features=15,
            n_components=10,
            subset=False,
        )
        assert result5.assays["protein"].var["selected_by_pca"].sum() == 15
        print("  Passed: 15 features annotated")
    except ImportError:
        print("  Skipped: scikit-learn not available")

    # Test 6: Invalid method
    print("\nTest 6: Invalid method (should error)")
    try:
        select_by_model_importance(container_test, method="invalid_method")
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        print("  Passed: Correctly raised ValueError")

    # Test 7: Edge case - n_top > n_features
    print("\nTest 7: n_top_features > n_features")
    result7 = select_by_model_importance(
        container_test,
        method="variance_threshold",
        n_top_features=200,
        subset=True,
    )
    assert result7.assays["protein"].n_features == n_features
    print("  Passed: All features kept")

    # Test 8: Sparse matrix support
    print("\nTest 8: Sparse matrix support")
    X_sparse = sparse.csr_matrix(X_test)
    assay_sparse = Assay(
        var=var_test,
        layers={"raw": ScpMatrix(X=X_sparse, M=None)},
        feature_id_col="_index",
    )
    container_sparse = ScpContainer(obs=obs_test, assays={"protein": assay_sparse})
    result8 = select_by_model_importance(
        container_sparse,
        method="variance_threshold",
        n_top_features=20,
        subset=True,
    )
    assert result8.assays["protein"].n_features == 20
    print("  Passed: Sparse matrix handled")

    print("\n" + "=" * 50)
    print("All model-based tests passed!")
    print("=" * 50)
