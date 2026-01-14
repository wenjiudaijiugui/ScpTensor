"""Outlier detection methods for single-cell proteomics data."""

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.ensemble import IsolationForest

from scptensor.core.exceptions import AssayNotFoundError, LayerNotFoundError, ScpValueError
from scptensor.core.structures import ScpContainer


def detect_outliers(
    container: ScpContainer,
    assay_name: str = "protein",
    layer: str = "raw",
    contamination: float = 0.05,
    random_state: int = 42,
) -> ScpContainer:
    """
    Detect outlier samples using Isolation Forest.

    Args:
        container: The ScpContainer object.
        assay_name: Name of the assay to use.
        layer: Layer to use for detection.
        contamination: Proportion of outliers in the data set (0, 0.5).
        random_state: Random state for reproducibility.

    Returns:
        ScpContainer with an added column 'is_outlier' in obs.

    Raises:
        AssayNotFoundError: If the specified assay does not exist.
        LayerNotFoundError: If the specified layer does not exist in the assay.
        ScpValueError: If contamination parameter is invalid.

    Examples:
        >>> container = detect_outliers(
        ...     container,
        ...     assay_name="protein",
        ...     contamination=0.05
        ... )
        >>> outlier_mask = container.obs['is_outlier'].to_numpy()
        >>> n_outliers = outlier_mask.sum()
    """
    # Validate parameters
    if not (0 < contamination < 0.5):
        raise ScpValueError(
            f"contamination must be in (0, 0.5), got {contamination}.",
            parameter="contamination",
            value=contamination,
        )

    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]
    if layer not in assay.layers:
        raise LayerNotFoundError(layer, assay_name)

    X = assay.layers[layer].X

    # Convert sparse to dense for IsolationForest
    if sp.issparse(X):
        X = X.toarray()

    # Handle NaN values using nan_to_num (faster than manual imputation)
    data_to_fit = np.nan_to_num(X, nan=0.0)

    clf = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    preds = clf.fit_predict(data_to_fit)  # 1 for inlier, -1 for outlier

    is_outlier = preds == -1

    # Add to obs
    new_obs = container.obs.with_columns(pl.Series("is_outlier", is_outlier))

    # Create new container
    new_container = ScpContainer(
        obs=new_obs,
        assays=container.assays,
        history=list(container.history),
    )

    new_container.log_operation(
        action="detect_outliers",
        params={"contamination": contamination, "method": "IsolationForest"},
        description=f"Detected {sum(is_outlier)} outliers.",
    )

    return new_container
