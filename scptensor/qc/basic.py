"""Basic Quality Control operations for single-cell proteomics data."""

import numpy as np

from scptensor.core.exceptions import AssayNotFoundError, ScpValueError
from scptensor.core.structures import ScpContainer


def _get_layer(assay, layer_name: str | None = None):
    """Get a layer from assay, defaulting to 'raw' or first available layer.

    Args:
        assay: The Assay object.
        layer_name: Specific layer name to retrieve, or None for default.

    Returns:
        The ScpMatrix layer.

    Raises:
        LayerNotFoundError: If no layers exist in the assay.
    """
    from scptensor.core.exceptions import LayerNotFoundError

    if layer_name:
        if layer_name not in assay.layers:
            raise LayerNotFoundError(layer_name, "<assay>")
        return assay.layers[layer_name]

    return assay.layers.get("raw") or next(iter(assay.layers.values()), None)


def basic_qc(
    container: ScpContainer,
    assay_name: str = "protein",
    min_features: int = 200,
    min_cells: int = 3,
    detection_threshold: float = 0.0,
    new_layer_name: str | None = None,
) -> ScpContainer:
    """
    Perform basic Quality Control (QC) on samples and features.

    This function filters samples with too few detected features and
    features detected in too few samples.

    Args:
        container: The ScpContainer object.
        assay_name: Name of the assay to perform QC on.
        min_features: Minimum number of features required for a cell to be kept.
        min_cells: Minimum number of cells a feature must be detected in.
        detection_threshold: Threshold for a value to be considered detected.
        new_layer_name: Unused parameter (kept for API compatibility).

    Returns:
        A new ScpContainer with filtered samples and features.

    Raises:
        AssayNotFoundError: If the specified assay does not exist.
        ScpValueError: If min_features or min_cells parameters are invalid.

    Examples:
        >>> container = basic_qc(
        ...     container,
        ...     assay_name="protein",
        ...     min_features=200,
        ...     min_cells=3
        ... )
    """
    # Validate parameters
    if min_features < 0:
        raise ScpValueError(
            f"min_features must be non-negative, got {min_features}.",
            parameter="min_features",
            value=min_features,
        )
    if min_cells < 0:
        raise ScpValueError(
            f"min_cells must be non-negative, got {min_cells}.",
            parameter="min_cells",
            value=min_cells,
        )

    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]
    layer = _get_layer(assay)
    X = layer.X

    # 1. Sample QC: Filter cells with too few features
    n_features_per_cell = np.sum(detection_threshold < X, axis=1)
    keep_samples_mask = n_features_per_cell >= min_features
    keep_samples_indices = np.where(keep_samples_mask)[0]

    container_filtered_samples = container.filter_samples(sample_indices=keep_samples_indices)

    # 2. Feature QC: Filter features detected in too few cells
    assay_filtered = container_filtered_samples.assays[assay_name]
    layer_filtered = _get_layer(assay_filtered)
    X_filtered = layer_filtered.X

    n_cells_per_feature = np.sum(X_filtered > detection_threshold, axis=0)
    keep_features_mask = n_cells_per_feature >= min_cells
    keep_features_indices = np.where(keep_features_mask)[0]

    container_final = container_filtered_samples.filter_features(
        assay_name, feature_indices=keep_features_indices
    )

    # Log QC stats
    n_samples_removed = container.n_samples - container_final.n_samples
    n_features_removed = assay.n_features - container_final.assays[assay_name].n_features

    container_final.log_operation(
        action="basic_qc",
        params={"assay": assay_name, "min_features": min_features, "min_cells": min_cells},
        description=f"Removed {n_samples_removed} samples and {n_features_removed} features.",
    )

    return container_final
