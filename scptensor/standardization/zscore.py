"""Z-score standardization for ScpTensor.

References:
    Lazar, C., et al. (2016). Accounting for the Multiple Natures of Missing
    Values in Label-Free Quantitative Proteomics Data. Journal of Proteome
    Research.

    Chawade, A., Alexandersson, E., & Levander, F. (2014). Normalyzer:
    a tool for rapid evaluation of normalization methods for omics data sets.
    Journal of Proteome Research.
"""

from __future__ import annotations

import numpy as np

from scptensor.core.exceptions import (
    AssayNotFoundError,
    LayerNotFoundError,
    ScpValueError,
    ValidationError,
)
from scptensor.core.structures import ScpContainer, ScpMatrix


def zscore(
    container: ScpContainer,
    assay_name: str = "protein",
    source_layer: str = "imputed",
    new_layer_name: str | None = "zscore",
    axis: int = 0,
    ddof: int = 1,
) -> ScpContainer:
    """Apply z-score standardization (mean=0, std=1).

    Parameters
    ----------
    container : ScpContainer
        The ScpContainer object.
    assay_name : str, default="protein"
        Name of the assay to transform.
    source_layer : str, default="imputed"
        Name of the layer to use as input.
    new_layer_name : str | None, default="zscore"
        Name for the new layer with z-scored values.
    axis : int, default=0
        0 to standardize features (columns), 1 to standardize samples (rows).
    ddof : int, default=1
        Delta degrees of freedom for std estimation.

    Returns
    -------
    ScpContainer
        The same container with an added z-score layer.
    """
    if axis not in (0, 1):
        raise ScpValueError(
            f"Axis must be 0 or 1, got {axis}.",
            parameter="axis",
            value=axis,
        )
    if ddof < 0:
        raise ScpValueError(
            f"Delta degrees of freedom must be non-negative, got {ddof}.",
            parameter="ddof",
            value=ddof,
        )

    if assay_name not in container.assays:
        available = ", ".join(f"'{k}'" for k in container.assays)
        raise AssayNotFoundError(
            assay_name,
            hint=f"Available assays: {available}. Use container.list_assays() to inspect.",
        )

    assay = container.assays[assay_name]
    if source_layer not in assay.layers:
        available = ", ".join(f"'{k}'" for k in assay.layers)
        raise LayerNotFoundError(
            source_layer,
            assay_name,
            hint=f"Available layers in assay '{assay_name}': {available}. Use assay.list_layers().",
        )

    input_layer = assay.layers[source_layer]
    if np.isnan(input_layer.X).any():
        raise ValidationError(
            "Z-score standardization requires a complete matrix (no missing values). "
            f"Layer '{source_layer}' contains NaNs. Please impute first.",
            field="X",
        )

    mean = np.mean(input_layer.X, axis=axis, keepdims=True)
    std = np.std(input_layer.X, axis=axis, keepdims=True, ddof=ddof)
    std[std == 0] = 1.0
    x_z = (input_layer.X - mean) / std

    layer_name = new_layer_name or "zscore"
    assay.add_layer(
        layer_name,
        ScpMatrix(
            X=x_z,
            M=input_layer.M.copy() if input_layer.M is not None else None,
        ),
    )

    container.log_operation(
        action="standardization_zscore",
        params={
            "assay": assay_name,
            "source_layer": source_layer,
            "new_layer_name": layer_name,
            "axis": axis,
            "ddof": ddof,
        },
        description=f"Z-score standardization on '{source_layer}' -> '{layer_name}'.",
    )

    return container
