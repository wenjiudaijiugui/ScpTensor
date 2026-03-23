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
import scipy.sparse as sp

from scptensor.core.assay_alias import resolve_assay_name
from scptensor.core.exceptions import (
    AssayNotFoundError,
    LayerNotFoundError,
    ScpValueError,
    ValidationError,
)
from scptensor.core.structures import ScpContainer, ScpMatrix


def zscore(
    container: ScpContainer,
    assay_name: str = "proteins",
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
    assay_name : str, default="proteins"
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

    resolved_assay_name = resolve_assay_name(container, assay_name)

    if resolved_assay_name not in container.assays:
        available = ", ".join(f"'{k}'" for k in container.assays)
        raise AssayNotFoundError(
            assay_name,
            hint=f"Available assays: {available}. Use container.list_assays() to inspect.",
        )

    assay = container.assays[resolved_assay_name]
    if source_layer not in assay.layers:
        available = ", ".join(f"'{k}'" for k in assay.layers)
        raise LayerNotFoundError(
            source_layer,
            resolved_assay_name,
            hint=(
                f"Available layers in assay '{resolved_assay_name}': {available}. "
                "Use assay.list_layers()."
            ),
        )

    input_layer = assay.layers[source_layer]
    x_source = input_layer.X
    x = (
        x_source.toarray()
        if isinstance(x_source, sp.spmatrix)
        else np.asarray(x_source, dtype=float)
    )
    if np.isnan(x).any():
        raise ValidationError(
            "Z-score standardization requires a complete matrix (no missing values). "
            f"Layer '{source_layer}' contains NaNs. Please impute first.",
            field="X",
        )
    if np.isinf(x).any():
        raise ValidationError(
            "Z-score standardization does not accept Inf values. "
            f"Layer '{source_layer}' contains infinite values. Please clean them first.",
            field="X",
        )

    axis_len = x.shape[axis]
    if axis_len <= ddof:
        raise ValidationError(
            f"Z-score standardization with ddof={ddof} requires at least {ddof + 1} "
            f"values along axis {axis}, got {axis_len}.",
            field="ddof",
        )

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True, ddof=ddof)
    std[std == 0] = 1.0
    x_z = (x - mean) / std

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
            "assay": resolved_assay_name,
            "source_layer": source_layer,
            "new_layer_name": layer_name,
            "axis": axis,
            "ddof": ddof,
        },
        description=f"Z-score standardization on '{source_layer}' -> '{layer_name}'.",
    )

    return container
