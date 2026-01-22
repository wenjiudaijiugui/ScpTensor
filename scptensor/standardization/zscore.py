"""Z-score standardization for ScpTensor.

References:
    Specht, H., et al. (2021). Single-cell proteomic and transcriptomic analysis
    of macrophage heterogeneity using SCoPE2. Genome Biology.

    Lazar, C., et al. (2016). Accounting for the Multiple Natures of Missing
    Values in Label-Free Quantitative Proteomics Data. Journal of Proteome Research.

    Vanderaa, C., & Gatto, L. (2023). Revisiting the analysis of single-cell
    proteomics data. Expert Review of Proteomics.
"""

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
    """
    Z-score standardization (feature scaling to mean=0, std=1).

    Z-score standardization transforms features to have zero mean and unit variance.
    This is a feature scaling method that makes different features comparable by
    removing scale differences.

    **Important:** Z-score standardization must be performed on a complete matrix
    (no missing values) as per Lazar et al. (2016) and Vanderaa & Gatto (2023).
    Apply imputation before using this function.

    **Mathematical Formulation:**
        z = (x - mean) / std

    **Default Behavior (axis=0):**
        Standardizes features (columns) - each feature will have mean=0, std=1.
        This is the standard use case for feature scaling in machine learning.

    **Alternative (axis=1):**
        Standardizes samples (rows) - each sample will have mean=0, std=1.

    Parameters
    ----------
    container : ScpContainer
        The ScpContainer object.
    assay_name : str, default="protein"
        Name of the assay to transform.
    source_layer : str, default="imputed"
        Name of the layer to use as input.
    new_layer_name : Optional[str], default="zscore"
        Name for the new layer with z-scores.
    axis : int, default=0
        0 to standardize features (columns), 1 to standardize samples (rows).
        Default is 0 (feature-wise standardization).
    ddof : int, default=1
        Delta degrees of freedom. 1 for unbiased estimator (sample std),
        0 for population std. R's scale() uses ddof=1.

    Returns
    -------
    ScpContainer
        ScpContainer with added z-score standardized layer.

    Raises
    ------
    ScpValueError
        If axis or ddof parameters are invalid.
    AssayNotFoundError
        If the specified assay does not exist.
    LayerNotFoundError
        If the specified layer does not exist in the assay.
    ValidationError
        If the layer contains missing values.

    Examples
    --------
    >>> import polars as pl
    >>> from scptensor.core.structures import ScpContainer, Assay, ScpMatrix
    >>> import numpy as np
    >>> container = ScpContainer(obs=pl.DataFrame({'_index': ['s1', 's2']}))
    >>> assay = Assay(var=pl.DataFrame({'_index': ['p1', 'p2']}))
    >>> assay.add_layer('imputed', ScpMatrix(X=np.array([[1.0, 2.0], [3.0, 4.0]])))
    >>> container.add_assay('protein', assay)
    >>> result = zscore(container)
    >>> 'zscore' in result.assays['protein'].layers
    True
    """
    # Validate parameters early (guard clause pattern)
    if axis not in (0, 1):
        raise ScpValueError(
            f"Axis must be 0 or 1, got {axis}. "
            "Use axis=0 to standardize features (columns), axis=1 for samples (rows).",
            parameter="axis",
            value=axis,
        )
    if ddof < 0:
        raise ScpValueError(
            f"Delta degrees of freedom must be non-negative, got {ddof}. "
            "Use ddof=1 for sample std (unbiased), ddof=0 for population std.",
            parameter="ddof",
            value=ddof,
        )

    # Validate assay and layer existence
    if assay_name not in container.assays:
        available = ", ".join(f"'{k}'" for k in container.assays.keys())
        raise AssayNotFoundError(
            assay_name,
            hint=f"Available assays: {available}. Use container.list_assays() to see all assays.",
        )

    assay = container.assays[assay_name]
    if source_layer not in assay.layers:
        available = ", ".join(f"'{k}'" for k in assay.layers.keys())
        raise LayerNotFoundError(
            source_layer,
            assay_name,
            hint=f"Available layers in assay '{assay_name}': {available}. "
            f"Use assay.list_layers() to see all layers.",
        )

    input_layer = assay.layers[source_layer]

    # Z-score requires complete matrix - no missing values allowed
    if np.isnan(input_layer.X).any():
        raise ValidationError(
            "Z-score standardization requires a complete matrix (no missing values). "
            f"Layer '{source_layer}' contains NaNs. "
            "Please apply imputation before z-score standardization. "
            "Reference: Vanderaa, C. & Gatto, L. (2023). Expert Review of Proteomics.",
            field="X",
        )

    # Compute statistics with vectorized operations
    mean = np.mean(input_layer.X, axis=axis, keepdims=True)
    std = np.std(input_layer.X, axis=axis, keepdims=True, ddof=ddof)

    # Handle zero std to avoid division by zero
    std[std == 0] = 1.0

    # Apply standardization: z = (x - mean) / std
    X_z = (input_layer.X - mean) / std

    # Create new layer with copied mask
    layer_name = new_layer_name or "zscore"
    new_matrix = ScpMatrix(
        X=X_z,
        M=input_layer.M.copy() if input_layer.M is not None else None,
    )

    assay.add_layer(layer_name, new_matrix)

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
