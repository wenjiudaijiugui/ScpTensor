"""Median centering normalization for ScpTensor."""

import numpy as np

from scptensor.core.exceptions import (
    AssayNotFoundError,
    LayerNotFoundError,
)
from scptensor.core.structures import ScpContainer, ScpMatrix


def norm_median(
    container: ScpContainer,
    assay_name: str = "protein",
    source_layer: str = "raw",
    new_layer_name: str | None = "median_centered",
    add_global_median: bool = False,
) -> ScpContainer:
    """
    Subtract the median of each sample with optional global median restoration.

    This method provides two modes for median-based normalization:

    1. Centering Mode (add_global_median=False):
        Centers each sample around its median value, removing sample-specific
        bias while preserving relative differences between features.

        Mathematical Formulation:
            X_centered = X - median(X, axis=1, keepdims=True)

        This mode is useful for removing sample-specific technical bias
        while maintaining the relative differences between features within
        each sample. The output will have median ~0 for each sample.

    2. Scaling Mode (add_global_median=True):
        Centers each sample around its median, then adds back the global median
        across all samples. This aligns samples to a common reference point
        while preserving the overall intensity distribution.

        Mathematical Formulation:
            X_centered = X - median(X, axis=1, keepdims=True) + global_median
            where global_median = median(X)

        This mode is useful when you want to remove sample-specific bias
        but maintain the overall data scale. The output will have median
        approximately equal to the global median for all samples.

    Parameters
    ----------
    container : ScpContainer
        ScpContainer containing the data.
    assay_name : str, default="protein"
        Name of the assay to process.
    source_layer : str, default="raw"
        Name of the layer to normalize.
    new_layer_name : Optional[str], default="median_centered"
        Name for the new normalized layer.
    add_global_median : bool, default=False
        If True, add global median after centering (scaling mode).
        If False, only center each sample (centering mode).

    Returns
    -------
    ScpContainer
        ScpContainer with added normalized layer.

    Raises
    ------
    AssayNotFoundError
        If the specified assay does not exist.
    LayerNotFoundError
        If the specified layer does not exist in the assay.

    Examples
    --------
    >>> import polars as pl
    >>> from scptensor.core.structures import ScpContainer, Assay, ScpMatrix
    >>> import numpy as np
    >>> container = ScpContainer(obs=pl.DataFrame({'_index': ['s1', 's2']}))
    >>> assay = Assay(var=pl.DataFrame({'_index': ['p1', 'p2']}))
    >>> assay.add_layer('raw', ScpMatrix(X=np.array([[1, 2], [3, 4]])))
    >>> container.add_assay('protein', assay)
    >>> result = norm_median(container)
    >>> 'median_centered' in result.assays['protein'].layers
    True
    >>> # Scaling mode: add global median back
    >>> result2 = norm_median(container, add_global_median=True)
    """
    if assay_name not in container.assays:
        available = ", ".join(f"'{k}'" for k in container.assays.keys())
        raise AssayNotFoundError(assay_name, hint=f"Available assays: {available}.")
    assay = container.assays[assay_name]
    if source_layer not in assay.layers:
        available = ", ".join(f"'{k}'" for k in assay.layers.keys())
        raise LayerNotFoundError(source_layer, assay_name, hint=f"Available layers: {available}.")
    input_layer = assay.layers[source_layer]

    X_centered = input_layer.X - np.nanmedian(input_layer.X, axis=1, keepdims=True)
    if add_global_median:
        X_centered = X_centered + np.nanmedian(input_layer.X)
    layer_name = new_layer_name or "median_centered"

    assay.add_layer(
        layer_name,
        ScpMatrix(X=X_centered, M=input_layer.M.copy() if input_layer.M is not None else None),
    )

    container.log_operation(
        action="normalization_median_centering",
        params={
            "assay": assay_name,
            "source_layer": source_layer,
            "new_layer_name": layer_name,
            "add_global_median": add_global_median,
        },
        description=f"Median centering on layer '{source_layer}' -> '{layer_name}' "
        f"({'scaling mode' if add_global_median else 'centering mode'}).",
    )

    return container
