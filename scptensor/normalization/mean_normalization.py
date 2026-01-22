"""Sample mean normalization for ScpTensor."""

import numpy as np

from scptensor.core.exceptions import (
    AssayNotFoundError,
    LayerNotFoundError,
)
from scptensor.core.structures import ScpContainer, ScpMatrix


def norm_mean(
    container: ScpContainer,
    assay_name: str = "protein",
    source_layer: str = "raw",
    new_layer_name: str | None = "sample_mean_norm",
    add_global_mean: bool = False,
) -> ScpContainer:
    """
    Subtract the mean of each sample with optional global mean restoration.

    This method provides two modes for mean-based normalization:

    1. Centering Mode (add_global_mean=False):
        Centers each sample around its mean value, removing sample-specific
        bias while preserving relative differences between features.

        Mathematical Formulation:
            X_centered = X - mean(X, axis=1, keepdims=True)

        This mode is useful for removing sample-specific technical bias
        while maintaining the relative differences between features within
        each sample. The output will have mean ~0 for each sample.

    2. Scaling Mode (add_global_mean=True):
        Centers each sample around its mean, then adds back the global mean
        across all samples. This aligns samples to a common reference point
        while preserving the overall intensity distribution.

        Mathematical Formulation:
            X_centered = X - mean(X, axis=1, keepdims=True) + global_mean
            where global_mean = mean(X)

        This mode is useful when you want to remove sample-specific bias
        but maintain the overall data scale. The output will have mean
        approximately equal to the global mean for all samples.

    **Note:** Mean normalization is more sensitive to outliers than median
    normalization. Use median normalization (norm_median) if your data
    contains extreme outliers.

    Parameters
    ----------
    container : ScpContainer
        ScpContainer containing the data.
    assay_name : str, default="protein"
        Name of the assay to process.
    source_layer : str, default="raw"
        Name of the layer to normalize.
    new_layer_name : Optional[str], default="sample_mean_norm"
        Name for the new normalized layer.
    add_global_mean : bool, default=False
        If True, add global mean after centering (scaling mode).
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
    >>> result = norm_mean(container)
    >>> 'sample_mean_norm' in result.assays['protein'].layers
    True
    >>> # Scaling mode: add global mean back
    >>> result2 = norm_mean(container, add_global_mean=True)
    """
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

    X_centered = input_layer.X - np.nanmean(input_layer.X, axis=1, keepdims=True)
    if add_global_mean:
        X_centered = X_centered + np.nanmean(input_layer.X)
    layer_name = new_layer_name or "sample_mean_norm"

    assay.add_layer(
        layer_name,
        ScpMatrix(X=X_centered, M=input_layer.M.copy() if input_layer.M is not None else None),
    )

    container.log_operation(
        action="normalization_sample_mean",
        params={
            "assay": assay_name,
            "source_layer": source_layer,
            "new_layer_name": layer_name,
            "add_global_mean": add_global_mean,
        },
        description=f"Sample mean normalization on layer '{source_layer}' -> '{layer_name}' "
        f"({'scaling mode' if add_global_mean else 'centering mode'}).",
    )

    return container
