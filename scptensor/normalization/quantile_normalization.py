"""Quantile normalization for ScpTensor.

Reference:
    Bolstad, B. M., Irizarry, R. A., Astrand, M., & Speed, T. P. (2003).
    A comparison of normalization methods for high density oligonucleotide
    array data based on variance and bias assessment. BMC Bioinformatics, 4, 9.

Mathematical Formulation:
    Quantile normalization forces all samples to have the same distribution
    by matching their empirical distributions.

    .. math::

        q_k = \\frac{1}{M} \\sum_{j=1}^{M} X_{(k),j}

    where :math:`X_{(k),j}` is the k-th order statistic of column j.

    The normalized value for element :math:`x_{i,j}` with rank :math:`r_{i,j}`:

    .. math::

        x_{i,j}^{normalized} = q_{r_{i,j}}

    Ties are handled using the average rank method.
"""

import numpy as np

from scptensor.core._layer_processing import ensure_dense_matrix, write_result_layer_and_log
from scptensor.core._rank_normalization import (
    map_reference_by_average_rank,
    quantile_normalize_dense_rows,
)
from scptensor.core._structure_container import ScpContainer

from ._context import resolve_normalization_context


def _map_reference_by_average_rank(
    row_valid: np.ndarray,
    reference_dist: np.ndarray,
) -> np.ndarray:
    """Compatibility wrapper over the shared internal rank-mapping kernel."""
    return map_reference_by_average_rank(row_valid, reference_dist)


def _quantile_normalize_rows(X: np.ndarray) -> np.ndarray:
    """Compatibility wrapper over the shared internal quantile kernel."""
    return quantile_normalize_dense_rows(X)


def norm_quantile(
    container: ScpContainer,
    assay_name: str = "protein",
    source_layer: str = "raw",
    new_layer_name: str = "quantile_norm",
) -> ScpContainer:
    """Quantile normalization for DIA-based single-cell proteomics data.

    Forces all samples to have the same distribution by matching their
    empirical distributions. This is a powerful normalization technique
    widely used in genomics and proteomics.

    **Important Note on Log Transformation:**
        It is strongly recommended to apply log2 transformation before
        quantile normalization. This ensures the data meets the normality
        assumptions and makes the distributions more comparable.
        Example:
            >>> from scptensor.transformation import log_transform
            >>> container = log_transform(container, base=2.0)
            >>> container = norm_quantile(container)

    Mathematical Formulation:
        Given matrix X with N samples (rows) and M proteins (cols):

        .. math::

            q_k = \\frac{1}{N} \\sum_{i=1}^{N} X_{i,(k)}

        where :math:`X_{i,(k)}` is the k-th order statistic of sample i.

        For element :math:`x_{i,j}` with rank :math:`r_{i,j}` in sample i:

        .. math::

            x_{i,j}^{normalized} = q_{r_{i,j}}

    **Tie Handling:**
        Tied values receive the average-rank interpolated quantile value.

    **Missing Value Handling:**
        Only non-missing observations are used for computing quantiles.
        NaN values are preserved in their original positions.

    Parameters
    ----------
    container : ScpContainer
        ScpContainer with data to normalize.
    assay_name : str, default="protein"
        Name of the assay to process.
    source_layer : str, default="raw"
        Name of the layer to normalize. Input matrix is expected as
        (samples x proteins), and normalization is performed per-sample.
    new_layer_name : str, default="quantile_norm"
        Name for the new normalized layer.

    Returns
    -------
    ScpContainer
        Container with added quantile normalized layer.

    Raises
    ------
    AssayNotFoundError
        If the specified assay does not exist.
    LayerNotFoundError
        If the specified layer does not exist in the assay.

    Examples
    --------
    >>> import polars as pl
    >>> import numpy as np
    >>> from scptensor.core.structures import ScpContainer, Assay, ScpMatrix
    >>> # Create test data
    >>> obs = pl.DataFrame({'_index': ['s1', 's2', 's3']})
    >>> var = pl.DataFrame({'_index': ['p1', 'p2', 'p3', 'p4']})
    >>> X = np.array([[1.0, 2.0, 5.0, 4.0],
    ...               [2.0, 3.0, 6.0, 5.0],
    ...               [3.0, 4.0, 7.0, 6.0]])
    >>> assay = Assay(var=var, layers={'raw': ScpMatrix(X=X)})
    >>> container = ScpContainer(obs=obs, assays={'protein': assay})
    >>> # Apply quantile normalization
    >>> result = norm_quantile(container)
    >>> 'quantile_norm' in result.assays['protein'].layers
    True

    Notes
    -----
    - This implementation preserves average-rank tie semantics while
      optimizing the common no-tie path for continuous intensities.
    - The algorithm is applied per-row (each row is a sample) to align all
      sample distributions.
    - Quantile normalization assumes most features are not differentially
      expressed, which is generally reasonable for large-scale proteomics
      data.
    - Log-transform before applying for best results.
    - ScpTensor AutoSelect only compares this method automatically on layers
      with explicit log provenance (for example, a `log` layer created by
      :func:`scptensor.transformation.log_transform`).

    """
    ctx = resolve_normalization_context(container, assay_name, source_layer)
    assay = ctx.assay
    input_layer = ctx.layer

    X = np.asarray(ensure_dense_matrix(input_layer.X), dtype=float)
    X_normalized = _quantile_normalize_rows(X)

    return write_result_layer_and_log(
        container,
        assay,
        source_layer=input_layer,
        layer_name=new_layer_name,
        x=X_normalized,
        action="normalization_quantile",
        params={
            "assay": ctx.resolved_assay_name,
            "source_layer": source_layer,
            "new_layer_name": new_layer_name,
        },
        description=f"Quantile normalization on layer '{source_layer}' -> '{new_layer_name}'.",
    )


__all__ = ["norm_quantile"]
