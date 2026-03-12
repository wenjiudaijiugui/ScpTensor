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
from scipy.stats import rankdata

from scptensor.core.structures import ScpContainer

from .base import create_result_layer, ensure_dense, log_operation, validate_assay_and_layer


def _quantile_normalize_rows(X: np.ndarray) -> np.ndarray:  # noqa: N803
    """Quantile-normalize a dense matrix row-wise.

    Parameters
    ----------
    X : np.ndarray
        Input matrix of shape (n_rows, n_cols). Each row is normalized
        against the shared reference distribution.

    Returns
    -------
    np.ndarray
        Quantile-normalized matrix with NaN positions preserved.
    """
    n_rows, n_cols = X.shape

    # Step 1: Sort each row independently (ignoring NaN)
    X_sorted = np.empty_like(X)
    nan_mask = np.isnan(X)

    for i in range(n_rows):
        row = X[i, :]
        non_nan = ~nan_mask[i, :]

        if non_nan.sum() == 0:
            X_sorted[i, :] = np.nan
            continue

        row_valid = row[non_nan]
        sorted_row = np.sort(row_valid)
        X_sorted[i, :] = np.nan
        X_sorted[i, : sorted_row.size] = sorted_row

    # Step 2: Compute rank-wise shared reference
    valid_counts = np.sum(~np.isnan(X_sorted), axis=0)
    rank_sums = np.nansum(X_sorted, axis=0)
    reference_dist = np.divide(
        rank_sums,
        valid_counts,
        out=np.zeros_like(rank_sums, dtype=float),
        where=valid_counts > 0,
    )

    # Step 3: Map row values back by rank
    X_normalized = np.empty_like(X)

    for i in range(n_rows):
        row = X[i, :]
        non_nan = ~nan_mask[i, :]

        if non_nan.sum() == 0:
            X_normalized[i, :] = np.nan
            continue

        row_valid = row[non_nan]
        ranks = rankdata(row_valid, method="average") - 1  # 0-indexed

        normalized_values = np.interp(
            ranks,
            np.arange(len(reference_dist)),
            reference_dist,
        )

        X_normalized[i, non_nan] = normalized_values
        X_normalized[i, ~non_nan] = np.nan

    return X_normalized


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
        Tied values receive the average of their corresponding quantiles using
        ``scipy.stats.rankdata(method="average")``.

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
    - This implementation uses scipy.stats.rankdata for efficient rank
      computation with proper tie handling.
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
    # Validate and get objects
    assay, input_layer = validate_assay_and_layer(container, assay_name, source_layer)

    X = ensure_dense(input_layer.X).copy()
    X_normalized = _quantile_normalize_rows(X)

    # Create and add new layer
    new_matrix = create_result_layer(X_normalized, input_layer)
    assay.add_layer(new_layer_name, new_matrix)

    # Log operation
    log_operation(
        container,
        action="normalization_quantile",
        params={
            "assay": assay_name,
            "source_layer": source_layer,
            "new_layer_name": new_layer_name,
        },
        description=f"Quantile normalization on layer '{source_layer}' -> '{new_layer_name}'.",
    )

    return container


__all__ = ["norm_quantile"]
