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

from scptensor.core.exceptions import ValidationError
from scptensor.core.structures import ScpContainer

from .base import ensure_dense, finalize_normalization_layer, validate_layer_context


def _map_reference_by_average_rank(
    row_valid: np.ndarray,
    reference_dist: np.ndarray,
) -> np.ndarray:
    """Map one valid row to the shared reference using average-rank ties."""
    n_valid = row_valid.size
    order = np.argsort(row_valid, kind="mergesort")
    mapped = np.empty(n_valid, dtype=float)

    if n_valid <= 1:
        mapped[order] = reference_dist[:n_valid]
        return mapped

    sorted_vals = row_valid[order]
    if not np.any(sorted_vals[1:] == sorted_vals[:-1]):
        mapped[order] = reference_dist[:n_valid]
        return mapped

    group_starts_mask = np.empty(n_valid, dtype=bool)
    group_starts_mask[0] = True
    group_starts_mask[1:] = sorted_vals[1:] != sorted_vals[:-1]

    group_starts = np.flatnonzero(group_starts_mask)
    group_ends = np.empty_like(group_starts)
    group_ends[:-1] = group_starts[1:] - 1
    group_ends[-1] = n_valid - 1

    avg_ranks = 0.5 * (group_starts + group_ends)
    left = avg_ranks.astype(np.intp)
    right = np.ceil(avg_ranks).astype(np.intp)
    weights = avg_ranks - left
    group_values = reference_dist[left] + weights * (reference_dist[right] - reference_dist[left])

    normalized_sorted = np.repeat(group_values, group_ends - group_starts + 1)
    mapped[order] = normalized_sorted
    return mapped


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
    X = np.asarray(X, dtype=float)
    if np.any(np.isinf(X)):
        raise ValidationError(
            "Quantile normalization does not accept Inf values. "
            "Use NaN for missing entries or clean infinite intensities first.",
            field="X",
        )

    n_rows, n_cols = X.shape

    # Step 1: Accumulate the rank-wise shared reference without materializing
    # a full sorted matrix. This keeps the stable dense-output contract while
    # avoiding an extra X-sized temporary.
    rank_sums = np.zeros(n_cols, dtype=float)
    valid_counts = np.zeros(n_cols, dtype=np.int32)

    for i in range(n_rows):
        row = X[i, :]
        row_valid = row[~np.isnan(row)]
        n_valid = row_valid.size

        if n_valid == 0:
            continue

        sorted_row = np.sort(row_valid)
        rank_sums[:n_valid] += sorted_row
        valid_counts[:n_valid] += 1

    # Step 2: Compute rank-wise shared reference
    reference_dist = np.divide(
        rank_sums,
        valid_counts,
        out=np.zeros_like(rank_sums, dtype=float),
        where=valid_counts > 0,
    )
    # Step 3: Map row values back by rank
    X_normalized = np.empty(X.shape, dtype=float)

    for i in range(n_rows):
        row = X[i, :]
        non_nan = ~np.isnan(row)

        if not np.any(non_nan):
            X_normalized[i, :] = np.nan
            continue

        row_valid = row[non_nan]
        X_normalized[i, non_nan] = _map_reference_by_average_rank(row_valid, reference_dist)
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
    # Validate and get objects
    ctx = validate_layer_context(container, assay_name, source_layer)
    assay = ctx.assay
    input_layer = ctx.layer

    X = np.asarray(ensure_dense(input_layer.X), dtype=float)
    X_normalized = _quantile_normalize_rows(X)

    return finalize_normalization_layer(
        container,
        assay,
        input_layer,
        X=X_normalized,
        new_layer_name=new_layer_name,
        action="normalization_quantile",
        params={
            "assay": ctx.resolved_assay_name,
            "source_layer": source_layer,
            "new_layer_name": new_layer_name,
        },
        description=f"Quantile normalization on layer '{source_layer}' -> '{new_layer_name}'.",
    )


__all__ = ["norm_quantile"]
