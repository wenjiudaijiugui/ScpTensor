"""Quantile normalization for ScpTensor.

Reference:
    Bolstad, B. M., Irizarry, R. A., Astrand, M., & Speed, T. P. (2003).
    A comparison of normalization methods for high density oligonucleotide
    array data based on variance and bias assessment. BMC Bioinformatics, 4, 9.

This implementation forces all samples to have the same distribution by:
1. Sorting each column (sample)
2. Computing row means (reference distribution)
3. Mapping back to original ranks
"""

import numpy as np
from scipy.stats import rankdata

from scptensor.core.exceptions import (
    AssayNotFoundError,
    LayerNotFoundError,
)
from scptensor.core.structures import ScpContainer, ScpMatrix


def norm_quantile(
    container: ScpContainer,
    assay_name: str = "protein",
    source_layer: str = "raw",
    new_layer_name: str = "quantile_norm",
) -> ScpContainer:
    """
    Quantile normalization for single-cell proteomics data.

    Forces all samples to have the same distribution by matching their
    empirical distributions. This is a powerful normalization technique
    widely used in genomics and proteomics.

    **Important Note on Log Transformation:**
        It is strongly recommended to apply log2 transformation before
        quantile normalization. This ensures the data meets the normality
        assumptions and makes the distributions more comparable.
        Example:
            >>> from scptensor.normalization import norm_log
            >>> container = norm_log(container, base=2.0)
            >>> container = norm_quantile(container)

    Mathematical Formulation:
        Given matrix X with N proteins (rows) and M samples (cols):

        1. Sort each column:
           X_sorted[k, j] = k-th order statistic of column j

        2. Compute reference distribution:
           q_bar[k] = (1/M) * sum(X_sorted[k, :])

        3. Inverse mapping:
           For element x[i,j] with rank r[i,j] in column j:
           x_normalized[i,j] = q_bar[r[i,j]]

    **Tie Handling:**
        Tied values receive the average of their corresponding quantiles
        using scipy.stats.rankdata with method='average'.

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
        Name of the layer to normalize. Should be log-transformed data
        for best results.
    new_layer_name : str, default="quantile_norm"
        Name for the new normalized layer.

    Returns
    -------
    ScpContainer
        ScpContainer with added quantile normalized layer.

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
    - The algorithm is applied per-column (per-sample) to align all
      sample distributions.
    - Quantile normalization assumes most features are not differentially
      expressed, which is generally reasonable for large-scale proteomics
      data.
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
    X = input_layer.X.copy()

    # Convert sparse to dense if necessary
    import scipy.sparse as sp

    if sp.issparse(X):
        X = X.toarray()

    # Get dimensions
    n_samples, n_features = X.shape

    # Step 1: Sort each column (sample) independently
    X_sorted = np.empty_like(X)
    nan_mask = np.isnan(X)

    for j in range(n_features):
        col = X[:, j]
        non_nan = ~nan_mask[:, j]

        if non_nan.sum() == 0:
            # All NaN column
            X_sorted[:, j] = np.nan
            continue

        # Sort non-NaN values
        col_valid = col[non_nan]
        sorted_col = np.sort(col_valid)

        # Place sorted values back
        X_sorted[non_nan, j] = sorted_col
        X_sorted[~non_nan, j] = np.nan

    # Step 2: Compute reference distribution as row means
    # For rows with all NaN, set reference to 0
    row_means = np.nanmean(X_sorted, axis=1)
    all_nan_rows = np.isnan(row_means)
    reference_dist = np.copy(row_means)
    reference_dist[all_nan_rows] = 0.0

    # Step 3: Inverse mapping - assign reference values based on ranks
    X_normalized = np.empty_like(X)

    for j in range(n_features):
        col = X[:, j]
        non_nan = ~nan_mask[:, j]

        if non_nan.sum() == 0:
            # All NaN column - preserve NaN
            X_normalized[:, j] = np.nan
            continue

        # Get ranks for non-NaN values
        col_valid = col[non_nan]
        ranks = rankdata(col_valid, method="average") - 1  # Convert to 0-indexed

        # Map ranks to reference distribution
        # Use linear interpolation for fractional ranks from ties
        normalized_values = np.interp(
            ranks,
            np.arange(len(reference_dist)),
            reference_dist,
        )

        # Fill in the normalized values
        X_normalized[non_nan, j] = normalized_values
        X_normalized[~non_nan, j] = np.nan

    # Create new layer
    new_matrix = ScpMatrix(
        X=X_normalized,
        M=input_layer.M.copy() if input_layer.M is not None else None,
    )
    assay.add_layer(new_layer_name, new_matrix)

    container.log_operation(
        action="normalization_quantile",
        params={
            "assay": assay_name,
            "source_layer": source_layer,
            "new_layer_name": new_layer_name,
        },
        description=f"Quantile normalization on layer '{source_layer}' -> '{new_layer_name}'.",
    )

    return container
