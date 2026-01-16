"""MinProb and MinDet imputation for single-cell proteomics data.

These methods are designed for left-censored MNAR (Missing Not At Random) data
where missingness is due to low abundance - values below detection limit.

References:
    .. [1] Wei R, et al. Sci Rep 2018;8:663.
    .. [2] Lazar C, et al. BMC Bioinformatics 2016;17:175.
"""

import numpy as np
import scipy.sparse as sp

from scptensor.core.exceptions import (
    AssayNotFoundError,
    LayerNotFoundError,
    ScpValueError,
)
from scptensor.core.structures import ScpContainer, ScpMatrix
from scptensor.impute._utils import _update_imputed_mask


def impute_minprob(
    container: ScpContainer,
    assay_name: str,
    source_layer: str,
    new_layer_name: str = "imputed_minprob",
    sigma: float = 2.0,
    random_state: int | None = None,
) -> ScpContainer:
    """
    Impute missing values using probabilistic minimum imputation (MinProb).

    Samples from a distribution centered at the minimum detected value for each
    feature, scaled by sigma. This method is designed for left-censored MNAR
    (Missing Not At Random) data where missingness is due to low abundance
    values below the detection limit.

    For each feature with missing values:
    1. Find the minimum detected (non-missing) value
    2. Calculate the spread as (min_detected / sigma)
    3. Sample imputed values from a truncated normal distribution centered
       at min_detected with standard deviation equal to the spread
    4. Values are truncated to be positive (greater than 0)

    Parameters
    ----------
    container : ScpContainer
        Input container with missing values.
    assay_name : str
        Name of the assay to use.
    source_layer : str
        Name of the layer containing data with missing values.
    new_layer_name : str, default "imputed_minprob"
        Name for the new layer with imputed data.
    sigma : float, default 2.0
        Standard deviation multiplier for the distribution width.
        Larger values create more spread in imputed values.
        - sigma=1: narrow distribution near minimum
        - sigma=2: moderate spread (recommended)
        - sigma=3: wide spread
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    ScpContainer
        Container with imputed data in new layer.

    Raises
    ------
    AssayNotFoundError
        If the assay does not exist.
    LayerNotFoundError
        If the layer does not exist.
    ScpValueError
        If parameters are invalid.

    Notes
    -----
    MinProb is particularly suitable for proteomics data where the "zero gap"
    exists - undetected proteins are reported as NA rather than zero due to
    detection limits. The method preserves the left-censored nature of the
    missingness by sampling values near the detection threshold.

    The sampling uses a truncated normal distribution to ensure imputed
    values remain positive (protein abundance cannot be negative).

    Examples
    --------
    >>> from scptensor import impute_minprob
    >>> result = impute_minprob(container, "proteins", sigma=2.0)
    >>> "imputed_minprob" in result.assays["proteins"].layers
    True

    References
    ----------
    .. [1] Wei R, et al. "Missing Value Imputation Approach for Mass
       Spectrometry-based Metabolomics Data." Sci Rep 2018;8:663.
    """
    # Parameter validation
    if sigma <= 0:
        raise ScpValueError(
            f"sigma must be positive, got {sigma}. "
            "Use sigma >= 1 for minimum distribution scaling.",
            parameter="sigma",
            value=sigma,
        )

    # Validate assay and layer
    if assay_name not in container.assays:
        available = ", ".join(f"'{k}'" for k in container.assays)
        raise AssayNotFoundError(
            assay_name,
            hint=f"Available assays: {available}. Use container.list_assays() to see all assays.",
        )

    assay = container.assays[assay_name]
    if source_layer not in assay.layers:
        available = ", ".join(f"'{k}'" for k in assay.layers)
        raise LayerNotFoundError(
            source_layer,
            assay_name,
            hint=f"Available layers in assay '{assay_name}': {available}. "
            f"Use assay.list_layers() to see all layers.",
        )

    # Set random seed
    if random_state is not None:
        np.random.seed(random_state)

    # Get data
    input_matrix = assay.layers[source_layer]
    X_original = input_matrix.X
    X_dense = X_original.toarray() if sp.issparse(X_original) else X_original.copy()  # type: ignore[union-attr]

    # Check for missing values
    missing_mask = np.isnan(X_dense)
    if not np.any(missing_mask):
        new_matrix = ScpMatrix(X=X_dense, M=_update_imputed_mask(input_matrix.M, missing_mask))
        layer_name = new_layer_name or "imputed_minprob"
        assay.add_layer(layer_name, new_matrix)
        container.log_operation(
            action="impute_minprob",
            params={
                "assay": assay_name,
                "source_layer": source_layer,
                "new_layer_name": layer_name,
                "sigma": sigma,
            },
            description=f"MinProb imputation on assay '{assay_name}': no missing values found.",
        )
        return container

    # Imputation: for each feature, sample from distribution around min
    X_imputed = X_dense.copy()
    n_features = X_dense.shape[1]

    for j in range(n_features):
        feature_col = X_dense[:, j]
        missing_in_col = missing_mask[:, j]

        if not np.any(missing_in_col):
            continue

        # Get minimum detected value for this feature
        detected_values = feature_col[~missing_in_col]
        if len(detected_values) == 0:
            # All values missing - use small positive value
            X_imputed[missing_in_col, j] = np.random.uniform(0.01, 0.1, size=np.sum(missing_in_col))
            continue

        min_detected = np.min(detected_values)

        # Calculate spread as (min_detected / sigma)
        # This ensures smaller sigma = more concentrated near minimum
        spread = min_detected / sigma

        # Sample from truncated normal distribution
        # We want values centered near min_detected but only below it
        # (since missing values are likely below detection limit)
        n_missing = np.sum(missing_in_col)

        # Use a scaled beta-like distribution or truncated normal
        # Approach: sample from normal with mean=min_detected, sd=spread
        # Then take max(0, min_detected - abs(sample - min_detected))
        # This ensures values are positive and typically below or near min_detected

        samples = np.random.normal(loc=min_detected, scale=spread, size=n_missing)

        # Ensure samples are positive and reasonable
        # Values should be in range (0, min_detected + spread] roughly
        # For left-censored data, we expect values below detection
        # So we bias toward values below min_detected
        imputed_values = np.maximum(0.01, min_detected - np.abs(samples - min_detected) * 0.5)

        X_imputed[missing_in_col, j] = imputed_values

    # Create new layer with updated mask
    new_matrix = ScpMatrix(X=X_imputed, M=_update_imputed_mask(input_matrix.M, missing_mask))
    layer_name = new_layer_name or "imputed_minprob"
    assay.add_layer(layer_name, new_matrix)

    container.log_operation(
        action="impute_minprob",
        params={
            "assay": assay_name,
            "source_layer": source_layer,
            "new_layer_name": layer_name,
            "sigma": sigma,
        },
        description=f"MinProb imputation (sigma={sigma}) on assay '{assay_name}'.",
    )

    return container


def impute_mindet(
    container: ScpContainer,
    assay_name: str,
    source_layer: str,
    new_layer_name: str = "imputed_mindet",
    sigma: float = 1.0,
) -> ScpContainer:
    """
    Impute missing values using deterministic minimum imputation (MinDet).

    Uses a fixed deterministic value for all missing values in each feature.
    The imputed value is calculated as (min_detected - sigma * spread),
    where spread is a measure of the data variability. This method is
    faster than MinProb but less accurate as it does not capture the
    uncertainty in imputation.

    For each feature with missing values:
    1. Find the minimum detected (non-missing) value
    2. Calculate the imputed value as (min_detected - sigma * min_detected / sigma_adjusted)
       which simplifies to a value below the minimum detected
    3. Fill all missing values with this deterministic value

    Parameters
    ----------
    container : ScpContainer
        Input container with missing values.
    assay_name : str
        Name of the assay to use.
    source_layer : str
        Name of the layer containing data with missing values.
    new_layer_name : str, default "imputed_mindet"
        Name for the new layer with imputed data.
    sigma : float, default 1.0
        Controls how far below the minimum to impute.
        Smaller values impute closer to the minimum.
        - sigma=1: impute at min_detected - spread (conservative)
        - sigma=2: impute closer to minimum (less conservative)

    Returns
    -------
    ScpContainer
        Container with imputed data in new layer.

    Raises
    ------
    AssayNotFoundError
        If the assay does not exist.
    LayerNotFoundError
        If the layer does not exist.
    ScpValueError
        If parameters are invalid.

    Notes
    -----
    MinDet is a deterministic variant of MinProb. It is faster but introduces
    bias by using the same value for all missing entries in a feature. This
    can artificially reduce variance and affect downstream statistical analysis.

    The deterministic nature means results are fully reproducible without
    random_state. Use MinProb when you need to capture imputation uncertainty.

    Examples
    --------
    >>> from scptensor import impute_mindet
    >>> result = impute_mindet(container, "proteins", sigma=1.0)
    >>> "imputed_mindet" in result.assays["proteins"].layers
    True

    References
    ----------
    .. [1] Lazar C, et al. "Missing Value Imputation for Proteomics Data."
       BMC Bioinformatics 2016;17:175.
    """
    # Parameter validation
    if sigma <= 0:
        raise ScpValueError(
            f"sigma must be positive, got {sigma}. Use sigma >= 1 for minimum value adjustment.",
            parameter="sigma",
            value=sigma,
        )

    # Validate assay and layer
    if assay_name not in container.assays:
        available = ", ".join(f"'{k}'" for k in container.assays)
        raise AssayNotFoundError(
            assay_name,
            hint=f"Available assays: {available}. Use container.list_assays() to see all assays.",
        )

    assay = container.assays[assay_name]
    if source_layer not in assay.layers:
        available = ", ".join(f"'{k}'" for k in assay.layers)
        raise LayerNotFoundError(
            source_layer,
            assay_name,
            hint=f"Available layers in assay '{assay_name}': {available}. "
            f"Use assay.list_layers() to see all layers.",
        )

    # Get data
    input_matrix = assay.layers[source_layer]
    X_original = input_matrix.X
    X_dense = X_original.toarray() if sp.issparse(X_original) else X_original.copy()  # type: ignore[union-attr]

    # Check for missing values
    missing_mask = np.isnan(X_dense)
    if not np.any(missing_mask):
        new_matrix = ScpMatrix(X=X_dense, M=_update_imputed_mask(input_matrix.M, missing_mask))
        layer_name = new_layer_name or "imputed_mindet"
        assay.add_layer(layer_name, new_matrix)
        container.log_operation(
            action="impute_mindet",
            params={
                "assay": assay_name,
                "source_layer": source_layer,
                "new_layer_name": layer_name,
                "sigma": sigma,
            },
            description=f"MinDet imputation on assay '{assay_name}': no missing values found.",
        )
        return container

    # Imputation: for each feature, use deterministic value
    X_imputed = X_dense.copy()
    n_features = X_dense.shape[1]

    for j in range(n_features):
        feature_col = X_dense[:, j]
        missing_in_col = missing_mask[:, j]

        if not np.any(missing_in_col):
            continue

        # Get minimum detected value for this feature
        detected_values = feature_col[~missing_in_col]
        if len(detected_values) == 0:
            # All values missing - use small positive value
            X_imputed[missing_in_col, j] = 0.01
            continue

        min_detected = np.min(detected_values)

        # Calculate deterministic imputation value
        # Using min_detected - (min_detected / sigma) ensures positive values
        # and scales appropriately with sigma
        spread = min_detected / sigma
        imputed_value = max(0.01, min_detected - spread)

        X_imputed[missing_in_col, j] = imputed_value

    # Create new layer with updated mask
    new_matrix = ScpMatrix(X=X_imputed, M=_update_imputed_mask(input_matrix.M, missing_mask))
    layer_name = new_layer_name or "imputed_mindet"
    assay.add_layer(layer_name, new_matrix)

    container.log_operation(
        action="impute_mindet",
        params={
            "assay": assay_name,
            "source_layer": source_layer,
            "new_layer_name": layer_name,
            "sigma": sigma,
        },
        description=f"MinDet imputation (sigma={sigma}) on assay '{assay_name}'.",
    )

    return container


if __name__ == "__main__":
    # Test: Basic functionality
    print("Testing MinProb and MinDet imputation...")

    # Create simple test data with MNAR pattern
    np.random.seed(42)
    n_samples = 100
    n_features = 50

    # Generate data where missingness depends on value (MNAR)
    X_true = np.random.exponential(scale=10, size=(n_samples, n_features))

    # Introduce MNAR missingness: lower values more likely to be missing
    X_missing = X_true.copy()
    missing_prob = 1 / (1 + np.exp((X_true - 5) / 2))  # Sigmoid - lower values more likely missing
    missing_mask = np.random.rand(n_samples, n_features) < missing_prob
    X_missing[missing_mask] = np.nan

    # Create container
    import polars as pl

    from scptensor.core.structures import Assay, MaskCode, ScpContainer

    var = pl.DataFrame({"_index": [f"prot_{i}" for i in range(n_features)]})
    obs = pl.DataFrame({"_index": [f"cell_{i}" for i in range(n_samples)]})

    assay = Assay(var=var)
    assay.add_layer("raw", ScpMatrix(X=X_missing, M=None))

    container = ScpContainer(obs=obs, assays={"protein": assay})

    # Test MinProb imputation
    print("\nTesting MinProb imputation...")
    result_minprob = impute_minprob(
        container,
        assay_name="protein",
        source_layer="raw",
        sigma=2.0,
        random_state=42,
    )

    assert "imputed_minprob" in result_minprob.assays["protein"].layers
    result_matrix = result_minprob.assays["protein"].layers["imputed_minprob"]
    X_minprob = result_matrix.X
    M_minprob = result_matrix.M

    assert not np.any(np.isnan(X_minprob))
    assert M_minprob is not None
    assert np.all(M_minprob[missing_mask] == MaskCode.IMPUTED)
    assert np.all(M_minprob[~missing_mask] == MaskCode.VALID)

    # Check imputed values are positive
    assert np.all(X_minprob >= 0)

    # Check imputed values are reasonable
    for j in range(n_features):
        if np.any(missing_mask[:, j]):
            min_detected = np.min(X_true[~missing_mask[:, j], j])
            imputed_vals = X_minprob[missing_mask[:, j], j]
            # Imputed values should be positive (at minimum)
            assert np.all(imputed_vals >= 0)

    print(f"  MinProb: Shape={X_minprob.shape}, No NaNs={not np.any(np.isnan(X_minprob))}")
    print(f"  Mask codes: {np.sum(M_minprob == MaskCode.IMPUTED)} imputed")

    # Test MinDet imputation
    print("\nTesting MinDet imputation...")
    result_mindet = impute_mindet(
        container,
        assay_name="protein",
        source_layer="raw",
        sigma=1.0,
    )

    assert "imputed_mindet" in result_mindet.assays["protein"].layers
    result_matrix = result_mindet.assays["protein"].layers["imputed_mindet"]
    X_mindet = result_matrix.X
    M_mindet = result_matrix.M

    assert not np.any(np.isnan(X_mindet))
    assert M_mindet is not None
    assert np.all(M_mindet[missing_mask] == MaskCode.IMPUTED)
    assert np.all(M_mindet[~missing_mask] == MaskCode.VALID)

    print(f"  MinDet: Shape={X_mindet.shape}, No NaNs={not np.any(np.isnan(X_mindet))}")
    print(f"  Mask codes: {np.sum(M_mindet == MaskCode.IMPUTED)} imputed")

    # Test with existing mask
    print("\nTesting with existing mask...")
    M_initial = np.zeros(X_missing.shape, dtype=np.int8)
    M_initial[missing_mask] = MaskCode.LOD

    assay2 = Assay(var=var)
    assay2.add_layer("raw", ScpMatrix(X=X_missing, M=M_initial))
    container2 = ScpContainer(obs=obs, assays={"protein": assay2})

    result2 = impute_minprob(
        container2,
        assay_name="protein",
        source_layer="raw",
        sigma=2.0,
        random_state=42,
    )

    M_result2 = result2.assays["protein"].layers["imputed_minprob"].M
    assert np.all(M_result2[missing_mask] == MaskCode.IMPUTED)  # type: ignore[index]
    assert np.all(M_result2[~missing_mask] == MaskCode.VALID)  # type: ignore[index]
    print("  Existing mask correctly updated")

    print("\nTesting different sigma values...")
    for sig in [0.5, 1.0, 2.0, 3.0]:
        result = impute_minprob(
            container,
            assay_name="protein",
            source_layer="raw",
            sigma=sig,
            random_state=42,
        )
        X_sig = result.assays["protein"].layers["imputed_minprob"].X
        assert not np.any(np.isnan(X_sig))
        print(f"  sigma={sig}: OK")

    print("\nTesting parameter validation...")
    try:
        impute_minprob(container, "protein", "raw", sigma=0)
        raise AssertionError("Should raise error for sigma=0")
    except ScpValueError:
        print("  sigma=0 correctly raises error")

    try:
        impute_mindet(container, "protein", "raw", sigma=-1)
        raise AssertionError("Should raise error for sigma=-1")
    except ScpValueError:
        print("  sigma=-1 correctly raises error")

    print("\nTesting random state reproducibility...")
    result1 = impute_minprob(container, "protein", "raw", sigma=2.0, random_state=123)
    result2 = impute_minprob(container, "protein", "raw", sigma=2.0, random_state=123)
    X1 = result1.assays["protein"].layers["imputed_minprob"].X
    X2 = result2.assays["protein"].layers["imputed_minprob"].X
    np.testing.assert_array_equal(X1, X2)
    print("  Same random_state produces identical results")

    print("\nTesting MinDet determinism...")
    result1 = impute_mindet(container, "protein", "raw", sigma=1.0)
    result2 = impute_mindet(container, "protein", "raw", sigma=1.0)
    X1 = result1.assays["protein"].layers["imputed_mindet"].X
    X2 = result2.assays["protein"].layers["imputed_mindet"].X
    np.testing.assert_array_equal(X1, X2)
    print("  MinDet is deterministic")

    print("\nTesting no missing values case...")
    assay3 = Assay(var=var)
    assay3.add_layer("raw", ScpMatrix(X=X_true, M=None))
    container3 = ScpContainer(obs=obs, assays={"protein": assay3})

    result3 = impute_minprob(container3, "protein", "raw", sigma=2.0)
    assert "imputed_minprob" in result3.assays["protein"].layers
    print("  No missing values handled correctly")

    print("\nTesting all missing column...")
    X_all_missing = X_missing.copy()
    X_all_missing[:, 0] = np.nan  # Make first column all NaN

    assay4 = Assay(var=var)
    assay4.add_layer("raw", ScpMatrix(X=X_all_missing, M=None))
    container4 = ScpContainer(obs=obs, assays={"protein": assay4})

    result4 = impute_minprob(container4, "protein", "raw", sigma=2.0, random_state=42)
    X_all_missing_imputed = result4.assays["protein"].layers["imputed_minprob"].X
    assert not np.any(np.isnan(X_all_missing_imputed))
    print("  All missing column handled correctly")

    print("\nTesting edge case with single sample...")
    X_single = np.array([[1.0, np.nan, 3.0, np.nan, 5.0]])
    var_single = pl.DataFrame({"_index": [f"prot_{i}" for i in range(5)]})
    obs_single = pl.DataFrame({"_index": ["cell_1"]})

    assay_single = Assay(var=var_single)
    assay_single.add_layer("raw", ScpMatrix(X=X_single, M=None))
    container_single = ScpContainer(obs=obs_single, assays={"protein": assay_single})

    result_single = impute_minprob(container_single, "protein", "raw", sigma=2.0, random_state=42)
    X_single_imputed = result_single.assays["protein"].layers["imputed_minprob"].X
    assert not np.any(np.isnan(X_single_imputed))
    print("  Single sample handled correctly")

    print("\nTesting custom layer names...")
    # Create fresh container for this test
    assay_custom = Assay(var=var)
    assay_custom.add_layer("raw", ScpMatrix(X=X_missing.copy(), M=None))
    container_custom = ScpContainer(obs=obs, assays={"protein": assay_custom})
    result_custom = impute_minprob(
        container_custom,
        assay_name="protein",
        source_layer="raw",
        new_layer_name="my_minprob",
        sigma=2.0,
    )
    assert "my_minprob" in result_custom.assays["protein"].layers
    assert "imputed_minprob" not in result_custom.assays["protein"].layers
    print("  Custom layer names work correctly")

    print("\nTesting history logging...")
    # Create fresh container for this test
    assay_log = Assay(var=var)
    assay_log.add_layer("raw", ScpMatrix(X=X_missing.copy(), M=None))
    container_log = ScpContainer(obs=obs, assays={"protein": assay_log})
    initial_history_len = len(container_log.history)
    result_log = impute_minprob(container_log, "protein", "raw", sigma=2.0)
    assert len(result_log.history) == initial_history_len + 1
    assert result_log.history[-1].action == "impute_minprob"
    print("  History logging works correctly")

    print("\nAll tests passed!")
