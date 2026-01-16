"""QRILC imputation for single-cell proteomics data.

QRILC (Quantile Regression Imputation of Left-Censored Data) is designed
specifically for MNAR (Missing Not At Random) data where missingness is due
to low abundance (left-censored) values.

Reference:
    Wei R, et al. "Missing Value Imputation Approach for Mass Spectrometry-based
    Metabolomics Data." Sci Rep 2018;8:663.
"""

from typing import overload

import numpy as np
import scipy.sparse as sp
from scipy.stats import norm

from scptensor.core.exceptions import (
    AssayNotFoundError,
    LayerNotFoundError,
    ScpValueError,
)
from scptensor.core.structures import MaskCode, ScpContainer, ScpMatrix
from scptensor.impute._utils import _update_imputed_mask


@overload
def impute_qrilc(
    container: ScpContainer,
    assay_name: str,
    source_layer: str,
    new_layer_name: str = "qrilc",
    q: float = 0.01,
    random_state: int | None = None,
) -> ScpContainer: ...


def impute_qrilc(
    container: ScpContainer,
    assay_name: str,
    source_layer: str,
    new_layer_name: str = "qrilc",
    q: float = 0.01,
    random_state: int | None = None,
) -> ScpContainer:
    """Impute missing values using Quantile Regression Imputation of Left-Censored Data.

    Designed specifically for MNAR (Missing Not At Random) data where missingness
    is due to low abundance (left-censored). For each feature, estimates the
    distribution of detected values and samples from the left-censored tail
    to fill missing values.

    The algorithm assumes that missing values are below the detection limit
    (left-censored) and models this using a truncated normal distribution.
    For each feature:
        1. Estimate detection limit as the q-th quantile of detected values
        2. Fit normal distribution to detected values above this threshold
        3. Sample from left-censored distribution for missing values

    Parameters
    ----------
    container : ScpContainer
        Input container with missing values.
    assay_name : str
        Name of the assay to use.
    source_layer : str
        Name of the layer containing data with missing values.
    new_layer_name : str, default "qrilc"
        Name for the new layer with imputed data.
    q : float, default 0.01
        Quantile for defining left-censoring threshold (0-1).
        Lower values assume more aggressive censoring (smaller detection limit).
        Recommended: 0.01 to 0.05 for proteomics data.
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
    QRILC is recommended for MNAR missingness by multiple studies (Wei 2018).
    It preserves the tail distribution better than mean-based methods.

    The method assumes:
        - Missing values are due to low abundance (below detection limit)
        - Detected values follow an approximately normal distribution
        - Each feature has sufficient detected values for estimation

    For features with too few detected values (< 3), falls back to
    global feature minimum imputation.

    Examples
    --------
    >>> from scptensor import impute_qrilc
    >>> result = impute_qrilc(container, "proteins", "raw", q=0.01)
    >>> "qrilc" in result.assays["proteins"].layers
    True
    """
    # Parameter validation
    if not 0 < q < 1:
        raise ScpValueError(
            f"Quantile q must be between 0 and 1, got {q}. "
            "Use q in (0, 1) for left-censoring threshold.",
            parameter="q",
            value=q,
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
    X_original = input_matrix.X.copy()
    n_samples, n_features = X_original.shape

    # Convert sparse to dense for QRILC
    X_dense = X_original.toarray() if sp.issparse(X_original) else X_original

    # Check for missing values
    missing_mask = np.isnan(X_dense)
    if not np.any(missing_mask):
        new_matrix = ScpMatrix(X=X_dense, M=_update_imputed_mask(input_matrix.M, missing_mask))
        layer_name = new_layer_name or "qrilc"
        assay.add_layer(layer_name, new_matrix)
        container.log_operation(
            action="impute_qrilc",
            params={
                "assay": assay_name,
                "source_layer": source_layer,
                "new_layer_name": layer_name,
                "q": q,
            },
            description=f"QRILC imputation on assay '{assay_name}': no missing values found.",
        )
        return container

    # Set random seed
    if random_state is not None:
        np.random.seed(random_state)

    X_imputed = X_dense.copy()

    # For each feature, estimate distribution and impute missing values
    for feat_idx in range(n_features):
        feature_col = X_dense[:, feat_idx]
        missing_in_feat = missing_mask[:, feat_idx]
        detected_mask = ~missing_in_feat

        detected_values = feature_col[detected_mask]
        n_missing = np.sum(missing_in_feat)
        n_detected = len(detected_values)

        if n_missing == 0:
            # No missing values for this feature
            continue

        if n_detected < 3:
            # Not enough detected values to estimate distribution
            # Fall back to minimum detected value
            min_detected = np.min(detected_values) if n_detected > 0 else 0.0
            X_imputed[missing_in_feat, feat_idx] = min_detected
            continue

        # Estimate detection limit as q-th quantile
        detection_limit = np.quantile(detected_values, q)

        # Use only values above detection limit for distribution fitting
        above_threshold = detected_values >= detection_limit
        values_for_fit = detected_values[above_threshold]

        if len(values_for_fit) < 3:
            # Fallback: use all detected values
            values_for_fit = detected_values

        # Fit normal distribution to detected values
        mu = np.mean(values_for_fit)
        sigma = np.std(values_for_fit, ddof=1)

        # Ensure sigma is positive for sampling
        if sigma <= 0 or not np.isfinite(sigma):
            sigma = np.std(detected_values, ddof=1)
            if sigma <= 0 or not np.isfinite(sigma):
                # Use range-based sigma estimate
                sigma = (np.max(detected_values) - np.min(detected_values)) / 4
                if sigma <= 0:
                    sigma = 1.0

        # Sample from left-censored normal distribution
        # Use truncated normal: sample from normal, but keep values below threshold
        n_samples_to_generate = n_missing * 2  # Generate extra candidates
        samples = np.random.randn(n_samples_to_generate) * sigma + mu

        # Keep only values below detection limit (left-censored)
        left_censored = samples[samples < detection_limit]

        if len(left_censored) >= n_missing:
            # Use sampled left-censored values
            imputed_values = left_censored[:n_missing]
        else:
            # Not enough left-censored samples, supplement with uniform range
            # between min(detected) and detection limit
            min_detected = np.min(detected_values)
            if min_detected < detection_limit:
                supplement = np.random.uniform(
                    min_detected,
                    detection_limit,
                    size=n_missing - len(left_censored),
                )
                imputed_values = np.concatenate([left_censored, supplement])
            else:
                # Fallback to truncated normal sampling
                from scipy.stats import truncnorm

                # Standardize the truncation point
                a = (min_detected - mu) / sigma
                b = (detection_limit - mu) / sigma
                imputed_values = truncnorm.rvs(
                    a, b, loc=mu, scale=sigma, size=n_missing, random_state=random_state
                )

        # Ensure imputed values are non-negative (abundance can't be negative)
        imputed_values = np.maximum(imputed_values, 0)

        # Assign imputed values
        X_imputed[missing_in_feat, feat_idx] = imputed_values

    # Create new layer with updated mask
    new_matrix = ScpMatrix(X=X_imputed, M=_update_imputed_mask(input_matrix.M, missing_mask))
    layer_name = new_layer_name or "qrilc"
    assay.add_layer(layer_name, new_matrix)

    container.log_operation(
        action="impute_qrilc",
        params={
            "assay": assay_name,
            "source_layer": source_layer,
            "new_layer_name": layer_name,
            "q": q,
        },
        description=f"QRILC imputation (q={q}) on assay '{assay_name}'.",
    )

    return container


if __name__ == "__main__":
    # Test: Basic functionality
    print("Testing QRILC imputation...")

    # Create simple test data
    np.random.seed(42)
    n_samples = 100
    n_features = 50

    # Generate log-normal data (typical for proteomics)
    X_true = np.exp(np.random.randn(n_samples, n_features) * 0.5 + 2)

    # Add MNAR missingness (lower values more likely to be missing)
    X_missing = X_true.copy()
    missing_prob = 1 - norm.cdf(X_true, loc=np.mean(X_true), scale=np.std(X_true))
    missing_prob = (missing_prob - missing_prob.min()) / (missing_prob.max() - missing_prob.min())
    missing_mask = np.random.rand(n_samples, n_features) < missing_prob * 0.3
    X_missing[missing_mask] = np.nan

    # Create container
    import polars as pl

    from scptensor.core.structures import Assay

    var = pl.DataFrame({"_index": [f"prot_{i}" for i in range(n_features)]})
    obs = pl.DataFrame({"_index": [f"cell_{i}" for i in range(n_samples)]})

    assay = Assay(var=var)
    assay.add_layer("raw", ScpMatrix(X=X_missing, M=None))

    container = ScpContainer(obs=obs, assays={"protein": assay})

    # Test QRILC imputation
    result = impute_qrilc(
        container,
        assay_name="protein",
        source_layer="raw",
        q=0.01,
        random_state=42,
    )

    # Check results
    assert "qrilc" in result.assays["protein"].layers
    result_matrix = result.assays["protein"].layers["qrilc"]
    X_imputed = result_matrix.X
    M_imputed = result_matrix.M

    # Check no NaNs
    assert not np.any(np.isnan(X_imputed))

    # Check mask was created and updated correctly
    assert M_imputed is not None
    assert np.all(M_imputed[missing_mask] == MaskCode.IMPUTED)
    assert np.all(M_imputed[~missing_mask] == MaskCode.VALID)

    # Check imputation accuracy
    imputed_values = X_imputed[missing_mask]
    true_values = X_true[missing_mask]

    # Compute correlation for imputed values
    correlation = np.corrcoef(imputed_values, true_values)[0, 1]

    print(f"  Imputation correlation: {correlation:.3f}")
    print(f"  Shape: {X_imputed.shape}")
    print(f"  NaN count: {np.sum(np.isnan(X_imputed))}")
    print(f"  Mask code check: {np.sum(M_imputed == MaskCode.IMPUTED)} imputed values")
    print(f"  History log: {len(result.history)} entries")

    # Test 2: With existing mask
    print("\nTesting QRILC imputation with existing mask...")

    # Create initial mask with MBR (1) and LOD (2) codes
    M_initial = np.zeros(X_missing.shape, dtype=np.int8)
    M_initial[missing_mask] = np.where(
        np.random.rand(np.sum(missing_mask)) < 0.5, MaskCode.MBR, MaskCode.LOD
    )

    assay2 = Assay(var=var)
    assay2.add_layer("raw", ScpMatrix(X=X_missing, M=M_initial))
    container2 = ScpContainer(obs=obs, assays={"protein": assay2})

    result2 = impute_qrilc(
        container2,
        assay_name="protein",
        source_layer="raw",
        q=0.01,
        random_state=42,
    )

    result_matrix2 = result2.assays["protein"].layers["qrilc"]
    M_imputed2 = result_matrix2.M

    # Check that imputed values now have IMPUTED (5) code
    assert M_imputed2 is not None
    assert np.all(M_imputed2[missing_mask] == MaskCode.IMPUTED)
    # Check that valid values remain VALID (0)
    assert np.all(M_imputed2[~missing_mask] == MaskCode.VALID)

    print("  Existing mask correctly updated to IMPUTED code")
    print(f"  Mask code check: {np.sum(M_imputed2 == MaskCode.IMPUTED)} imputed values")

    # Test 3: Different q values
    print("\nTesting QRILC with different q values...")
    for q_val in [0.001, 0.01, 0.05]:
        result_q = impute_qrilc(
            container,
            assay_name="protein",
            source_layer="raw",
            q=q_val,
            random_state=42,
        )
        X_q = (
            result_q.assays["protein"].layers[f"qrilc_{q_val}"].X
            if f"qrilc_{q_val}" in result_q.assays["protein"].layers
            else result_q.assays["protein"].layers["qrilc"].X
        )
        print(f"  q={q_val}: mean(imputed)={np.mean(X_q[missing_mask]):.3f}")

    print("\nAll tests passed!")
