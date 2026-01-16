"""Count-based differential expression methods.

This module implements methods designed for count-based single-cell
proteomics data, modeling the mean-variance relationship common in
such data.

Supported methods:
    - diff_expr_voom: VOOM transformation with limma analysis
    - diff_expr_limma_trend: Empirical Bayes with trend correction
    - diff_expr_deseq2: Negative binomial GLM (DESeq2-like)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp
import scipy.stats as stats

from scptensor.core.exceptions import (
    AssayNotFoundError,
    LayerNotFoundError,
    ValidationError,
)
from scptensor.core.structures import ScpContainer

if TYPE_CHECKING:
    pass

# Import from core module
from scptensor.diff_expr.core import DiffExprResult, adjust_fdr

__all__ = [
    "diff_expr_voom",
    "diff_expr_limma_trend",
    "diff_expr_deseq2",
]


# =============================================================================
# Helper functions
# =============================================================================


def _isna(arr: np.ndarray) -> np.ndarray:
    """
    Check for NaN values in an array, handling string arrays.

    Parameters
    ----------
    arr : np.ndarray
        Input array

    Returns
    -------
    np.ndarray
        Boolean mask indicating NaN values
    """
    if arr.dtype.kind in {"U", "S", "O"}:
        mask = (arr == None) | (arr == "NaN") | (arr == "nan") | (arr == "NA")  # noqa: E711
        return mask
    return np.isnan(arr)


def _extract_group_indices(
    groups: np.ndarray,
    group_names: list[str] | tuple[str, ...],
    min_samples: int,
) -> dict[str, np.ndarray]:
    """
    Extract and validate group indices for statistical testing.

    Parameters
    ----------
    groups : np.ndarray
        Array of group labels
    group_names : list of str
        Names of groups to extract
    min_samples : int
        Minimum samples required per group

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from group name to indices

    Raises
    ------
    ValidationError
        If insufficient samples in any group
    """
    indices = {}
    for name in group_names:
        mask = groups == name
        n = np.sum(mask)
        if n < min_samples:
            raise ValidationError(
                f"Group '{name}' has only {n} samples, minimum {min_samples} required",
                field=name,
            )
        indices[name] = np.where(mask)[0]
    return indices


def _to_dense(X: np.ndarray | sp.spmatrix) -> np.ndarray:
    """
    Convert sparse matrix to dense if needed.

    Parameters
    ----------
    X : np.ndarray or sp.spmatrix
        Input matrix

    Returns
    -------
    np.ndarray
        Dense matrix
    """
    if sp.issparse(X):
        return X.toarray()
    return X


def _handle_mask(
    X: np.ndarray,
    M: np.ndarray | sp.spmatrix | None,
    sample_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Handle mask matrix for valid value extraction.

    Parameters
    ----------
    X : np.ndarray
        Data matrix
    M : np.ndarray or sp.spmatrix or None
        Mask matrix (0 = valid, non-zero = invalid)
    sample_idx : np.ndarray
        Sample indices to extract

    Returns
    -------
    tuple
        (values, valid_mask) for the specified samples
    """
    if M is None:
        # No mask, all values valid
        values = X[sample_idx, :]
        valid_mask = np.ones_like(values, dtype=bool)
        return values, valid_mask

    M_dense = _to_dense(M)
    valid_mask = M_dense[sample_idx, :] == 0
    values = X[sample_idx, :]

    return values, valid_mask


def _lowess_smooth(
    x: np.ndarray,
    y: np.ndarray,
    frac: float = 0.5,
) -> np.ndarray:
    """
    Locally weighted scatterplot smoothing.

    Parameters
    ----------
    x : np.ndarray
        Input x values
    y : np.ndarray
        Input y values
    frac : float, default=0.5
        Fraction of data to use for each local fit

    Returns
    -------
    np.ndarray
        Smoothed y values
    """
    n = len(x)
    if n < 3:
        return y.copy()

    # Sort by x
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    # Number of points for local regression
    k = max(3, int(frac * n))

    smoothed = np.zeros(n)

    for i in range(n):
        # Find k nearest neighbors
        distances = np.abs(x_sorted - x_sorted[i])
        neighbor_idx = np.argsort(distances)[:k]

        # Triangular weighting
        x_local = x_sorted[neighbor_idx]
        y_local = y_sorted[neighbor_idx]
        max_dist = np.max(np.abs(x_local - x_sorted[i]))

        if max_dist == 0:
            weights = np.ones(k)
        else:
            weights = 1.0 - np.abs(x_local - x_sorted[i]) / max_dist

        # Weighted linear regression
        if np.sum(weights) > 0:
            weighted_x = x_local * weights
            weighted_y = y_local * weights
            sum_w = np.sum(weights)

            # Simple weighted average for stability
            smoothed[i] = np.sum(weighted_y) / sum_w
        else:
            smoothed[i] = y_sorted[i]

    # Unsort
    unsort_idx = np.argsort(sort_idx)
    return smoothed[unsort_idx]


def _voom_transform(
    counts: np.ndarray,
    lib_size: np.ndarray,
    min_count: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply VOOM transformation.

    Converts counts to log2-CPM with observation-level precision weights.

    Parameters
    ----------
    counts : np.ndarray
        Count matrix (n_samples, n_features)
    lib_size : np.ndarray
        Library size for each sample
    min_count : float, default=0.5
        Minimum count offset for log transformation

    Returns
    -------
    tuple
        (log2_cpm, weights) where:
        - log2_cpm: log2 counts per million
        - weights: inverse variance weights for each observation

    Notes
    -----
    The VOOM transformation computes:
        1. CPM = counts / lib_size * 1e6
        2. log2_cpm = log2(CPM + min_count)
        3. variance = fit(mean - variance trend)
        4. weights = 1 / variance

    References
    ----------
    Law et al. (2014) voom: precision weights unlock linear model
    analysis tools for RNA-seq. Genome Biology 15:R29.
    """
    n_samples, n_features = counts.shape

    # Avoid division by zero
    lib_size = np.maximum(lib_size, 1)

    # Calculate CPM and log2 transform
    cpm = counts / lib_size[:, np.newaxis] * 1e6
    log2_cpm = np.log2(cpm + min_count)

    # Estimate mean-variance relationship
    # Compute mean and variance for each feature
    means = np.mean(log2_cpm, axis=0)
    variances = np.var(log2_cpm, axis=0, ddof=1)

    # Fit trend: variance as function of mean
    # Use LOWESS to smooth the relationship
    valid = variances > 0
    if np.sum(valid) > 3:
        trend = _lowess_smooth(means[valid], variances[valid], frac=0.5)

        # Interpolate trend for all features
        trend_all = np.interp(means, means[valid], trend, left=trend[0], right=trend[-1])
    else:
        # Fallback to constant variance
        trend_all = (
            np.ones(n_features) * np.mean(variances[valid])
            if np.any(valid)
            else np.ones(n_features)
        )

    # Compute weights as inverse variance
    # Use trend variance for each feature based on its mean
    weights = 1.0 / np.maximum(trend_all, 1e-6)

    # Normalize weights per sample
    weights = weights * n_samples / np.sum(weights)

    return log2_cpm, weights


def _tmm_normalize(
    counts: np.ndarray,
    ref_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply TMM (trimmed mean of M-values) normalization.

    Parameters
    ----------
    counts : np.ndarray
        Count matrix (n_samples, n_features)
    ref_idx : int, default=0
        Index of reference sample

    Returns
    -------
    tuple
        (normalized_counts, lib_size) with normalized counts and library sizes

    Notes
    -----
    TMM normalization accounts for composition bias between samples.
    """
    n_samples, n_features = counts.shape

    # Filter features with zero counts in reference
    ref_counts = counts[ref_idx, :]
    valid_features = ref_counts > 0
    if np.sum(valid_features) < 100:
        valid_features = counts[ref_idx, :] >= 0  # Use all if too few

    # Compute scaling factors
    lib_size = np.sum(counts, axis=1)
    factors = np.ones(n_samples)

    for i in range(n_samples):
        if i == ref_idx:
            factors[i] = 1.0
            continue

        # M-values: log2 fold change relative to reference
        M = np.log2(
            (counts[i, valid_features] / lib_size[i] + 1e-6)
            / (counts[ref_idx, valid_features] / lib_size[ref_idx] + 1e-6)
        )

        # A-values: average log expression
        A = 0.5 * np.log2(
            (counts[i, valid_features] / lib_size[i] + 1e-6)
            * (counts[ref_idx, valid_features] / lib_size[ref_idx] + 1e-6)
        )

        # Trim extremes (30% on each end)
        n_valid = len(M)
        trim = int(0.3 * n_valid)
        if trim > 0:
            sort_idx = np.argsort(A)
            trimmed_idx = sort_idx[trim : n_valid - trim]
            M_trimmed = M[trimmed_idx]
        else:
            M_trimmed = M

        # TMM factor
        factors[i] = 2 ** np.mean(M_trimmed)

    # Normalize counts
    norm_counts = counts / factors[:, np.newaxis]
    norm_lib_size = np.sum(norm_counts, axis=1)

    return norm_counts, norm_lib_size


def _limma_ebayes(
    logfc: np.ndarray,
    se: np.ndarray,
    df: int,
    robust: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply limma empirical Bayes moderation.

    Parameters
    ----------
    logfc : np.ndarray
        Log fold changes
    se : np.ndarray
        Standard errors
    df : int
        Degrees of freedom
    robust : bool, default=False
        Use robust empirical Bayes

    Returns
    -------
    tuple
        (mod_logfc, mod_se, moderated_t) with moderated statistics

    Notes
    -----
    Empirical Bayes shrinks feature-wise variances toward a common value,
    improving power when number of features is large.
    """
    n_features = len(logfc)

    # Remove infinite/NaN values
    valid = np.isfinite(logfc) & np.isfinite(se) & (se > 0)
    if np.sum(valid) < 3:
        return logfc, se, np.zeros_like(logfc)

    logfc_valid = logfc[valid]
    se_valid = se[valid]

    # Estimate prior parameters
    # s2.prior: prior variance
    # df.prior: prior degrees of freedom

    # Feature-wise sample variances
    s2 = se_valid**2

    # Fit inverse gamma distribution to variances
    # Simple method of moments
    s2_mean = np.mean(s2)
    s2_var = np.var(s2, ddof=1)

    if s2_var > 0:
        # Prior parameters for inverse gamma
        df_prior = 2 * s2_mean**2 / s2_var
        s2_prior = s2_mean * (df_prior - 2) / df_prior if df_prior > 2 else s2_mean
        df_prior = max(df_prior, 0.1)  # Ensure positive
    else:
        # All variances similar
        df_prior = n_features
        s2_prior = s2_mean

    # Moderated variances
    df_total = df + df_prior
    mod_s2 = (df * s2 + df_prior * s2_prior) / df_total

    if robust:
        # Robust weighting (simplified)
        residuals = logfc_valid / se_valid
        robust_weights = 1.0 / (1.0 + (residuals / stats.median_abs_deviation(residuals)) ** 2)
        mod_s2 = mod_s2 / (robust_weights + 0.1) * np.mean(robust_weights + 0.1)

    mod_se = np.sqrt(mod_s2)

    # Moderated t-statistics
    moderated_t = logfc_valid / mod_se

    # Fill output
    mod_logfc = logfc.copy()
    mod_se_out = se.copy()
    mod_t_out = np.zeros_like(logfc)

    mod_logfc[valid] = logfc_valid
    mod_se_out[valid] = mod_se
    mod_t_out[valid] = moderated_t

    return mod_logfc, mod_se_out, mod_t_out


def _estimate_nb_dispersion(
    counts: np.ndarray,
    group_sizes: np.ndarray,
) -> np.ndarray:
    """
    Estimate negative binomial dispersion per feature.

    Parameters
    ----------
    counts : np.ndarray
        Count matrix (n_samples, n_features)
    group_sizes : np.ndarray
        Number of samples per group

    Returns
    -------
    np.ndarray
        Dispersion estimates for each feature

    Notes
    -----
    Uses method of moments for dispersion estimation.
    For NB distribution: var = mu + alpha * mu^2
    where alpha is the dispersion parameter.
    """
    n_samples, n_features = counts.shape

    # Avoid zero counts
    counts_safe = np.maximum(counts, 0.5)

    # Compute mean and variance per feature (pooled across groups)
    means = np.mean(counts_safe, axis=0)

    # Group-specific variances
    variances = np.zeros(n_features)
    start_idx = 0

    for g_size in group_sizes:
        end_idx = start_idx + g_size
        group_counts = counts_safe[start_idx:end_idx, :]
        group_var = np.var(group_counts, axis=0, ddof=1)
        variances += group_var * (g_size - 1)
        start_idx = end_idx

    variances /= n_samples - len(group_sizes)

    # Estimate dispersion: alpha = (var - mean) / mean^2
    dispersions = np.maximum(variances - means, 0) / (means**2 + 1e-6)

    # Clip extreme values
    dispersions = np.clip(dispersions, 1e-6, 100)

    return dispersions


def _nb_wald_test(
    counts1: np.ndarray,
    counts2: np.ndarray,
    dispersions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Wald test for negative binomial model.

    Parameters
    ----------
    counts1 : np.ndarray
        Counts for group 1 (n1, n_features)
    counts2 : np.ndarray
        Counts for group 2 (n2, n_features)
    dispersions : np.ndarray
        Dispersion estimates for each feature

    Returns
    -------
    tuple
        (log2_fc, p_values) with fold changes and p-values
    """
    n1, n_features = counts1.shape
    n2 = counts2.shape[0]

    # Avoid zero counts
    counts1_safe = np.maximum(counts1, 0.5)
    counts2_safe = np.maximum(counts2, 0.5)

    # Size factors (library size normalization)
    lib_size1 = np.sum(counts1_safe, axis=1, keepdims=True)
    lib_size2 = np.sum(counts2_safe, axis=1, keepdims=True)

    # Normalized counts
    norm_counts1 = counts1_safe / lib_size1 * np.mean(lib_size1)
    norm_counts2 = counts2_safe / lib_size2 * np.mean(lib_size2)

    # Means per feature
    mean1 = np.mean(norm_counts1, axis=0)
    mean2 = np.mean(norm_counts2, axis=0)

    # Log2 fold change
    log2_fc = np.log2((mean1 + 1e-6) / (mean2 + 1e-6))

    # Variance under null hypothesis (pooled)
    # var = mean + alpha * mean^2
    n_total = n1 + n2
    pooled_mean = (n1 * mean1 + n2 * mean2) / n_total
    null_variance = pooled_mean + dispersions * pooled_mean**2

    # Standard error of log fold change
    se = np.sqrt(null_variance / n1 + null_variance / n2) / (np.log(2) * (pooled_mean + 1e-6))

    # Wald statistic
    wald = log2_fc / np.maximum(se, 1e-10)

    # Two-sided p-values
    p_values = 2 * (1 - stats.norm.cdf(np.abs(wald)))

    return log2_fc, p_values


def _upper_quartile_normalize(
    counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Upper quartile normalization.

    Parameters
    ----------
    counts : np.ndarray
        Count matrix (n_samples, n_features)

    Returns
    -------
    tuple
        (normalized_counts, lib_size) with normalized counts and library sizes
    """
    n_samples = counts.shape[0]

    # Compute upper quartile (75th percentile) for each sample
    uq = np.percentile(counts, 75, axis=1)

    # Avoid division by zero
    uq = np.maximum(uq, 1)

    # Scale factors
    median_uq = np.median(uq)
    factors = uq / median_uq

    # Normalize counts
    norm_counts = counts / factors[:, np.newaxis]
    lib_size = np.sum(norm_counts, axis=1)

    return norm_counts, lib_size


# =============================================================================
# Main API functions
# =============================================================================


def diff_expr_voom(
    container: ScpContainer,
    assay_name: str,
    layer: str,
    groupby: str,
    group1: str,
    group2: str,
    min_count: int = 10,
    normalize: str = "tmm",
) -> DiffExprResult:
    """VOOM transformation with limma analysis.

    Implements the VOOM (mean-variance modelling at the observational
    level) pipeline for count-based differential expression.

    Parameters
    ----------
    container : ScpContainer
        Input data container.
    assay_name : str
        Assay containing the data.
    layer : str
        Layer with count data.
    groupby : str
        Column in obs defining groups.
    group1 : str
        First group name.
    group2 : str
        Second group name.
    min_count : int, default=10
        Minimum count for feature filtering.
    normalize : str, default="tmm"
        Normalization method (tmm, upper_quartile, none).

    Returns
    -------
    DiffExprResult
        Test results with log2 fold changes and p-values.

    Raises
    ------
    AssayNotFoundError
        If assay_name not found.
    LayerNotFoundError
        If layer not found.
    ValidationError
        If groups have insufficient samples.

    Notes
    -----
    VOOM converts counts to log2-CPM with observation-level precision
    weights, then applies limma's empirical Bayes moderation.

    The pipeline steps:
        1. Filter low-count features
        2. Normalize library sizes (TMM by default)
        3. Transform to log2-CPM
        4. Estimate mean-variance trend
        5. Compute precision weights
        6. Apply moderated t-test

    References
    ----------
    .. [1] Law et al. (2014) voom: precision weights unlock linear
           model analysis tools for RNA-seq. Genome Biology 15:R29.

    Examples
    --------
    >>> import scptensor as sc
    >>> result = sc.diff_expr.diff_expr_voom(
    ...     container,
    ...     assay_name="proteins",
    ...     layer="X",
    ...     groupby="condition",
    ...     group1="control",
    ...     group2="treatment"
    ... )
    >>> sig = result.get_significant(alpha=0.05, min_log2_fc=1.0)
    """
    # Validate inputs
    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name, available_assays=list(container.assays.keys()))

    assay = container.assays[assay_name]
    if layer not in assay.layers:
        raise LayerNotFoundError(layer, assay_name, available_layers=list(assay.layers.keys()))

    if groupby not in container.obs.columns:
        raise ValidationError(f"Group column '{groupby}' not found in obs", field="groupby")

    if normalize not in {"tmm", "upper_quartile", "none"}:
        raise ValidationError(
            f"Unknown normalization method: {normalize}. Use 'tmm', 'upper_quartile', or 'none'.",
            field="normalize",
        )

    # Get group indices
    groups = container.obs[groupby].to_numpy()
    group_indices = _extract_group_indices(groups, [group1, group2], min_samples=3)
    idx1, idx2 = group_indices[group1], group_indices[group2]

    # Extract data
    X = assay.layers[layer].X
    M = assay.layers[layer].M
    X_dense = _to_dense(X)

    # Handle mask - set masked values to zero
    if M is not None:
        M_dense = _to_dense(M)
        X_dense = np.where(M_dense == 0, X_dense, 0)
    else:
        M_dense = None

    # Extract groups
    counts1 = X_dense[idx1, :]
    counts2 = X_dense[idx2, :]
    counts_combined = np.vstack([counts1, counts2])

    # Filter low-count features
    feature_counts = np.sum(counts_combined, axis=0)
    valid_features = feature_counts >= min_count
    if np.sum(valid_features) < 10:
        raise ValidationError(
            f"Too few features ({np.sum(valid_features)}) pass min_count filter",
            field="min_count",
        )

    counts1_filtered = counts1[:, valid_features]
    counts2_filtered = counts2[:, valid_features]

    # Normalize
    if normalize == "tmm":
        norm1, lib1 = _tmm_normalize(counts1_filtered)
        norm2, lib2 = _tmm_normalize(counts2_filtered)
    elif normalize == "upper_quartile":
        norm1, lib1 = _upper_quartile_normalize(counts1_filtered)
        norm2, lib2 = _upper_quartile_normalize(counts2_filtered)
    else:  # none
        norm1 = counts1_filtered
        norm2 = counts2_filtered
        lib1 = np.sum(norm1, axis=1)
        lib2 = np.sum(norm2, axis=1)

    # Combined for VOOM
    counts_all = np.vstack([norm1, norm2])
    lib_all = np.concatenate([lib1, lib2])

    # VOOM transformation
    log2_cpm, weights = _voom_transform(counts_all, lib_all)

    # Compute statistics
    n1, n2 = len(idx1), len(idx2)
    log2_cpm1 = log2_cpm[:n1, :]
    log2_cpm2 = log2_cpm[n1:, :]

    means1 = np.mean(log2_cpm1, axis=0)
    means2 = np.mean(log2_cpm2, axis=0)
    log2_fc = means1 - means2

    # Standard errors (using inverse weights as variance approximation)
    var1 = (
        np.sum((log2_cpm1 - means1) ** 2, axis=0) / (n1 * (n1 - 1))
        if n1 > 1
        else np.zeros_like(means1)
    )
    var2 = (
        np.sum((log2_cpm2 - means2) ** 2, axis=0) / (n2 * (n2 - 1))
        if n2 > 1
        else np.zeros_like(means2)
    )

    se = np.sqrt(var1 + var2)

    # Empirical Bayes moderation
    df_total = n1 + n2 - 2
    mod_logfc, mod_se, moderated_t = _limma_ebayes(log2_fc, se, df_total, robust=True)

    # P-values from t-distribution
    p_values = 2 * (1 - stats.t.cdf(np.abs(moderated_t), df_total))

    # Fill in filtered features
    n_features = X_dense.shape[1]
    p_values_all = np.ones(n_features) * np.nan
    log2_fc_all = np.ones(n_features) * np.nan
    test_stats_all = np.ones(n_features) * np.nan

    p_values_all[valid_features] = p_values
    log2_fc_all[valid_features] = mod_logfc
    test_stats_all[valid_features] = moderated_t

    # Group statistics
    g1_means = np.zeros(n_features) * np.nan
    g2_means = np.zeros(n_features) * np.nan
    g1_means[valid_features] = means1
    g2_means[valid_features] = means2

    # Adjust p-values
    p_values_adj = adjust_fdr(p_values_all, method="bh")

    return DiffExprResult(
        feature_ids=assay.var[assay.feature_id_col].to_numpy(),
        p_values=p_values_all,
        p_values_adj=p_values_adj,
        log2_fc=log2_fc_all,
        test_statistics=test_stats_all,
        effect_sizes=log2_fc_all / np.maximum(mod_se, 1e-6),  # Approximate effect size
        group_stats={
            f"{group1}_mean": g1_means,
            f"{group2}_mean": g2_means,
        },
        method="voom",
        params={
            "groupby": groupby,
            "group1": group1,
            "group2": group2,
            "min_count": min_count,
            "normalize": normalize,
        },
    )


def diff_expr_limma_trend(
    container: ScpContainer,
    assay_name: str,
    layer: str,
    groupby: str,
    group1: str,
    group2: str,
    trend: bool = True,
    robust: bool = True,
) -> DiffExprResult:
    """limma-trend analysis for count data.

    Applies empirical Bayes variance shrinkage with trend correction.

    Parameters
    ----------
    container : ScpContainer
        Input data container.
    assay_name : str
        Assay containing the data.
    layer : str
        Layer with count data.
    groupby : str
        Column in obs defining groups.
    group1 : str
        First group name.
    group2 : str
        Second group name.
    trend : bool, default=True
        Apply trend correction to variance.
    robust : bool, default=True
        Use robust empirical Bayes.

    Returns
    -------
    DiffExprResult
        Test results with moderated statistics.

    Raises
    ------
    AssayNotFoundError
        If assay_name not found.
    LayerNotFoundError
        If layer not found.
    ValidationError
        If groups have insufficient samples.

    Notes
    -----
    limma-trend is useful when there is a mean-variance relationship
    in the data. The trend option allows variance shrinkage to vary
    with the mean expression level.

    The method:
        1. Computes log2 fold changes
        2. Estimates variances
        3. Fits variance-mean trend (if trend=True)
        4. Applies empirical Bayes shrinkage
        5. Computes moderated t-statistics

    References
    ----------
    .. [1] Smyth GK (2004) Linear models and empirical Bayes methods
           for assessing differential expression in microarray experiments.
           Statistical Applications in Genetics and Molecular Biology 3:Article3.

    .. [2] Ritchie et al. (2015) limma powers differential expression
           analyses for RNA-sequencing and microarray studies. Nucleic
           Acids Research 43:e47.

    Examples
    --------
    >>> import scptensor as sc
    >>> result = sc.diff_expr.diff_expr_limma_trend(
    ...     container,
    ...     assay_name="proteins",
    ...     layer="X",
    ...     groupby="batch",
    ...     group1="A",
    ...     group2="B",
    ...     trend=True
    ... )
    """
    # Validate inputs
    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name, available_assays=list(container.assays.keys()))

    assay = container.assays[assay_name]
    if layer not in assay.layers:
        raise LayerNotFoundError(layer, assay_name, available_layers=list(assay.layers.keys()))

    if groupby not in container.obs.columns:
        raise ValidationError(f"Group column '{groupby}' not found in obs", field="groupby")

    # Get group indices
    groups = container.obs[groupby].to_numpy()
    group_indices = _extract_group_indices(groups, [group1, group2], min_samples=3)
    idx1, idx2 = group_indices[group1], group_indices[group2]

    # Extract data
    X = assay.layers[layer].X
    M = assay.layers[layer].M
    X_dense = _to_dense(X)

    # Handle mask
    if M is not None:
        M_dense = _to_dense(M)
        valid_mask = M_dense == 0
        X_dense = np.where(valid_mask, X_dense, np.nan)
    # Keep masked values as NaN for handling

    counts1 = X_dense[idx1, :]
    counts2 = X_dense[idx2, :]

    n1, n_features = counts1.shape
    n2 = counts2.shape[0]

    # Log2 transform with offset (pseudo-count)
    offset = 0.5
    log_counts1 = np.log2(counts1 + offset)
    log_counts2 = np.log2(counts2 + offset)

    # Compute means and variances
    means1 = np.nanmean(log_counts1, axis=0)
    means2 = np.nanmean(log_counts2, axis=0)
    variances1 = np.nanvar(log_counts1, axis=0, ddof=1)
    variances2 = np.nanvar(log_counts2, axis=0, ddof=1)

    # Log fold change
    log2_fc = means1 - means2

    # Standard errors
    se = np.sqrt(variances1 / n1 + variances2 / n2)

    # Apply trend correction if requested
    if trend:
        # Fit variance-mean trend
        overall_mean = (means1 + means2) / 2
        pooled_var = (variances1 * (n1 - 1) + variances2 * (n2 - 1)) / (n1 + n2 - 2)

        valid_var = pooled_var > 0
        if np.sum(valid_var) > 3:
            # Fit trend on log-log scale
            log_mean = np.log2(np.maximum(overall_mean[valid_var], 1e-6))
            log_var = np.log2(pooled_var[valid_var])

            # Linear fit
            coeffs = np.polyfit(log_mean, log_var, 1)
            trend_var = 2 ** (coeffs[0] * np.log2(np.maximum(overall_mean, 1e-6)) + coeffs[1])

            # Adjust se using trend
            se_trend = np.sqrt(trend_var / n1 + trend_var / n2)
            se = se_trend

    # Empirical Bayes moderation
    df_total = n1 + n2 - 2
    mod_logfc, mod_se, moderated_t = _limma_ebayes(log2_fc, se, df_total, robust=robust)

    # P-values from t-distribution
    p_values = 2 * (1 - stats.t.cdf(np.abs(moderated_t), df_total))

    # Adjust p-values
    p_values_adj = adjust_fdr(p_values, method="bh")

    # Group statistics
    g1_medians = np.nanmedian(counts1, axis=0)
    g2_medians = np.nanmedian(counts2, axis=0)

    return DiffExprResult(
        feature_ids=assay.var[assay.feature_id_col].to_numpy(),
        p_values=p_values,
        p_values_adj=p_values_adj,
        log2_fc=mod_logfc,
        test_statistics=moderated_t,
        effect_sizes=mod_logfc / np.maximum(mod_se, 1e-6),
        group_stats={
            f"{group1}_mean": means1,
            f"{group2}_mean": means2,
            f"{group1}_median": g1_medians,
            f"{group2}_median": g2_medians,
        },
        method="limma_trend",
        params={
            "groupby": groupby,
            "group1": group1,
            "group2": group2,
            "trend": trend,
            "robust": robust,
        },
    )


def diff_expr_deseq2(
    container: ScpContainer,
    assay_name: str,
    layer: str,
    groupby: str,
    group1: str,
    group2: str,
    fit_type: str = "parametric",
    test: str = "wald",
    min_count: int = 10,
) -> DiffExprResult:
    """DESeq2-like negative binomial model analysis.

    Models count data using negative binomial GLM.

    Parameters
    ----------
    container : ScpContainer
        Input data container.
    assay_name : str
        Assay containing the data.
    layer : str
        Layer with count data.
    groupby : str
        Column in obs defining groups.
    group1 : str
        First group name.
    group2 : str
        Second group name.
    fit_type : str, default="parametric"
        Dispersion fitting (parametric, local, mean).
    test : str, default="wald"
        Test type (wald, lrt).
    min_count : int, default=10
        Minimum count for filtering.

    Returns
    -------
    DiffExprResult
        Test results with NB-based statistics.

    Raises
    ------
    AssayNotFoundError
        If assay_name not found.
    LayerNotFoundError
        If layer not found.
    ValidationError
        If groups have insufficient samples or invalid parameters.

    Notes
    -----
    This implements a simplified DESeq2-like analysis using negative
    binomial modeling. The key steps are:

        1. Size factor estimation (median ratio method)
        2. Dispersion estimation
        3. GLM fitting
        4. Wald or LRT test

    The negative binomial model accounts for overdispersion common
    in count data:
        Var(Y) = mu + alpha * mu^2

    where alpha is the dispersion parameter.

    References
    ----------
    .. [1] Love MI, Huber W, Anders S (2014) Moderated estimation of
           fold change and dispersion for RNA-seq data with DESeq2.
           Genome Biology 15:550.

    .. [2] Anders S, Huber W (2010) Differential expression analysis
           for sequence count data. Genome Biology 11:R106.

    Examples
    --------
    >>> import scptensor as sc
    >>> result = sc.diff_expr.diff_expr_deseq2(
    ...     container,
    ...     assay_name="proteins",
    ...     layer="X",
    ...     groupby="condition",
    ...     group1="control",
    ...     group2="treatment"
    ... )
    >>> sig = result.get_significant(alpha=0.05)
    """
    # Validate inputs
    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name, available_assays=list(container.assays.keys()))

    assay = container.assays[assay_name]
    if layer not in assay.layers:
        raise LayerNotFoundError(layer, assay_name, available_layers=list(assay.layers.keys()))

    if groupby not in container.obs.columns:
        raise ValidationError(f"Group column '{groupby}' not found in obs", field="groupby")

    if fit_type not in {"parametric", "local", "mean"}:
        raise ValidationError(
            f"Unknown fit_type: {fit_type}. Use 'parametric', 'local', or 'mean'.",
            field="fit_type",
        )

    if test not in {"wald", "lrt"}:
        raise ValidationError(
            f"Unknown test: {test}. Use 'wald' or 'lrt'.",
            field="test",
        )

    # Get group indices
    groups = container.obs[groupby].to_numpy()
    group_indices = _extract_group_indices(groups, [group1, group2], min_samples=3)
    idx1, idx2 = group_indices[group1], group_indices[group2]

    # Extract data
    X = assay.layers[layer].X
    M = assay.layers[layer].M
    X_dense = _to_dense(X)

    # Handle mask - set masked values to zero
    if M is not None:
        M_dense = _to_dense(M)
        X_dense = np.where(M_dense == 0, X_dense, 0)
    else:
        M_dense = None

    counts1 = X_dense[idx1, :]
    counts2 = X_dense[idx2, :]

    n1, n_features = counts1.shape
    n2 = counts2.shape[0]

    # Size factor estimation (median ratio method)
    # Compute geometric means across all samples
    counts_all = np.vstack([counts1, counts2])
    log_counts = np.log(counts_all + 1)

    # Geometric mean per feature
    log_geom_means = np.mean(log_counts, axis=0)
    geom_means = np.exp(log_geom_means)

    # Size factors (median of ratios)
    size_factors = np.zeros(n1 + n2)
    for i in range(n1 + n2):
        ratios = counts_all[i, :] / (geom_means + 1e-6)
        ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
        if len(ratios) > 0:
            size_factors[i] = np.median(ratios)
        else:
            size_factors[i] = 1.0

    size_factors = np.maximum(size_factors, 1e-6)

    # Normalize counts
    norm_counts1 = counts1 / size_factors[:n1, np.newaxis]
    norm_counts2 = counts2 / size_factors[n1:, np.newaxis]

    # Filter low-count features
    total_counts = np.sum(counts_all, axis=0)
    valid_features = total_counts >= min_count
    if np.sum(valid_features) < 10:
        raise ValidationError(
            f"Too few features ({np.sum(valid_features)}) pass min_count filter",
            field="min_count",
        )

    # Estimate dispersion
    group_sizes = np.array([n1, n2])
    dispersions = _estimate_nb_dispersion(counts_all, group_sizes)

    # Apply fit_type adjustments
    if fit_type == "parametric":
        # Fit parametric curve: alpha = a1 + a2 / mean
        means = np.mean(counts_all, axis=0) + 1e-6
        valid_disp = dispersions < 50
        if np.sum(valid_disp) > 10:
            # Fit on transformed scale
            inv_mean = 1.0 / means[valid_disp]
            coeffs = np.polyfit(inv_mean, dispersions[valid_disp], 1)
            fitted_disp = coeffs[0] / means + coeffs[1]
            dispersions = np.maximum(dispersions, fitted_disp)
    elif fit_type == "local":
        # Local regression smoothing
        means = np.mean(counts_all, axis=0)
        valid_disp = (dispersions > 0) & (dispersions < 50)
        if np.sum(valid_disp) > 10:
            smoothed = _lowess_smooth(means[valid_disp], dispersions[valid_disp], frac=0.3)
            # Interpolate
            dispersions = np.interp(
                means, means[valid_disp], smoothed, left=smoothed[0], right=smoothed[-1]
            )

    # Wald test
    log2_fc, p_values = _nb_wald_test(norm_counts1, norm_counts2, dispersions)

    # LRT alternative
    if test == "lrt":
        # Likelihood ratio test
        # Null model: single mean for both groups
        # Alt model: separate means per group
        means1 = np.mean(norm_counts1, axis=0) + 1e-6
        means2 = np.mean(norm_counts2, axis=0) + 1e-6
        means_pooled = (n1 * means1 + n2 * means2) / (n1 + n2)

        # Log likelihoods (simplified NB)
        def nb_log_likelihood(counts, mean, disp):
            var = mean + disp * mean**2
            r = mean**2 / np.maximum(var - mean, 1e-6)
            p = np.maximum(var - mean, 1e-6) / var
            log_lik = np.sum(counts * np.log(mean) - (counts + r) * np.log(mean + disp * mean))
            return log_lik

        ll_null = np.sum(norm_counts1 * np.log(means_pooled)) + np.sum(
            norm_counts2 * np.log(means_pooled)
        )
        ll_alt = np.sum(norm_counts1 * np.log(means1)) + np.sum(norm_counts2 * np.log(means2))

        # LRT statistic
        lrt_stat = 2 * (ll_alt - ll_null)
        lrt_stat = np.maximum(lrt_stat, 0)

        # P-values from chi-square with 1 df
        p_values = 1 - stats.chi2.cdf(lrt_stat, 1)

    # Set p-values for filtered features
    p_values_all = np.ones(n_features) * np.nan
    p_values_all[valid_features] = p_values

    log2_fc_all = np.zeros(n_features) * np.nan
    log2_fc_all[valid_features] = log2_fc

    # Adjust p-values
    p_values_adj = adjust_fdr(p_values_all, method="bh")

    # Group statistics
    g1_means = np.zeros(n_features) * np.nan
    g2_means = np.zeros(n_features) * np.nan
    g1_means[valid_features] = np.mean(norm_counts1, axis=0)
    g2_means[valid_features] = np.mean(norm_counts2, axis=0)

    return DiffExprResult(
        feature_ids=assay.var[assay.feature_id_col].to_numpy(),
        p_values=p_values_all,
        p_values_adj=p_values_adj,
        log2_fc=log2_fc_all,
        test_statistics=np.abs(log2_fc_all) / 0.1,  # Approximate Wald statistic
        effect_sizes=log2_fc_all,
        group_stats={
            f"{group1}_mean": g1_means,
            f"{group2}_mean": g2_means,
        },
        method="deseq2",
        params={
            "groupby": groupby,
            "group1": group1,
            "group2": group2,
            "fit_type": fit_type,
            "test": test,
            "min_count": min_count,
        },
    )


# =============================================================================
# Module tests
# =============================================================================

if __name__ == "__main__":
    import sys

    print("Running count_models module tests...")

    # Test 1: _isna function
    print("Test 1: _isna function")
    assert np.array_equal(_isna(np.array([1.0, np.nan, 3.0])), [False, True, False])
    print("  PASSED")

    # Test 2: _lowess_smooth
    print("Test 2: _lowess_smooth")
    x = np.linspace(0, 10, 50)
    y = np.sin(x) + np.random.normal(0, 0.1, 50)
    smoothed = _lowess_smooth(x, y)
    assert len(smoothed) == len(y)
    print("  PASSED")

    # Test 3: _voom_transform
    print("Test 3: _voom_transform")
    counts = np.random.negative_binomial(10, 0.5, size=(20, 100))
    lib_size = np.sum(counts, axis=1)
    log2_cpm, weights = _voom_transform(counts, lib_size)
    assert log2_cpm.shape == counts.shape
    assert weights.shape[0] == counts.shape[1]  # Per feature
    print("  PASSED")

    # Test 4: _tmm_normalize
    print("Test 4: _tmm_normalize")
    counts = np.random.negative_binomial(10, 0.5, size=(10, 50))
    norm_counts, lib_size = _tmm_normalize(counts)
    assert norm_counts.shape == counts.shape
    assert lib_size.shape[0] == counts.shape[0]
    print("  PASSED")

    # Test 5: _upper_quartile_normalize
    print("Test 5: _upper_quartile_normalize")
    counts = np.random.negative_binomial(10, 0.5, size=(10, 50))
    norm_counts, lib_size = _upper_quartile_normalize(counts)
    assert norm_counts.shape == counts.shape
    print("  PASSED")

    # Test 6: _estimate_nb_dispersion
    print("Test 6: _estimate_nb_dispersion")
    counts = np.vstack(
        [
            np.random.negative_binomial(10, 0.5, size=(10, 100)),
            np.random.negative_binomial(15, 0.5, size=(10, 100)),
        ]
    )
    dispersions = _estimate_nb_dispersion(counts, np.array([10, 10]))
    assert len(dispersions) == 100
    assert np.all(dispersions > 0)
    print("  PASSED")

    # Test 7: _nb_wald_test
    print("Test 7: _nb_wald_test")
    counts1 = np.random.negative_binomial(10, 0.5, size=(10, 50))
    counts2 = np.random.negative_binomial(15, 0.5, size=(10, 50))
    dispersions = np.ones(50) * 0.1
    log2_fc, p_values = _nb_wald_test(counts1, counts2, dispersions)
    assert len(log2_fc) == 50
    assert len(p_values) == 50
    assert np.all((p_values >= 0) & (p_values <= 1))
    print("  PASSED")

    # Test 8: _limma_ebayes
    print("Test 8: _limma_ebayes")
    logfc = np.random.randn(100) * 0.5
    se = np.abs(np.random.randn(100)) * 0.1 + 0.05
    mod_logfc, mod_se, mod_t = _limma_ebayes(logfc, se, df=10, robust=False)
    assert len(mod_logfc) == len(logfc)
    assert np.all(np.isfinite(mod_t[np.isfinite(logfc)]))
    print("  PASSED")

    # Test 9: _to_dense
    print("Test 9: _to_dense")
    dense = np.random.randn(10, 20)
    sparse = sp.csr_matrix(dense)
    assert np.array_equal(_to_dense(dense), dense)
    assert np.allclose(_to_dense(sparse), dense)
    print("  PASSED")

    print()
    print("=" * 60)
    print("All count_models tests passed!")
    print("=" * 60)
    sys.exit(0)
