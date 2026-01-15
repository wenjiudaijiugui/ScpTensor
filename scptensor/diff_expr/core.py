"""Differential expression analysis core module.

This module provides statistical tests for identifying features (proteins/peptides)
that differ significantly between groups in single-cell proteomics data.

Supported tests:
    - t-test: Two-group comparison (Student's and Welch's)
    - Mann-Whitney U: Non-parametric two-group comparison
    - ANOVA: Multi-group comparison
    - Kruskal-Wallis: Non-parametric multi-group comparison
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
import scipy.sparse as sp
import scipy.stats as stats

from scptensor.core.exceptions import (
    AssayNotFoundError,
    LayerNotFoundError,
    ValidationError,
)
from scptensor.core.structures import ScpContainer


@dataclass(frozen=True)
class DiffExprResult:
    """
    Result container for differential expression analysis.

    Attributes
    ----------
    feature_ids : np.ndarray
        Feature identifiers (protein/peptide IDs)
    p_values : np.ndarray
        Raw p-values from statistical test
    p_values_adj : np.ndarray
        Adjusted p-values (FDR-corrected)
    log2_fc : np.ndarray
        Log2 fold changes (for two-group tests)
    test_statistics : np.ndarray
        Test statistic values
    effect_sizes : np.ndarray | None
        Effect size metrics (e.g., Cohen's d)
    group_stats : dict[str, np.ndarray]
        Group-wise statistics (mean, median, etc.)
    method : str
        Statistical method used
    params : dict[str, Any]
        Parameters used in the analysis
    """

    feature_ids: np.ndarray
    p_values: np.ndarray
    p_values_adj: np.ndarray
    log2_fc: np.ndarray
    test_statistics: np.ndarray
    effect_sizes: np.ndarray | None
    group_stats: dict[str, np.ndarray]
    method: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pl.DataFrame:
        """
        Convert results to a Polars DataFrame.

        Returns
        -------
        pl.DataFrame
            Results as a DataFrame with columns for feature_id, p_value,
            p_value_adj, log2_fc, test_statistic, effect_size.
        """
        data: dict[str, np.ndarray | list] = {
            "feature_id": self.feature_ids,
            "p_value": self.p_values,
            "p_value_adj": self.p_values_adj,
            "log2_fc": self.log2_fc,
            "test_statistic": self.test_statistics,
        }

        if self.effect_sizes is not None:
            data["effect_size"] = self.effect_sizes

        data.update(self.group_stats)

        return pl.DataFrame(data).sort("p_value")

    def get_significant(
        self, alpha: float = 0.05, min_log2_fc: float | None = None
    ) -> pl.DataFrame:
        """
        Get significant features based on adjusted p-value threshold.

        Parameters
        ----------
        alpha : float, default=0.05
            FDR threshold for significance
        min_log2_fc : float, optional
            Minimum absolute log2 fold change for significance

        Returns
        -------
        pl.DataFrame
            Significant features sorted by adjusted p-value
        """
        df = self.to_dataframe()
        mask = pl.col("p_value_adj") < alpha

        if min_log2_fc is not None:
            mask = mask & (pl.col("log2_fc").abs() >= min_log2_fc)

        return df.filter(mask)


def adjust_fdr(
    p_values: np.ndarray,
    method: str = "bh",
) -> np.ndarray:
    """
    Adjust p-values for multiple testing using False Discovery Rate methods.

    Parameters
    ----------
    p_values : np.ndarray
        Raw p-values to adjust
    method : str, default="bh"
        FDR correction method:
        - "bh": Benjamini-Hochberg (step-up)
        - "by": Benjamini-Yekutieli (more conservative, assumes dependence)

    Returns
    -------
    np.ndarray
        Adjusted p-values (FDR-corrected)

    Notes
    -----
    The Benjamini-Hochberg procedure controls the FDR at level alpha:
        1. Sort p-values: p_(1) <= p_(2) <= ... <= p_(m)
        2. Find largest k such that p_(k) <= (k/m) * alpha
        3. Reject all hypotheses 1, 2, ..., k

    References
    ----------
    Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery
    rate: a practical and powerful approach to multiple testing. Journal of
    the Royal Statistical Society Series B, 57(1), 289-300.
    """
    p_values = np.asarray(p_values, dtype=np.float64)
    n = len(p_values)

    if n == 0:
        return np.array([], dtype=np.float64)

    nan_mask = np.isnan(p_values)

    if nan_mask.all():
        return p_values

    valid_idx = ~nan_mask
    p_clean = p_values[valid_idx]
    n_valid = len(p_clean)

    sorted_idx = np.argsort(p_clean)
    sorted_p = p_clean[sorted_idx]
    ranks = np.arange(1, n_valid + 1, dtype=np.float64)

    multiplier: float = float(n)
    if method == "by":
        multiplier *= float(np.sum(1.0 / ranks))
    elif method != "bh":
        raise ValidationError(
            f"Unknown FDR correction method: {method}. Use 'bh' or 'by'.",
            field="method",
        )

    adjusted = sorted_p * multiplier / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    result = np.full_like(p_values, np.nan, dtype=np.float64)
    result[valid_idx] = adjusted[np.argsort(sorted_idx)]

    return result


def _handle_missing_values(
    X: np.ndarray | sp.spmatrix,
    strategy: str = "ignore",
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Handle missing values for statistical testing.

    Parameters
    ----------
    X : np.ndarray or sp.spmatrix
        Input data matrix (n_samples, n_features)
    strategy : str, default="ignore"
        Strategy for handling missing values:
        - "ignore": Skip missing values in calculations
        - "zero": Replace missing values with zeros
        - "median": Replace missing values with feature median

    Returns
    -------
    tuple
        (X_processed, mask) where mask indicates valid values
    """
    if sp.issparse(X):
        X = X.toarray()  # type: ignore[union-attr]

    X_arr = X.astype(np.float64, copy=False)

    if strategy == "ignore":
        return X_arr, ~np.isnan(X_arr)

    if strategy == "zero":
        np.nan_to_num(X_arr, copy=False, nan=0.0)
        return X_arr, None

    if strategy == "median":
        nan_mask = np.isnan(X_arr)
        if nan_mask.any():
            col_medians = np.nanmedian(X_arr, axis=0)
            np.where(col_medians == 0, np.finfo(np.float64).eps, col_medians)
            X_arr = np.where(nan_mask, col_medians, X_arr)
        return X_arr, None

    raise ValidationError(
        f"Unknown missing value strategy: {strategy}. Use 'ignore', 'zero', or 'median'.",
        field="strategy",
    )


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size for two groups.

    Parameters
    ----------
    x, y : np.ndarray
        Data values for two groups

    Returns
    -------
    float
        Cohen's d effect size

    Notes
    -----
    Cohen's d is calculated as:
        d = (mean(x) - mean(y)) / pooled_std

    where pooled_std = sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))

    Interpretation:
        |d| < 0.2: Small effect
        |d| < 0.5: Medium effect
        |d| >= 0.8: Large effect
    """
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return 0.0

    var1, var2 = float(np.var(x, ddof=1)), float(np.var(y, ddof=1))
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    return 0.0 if pooled_std == 0 else (float(np.mean(x) - np.mean(y)) / float(pooled_std))


def _log2_fold_change(
    group1_values: np.ndarray,
    group2_values: np.ndarray,
    offset: float = 1.0,
) -> float:
    """
    Calculate log2 fold change between two groups.

    Parameters
    ----------
    group1_values : np.ndarray
        Values from group 1
    group2_values : np.ndarray
        Values from group 2
    offset : float, default=1.0
        Offset added to avoid log(0)

    Returns
    -------
    float
        Log2 fold change (group1 relative to group2)
    """
    median1, median2 = float(np.median(group1_values)), float(np.median(group2_values))
    return float(np.log2((median1 + offset) / (median2 + offset)))


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
        return mask  # type: ignore[no-any-return]
    return np.isnan(arr)  # type: ignore[no-any-return]


def diff_expr_ttest(
    container: ScpContainer,
    assay_name: str,
    group_col: str,
    group1: str,
    group2: str,
    layer_name: str = "X",
    equal_var: bool = False,
    missing_strategy: str = "ignore",
    min_samples_per_group: int = 3,
    log2_fc_offset: float = 1.0,
) -> DiffExprResult:
    """
    Perform two-group differential expression using t-test.

    Parameters
    ----------
    container : ScpContainer
        Input container
    assay_name : str
        Name of assay to analyze
    group_col : str
        Column name in obs containing group labels
    group1, group2 : str
        Names of the two groups to compare
    layer_name : str, default="X"
        Layer to use for analysis
    equal_var : bool, default=False
        If True, use Student's t-test (assumes equal variance)
        If False, use Welch's t-test (does not assume equal variance)
    missing_strategy : str, default="ignore"
        How to handle missing values: "ignore", "zero", or "median"
    min_samples_per_group : int, default=3
        Minimum samples required per group for testing
    log2_fc_offset : float, default=1.0
        Offset for log2 fold change calculation

    Returns
    -------
    DiffExprResult
        Results including p-values, adjusted p-values, log2 fold changes,
        test statistics, and effect sizes

    Raises
    ------
    AssayNotFoundError
        If specified assay does not exist
    LayerNotFoundError
        If specified layer does not exist
    ValidationError
        If group labels not found or insufficient samples

    Notes
    -----
    Welch's t-test (equal_var=False) is recommended for proteomics data
    as it handles unequal variances between groups, which is common when
    comparing different experimental conditions.

    The test statistic for Welch's t-test is:
        t = (mean1 - mean2) / sqrt(var1/n1 + var2/n2)

    Degrees of freedom (Welch-Satterthwaite equation):
        df = (var1/n1 + var2/n2)^2 / ((var1/n1)^2/(n1-1) + (var2/n2)^2/(n2-1))
    """
    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]
    if layer_name not in assay.layers:
        raise LayerNotFoundError(layer_name, assay_name)

    if group_col not in container.obs.columns:
        raise ValidationError(f"Group column '{group_col}' not found in obs", field="group_col")

    groups = container.obs[group_col].to_numpy()
    group_indices = _extract_group_indices(groups, [group1, group2], min_samples_per_group)
    idx1, idx2 = group_indices[group1], group_indices[group2]

    X = assay.layers[layer_name].X
    n_features = X.shape[1]
    X_proc, valid_mask = _handle_missing_values(X, strategy=missing_strategy)

    p_values = np.full(n_features, np.nan, dtype=np.float64)
    test_stats = np.full(n_features, np.nan, dtype=np.float64)
    log2_fc = np.full(n_features, np.nan, dtype=np.float64)
    effect_sizes = np.full(n_features, np.nan, dtype=np.float64)

    g1_means = np.full(n_features, np.nan, dtype=np.float64)
    g2_means = np.full(n_features, np.nan, dtype=np.float64)
    g1_medians = np.full(n_features, np.nan, dtype=np.float64)
    g2_medians = np.full(n_features, np.nan, dtype=np.float64)

    for j in range(n_features):
        g1_vals = X_proc[idx1, j]
        g2_vals = X_proc[idx2, j]

        if valid_mask is not None:
            g1_vals = g1_vals[valid_mask[idx1, j]]
            g2_vals = g2_vals[valid_mask[idx2, j]]

        if len(g1_vals) < 2 or len(g2_vals) < 2:
            continue

        g1_means[j] = np.mean(g1_vals)
        g2_means[j] = np.mean(g2_vals)
        g1_medians[j] = np.median(g1_vals)
        g2_medians[j] = np.median(g2_vals)

        result = stats.ttest_ind(g1_vals, g2_vals, equal_var=equal_var)
        p_values[j] = result.pvalue
        test_stats[j] = result.statistic
        log2_fc[j] = _log2_fold_change(g1_vals, g2_vals, offset=log2_fc_offset)
        effect_sizes[j] = _cohens_d(g1_vals, g2_vals)

    method_name = "welch_ttest" if not equal_var else "student_ttest"

    return DiffExprResult(
        feature_ids=assay.var[assay.feature_id_col].to_numpy(),
        p_values=p_values,
        p_values_adj=adjust_fdr(p_values, method="bh"),
        log2_fc=log2_fc,
        test_statistics=test_stats,
        effect_sizes=effect_sizes,
        group_stats={
            f"{group1}_mean": g1_means,
            f"{group2}_mean": g2_means,
            f"{group1}_median": g1_medians,
            f"{group2}_median": g2_medians,
        },
        method=method_name,
        params={
            "group_col": group_col,
            "group1": group1,
            "group2": group2,
            "equal_var": equal_var,
            "missing_strategy": missing_strategy,
        },
    )


def diff_expr_mannwhitney(
    container: ScpContainer,
    assay_name: str,
    group_col: str,
    group1: str,
    group2: str,
    layer_name: str = "X",
    alternative: str = "two-sided",
    missing_strategy: str = "ignore",
    min_samples_per_group: int = 3,
    log2_fc_offset: float = 1.0,
) -> DiffExprResult:
    """
    Perform two-group differential expression using Mann-Whitney U test.

    Parameters
    ----------
    container : ScpContainer
        Input container
    assay_name : str
        Name of assay to analyze
    group_col : str
        Column name in obs containing group labels
    group1, group2 : str
        Names of the two groups to compare
    layer_name : str, default="X"
        Layer to use for analysis
    alternative : str, default="two-sided"
        Alternative hypothesis: "two-sided", "less", or "greater"
    missing_strategy : str, default="ignore"
        How to handle missing values: "ignore", "zero", or "median"
    min_samples_per_group : int, default=3
        Minimum samples required per group for testing
    log2_fc_offset : float, default=1.0
        Offset for log2 fold change calculation

    Returns
    -------
    DiffExprResult
        Results including p-values, adjusted p-values, log2 fold changes,
        and test statistics

    Raises
    ------
    AssayNotFoundError
        If specified assay does not exist
    LayerNotFoundError
        If specified layer does not exist
    ValidationError
        If group labels not found or insufficient samples

    Notes
    -----
    The Mann-Whitney U test (also called Wilcoxon rank-sum test) is a
    non-parametric test that does not assume normality. It tests whether
    samples from one group tend to have higher values than the other.

    The test statistic U is calculated as:
        U = R1 - n1*(n1+1)/2

    where R1 is the sum of ranks for group 1.

    This test is more robust to outliers than the t-test but has less
    statistical power when data is normally distributed.

    References
    ----------
    Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of two
    random variables is stochastically larger than the other. The Annals
    of Mathematical Statistics, 18(1), 50-60.
    """
    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]
    if layer_name not in assay.layers:
        raise LayerNotFoundError(layer_name, assay_name)

    if group_col not in container.obs.columns:
        raise ValidationError(f"Group column '{group_col}' not found in obs", field="group_col")

    groups = container.obs[group_col].to_numpy()
    group_indices = _extract_group_indices(groups, [group1, group2], min_samples_per_group)
    idx1, idx2 = group_indices[group1], group_indices[group2]

    X = assay.layers[layer_name].X
    n_features = X.shape[1]
    X_proc, valid_mask = _handle_missing_values(X, strategy=missing_strategy)

    p_values = np.full(n_features, np.nan, dtype=np.float64)
    test_stats = np.full(n_features, np.nan, dtype=np.float64)
    log2_fc = np.full(n_features, np.nan, dtype=np.float64)

    g1_medians = np.full(n_features, np.nan, dtype=np.float64)
    g2_medians = np.full(n_features, np.nan, dtype=np.float64)

    for j in range(n_features):
        g1_vals = X_proc[idx1, j]
        g2_vals = X_proc[idx2, j]

        if valid_mask is not None:
            g1_vals = g1_vals[valid_mask[idx1, j]]
            g2_vals = g2_vals[valid_mask[idx2, j]]

        if len(g1_vals) < 2 or len(g2_vals) < 2:
            continue

        g1_medians[j] = np.median(g1_vals)
        g2_medians[j] = np.median(g2_vals)

        try:
            result = stats.mannwhitneyu(g1_vals, g2_vals, alternative=alternative)
            p_values[j] = result.pvalue
            test_stats[j] = result.statistic
        except ValueError:
            p_values[j] = 1.0
            test_stats[j] = 0.0

        log2_fc[j] = _log2_fold_change(g1_vals, g2_vals, offset=log2_fc_offset)

    return DiffExprResult(
        feature_ids=assay.var[assay.feature_id_col].to_numpy(),
        p_values=p_values,
        p_values_adj=adjust_fdr(p_values, method="bh"),
        log2_fc=log2_fc,
        test_statistics=test_stats,
        effect_sizes=None,
        group_stats={
            f"{group1}_median": g1_medians,
            f"{group2}_median": g2_medians,
        },
        method="mannwhitney",
        params={
            "group_col": group_col,
            "group1": group1,
            "group2": group2,
            "alternative": alternative,
            "missing_strategy": missing_strategy,
        },
    )


def diff_expr_anova(
    container: ScpContainer,
    assay_name: str,
    group_col: str,
    layer_name: str = "X",
    missing_strategy: str = "ignore",
    min_samples_per_group: int = 3,
) -> DiffExprResult:
    """
    Perform multi-group differential expression using one-way ANOVA.

    Parameters
    ----------
    container : ScpContainer
        Input container
    assay_name : str
        Name of assay to analyze
    group_col : str
        Column name in obs containing group labels
    layer_name : str, default="X"
        Layer to use for analysis
    missing_strategy : str, default="ignore"
        How to handle missing values: "ignore", "zero", or "median"
    min_samples_per_group : int, default=3
        Minimum samples required per group for testing

    Returns
    -------
    DiffExprResult
        Results including p-values, adjusted p-values, F-statistics,
        and group-wise statistics

    Raises
    ------
    AssayNotFoundError
        If specified assay does not exist
    LayerNotFoundError
        If specified layer does not exist
    ValidationError
        If group labels not found or insufficient samples

    Notes
    -----
    One-way ANOVA tests whether the means of multiple groups are equal.
    The null hypothesis is that all group means are equal.

    The F-statistic is calculated as:
        F = (MS_between) / (MS_within)

    where:
        MS_between = sum(n_i * (mean_i - grand_mean)^2) / (k - 1)
        MS_within = sum((n_i - 1) * var_i) / (N - k)

    with k groups and total N samples.

    ANOVA assumes:
        1. Normality within each group
        2. Homogeneity of variances
        3. Independence of observations

    If these assumptions are violated, consider using Kruskal-Wallis test.

    References
    ----------
    Fisher, R. A. (1918). The correlation between relatives on the
    supposition of Mendelian inheritance. Transactions of the Royal
    Society of Edinburgh, 52(2), 399-433.
    """
    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]
    if layer_name not in assay.layers:
        raise LayerNotFoundError(layer_name, assay_name)

    if group_col not in container.obs.columns:
        raise ValidationError(f"Group column '{group_col}' not found in obs", field="group_col")

    groups = container.obs[group_col].to_numpy()
    unique_groups = groups[~_isna(groups)]
    unique_groups = np.unique(unique_groups)
    n_groups = len(unique_groups)

    if n_groups < 2:
        raise ValidationError(
            f"Need at least 2 groups for ANOVA, found {n_groups}",
            field="group_col",
        )

    group_indices = _extract_group_indices(groups, list(unique_groups), min_samples_per_group)

    X = assay.layers[layer_name].X
    n_features = X.shape[1]
    X_proc, valid_mask = _handle_missing_values(X, strategy=missing_strategy)

    nan_value = np.nan
    p_values = np.full(n_features, nan_value, dtype=np.float64)
    test_stats = np.full(n_features, nan_value, dtype=np.float64)

    group_means = {
        f"{g}_mean": np.full(n_features, nan_value, dtype=np.float64) for g in unique_groups
    }
    group_medians = {
        f"{g}_median": np.full(n_features, nan_value, dtype=np.float64) for g in unique_groups
    }

    for j in range(n_features):
        group_data: list[np.ndarray] | None = []
        for g in unique_groups:
            idx = group_indices[g]
            g_vals = X_proc[idx, j]

            if valid_mask is not None:
                g_vals = g_vals[valid_mask[idx, j]]

            if len(g_vals) < 2:
                group_data = None
                break

            assert group_data is not None
            group_data.append(g_vals)
            group_means[f"{g}_mean"][j] = np.mean(g_vals)
            group_medians[f"{g}_median"][j] = np.median(g_vals)

        if group_data is None:
            continue

        try:
            result = stats.f_oneway(*group_data)
            p_values[j] = result.pvalue
            test_stats[j] = result.statistic
        except ValueError:
            p_values[j] = 1.0
            test_stats[j] = 0.0

    group_stats = {**group_means, **group_medians}
    nan_array = np.full(n_features, np.nan, dtype=np.float64)

    return DiffExprResult(
        feature_ids=assay.var[assay.feature_id_col].to_numpy(),
        p_values=p_values,
        p_values_adj=adjust_fdr(p_values, method="bh"),
        log2_fc=nan_array,
        test_statistics=test_stats,
        effect_sizes=nan_array,
        group_stats=group_stats,
        method="anova",
        params={
            "group_col": group_col,
            "n_groups": n_groups,
            "groups": list(unique_groups),
            "missing_strategy": missing_strategy,
        },
    )


def diff_expr_kruskal(
    container: ScpContainer,
    assay_name: str,
    group_col: str,
    layer_name: str = "X",
    missing_strategy: str = "ignore",
    min_samples_per_group: int = 3,
) -> DiffExprResult:
    """
    Perform multi-group differential expression using Kruskal-Wallis test.

    Parameters
    ----------
    container : ScpContainer
        Input container
    assay_name : str
        Name of assay to analyze
    group_col : str
        Column name in obs containing group labels
    layer_name : str, default="X"
        Layer to use for analysis
    missing_strategy : str, default="ignore"
        How to handle missing values: "ignore", "zero", or "median"
    min_samples_per_group : int, default=3
        Minimum samples required per group for testing

    Returns
    -------
    DiffExprResult
        Results including p-values, adjusted p-values, H-statistics,
        and group-wise statistics

    Raises
    ------
    AssayNotFoundError
        If specified assay does not exist
    LayerNotFoundError
        If specified layer does not exist
    ValidationError
        If group labels not found or insufficient samples

    Notes
    -----
    The Kruskal-Wallis test is a non-parametric alternative to one-way ANOVA.
    It tests whether samples from different groups originate from the same
    distribution without assuming normality.

    The H-statistic is based on rank sums:
        H = (12 / (N*(N+1))) * sum(R_i^2 / n_i) - 3*(N+1)

    where R_i is the sum of ranks for group i and n_i is the size of group i.

    This test is more robust to outliers and non-normal distributions than
    ANOVA, but has less statistical power when data is normally distributed.

    References
    ----------
    Kruskal, W. H., & Wallis, W. A. (1952). Use of ranks in one-criterion
    variance analysis. Journal of the American Statistical Association,
    47(260), 583-621.
    """
    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]
    if layer_name not in assay.layers:
        raise LayerNotFoundError(layer_name, assay_name)

    if group_col not in container.obs.columns:
        raise ValidationError(f"Group column '{group_col}' not found in obs", field="group_col")

    groups = container.obs[group_col].to_numpy()
    unique_groups = groups[~_isna(groups)]
    unique_groups = np.unique(unique_groups)
    n_groups = len(unique_groups)

    if n_groups < 2:
        raise ValidationError(
            f"Need at least 2 groups for Kruskal-Wallis, found {n_groups}",
            field="group_col",
        )

    group_indices = _extract_group_indices(groups, list(unique_groups), min_samples_per_group)

    X = assay.layers[layer_name].X
    n_features = X.shape[1]
    X_proc, valid_mask = _handle_missing_values(X, strategy=missing_strategy)

    nan_value = np.nan
    p_values = np.full(n_features, nan_value, dtype=np.float64)
    test_stats = np.full(n_features, nan_value, dtype=np.float64)

    group_means = {
        f"{g}_mean": np.full(n_features, nan_value, dtype=np.float64) for g in unique_groups
    }
    group_medians = {
        f"{g}_median": np.full(n_features, nan_value, dtype=np.float64) for g in unique_groups
    }

    for j in range(n_features):
        group_data: list[np.ndarray] | None = []
        for g in unique_groups:
            idx = group_indices[g]
            g_vals = X_proc[idx, j]

            if valid_mask is not None:
                g_vals = g_vals[valid_mask[idx, j]]

            if len(g_vals) < 2:
                group_data = None
                break

            assert group_data is not None
            group_data.append(g_vals)
            group_means[f"{g}_mean"][j] = np.mean(g_vals)
            group_medians[f"{g}_median"][j] = np.median(g_vals)

        if group_data is None:
            continue

        try:
            result = stats.kruskal(*group_data)
            p_values[j] = result.pvalue
            test_stats[j] = result.statistic
        except ValueError:
            p_values[j] = 1.0
            test_stats[j] = 0.0

    group_stats = {**group_means, **group_medians}
    nan_array = np.full(n_features, np.nan, dtype=np.float64)

    return DiffExprResult(
        feature_ids=assay.var[assay.feature_id_col].to_numpy(),
        p_values=p_values,
        p_values_adj=adjust_fdr(p_values, method="bh"),
        log2_fc=nan_array,
        test_statistics=test_stats,
        effect_sizes=nan_array,
        group_stats=group_stats,
        method="kruskal",
        params={
            "group_col": group_col,
            "n_groups": n_groups,
            "groups": list(unique_groups),
            "missing_strategy": missing_strategy,
        },
    )


if __name__ == "__main__":
    import sys

    print("Running diff_expr module tests...")

    # Test 1: FDR adjustment
    print("Test 1: FDR adjustment")
    p_values = np.array([0.001, 0.01, 0.05, 0.1, 0.5])
    adjusted = adjust_fdr(p_values, method="bh")
    assert np.all(adjusted >= p_values), "Adjusted p-values should be >= original"
    assert np.all(adjusted <= 1.0), "Adjusted p-values should be <= 1"
    print("  PASSED")

    # Test 2: NaN handling
    print("Test 2: NaN handling")
    p_with_nan = np.array([0.01, np.nan, 0.05, 0.1, np.nan])
    adjusted_nan = adjust_fdr(p_with_nan, method="bh")
    assert np.isnan(adjusted_nan[1]) and np.isnan(adjusted_nan[4])
    assert not np.isnan(adjusted_nan[0])
    print("  PASSED")

    # Test 3: Cohen's d
    print("Test 3: Cohen's d")
    x, y = np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    d = _cohens_d(x, y)
    assert -2 < d < 2
    print("  PASSED")

    # Test 4: Log2 fold change
    print("Test 4: Log2 fold change")
    g1, g2 = np.array([10.0, 12.0, 11.0, 13.0, 10.0]), np.array([5.0, 6.0, 5.5, 6.5, 5.0])
    fc = _log2_fold_change(g1, g2)
    assert fc > 0
    print("  PASSED")

    # Test 5: DiffExprResult
    print("Test 5: DiffExprResult")
    result = DiffExprResult(
        feature_ids=np.array(["P1", "P2", "P3"]),
        p_values=np.array([0.001, 0.05, 0.5]),
        p_values_adj=np.array([0.003, 0.075, 0.5]),
        log2_fc=np.array([1.5, -0.8, 0.1]),
        test_statistics=np.array([3.2, -1.5, 0.5]),
        effect_sizes=np.array([0.8, -0.4, 0.1]),
        group_stats={"A_mean": np.array([10, 8, 5]), "B_mean": np.array([4, 10, 5])},
        method="ttest",
        params={"group1": "A", "group2": "B"},
    )
    df = result.to_dataframe()
    assert df.shape[0] == 3
    assert "p_value" in df.columns and "log2_fc" in df.columns
    sig = result.get_significant(alpha=0.05, min_log2_fc=0.5)
    assert len(sig) == 1
    print("  PASSED")

    # Test 6: Missing value handling
    print("Test 6: Missing value handling")
    X_missing = np.array([[1.0, 2.0, np.nan], [3.0, np.nan, 4.0], [5.0, 6.0, 7.0]])
    X_proc, mask = _handle_missing_values(X_missing, strategy="ignore")
    assert mask is not None and mask.sum() == 7
    print("  PASSED")

    # Test 7: BY correction
    print("Test 7: Benjamini-Yekutieli correction")
    p_values = np.array([0.001, 0.01, 0.05, 0.1, 0.5])
    adjusted_bh = adjust_fdr(p_values, method="bh")
    adjusted_by = adjust_fdr(p_values, method="by")
    assert np.all(adjusted_by >= adjusted_bh)
    print("  PASSED")

    # Test 8: _isna function
    print("Test 8: _isna function")
    assert np.array_equal(_isna(np.array([1.0, np.nan, 3.0])), [False, True, False])
    assert _isna(np.array(["a", None, "NaN"]))[1]
    print("  PASSED")

    print()
    print("=" * 60)
    print("All diff_expr core tests passed!")
    print("=" * 60)
    sys.exit(0)
