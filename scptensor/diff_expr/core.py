"""Differential expression analysis core module.

This module provides statistical tests for identifying features (proteins/peptides)
that differ significantly between groups in single-cell proteomics data.

Supported tests:
    - t-test: Two-group comparison (Student's and Welch's)
    - Paired t-test: Paired sample comparison
    - Mann-Whitney U: Non-parametric two-group comparison
    - ANOVA: Multi-group comparison
    - Kruskal-Wallis: Non-parametric multi-group comparison
    - Permutation test: Non-parametric resampling test
    - Homoscedasticity tests: Levene and Brown-Forsythe tests
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
import scipy.stats as stats

from scptensor.core.exceptions import (
    AssayNotFoundError,
    LayerNotFoundError,
    ValidationError,
)
from scptensor.core.structures import ScpContainer

# Import shared utilities
from scptensor.diff_expr._utils import (
    cohens_d,
    extract_group_indices,
    handle_missing_values,
    isna,
    log2_fold_change,
    validate_pairing,
)


@dataclass(frozen=True)
class DiffExprResult:
    """Result container for differential expression analysis.

    Attributes
    ----------
    feature_ids : np.ndarray
        Feature identifiers
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
        """Convert results to a Polars DataFrame.

        Returns
        -------
        pl.DataFrame
            Results as a DataFrame
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
        """Get significant features based on adjusted p-value threshold.

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
    """Adjust p-values for multiple testing.

    Parameters
    ----------
    p_values : np.ndarray
        Raw p-values to adjust
    method : str, default="bh"
        Correction method: "bh", "by", "bonferroni", "holm", or "hommel"

    Returns
    -------
    np.ndarray
        Adjusted p-values
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

    if method in {"bh", "by"}:
        sorted_idx = np.argsort(p_clean)
        sorted_p = p_clean[sorted_idx]
        ranks = np.arange(1, n_valid + 1, dtype=np.float64)

        multiplier: float = float(n_valid)
        if method == "by":
            multiplier *= float(np.sum(1.0 / ranks))

        adjusted = sorted_p * multiplier / ranks
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        adjusted = np.clip(adjusted, 0.0, 1.0)

        result = np.full_like(p_values, np.nan, dtype=np.float64)
        result[valid_idx] = adjusted[np.argsort(sorted_idx)]
        return result

    if method == "bonferroni":
        result = np.full_like(p_values, np.nan, dtype=np.float64)
        result[valid_idx] = np.minimum(p_clean * n_valid, 1.0)
        return result

    if method == "holm":
        sorted_idx = np.argsort(p_clean)
        sorted_p = p_clean[sorted_idx]

        adjusted = np.zeros_like(sorted_p)
        n_tests = n_valid
        for i, p_val in enumerate(sorted_p):
            adjusted[i] = min(p_val * (n_tests - i), 1.0)

        for i in range(len(adjusted) - 2, -1, -1):
            adjusted[i] = min(adjusted[i], adjusted[i + 1])

        result = np.full_like(p_values, np.nan, dtype=np.float64)
        result[valid_idx] = adjusted[np.argsort(sorted_idx)]
        return result

    if method == "hommel":
        sorted_idx = np.argsort(p_clean)
        sorted_p = p_clean[sorted_idx]
        n_tests = n_valid

        adjusted = np.full_like(sorted_p, np.nan)
        for i in range(n_tests):
            p_i = sorted_p[i]
            max_val = p_i * n_tests / (i + 1)
            for j in range(i, n_tests):
                candidate = sorted_p[j] * n_tests / (j - i + 1)
                max_val = max(max_val, candidate)
            adjusted[i] = min(max_val, 1.0)

        for i in range(len(adjusted) - 2, -1, -1):
            adjusted[i] = min(adjusted[i], adjusted[i + 1])

        result = np.full_like(p_values, np.nan, dtype=np.float64)
        result[valid_idx] = adjusted[np.argsort(sorted_idx)]
        return result

    raise ValidationError(
        f"Unknown correction method: {method}. Use 'bh', 'by', 'bonferroni', 'holm', or 'hommel'.",
        field="method",
    )


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
    """Two-group differential expression using t-test.

    Parameters
    ----------
    container : ScpContainer
        Input container
    assay_name : str
        Name of assay to analyze
    group_col : str
        Column in obs containing group labels
    group1, group2 : str
        Names of the two groups to compare
    layer_name : str, default="X"
        Layer to use for analysis
    equal_var : bool, default=False
        If True, use Student's t-test; otherwise Welch's t-test
    missing_strategy : str, default="ignore"
        How to handle missing values
    min_samples_per_group : int, default=3
        Minimum samples required per group
    log2_fc_offset : float, default=1.0
        Offset for log2 fold change calculation

    Returns
    -------
    DiffExprResult
        Test results
    """
    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]
    if layer_name not in assay.layers:
        raise LayerNotFoundError(layer_name, assay_name)

    if group_col not in container.obs.columns:
        raise ValidationError(f"Group column '{group_col}' not found in obs", field="group_col")

    groups = container.obs[group_col].to_numpy()
    group_indices = extract_group_indices(groups, [group1, group2], min_samples_per_group)
    idx1, idx2 = group_indices[group1], group_indices[group2]

    X = assay.layers[layer_name].X
    n_features = X.shape[1]
    X_proc, valid_mask = handle_missing_values(X, strategy=missing_strategy)

    p_values = np.full(n_features, np.nan, dtype=np.float64)
    test_stats = np.full(n_features, np.nan, dtype=np.float64)
    log2_fc_arr = np.full(n_features, np.nan, dtype=np.float64)
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
        log2_fc_arr[j] = log2_fold_change(g1_vals, g2_vals, offset=log2_fc_offset)
        effect_sizes[j] = cohens_d(g1_vals, g2_vals)

    method_name = "welch_ttest" if not equal_var else "student_ttest"

    return DiffExprResult(
        feature_ids=assay.var[assay.feature_id_col].to_numpy(),
        p_values=p_values,
        p_values_adj=adjust_fdr(p_values, method="bh"),
        log2_fc=log2_fc_arr,
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
    """Two-group differential expression using Mann-Whitney U test.

    Parameters
    ----------
    container : ScpContainer
        Input container
    assay_name : str
        Name of assay to analyze
    group_col : str
        Column in obs containing group labels
    group1, group2 : str
        Names of the two groups to compare
    layer_name : str, default="X"
        Layer to use for analysis
    alternative : str, default="two-sided"
        Alternative hypothesis
    missing_strategy : str, default="ignore"
        How to handle missing values
    min_samples_per_group : int, default=3
        Minimum samples required per group
    log2_fc_offset : float, default=1.0
        Offset for log2 fold change calculation

    Returns
    -------
    DiffExprResult
        Test results
    """
    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]
    if layer_name not in assay.layers:
        raise LayerNotFoundError(layer_name, assay_name)

    if group_col not in container.obs.columns:
        raise ValidationError(f"Group column '{group_col}' not found in obs", field="group_col")

    groups = container.obs[group_col].to_numpy()
    group_indices = extract_group_indices(groups, [group1, group2], min_samples_per_group)
    idx1, idx2 = group_indices[group1], group_indices[group2]

    X = assay.layers[layer_name].X
    n_features = X.shape[1]
    X_proc, valid_mask = handle_missing_values(X, strategy=missing_strategy)

    p_values = np.full(n_features, np.nan, dtype=np.float64)
    test_stats = np.full(n_features, np.nan, dtype=np.float64)
    log2_fc_arr = np.full(n_features, np.nan, dtype=np.float64)

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

        log2_fc_arr[j] = log2_fold_change(g1_vals, g2_vals, offset=log2_fc_offset)

    return DiffExprResult(
        feature_ids=assay.var[assay.feature_id_col].to_numpy(),
        p_values=p_values,
        p_values_adj=adjust_fdr(p_values, method="bh"),
        log2_fc=log2_fc_arr,
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


def _run_multi_group_test(
    X_proc: np.ndarray,
    group_indices: dict[str, np.ndarray],
    valid_mask: np.ndarray | None,
    n_features: int,
    test_fn,
    unique_groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Run multi-group statistical test across features.

    Parameters
    ----------
    X_proc : np.ndarray
        Processed data matrix
    group_indices : dict
        Mapping from group name to indices
    valid_mask : np.ndarray or None
        Valid value mask
    n_features : int
        Number of features
    test_fn : callable
        Statistical test function
    unique_groups : np.ndarray
        Unique group names

    Returns
    -------
    tuple
        (p_values, test_stats, group_means, group_medians)
    """
    p_values = np.full(n_features, np.nan, dtype=np.float64)
    test_stats = np.full(n_features, np.nan, dtype=np.float64)

    group_means = {
        f"{g}_mean": np.full(n_features, np.nan, dtype=np.float64) for g in unique_groups
    }
    group_medians = {
        f"{g}_median": np.full(n_features, np.nan, dtype=np.float64) for g in unique_groups
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

            group_data.append(g_vals)
            group_means[f"{g}_mean"][j] = np.mean(g_vals)
            group_medians[f"{g}_median"][j] = np.median(g_vals)

        if group_data is None:
            continue

        try:
            result = test_fn(*group_data)
            p_values[j] = result.pvalue
            test_stats[j] = result.statistic
        except ValueError:
            p_values[j] = 1.0
            test_stats[j] = 0.0

    return p_values, test_stats, group_means, group_medians


def diff_expr_anova(
    container: ScpContainer,
    assay_name: str,
    group_col: str,
    layer_name: str = "X",
    missing_strategy: str = "ignore",
    min_samples_per_group: int = 3,
) -> DiffExprResult:
    """Multi-group differential expression using one-way ANOVA.

    Parameters
    ----------
    container : ScpContainer
        Input container
    assay_name : str
        Name of assay to analyze
    group_col : str
        Column in obs containing group labels
    layer_name : str, default="X"
        Layer to use for analysis
    missing_strategy : str, default="ignore"
        How to handle missing values
    min_samples_per_group : int, default=3
        Minimum samples required per group

    Returns
    -------
    DiffExprResult
        Test results
    """
    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]
    if layer_name not in assay.layers:
        raise LayerNotFoundError(layer_name, assay_name)

    if group_col not in container.obs.columns:
        raise ValidationError(f"Group column '{group_col}' not found in obs", field="group_col")

    groups = container.obs[group_col].to_numpy()
    unique_groups = groups[~isna(groups)]
    unique_groups = np.unique(unique_groups)
    n_groups = len(unique_groups)

    if n_groups < 2:
        raise ValidationError(
            f"Need at least 2 groups for ANOVA, found {n_groups}",
            field="group_col",
        )

    group_indices = extract_group_indices(groups, list(unique_groups), min_samples_per_group)

    X = assay.layers[layer_name].X
    n_features = X.shape[1]
    X_proc, valid_mask = handle_missing_values(X, strategy=missing_strategy)

    p_values, test_stats, group_means, group_medians = _run_multi_group_test(
        X_proc, group_indices, valid_mask, n_features, stats.f_oneway, unique_groups
    )

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
    """Multi-group differential expression using Kruskal-Wallis test.

    Parameters
    ----------
    container : ScpContainer
        Input container
    assay_name : str
        Name of assay to analyze
    group_col : str
        Column in obs containing group labels
    layer_name : str, default="X"
        Layer to use for analysis
    missing_strategy : str, default="ignore"
        How to handle missing values
    min_samples_per_group : int, default=3
        Minimum samples required per group

    Returns
    -------
    DiffExprResult
        Test results
    """
    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]
    if layer_name not in assay.layers:
        raise LayerNotFoundError(layer_name, assay_name)

    if group_col not in container.obs.columns:
        raise ValidationError(f"Group column '{group_col}' not found in obs", field="group_col")

    groups = container.obs[group_col].to_numpy()
    unique_groups = groups[~isna(groups)]
    unique_groups = np.unique(unique_groups)
    n_groups = len(unique_groups)

    if n_groups < 2:
        raise ValidationError(
            f"Need at least 2 groups for Kruskal-Wallis, found {n_groups}",
            field="group_col",
        )

    group_indices = extract_group_indices(groups, list(unique_groups), min_samples_per_group)

    X = assay.layers[layer_name].X
    n_features = X.shape[1]
    X_proc, valid_mask = handle_missing_values(X, strategy=missing_strategy)

    p_values, test_stats, group_means, group_medians = _run_multi_group_test(
        X_proc, group_indices, valid_mask, n_features, stats.kruskal, unique_groups
    )

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


def diff_expr_paired_ttest(
    container: ScpContainer,
    assay_name: str,
    group_col: str,
    pair_id_col: str,
    layer_name: str = "X",
    missing_strategy: str = "ignore",
    min_pairs_per_group: int = 3,
    log2_fc_offset: float = 1.0,
) -> DiffExprResult:
    """Paired sample t-test for differential expression.

    Parameters
    ----------
    container : ScpContainer
        Input container
    assay_name : str
        Name of assay to analyze
    group_col : str
        Column in obs containing group labels
    pair_id_col : str
        Column in obs containing pair IDs
    layer_name : str, default="X"
        Layer to use for analysis
    missing_strategy : str, default="ignore"
        How to handle missing values
    min_pairs_per_group : int, default=3
        Minimum pairs required per group
    log2_fc_offset : float, default=1.0
        Offset for log2 fold change calculation

    Returns
    -------
    DiffExprResult
        Test results
    """
    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]
    if layer_name not in assay.layers:
        raise LayerNotFoundError(layer_name, assay_name)

    if group_col not in container.obs.columns:
        raise ValidationError(f"Group column '{group_col}' not found in obs", field="group_col")

    if pair_id_col not in container.obs.columns:
        raise ValidationError(
            f"Pair ID column '{pair_id_col}' not found in obs", field="pair_id_col"
        )

    groups = container.obs[group_col].to_numpy()
    pair_ids = container.obs[pair_id_col].to_numpy()

    unique_groups = groups[~isna(groups)]
    unique_groups = np.unique(unique_groups)

    if len(unique_groups) != 2:
        raise ValidationError(
            f"Paired t-test requires exactly 2 groups, found {len(unique_groups)}",
            field="group_col",
        )

    group1, group2 = unique_groups[0], unique_groups[1]

    valid_pairs = validate_pairing(container.obs, group_col, pair_id_col, group1, group2)

    n_pairs = len(valid_pairs)
    if n_pairs < min_pairs_per_group:
        raise ValidationError(
            f"Only {n_pairs} complete pairs found, minimum {min_pairs_per_group} required",
            field="pair_id_col",
        )

    X = assay.layers[layer_name].X
    n_features = X.shape[1]
    X_proc, valid_mask = handle_missing_values(X, strategy=missing_strategy)

    p_values = np.full(n_features, np.nan, dtype=np.float64)
    test_stats = np.full(n_features, np.nan, dtype=np.float64)
    log2_fc_arr = np.full(n_features, np.nan, dtype=np.float64)
    effect_sizes = np.full(n_features, np.nan, dtype=np.float64)

    g1_means = np.full(n_features, np.nan, dtype=np.float64)
    g2_means = np.full(n_features, np.nan, dtype=np.float64)
    g1_medians = np.full(n_features, np.nan, dtype=np.float64)
    g2_medians = np.full(n_features, np.nan, dtype=np.float64)

    for j in range(n_features):
        differences = []
        g1_vals_list = []
        g2_vals_list = []

        for idx1, idx2 in valid_pairs:
            v1 = X_proc[idx1, j]
            v2 = X_proc[idx2, j]

            if valid_mask is not None:
                if not valid_mask[idx1, j] or not valid_mask[idx2, j]:
                    continue

            differences.append(v1 - v2)
            g1_vals_list.append(v1)
            g2_vals_list.append(v2)

        if len(differences) < 2:
            continue

        diff_arr = np.array(differences)
        g1_arr = np.array(g1_vals_list)
        g2_arr = np.array(g2_vals_list)

        g1_means[j] = np.mean(g1_arr)
        g2_means[j] = np.mean(g2_arr)
        g1_medians[j] = np.median(g1_arr)
        g2_medians[j] = np.median(g2_arr)

        result = stats.ttest_rel(g1_arr, g2_arr)
        p_values[j] = result.pvalue
        test_stats[j] = result.statistic

        log2_fc_arr[j] = log2_fold_change(g1_arr, g2_arr, offset=log2_fc_offset)

        mean_diff = np.mean(diff_arr)
        sd_diff = np.std(diff_arr, ddof=1)
        if sd_diff > 0:
            effect_sizes[j] = mean_diff / sd_diff
        else:
            effect_sizes[j] = 0.0

    return DiffExprResult(
        feature_ids=assay.var[assay.feature_id_col].to_numpy(),
        p_values=p_values,
        p_values_adj=adjust_fdr(p_values, method="bh"),
        log2_fc=log2_fc_arr,
        test_statistics=test_stats,
        effect_sizes=effect_sizes,
        group_stats={
            f"{group1}_mean": g1_means,
            f"{group2}_mean": g2_means,
            f"{group1}_median": g1_medians,
            f"{group2}_median": g2_medians,
        },
        method="paired_ttest",
        params={
            "group_col": group_col,
            "pair_id_col": pair_id_col,
            "n_pairs": n_pairs,
            "group1": group1,
            "group2": group2,
            "missing_strategy": missing_strategy,
        },
    )


def check_homoscedasticity(
    container: ScpContainer,
    assay_name: str,
    group_col: str,
    layer_name: str = "X",
    test_type: str = "levene",
    center: str = "median",
    min_samples_per_group: int = 3,
) -> pl.DataFrame:
    """Test for equality of variances (homoscedasticity) across groups.

    Parameters
    ----------
    container : ScpContainer
        Input container
    assay_name : str
        Name of assay to analyze
    group_col : str
        Column in obs containing group labels
    layer_name : str, default="X"
        Layer to use for analysis
    test_type : str, default="levene"
        Type of test: "levene" or "brown-forsythe"
    center : str, default="median"
        Center measure for Levene's test
    min_samples_per_group : int, default=3
        Minimum samples required per group

    Returns
    -------
    pl.DataFrame
        Results with feature_id, statistic, p_value, p_value_adj
    """
    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]
    if layer_name not in assay.layers:
        raise LayerNotFoundError(layer_name, assay_name)

    if group_col not in container.obs.columns:
        raise ValidationError(f"Group column '{group_col}' not found in obs", field="group_col")

    groups = container.obs[group_col].to_numpy()
    unique_groups = groups[~isna(groups)]
    unique_groups = np.unique(unique_groups)

    group_indices = extract_group_indices(groups, list(unique_groups), min_samples_per_group)

    X = assay.layers[layer_name].X
    n_features = X.shape[1]
    X_proc, valid_mask = handle_missing_values(X, strategy="ignore")

    test_stats = np.full(n_features, np.nan, dtype=np.float64)
    p_values = np.full(n_features, np.nan, dtype=np.float64)

    test_fn = stats.levene if test_type == "levene" else stats.brownforsythe

    for j in range(n_features):
        group_data = []
        for g in unique_groups:
            idx = group_indices[g]
            g_vals = X_proc[idx, j]

            if valid_mask is not None:
                g_vals = g_vals[valid_mask[idx, j]]

            if len(g_vals) < 2:
                group_data = None
                break

            group_data.append(g_vals)

        if group_data is None:
            continue

        try:
            result = test_fn(*group_data, center=center)
            p_values[j] = result.pvalue
            test_stats[j] = result.statistic
        except ValueError:
            p_values[j] = 1.0
            test_stats[j] = 0.0

    p_values_adj = adjust_fdr(p_values, method="bh")

    return pl.DataFrame({
        "feature_id": assay.var[assay.feature_id_col].to_numpy(),
        "statistic": test_stats,
        "p_value": p_values,
        "p_value_adj": p_values_adj,
    }).sort("p_value")


def diff_expr_permutation_test(
    container: ScpContainer,
    assay_name: str,
    group_col: str,
    group1: str,
    group2: str,
    layer_name: str = "X",
    n_permutations: int = 1000,
    missing_strategy: str = "ignore",
    min_samples_per_group: int = 3,
    log2_fc_offset: float = 1.0,
    random_state: int | None = None,
) -> DiffExprResult:
    """Permutation-based non-parametric test for differential expression.

    Parameters
    ----------
    container : ScpContainer
        Input container
    assay_name : str
        Name of assay to analyze
    group_col : str
        Column in obs containing group labels
    group1, group2 : str
        Names of the two groups to compare
    layer_name : str, default="X"
        Layer to use for analysis
    n_permutations : int, default=1000
        Number of permutations
    missing_strategy : str, default="ignore"
        How to handle missing values
    min_samples_per_group : int, default=3
        Minimum samples required per group
    log2_fc_offset : float, default=1.0
        Offset for log2 fold change calculation
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    DiffExprResult
        Test results
    """
    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]
    if layer_name not in assay.layers:
        raise LayerNotFoundError(layer_name, assay_name)

    if group_col not in container.obs.columns:
        raise ValidationError(f"Group column '{group_col}' not found in obs", field="group_col")

    groups = container.obs[group_col].to_numpy()
    group_indices = extract_group_indices(groups, [group1, group2], min_samples_per_group)
    idx1, idx2 = group_indices[group1], group_indices[group2]

    X = assay.layers[layer_name].X
    n_features = X.shape[1]
    X_proc, valid_mask = handle_missing_values(X, strategy=missing_strategy)

    all_indices = np.concatenate([idx1, idx2])
    n1, n2 = len(idx1), len(idx2)

    rng = np.random.default_rng(random_state)

    p_values = np.full(n_features, np.nan, dtype=np.float64)
    test_stats = np.full(n_features, np.nan, dtype=np.float64)
    log2_fc_arr = np.full(n_features, np.nan, dtype=np.float64)

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

        observed_diff = np.median(g1_vals) - np.median(g2_vals)
        test_stats[j] = observed_diff
        log2_fc_arr[j] = log2_fold_change(g1_vals, g2_vals, offset=log2_fc_offset)

        combined = np.concatenate([g1_vals, g2_vals])
        n_total = len(combined)

        extreme_count = 0
        for _ in range(n_permutations):
            permuted = rng.permutation(combined)
            perm_g1 = permuted[:len(g1_vals)]
            perm_g2 = permuted[len(g1_vals):]

            perm_diff = np.median(perm_g1) - np.median(perm_g2)

            if abs(perm_diff) >= abs(observed_diff):
                extreme_count += 1

        p_values[j] = extreme_count / n_permutations

    return DiffExprResult(
        feature_ids=assay.var[assay.feature_id_col].to_numpy(),
        p_values=p_values,
        p_values_adj=adjust_fdr(p_values, method="bh"),
        log2_fc=log2_fc_arr,
        test_statistics=test_stats,
        effect_sizes=None,
        group_stats={
            f"{group1}_median": g1_medians,
            f"{group2}_median": g2_medians,
        },
        method="permutation_test",
        params={
            "group_col": group_col,
            "group1": group1,
            "group2": group2,
            "n_permutations": n_permutations,
            "random_state": random_state,
            "missing_strategy": missing_strategy,
        },
    )


if __name__ == "__main__":
    import sys

    print("Running diff_expr module tests...")

    print("Test 1: FDR adjustment")
    p_values = np.array([0.001, 0.01, 0.05, 0.1, 0.5])
    adjusted = adjust_fdr(p_values, method="bh")
    assert np.all(adjusted >= p_values), "Adjusted p-values should be >= original"
    assert np.all(adjusted <= 1.0), "Adjusted p-values should be <= 1"
    print("  PASSED")

    print("Test 2: NaN handling")
    p_with_nan = np.array([0.01, np.nan, 0.05, 0.1, np.nan])
    adjusted_nan = adjust_fdr(p_with_nan, method="bh")
    assert np.isnan(adjusted_nan[1]) and np.isnan(adjusted_nan[4])
    assert not np.isnan(adjusted_nan[0])
    print("  PASSED")

    print("Test 3: Cohen's d")
    x, y = np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    d = cohens_d(x, y)
    assert -2 < d < 2
    print("  PASSED")

    print("Test 4: Log2 fold change")
    g1, g2 = np.array([10.0, 12.0, 11.0, 13.0, 10.0]), np.array([5.0, 6.0, 5.5, 6.5, 5.0])
    fc = log2_fold_change(g1, g2)
    assert fc > 0
    print("  PASSED")

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

    print("Test 6: Missing value handling")
    X_missing = np.array([[1.0, 2.0, np.nan], [3.0, np.nan, 4.0], [5.0, 6.0, 7.0]])
    X_proc, mask = handle_missing_values(X_missing, strategy="ignore")
    assert mask is not None and mask.sum() == 7
    print("  PASSED")

    print("Test 7: BY correction")
    p_values = np.array([0.001, 0.01, 0.05, 0.1, 0.5])
    adjusted_bh = adjust_fdr(p_values, method="bh")
    adjusted_by = adjust_fdr(p_values, method="by")
    assert np.all(adjusted_by >= adjusted_bh)
    print("  PASSED")

    print("Test 8: isna function")
    assert np.array_equal(isna(np.array([1.0, np.nan, 3.0])), [False, True, False])
    assert isna(np.array(["a", None, "NaN"]))[1]
    print("  PASSED")

    print()
    print("=" * 60)
    print("All diff_expr core tests passed!")
    print("=" * 60)
    sys.exit(0)
