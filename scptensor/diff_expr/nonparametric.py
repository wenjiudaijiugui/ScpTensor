"""Non-parametric differential expression methods.

This module implements distribution-free tests for differential expression,
suitable for data that violates normality assumptions or has unequal
variances.

Supported methods:
    - diff_expr_wilcoxon: Wilcoxon rank-sum test (Mann-Whitney)
    - diff_expr_brunner_munzel: Brunner-Munzel test for heteroscedastic data
"""

from __future__ import annotations

from typing import cast

import numpy as np
import scipy.sparse as sp
import scipy.stats as stats

from scptensor.core.exceptions import (
    AssayNotFoundError,
    LayerNotFoundError,
    ValidationError,
)
from scptensor.core.structures import ScpContainer
from scptensor.diff_expr._utils import (
    extract_group_indices_from_obs,
    log2_fold_change,
    rank_biserial_correlation,
    validate_pairing,
)
from scptensor.diff_expr.core import DiffExprResult, adjust_fdr


def brunner_munzel_statistic(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float]:
    """Calculate Brunner-Munzel statistic and p-value.

    The Brunner-Munzel test is a non-parametric test for the equality of
    stochastic dominance between two groups. It is robust to unequal
    variances and different sample sizes.

    Parameters
    ----------
    x, y : np.ndarray
        Data values from two groups

    Returns
    -------
    tuple[float, float]
        (statistic, p_value) where statistic is approximately standard normal

    References
    ----------
    Brunner, E., & Munzel, U. (2000). The nonparametric Behrens-Fisher problem:
    Asymptotic theory and a small-sample approximation. Biometrical Journal, 42(1), 17-25.
    """
    n1, n2 = len(x), len(y)

    if n1 < 2 or n2 < 2:
        return 0.0, 1.0

    combined = np.concatenate([x, y])
    ranks = stats.rankdata(combined)

    r1 = np.mean(ranks[:n1])
    r2 = np.mean(ranks[n1:])

    n = n1 + n2
    pHat = r1 / n

    rank_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if combined[i] < combined[j]:
                rank_matrix[i, j] = 1.0
            elif combined[i] == combined[j]:
                rank_matrix[i, j] = 0.5

    r1_rows = rank_matrix[:n1, :]
    r2_rows = rank_matrix[n1:, :]

    h1 = np.mean(r1_rows, axis=1)
    h2 = np.mean(r2_rows, axis=1)

    s1_sq = np.var(h1, ddof=1) if n1 > 1 else 0.0
    s2_sq = np.var(h2, ddof=1) if n2 > 1 else 0.0

    se = np.sqrt(s1_sq / n1 + s2_sq / n2)

    if se < 1e-10:
        return 0.0, 1.0

    statistic = (pHat - 0.5) / se
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(statistic)))

    return float(statistic), float(p_value)


def diff_expr_wilcoxon(
    container: ScpContainer,
    assay_name: str,
    layer: str,
    groupby: str,
    group1: str,
    group2: str,
    paired: bool = False,
    zero_method: str = "pratt",
    alternative: str = "two-sided",
    min_samples_per_group: int = 3,
    missing_strategy: str = "ignore",
    log2_fc_offset: float = 1.0,
) -> DiffExprResult:
    """Wilcoxon rank-sum test (paired or unpaired).

    Non-parametric test for comparing two groups. More robust than
    t-test for non-normal data.

    Parameters
    ----------
    container : ScpContainer
        Input data container
    assay_name : str
        Assay containing the data
    layer : str
        Layer with expression data
    groupby : str
        Column in obs defining groups
    group1, group2 : str
        Group names to compare
    paired : bool, default=False
        Use paired test. Requires matching pairs in obs
    zero_method : str, default="pratt"
        Method for handling zero values (pratt, wilcox, zsplit)
    alternative : str, default="two-sided"
        Alternative hypothesis (two-sided, greater, less)
    min_samples_per_group : int, default=3
        Minimum samples required per group
    missing_strategy : str, default="ignore"
        How to handle missing values
    log2_fc_offset : float, default=1.0
        Offset for log2 fold change calculation

    Returns
    -------
    DiffExprResult
        Test results with rank-based statistics
    """
    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]
    if layer not in assay.layers:
        raise LayerNotFoundError(layer, assay_name)

    if zero_method not in {"pratt", "wilcox", "zsplit"}:
        raise ValidationError(
            f"Unknown zero_method: {zero_method}. Use 'pratt', 'wilcox', or 'zsplit'.",
            field="zero_method",
        )

    if alternative not in {"two-sided", "greater", "less"}:
        raise ValidationError(
            f"Unknown alternative: {alternative}. Use 'two-sided', 'greater', or 'less'.",
            field="alternative",
        )

    X = assay.layers[layer].X
    M = assay.layers[layer].M
    n_features = X.shape[1]

    if sp.issparse(X):
        X = cast(sp.spmatrix, X).toarray()
    if M is not None and sp.issparse(M):
        M = cast(sp.spmatrix, M).toarray()

    X_arr = X.astype(np.float64, copy=False)
    M_arr = M.astype(np.float64, copy=False) if M is not None else None

    # Handle missing values
    nan_mask = np.isnan(X_arr)
    if missing_strategy == "zero" and nan_mask.any():
        X_arr = np.nan_to_num(X_arr, copy=False, nan=0.0)
        if M_arr is not None:
            M_arr = M_arr.copy()
            M_arr[nan_mask] = 1
    elif missing_strategy == "median" and nan_mask.any():
        col_medians = np.nanmedian(X_arr, axis=0, keepdims=True)
        X_arr = np.where(nan_mask, col_medians, X_arr)
        if M_arr is not None:
            M_arr = M_arr.copy()
            M_arr[nan_mask] = 1

    p_values = np.full(n_features, np.nan, dtype=np.float64)
    test_stats = np.full(n_features, np.nan, dtype=np.float64)
    log2_fc_arr = np.full(n_features, np.nan, dtype=np.float64)
    effect_sizes = np.full(n_features, np.nan, dtype=np.float64)

    g1_medians = np.full(n_features, np.nan, dtype=np.float64)
    g2_medians = np.full(n_features, np.nan, dtype=np.float64)

    if paired:
        pair_id_col = f"{groupby}_pair_id"
        if pair_id_col not in container.obs.columns:
            for col in ["pair_id", "subject_id", "sample_id", "patient_id"]:
                if col in container.obs.columns:
                    pair_id_col = col
                    break
            else:
                raise ValidationError(
                    "Paired test requires a pair ID column in obs. "
                    f"Expected '{groupby}_pair_id' or 'pair_id'.",
                    field="pair_id_col",
                )

        valid_pairs = validate_pairing(container.obs, groupby, pair_id_col, group1, group2)

        if len(valid_pairs) < min_samples_per_group:
            raise ValidationError(
                f"Only {len(valid_pairs)} complete pairs found, "
                f"minimum {min_samples_per_group} required",
                field="pair_id_col",
            )

        for j in range(n_features):
            differences = []
            g1_vals_list = []
            g2_vals_list = []

            for pair_idx1, pair_idx2 in valid_pairs:
                v1 = X_arr[pair_idx1, j]
                v2 = X_arr[pair_idx2, j]

                if M_arr is not None:
                    if M_arr[pair_idx1, j] != 0 or M_arr[pair_idx2, j] != 0:
                        continue
                elif missing_strategy == "ignore":
                    if np.isnan(v1) or np.isnan(v2):
                        continue

                differences.append(v1 - v2)
                g1_vals_list.append(v1)
                g2_vals_list.append(v2)

            if len(differences) < min_samples_per_group:
                continue

            diff_arr = np.array(differences)
            g1_arr = np.array(g1_vals_list)
            g2_arr = np.array(g2_vals_list)

            g1_medians[j] = np.median(g1_arr)
            g2_medians[j] = np.median(g2_arr)

            try:
                result = stats.wilcoxon(
                    diff_arr,
                    zero_method=zero_method,
                    alternative=alternative,
                )
                p_values[j] = result.pvalue
                test_stats[j] = float(result.statistic)
            except ValueError:
                p_values[j] = 1.0
                test_stats[j] = 0.0

            log2_fc_arr[j] = log2_fold_change(g1_arr, g2_arr, offset=log2_fc_offset)
            effect_sizes[j] = rank_biserial_correlation(g1_arr, g2_arr)

        method_name = "wilcoxon_paired"

    else:
        idx1, idx2 = extract_group_indices_from_obs(container.obs, groupby, group1, group2)

        if len(idx1) < min_samples_per_group:
            raise ValidationError(
                f"Group '{group1}' has only {len(idx1)} samples, "
                f"minimum {min_samples_per_group} required",
                field=group1,
            )
        if len(idx2) < min_samples_per_group:
            raise ValidationError(
                f"Group '{group2}' has only {len(idx2)} samples, "
                f"minimum {min_samples_per_group} required",
                field=group2,
            )

        for j in range(n_features):
            g1_vals = X_arr[idx1, j]
            g2_vals = X_arr[idx2, j]

            if M_arr is not None:
                g1_valid = M_arr[idx1, j] == 0
                g2_valid = M_arr[idx2, j] == 0
                g1_vals = g1_vals[g1_valid]
                g2_vals = g2_vals[g2_valid]
            elif missing_strategy == "ignore":
                g1_valid = ~np.isnan(g1_vals)
                g2_valid = ~np.isnan(g2_vals)
                g1_vals = g1_vals[g1_valid]
                g2_vals = g2_vals[g2_valid]

            if len(g1_vals) < 2 or len(g2_vals) < 2:
                continue

            g1_medians[j] = np.median(g1_vals)
            g2_medians[j] = np.median(g2_vals)

            try:
                result = stats.ranksums(g1_vals, g2_vals)
                p_values[j] = result.pvalue
                test_stats[j] = float(result.statistic)
            except ValueError:
                p_values[j] = 1.0
                test_stats[j] = 0.0

            log2_fc_arr[j] = log2_fold_change(g1_vals, g2_vals, offset=log2_fc_offset)
            effect_sizes[j] = rank_biserial_correlation(g1_vals, g2_vals)

        method_name = "wilcoxon_ranksum"

    return DiffExprResult(
        feature_ids=assay.var[assay.feature_id_col].to_numpy(),
        p_values=p_values,
        p_values_adj=adjust_fdr(p_values, method="bh"),
        log2_fc=log2_fc_arr,
        test_statistics=test_stats,
        effect_sizes=effect_sizes,
        group_stats={
            f"{group1}_median": g1_medians,
            f"{group2}_median": g2_medians,
        },
        method=method_name,
        params={
            "groupby": groupby,
            "group1": group1,
            "group2": group2,
            "paired": paired,
            "zero_method": zero_method,
            "alternative": alternative,
            "missing_strategy": missing_strategy,
        },
    )


def diff_expr_brunner_munzel(
    container: ScpContainer,
    assay_name: str,
    layer: str,
    groupby: str,
    group1: str,
    group2: str,
    alternative: str = "two-sided",
    min_samples_per_group: int = 3,
    missing_strategy: str = "ignore",
    log2_fc_offset: float = 1.0,
) -> DiffExprResult:
    """Brunner-Munzel test for heteroscedastic data.

    Tests for difference in stochastic equality between groups.
    More robust than Wilcoxon when variances differ significantly.

    Parameters
    ----------
    container : ScpContainer
        Input data container
    assay_name : str
        Assay containing the data
    layer : str
        Layer with expression data
    groupby : str
        Column in obs defining groups
    group1, group2 : str
        Group names to compare
    alternative : str, default="two-sided"
        Alternative hypothesis (two-sided, greater, less)
    min_samples_per_group : int, default=3
        Minimum samples required per group
    missing_strategy : str, default="ignore"
        How to handle missing values
    log2_fc_offset : float, default=1.0
        Offset for log2 fold change calculation

    Returns
    -------
    DiffExprResult
        Test results with Brunner-Munzel statistics
    """
    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]
    if layer not in assay.layers:
        raise LayerNotFoundError(layer, assay_name)

    if alternative not in {"two-sided", "greater", "less"}:
        raise ValidationError(
            f"Unknown alternative: {alternative}. Use 'two-sided', 'greater', or 'less'.",
            field="alternative",
        )

    idx1, idx2 = extract_group_indices_from_obs(container.obs, groupby, group1, group2)

    if len(idx1) < min_samples_per_group:
        raise ValidationError(
            f"Group '{group1}' has only {len(idx1)} samples, "
            f"minimum {min_samples_per_group} required",
            field=group1,
        )
    if len(idx2) < min_samples_per_group:
        raise ValidationError(
            f"Group '{group2}' has only {len(idx2)} samples, "
            f"minimum {min_samples_per_group} required",
            field=group2,
        )

    X = assay.layers[layer].X
    M = assay.layers[layer].M
    n_features = X.shape[1]

    if sp.issparse(X):
        X = cast(sp.spmatrix, X).toarray()
    if M is not None and sp.issparse(M):
        M = cast(sp.spmatrix, M).toarray()

    X_arr = X.astype(np.float64, copy=False)
    M_arr = M.astype(np.float64, copy=False) if M is not None else None

    # Handle missing values
    nan_mask = np.isnan(X_arr)
    if missing_strategy == "zero" and nan_mask.any():
        X_arr = np.nan_to_num(X_arr, copy=False, nan=0.0)
        if M_arr is not None:
            M_arr = M_arr.copy()
            M_arr[nan_mask] = 1
    elif missing_strategy == "median" and nan_mask.any():
        col_medians = np.nanmedian(X_arr, axis=0, keepdims=True)
        X_arr = np.where(nan_mask, col_medians, X_arr)
        if M_arr is not None:
            M_arr = M_arr.copy()
            M_arr[nan_mask] = 1

    p_values = np.full(n_features, np.nan, dtype=np.float64)
    test_stats = np.full(n_features, np.nan, dtype=np.float64)
    log2_fc_arr = np.full(n_features, np.nan, dtype=np.float64)
    relative_effects = np.full(n_features, np.nan, dtype=np.float64)

    g1_medians = np.full(n_features, np.nan, dtype=np.float64)
    g2_medians = np.full(n_features, np.nan, dtype=np.float64)

    for j in range(n_features):
        g1_vals = X_arr[idx1, j]
        g2_vals = X_arr[idx2, j]

        if M_arr is not None:
            g1_valid = M_arr[idx1, j] == 0
            g2_valid = M_arr[idx2, j] == 0
            g1_vals = g1_vals[g1_valid]
            g2_vals = g2_vals[g2_valid]
        elif missing_strategy == "ignore":
            g1_valid = ~np.isnan(g1_vals)
            g2_valid = ~np.isnan(g2_vals)
            g1_vals = g1_vals[g1_valid]
            g2_vals = g2_vals[g2_valid]

        if len(g1_vals) < 2 or len(g2_vals) < 2:
            continue

        g1_medians[j] = np.median(g1_vals)
        g2_medians[j] = np.median(g2_vals)

        statistic, p_value = brunner_munzel_statistic(g1_vals, g2_vals)

        if alternative == "greater":
            if statistic < 0:
                p_value = 1.0 - p_value / 2.0
            else:
                p_value = p_value / 2.0 if p_value < 0.5 else 1.0 - p_value / 2.0
        elif alternative == "less":
            if statistic > 0:
                p_value = 1.0 - p_value / 2.0
            else:
                p_value = p_value / 2.0 if p_value < 0.5 else 1.0 - p_value / 2.0

        p_values[j] = p_value
        test_stats[j] = statistic

        n1, n2 = len(g1_vals), len(g2_vals)
        comparisons = np.subtract.outer(g1_vals, g2_vals)
        n_less = np.sum(comparisons < 0)
        n_equal = np.sum(comparisons == 0)
        pHat = (n_less + 0.5 * n_equal) / (n1 * n2)
        relative_effects[j] = pHat

        log2_fc_arr[j] = log2_fold_change(g1_vals, g2_vals, offset=log2_fc_offset)

    return DiffExprResult(
        feature_ids=assay.var[assay.feature_id_col].to_numpy(),
        p_values=p_values,
        p_values_adj=adjust_fdr(p_values, method="bh"),
        log2_fc=log2_fc_arr,
        test_statistics=test_stats,
        effect_sizes=relative_effects,
        group_stats={
            f"{group1}_median": g1_medians,
            f"{group2}_median": g2_medians,
        },
        method="brunner_munzel",
        params={
            "groupby": groupby,
            "group1": group1,
            "group2": group2,
            "alternative": alternative,
            "missing_strategy": missing_strategy,
        },
    )


if __name__ == "__main__":
    import sys

    print("Running nonparametric module tests...")

    print("Test 1: Brunner-Munzel statistic")
    x, y = np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    stat, p_val = brunner_munzel_statistic(x, y)
    assert isinstance(stat, float)
    assert 0 <= p_val <= 1
    print("  PASSED")

    print("Test 2: Rank-biserial correlation")
    effect = rank_biserial_correlation(x, y)
    assert -1 <= effect <= 1
    print("  PASSED")

    print("Test 3: Log2 fold change")
    fc = log2_fold_change(np.array([10.0, 12.0, 11.0]), np.array([5.0, 6.0, 5.5]))
    assert fc > 0
    print("  PASSED")

    print("Test 4: DiffExprResult compatibility")
    result = DiffExprResult(
        feature_ids=np.array(["P1", "P2"]),
        p_values=np.array([0.01, 0.05]),
        p_values_adj=np.array([0.02, 0.10]),
        log2_fc=np.array([1.0, -0.5]),
        test_statistics=np.array([2.5, -1.2]),
        effect_sizes=np.array([0.5, 0.3]),
        group_stats={"A_median": np.array([10.0, 5.0]), "B_median": np.array([5.0, 8.0])},
        method="wilcoxon",
        params={"group1": "A", "group2": "B"},
    )
    df = result.to_dataframe()
    assert df.shape[0] == 2
    print("  PASSED")

    print()
    print("=" * 60)
    print("All nonparametric tests passed!")
    print("=" * 60)
    sys.exit(0)
