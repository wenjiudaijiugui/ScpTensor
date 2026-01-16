"""Non-parametric differential expression methods.

This module implements distribution-free tests for differential expression,
suitable for data that violates normality assumptions or has unequal
variances.

Supported methods:
    - diff_expr_wilcoxon: Wilcoxon rank-sum test (Mann-Whitney)
    - diff_expr_brunner_munzel: Brunner-Munzel test for heteroscedastic data
"""

from __future__ import annotations

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

# Import from core module
from scptensor.diff_expr.core import DiffExprResult, adjust_fdr


def _extract_group_indices(
    obs: pl.DataFrame,
    groupby: str,
    group1: str,
    group2: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract and validate group indices.

    Parameters
    ----------
    obs : pl.DataFrame
        Sample metadata
    groupby : str
        Column name in obs defining groups
    group1 : str
        First group name
    group2 : str
        Second group name

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Indices for group1 and group2

    Raises
    ------
    ValidationError
        If groupby column not found or groups have no samples
    """
    if groupby not in obs.columns:
        raise ValidationError(f"Group column '{groupby}' not found in obs", field="groupby")

    groups = obs[groupby].to_numpy()

    idx1 = np.where(groups == group1)[0]
    idx2 = np.where(groups == group2)[0]

    if len(idx1) == 0:
        raise ValidationError(f"Group '{group1}' has no samples", field=group1)
    if len(idx2) == 0:
        raise ValidationError(f"Group '{group2}' has no samples", field=group2)

    return idx1, idx2


def _validate_pairing(
    obs: pl.DataFrame,
    groupby: str,
    pair_id_col: str,
    group1: str,
    group2: str,
) -> list[tuple[int, int]]:
    """Validate and extract paired sample indices.

    Parameters
    ----------
    obs : pl.DataFrame
        Sample metadata
    groupby : str
        Column name in obs defining groups
    pair_id_col : str
        Column name in obs containing pair IDs
    group1 : str
        First group name
    group2 : str
        Second group name

    Returns
    -------
    list[tuple[int, int]]
        List of (index1, index2) tuples for each valid pair

    Raises
    ------
    ValidationError
        If pairing is invalid or insufficient pairs found
    """
    if pair_id_col not in obs.columns:
        raise ValidationError(
            f"Pair ID column '{pair_id_col}' not found in obs", field="pair_id_col"
        )

    groups = obs[groupby].to_numpy()
    pair_ids = obs[pair_id_col].to_numpy()

    # Build pairing: find samples that share the same pair_id
    pair_to_idx: dict[str, dict[str, int]] = {}
    for i, (g, pid) in enumerate(zip(groups, pair_ids, strict=False)):
        if _isna_array(np.array(g)) or _isna_array(np.array(pid)):
            continue
        pid_str = str(pid)
        if pid_str not in pair_to_idx:
            pair_to_idx[pid_str] = {}
        pair_to_idx[pid_str][str(g)] = i

    # Extract complete pairs
    valid_pairs = []
    for idx_map in pair_to_idx.values():
        if str(group1) in idx_map and str(group2) in idx_map:
            valid_pairs.append((idx_map[str(group1)], idx_map[str(group2)]))

    return valid_pairs


def _isna_array(arr: np.ndarray) -> np.ndarray:
    """Check if array or value is NaN.

    Parameters
    ----------
    arr : np.ndarray
        Input array or scalar

    Returns
    -------
    np.ndarray
        True if NaN, False otherwise
    """
    if arr.ndim == 0:
        val = arr.item()
        if isinstance(val, str):
            return np.array(val in {"NaN", "nan", "NA", "None", ""})
        return np.array(bool(np.isnan(val)))
    if arr.dtype.kind in {"U", "S", "O"}:
        return np.array([str(v) in {"NaN", "nan", "NA", "None", ""} for v in arr])
    return np.isnan(arr)


def _wilcoxon_effect_size(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate rank-biserial correlation (effect size) for Wilcoxon test.

    The rank-biserial correlation is derived from the Mann-Whitney U statistic
    and measures the strength of association between group membership and ranks.

    Parameters
    ----------
    x : np.ndarray
        Data values from group 1
    y : np.ndarray
        Data values from group 2

    Returns
    -------
    float
        Rank-biserial correlation coefficient

    Notes
    -----
    The rank-biserial correlation is calculated as:
        rrb = 1 - (2 * U) / (n1 * n2)

    where U is the Mann-Whitney U statistic.

    Interpretation:
        |rrb| < 0.1: Negligible effect
        |rrb| < 0.3: Small effect
        |rrb| < 0.5: Medium effect
        |rrb| >= 0.5: Large effect

    References
    ----------
    Cureton, E. E. (1956). Rank-biserial correlation. Psychometrika, 21(3), 287-290.
    """
    from scipy.stats import mannwhitneyu

    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return 0.0

    try:
        result = mannwhitneyu(x, y, alternative="two-sided")
        u_stat = result.statistic
        # The rank-biserial correlation
        rrb = 1.0 - (2.0 * u_stat) / (n1 * n2)
        return float(rrb)
    except ValueError:
        return 0.0


def _brunner_munzel_statistic(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float]:
    """Calculate Brunner-Munzel statistic and p-value.

    The Brunner-Munzel test is a non-parametric test for the equality of
    stochastic dominance between two groups. It is robust to unequal
    variances and different sample sizes.

    Parameters
    ----------
    x : np.ndarray
        Data values from group 1
    y : np.ndarray
        Data values from group 2

    Returns
    -------
    tuple[float, float]
        (statistic, p_value) where statistic is approximately standard normal

    Notes
    -----
    The Brunner-Munzel statistic tests the hypothesis:
        H0: P(X < Y) = P(Y < X) = 0.5

    The test statistic is calculated as:
        W = (n1 * n2 * (pHat - 0.5)) / sqrt(var)

    where:
        pHat = (1 / (n1 * n2)) * sum(I(x_i < y_j) + 0.5 * I(x_i = y_j))

    References
    ----------
    Brunner, E., & Munzel, U. (2000). The nonparametric Behrens-Fisher problem:
    Asymptotic theory and a small-sample approximation. Biometrical Journal, 42(1), 17-25.
    """
    n1, n2 = len(x), len(y)

    if n1 < 2 or n2 < 2:
        return 0.0, 1.0

    # Calculate relative effects
    # pHat estimates P(X < Y) + 0.5 * P(X = Y)
    combined = np.concatenate([x, y])
    ranks = stats.rankdata(combined)

    # Average ranks for ties
    r1 = np.mean(ranks[:n1])
    r2 = np.mean(ranks[n1:])

    n = n1 + n2

    # Brunner-Munzel statistic
    pHat = r1 / n  # Estimated relative effect

    # Calculate variance components
    # Rank-based variance estimator
    rank_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if combined[i] < combined[j]:
                rank_matrix[i, j] = 1.0
            elif combined[i] == combined[j]:
                rank_matrix[i, j] = 0.5

    # Variance of relative effects
    r1_rows = rank_matrix[:n1, :]
    r2_rows = rank_matrix[n1:, :]

    # Mean ranks for each comparison
    h1 = np.mean(r1_rows, axis=1)
    h2 = np.mean(r2_rows, axis=1)

    # Sample variance of relative effects
    s1_sq = np.var(h1, ddof=1) if n1 > 1 else 0.0
    s2_sq = np.var(h2, ddof=1) if n2 > 1 else 0.0

    # Standard error
    se = np.sqrt(s1_sq / n1 + s2_sq / n2)

    if se < 1e-10:
        return 0.0, 1.0

    # Test statistic (approximately standard normal)
    statistic = (pHat - 0.5) / se

    # Two-sided p-value from standard normal
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(statistic)))

    return float(statistic), float(p_value)


def _handle_mask_values(
    X: np.ndarray | sp.spmatrix,
    M: np.ndarray | sp.spmatrix | None,
    idx: np.ndarray,
    feature_idx: int,
) -> np.ndarray:
    """Extract valid values from a feature, handling missing data.

    Parameters
    ----------
    X : np.ndarray or sp.spmatrix
        Data matrix
    M : np.ndarray or sp.spmatrix or None
        Mask matrix (0 = valid, non-zero = invalid)
    idx : np.ndarray
        Sample indices to extract
    feature_idx : int
        Feature column index

    Returns
    -------
    np.ndarray
        Valid values for the specified samples and feature
    """
    if sp.issparse(X):
        X_dense: np.ndarray = X.toarray()
        X = X_dense

    values = X[idx, feature_idx]

    if M is not None:
        if sp.issparse(M):
            M_dense: np.ndarray = M.toarray()
            M = M_dense
        mask = M[idx, feature_idx]
        # Keep only values where mask == 0 (valid)
        valid_mask = mask == 0
        values = values[valid_mask]

    return values


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
        Input data container.
    assay_name : str
        Assay containing the data.
    layer : str
        Layer with expression data.
    groupby : str
        Column in obs defining groups.
    group1 : str
        First group name.
    group2 : str
        Second group name.
    paired : bool, default=False
        Use paired test. Requires matching pairs in obs.
    zero_method : str, default="pratt"
        Method for handling zero values (pratt, wilcox, zsplit).
    alternative : str, default="two-sided"
        Alternative hypothesis (two-sided, greater, less).
    min_samples_per_group : int, default=3
        Minimum samples required per group.
    missing_strategy : str, default="ignore"
        How to handle missing values: "ignore", "zero", or "median".
    log2_fc_offset : float, default=1.0
        Offset for log2 fold change calculation.

    Returns
    -------
    DiffExprResult
        Test results with rank-based statistics.

    Raises
    ------
    AssayNotFoundError
        If assay_name not found.
    LayerNotFoundError
        If layer not found.
    ValidationError
        If groups have insufficient samples or pairs don't match.

    Notes
    -----
    The Wilcoxon rank-sum test (Mann-Whitney U test) compares the
    distributions of two groups without assuming normality.

    For paired tests, the Wilcoxon signed-rank test is used.

    The zero_method parameter controls how zero differences are handled:
        - "pratt": Include zeros in ranking (default, recommended)
        - "wilcox": Discard zero differences
        - "zsplit": Split zeros between positive and negative ranks

    References
    ----------
    .. [1] Mann, H. B., & Whitney, D. R. (1947). On a test of whether
           one of two random variables is stochastically larger than
           the other. Annals of Mathematical Statistics 18:50-60.

    .. [2] Wilcoxon, F. (1945). Individual comparisons by ranking methods.
           Biometrics Bulletin 1(6):80-83.
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
        X_dense: np.ndarray = X.toarray()
        X = X_dense
    if M is not None and sp.issparse(M):
        M_dense: np.ndarray = M.toarray()
        M = M_dense

    # Convert sparse to dense for processing
    X_arr = X.astype(np.float64, copy=False)
    M_arr = M.astype(np.float64, copy=False) if M is not None else None

    # Handle missing values in X
    nan_mask = np.isnan(X_arr)
    if missing_strategy == "zero" and nan_mask.any():
        X_arr = np.nan_to_num(X_arr, copy=False, nan=0.0)
        if M_arr is not None:
            # Update mask for previously NaN values
            M_arr = M_arr.copy()
            M_arr[nan_mask] = 1  # Mark as imputed
    elif missing_strategy == "median" and nan_mask.any():
        col_medians = np.nanmedian(X_arr, axis=0, keepdims=True)
        X_arr = np.where(nan_mask, col_medians, X_arr)
        if M_arr is not None:
            M_arr = M_arr.copy()
            M_arr[nan_mask] = 1  # Mark as imputed

    # Initialize result arrays
    p_values = np.full(n_features, np.nan, dtype=np.float64)
    test_stats = np.full(n_features, np.nan, dtype=np.float64)
    log2_fc = np.full(n_features, np.nan, dtype=np.float64)
    effect_sizes = np.full(n_features, np.nan, dtype=np.float64)

    g1_medians = np.full(n_features, np.nan, dtype=np.float64)
    g2_medians = np.full(n_features, np.nan, dtype=np.float64)

    if paired:
        # Validate pairing
        pair_id_col = f"{groupby}_pair_id"
        if pair_id_col not in container.obs.columns:
            # Try common pair column names
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

        valid_pairs = _validate_pairing(container.obs, groupby, pair_id_col, group1, group2)

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

                # Check mask
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

            # Wilcoxon signed-rank test for paired data
            try:
                result = stats.wilcoxon(
                    diff_arr,
                    zero_method=zero_method,
                    alternative=alternative,
                )
                p_values[j] = result.pvalue
                test_stats[j] = float(result.statistic)
            except ValueError:
                # Too few non-zero differences or all zeros
                p_values[j] = 1.0
                test_stats[j] = 0.0

            log2_fc[j] = _log2_fold_change(g1_arr, g2_arr, offset=log2_fc_offset)

            # Effect size for paired test (rank-biserial)
            effect_sizes[j] = _wilcoxon_effect_size(g1_arr, g2_arr)

        method_name = "wilcoxon_paired"

    else:
        # Unpaired test
        idx1, idx2 = _extract_group_indices(container.obs, groupby, group1, group2)

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

            # Filter by mask
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

            # Wilcoxon rank-sum test (Mann-Whitney U)
            try:
                result = stats.ranksums(g1_vals, g2_vals)
                p_values[j] = result.pvalue
                test_stats[j] = float(result.statistic)
            except ValueError:
                p_values[j] = 1.0
                test_stats[j] = 0.0

            log2_fc[j] = _log2_fold_change(g1_vals, g2_vals, offset=log2_fc_offset)
            effect_sizes[j] = _wilcoxon_effect_size(g1_vals, g2_vals)

        method_name = "wilcoxon_ranksum"

    return DiffExprResult(
        feature_ids=assay.var[assay.feature_id_col].to_numpy(),
        p_values=p_values,
        p_values_adj=adjust_fdr(p_values, method="bh"),
        log2_fc=log2_fc,
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
        Input data container.
    assay_name : str
        Assay containing the data.
    layer : str
        Layer with expression data.
    groupby : str
        Column in obs defining groups.
    group1 : str
        First group name.
    group2 : str
        Second group name.
    alternative : str, default="two-sided"
        Alternative hypothesis (two-sided, greater, less).
    min_samples_per_group : int, default=3
        Minimum samples required per group.
    missing_strategy : str, default="ignore"
        How to handle missing values: "ignore", "zero", or "median".
    log2_fc_offset : float, default=1.0
        Offset for log2 fold change calculation.

    Returns
    -------
    DiffExprResult
        Test results with Brunner-Munzel statistics.

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
    The Brunner-Munzel test is robust to unequal variances and
    different sample sizes. It tests the null hypothesis of
    stochastic equality (P(X < Y) = P(Y < X) = 0.5).

    The test statistic is approximately standard normal.

    Advantages over Wilcoxon/Mann-Whitney:
        - Robust to unequal variances (heteroscedasticity)
        - Handles unequal sample sizes well
        - Better Type I error control when variances differ

    The relative effect pHat estimates the probability that a randomly
    selected value from group 1 is less than a randomly selected value
    from group 2:
        pHat = P(X < Y) + 0.5 * P(X = Y)

    Interpretation of pHat:
        - pHat = 0.5: Stochastic equality (no difference)
        - pHat > 0.5: Group 1 tends to be larger
        - pHat < 0.5: Group 2 tends to be larger

    References
    ----------
    .. [1] Brunner, E., & Munzel, U. (2000). The nonparametric
           Behrens-Fisher problem: Asymptotic theory and a small-
           sample approximation. Biometrical Journal 42:17-25.

    .. [2] Neubert, K., & Brunner, E. (2007). A studentized permutation
           test for the non-parametric Behrens-Fisher problem.
           Computational Statistics & Data Analysis 51(10):5192-5204.
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

    idx1, idx2 = _extract_group_indices(container.obs, groupby, group1, group2)

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
        X_dense: np.ndarray = X.toarray()
        X = X_dense
    if M is not None and sp.issparse(M):
        M_dense: np.ndarray = M.toarray()
        M = M_dense

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

    # Initialize result arrays
    p_values = np.full(n_features, np.nan, dtype=np.float64)
    test_stats = np.full(n_features, np.nan, dtype=np.float64)
    log2_fc = np.full(n_features, np.nan, dtype=np.float64)
    relative_effects = np.full(n_features, np.nan, dtype=np.float64)

    g1_medians = np.full(n_features, np.nan, dtype=np.float64)
    g2_medians = np.full(n_features, np.nan, dtype=np.float64)

    for j in range(n_features):
        g1_vals = X_arr[idx1, j]
        g2_vals = X_arr[idx2, j]

        # Filter by mask
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

        # Brunner-Munzel test
        statistic, p_value = _brunner_munzel_statistic(g1_vals, g2_vals)

        # Adjust for one-sided alternatives
        if alternative == "greater":
            # Test if group1 > group2
            if statistic < 0:
                p_value = 1.0 - p_value / 2.0
            else:
                p_value = p_value / 2.0 if p_value < 0.5 else 1.0 - p_value / 2.0
        elif alternative == "less":
            # Test if group1 < group2
            if statistic > 0:
                p_value = 1.0 - p_value / 2.0
            else:
                p_value = p_value / 2.0 if p_value < 0.5 else 1.0 - p_value / 2.0

        p_values[j] = p_value
        test_stats[j] = statistic

        # Calculate relative effect (pHat)
        # pHat = P(X < Y) + 0.5 * P(X = Y)
        n1, n2 = len(g1_vals), len(g2_vals)
        comparisons = np.subtract.outer(g1_vals, g2_vals)
        n_less = np.sum(comparisons < 0)
        n_equal = np.sum(comparisons == 0)
        pHat = (n_less + 0.5 * n_equal) / (n1 * n2)
        relative_effects[j] = pHat

        log2_fc[j] = _log2_fold_change(g1_vals, g2_vals, offset=log2_fc_offset)

    return DiffExprResult(
        feature_ids=assay.var[assay.feature_id_col].to_numpy(),
        p_values=p_values,
        p_values_adj=adjust_fdr(p_values, method="bh"),
        log2_fc=log2_fc,
        test_statistics=test_stats,
        effect_sizes=relative_effects,  # Relative effects instead of Cohen's d
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


def _log2_fold_change(
    group1_values: np.ndarray,
    group2_values: np.ndarray,
    offset: float = 1.0,
) -> float:
    """Calculate log2 fold change between two groups.

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


if __name__ == "__main__":
    import sys

    print("Running nonparametric module tests...")

    # Test 1: Wilcoxon effect size
    print("Test 1: Wilcoxon effect size calculation")
    x, y = np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    effect = _wilcoxon_effect_size(x, y)
    assert -1 <= effect <= 1
    print("  PASSED")

    # Test 2: Brunner-Munzel statistic
    print("Test 2: Brunner-Munzel statistic")
    stat, p_val = _brunner_munzel_statistic(x, y)
    assert isinstance(stat, float)
    assert 0 <= p_val <= 1
    print("  PASSED")

    # Test 3: _isna_array function
    print("Test 3: _isna_array function")
    assert _isna_array(np.array([1.0, np.nan, 3.0]))[1]
    assert not _isna_array(np.array([1.0, 2.0, 3.0]))[0]
    print("  PASSED")

    # Test 4: _log2_fold_change
    print("Test 4: Log2 fold change calculation")
    g1, g2 = np.array([10.0, 12.0, 11.0]), np.array([5.0, 6.0, 5.5])
    fc = _log2_fold_change(g1, g2)
    assert fc > 0  # group1 has higher values
    print("  PASSED")

    # Test 5: _extract_group_indices
    print("Test 5: Extract group indices")
    obs = pl.DataFrame({"_index": ["s1", "s2", "s3", "s4"], "group": ["A", "A", "B", "B"]})
    idx1, idx2 = _extract_group_indices(obs, "group", "A", "B")
    assert len(idx1) == 2 and len(idx2) == 2
    print("  PASSED")

    # Test 6: DiffExprResult compatibility
    print("Test 6: DiffExprResult compatibility")
    from scptensor.diff_expr.core import DiffExprResult

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
