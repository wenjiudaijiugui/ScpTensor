"""Differential expression analysis for single-cell proteomics.

This module provides statistical tests for identifying features (proteins/peptides)
that differ significantly between groups.

Key Functions:
    diff_expr_ttest: Two-group comparison using t-test (Welch's or Student's)
    diff_expr_paired_ttest: Paired sample comparison for matched samples
    diff_expr_mannwhitney: Non-parametric two-group comparison
    diff_expr_anova: Multi-group comparison using ANOVA
    diff_expr_kruskal: Non-parametric multi-group comparison
    diff_expr_permutation_test: Non-parametric permutation-based test
    check_homoscedasticity: Test for equality of variances
    adjust_fdr: Multiple testing correction (FDR, Bonferroni, Holm, Hommel)

Result Structures:
    DiffExprResult: Container for test results with p-values, fold changes,
                    effect sizes, and group statistics
"""

from .core import (
    DiffExprResult,
    adjust_fdr,
    check_homoscedasticity,
    diff_expr_anova,
    diff_expr_kruskal,
    diff_expr_mannwhitney,
    diff_expr_paired_ttest,
    diff_expr_permutation_test,
    diff_expr_ttest,
)

__all__ = [
    "DiffExprResult",
    "adjust_fdr",
    "diff_expr_ttest",
    "diff_expr_paired_ttest",
    "diff_expr_mannwhitney",
    "diff_expr_anova",
    "diff_expr_kruskal",
    "diff_expr_permutation_test",
    "check_homoscedasticity",
]
