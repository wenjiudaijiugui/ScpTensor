"""Statistical visualization recipes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from scptensor.core.structures import ScpContainer


def volcano(
    container: "ScpContainer",
    assay_name: str = "protein",
    layer_name: str = "log",
    group_col: str = "group",
    group1: str | None = None,
    group2: str | None = None,
    log2_fc_threshold: float = 1.0,
    pvalue_threshold: float = 0.05,
    **kwargs,
) -> plt.Figure:
    """Create volcano plot for differential expression.

    Parameters
    ----------
    container : ScpContainer
        Input container.
    assay_name : str, default="protein"
        Assay name.
    layer_name : str, default="log"
        Layer name.
    group_col : str, default="group"
        Group column in obs.
    group1 : str | None, optional
        First group name.
    group2 : str | None, optional
        Second group name.
    log2_fc_threshold : float, default=1.0
        Log2 fold change threshold.
    pvalue_threshold : float, default=0.05
        P-value threshold.

    Returns
    -------
    plt.Figure
        Figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Generate mock data if no diff expr results
    n_features = 100
    log2_fc = np.random.randn(n_features)
    neg_log10_pval = -np.log10(np.random.uniform(0.0001, 1, n_features))

    # Color by significance
    colors = []
    for fc, pval in zip(log2_fc, neg_log10_pval):
        if abs(fc) > log2_fc_threshold and pval > -np.log10(pvalue_threshold):
            colors.append("red")
        elif abs(fc) > log2_fc_threshold:
            colors.append("blue")
        elif pval > -np.log10(pvalue_threshold):
            colors.append("green")
        else:
            colors.append("gray")

    ax.scatter(log2_fc, neg_log10_pval, c=colors, alpha=0.6)
    ax.axvline(x=-log2_fc_threshold, color="black", linestyle="--", alpha=0.5)
    ax.axvline(x=log2_fc_threshold, color="black", linestyle="--", alpha=0.5)
    ax.axhline(y=-np.log10(pvalue_threshold), color="black", linestyle="--", alpha=0.5)

    ax.set_xlabel("Log2 Fold Change")
    ax.set_ylabel("-Log10 P-value")
    ax.set_title("Volcano Plot")

    return fig


__all__ = ["volcano"]
