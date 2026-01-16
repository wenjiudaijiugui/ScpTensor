import matplotlib.pyplot as plt
import numpy as np

from scptensor.core.structures import ScpContainer
from scptensor.viz.base.scatter import scatter


def volcano(
    container: ScpContainer,
    comparison: str,
    assay_name: str = "protein",
    logfc_col: str | None = None,
    pval_col: str | None = None,
    fc_threshold: float = 1.0,
    pval_threshold: float = 0.05,
) -> plt.Axes:
    """Create a volcano plot for differential expression results.

    Displays log2 fold change vs -log10 p-value, with significant
    up/down-regulated points highlighted in red/blue.

    Parameters
    ----------
    container : ScpContainer
        Input data container with DE results in ``var``.
    comparison : str
        Comparison name. Used to construct default column names
        (e.g., "Treat_vs_Ctrl" -> "{comparison}_logFC").
    assay_name : str, default "protein"
        Assay containing DE results.
    logfc_col : str | None
        Explicit logFC column name. If None, uses "{comparison}_logFC".
    pval_col : str | None
        Explicit p-value column name. If None, uses "{comparison}_pval".
    fc_threshold : float, default 1.0
        Log2 fold change threshold for significance.
    pval_threshold : float, default 0.05
        P-value threshold for significance.

    Returns
    -------
    plt.Axes
        The axes containing the plot.

    Raises
    ------
    ValueError
        If assay or required columns are not found.

    """
    assay = container.assays.get(assay_name)
    if assay is None:
        raise ValueError(f"Assay '{assay_name}' not found.")

    var = assay.var

    # Resolve column names
    logfc_col = logfc_col or f"{comparison}_logFC"
    pval_col = pval_col or f"{comparison}_pval"

    missing = [c for c in (logfc_col, pval_col) if c not in var.columns]
    if missing:
        raise ValueError(f"Columns not found in {assay_name}.var: {missing}")

    logfc = var[logfc_col].to_numpy()
    pval = var[pval_col].to_numpy()

    # Avoid log(0) in -log10(pval)
    pval_clipped = np.clip(pval, 1e-300, None)
    neg_log_pval = -np.log10(pval_clipped)

    # Color coding: grey=NS, red=up, blue=down
    colors = np.full(len(logfc), "grey", dtype=object)
    is_sig = pval < pval_threshold
    colors[is_sig & (logfc > fc_threshold)] = "red"
    colors[is_sig & (logfc < -fc_threshold)] = "blue"

    X = np.column_stack((logfc, neg_log_pval))
    m = (np.isnan(logfc) | np.isnan(pval)).astype(np.int8)

    ax = scatter(
        X=X,
        c=colors,
        m=m,
        mask_style="subtle",
        title=f"Volcano Plot: {comparison}",
        xlabel="log2 Fold Change",
        ylabel="-log10 P-value",
    )

    # Threshold lines
    line_kwargs = {"linestyle": "--", "color": "black", "linewidth": 0.5, "alpha": 0.5}
    ax.axhline(-np.log10(pval_threshold), **line_kwargs)  # type: ignore[arg-type]
    ax.axvline(fc_threshold, **line_kwargs)  # type: ignore[arg-type]
    ax.axvline(-fc_threshold, **line_kwargs)  # type: ignore[arg-type]

    return ax
