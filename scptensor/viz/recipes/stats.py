import numpy as np
import matplotlib.pyplot as plt
from scptensor.core.structures import ScpContainer
from scptensor.viz.base import scatter

def volcano(
    container: ScpContainer, 
    comparison: str,
    assay_name: str = 'protein',
    logfc_col: str = None,
    pval_col: str = None,
    fc_threshold: float = 1.0,
    pval_threshold: float = 0.05
):
    """
    Volcano Plot for Differential Expression.
    
    Args:
        container: ScpContainer.
        comparison: Name of the comparison (used to find columns in var if defaults not set).
                    e.g. "Treat_vs_Ctrl" -> expects "Treat_vs_Ctrl_logFC" and "Treat_vs_Ctrl_pval".
        assay_name: Assay containing DE results in var.
        logfc_col: Explicit column name for logFC.
        pval_col: Explicit column name for P-value.
        fc_threshold: Fold Change threshold (log2 scale).
        pval_threshold: P-value threshold.
    """
    if assay_name not in container.assays:
        raise ValueError(f"Assay '{assay_name}' not found.")
        
    var = container.assays[assay_name].var
    
    # Determine columns
    if logfc_col is None:
        logfc_col = f"{comparison}_logFC"
    if pval_col is None:
        pval_col = f"{comparison}_pval"
        
    if logfc_col not in var.columns:
        raise ValueError(f"Column '{logfc_col}' not found in {assay_name}.var")
    if pval_col not in var.columns:
        raise ValueError(f"Column '{pval_col}' not found in {assay_name}.var")
        
    logfc = var[logfc_col].to_numpy()
    pval = var[pval_col].to_numpy()
    
    # Calculate -log10(pval)
    # Handle zeros to avoid inf
    pval_safe = pval.copy()
    pval_safe[pval_safe < 1e-300] = 1e-300
    neg_log_pval = -np.log10(pval_safe)
    
    # Classify points
    # 0: NS, 1: Sig (FC only), 2: Sig (Pval only), 3: Sig (Both) -> simplified coloring
    # Usually: Up (Red), Down (Blue), NS (Grey)
    
    colors = np.array(['grey'] * len(logfc), dtype=object)
    
    is_sig_pval = pval < pval_threshold
    is_up = (logfc > fc_threshold) & is_sig_pval
    is_down = (logfc < -fc_threshold) & is_sig_pval
    
    colors[is_up] = 'red'
    colors[is_down] = 'blue'
    
    # Map colors to numbers/RGBA for scatter primitive? 
    # Primitive scatter handles 'c' array. If strings, we need to pass them carefully or use mask.
    # Our primitive 'scatter' logic for 'c' with mask:
    # "if c is not None ... c_valid = c[valid_mask]"
    # Matplotlib scatter accepts array of color strings.
    
    # Prepare X for scatter: (logFC, -log10P)
    X_plot = np.column_stack((logfc, neg_log_pval))
    
    # All valid (no missingness in DE results usually, or we filter NaNs)
    # If logFC or pval is NaN, mask it
    mask = np.isnan(logfc) | np.isnan(pval)
    # Convert bool mask to int (0=Valid, 1=Invalid)
    m = mask.astype(int)
    
    ax = scatter(
        X=X_plot,
        c=colors,
        m=m,
        mask_style='subtle', # NaNs will be greyed out/transparent
        title=f"Volcano Plot: {comparison}",
        xlabel="log2 Fold Change",
        ylabel="-log10 P-value"
    )
    
    # Add threshold lines
    ax.axhline(-np.log10(pval_threshold), linestyle='--', color='black', linewidth=0.5, alpha=0.5)
    ax.axvline(fc_threshold, linestyle='--', color='black', linewidth=0.5, alpha=0.5)
    ax.axvline(-fc_threshold, linestyle='--', color='black', linewidth=0.5, alpha=0.5)
    
    return ax
