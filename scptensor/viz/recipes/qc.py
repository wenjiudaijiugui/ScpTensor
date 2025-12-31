import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Optional, List, Any
from scptensor.core.structures import ScpContainer
from scptensor.viz.base import violin, heatmap
from scptensor.viz.base.style import setup_style

def qc_completeness(
    container: ScpContainer, 
    assay_name: str = 'protein',
    layer: str = 'raw',
    group_by: str = 'batch',
    figsize: tuple[int, int] = (6, 4),
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Visualize data completeness (valid values count) per sample, grouped by metadata.
    
    Args:
        container: ScpContainer.
        assay_name: Target assay name.
        layer: Target layer name (usually 'raw').
        group_by: Column name in obs to group samples by.
        figsize: Figure size if creating new axes.
        ax: Optional matplotlib Axes object.

    Returns:
        plt.Axes: The plot axes.
    """
    setup_style()
    
    if assay_name not in container.assays:
        raise ValueError(f"Assay '{assay_name}' not found.")
        
    assay = container.assays[assay_name]
    if layer not in assay.layers:
        raise ValueError(f"Layer '{layer}' not found.")
        
    matrix = assay.layers[layer]
    
    # Calculate valid counts (Measured: M=0)
    M: np.ndarray = matrix.M
    valid_counts: np.ndarray = np.sum(M == 0, axis=1)
    
    # Get groups
    if group_by not in container.obs.columns:
        # Fallback to single group if key missing
        groups = np.array(['All'] * container.n_samples)
    else:
        groups = container.obs[group_by].to_numpy()
        
    unique_groups: np.ndarray = np.unique(groups)
    data_list: List[np.ndarray] = []
    labels_list: List[str] = []
    
    for g in unique_groups:
        mask = (groups == g)
        data_list.append(valid_counts[mask])
        labels_list.append(str(g))
        
    # Use base violin plot
    return violin(
        data=data_list, 
        labels=labels_list, 
        ax=ax,
        title=f"Data Completeness by {group_by}",
        ylabel="Number of Measured Features"
    )

def qc_matrix_spy(
    container: ScpContainer, 
    assay_name: str = 'protein',
    layer: str = 'raw',
    figsize: tuple[int, int] = (8, 6),
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Visualize missing value distribution (Spy Plot).
    
    Args:
        container: ScpContainer.
        assay_name: Target assay name.
        layer: Target layer name.
        figsize: Figure size if creating new axes.
        ax: Optional matplotlib Axes object.

    Returns:
        plt.Axes: The plot axes.
    """
    setup_style()

    if assay_name not in container.assays:
        raise ValueError(f"Assay '{assay_name}' not found.")
        
    assay = container.assays[assay_name]
    if layer not in assay.layers:
        raise ValueError(f"Layer '{layer}' not found.")
        
    matrix = assay.layers[layer]
    M: np.ndarray = matrix.M
    
    # We want to visualize M. 
    # M=0 is Measured (Valid), M!=0 is Missing/Other.
    # Create a binary view for spy plot: 0 (Measured), 1 (Missing)
    # Or use the actual M values if they have meaning (0=Measured, 1=MBR, etc.)
    # For aesthetic spy plot, let's stick to Binary or Categorical.
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        
    # Custom colormap for missing/measured
    # 0: Valid (Dark Blue or Grey), >0: Missing (Light or White or Yellow)
    # Let's use a discrete map.
    # We can map M to [0, 1] for simple spy
    spy_data = (M > 0).astype(int)
    
    cmap = mcolors.ListedColormap(['#0C5DA5', '#E5E5E5']) # SciencePlots Blue vs Grey
    bounds = [-0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    im = ax.imshow(spy_data, aspect='auto', interpolation='nearest', cmap=cmap, norm=norm)
    
    # Create legend/colorbar manually for clarity
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(['Measured', 'Missing'])
    
    ax.set_title(f"Matrix Spy Plot ({assay_name}/{layer})")
    ax.set_xlabel("Features")
    ax.set_ylabel("Samples")
    
    return ax
