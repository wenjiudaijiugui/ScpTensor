import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Any
from .style import setup_style

def violin(
    data: List[Any],
    labels: List[str],
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    ylabel: Optional[str] = None
) -> plt.Axes:
    """
    Primitive Violin Plot.
    """
    setup_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        
    parts = ax.violinplot(data, showmeans=False, showmedians=True)
    
    for pc in parts['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
        
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    if title: ax.set_title(title)
    if ylabel: ax.set_ylabel(ylabel)
    
    return ax
