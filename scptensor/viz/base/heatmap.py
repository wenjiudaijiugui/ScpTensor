import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, Any, List
from .style import setup_style

def heatmap(
    X: np.ndarray,
    m: Optional[np.ndarray] = None,
    xticklabels: Optional[List[str]] = None,
    yticklabels: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
    cmap: str = 'viridis',
    **kwargs: Any
) -> plt.Axes:
    """
    Primitive Heatmap with Mask hatching.
    """
    setup_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Extract title from kwargs if present, as imshow doesn't support it
    title = kwargs.pop('title', None)
        
    # Draw main heatmap
    im = ax.imshow(X, aspect='auto', cmap=cmap, **kwargs)

    if title:
        ax.set_title(title)
    
    # Draw mask hatching
    if m is not None:
        # Create a masked array where m != 0
        # We overlay a hatch pattern
        # pcolorfast or pcolormesh is better for this
        # We need coordinates
        rows, cols = X.shape
        # Create meshgrid
        x = np.arange(cols + 1)
        y = np.arange(rows + 1)
        
        masked_data = np.ma.masked_where(m == 0, m)
        # We use pcolormesh to show hatches only where mask is True (m!=0)
        # Use a transparent facecolor, and black hatch
        ax.pcolormesh(x, y, masked_data, hatch='////', alpha=0.0, shading='flat')
        
    if xticklabels is not None:
        ax.set_xticks(np.arange(len(xticklabels)))
        ax.set_xticklabels(xticklabels, rotation=90)
        
    if yticklabels is not None:
        ax.set_yticks(np.arange(len(yticklabels)))
        ax.set_yticklabels(yticklabels)
        
    plt.colorbar(im, ax=ax)
    
    return ax
