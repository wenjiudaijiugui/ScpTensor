import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, Any
from .style import setup_style

def scatter(
    X: np.ndarray,
    c: Optional[np.ndarray] = None,
    m: Optional[np.ndarray] = None,
    mask_style: str = 'subtle',
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    **kwargs: Any
) -> plt.Axes:
    """
    Primitive Scatter Plot with Mask handling.
    
    Args:
        X: Coordinates (N, 2).
        c: Color values (N,).
        m: Mask values (N,). 0=Valid.
        mask_style: 'subtle' (alpha coding) or 'explicit' (shape coding).
        ax: Matplotlib axes.
        **kwargs: Passed to scatter.
    """
    setup_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    
    if m is None:
        m = np.zeros(X.shape[0], dtype=int)
        
    valid_mask = (m == 0)
    invalid_mask = ~valid_mask
    
    # Base kwargs
    scatter_kwargs = {'s': 20, 'edgecolor': 'none'}
    scatter_kwargs.update(kwargs)
    
    if mask_style == 'subtle':
        # Valid data: alpha=1.0, zorder=10
        if np.any(valid_mask):
            c_valid = c[valid_mask] if c is not None else None
            ax.scatter(X[valid_mask, 0], X[valid_mask, 1], c=c_valid, alpha=1.0, zorder=10, label='Measured', **scatter_kwargs)
            
        # Invalid data: alpha=0.3, zorder=0
        if np.any(invalid_mask):
            c_invalid = c[invalid_mask] if c is not None else 'gray'
            # If c is continuous, c_invalid might be numeric, which is fine.
            # If c is categorical/colors, it works too.
            # But usually missing values might imply missing color source? 
            # If color source is Metadata (e.g. Batch), it exists for invalid data.
            # If color source is Expression, it might be imputed or missing.
            ax.scatter(X[invalid_mask, 0], X[invalid_mask, 1], c=c_invalid, alpha=0.3, zorder=0, label='Imputed/Missing', **scatter_kwargs)
            
    elif mask_style == 'explicit':
        # Valid data: marker='o'
        if np.any(valid_mask):
            c_valid = c[valid_mask] if c is not None else None
            ax.scatter(X[valid_mask, 0], X[valid_mask, 1], c=c_valid, marker='o', label='Measured', **scatter_kwargs)
            
        # Invalid data: marker='x'
        if np.any(invalid_mask):
            c_invalid = c[invalid_mask] if c is not None else 'gray'
            ax.scatter(X[invalid_mask, 0], X[invalid_mask, 1], c=c_invalid, marker='x', label='Imputed/Missing', **scatter_kwargs)
            
    else:
        # Default fallback
        ax.scatter(X[:, 0], X[:, 1], c=c, **scatter_kwargs)
        
    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    
    return ax
