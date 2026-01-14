"""Missing value visualization handling."""

import matplotlib.pyplot as plt
import numpy as np


class MissingValueHandler:
    """Unified missing value visualization handling.

    Provides utilities for visualizing missing values in scatter plots
    with different colors for different missing value types.
    """

    # Missing value type colors
    MISSING_COLORS = {
        1: "#d3d3d3",  # MBR - Light gray
        2: "#add8e6",  # LOD - Light blue
        3: "#ffcccb",  # FILTERED - Light red
        "missing": "#d3d3d3",  # Generic missing
    }

    @staticmethod
    def separate_mask(
        X: np.ndarray,  # noqa: N803 - X is standard notation for data matrix
        M: np.ndarray,  # noqa: N803 - M is standard notation for mask matrix
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Separate valid and missing values.

        Parameters
        ----------
        X : ndarray
            Data matrix
        M : ndarray
            Mask matrix

        Returns
        -------
        x_valid : ndarray
            Valid values (where M == 0)
        x_missing : ndarray
            Missing values (where M > 0)
        m_types : ndarray
            Missing value types
        """
        valid_mask = M == 0
        missing_mask = M > 0

        x_valid = X[valid_mask]  # noqa: N806
        x_missing = X[missing_mask]  # noqa: N806
        m_types = M[missing_mask]  # noqa: N806

        return x_valid, x_missing, m_types

    @staticmethod
    def create_overlay_scatter(
        x: np.ndarray,
        y: np.ndarray,
        M: np.ndarray,  # noqa: N803 - M is standard notation for mask matrix
        ax: plt.Axes,
        size: float = 5.0,
        alpha: float = 0.8,
        **scatter_kwargs,
    ) -> dict:
        """Create layered scatter plot with missing values.

        Creates a scatter plot where valid values (M==0) are plotted first,
        followed by overlay of each missing value type with distinct colors.

        Parameters
        ----------
        x, y : ndarray
            Coordinates
        M : ndarray
            Mask values (0=valid, >0=missing)
        ax : Axes
            Matplotlib axes
        size : float, default=5.0
            Point size
        alpha : float, default=0.8
            Transparency for valid points (missing points use alpha*0.7)
        **scatter_kwargs
            Additional scatter arguments (e.g., color, cmap) for valid points

        Returns
        -------
        dict
            Dictionary with legend handles/labels as keys
        """
        valid_mask = M == 0
        handles = {}

        # Plot valid values first
        if valid_mask.sum() > 0:
            ax.scatter(
                x[valid_mask],
                y[valid_mask],
                s=size,
                alpha=alpha,
                label="Detected",
                **scatter_kwargs,
            )
            handles["Detected"] = True

        # Plot each missing type separately
        for m_type, color in MissingValueHandler.MISSING_COLORS.items():
            if isinstance(m_type, str):
                continue  # Skip generic color entry
            type_mask = m_type == M
            if type_mask.sum() > 0:
                ax.scatter(
                    x[type_mask],
                    y[type_mask],
                    s=size,
                    alpha=alpha * 0.7,
                    color=color,
                    label=f"Missing ({m_type})",
                    edgecolors="none",
                )
                handles[f"Missing ({m_type})"] = True

        return handles
