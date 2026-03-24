"""Missing value visualization handling."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


class MissingValueHandler:
    """Unified missing value visualization handling.

    Provides utilities for visualizing missing values in scatter plots
    with different colors for different missing value types.
    """

    # Mask code colors (0 is valid and not rendered as missing overlay)
    MISSING_COLORS = {
        1: "#d3d3d3",  # MBR - Light gray
        2: "#add8e6",  # LOD - Light blue
        3: "#ffcccb",  # FILTERED - Light red
        4: "#f4a261",  # OUTLIER - Orange
        5: "#9467bd",  # IMPUTED - Purple
        6: "#8c8c8c",  # UNCERTAIN - Dark gray
        "missing": "#d3d3d3",  # Generic missing
    }

    @staticmethod
    def separate_mask(
        X: np.ndarray,
        M: np.ndarray,
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

        x_valid = X[valid_mask]
        x_missing = X[missing_mask]
        m_types = M[missing_mask]

        return x_valid, x_missing, m_types

    @staticmethod
    def get_present_mask_codes(M: np.ndarray) -> list[int]:
        """Return sorted non-zero mask codes present in input mask."""
        mask_array = np.asarray(M).ravel()
        if mask_array.size == 0:
            return []
        return [int(code) for code in np.unique(mask_array[mask_array > 0])]

    @classmethod
    def get_color_for_mask_code(cls, code: int) -> str:
        """Return color for a mask code, falling back to generic missing color."""
        return cls.MISSING_COLORS.get(int(code), cls.MISSING_COLORS["missing"])

    @staticmethod
    def _validate_overlay_inputs(
        x: np.ndarray,
        y: np.ndarray,
        M: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Validate and flatten overlay inputs to 1D arrays."""
        x_arr = np.asarray(x).ravel()
        y_arr = np.asarray(y).ravel()
        m_arr = np.asarray(M).ravel()

        if x_arr.shape != y_arr.shape or x_arr.shape != m_arr.shape:
            raise ValueError(
                "x, y, and M must have the same number of elements for overlay scatter",
            )
        return x_arr, y_arr, m_arr

    @staticmethod
    def create_overlay_scatter(
        x: np.ndarray,
        y: np.ndarray,
        M: np.ndarray,
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
        x_arr, y_arr, m_arr = MissingValueHandler._validate_overlay_inputs(x, y, M)
        mask_codes = MissingValueHandler.get_present_mask_codes(m_arr)

        valid_mask = m_arr == 0
        handles = {}

        # Plot valid values first
        if valid_mask.sum() > 0:
            ax.scatter(
                x_arr[valid_mask],
                y_arr[valid_mask],
                s=size,
                alpha=alpha,
                label="Detected",
                **scatter_kwargs,
            )
            handles["Detected"] = True

        # Plot each non-zero mask code separately
        for m_type in mask_codes:
            type_mask = m_arr == m_type
            if type_mask.sum() > 0:
                ax.scatter(
                    x_arr[type_mask],
                    y_arr[type_mask],
                    s=size,
                    alpha=alpha * 0.7,
                    color=MissingValueHandler.get_color_for_mask_code(m_type),
                    label=f"Missing ({m_type})",
                    edgecolors="none",
                )
                handles[f"Missing ({m_type})"] = True

        return handles
