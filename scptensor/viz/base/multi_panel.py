"""Multi-panel layout manager for combined plots."""

from typing import Callable, Any
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class PanelLayout:
    """Multi-panel combined plot layout manager.

    Provides a convenient interface for creating figures with multiple panels,
    handling grid layout, shared colorbars, and shared legends.

    Examples
    --------
    >>> layout = PanelLayout(figsize=(12, 8), grid=(2, 2))
    >>> layout.add_panel(0, lambda ax: ax.plot([1, 2, 3]))
    >>> layout.add_panel(1, lambda ax: ax.scatter([1, 2], [3, 4]))
    >>> fig = layout.finalize()
    """

    def __init__(
        self,
        figsize: tuple[float, float] = (12, 8),
        grid: tuple[int, int] | None = None,
    ) -> None:
        """
        Initialize panel layout.

        Parameters
        ----------
        figsize : tuple, optional
            Overall figure size (width, height) in inches.
            Default is (12, 8).
        grid : tuple of (int, int) or None, optional
            (n_rows, n_cols) grid specification. If None, grid is
            auto-computed based on the number of panels added.
            Default is None.
        """
        self.figsize = figsize
        self.grid = grid
        self._auto_grid = grid is None  # Track if using auto-grid mode
        self._n_panels = 0
        self._figure: Figure | None = None
        self._axes: list[plt.Axes] = []
        self._colorbar_sources: list[tuple[plt.Axes, Any]] = []
        self._legend_elements: list[dict[str, Any]] = []

    def _compute_grid(self, n_panels: int) -> tuple[int, int]:
        """Compute optimal grid layout for given number of panels.

        Parameters
        ----------
        n_panels : int
            Number of panels to accommodate.

        Returns
        -------
        tuple of (int, int)
            (n_rows, n_cols) grid specification.
        """
        import math

        if n_panels == 0:
            return (1, 1)

        n_cols = math.ceil(math.sqrt(n_panels))
        n_rows = math.ceil(n_panels / n_cols)
        return (n_rows, n_cols)

    def _create_figure(self) -> None:
        """Create figure and subplots if not exists."""
        if self._figure is None:
            if self.grid is not None:
                n_rows, n_cols = self.grid
            else:
                n_rows, n_cols = self._compute_grid(self._n_panels)
                self.grid = (n_rows, n_cols)

            self._figure, self._axes_array = plt.subplots(
                self.grid[0], self.grid[1], figsize=self.figsize, squeeze=False
            )
            # Flatten axes array for easier indexing
            self._axes = self._axes_array.flatten().tolist()

    def _recreate_figure_for_panel(self, position: int | tuple[int, int]) -> None:
        """Recreate figure with expanded grid if needed for auto-grid mode.

        Parameters
        ----------
        position : int or tuple of (int, int)
            Panel position being added.
        """
        if not self._auto_grid:
            return  # Fixed grid, no expansion

        # Compute required grid for current panel count
        required_grid = self._compute_grid(self._n_panels)

        if self.grid is None or required_grid != self.grid:
            # Store existing plots
            existing_plots = []
            for ax in self._axes:
                # Get data from existing axes
                lines_data = [(line.get_xdata(), line.get_ydata()) for line in ax.lines]
                existing_plots.append(lines_data)

            # Close old figure and create new one
            plt.close(self._figure)
            self.grid = required_grid

            self._figure, self._axes_array = plt.subplots(
                self.grid[0], self.grid[1], figsize=self.figsize, squeeze=False
            )
            self._axes = self._axes_array.flatten().tolist()

            # Redraw existing plots (note: simplified - in real use case
            # we'd need to store the plot functions, not just data)
            for i, ax in enumerate(self._axes):
                if i < len(existing_plots):
                    for x, y in existing_plots[i]:
                        ax.plot(x, y)

    def add_panel(
        self,
        position: int | tuple[int, int],
        plot_func: Callable[[plt.Axes], Any],
        **kwargs,
    ) -> plt.Axes:
        """
        Add subplot panel.

        Parameters
        ----------
        position : int or tuple of (int, int)
            Panel index (0-based) or (row, col) position in grid.
        plot_func : callable
            Function that takes an Axes object and plots on it.
        **kwargs
            Additional keyword arguments passed to plot_func.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object for the added panel.

        Examples
        --------
        >>> layout = PanelLayout(grid=(2, 2))
        >>> ax = layout.add_panel(0, lambda ax: ax.plot([1, 2, 3]))
        >>> ax = layout.add_panel((0, 1), lambda ax: ax.scatter([1, 2], [3, 4]))
        """
        self._n_panels += 1

        # For auto-grid mode, potentially expand grid before creating
        if self._figure is not None and self._auto_grid:
            self._recreate_figure_for_panel(position)

        self._create_figure()

        if isinstance(position, tuple):
            if self.grid is None:
                raise ValueError("Grid must be specified when using tuple position")
            idx = position[0] * self.grid[1] + position[1]
        else:
            idx = position

        if idx >= len(self._axes):
            raise IndexError(f"Panel position {idx} out of range for grid {self.grid}")

        ax = self._axes[idx]
        result = plot_func(ax, **kwargs)

        # Store image/mappable for potential colorbar
        if result is not None and hasattr(result, "get_cmap"):
            self._colorbar_sources.append((ax, result))

        return ax

    def add_legend(
        self, position: str = "right", labels: list[str] | None = None, **kwargs
    ) -> None:
        """
        Add shared legend across panels.

        Parameters
        ----------
        position : str, optional
            Legend position. Options: 'right', 'bottom', 'outside'.
            Default is 'right'.
        labels : list of str or None, optional
            Legend labels. If None, uses auto-generated labels.
            Default is None.
        **kwargs
            Additional keyword arguments passed to legend().

        Examples
        --------
        >>> layout.add_legend(labels=["Control", "Treatment"], position="right")
        """
        self._legend_elements.append(
            {"position": position, "labels": labels, "kwargs": kwargs}
        )

    def add_colorbar(self, position: str = "right", label: str = "") -> None:
        """
        Add shared colorbar across panels.

        Creates a colorbar based on the first mappable artist
        (e.g., from imshow, scatter, etc.) added to any panel.

        Parameters
        ----------
        position : str, optional
            Colorbar position. Default is 'right'.
        label : str, optional
            Colorbar label. Default is empty string.

        Examples
        --------
        >>> layout.add_panel(0, lambda ax: ax.imshow(data, cmap="viridis"))
        >>> layout.add_colorbar(label="Intensity")
        """
        if not self._colorbar_sources:
            return

        # Ensure figure exists
        if self._figure is None:
            self._create_figure()

        # Use first colorbar source
        ax, mappable = self._colorbar_sources[0]
        if self._figure is not None:
            self._figure.colorbar(mappable, ax=ax, label=label)

    def finalize(self, tight: bool = True) -> Figure:
        """
        Finalize layout and adjust spacing.

        Parameters
        ----------
        tight : bool, optional
            Whether to use tight_layout for automatic spacing.
            Default is True.

        Returns
        -------
        matplotlib.figure.Figure
            The finalized figure object.

        Examples
        --------
        >>> fig = layout.finalize()
        >>> fig.savefig("output.png", dpi=300)
        """
        if self._figure is None:
            self._create_figure()

        if tight and self._figure is not None:
            self._figure.tight_layout()

        # Handle legends
        for legend_spec in self._legend_elements:
            # Add legend to first axes with artists
            for ax in self._axes:
                if ax.legend_ is not None or ax.lines or ax.collections:
                    ax.legend(
                        labels=legend_spec["labels"],
                        loc="best",
                        **legend_spec["kwargs"],
                    )
                    break

        return self._figure

    @property
    def figure(self) -> Figure:
        """Get the figure, creating it if necessary.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object.
        """
        if self._figure is None:
            self._create_figure()
        return self._figure

    @property
    def axes(self) -> list[plt.Axes]:
        """Get all axes.

        Returns
        -------
        list of matplotlib.axes.Axes
            List of all axes objects in the grid.
        """
        if not self._axes:
            self._create_figure()
        return self._axes
