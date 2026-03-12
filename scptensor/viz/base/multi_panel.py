"""Multi-panel layout manager for combined plots."""

import math
import warnings
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
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
        self._auto_grid = grid is None
        self._figure: Figure | None = None
        self._axes: list[Axes] = []
        self._panel_specs: dict[int, tuple[Callable[[Axes], Any], dict[str, Any]]] = {}
        self._panel_mappables: dict[int, Any] = {}
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
        if n_panels == 0:
            return (1, 1)

        n_cols = math.ceil(math.sqrt(n_panels))
        n_rows = math.ceil(n_panels / n_cols)
        return (n_rows, n_cols)

    def _create_figure(self, target_grid: tuple[int, int]) -> None:
        """Create (or recreate) figure with the specified grid."""
        if self._figure is not None:
            plt.close(self._figure)

        self.grid = target_grid
        figure, axes_array = plt.subplots(
            target_grid[0], target_grid[1], figsize=self.figsize, squeeze=False
        )
        self._figure = figure
        self._axes = axes_array.flatten().tolist()

    def _ensure_figure(self, required_panels: int = 1) -> None:
        """Ensure a figure exists and has enough axes for required panels."""
        target_grid = self.grid
        if self._auto_grid:
            target_grid = self._compute_grid(required_panels)

        if target_grid is None:
            target_grid = (1, 1)

        needs_recreate = self._figure is None or (
            self._auto_grid and self.grid is not None and self.grid != target_grid
        )
        if not needs_recreate:
            return

        self._create_figure(target_grid)
        self._rerender_panels()

    def _rerender_panels(self) -> None:
        """Render all registered panel functions to current axes."""
        self._panel_mappables.clear()
        for idx, (plot_func, kwargs) in sorted(self._panel_specs.items()):
            self._render_panel(idx, plot_func, kwargs)

    def _required_capacity(self) -> int:
        """Return minimum axis capacity required by current panel registrations."""
        if not self._panel_specs:
            return 1
        return max(self._panel_specs) + 1

    def _position_to_index(self, position: int | tuple[int, int]) -> int:
        """Convert user-specified panel position into a flattened axis index."""
        if isinstance(position, tuple):
            if self._auto_grid:
                raise ValueError("Tuple positions require explicit grid in PanelLayout")
            if self.grid is None:
                raise ValueError("Grid must be specified when using tuple position")
            row, col = position
            if row < 0 or col < 0:
                raise IndexError("Panel row/col indices must be non-negative")
            return row * self.grid[1] + col
        if position < 0:
            raise IndexError("Panel index must be non-negative")
        return position

    def _render_panel(
        self, idx: int, plot_func: Callable[[Axes], Any], kwargs: dict[str, Any]
    ) -> Axes:
        """Execute panel plotting function and track mappable outputs."""
        if idx >= len(self._axes):
            raise IndexError(f"Panel position {idx} out of range for grid {self.grid}")

        ax = self._axes[idx]
        ax.clear()
        result = plot_func(ax, **kwargs)
        if result is not None and hasattr(result, "get_cmap"):
            self._panel_mappables[idx] = result
        else:
            self._panel_mappables.pop(idx, None)
        return ax

    def add_panel(
        self,
        position: int | tuple[int, int],
        plot_func: Callable[[Axes], Any],
        **kwargs: Any,
    ) -> Axes:
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
        idx = self._position_to_index(position)

        required_panels = max(
            len(self._panel_specs) + (0 if idx in self._panel_specs else 1), idx + 1
        )
        self._ensure_figure(required_panels)

        self._panel_specs[idx] = (plot_func, dict(kwargs))
        return self._render_panel(idx, plot_func, dict(kwargs))

    def add_legend(
        self,
        position: str = "right",
        labels: list[str] | None = None,
        **kwargs: Any,
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
        self._legend_elements.append({"position": position, "labels": labels, "kwargs": kwargs})

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
        if not self._panel_mappables:
            return

        if position not in {"right", "left", "top", "bottom"}:
            raise ValueError("position must be one of: right, left, top, bottom")

        self._ensure_figure(self._required_capacity())

        first_idx = min(self._panel_mappables)
        mappable = self._panel_mappables[first_idx]
        axes = [self._axes[idx] for idx in sorted(self._panel_specs)]
        if self._figure is not None:
            try:
                self._figure.colorbar(mappable, ax=axes, label=label, location=position)
            except TypeError:
                # Fallback for older Matplotlib versions without `location`.
                self._figure.colorbar(mappable, ax=axes, label=label)

    def finalize(self, tight: bool = True) -> Figure | None:
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
        self._ensure_figure(self._required_capacity())

        if tight and self._figure is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="This figure includes Axes that are not compatible with tight_layout.*",
                    category=UserWarning,
                )
                self._figure.tight_layout()

        # Handle legends
        for legend_spec in self._legend_elements:
            position = legend_spec["position"]
            legend_kwargs = dict(legend_spec["kwargs"])
            if position in {"right", "outside"}:
                legend_kwargs.setdefault("loc", "center left")
                legend_kwargs.setdefault("bbox_to_anchor", (1.02, 0.5))
                legend_kwargs.setdefault("borderaxespad", 0.0)
            elif position == "bottom":
                legend_kwargs.setdefault("loc", "upper center")
                legend_kwargs.setdefault("bbox_to_anchor", (0.5, -0.1))
            elif position != "best":
                raise ValueError("legend position must be one of: right, bottom, outside, best")

            # Add legend to first axes with artists
            for ax in self._axes:
                if ax.legend_ is not None or ax.lines or ax.collections:
                    ax.legend(
                        labels=legend_spec["labels"],
                        **legend_kwargs,
                    )
                    break

        return self._figure

    @property
    def figure(self) -> Figure | None:
        """Get the figure, creating it if necessary.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object.
        """
        self._ensure_figure(self._required_capacity())
        return self._figure

    @property
    def axes(self) -> list[Axes]:
        """Get all axes.

        Returns
        -------
        list of matplotlib.axes.Axes
            List of all axes objects in the grid.
        """
        if not self._axes:
            self._ensure_figure(self._required_capacity())
        return self._axes
