"""Base classes for benchmark display modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

__all__ = ["DisplayBase", "ComparisonDisplay"]


class DisplayBase(ABC):
    """Abstract base class for benchmark display handlers.

    Parameters
    ----------
    output_dir : str | Path, optional
        Directory path for display outputs. Default is "benchmark_results".
    """

    def __init__(self, output_dir: str | Path = "benchmark_results") -> None:
        self.output_dir: Path = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def render(self) -> Path:
        """Render the display output.

        Returns
        -------
        Path
            Path to the rendered output file.
        """


class ComparisonDisplay(DisplayBase):
    """Abstract base for comparison display handlers.

    Extends DisplayBase with methods for rendering comparisons between
    ScpTensor and competitor preprocessing methods.
    """

    @abstractmethod
    def render_comparison(self, result: dict[str, Any]) -> Path:
        """Render a single comparison result.

        Parameters
        ----------
        result : dict[str, Any]
            Dictionary containing comparison result data.

        Returns
        -------
        Path
            Path to the rendered comparison output file.
        """

    @abstractmethod
    def render_summary(self, results: list[dict[str, Any]]) -> Path:
        """Render a summary of multiple comparison results.

        Parameters
        ----------
        results : list[dict[str, Any]]
            List of comparison result dictionaries.

        Returns
        -------
        Path
            Path to the rendered summary output file.
        """
