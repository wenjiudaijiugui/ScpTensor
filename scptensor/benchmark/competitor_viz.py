"""Visualization module for competitor benchmark results.

Generates publication-quality plots comparing ScpTensor with competitor tools.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd

# =============================================================================
# Constants
# =============================================================================
_DPI = 300
_SCPTENSOR_COLOR = "#2E86AB"
_COMPETITOR_COLOR = "#A23B72"
_IMPLEMENTATION_LABELS = {
    "scptensor_time_ms": "ScpTensor",
    "competitor_time_ms": "Competitor",
    "scptensor_memory_mb": "ScpTensor",
    "competitor_memory_mb": "Competitor",
}
_COLUMN_MAPPING = {
    "scptensor_time": "scptensor_time_ms",
    "competitor_time": "competitor_time_ms",
    "scptensor_memory": "scptensor_memory_mb",
    "competitor_memory": "competitor_memory_mb",
    "accuracy_correlation": "accuracy",
    "competitor_name": "competitor",
}


# =============================================================================
# Style Setup
# =============================================================================


def _setup_style() -> None:
    """Setup matplotlib style for publication-quality plots."""
    import matplotlib.pyplot as plt

    try:
        plt.style.use(["science", "no-latex"])
    except OSError:
        plt.style.use("default")


# Setup style on import
_setup_style()


# =============================================================================
# Data Conversion Utilities
# =============================================================================


def _results_to_dataframe(results: dict[str, Any]) -> pd.DataFrame:
    """Convert benchmark results to pandas DataFrame.

    Parameters
    ----------
    results : dict[str, Any]
        Raw benchmark results dictionary.

    Returns
    -------
    pd.DataFrame
        Formatted DataFrame with all metrics.
    """
    rows = []

    for operation, operation_results in results.items():
        if operation.startswith("_"):
            continue

        for dataset_index, result in enumerate(operation_results):
            rows.append(
                {
                    "operation": result["operation"],
                    "dataset_index": dataset_index,
                    "scptensor_time_ms": result["scptensor_time"] * 1000,
                    "competitor_time_ms": result["competitor_time"] * 1000,
                    "speedup_factor": result["speedup_factor"],
                    "scptensor_memory_mb": result["scptensor_memory"],
                    "competitor_memory_mb": result["competitor_memory"],
                    "memory_ratio": result["memory_ratio"],
                    "accuracy": result["accuracy_correlation"],
                    "competitor": result["competitor_name"],
                }
            )

    return pd.DataFrame(rows)


def _melt_implementation_data(
    df: pd.DataFrame,
    id_vars: list[str],
    value_vars: list[str],
    value_name: str = "value",
) -> pd.DataFrame:
    """Melt DataFrame for side-by-side implementation comparison.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    id_vars : list[str]
        Identifier columns.
    value_vars : list[str]
        Value columns to melt.
    value_name : str, default "value"
        Name for melted value column.

    Returns
    -------
    pd.DataFrame
        Melted DataFrame with labeled implementations.
    """
    df_melt = df.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name="implementation",
        value_name=value_name,
    )
    df_melt["implementation"] = df_melt["implementation"].map(_IMPLEMENTATION_LABELS)
    return df_melt


def _save_or_show_figure(
    output_path: str | Path | None,
    message: str,
) -> None:
    """Save figure to file or display it.

    Parameters
    ----------
    output_path : str | Path | None
        Path to save figure, or None to display.
    message : str
        Message to print when saving.
    """
    import matplotlib.pyplot as plt

    if output_path:
        plt.savefig(output_path, dpi=_DPI, bbox_inches="tight")
        print(f"Saved {message} to: {output_path}")
    else:
        plt.show()
    plt.close()


# =============================================================================
# Visualization Class
# =============================================================================


class CompetitorResultVisualizer:
    """Visualizer for competitor benchmark results.

    Creates comparison plots for:
    - Runtime performance
    - Memory usage
    - Accuracy correlation
    - Speedup factors

    Examples
    --------
    >>> viz = CompetitorResultVisualizer("results.json")
    >>> viz.create_all_plots()
    """

    __slots__ = ("results_path", "results", "summaries", "_df_cache")

    def __init__(self, results_path: str | Path | None = None) -> None:
        """Initialize visualizer.

        Parameters
        ----------
        results_path : str | Path | None
            Path to benchmark results JSON file.
        """
        self.results_path = Path(results_path) if results_path else None
        self.results: dict[str, Any] = {}
        self.summaries: dict[str, Any] = {}
        self._df_cache: pd.DataFrame | None = None

        if self.results_path and self.results_path.exists():
            self.load_results(self.results_path)

    def load_results(self, path: str | Path) -> None:
        """Load benchmark results from JSON file.

        Parameters
        ----------
        path : str | Path
            Path to JSON file containing benchmark results.
        """
        path = Path(path)
        with open(path) as f:
            self.results = json.load(f)

        # Extract summary
        if "_summary" in self.results:
            self.summaries = self.results["_summary"]

        # Clear cache
        self._df_cache = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            Benchmark results as DataFrame.
        """
        if self._df_cache is None:
            self._df_cache = _results_to_dataframe(self.results)
        return self._df_cache

    def plot_speedup_comparison(
        self,
        output_path: str | Path | None = None,
        figsize: tuple[float, float] = (10, 6),
    ) -> None:
        """Plot speedup comparison across operations.

        Parameters
        ----------
        output_path : str | Path | None
            Path to save figure.
        figsize : tuple[float, float]
            Figure size (width, height).
        """
        import matplotlib.pyplot as plt

        df = self.to_dataframe()
        pivot_df = df.pivot(
            index="operation",
            columns="dataset_index",
            values="speedup_factor",
        )

        fig, ax = plt.subplots(figsize=figsize)
        pivot_df.plot(kind="bar", ax=ax, rot=45)

        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7, linewidth=1)

        ax.set_xlabel("Operation", fontweight="bold")
        ax.set_ylabel("Speedup Factor (ScpTensor / Competitor)", fontweight="bold")
        ax.set_title(
            "ScpTensor Performance vs Competitors", fontsize=14, fontweight="bold"
        )
        ax.legend(title="Dataset")
        ax.grid(axis="y", alpha=0.3)

        for container in ax.containers:
            ax.bar_label(container, fmt="%.2fx", fontsize=8, padding=2)

        plt.tight_layout()
        _save_or_show_figure(output_path, "speedup comparison")

    def plot_runtime_comparison(
        self,
        output_path: str | Path | None = None,
        figsize: tuple[float, float] = (12, 6),
    ) -> None:
        """Plot runtime comparison (side-by-side bars).

        Parameters
        ----------
        output_path : str | Path | None
            Path to save figure.
        figsize : tuple[float, float]
            Figure size (width, height).
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        df = self.to_dataframe()
        df_melt = _melt_implementation_data(
            df,
            id_vars=["operation", "dataset_index"],
            value_vars=["scptensor_time_ms", "competitor_time_ms"],
            value_name="runtime_ms",
        )

        fig, ax = plt.subplots(figsize=figsize)

        sns.barplot(
            data=df_melt,
            x="operation",
            y="runtime_ms",
            hue="implementation",
            ax=ax,
            palette=[_SCPTENSOR_COLOR, _COMPETITOR_COLOR],
        )

        ax.set_xlabel("Operation", fontweight="bold")
        ax.set_ylabel("Runtime (ms)", fontweight="bold")
        ax.set_title(
            "Runtime Comparison: ScpTensor vs Competitors",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(title="Implementation")
        ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()
        _save_or_show_figure(output_path, "runtime comparison")

    def plot_memory_comparison(
        self,
        output_path: str | Path | None = None,
        figsize: tuple[float, float] = (10, 6),
    ) -> None:
        """Plot memory usage comparison.

        Parameters
        ----------
        output_path : str | Path | None
            Path to save figure.
        figsize : tuple[float, float]
            Figure size (width, height).
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        df = self.to_dataframe()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot 1: Absolute memory usage
        df_melt = _melt_implementation_data(
            df,
            id_vars=["operation"],
            value_vars=["scptensor_memory_mb", "competitor_memory_mb"],
            value_name="memory_mb",
        )

        sns.barplot(
            data=df_melt,
            x="operation",
            y="memory_mb",
            hue="implementation",
            ax=ax1,
            palette=[_SCPTENSOR_COLOR, _COMPETITOR_COLOR],
        )
        ax1.set_xlabel("Operation", fontweight="bold")
        ax1.set_ylabel("Memory (MB)", fontweight="bold")
        ax1.set_title("Memory Usage", fontsize=12, fontweight="bold")
        ax1.legend(title=None)
        ax1.grid(axis="y", alpha=0.3)
        ax1.tick_params(axis="x", rotation=45)

        # Plot 2: Memory ratio
        for op in df["operation"].unique():
            op_data = df[df["operation"] == op]
            ax2.scatter(
                [op] * len(op_data),
                op_data["memory_ratio"],
                s=100,
                alpha=0.7,
                edgecolors="black",
            )

        ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7)
        ax2.set_xlabel("Operation", fontweight="bold")
        ax2.set_ylabel("Memory Ratio (ScpTensor / Competitor)", fontweight="bold")
        ax2.set_title("Memory Efficiency", fontsize=12, fontweight="bold")
        ax2.grid(axis="y", alpha=0.3)
        ax2.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        _save_or_show_figure(output_path, "memory comparison")

    def plot_accuracy_comparison(
        self,
        output_path: str | Path | None = None,
        figsize: tuple[float, float] = (10, 6),
    ) -> None:
        """Plot accuracy (correlation) comparison.

        Parameters
        ----------
        output_path : str | Path | None
            Path to save figure.
        figsize : tuple[float, float]
            Figure size (width, height).
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        df = self.to_dataframe()

        fig, ax = plt.subplots(figsize=figsize)

        sns.barplot(
            data=df,
            x="operation",
            y="accuracy",
            ax=ax,
            color=_SCPTENSOR_COLOR,
            errorbar="sd",
        )

        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7, linewidth=1)

        ax.set_xlabel("Operation", fontweight="bold")
        ax.set_ylabel("Output Correlation", fontweight="bold")
        ax.set_title(
            "Result Agreement: ScpTensor vs Competitors", fontsize=14, fontweight="bold"
        )
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()
        _save_or_show_figure(output_path, "accuracy comparison")

    def plot_comprehensive_summary(
        self,
        output_path: str | Path | None = None,
        figsize: tuple[float, float] = (14, 10),
    ) -> None:
        """Create a comprehensive summary plot with all metrics.

        Parameters
        ----------
        output_path : str | Path | None
            Path to save figure.
        figsize : tuple[float, float]
            Figure size (width, height).
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        df = self.to_dataframe()
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Plot 1: Speedup (main metric)
        ax = axes[0, 0]
        pivot_df = df.pivot(
            index="operation",
            columns="dataset_index",
            values="speedup_factor",
        )
        pivot_df.plot(kind="bar", ax=ax, rot=45)
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7)
        ax.set_xlabel("Operation", fontweight="bold")
        ax.set_ylabel("Speedup Factor", fontweight="bold")
        ax.set_title("Performance Speedup", fontsize=12, fontweight="bold")
        ax.legend(title="Dataset", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        # Plot 2: Runtime
        ax = axes[0, 1]
        df_grouped = df.groupby("operation")[
            ["scptensor_time_ms", "competitor_time_ms"]
        ].mean()
        df_grouped.plot(
            kind="bar", ax=ax, rot=45, color=[_SCPTENSOR_COLOR, _COMPETITOR_COLOR]
        )
        ax.set_xlabel("Operation", fontweight="bold")
        ax.set_ylabel("Runtime (ms)", fontweight="bold")
        ax.set_title("Mean Runtime", fontsize=12, fontweight="bold")
        ax.legend(["ScpTensor", "Competitor"], fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        # Plot 3: Memory ratio
        ax = axes[1, 0]
        for op in df["operation"].unique():
            op_data = df[df["operation"] == op]
            ax.scatter([op] * len(op_data), op_data["memory_ratio"], s=100, alpha=0.7)
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7)
        ax.set_xlabel("Operation", fontweight="bold")
        ax.set_ylabel("Memory Ratio", fontweight="bold")
        ax.set_title("Memory Efficiency", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

        # Plot 4: Accuracy
        ax = axes[1, 1]
        sns.barplot(
            data=df,
            x="operation",
            y="accuracy",
            ax=ax,
            color=_SCPTENSOR_COLOR,
            errorbar="sd",
        )
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7)
        ax.set_xlabel("Operation", fontweight="bold")
        ax.set_ylabel("Output Correlation", fontweight="bold")
        ax.set_title("Result Agreement", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

        plt.suptitle(
            "ScpTensor vs Competitors: Comprehensive Benchmark",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()

        _save_or_show_figure(output_path, "comprehensive summary")

    def create_all_plots(
        self,
        output_dir: str | Path = "competitor_benchmark_plots",
    ) -> dict[str, str]:
        """Create all visualization plots.

        Parameters
        ----------
        output_dir : str | Path
            Directory to save plots.

        Returns
        -------
        dict[str, str]
            Dictionary mapping plot names to file paths.
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        plots = {}

        # Comprehensive summary (first for thumbnail)
        path = output_path / "00_comprehensive_summary.png"
        self.plot_comprehensive_summary(path)
        plots["summary"] = str(path)

        # Speedup comparison
        path = output_path / "01_speedup_comparison.png"
        self.plot_speedup_comparison(path)
        plots["speedup"] = str(path)

        # Runtime comparison
        path = output_path / "02_runtime_comparison.png"
        self.plot_runtime_comparison(path)
        plots["runtime"] = str(path)

        # Memory comparison
        path = output_path / "03_memory_comparison.png"
        self.plot_memory_comparison(path)
        plots["memory"] = str(path)

        # Accuracy comparison
        path = output_path / "04_accuracy_comparison.png"
        self.plot_accuracy_comparison(path)
        plots["accuracy"] = str(path)

        print(f"\nAll plots saved to: {output_path}")

        return plots


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        results_path = sys.argv[1]
    else:
        results_path = "competitor_benchmark_results/competitor_benchmark_results.json"

    viz = CompetitorResultVisualizer(results_path)
    viz.create_all_plots()
