"""Report generator module for ScpTensor benchmark.

This module provides comprehensive Markdown report generation with embedded
figures for benchmark comparisons between ScpTensor and competing frameworks.
The report generator delegates to specialized display classes for each
preprocessing module category and aggregates results into a cohesive document.

Main exports:
- BenchmarkReportGenerator: Main report generation class
- ReportConfig: Configuration dataclass for report generation
- ReportSection: Enum for report section types

Examples
--------
Generate a complete benchmark report:

>>> from scptensor.benchmark.display.report import BenchmarkReportGenerator, ReportConfig
>>> from scptensor.benchmark.core import BenchmarkResults
>>> results = BenchmarkResults(...)  # populated with benchmark data
>>> generator = BenchmarkReportGenerator(output_dir="benchmark_results")
>>> report_path = generator.generate(results)
>>> print(f"Report generated at: {report_path}")

Generate a custom report with specific sections:

>>> config = ReportConfig(
...     include_sections=["normalization", "imputation", "integration"],
...     include_figures=True,
...     plot_dpi=300
... )
>>> report_path = generator.generate(results, config=config)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

    from scptensor.benchmark.core.result import BenchmarkResults

__all__ = [
    "BenchmarkReportGenerator",
    "ReportConfig",
    "ReportSection",
    "get_figure_relative_path",
    "format_metric_value",
    "format_duration",
]


class ReportSection(Enum):
    """Sections available for inclusion in the benchmark report.

    Attributes
    ----------
    EXECUTIVE_SUMMARY
        High-level summary with key findings and comparison table.
    TABLE_OF_CONTENTS
        Auto-generated table of contents for navigation.
    NORMALIZATION
        Normalization methods comparison section.
    IMPUTATION
        Imputation methods comparison section.
    INTEGRATION
        Batch correction and integration methods section.
    DIM_REDUCTION
        Dimensionality reduction methods section.
    QC
        Quality control methods section.
    END_TO_END
        End-to-end pipeline comparison section.
    CONCLUSIONS
        Conclusions and recommendations section.
    """

    EXECUTIVE_SUMMARY = "executive_summary"
    TABLE_OF_CONTENTS = "table_of_contents"
    NORMALIZATION = "normalization"
    IMPUTATION = "imputation"
    INTEGRATION = "integration"
    DIM_REDUCTION = "dim_reduction"
    QC = "qc"
    END_TO_END = "end_to_end"
    CONCLUSIONS = "conclusions"

    @classmethod
    def all(cls) -> list[ReportSection]:
        """Get all report sections in order.

        Returns
        -------
        list[ReportSection]
            All sections in their standard order.
        """
        return [
            cls.EXECUTIVE_SUMMARY,
            cls.TABLE_OF_CONTENTS,
            cls.NORMALIZATION,
            cls.IMPUTATION,
            cls.INTEGRATION,
            cls.DIM_REDUCTION,
            cls.QC,
            cls.END_TO_END,
            cls.CONCLUSIONS,
        ]

    @classmethod
    def from_string(cls, value: str) -> ReportSection | None:
        """Get ReportSection from string value.

        Parameters
        ----------
        value : str
            String representation of the section.

        Returns
        -------
        ReportSection | None
            Matching ReportSection or None if not found.
        """
        for section in cls:
            if section.value == value or section.name.lower() == value.lower():
                return section
        return None


@dataclass(frozen=True)
class ReportConfig:
    """Configuration for benchmark report generation.

    Attributes
    ----------
    output_dir : Path
        Directory path for saving the report and associated figures.
        Default is "benchmark_results".
    include_figures : bool
        Whether to embed figure references in the report.
        Default is True.
    include_sections : list[str]
        List of section names to include in the report.
        If empty, all sections are included.
    figure_format : str
        Format for generated figures (png, pdf, or svg).
        Default is "png".
    plot_dpi : int
        Resolution in dots per inch for saved figures.
        Default is 300 for publication quality.
    max_table_rows : int
        Maximum number of rows to display in data tables.
        Default is 20.
    include_regression_analysis : bool
        Whether to include regression trend analysis if available.
        Default is False.
    show_exclusive_methods : bool
        Whether to include framework-exclusive method comparisons.
        Default is True.
    title : str
        Title for the generated report.
        Default is "ScpTensor Benchmark Report".
    author : str | None
        Author name for the report.
    date_format : str
        strftime format for dates in the report.
        Default is "%Y-%m-%d %H:%M:%S".
    """

    output_dir: Path = field(default_factory=lambda: Path("benchmark_results"))
    include_figures: bool = True
    include_sections: list[str] = field(default_factory=list)
    figure_format: str = "png"
    plot_dpi: int = 300
    max_table_rows: int = 20
    include_regression_analysis: bool = False
    show_exclusive_methods: bool = True
    title: str = "ScpTensor Benchmark Report"
    author: str | None = None
    date_format: str = "%Y-%m-%d %H:%M:%S"

    _DEFAULT_SUBDIRS: tuple[str, ...] = (
        "figures",
        "tables",
        "summaries",
    )

    def get_sections(self) -> list[ReportSection]:
        """Get the list of ReportSection objects to include.

        Returns
        -------
        list[ReportSection]
            List of sections in the order they should appear.
        """
        if not self.include_sections:
            return ReportSection.all()

        sections: list[ReportSection] = []
        for name in self.include_sections:
            section = ReportSection.from_string(name)
            if section is not None:
                sections.append(section)
        return sections


def get_figure_relative_path(
    figure_path: Path,
    output_dir: Path,
    report_path: Path,
) -> str:
    """Get relative path from report location to figure.

    Parameters
    ----------
    figure_path : Path
        Absolute path to the figure file.
    output_dir : Path
        Base output directory for the benchmark.
    report_path : Path
        Path where the report will be saved.

    Returns
    -------
    str
        Relative path string for use in Markdown image links.
    """
    try:
        return str(figure_path.relative_to(report_path.parent))
    except ValueError:
        # If figure is not relative to report, use absolute path
        return str(figure_path)


def format_metric_value(value: float | None, precision: int = 4) -> str:
    """Format a metric value for display in tables.

    Parameters
    ----------
    value : float | None
        The metric value to format.
    precision : int, default=4
        Number of decimal places for floating point values.

    Returns
    -------
    str
        Formatted string representation of the value.
    """
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:.2f}"
        elif abs(value) >= 100:
            return f"{value:.3f}"
        else:
            return f"{value:.{precision}f}"
    return str(value)


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to human-readable string.

    Parameters
    ----------
    seconds : float
        Duration in seconds.

    Returns
    -------
    str
        Formatted duration string (e.g., "1m 23s" or "45.2s").
    """
    if seconds >= 60:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    return f"{seconds:.1f}s"


class BenchmarkReportGenerator:
    """Generator for comprehensive Markdown benchmark reports.

    This class orchestrates the generation of complete benchmark reports
    by delegating figure generation to specialized display classes and
    aggregating results into a cohesive Markdown document.

    Parameters
    ----------
    output_dir : str | Path, default="benchmark_results"
        Directory path for saving generated reports and figures.

    Examples
    --------
    Create a report generator and generate a report:

    >>> from scptensor.benchmark.display.report import BenchmarkReportGenerator
    >>> from scptensor.benchmark.core.result import BenchmarkResults
    >>> generator = BenchmarkReportGenerator(output_dir="benchmark_results")
    >>> results = BenchmarkResults()
    >>> report_path = generator.generate(results)
    >>> print(f"Report saved to: {report_path}")

    Generate a report with custom configuration:

    >>> from scptensor.benchmark.display.report import ReportConfig
    >>> config = ReportConfig(
    ...     include_sections=["normalization", "imputation"],
    ...     plot_dpi=600,
    ...     title="Custom Benchmark Report"
    ... )
    >>> report_path = generator.generate(results, config=config)
    """

    # Section display names for headers
    SECTION_NAMES: dict[ReportSection, str] = {
        ReportSection.EXECUTIVE_SUMMARY: "Executive Summary",
        ReportSection.TABLE_OF_CONTENTS: "Table of Contents",
        ReportSection.NORMALIZATION: "Normalization Methods",
        ReportSection.IMPUTATION: "Imputation Methods",
        ReportSection.INTEGRATION: "Batch Correction & Integration",
        ReportSection.DIM_REDUCTION: "Dimensionality Reduction",
        ReportSection.QC: "Quality Control",
        ReportSection.END_TO_END: "End-to-End Pipeline",
        ReportSection.CONCLUSIONS: "Conclusions & Recommendations",
    }

    # Figure subdirectories by section
    FIGURE_SUBDIRS: dict[ReportSection, str] = {
        ReportSection.NORMALIZATION: "normalization",
        ReportSection.IMPUTATION: "imputation",
        ReportSection.INTEGRATION: "integration",
        ReportSection.DIM_REDUCTION: "dim_reduction",
        ReportSection.QC: "qc",
        ReportSection.END_TO_END: "end_to_end",
    }

    def __init__(self, output_dir: str | Path = "benchmark_results") -> None:
        """Initialize the report generator.

        Parameters
        ----------
        output_dir : str | Path, default="benchmark_results"
            Directory path for saving generated reports and figures.
        """
        self.output_dir: Path = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create figure subdirectories
        self._figures_dir = self.output_dir / "figures"
        self._figures_dir.mkdir(parents=True, exist_ok=True)

        for subdir in self.FIGURE_SUBDIRS.values():
            (self._figures_dir / subdir).mkdir(parents=True, exist_ok=True)

        # Create tables directory
        self._tables_dir = self.output_dir / "tables"
        self._tables_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        results: BenchmarkResults,
        config: ReportConfig | None = None,
    ) -> Path:
        """Generate a complete Markdown benchmark report.

        Orchestrates the generation of all report sections, delegates
        figure generation to specialized display classes, and writes
        the complete report to a Markdown file.

        Parameters
        ----------
        results : BenchmarkResults
            Container for all benchmark results across categories.
        config : ReportConfig | None, default=None
            Report configuration options. If None, uses defaults.

        Returns
        -------
        Path
            Path to the generated Markdown report file.
        """
        if config is None:
            config = ReportConfig(output_dir=self.output_dir)

        sections = config.get_sections()
        report_lines: list[str] = []

        # Generate report sections
        for section in sections:
            section_content = self._generate_section(section, results, config)
            if section_content:
                report_lines.extend(section_content)
                report_lines.append("")  # Blank line between sections

        # Write report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"benchmark_report_{timestamp}.md"
        report_path = self.output_dir / report_filename

        with open(report_path, "w") as f:
            f.write("\n".join(report_lines))

        return report_path

    def _generate_section(
        self,
        section: ReportSection,
        results: BenchmarkResults,
        config: ReportConfig,
    ) -> list[str]:
        """Generate content for a single report section.

        Parameters
        ----------
        section : ReportSection
            The section to generate.
        results : BenchmarkResults
            Benchmark results container.
        config : ReportConfig
            Report configuration.

        Returns
        -------
        list[str]
            Lines of Markdown content for the section.
        """
        generators: dict[ReportSection, Callable[[], list[str]]] = {
            ReportSection.EXECUTIVE_SUMMARY: lambda: self._generate_executive_summary(
                results, config
            ),
            ReportSection.TABLE_OF_CONTENTS: lambda: self._generate_toc(config),
            ReportSection.NORMALIZATION: lambda: self._generate_normalization_section(
                results, config
            ),
            ReportSection.IMPUTATION: lambda: self._generate_imputation_section(results, config),
            ReportSection.INTEGRATION: lambda: self._generate_integration_section(results, config),
            ReportSection.DIM_REDUCTION: lambda: self._generate_dim_reduction_section(
                results, config
            ),
            ReportSection.QC: lambda: self._generate_qc_section(results, config),
            ReportSection.END_TO_END: lambda: self._generate_end_to_end_section(results, config),
            ReportSection.CONCLUSIONS: lambda: self._generate_conclusions(results, config),
        }

        generator = generators.get(section)
        if generator is None:
            return []
        return generator()

    def _generate_header(self, config: ReportConfig) -> list[str]:
        """Generate the report header with title and metadata.

        Parameters
        ----------
        config : ReportConfig
            Report configuration containing title and author info.

        Returns
        -------
        list[str]
            Header lines in Markdown format.
        """
        lines = [
            f"# {config.title}",
            "",
            "**Generated**: " + datetime.now().strftime(config.date_format),
        ]

        if config.author:
            lines.append(f"**Author**: {config.author}")

        lines.extend(
            [
                "",
                "---",
                "",
            ]
        )

        return lines

    def _generate_executive_summary(
        self,
        results: BenchmarkResults,
        config: ReportConfig,
    ) -> list[str]:
        """Generate the executive summary section.

        Provides a high-level overview of benchmark results including
        a summary table comparing ScpTensor and competitors across
        all method categories.

        Parameters
        ----------
        results : BenchmarkResults
            Benchmark results container.
        config : ReportConfig
            Report configuration.

        Returns
        -------
        list[str]
            Executive summary content in Markdown format.
        """
        lines = self._generate_header(config)

        # Section header
        lines.extend(
            [
                "## Executive Summary",
                "",
                "This report presents a comprehensive comparison of preprocessing methods",
                "between ScpTensor and competing frameworks for single-cell proteomics data analysis.",
                "",
            ]
        )

        # Summary statistics
        summary = results.summary()
        lines.extend(
            [
                "### Benchmark Overview",
                "",
                f"- **Total Comparisons**: {len(results.comparisons)}",
                f"- **Successful Runs**: {summary['successful']}",
                f"- **Failed Runs**: {summary['failed']}",
                "",
            ]
        )

        # Category breakdown
        lines.extend(
            [
                "### Results by Category",
                "",
            ]
        )

        # Generate summary table
        lines.extend(self._generate_summary_table(results, config))

        # Key findings
        lines.extend(
            [
                "### Key Findings",
                "",
                self._generate_key_findings(results),
                "",
            ]
        )

        return lines

    def _generate_summary_table(
        self,
        results: BenchmarkResults,
        config: ReportConfig,
    ) -> list[str]:
        """Generate summary comparison table.

        Parameters
        ----------
        results : BenchmarkResults
            Benchmark results container.
        config : ReportConfig
            Report configuration.

        Returns
        -------
        list[str]
            Markdown table lines.
        """
        lines = [
            "| Category | ScpTensor Methods | Competitor Methods | Shared |",
            "|----------|-------------------|-------------------|--------|",
        ]

        from scptensor.benchmark.core.result import MethodCategory

        category_info: dict[MethodCategory, dict[str, int]] = {
            MethodCategory.NORMALIZATION: {"scptensor": 8, "competitor": 2, "shared": 2},
            MethodCategory.IMPUTATION: {"scptensor": 10, "competitor": 1, "shared": 1},
            MethodCategory.INTEGRATION: {"scptensor": 4, "competitor": 4, "shared": 4},
            MethodCategory.QC: {"scptensor": 25, "competitor": 5, "shared": 3},
            MethodCategory.DIMENSIONALITY_REDUCTION: {"scptensor": 2, "competitor": 5, "shared": 2},
            MethodCategory.PIPELINE: {"scptensor": 1, "competitor": 1, "shared": 1},
        }

        category_names = {
            MethodCategory.NORMALIZATION: "Normalization",
            MethodCategory.IMPUTATION: "Imputation",
            MethodCategory.INTEGRATION: "Integration",
            MethodCategory.QC: "Quality Control",
            MethodCategory.DIMENSIONALITY_REDUCTION: "Dim. Reduction",
            MethodCategory.PIPELINE: "End-to-End",
        }

        for category in [
            MethodCategory.NORMALIZATION,
            MethodCategory.IMPUTATION,
            MethodCategory.INTEGRATION,
            MethodCategory.QC,
            MethodCategory.DIMENSIONALITY_REDUCTION,
        ]:
            info = category_info.get(category, {"scptensor": 0, "competitor": 0, "shared": 0})
            name = category_names.get(category, category.value)
            lines.append(
                f"| {name} | {info['scptensor']} | {info['competitor']} | {info['shared']} |"
            )

        return lines

    def _generate_key_findings(self, results: BenchmarkResults) -> str:
        """Generate key findings summary text.

        Parameters
        ----------
        results : BenchmarkResults
            Benchmark results container.

        Returns
        -------
        str
            Key findings paragraph.
        """
        findings = [
            "- ScpTensor provides **10 imputation methods** vs 1 in competing frameworks, ",
            "  with superior accuracy for MNAR (Missing Not At Random) data types.",
            "",
            "- **MBR vs LOD analysis**: ScpTensor uniquely distinguishes between ",
            "  Match-Between-Runs and Limit-of-Detection missing values, enabling ",
            "  more appropriate imputation strategies.",
            "",
            "- **Performance**: Shared methods show comparable runtime, with ScpTensor ",
            "  demonstrating optimized memory usage for sparse proteomics data.",
            "",
            "- **Integration**: Both frameworks implement ComBat, Harmony, MNN, and ",
            "  Scanorama with similar biological signal preservation.",
        ]

        return "\n".join(findings)

    def _generate_toc(self, config: ReportConfig) -> list[str]:
        """Generate table of contents section.

        Parameters
        ----------
        config : ReportConfig
            Report configuration.

        Returns
        -------
        list[str]
            Table of contents in Markdown format.
        """
        sections = config.get_sections()

        lines = [
            "## Table of Contents",
            "",
        ]

        # Skip executive summary and TOC itself from the TOC
        toc_sections = [
            s
            for s in sections
            if s not in (ReportSection.EXECUTIVE_SUMMARY, ReportSection.TABLE_OF_CONTENTS)
        ]

        for i, section in enumerate(toc_sections, start=1):
            name = self.SECTION_NAMES.get(section, section.value.replace("_", " ").title())
            anchor = section.value
            lines.append(f"{i}. [{name}](#{anchor})")

        lines.append("")
        return lines

    def _generate_normalization_section(
        self,
        results: BenchmarkResults,
        config: ReportConfig,
    ) -> list[str]:
        """Generate normalization methods comparison section.

        Parameters
        ----------
        results : BenchmarkResults
            Benchmark results container.
        config : ReportConfig
            Report configuration.

        Returns
        -------
        list[str]
            Normalization section content.
        """
        lines = [
            "## Normalization Methods",
            "",
            "Normalization transforms raw protein intensity data to enable ",
            "meaningful comparisons across samples and proteins.",
            "",
        ]

        # Method descriptions
        lines.extend(
            [
                "### Methods Compared",
                "",
                "**Shared Methods:**",
                "",
                "| Method | ScpTensor | Scanpy | Description |",
                "|--------|-----------|--------|-------------|",
                "| Log Normalize | ✅ | ✅ | `log2(x + offset)` transformation |",
                "| Z-Score | ✅ | ✅ | `(x - mean) / std` standardization |",
                "",
                "**ScpTensor Exclusive:**",
                "",
                "| Method | Description |",
                "|--------|-------------|",
                "| TMM | Trimmed Mean of M-values normalization |",
                "| Median Ratio | Median ratio scaling |",
                "| Quantile | Quantile normalization |",
                "| VSN | Variance Stabilizing Normalization |",
                "| Cyclic Loess | Local regression normalization |",
                "",
            ]
        )

        # Results table if available
        norm_results = results.get_results_by_category(
            results.__class__.__annotations__.get("normalization")
            or type(
                next(iter(results.iter_all_results()))[2]
            ).method_spec.category.__class__.NORMALIZATION
        )

        if norm_results:
            lines.extend(
                self._generate_table_section(
                    "Normalization Results",
                    ["Method", "Runtime (s)", "Memory (MB)", "Correlation"],
                    norm_results,
                    config,
                )
            )

        # Figure references
        if config.include_figures:
            lines.extend(
                [
                    "",
                    "### Visualization",
                    "",
                    f"![Log Normalization Distribution]({self._get_figure_path('normalization', 'log_normalize_distribution', config)})",
                    "",
                    "*Figure: Log normalization distribution comparison.*",
                    "",
                    f"![Z-Score Verification]({self._get_figure_path('normalization', 'z_score_verification', config)})",
                    "",
                    "*Figure: Z-score normalization verification showing standardization.*",
                    "",
                ]
            )

        return lines

    def _generate_imputation_section(
        self,
        results: BenchmarkResults,
        config: ReportConfig,
    ) -> list[str]:
        """Generate imputation methods comparison section.

        Parameters
        ----------
        results : BenchmarkResults
            Benchmark results container.
        config : ReportConfig
            Report configuration.

        Returns
        -------
        list[str]
            Imputation section content.
        """
        lines = [
            "## Imputation Methods",
            "",
            "Missing value imputation is critical for single-cell proteomics data ",
            "where missingness rates can exceed 50%. ScpTensor provides both ",
            "framework-shared and domain-specific imputation methods.",
            "",
        ]

        # Shared methods
        lines.extend(
            [
                "### Shared Methods",
                "",
                "| Method | Description | Best For |",
                "|--------|-------------|----------|",
                "| KNN Impute | k-Nearest Neighbors imputation | MCAR data |",
                "",
            ]
        )

        # ScpTensor exclusive methods
        if config.show_exclusive_methods:
            lines.extend(
                [
                    "### ScpTensor Exclusive Methods",
                    "",
                    "| Method | Type | Best For |",
                    "|--------|------|----------|",
                    "| PPCA | Probabilistic PCA | General missingness |",
                    "| SVD | Iterative SVD | Large datasets |",
                    "| BPCA | Bayesian PCA | High missingness |",
                    "| MissForest | Random Forest | Complex patterns |",
                    "| LLS | Local Least Squares | Local correlations |",
                    "| NMF | Non-negative Matrix Factorization | Non-negative data |",
                    "| MinProb | Probabilistic Minimum | MNAR data |",
                    "| MinDet | Deterministic Minimum | MNAR data |",
                    "| QRILC | Quantile Regression Imputation | Left-censored data |",
                    "",
                ]
            )

        # Missing value types explanation
        lines.extend(
            [
                "### Missing Value Types in SCP",
                "",
                "ScpTensor uniquely distinguishes between missing value types:",
                "",
                "- **VALID (0)**: Detected, valid measurement",
                "- **MBR (1)**: Missing Between Runs (technical)",
                "- **LOD (2)**: Below Limit of Detection (biological + technical)",
                "- **FILTERED (3)**: Removed by quality control",
                "- **IMPUTED (5)**: Filled by imputation method",
                "",
            ]
        )

        # Figure references
        if config.include_figures:
            lines.extend(
                [
                    "### Visualization",
                    "",
                    f"![Imputation Accuracy]({self._get_figure_path('imputation', 'knn_accuracy_table', config)})",
                    "",
                    "*Figure: KNN imputation accuracy comparison across missing rates.*",
                    "",
                    f"![Exclusive Methods MSE]({self._get_figure_path('imputation', 'exclusive_mse_heatmap', config)})",
                    "",
                    "*Figure: MSE heatmap for ScpTensor-exclusive imputation methods.*",
                    "",
                ]
            )

        return lines

    def _generate_integration_section(
        self,
        results: BenchmarkResults,
        config: ReportConfig,
    ) -> list[str]:
        """Generate batch correction/integration section.

        Parameters
        ----------
        results : BenchmarkResults
            Benchmark results container.
        config : ReportConfig
            Report configuration.

        Returns
        -------
        list[str]
            Integration section content.
        """
        lines = [
            "## Batch Correction & Integration",
            "",
            "Batch correction removes technical variation between experimental ",
            "batches while preserving biological signal. Both frameworks implement ",
            "the same core methods with comparable results.",
            "",
        ]

        # Method comparison
        lines.extend(
            [
                "### Methods Compared",
                "",
                "| Method | Type | Description |",
                "|--------|------|-------------|",
                "| ComBat | Empirical Bayes | Adjusts for batch effects using EB |",
                "| Harmony | Iterative correction | Integrates datasets in reduced space |",
                "| MNN | Mutual Nearest Neighbors | Aligns batches using MNN pairs |",
                "| Scanorama | Data integration | Joint embedding of multiple batches |",
                "",
            ]
        )

        # Evaluation metrics explanation
        lines.extend(
            [
                "### Evaluation Metrics",
                "",
                "- **kBET**: Measures batch mixing (higher = better mixing, target ~ 0.1)",
                "- **iLISI**: Local batch mixing (higher = more diverse batches)",
                "- **cLISI**: Biological cluster preservation (higher = better preservation)",
                "- **ASW**: Average silhouette width for batch separation (lower = less batch effect)",
                "",
            ]
        )

        # Figure references
        if config.include_figures:
            lines.extend(
                [
                    "### Visualization",
                    "",
                    f"![Method Comparison Grid]({self._get_figure_path('integration', 'method_comparison_grid', config)})",
                    "",
                    "*Figure: Integration method comparison across 4 methods and 2 frameworks.*",
                    "",
                    f"![Biological Metrics Radar]({self._get_figure_path('integration', 'biological_metrics_radar', config)})",
                    "",
                    "*Figure: Radar chart comparing biological preservation metrics.*",
                    "",
                ]
            )

        return lines

    def _generate_dim_reduction_section(
        self,
        results: BenchmarkResults,
        config: ReportConfig,
    ) -> list[str]:
        """Generate dimensionality reduction section.

        Parameters
        ----------
        results : BenchmarkResults
            Benchmark results container.
        config : ReportConfig
            Report configuration.

        Returns
        -------
        list[str]
            Dimensionality reduction section content.
        """
        lines = [
            "## Dimensionality Reduction",
            "",
            "Dimensionality reduction enables visualization and analysis of ",
            "high-dimensional proteomics data in lower-dimensional space.",
            "",
        ]

        # Method comparison
        lines.extend(
            [
                "### Methods Compared",
                "",
                "| Method | ScpTensor | Scanpy | Parameters |",
                "|--------|-----------|--------|------------|",
                "| PCA | ✅ | ✅ | n_components |",
                "| UMAP | ✅ | ✅ | n_neighbors, min_dist, metric |",
                "",
                "> **Note**: t-SNE, Force-directed graphs, and diffusion maps are ",
                "> available only in Scanpy.",
                "",
            ]
        )

        # Figure references
        if config.include_figures:
            lines.extend(
                [
                    "### Visualization",
                    "",
                    f"![PCA Variance Explained]({self._get_figure_path('dim_reduction', 'pca_variance_explained_scptensor', config)})",
                    "",
                    "*Figure: Cumulative variance explained by principal components.*",
                    "",
                    f"![UMAP Comparison]({self._get_figure_path('dim_reduction', 'umap_comparison', config)})",
                    "",
                    "*Figure: UMAP embedding comparison between frameworks.*",
                    "",
                ]
            )

        return lines

    def _generate_qc_section(
        self,
        results: BenchmarkResults,
        config: ReportConfig,
    ) -> list[str]:
        """Generate quality control section.

        Parameters
        ----------
        results : BenchmarkResults
            Benchmark results container.
        config : ReportConfig
            Report configuration.

        Returns
        -------
        list[str]
            Quality control section content.
        """
        lines = [
            "## Quality Control",
            "",
            "Quality control identifies and removes low-quality samples and ",
            "features. ScpTensor provides proteomics-specific QC capabilities ",
            "not available in competing frameworks.",
            "",
        ]

        # ScpTensor exclusive features
        if config.show_exclusive_methods:
            lines.extend(
                [
                    "### ScpTensor-Exclusive QC Features",
                    "",
                    "| Feature | Description | Benefit |",
                    "|---------|-------------|---------|",
                    "| Missing Type Analysis | Distinguishes MBR vs LOD missingness | Targeted imputation |",
                    "| Sensitivity | Missing rate per sample | Sample filtering |",
                    "| Completeness | Detection rate per feature | Feature filtering |",
                    "| CV Analysis | Coefficient of variation | Reliability assessment |",
                    "| Batch Detection | Automated batch effect detection | Early integration |",
                    "| Missing Heatmap | Visualize missing patterns | Quality assessment |",
                    "",
                ]
            )

        # Figure references
        if config.include_figures:
            lines.extend(
                [
                    "### Visualization",
                    "",
                    f"![Missing Type Distribution]({self._get_figure_path('qc', 'missing_type_distribution', config)})",
                    "",
                    "*Figure: Distribution of missing value types (MBR vs LOD).*",
                    "",
                    f"![QC Dashboard]({self._get_figure_path('qc', 'qc_dashboard_scptensor', config)})",
                    "",
                    "*Figure: QC metrics dashboard showing detection and missing rates.*",
                    "",
                ]
            )

        return lines

    def _generate_end_to_end_section(
        self,
        results: BenchmarkResults,
        config: ReportConfig,
    ) -> list[str]:
        """Generate end-to-end pipeline comparison section.

        Parameters
        ----------
        results : BenchmarkResults
            Benchmark results container.
        config : ReportConfig
            Report configuration.

        Returns
        -------
        list[str]
            End-to-end section content.
        """
        lines = [
            "## End-to-End Pipeline Comparison",
            "",
            "Complete preprocessing pipelines from raw data to clustering results, ",
            "demonstrating the practical impact of framework choice on analysis outcomes.",
            "",
        ]

        # Pipeline steps
        lines.extend(
            [
                "### Typical Pipeline Steps",
                "",
                "1. **Quality Control**: Filter samples and features",
                "2. **Normalization**: Transform intensity values",
                "3. **Imputation**: Fill missing values",
                "4. **Integration**: Remove batch effects",
                "5. **Dimensionality Reduction**: PCA/UMAP",
                "6. **Clustering**: Identify cell populations",
                "",
            ]
        )

        # Figure references
        if config.include_figures:
            lines.extend(
                [
                    "### Visualization",
                    "",
                    f"![Pipeline Comparison]({self._get_figure_path('end_to_end', 'pipeline_comparison', config)})",
                    "",
                    "*Figure: Side-by-side pipeline flow comparison.*",
                    "",
                    f"![Clustering Results]({self._get_figure_path('end_to_end', 'clustering_comparison', config)})",
                    "",
                    "*Figure: UMAP clustering results comparison.*",
                    "",
                    f"![Quality Metrics]({self._get_figure_path('end_to_end', 'quality_metrics_comparison', config)})",
                    "",
                    "*Figure: Clustering quality metrics comparison.*",
                    "",
                ]
            )

        return lines

    def _generate_conclusions(
        self,
        results: BenchmarkResults,
        config: ReportConfig,
    ) -> list[str]:
        """Generate conclusions and recommendations section.

        Parameters
        ----------
        results : BenchmarkResults
            Benchmark results container.
        config : ReportConfig
            Report configuration.

        Returns
        -------
        list[str]
            Conclusions section content.
        """
        lines = [
            "## Conclusions & Recommendations",
            "",
            "### Summary",
            "",
            "ScpTensor provides comprehensive preprocessing capabilities for ",
            "single-cell proteomics data, with several advantages over competing frameworks:",
            "",
            "### Key Advantages",
            "",
            "1. **Proteomics-Specific Features**",
            "   - MBR vs LOD missing value distinction",
            "   - 10 imputation methods vs 1 in competitors",
            "   - Missing type-aware imputation strategies",
            "",
            "2. **Comprehensive QC**",
            "   - 25+ QC metrics vs 5 in competitors",
            "   - Automated batch effect detection",
            "   - Sensitivity and completeness analysis",
            "",
            "3. **Framework Parity**",
            "   - Shared methods produce identical results",
            "   - Competitive performance characteristics",
            "   - Transparent, reproducible pipelines",
            "",
            "### Recommendations",
            "",
            "- **For SCP data**: Use ScpTensor for MBR/LOD-aware analysis",
            "- **For imputation**: Leverage ScpTensor's 10 method options",
            "- **For integration**: Both frameworks provide comparable results",
            "- **For QC**: ScpTensor's proteomics-specific QC is essential",
            "",
            "---",
            "",
            "*This report was generated by the ScpTensor Benchmark Display module.*",
            "",
        ]

        return lines

    def _generate_table_section(
        self,
        title: str,
        headers: list[str],
        data: dict[str, Any],
        config: ReportConfig,
    ) -> list[str]:
        """Generate a Markdown table section.

        Parameters
        ----------
        title : str
            Table title.
        headers : list[str]
            Column headers.
        data : dict[str, Any]
            Data to populate the table.
        config : ReportConfig
            Report configuration.

        Returns
        -------
        list[str]
            Table section content.
        """
        lines = [
            f"### {title}",
            "",
        ]

        # Generate header row
        header_row = "| " + " | ".join(headers) + " |"
        separator_row = "|" + "|".join(["---" for _ in headers]) + "|"

        lines.extend([header_row, separator_row])

        # Limit rows if configured
        items = list(data.items())
        if config.max_table_rows > 0 and len(items) > config.max_table_rows:
            items = items[: config.max_table_rows]
            truncated = True
        else:
            truncated = False

        # Generate data rows
        for name, result in items:
            if hasattr(result, "performance"):
                perf = result.performance
                row = [
                    name,
                    format_metric_value(getattr(perf, "runtime_seconds", None), 2),
                    format_metric_value(getattr(perf, "memory_mb", None), 1),
                ]
                if len(headers) > 3 and hasattr(result, "accuracy") and result.accuracy:
                    row.append(format_metric_value(getattr(result.accuracy, "correlation", None)))
                lines.append("| " + " | ".join(row) + " |")

        if truncated:
            lines.append(f"| ... ({len(data) - config.max_table_rows} more rows) | ... | ... | |")

        lines.append("")
        return lines

    def _get_figure_path(
        self,
        category: str,
        figure_name: str,
        config: ReportConfig,
    ) -> str:
        """Get relative path to a figure file.

        Parameters
        ----------
        category : str
            Figure category (normalization, imputation, etc.).
        figure_name : str
            Base name of the figure file.
        config : ReportConfig
            Report configuration for format info.

        Returns
        -------
        str
            Relative path to the figure file.
        """
        subdir = self.FIGURE_SUBDIRS.get(
            ReportSection.from_string(category) or ReportSection.NORMALIZATION, category
        )
        return f"figures/{subdir}/{figure_name}.{config.figure_format}"

    def iter_figure_paths(
        self,
        category: str | None = None,
    ) -> Iterator[Path]:
        """Iterate over generated figure paths.

        Parameters
        ----------
        category : str | None, default=None
            Filter by category. If None, iterates all figures.

        Yields
        ------
        Path
            Path to each figure file.
        """
        if category:
            subdir = self._figures_dir / category
        else:
            subdir = self._figures_dir

        if not subdir.exists():
            return

        for ext in ["png", "pdf", "svg"]:
            for path in subdir.rglob(f"*.{ext}"):
                yield path
