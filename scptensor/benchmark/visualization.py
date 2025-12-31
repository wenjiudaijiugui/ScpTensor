"""
Efficient results visualization for benchmarking with multiple output formats.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scienceplots
from .core import BenchmarkResults, MethodRunResult


class ResultsVisualizer:
    """
    Efficient visualization suite for benchmark results.

    Features:
    - Publication-quality static plots (matplotlib + scienceplots)
    - Interactive dashboards (plotly)
    - Efficient computation with caching
    - Multiple output formats
    """

    def __init__(self, style: str = "science"):
        """
        Initialize visualizer with plotting style.

        Args:
            style: Plotting style ('science', 'science+no-latex', 'default')
        """
        self.style = style
        self._setup_style()
        self._color_palette = px.colors.qualitative.Set1

        # Cache for expensive computations
        self._cache = {}

    def _setup_style(self):
        """Setup matplotlib style for publication-quality plots."""
        if self.style == "science":
            plt.style.use(["science", "no-latex"])
        elif self.style == "science+no-latex":
            plt.style.use(["science", "no-latex"])
        else:
            plt.style.use("default")

    def create_comprehensive_report(
        self,
        results: BenchmarkResults,
        output_dir: str = "benchmark_results",
        create_interactive: bool = True
    ) -> Dict[str, str]:
        """
        Create a comprehensive benchmark report with all visualizations.

        Args:
            results: BenchmarkResults to visualize
            output_dir: Directory to save plots
            create_interactive: Whether to create interactive HTML report

        Returns:
            Dictionary mapping plot types to file paths
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        file_paths = {}

        # Static plots (publication quality)
        file_paths['method_comparison_heatmap'] = self.method_comparison_heatmap(
            results, save_path=f"{output_dir}/method_comparison_heatmap.png"
        )

        file_paths['computational_efficiency'] = self.computational_efficiency_scatter(
            results, save_path=f"{output_dir}/computational_efficiency.png"
        )

        file_paths['biological_vs_technical'] = self.biological_vs_technical_tradeoff(
            results, save_path=f"{output_dir}/biological_vs_technical.png"
        )

        file_paths['method_rankings'] = self.method_rankings_table(
            results, save_path=f"{output_dir}/method_rankings.png"
        )

        # Interactive visualizations
        if create_interactive:
            file_paths['interactive_dashboard'] = self.create_interactive_dashboard(
                results, save_path=f"{output_dir}/interactive_dashboard.html"
            )

        return file_paths

    def method_comparison_heatmap(
        self,
        results: BenchmarkResults,
        dataset_name: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> str:
        """
        Create heatmap comparing methods across multiple metrics.

        Optimized for efficiency by computing metrics once.
        """
        cache_key = f"heatmap_{dataset_name}_{id(results)}"
        if cache_key in self._cache:
            comparison_df = self._cache[cache_key]
        else:
            comparison_df = self._compute_method_comparison_matrix(results, dataset_name)
            self._cache[cache_key] = comparison_df

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))

        # Prepare data for heatmap
        # Use actual column names from the data
        available_columns = comparison_df.columns.tolist()
        metrics_mapping = {
            'Data Recovery Rate': 'mean_data_recovery_rate',
            'Group Separation': 'mean_group_separation',
            'Batch Mixing Score': 'mean_batch_mixing_score',
            'Runtime': 'mean_runtime_seconds',
            'Memory Usage': 'mean_memory_usage_mb'
        }

        metrics_to_plot = []
        plot_data = comparison_df.copy()

        for display_name, column_name in metrics_mapping.items():
            if column_name in available_columns:
                if display_name == 'Runtime':
                    display_name = 'Runtime (log scale)'
                plot_data[display_name] = plot_data[column_name]
                metrics_to_plot.append(display_name)

        if not metrics_to_plot:
            print("Warning: No valid metrics found for visualization")
            return ""

        # Only use the metrics we want to plot
        plot_data = plot_data[metrics_to_plot]

        # Handle runtime log scaling
        if 'Runtime (log scale)' in plot_data.columns:
            plot_data['Runtime (log scale)'] = np.log10(plot_data['Runtime (log scale)'] + 1)

        # Scale all metrics to [0, 1] for fair comparison
        for col in plot_data.columns:
            col_min, col_max = plot_data[col].min(), plot_data[col].max()
            if col_max > col_min:
                plot_data[col] = (plot_data[col] - col_min) / (col_max - col_min)

        # Create heatmap
        sns.heatmap(
            plot_data,
            annot=True,
            cmap='RdYlBu_r',
            center=0.5,
            fmt='.2f',
            ax=ax,
            cbar_kws={'label': 'Normalized Performance (0-1)'}
        )

        ax.set_title(f'Method Comparison Heatmap{f" - {dataset_name}" if dataset_name else ""}',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Methods', fontsize=12)
        ax.set_ylabel('Metrics', fontsize=12)

        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            plt.show()
            return ""

    def computational_efficiency_scatter(
        self,
        results: BenchmarkResults,
        save_path: Optional[str] = None
    ) -> str:
        """
        Create scatter plot of runtime vs performance (Pareto frontier).

        Efficiently identifies Pareto-optimal methods.
        """
        comparison_df = results.get_method_comparison()

        if comparison_df.empty:
            return ""

        fig, ax = plt.subplots(figsize=(10, 8))

        # Use biological performance as primary metric
        x_metric = 'mean_runtime_seconds'
        y_metric = 'mean_group_separation'
        size_metric = 'mean_data_recovery_rate'

        # Remove methods with no biological scores
        plot_df = comparison_df.dropna(subset=[y_metric])

        if plot_df.empty:
            return ""

        # Calculate Pareto frontier
        pareto_indices = self._calculate_pareto_frontier(
            plot_df[x_metric].values,
            plot_df[y_metric].values
        )

        # Plot all methods
        scatter = ax.scatter(
            plot_df[x_metric],
            plot_df[y_metric],
            s=plot_df[size_metric] * 100,  # Size based on data recovery
            c=range(len(plot_df)),
            cmap='viridis',
            alpha=0.7,
            edgecolors='black',
            linewidth=1
        )

        # Highlight Pareto-optimal methods
        pareto_df = plot_df.iloc[pareto_indices]
        ax.scatter(
            pareto_df[x_metric],
            pareto_df[y_metric],
            s=200,
            facecolors='none',
            edgecolors='red',
            linewidth=3,
            label='Pareto Optimal',
            zorder=10
        )

        # Add method labels for Pareto-optimal points
        for idx, row in pareto_df.iterrows():
            ax.annotate(
                idx,
                (row[x_metric], row[y_metric]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=10,
                fontweight='bold',
                color='red'
            )

        ax.set_xlabel('Runtime (seconds)', fontsize=12)
        ax.set_ylabel('Group Separation (Silhouette Score)', fontsize=12)
        ax.set_title('Computational Efficiency Analysis', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add colorbar for data recovery rate
        cbar = plt.colorbar(scatter)
        cbar.set_label('Data Recovery Rate', rotation=270, labelpad=15)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            plt.show()
            return ""

    def biological_vs_technical_tradeoff(
        self,
        results: BenchmarkResults,
        save_path: Optional[str] = None
    ) -> str:
        """
        Create plot showing biological vs technical performance tradeoffs.
        """
        comparison_df = results.get_method_comparison()

        if comparison_df.empty:
            return ""

        fig, ax = plt.subplots(figsize=(10, 8))

        x_metric = 'mean_group_separation'  # Biological
        y_metric = 'mean_data_recovery_rate'  # Technical

        # Remove methods with missing biological scores
        plot_df = comparison_df.dropna(subset=[x_metric])

        if plot_df.empty:
            return ""

        # Create scatter plot
        scatter = ax.scatter(
            plot_df[x_metric],
            plot_df[y_metric],
            s=plot_df['mean_batch_mixing_score'] * 200,
            c=plot_df['mean_runtime_seconds'],
            cmap='plasma',
            alpha=0.8,
            edgecolors='black',
            linewidth=1
        )

        # Add method labels
        for idx, row in plot_df.iterrows():
            ax.annotate(
                idx,
                (row[x_metric], row[y_metric]),
                xytext=(3, 3),
                textcoords='offset points',
                fontsize=9,
                alpha=0.8
            )

        ax.set_xlabel('Group Separation (Biological Quality)', fontsize=12)
        ax.set_ylabel('Data Recovery Rate (Technical Quality)', fontsize=12)
        ax.set_title('Biological vs Technical Performance Tradeoff', fontsize=14, fontweight='bold')

        # Add colorbar for runtime
        cbar = plt.colorbar(scatter)
        cbar.set_label('Runtime (seconds)', rotation=270, labelpad=15)

        # Add quadrant lines for "ideal" regions
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)

        # Add quadrant labels
        ax.text(0.1, 0.9, 'High Technical\nLow Biological',
                transform=ax.transAxes, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.text(0.9, 0.9, 'High Technical\nHigh Biological',
                transform=ax.transAxes, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        ax.text(0.1, 0.1, 'Low Technical\nLow Biological',
                transform=ax.transAxes, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
        ax.text(0.9, 0.1, 'Low Technical\nHigh Biological',
                transform=ax.transAxes, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            plt.show()
            return ""

    def method_rankings_table(
        self,
        results: BenchmarkResults,
        save_path: Optional[str] = None
    ) -> str:
        """
        Create a comprehensive method rankings table visualization.
        """
        comparison_df = results.get_method_comparison()

        if comparison_df.empty:
            return ""

        # Calculate composite scores
        comparison_df['biological_score'] = (
            comparison_df['mean_group_separation'].fillna(0) * 0.6 +
            comparison_df['mean_batch_mixing_score'] * 0.4
        )

        comparison_df['technical_score'] = (
            comparison_df['mean_data_recovery_rate'] * 0.5 +
            comparison_df['mean_batch_mixing_score'] * 0.5
        )

        # Normalize runtime (lower is better)
        max_runtime = comparison_df['mean_runtime_seconds'].max()
        comparison_df['speed_score'] = 1 - (comparison_df['mean_runtime_seconds'] / max_runtime)

        # Overall score (weighted combination)
        comparison_df['overall_score'] = (
            comparison_df['biological_score'] * 0.4 +
            comparison_df['technical_score'] * 0.3 +
            comparison_df['speed_score'] * 0.3
        )

        # Sort by overall score
        comparison_df = comparison_df.sort_values('overall_score', ascending=False)

        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Overall rankings
        y_pos = np.arange(len(comparison_df))
        bars1 = ax1.barh(y_pos, comparison_df['overall_score'], color='skyblue')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(comparison_df.index)
        ax1.set_xlabel('Overall Score')
        ax1.set_title('Method Overall Rankings')
        ax1.grid(axis='x', alpha=0.3)

        # 2. Biological performance
        bars2 = ax2.barh(y_pos, comparison_df['biological_score'], color='lightgreen')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(comparison_df.index)
        ax2.set_xlabel('Biological Score')
        ax2.set_title('Biological Performance')
        ax2.grid(axis='x', alpha=0.3)

        # 3. Technical performance
        bars3 = ax3.barh(y_pos, comparison_df['technical_score'], color='salmon')
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(comparison_df.index)
        ax3.set_xlabel('Technical Score')
        ax3.set_title('Technical Performance')
        ax3.grid(axis='x', alpha=0.3)

        # 4. Computational efficiency
        bars4 = ax4.barh(y_pos, comparison_df['speed_score'], color='plum')
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(comparison_df.index)
        ax4.set_xlabel('Speed Score')
        ax4.set_title('Computational Efficiency')
        ax4.grid(axis='x', alpha=0.3)

        # Add value labels on bars
        for ax, bars in [(ax1, bars1), (ax2, bars2), (ax3, bars3), (ax4, bars4)]:
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{width:.2f}', ha='left', va='center', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            plt.show()
            return ""

    def create_interactive_dashboard(
        self,
        results: BenchmarkResults,
        save_path: Optional[str] = None
    ) -> str:
        """
        Create interactive dashboard with plotly for exploration.

        Efficiently creates multiple linked plots.
        """
        comparison_df = results.get_method_comparison()

        if comparison_df.empty:
            return ""

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Efficiency Analysis', 'Performance Tradeoff',
                          'Method Rankings', 'Detailed Comparison'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "table"}]]
        )

        # 1. Efficiency scatter plot
        scatter = go.Scatter(
            x=comparison_df['mean_runtime_seconds'],
            y=comparison_df['mean_group_separation'].fillna(0),
            mode='markers+text',
            text=comparison_df.index,
            textposition="top center",
            marker=dict(
                size=comparison_df['mean_data_recovery_rate'] * 20,
                color=comparison_df['mean_batch_mixing_score'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Batch Mixing Score", x=1.02)
            ),
            name='Methods'
        )
        fig.add_trace(scatter, row=1, col=1)

        # 2. Performance tradeoff
        tradeoff = go.Scatter(
            x=comparison_df['mean_group_separation'].fillna(0),
            y=comparison_df['mean_data_recovery_rate'],
            mode='markers+text',
            text=comparison_df.index,
            textposition="top center",
            marker=dict(
                size=15,
                color=comparison_df['mean_runtime_seconds'],
                colorscale='Plasma',
                showscale=True,
                colorbar=dict(title="Runtime (s)", x=1.02)
            ),
            name='Methods'
        )
        fig.add_trace(tradeoff, row=1, col=2)

        # 3. Rankings bar chart
        comparison_df['overall_score'] = (
            comparison_df['mean_group_separation'].fillna(0) * 0.4 +
            comparison_df['mean_data_recovery_rate'] * 0.3 +
            (1 - comparison_df['mean_runtime_seconds'] / comparison_df['mean_runtime_seconds'].max()) * 0.3
        )
        comparison_df_sorted = comparison_df.sort_values('overall_score', ascending=True)

        rankings = go.Bar(
            x=comparison_df_sorted['overall_score'],
            y=comparison_df_sorted.index,
            orientation='h',
            marker_color='lightblue',
            name='Overall Score'
        )
        fig.add_trace(rankings, row=2, col=1)

        # 4. Detailed comparison table
        table_data = comparison_df[['mean_runtime_seconds', 'mean_memory_usage_mb',
                                 'mean_data_recovery_rate', 'mean_group_separation']].round(3)

        table = go.Table(
            header=dict(values=['Method'] + list(table_data.columns)),
            cells=dict(values=[table_data.index] + [table_data[col] for col in table_data.columns])
        )
        fig.add_trace(table, row=2, col=2)

        # Update layout
        fig.update_layout(
            title="Interactive Benchmark Results Dashboard",
            height=800,
            showlegend=False
        )

        # Update subplot titles and labels
        fig.update_xaxes(title_text="Runtime (seconds)", row=1, col=1)
        fig.update_yaxes(title_text="Group Separation", row=1, col=1)
        fig.update_xaxes(title_text="Group Separation", row=1, col=2)
        fig.update_yaxes(title_text="Data Recovery Rate", row=1, col=2)
        fig.update_xaxes(title_text="Overall Score", row=2, col=1)

        if save_path:
            fig.write_html(save_path)
            return save_path
        else:
            fig.show()
            return ""

    def _compute_method_comparison_matrix(
        self,
        results: BenchmarkResults,
        dataset_name: Optional[str] = None
    ) -> pd.DataFrame:
        """Efficiently compute comparison matrix with caching."""
        return results.get_method_comparison(dataset_name)

    def _calculate_pareto_frontier(self, costs: np.ndarray, benefits: np.ndarray) -> np.ndarray:
        """
        Efficiently calculate Pareto frontier indices.

        Args:
            costs: Cost values (lower is better)
            benefits: Benefit values (higher is better)

        Returns:
            Indices of Pareto-optimal points
        """
        points = np.column_stack((costs, benefits))
        n_points = len(points)

        # Initialize all points as Pareto-optimal
        is_pareto = np.ones(n_points, dtype=bool)

        for i in range(n_points):
            if is_pareto[i]:
                # Remove points dominated by current point
                dominated = np.all(
                    (costs >= costs[i]) & (benefits <= benefits[i]) &
                    ((costs > costs[i]) | (benefits < benefits[i])),
                    axis=0
                )
                is_pareto[dominated] = False

        return np.where(is_pareto)[0]