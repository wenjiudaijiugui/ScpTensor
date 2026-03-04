"""Plotting helpers for aggregation benchmark outputs."""

from __future__ import annotations

from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _ordered_methods(df: pd.DataFrame) -> list[str]:
    return [str(m) for m in df["method"].drop_duplicates().tolist()]


def plot_summary_metrics(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Create a multi-panel comparison of core benchmark metrics."""
    _ensure_parent(output_path)

    work = summary_df.copy()
    work["abs_bias"] = np.abs(work["bias"])

    panel_metrics: list[tuple[str, str, bool]] = [
        ("mae", "MAE (lower is better)", False),
        ("rmse", "RMSE (lower is better)", False),
        ("abs_bias", "|Bias| (lower is better)", False),
        ("species_overlap_auc_mean", "Pairwise AUC (higher is better)", True),
        ("changed_vs_background_auc", "Changed-vs-Background AUC", True),
        ("coverage_ratio", "Coverage Ratio (higher is better)", True),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(24, 12), constrained_layout=True)
    axes_flat = axes.ravel()

    for ax, (metric, title, higher_better) in zip(axes_flat, panel_metrics, strict=False):
        plot_df = work[["method", metric]].dropna().copy()
        if plot_df.empty:
            ax.set_visible(False)
            continue

        plot_df = plot_df.sort_values(metric, ascending=not higher_better)
        sns.barplot(data=plot_df, x="method", y=metric, ax=ax, color="#4C78A8")

        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Aggregation Method Benchmark Summary", fontsize=24)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_log2fc_distribution(
    protein_df: pd.DataFrame,
    output_path: Path,
    *,
    species_order: tuple[str, ...] = ("HUMAN", "YEAST", "ECOLI"),
) -> None:
    """Plot per-method log2FC distribution by species with expected-value overlays."""
    _ensure_parent(output_path)

    methods = _ordered_methods(protein_df)
    n_cols = 3
    n_rows = ceil(len(methods) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 6 * n_rows), sharey=True)
    axes_flat = np.asarray(axes).ravel()

    for ax, method in zip(axes_flat, methods, strict=False):
        sub = protein_df[protein_df["method"] == method]
        sns.boxplot(
            data=sub,
            x="species",
            y="log2_fc_ab",
            order=species_order,
            ax=ax,
            color="#72B7B2",
            fliersize=1,
        )
        for idx, sp in enumerate(species_order):
            exp_vals = sub.loc[sub["species"] == sp, "expected_log2_fc_ab"].dropna()
            if exp_vals.empty:
                continue
            ax.hlines(
                float(exp_vals.iloc[0]),
                idx - 0.3,
                idx + 0.3,
                colors="crimson",
                linestyles="--",
                linewidth=1.5,
            )

        ax.set_title(method)
        ax.set_xlabel("")
        ax.set_ylabel("Observed log2FC (A/B)")

    for ax in axes_flat[len(methods) :]:
        ax.set_visible(False)

    fig.suptitle("Species Log2FC Distribution by Aggregation Method", fontsize=24)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_observed_vs_expected(
    protein_df: pd.DataFrame,
    output_path: Path,
    *,
    species_order: tuple[str, ...] = ("HUMAN", "YEAST", "ECOLI"),
) -> None:
    """Scatter plot of observed vs expected log2FC for each aggregation method."""
    _ensure_parent(output_path)

    methods = _ordered_methods(protein_df)
    n_cols = 3
    n_rows = ceil(len(methods) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 6 * n_rows), sharex=True, sharey=True)
    axes_flat = np.asarray(axes).ravel()

    palette = {
        "HUMAN": "#4C78A8",
        "YEAST": "#F58518",
        "ECOLI": "#54A24B",
    }

    for panel_idx, (ax, method) in enumerate(zip(axes_flat, methods, strict=False)):
        sub = protein_df[protein_df["method"] == method]
        rng = np.random.default_rng(42 + panel_idx)

        for sp in species_order:
            block = sub[sub["species"] == sp]
            if block.empty:
                continue

            x = block["expected_log2_fc_ab"].to_numpy(dtype=np.float64)
            y = block["log2_fc_ab"].to_numpy(dtype=np.float64)
            jitter = rng.normal(0.0, 0.03, size=x.shape[0])
            ax.scatter(
                x + jitter,
                y,
                s=14,
                alpha=0.35,
                color=palette.get(sp, "#666666"),
                label=sp,
            )

        ax.plot([-2.5, 1.5], [-2.5, 1.5], linestyle="--", linewidth=1.2, color="black")
        ax.set_title(method)
        ax.set_xlabel("Expected log2FC (A/B)")
        ax.set_ylabel("Observed log2FC (A/B)")
        ax.set_xlim(-2.4, 1.4)

    for ax in axes_flat[len(methods) :]:
        ax.set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        unique: dict[str, object] = {}
        for handle, label in zip(handles, labels, strict=False):
            if label not in unique:
                unique[label] = handle
        fig.legend(unique.values(), unique.keys(), loc="upper right", frameon=True)

    fig.suptitle("Observed vs Expected Species Ratios", fontsize=24)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_cv_distribution(
    protein_df: pd.DataFrame,
    output_path: Path,
    *,
    species_order: tuple[str, ...] = ("HUMAN", "YEAST", "ECOLI"),
) -> None:
    """Plot replicate CV distributions across methods and species."""
    _ensure_parent(output_path)

    plot_df = protein_df[np.isfinite(protein_df["cv_mean"])].copy()
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(24, 8), constrained_layout=True)
    sns.boxplot(
        data=plot_df,
        x="method",
        y="cv_mean",
        hue="species",
        order=_ordered_methods(plot_df),
        hue_order=species_order,
        ax=ax,
        fliersize=1,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Replicate CV (mean of group A/B)")
    ax.set_title("Replicate Precision by Method and Species")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Species", loc="upper right")

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_metric_heatmap(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Plot normalized multi-metric score heatmap."""
    _ensure_parent(output_path)

    metrics = {
        "mae": False,
        "rmse": False,
        "cv_median_all": False,
        "species_overlap_auc_mean": True,
        "changed_vs_background_auc": True,
        "coverage_ratio": True,
    }

    work = summary_df.set_index("method")
    score_matrix = pd.DataFrame(index=work.index)

    for metric, higher_better in metrics.items():
        vals = work[metric].astype(float)
        finite = np.isfinite(vals.to_numpy(dtype=np.float64))
        if not finite.any():
            score_matrix[metric] = np.nan
            continue

        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
        if np.isclose(vmin, vmax):
            score_matrix[metric] = 1.0
            continue

        if higher_better:
            score_matrix[metric] = (vals - vmin) / (vmax - vmin)
        else:
            score_matrix[metric] = (vmax - vals) / (vmax - vmin)

    fig, ax = plt.subplots(figsize=(12, max(4, len(score_matrix) * 0.6)), constrained_layout=True)
    sns.heatmap(
        score_matrix,
        cmap="YlGnBu",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Normalized score (0-1, higher is better)"},
        ax=ax,
    )
    ax.set_title("Multi-Metric Method Ranking Heatmap")
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Method")

    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_species_coverage(
    species_summary_df: pd.DataFrame,
    output_path: Path,
    *,
    species_order: tuple[str, ...] = ("HUMAN", "YEAST", "ECOLI"),
) -> None:
    """Plot per-species quantification coverage for each method."""
    _ensure_parent(output_path)

    fig, ax = plt.subplots(figsize=(24, 8), constrained_layout=True)
    sns.barplot(
        data=species_summary_df,
        x="method",
        y="coverage_ratio",
        hue="species",
        order=_ordered_methods(species_summary_df),
        hue_order=species_order,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Coverage ratio")
    ax.set_ylim(0, 1.05)
    ax.set_title("Species-Specific Coverage by Aggregation Method")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Species", loc="upper right")

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
