"""Plotting helpers for integration benchmark outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_summary_metrics(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Plot key batch-removal and bio-conservation metrics."""
    _ensure_parent(output_path)
    if summary_df.empty:
        return

    panel_metrics: list[tuple[str, str]] = [
        ("between_batch_ratio", "Between-Batch Ratio (Lower Better)"),
        ("batch_asw", "Batch ASW (Lower Better)"),
        ("batch_mixing", "Batch Mixing (Higher Better)"),
        ("lisi_approx", "Approx. LISI (Higher Better)"),
        ("condition_asw", "Condition ASW (Higher Better)"),
        ("condition_ari", "Condition ARI (Higher Better)"),
        ("condition_nmi", "Condition NMI (Higher Better)"),
        ("condition_knn_purity", "Condition kNN Purity (Higher Better)"),
        ("runtime_sec", "Runtime (sec, Lower Better)"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(24, 16), constrained_layout=True)
    axes_flat = axes.ravel()

    for ax, (metric, title) in zip(axes_flat, panel_metrics, strict=False):
        if metric not in summary_df.columns:
            ax.set_visible(False)
            continue

        plot_df = summary_df[["scenario", "method", metric]].dropna().copy()
        if plot_df.empty:
            ax.set_visible(False)
            continue

        sns.barplot(
            data=plot_df,
            x="method",
            y=metric,
            hue="scenario",
            errorbar=None,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(title="Scenario", fontsize=9, title_fontsize=10)

    fig.suptitle("Integration Benchmark Summary", fontsize=24)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_score_heatmap(scores_df: pd.DataFrame, output_path: Path) -> None:
    """Heatmap of overall normalized scores by dataset/scenario and method."""
    _ensure_parent(output_path)
    if scores_df.empty or "overall_score" not in scores_df.columns:
        return

    work = scores_df.copy()
    if "scenario" in work.columns:
        work["dataset_scenario"] = work["dataset"].astype(str) + " | " + work["scenario"].astype(str)
        index_col = "dataset_scenario"
    else:
        index_col = "dataset"

    pivot = work.pivot(index=index_col, columns="method", values="overall_score")
    if pivot.empty:
        return

    fig, ax = plt.subplots(
        figsize=(max(8, pivot.shape[1] * 1.2), max(5, pivot.shape[0] * 0.9)),
    )
    sns.heatmap(
        pivot,
        cmap="YlGnBu",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Overall normalized score (0-1)"},
        ax=ax,
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_title("Integration Method Ranking Heatmap")
    ax.set_xlabel("Method")
    ax.set_ylabel(index_col.replace("_", " ").title())
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_overall_scores(scores_df: pd.DataFrame, output_path: Path) -> None:
    """Bar plot of mean overall score per method."""
    _ensure_parent(output_path)
    if scores_df.empty or "overall_score" not in scores_df.columns:
        return

    agg = (
        scores_df.groupby("method", as_index=False)["overall_score"]
        .mean()
        .sort_values("overall_score", ascending=False)
    )
    if agg.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    sns.barplot(data=agg, x="method", y="overall_score", color="#4C78A8", ax=ax)
    ax.set_title("Integration Method Overall Scores")
    ax.set_xlabel("")
    ax.set_ylabel("Mean normalized score (0-1)")
    ax.tick_params(axis="x", rotation=30)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


__all__ = ["plot_summary_metrics", "plot_score_heatmap", "plot_overall_scores"]
