"""Plotting helpers for normalization benchmark outputs."""

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


def plot_summary_metrics(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Plot selected metrics across methods and datasets."""
    _ensure_parent(output_path)
    work = summary_df.copy()

    panel_metrics: list[tuple[str, str]] = [
        ("coverage_ratio", "Coverage Ratio"),
        ("rle_mad_median", "RLE MAD (Lower Better)"),
        ("pairwise_wasserstein_median", "Pairwise Wasserstein (Lower Better)"),
        ("within_group_sd_median", "Within-Group SD (Lower Better)"),
        ("ratio_mae", "Ratio MAE (Lower Better)"),
        ("ratio_pairwise_auc_mean", "Species Pairwise AUC (Higher Better)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(24, 12), constrained_layout=True)
    axes_flat = axes.ravel()

    for ax, (metric, title) in zip(axes_flat, panel_metrics, strict=False):
        plot_df = work[["dataset", "method", metric]].dropna().copy()
        if plot_df.empty:
            ax.set_visible(False)
            continue

        sns.barplot(
            data=plot_df,
            x="method",
            y=metric,
            hue="dataset",
            ax=ax,
            errorbar=None,
        )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(title="Dataset", fontsize=10, title_fontsize=10)

    fig.suptitle("Normalization Benchmark Summary", fontsize=24)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_score_heatmap(score_df: pd.DataFrame, output_path: Path) -> None:
    """Plot per-dataset/per-method overall normalized score heatmap."""
    _ensure_parent(output_path)
    if score_df.empty:
        return

    pivot = score_df.pivot(index="dataset", columns="method", values="overall_score")
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(max(8, pivot.shape[1] * 1.2), 6), constrained_layout=True)
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
    ax.set_title("Method Ranking Heatmap by Dataset")
    ax.set_xlabel("Method")
    ax.set_ylabel("Dataset")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_overall_scores(score_df: pd.DataFrame, output_path: Path) -> None:
    """Plot mean overall score across datasets."""
    _ensure_parent(output_path)
    if score_df.empty:
        return

    agg = (
        score_df.groupby("method", as_index=False)["overall_score"]
        .mean()
        .sort_values("overall_score", ascending=False)
    )
    if agg.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    sns.barplot(data=agg, x="method", y="overall_score", ax=ax, color="#4C78A8")
    ax.set_title("Overall Method Score Across Datasets")
    ax.set_xlabel("")
    ax.set_ylabel("Mean normalized score (0-1)")
    ax.tick_params(axis="x", rotation=30)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_ratio_distributions(
    ratio_df: pd.DataFrame,
    output_path: Path,
    *,
    species_order: tuple[str, ...] = ("HUMAN", "YEAST", "ECOLI"),
) -> None:
    """Plot observed log2FC distributions by species for ratio-ground-truth datasets."""
    _ensure_parent(output_path)
    if ratio_df.empty:
        return

    work = ratio_df[ratio_df["species"].isin(species_order)].copy()
    if work.empty:
        return

    datasets = [str(ds) for ds in work["dataset"].drop_duplicates().tolist()]
    n_cols = 2
    n_rows = ceil(len(datasets) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(18, max(6, n_rows * 6)),
        sharey=True,
        constrained_layout=True,
    )
    axes_flat = np.asarray(axes).ravel()

    for ax, dataset in zip(axes_flat, datasets, strict=False):
        sub = work[work["dataset"] == dataset]
        sns.boxplot(
            data=sub,
            x="species",
            y="log2_fc_ab",
            hue="method",
            order=species_order,
            ax=ax,
            fliersize=1,
        )

        expected = (
            sub[["species", "expected_log2_fc_ab"]].dropna().drop_duplicates().set_index("species")
        )
        for idx, species in enumerate(species_order):
            if species not in expected.index:
                continue
            exp_val = float(expected.loc[species, "expected_log2_fc_ab"])
            ax.hlines(
                exp_val,
                idx - 0.35,
                idx + 0.35,
                colors="crimson",
                linestyles="--",
                linewidth=1.5,
            )

        ax.set_title(dataset)
        ax.set_xlabel("")
        ax.set_ylabel("Observed log2FC (A/B)")
        ax.legend_.set_title("Method")  # type: ignore[union-attr]

    for ax in axes_flat[len(datasets) :]:
        ax.set_visible(False)

    fig.suptitle("Observed vs Expected Species Ratios After Normalization", fontsize=22)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
