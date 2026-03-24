"""Plotting helpers for imputation benchmark outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_overall_scores(scores_df: pd.DataFrame, output_path: Path) -> None:
    """Bar plot of overall score per method (faceted by dataset)."""
    _ensure_parent(output_path)
    if scores_df.empty:
        return

    plot_df = scores_df.copy()
    group_cols = ["dataset", "method"]
    if "holdout_state" in plot_df.columns:
        group_cols.insert(1, "holdout_state")
    plot_df = (
        plot_df.groupby(group_cols, as_index=False)["overall_score"]
        .mean(numeric_only=True)
        .sort_values(group_cols)
    )

    facet_kwargs: dict[str, object] = {"col": "dataset"}
    if "holdout_state" in plot_df.columns:
        facet_kwargs["row"] = "holdout_state"

    fig = sns.catplot(
        data=plot_df,
        x="method",
        y="overall_score",
        kind="bar",
        sharey=False,
        height=4.5,
        aspect=1.6,
        color="#4C78A8",
        **facet_kwargs,
    )
    fig.set_axis_labels("Method", "Overall score (0-1)")
    if "holdout_state" in plot_df.columns:
        fig.set_titles("{row_name} | {col_name}")
    else:
        fig.set_titles("{col_name}")
    for ax in fig.axes.flat:
        ax.tick_params(axis="x", rotation=35)
    fig.figure.suptitle("Imputation Method Overall Scores", y=1.05, fontsize=20)
    fig.figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig.figure)


def plot_metric_heatmap(scores_df: pd.DataFrame, output_path: Path) -> None:
    """Heatmap of per-metric normalized scores for each method."""
    _ensure_parent(output_path)
    if scores_df.empty:
        return

    score_cols = [c for c in scores_df.columns if c.startswith("score_")]
    if not score_cols:
        return

    work = scores_df.copy()
    group_cols = ["dataset", "method"]
    if "holdout_state" in work.columns:
        group_cols.insert(1, "holdout_state")
    work = work.groupby(group_cols, as_index=False)[score_cols].mean(numeric_only=True)

    if "holdout_state" in work.columns:
        work["dataset_method"] = (
            work["dataset"].astype(str)
            + " | "
            + work["holdout_state"].astype(str)
            + " | "
            + work["method"].astype(str)
        )
    else:
        work["dataset_method"] = work["dataset"].astype(str) + " | " + work["method"].astype(str)
    matrix = work.set_index("dataset_method")[score_cols]

    fig, ax = plt.subplots(
        figsize=(max(10, len(score_cols) * 1.2), max(5, matrix.shape[0] * 0.4)),
        constrained_layout=True,
    )
    sns.heatmap(
        matrix,
        cmap="YlGnBu",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Normalized score (higher is better)"},
        ax=ax,
    )
    ax.set_title("Imputation Benchmark Score Heatmap")
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Dataset | Method")

    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_nrmse_curves(raw_df: pd.DataFrame, output_path: Path, top_n: int = 8) -> None:
    """NRMSE vs holdout rate curves for top methods per dataset."""
    _ensure_parent(output_path)
    if raw_df.empty or "nrmse" not in raw_df.columns:
        return

    plot_df = raw_df.copy()
    plot_df = plot_df[np.isfinite(plot_df["nrmse"].to_numpy(dtype=np.float64))]
    if plot_df.empty:
        return

    rank_group_cols = ["dataset", "method"]
    if "holdout_state" in plot_df.columns:
        rank_group_cols.insert(1, "holdout_state")
    rank_df = (
        plot_df.groupby(rank_group_cols, as_index=False)["nrmse"]
        .mean()
        .sort_values(
            rank_group_cols[:-1] + ["nrmse"], ascending=[True] * (len(rank_group_cols) - 1) + [True]
        )
    )
    keep_by = ["dataset"]
    if "holdout_state" in rank_df.columns:
        keep_by.append("holdout_state")
    keep_pairs = rank_df.groupby(keep_by).head(top_n)[rank_group_cols]
    keep_set = {
        tuple(str(getattr(r, col)) for col in rank_group_cols)
        for r in keep_pairs.itertuples(index=False)
    }
    mask = plot_df.apply(
        lambda r: tuple(str(r[col]) for col in rank_group_cols) in keep_set,
        axis=1,
    )
    plot_df = plot_df[mask]

    facet_kwargs: dict[str, object] = {"col": "dataset"}
    if "holdout_state" in plot_df.columns:
        facet_kwargs["row"] = "holdout_state"

    fig = sns.relplot(
        data=plot_df,
        x="holdout_rate",
        y="nrmse",
        hue="method",
        style="mechanism",
        kind="line",
        marker="o",
        height=4.5,
        aspect=1.4,
        facet_kws={"sharey": False},
        **facet_kwargs,
    )
    fig.set_axis_labels("Holdout rate", "NRMSE (lower is better)")
    if "holdout_state" in plot_df.columns:
        fig.set_titles("{row_name} | {col_name}")
    else:
        fig.set_titles("{col_name}")
    fig.figure.suptitle("Masked-Value Recovery Curves", y=1.05, fontsize=20)
    fig.figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig.figure)


def plot_runtime_vs_accuracy(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Scatter plot of runtime vs NRMSE."""
    _ensure_parent(output_path)
    if (
        summary_df.empty
        or "runtime_sec" not in summary_df.columns
        or "nrmse" not in summary_df.columns
    ):
        return

    plot_df = summary_df.copy()
    plot_df = plot_df[
        np.isfinite(plot_df["runtime_sec"].to_numpy(dtype=np.float64))
        & np.isfinite(plot_df["nrmse"].to_numpy(dtype=np.float64))
    ]
    if plot_df.empty:
        return

    group_cols = ["dataset", "method"]
    if "holdout_state" in plot_df.columns:
        group_cols.insert(1, "holdout_state")
    plot_df = plot_df.groupby(group_cols, as_index=False)[["runtime_sec", "nrmse"]].mean(
        numeric_only=True
    )

    facet_kwargs: dict[str, object] = {"col": "dataset"}
    if "holdout_state" in plot_df.columns:
        facet_kwargs["row"] = "holdout_state"

    fig = sns.relplot(
        data=plot_df,
        x="runtime_sec",
        y="nrmse",
        hue="method",
        kind="scatter",
        height=4.5,
        aspect=1.5,
        facet_kws={"sharey": False},
        **facet_kwargs,
    )
    fig.set_axis_labels("Runtime (sec)", "NRMSE (lower is better)")
    if "holdout_state" in plot_df.columns:
        fig.set_titles("{row_name} | {col_name}")
    else:
        fig.set_titles("{col_name}")
    fig.figure.suptitle("Accuracy-Speed Tradeoff", y=1.05, fontsize=20)
    fig.figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig.figure)
