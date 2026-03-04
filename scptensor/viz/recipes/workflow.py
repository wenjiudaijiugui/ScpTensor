"""Workflow-oriented visualization recipes.

These helpers provide quick, stage-level visual summaries for a typical
ScpTensor analysis workflow:

1. Data loading overview
2. QC filtering impact
3. Preprocessing + imputation changes
4. Reduction + clustering summaries
5. Saved artifact sizes
6. Recent operation history
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

from scptensor.core.structures import ScpContainer
from scptensor.viz.base.style import setup_style
from scptensor.viz.base.validation import validate_container, validate_layer

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = [
    "plot_data_overview",
    "plot_qc_filtering_summary",
    "plot_preprocessing_summary",
    "plot_missingness_reduction",
    "plot_reduction_summary",
    "plot_embedding_panels",
    "plot_saved_artifact_sizes",
    "plot_recent_operations",
]


def _to_dense_array(x: np.ndarray | sp.spmatrix) -> np.ndarray:
    """Convert matrix-like input to dense NumPy array."""
    if sp.issparse(x):
        return x.toarray()  # type: ignore[no-any-return]
    return np.asarray(x)


def _detected_mask(x: np.ndarray | sp.spmatrix) -> np.ndarray:
    """Get dense detected-value mask using QC-consistent definition."""
    dense = _to_dense_array(x)
    return (dense > 0) & np.isfinite(dense)


def _sample_finite_values(
    x: np.ndarray | sp.spmatrix, max_points: int = 200_000, seed: int = 42
) -> np.ndarray:
    """Sample finite values for stable histogram rendering."""
    dense = _to_dense_array(x)
    vals = dense[np.isfinite(dense)]
    if vals.size > max_points:
        rng = np.random.default_rng(seed)
        vals = rng.choice(vals, size=max_points, replace=False)
    return vals


def _missing_rate(x: np.ndarray | sp.spmatrix) -> float:
    """Compute missing rate as non-finite proportion."""
    dense = _to_dense_array(x)
    return float(np.mean(~np.isfinite(dense)))


def plot_data_overview(
    container: ScpContainer,
    assay_name: str = "proteins",
    layer: str = "raw",
    groupby: str = "cell_cycle",
    max_points: int = 200_000,
    figsize: tuple[float, float] = (15, 4),
) -> np.ndarray:
    """Plot loading-stage overview with distributions and sample composition."""
    validate_container(container)
    validate_layer(container, assay_name, layer)
    setup_style()

    x = container.assays[assay_name].layers[layer].X
    sampled_vals = _sample_finite_values(x, max_points=max_points)
    detected = _detected_mask(x)

    sample_detection = np.mean(detected, axis=1)
    feature_missing = 1.0 - np.mean(detected, axis=0)

    _, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].hist(sampled_vals, bins=60, color="#4C72B0", alpha=0.85)
    axes[0].set_title("Raw Intensity Distribution")
    axes[0].set_xlabel("Intensity")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(sample_detection, bins=40, color="#55A868", alpha=0.85)
    axes[1].set_title("Per-Sample Detection Rate")
    axes[1].set_xlabel("Detection rate")
    axes[1].set_ylabel("# Samples")

    if groupby in container.obs.columns:
        counts = container.obs[groupby].value_counts().sort("count", descending=True)
        axes[2].bar(
            counts[groupby].cast(str).to_list(),
            counts["count"].to_numpy(),
            color="#C44E52",
        )
        axes[2].set_title(f"{groupby} Composition")
        axes[2].set_xlabel(groupby)
        axes[2].set_ylabel("# Samples")
        axes[2].tick_params(axis="x", rotation=30)
    else:
        axes[2].hist(feature_missing, bins=40, color="#C44E52", alpha=0.85)
        axes[2].set_title("Per-Feature Missing Rate")
        axes[2].set_xlabel("Missing rate")
        axes[2].set_ylabel("# Features")

    plt.tight_layout()
    return axes


def plot_qc_filtering_summary(
    container_before: ScpContainer,
    container_after: ScpContainer,
    assay_name: str = "proteins",
    layer: str = "raw",
    min_features: float | None = None,
    max_missing_rate: float | None = None,
    sample_qc_col: str | None = None,
    feature_missing_col: str = "missing_rate",
    figsize: tuple[float, float] = (15, 4),
) -> np.ndarray:
    """Plot QC filtering impact across samples and features."""
    validate_container(container_before)
    validate_container(container_after)
    validate_layer(container_before, assay_name, layer)
    validate_layer(container_after, assay_name, layer)
    setup_style()

    sample_qc_col = sample_qc_col or f"n_features_{assay_name}"
    if sample_qc_col in container_before.obs.columns:
        sample_qc_before = container_before.obs[sample_qc_col].to_numpy()
    else:
        sample_qc_before = np.sum(
            _detected_mask(container_before.assays[assay_name].layers[layer].X), axis=1
        )

    before_samples = container_before.n_samples
    after_samples = container_after.n_samples
    before_features = container_before.assays[assay_name].n_features
    after_features = container_after.assays[assay_name].n_features

    if feature_missing_col in container_before.assays[assay_name].var.columns:
        feature_missing_before = (
            container_before.assays[assay_name].var[feature_missing_col].to_numpy()
        )
    else:
        detected_before = _detected_mask(container_before.assays[assay_name].layers[layer].X)
        feature_missing_before = 1.0 - np.mean(detected_before, axis=0)

    if feature_missing_col in container_after.assays[assay_name].var.columns:
        feature_missing_after = (
            container_after.assays[assay_name].var[feature_missing_col].to_numpy()
        )
    else:
        detected_after = _detected_mask(container_after.assays[assay_name].layers[layer].X)
        feature_missing_after = 1.0 - np.mean(detected_after, axis=0)

    _, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].hist(sample_qc_before, bins=40, color="#4C72B0", alpha=0.85)
    if min_features is not None:
        axes[0].axvline(min_features, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_title("n_features per Sample (Before Filter)")
    axes[0].set_xlabel(sample_qc_col)
    axes[0].set_ylabel("# Samples")

    x = np.arange(2)
    labels = ["Samples", "Features"]
    axes[1].bar(x - 0.18, [before_samples, before_features], width=0.36, color="#8172B2")
    axes[1].bar(x + 0.18, [after_samples, after_features], width=0.36, color="#64B5CD")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_title("QC Filtering Impact")
    axes[1].set_ylabel("Count")
    axes[1].legend(["Before", "After"])

    axes[2].hist(feature_missing_before, bins=40, alpha=0.55, label="Before", color="#C44E52")
    axes[2].hist(feature_missing_after, bins=40, alpha=0.55, label="After", color="#55A868")
    if max_missing_rate is not None:
        axes[2].axvline(max_missing_rate, color="red", linestyle="--", linewidth=1.5)
    axes[2].set_title("Feature Missing Rate")
    axes[2].set_xlabel(feature_missing_col)
    axes[2].set_ylabel("# Features")
    axes[2].legend()

    plt.tight_layout()
    return axes


def plot_preprocessing_summary(
    container: ScpContainer,
    assay_name: str = "proteins",
    raw_layer: str = "raw",
    transformed_layers: Sequence[str] = ("log2", "norm", "imputed"),
    max_points: int = 150_000,
    figsize: tuple[float, float] = (15, 4),
) -> np.ndarray:
    """Plot distribution and sample-median changes across preprocessing layers."""
    validate_container(container)
    validate_layer(container, assay_name, raw_layer)
    for layer in transformed_layers:
        validate_layer(container, assay_name, layer)
    setup_style()

    assay = container.assays[assay_name]

    _, axes = plt.subplots(1, 3, figsize=figsize)

    x_raw = assay.layers[raw_layer].X
    axes[0].hist(
        _sample_finite_values(x_raw, max_points=max_points),
        bins=60,
        color="#4C72B0",
        alpha=0.85,
    )
    axes[0].set_title(f"{raw_layer} Distribution")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Frequency")

    palette = ["#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]
    medians_by_layer: list[np.ndarray] = []
    for i, layer in enumerate(transformed_layers):
        x = assay.layers[layer].X
        axes[1].hist(
            _sample_finite_values(x, max_points=max_points),
            bins=60,
            alpha=0.45,
            label=layer,
            color=palette[i % len(palette)],
        )
        dense_x = _to_dense_array(x)
        medians_by_layer.append(np.nanmedian(dense_x, axis=1))

    axes[1].set_title("Processed Layers Distribution")
    axes[1].set_xlabel("Value")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()

    axes[2].boxplot(medians_by_layer, tick_labels=list(transformed_layers), patch_artist=True)
    axes[2].set_title("Per-Sample Median")
    axes[2].set_ylabel("Median value")

    plt.tight_layout()
    return axes


def plot_missingness_reduction(
    container: ScpContainer,
    assay_name: str = "proteins",
    before_layer: str = "norm",
    after_layer: str = "imputed",
    figsize: tuple[float, float] = (4.5, 3.5),
    ax: Axes | None = None,
) -> Axes:
    """Plot missing-rate change between two layers."""
    validate_container(container)
    validate_layer(container, assay_name, before_layer)
    validate_layer(container, assay_name, after_layer)
    setup_style()

    before_rate = _missing_rate(container.assays[assay_name].layers[before_layer].X)
    after_rate = _missing_rate(container.assays[assay_name].layers[after_layer].X)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    rates = [before_rate, after_rate]
    labels = [f"Before ({before_layer})", f"After ({after_layer})"]
    ax.bar(labels, rates, color=["#C44E52", "#55A868"])
    ax.set_ylabel("Missing rate")
    ax.set_title("Missingness Reduction")
    for i, val in enumerate(rates):
        ax.text(i, val + 0.005, f"{val:.2%}", ha="center", va="bottom")

    return ax


def plot_reduction_summary(
    container: ScpContainer,
    pca_assay_name: str = "pca",
    cluster_col: str = "kmeans_cluster",
    explained_var_col: str = "explained_variance_ratio",
    figsize: tuple[float, float] = (12, 4),
) -> np.ndarray:
    """Plot PCA explained variance and cluster-size distribution."""
    validate_container(container)
    setup_style()

    _, axes = plt.subplots(1, 2, figsize=figsize)

    if (
        pca_assay_name in container.assays
        and explained_var_col in container.assays[pca_assay_name].var.columns
    ):
        ratios = container.assays[pca_assay_name].var[explained_var_col].to_numpy()
        cumulative = np.cumsum(ratios)
        axes[0].plot(
            np.arange(1, len(cumulative) + 1),
            cumulative,
            marker="o",
            markersize=3,
            color="#4C72B0",
        )
        axes[0].axhline(0.8, color="red", linestyle="--", linewidth=1.2)
        axes[0].set_title("PCA Cumulative Explained Variance")
        axes[0].set_xlabel("# Components")
        axes[0].set_ylabel("Cumulative ratio")
    else:
        axes[0].axis("off")
        axes[0].text(0.5, 0.5, "No PCA variance ratio available", ha="center", va="center")

    if cluster_col in container.obs.columns:
        counts = container.obs[cluster_col].value_counts().sort(cluster_col)
        axes[1].bar(
            counts[cluster_col].cast(str).to_list(),
            counts["count"].to_numpy(),
            color="#55A868",
        )
        axes[1].set_title("Cluster Size Distribution")
        axes[1].set_xlabel(cluster_col)
        axes[1].set_ylabel("# Samples")
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, f"Column '{cluster_col}' not found in obs", ha="center", va="center")

    plt.tight_layout()
    return axes


def plot_embedding_panels(
    container: ScpContainer,
    assay_names: Sequence[str] = ("pca", "umap"),
    layer: str = "X",
    color_by: str | None = "kmeans_cluster",
    figsize: tuple[float, float] = (12, 5),
) -> np.ndarray:
    """Plot 2D embeddings directly from reduced assays (e.g., PCA/UMAP)."""
    validate_container(container)
    setup_style()

    valid_assays = [
        a for a in assay_names if a in container.assays and layer in container.assays[a].layers
    ]
    if not valid_assays:
        raise ValueError(
            f"No valid embedding assays found. Requested={list(assay_names)}, layer='{layer}'."
        )

    _, axes = plt.subplots(1, len(valid_assays), figsize=figsize)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    color_values: np.ndarray | str = "#4C72B0"
    is_numeric_color = False
    category_mapping: dict[str, int] | None = None
    if color_by is not None and color_by in container.obs.columns:
        raw_color = container.obs[color_by].to_numpy()
        if np.issubdtype(raw_color.dtype, np.number):
            color_values = raw_color
            is_numeric_color = True
        else:
            uniq = np.unique(raw_color)
            mapper = {str(label): i for i, label in enumerate(uniq)}
            color_values = np.array([mapper[str(v)] for v in raw_color], dtype=float)
            category_mapping = mapper

    scatter_ref = None
    for ax, assay_name in zip(axes, valid_assays, strict=False):
        emb = _to_dense_array(container.assays[assay_name].layers[layer].X)
        if emb.shape[1] < 2:
            raise ValueError(
                f"Assay '{assay_name}/{layer}' must have at least 2 dimensions, got {emb.shape[1]}."
            )
        scatter_ref = ax.scatter(emb[:, 0], emb[:, 1], c=color_values, s=12, cmap="tab10")
        ax.set_title(f"{assay_name.upper()} (colored by {color_by or 'default'})")
        ax.set_xlabel(f"{assay_name.upper()}1")
        ax.set_ylabel(f"{assay_name.upper()}2")

    if (
        color_by is not None
        and color_by in container.obs.columns
        and is_numeric_color
        and scatter_ref is not None
    ):
        axes[0].figure.colorbar(
            scatter_ref, ax=axes.tolist(), fraction=0.03, pad=0.02, label=color_by
        )
    elif category_mapping:
        from matplotlib.patches import Patch

        cmap = plt.get_cmap("tab10")
        legend_items = [
            Patch(facecolor=cmap(code), label=label)
            for label, code in sorted(category_mapping.items())
        ]
        axes[0].legend(handles=legend_items, title=color_by, loc="best")

    plt.tight_layout()
    return axes


def plot_saved_artifact_sizes(
    paths: Sequence[str | Path],
    figsize: tuple[float, float] = (6, 4),
    ax: Axes | None = None,
) -> Axes:
    """Plot file sizes for saved artifacts."""
    setup_style()
    resolved = [Path(p) for p in paths]
    if not resolved:
        raise ValueError("paths cannot be empty")

    sizes_mb = [p.stat().st_size / (1024 * 1024) for p in resolved]
    labels = [p.suffix.upper().lstrip(".") or p.name for p in resolved]

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.bar(labels, sizes_mb, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"][: len(labels)])
    ax.set_ylabel("File size (MB)")
    ax.set_title("Saved Artifact Size")
    for i, val in enumerate(sizes_mb):
        ax.text(i, val + 0.01, f"{val:.2f} MB", ha="center", va="bottom")

    return ax


def plot_recent_operations(
    container: ScpContainer,
    n_recent: int = 12,
    figsize: tuple[float, float] = (7, 4),
    ax: Axes | None = None,
) -> Axes:
    """Plot frequency of recent operation history entries."""
    validate_container(container)
    setup_style()

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    recent_logs = container.history[-n_recent:]
    if not recent_logs:
        ax.text(0.5, 0.5, "No history available", ha="center", va="center")
        ax.set_axis_off()
        return ax

    actions = np.array([log.action for log in recent_logs], dtype=object)
    unique_actions, counts = np.unique(actions, return_counts=True)
    order = np.argsort(counts)[::-1]
    unique_actions = unique_actions[order]
    counts = counts[order]

    ax.barh(unique_actions.astype(str), counts, color="#8172B2")
    ax.set_title("Recent Operation Frequency")
    ax.set_xlabel("Count")
    ax.invert_yaxis()

    return ax
