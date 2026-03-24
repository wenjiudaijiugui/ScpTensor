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
from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

from scptensor.core._structure_container import ScpContainer
from scptensor.viz.base.style import setup_style
from scptensor.viz.base.validation import validate_container, validate_layer

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = [
    "plot_aggregation_summary",
    "plot_data_overview",
    "plot_normalization_summary",
    "plot_qc_filtering_summary",
    "plot_preprocessing_summary",
    "plot_missingness_reduction",
    "plot_integration_batch_summary",
    "plot_reduction_summary",
    "plot_embedding_panels",
    "plot_saved_artifact_sizes",
    "plot_recent_operations",
]


def _to_dense_array(x: np.ndarray | sp.spmatrix) -> np.ndarray:
    """Convert matrix-like input to dense NumPy array."""
    if sp.issparse(x):
        return cast("sp.spmatrix", x).toarray()
    return np.asarray(x)


def _detected_mask(x: np.ndarray | sp.spmatrix) -> np.ndarray:
    """Get dense detected-value mask using QC-consistent definition."""
    dense = _to_dense_array(x)
    return (dense > 0) & np.isfinite(dense)


def _sample_finite_values(
    x: np.ndarray | sp.spmatrix,
    max_points: int = 200_000,
    seed: int = 42,
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


def _safe_row_cv(x: np.ndarray | sp.spmatrix) -> np.ndarray:
    """Compute per-sample CV with robust finite-value handling."""
    dense = _to_dense_array(x)
    cvs = np.full(dense.shape[0], np.nan, dtype=np.float64)
    for i in range(dense.shape[0]):
        row = dense[i]
        finite = np.isfinite(row)
        if np.sum(finite) < 2:
            continue
        mu = float(np.mean(row[finite]))
        if np.isclose(mu, 0.0):
            continue
        sigma = float(np.std(row[finite], ddof=1))
        cvs[i] = sigma / abs(mu)
    return cvs


def _impute_finite_column_mean(x: np.ndarray) -> np.ndarray:
    """Fill NaN/inf values with column means for stable projection/metrics."""
    out = np.asarray(x, dtype=np.float64).copy()
    finite = np.isfinite(out)
    if np.all(finite):
        return out

    col_means = np.zeros(out.shape[1], dtype=np.float64)
    for j in range(out.shape[1]):
        col = out[:, j]
        mask = np.isfinite(col)
        col_means[j] = float(np.mean(col[mask])) if np.any(mask) else 0.0

    bad_rows, bad_cols = np.where(~finite)
    out[bad_rows, bad_cols] = col_means[bad_cols]
    return out


def _batch_dispersion_score(x_embed: np.ndarray, batch_codes: np.ndarray) -> float:
    """Return 1 - between/(between+within) dispersion ratio in [0, 1]."""
    global_mean = np.mean(x_embed, axis=0)
    between = 0.0
    within = 0.0
    for b in np.unique(batch_codes):
        mask = batch_codes == b
        xb = x_embed[mask]
        if xb.size == 0:
            continue
        mean_b = np.mean(xb, axis=0)
        between += float(xb.shape[0] * np.sum((mean_b - global_mean) ** 2))
        within += float(np.sum((xb - mean_b) ** 2))

    total = between + within + 1e-12
    return float(np.clip(1.0 - (between / total), 0.0, 1.0))


def _compute_batch_quality_metrics(
    x_embed: np.ndarray,
    batch_labels: np.ndarray,
) -> dict[str, float]:
    """Compute batch-mixing quality metrics in a higher-is-better direction."""
    from scptensor.autoselect.metrics.batch import batch_asw, batch_mixing_score

    _, encoded = np.unique(batch_labels.astype(str), return_inverse=True)
    batch_codes = encoded.astype(np.int64)
    return {
        "batch_asw_mix": float(batch_asw(x_embed, batch_codes)),
        "batch_mixing_knn": float(batch_mixing_score(x_embed, batch_codes)),
        "batch_dispersion": float(_batch_dispersion_score(x_embed, batch_codes)),
    }


def plot_aggregation_summary(
    container: ScpContainer,
    source_assay: str = "peptides",
    target_assay: str = "proteins",
    top_n_targets: int = 15,
    figsize: tuple[float, float] = (15, 4),
) -> np.ndarray:
    """Visualize peptide/precursor -> protein aggregation coverage and density."""
    validate_container(container)
    setup_style()

    if source_assay not in container.assays:
        raise ValueError(f"Source assay '{source_assay}' not found.")
    if target_assay not in container.assays:
        raise ValueError(f"Target assay '{target_assay}' not found.")

    link = next(
        (
            lk
            for lk in reversed(container.links)
            if lk.source_assay == source_assay and lk.target_assay == target_assay
        ),
        None,
    )
    if link is None:
        raise ValueError(
            "No aggregation link found for requested assays. "
            "Run aggregate_to_protein first or provide matching source/target assay names.",
        )

    linkage = link.linkage
    if linkage.height == 0:
        raise ValueError("Aggregation link is empty; cannot visualize aggregation summary.")

    peptides_per_target = linkage.group_by("target_id").len().rename({"len": "n_source"})
    source_total = container.assays[source_assay].n_features
    target_total = container.assays[target_assay].n_features
    mapped_source = int(linkage.select("source_id").n_unique())
    mapped_target = int(linkage.select("target_id").n_unique())

    _, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].bar(
        ["Source features", "Mapped source", "Mapped target", "Target features"],
        [source_total, mapped_source, mapped_target, target_total],
        color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"],
    )
    axes[0].set_title("Aggregation Coverage")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=15)

    counts = peptides_per_target["n_source"].to_numpy()
    axes[1].hist(
        counts,
        bins=min(40, max(8, int(np.sqrt(counts.size)))),
        color="#64B5CD",
        alpha=0.9,
    )
    axes[1].set_title("Sources per Target Distribution")
    axes[1].set_xlabel("# source features per target")
    axes[1].set_ylabel("# targets")

    top_tbl = peptides_per_target.sort("n_source", descending=True).head(max(1, top_n_targets))
    axes[2].bar(
        top_tbl["target_id"].cast(str).to_list(),
        top_tbl["n_source"].to_numpy(),
        color="#CCB974",
    )
    axes[2].set_title(f"Top {top_tbl.height} Targets by Source Count")
    axes[2].set_xlabel("Target ID")
    axes[2].set_ylabel("# source features")
    axes[2].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    return axes


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
            _detected_mask(container_before.assays[assay_name].layers[layer].X),
            axis=1,
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


def plot_normalization_summary(
    container: ScpContainer,
    assay_name: str = "proteins",
    before_layer: str = "raw",
    after_layer: str = "norm",
    max_points: int = 150_000,
    figsize: tuple[float, float] = (15, 4),
) -> np.ndarray:
    """Plot normalization-focused diagnostics (shift, stability, distribution)."""
    validate_container(container)
    validate_layer(container, assay_name, before_layer)
    validate_layer(container, assay_name, after_layer)
    setup_style()

    x_before = container.assays[assay_name].layers[before_layer].X
    x_after = container.assays[assay_name].layers[after_layer].X

    before_dense = _to_dense_array(x_before)
    after_dense = _to_dense_array(x_after)

    before_median = np.nanmedian(before_dense, axis=1)
    after_median = np.nanmedian(after_dense, axis=1)

    cv_before = _safe_row_cv(x_before)
    cv_after = _safe_row_cv(x_after)

    _, axes = plt.subplots(1, 3, figsize=figsize)

    min_lim = float(np.nanmin(np.concatenate([before_median, after_median])))
    max_lim = float(np.nanmax(np.concatenate([before_median, after_median])))
    axes[0].scatter(before_median, after_median, s=12, alpha=0.8, color="#4C72B0")
    axes[0].plot([min_lim, max_lim], [min_lim, max_lim], linestyle="--", color="red", linewidth=1)
    axes[0].set_title("Per-sample Median Shift")
    axes[0].set_xlabel(f"Before ({before_layer})")
    axes[0].set_ylabel(f"After ({after_layer})")

    finite_before = cv_before[np.isfinite(cv_before)]
    finite_after = cv_after[np.isfinite(cv_after)]
    axes[1].boxplot(
        [finite_before, finite_after],
        tick_labels=[before_layer, after_layer],
        patch_artist=True,
        boxprops={"facecolor": "#A1C9F4"},
        medianprops={"color": "black"},
    )
    axes[1].set_title("Per-sample CV")
    axes[1].set_ylabel("Coefficient of variation")

    axes[2].hist(
        _sample_finite_values(x_before, max_points=max_points),
        bins=60,
        alpha=0.5,
        label=before_layer,
        color="#C44E52",
    )
    axes[2].hist(
        _sample_finite_values(x_after, max_points=max_points),
        bins=60,
        alpha=0.5,
        label=after_layer,
        color="#55A868",
    )
    axes[2].set_title("Global Distribution Overlap")
    axes[2].set_xlabel("Value")
    axes[2].set_ylabel("Frequency")
    axes[2].legend()

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


def plot_integration_batch_summary(
    container_before: ScpContainer,
    container_after: ScpContainer,
    assay_name: str = "proteins",
    before_layer: str = "norm",
    after_layer: str = "integrated",
    batch_key: str = "batch",
    figsize: tuple[float, float] = (15, 4.5),
) -> np.ndarray:
    """Plot batch-correction quality before/after with embedding and metrics."""
    from sklearn.decomposition import PCA

    validate_container(container_before)
    validate_container(container_after)
    validate_layer(container_before, assay_name, before_layer)
    validate_layer(container_after, assay_name, after_layer)
    setup_style()

    if batch_key not in container_before.obs.columns:
        raise ValueError(f"Batch key '{batch_key}' not found in container_before.obs.")
    if batch_key not in container_after.obs.columns:
        raise ValueError(f"Batch key '{batch_key}' not found in container_after.obs.")

    batch_before = container_before.obs[batch_key].to_numpy()
    batch_after = container_after.obs[batch_key].to_numpy()
    if len(batch_before) != len(batch_after):
        raise ValueError("container_before and container_after have different sample counts.")

    x_before = _impute_finite_column_mean(
        _to_dense_array(container_before.assays[assay_name].layers[before_layer].X),
    )
    x_after = _impute_finite_column_mean(
        _to_dense_array(container_after.assays[assay_name].layers[after_layer].X),
    )

    z_before = PCA(n_components=2, random_state=42).fit_transform(x_before)
    z_after = PCA(n_components=2, random_state=42).fit_transform(x_after)

    uniq_labels = np.unique(batch_before.astype(str))
    label_to_code = {label: i for i, label in enumerate(uniq_labels)}
    code_before = np.array([label_to_code[str(v)] for v in batch_before], dtype=np.int64)
    code_after = np.array([label_to_code.get(str(v), -1) for v in batch_after], dtype=np.int64)

    metrics_before = _compute_batch_quality_metrics(z_before, batch_before.astype(str))
    metrics_after = _compute_batch_quality_metrics(z_after, batch_after.astype(str))

    _, axes = plt.subplots(1, 3, figsize=figsize)
    cmap = plt.get_cmap("tab10")

    axes[0].scatter(z_before[:, 0], z_before[:, 1], c=code_before, cmap=cmap, s=12, alpha=0.85)
    axes[0].set_title(f"Before Integration ({before_layer})")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    axes[1].scatter(z_after[:, 0], z_after[:, 1], c=code_after, cmap=cmap, s=12, alpha=0.85)
    axes[1].set_title(f"After Integration ({after_layer})")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")

    metric_names = list(metrics_before.keys())
    before_vals = [metrics_before[m] for m in metric_names]
    after_vals = [metrics_after[m] for m in metric_names]
    x = np.arange(len(metric_names))
    axes[2].bar(x - 0.18, before_vals, width=0.36, label="Before", color="#C44E52")
    axes[2].bar(x + 0.18, after_vals, width=0.36, label="After", color="#55A868")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(metric_names, rotation=20, ha="right")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_title("Batch Mixing Metrics (Higher Better)")
    axes[2].set_ylabel("Score")
    axes[2].legend()

    from matplotlib.patches import Patch

    legend_items = [
        Patch(facecolor=cmap(code), label=label) for label, code in label_to_code.items()
    ]
    axes[1].legend(handles=legend_items, title=batch_key, loc="best")

    plt.tight_layout()
    return axes


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
            f"No valid embedding assays found. Requested={list(assay_names)}, layer='{layer}'.",
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
                f"Assay '{assay_name}/{layer}' must have at least 2 dimensions, "
                f"got {emb.shape[1]}.",
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
            scatter_ref,
            ax=axes.tolist(),
            fraction=0.03,
            pad=0.02,
            label=color_by,
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
