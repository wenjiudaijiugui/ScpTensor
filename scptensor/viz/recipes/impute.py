"""Visualization recipes for imputation assessment.

This module provides specialized visualization functions for assessing imputation
quality and comparing different imputation methods in single-cell proteomics data.

Functions include:
- plot_imputation_comparison: Compare multiple methods side-by-side
- plot_imputation_scatter: Scatter plot imputed vs true values
- plot_imputation_metrics: Bar chart of NRMSE, PCC metrics
- plot_missing_pattern: Heatmap of missing value patterns
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import seaborn as sns
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.stats import pearsonr

from scptensor.core.exceptions import AssayNotFoundError, LayerNotFoundError
from scptensor.core.structures import ScpContainer
from scptensor.viz.base.style import PlotStyle

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def plot_imputation_comparison(
    container: ScpContainer,
    assay_name: str,
    layer_name: str,
    methods: list[str] | None = None,
    metrics: list[str] = ["nrmse", "pcc"],
    figsize: tuple[float, float] = (12, 6),
) -> Axes:
    """Compare multiple imputation methods using validation metrics.

    Creates a grouped bar chart comparing different imputation methods
    based on validation metrics. For imputed layers, compares against
    the original non-missing values to compute accuracy metrics.

    Parameters
    ----------
    container : ScpContainer
        Container with original (missing) data and multiple imputed layers.
    assay_name : str
        Assay name containing the layers to compare.
    layer_name : str
        Base layer name with original data (will have mask codes).
    methods : list of str or None, default None
        Imputation method layer names to compare. If None, uses all
        layers starting with common imputation prefixes (knn_, qrilc_, etc.).
    metrics : list of str, default ["nrmse", "pcc"]
        Metrics to compute. Options: "nrmse", "pcc", "cosine".
    figsize : tuple, default (12, 6)
        Figure size.

    Returns
    -------
    matplotlib.axes.Axes
        The plot axes with attached comparison results.

    Raises
    ------
    AssayNotFoundError
        If assay_name is not found in container.
    LayerNotFoundError
        If layer_name is not found in the assay.

    Examples
    --------
    >>> from scptensor import create_test_container
    >>> from scptensor.impute import impute_knn, impute_qrilc
    >>> from scptensor.viz.recipes.impute import plot_imputation_comparison
    >>> container = create_test_container()
    >>> container = impute_knn(container, "proteins", "raw", "knn")
    >>> container = impute_qrilc(container, "proteins", "raw", "qrilc")
    >>> ax = plot_imputation_comparison(
    ...     container, "proteins", "raw", methods=["knn", "qrilc"]
    ... )
    >>> results = ax.imputation_results  # Access comparison data
    """
    # Apply style
    PlotStyle.apply_style(theme="science")

    # Validate assay and layer
    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]
    if layer_name not in assay.layers:
        raise LayerNotFoundError(layer_name, assay_name)

    original_matrix = assay.layers[layer_name]

    # Get original data and mask
    if sp.issparse(original_matrix.X):
        X_orig = original_matrix.X.toarray()  # type: ignore[union-attr]
    else:
        X_orig = original_matrix.X.copy()

    if original_matrix.M is not None:
        if sp.issparse(original_matrix.M):
            M = original_matrix.M.toarray()  # type: ignore[union-attr]
        else:
            M = original_matrix.M.copy()
    else:
        M = np.zeros_like(X_orig, dtype=np.int8)

    # Auto-detect imputed layers if methods not specified
    if methods is None:
        impute_prefixes = [
            "knn_",
            "qrilc_",
            "bpca_",
            "nmf_",
            "lls_",
            "svd_",
            "mf_",
            "ppca_",
            "minprob_",
            "mindet_",
        ]
        methods = []
        for layer in assay.layers:
            if any(layer.startswith(prefix) for prefix in impute_prefixes):
                methods.append(layer)

    if not methods:
        raise ValueError(
            "No imputed layers found. Please specify methods parameter or "
            "run imputation functions to create imputed layers."
        )

    # Compute metrics for each method
    results = {}
    for method in methods:
        if method not in assay.layers:
            continue

        imputed_matrix = assay.layers[method]
        if sp.issparse(imputed_matrix.X):
            X_imp = imputed_matrix.X.toarray()  # type: ignore[union-attr]
        else:
            X_imp = imputed_matrix.X.copy()

        # Find imputed positions (mask code 5 or originally missing positions)
        imputed_mask = (M != 0) | (imputed_matrix.M == 5 if imputed_matrix.M is not None else False)

        if not np.any(imputed_mask):
            continue

        method_metrics = _compute_imputation_metrics(
            X_orig, X_imp, imputed_mask, metric_names=metrics
        )
        results[method] = method_metrics

    if not results:
        raise ValueError("No valid imputed data found for comparison.")

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data for plotting
    method_names = list(results.keys())
    n_methods = len(method_names)
    n_metrics = len(metrics)

    # Set up color palette (colorblind-friendly)
    colors = sns.color_palette("colorblind", n_colors=n_metrics)

    # Set up bar positions
    x = np.arange(n_methods)
    width = 0.8 / n_metrics

    # Plot bars for each metric
    bars = []
    for i, metric in enumerate(metrics):
        values = [results[m].get(metric, 0) for m in method_names]
        offset = (i - n_metrics / 2 + 0.5) * width
        bar = ax.bar(
            x + offset,
            values,
            width,
            label=metric.upper(),
            color=colors[i],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.8,
        )
        bars.append(bar)

    # Customize plot
    ax.set_xlabel("Imputation Method")
    ax.set_ylabel("Metric Value")
    ax.set_title("Imputation Method Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=45, ha="right")
    ax.legend(loc="upper right")

    # Add grid
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar_group in bars:
        for bar in bar_group:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    # Attach results to axes
    ax.imputation_results = results  # type: ignore[attr-defined]

    return ax


def plot_imputation_scatter(
    container_true: ScpContainer,
    container_imputed: ScpContainer,
    assay_name: str,
    layer_name: str,
    layer_imputed: str,
    figsize: tuple[float, float] = (8, 8),
) -> Axes:
    """Scatter plot comparing imputed values against true values.

    Creates a scatter plot with true values on x-axis and imputed values
    on y-axis. Red points indicate imputed values (originally missing),
    gray points indicate observed values. Includes correlation coefficient
    and identity line.

    Parameters
    ----------
    container_true : ScpContainer
        Container with true (complete) values.
    container_imputed : ScpContainer
        Container with imputed values (may have different mask codes).
    assay_name : str
        Assay name in both containers.
    layer_name : str
        Layer name in true container.
    layer_imputed : str
        Imputed layer name in imputed container.
    figsize : tuple, default (8, 8)
        Figure size.

    Returns
    -------
    matplotlib.axes.Axes
        The plot axes with attached correlation statistics.

    Raises
    ------
    AssayNotFoundError
        If assay_name is not found in either container.
    LayerNotFoundError
        If specified layers are not found.

    Examples
    --------
    >>> from scptensor import create_test_container
    >>> from scptensor.impute import impute_knn
    >>> from scptensor.viz.recipes.impute import plot_imputation_scatter
    >>> container_true = create_test_container()  # Complete data
    >>> container_imp = impute_knn(container, "proteins", "raw", "knn")
    >>> ax = plot_imputation_scatter(
    ...     container_true, container_imp, "proteins", "raw", "knn"
    ... )
    """
    # Apply style
    PlotStyle.apply_style(theme="science")

    # Validate containers
    if assay_name not in container_true.assays:
        raise AssayNotFoundError(assay_name)
    if assay_name not in container_imputed.assays:
        raise AssayNotFoundError(assay_name)

    assay_true = container_true.assays[assay_name]
    assay_imp = container_imputed.assays[assay_name]

    if layer_name not in assay_true.layers:
        raise LayerNotFoundError(layer_name, assay_name)
    if layer_imputed not in assay_imp.layers:
        raise LayerNotFoundError(layer_imputed, assay_name)

    # Get data
    matrix_true = assay_true.layers[layer_name]
    matrix_imp = assay_imp.layers[layer_imputed]

    if sp.issparse(matrix_true.X):
        X_true = matrix_true.X.toarray()  # type: ignore[union-attr]
    else:
        X_true = matrix_true.X.copy()

    if sp.issparse(matrix_imp.X):
        X_imp = matrix_imp.X.toarray()  # type: ignore[union-attr]
    else:
        X_imp = matrix_imp.X.copy()

    # Get mask to identify imputed vs observed
    if matrix_imp.M is not None:
        if sp.issparse(matrix_imp.M):
            M_imp = matrix_imp.M.toarray()  # type: ignore[union-attr]
        else:
            M_imp = matrix_imp.M.copy()
    else:
        M_imp = np.zeros_like(X_imp, dtype=np.int8)

    # Imputed values have mask code 5
    imputed_mask = M_imp == 5

    # Flatten for scatter plot
    true_flat = X_true.flatten()
    imp_flat = X_imp.flatten()
    imp_mask_flat = imputed_mask.flatten()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot observed values (gray)
    obs_true = true_flat[~imp_mask_flat]
    obs_imp = imp_flat[~imp_mask_flat]
    ax.scatter(
        obs_true,
        obs_imp,
        c="gray",
        alpha=0.3,
        s=20,
        label=f"Observed (n={len(obs_true)})",
        edgecolors="none",
    )

    # Plot imputed values (red)
    imp_true = true_flat[imp_mask_flat]
    imp_imp = imp_flat[imp_mask_flat]
    ax.scatter(
        imp_true,
        imp_imp,
        c="red",
        alpha=0.6,
        s=30,
        label=f"Imputed (n={len(imp_true)})",
        edgecolors="black",
        linewidth=0.5,
    )

    # Add identity line
    min_val = min(np.min(true_flat), np.min(imp_flat))
    max_val = max(np.max(true_flat), np.max(imp_flat))
    ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5, label="Identity")

    # Compute correlation statistics
    pcc_all = pearsonr(true_flat, imp_flat)[0]
    pcc_imp = pearsonr(imp_true, imp_imp)[0]

    # Compute NRMSE for imputed values
    nrmse = _compute_nrmse(imp_true, imp_imp)

    # Add statistics text box
    stats_text = (
        f"PCC (all): {pcc_all:.4f}\nPCC (imputed): {pcc_imp:.4f}\nNRMSE (imputed): {nrmse:.4f}"
    )
    ax.text(
        0.98,
        0.02,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
        fontsize=10,
        family="monospace",
    )

    # Set labels and title
    ax.set_xlabel("True Values")
    ax.set_ylabel("Imputed Values")
    ax.set_title(f"Imputation Accuracy: {layer_imputed}")
    ax.legend(loc="upper left")

    # Add grid
    ax.grid(True, alpha=0.3)

    # Attach statistics to axes
    ax.scatter_stats = {  # type: ignore[attr-defined]
        "pcc_all": pcc_all,
        "pcc_imputed": pcc_imp,
        "nrmse": nrmse,
    }

    return ax


def plot_imputation_metrics(
    metrics: dict[str, dict[str, float]],
    metric_names: list[str] = ["NRMSE", "PCC"],
    figsize: tuple[float, float] = (10, 6),
) -> Axes:
    """Bar chart of imputation performance metrics.

    Creates a grouped bar chart comparing different imputation methods
    across multiple performance metrics.

    Parameters
    ----------
    metrics : dict
        Nested dictionary: {method: {metric: value}}.
        Example: {"knn": {"nrmse": 0.2, "pcc": 0.95}, "qrilc": {...}}
    metric_names : list of str, default ["NRMSE", "PCC"]
        Display names for metrics (maps to lowercase keys in metrics dict).
    figsize : tuple, default (10, 6)
        Figure size.

    Returns
    -------
    matplotlib.axes.Axes
        The plot axes.

    Examples
    --------
    >>> from scptensor.viz.recipes.impute import plot_imputation_metrics
    >>> metrics = {
    ...     "KNN": {"nrmse": 0.15, "pcc": 0.92},
    ...     "QRILC": {"nrmse": 0.18, "pcc": 0.89},
    ...     "BPCA": {"nrmse": 0.12, "pcc": 0.95},
    ... }
    >>> ax = plot_imputation_metrics(metrics, metric_names=["NRMSE", "PCC"])
    """
    # Apply style
    PlotStyle.apply_style(theme="science")

    # Convert metric names to lowercase for dict lookup
    metric_keys = [name.lower() for name in metric_names]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data
    methods = list(metrics.keys())
    n_methods = len(methods)
    n_metrics = len(metric_names)

    # Set up color palette
    colors = sns.color_palette("colorblind", n_colors=n_metrics)

    # Set up bar positions
    x = np.arange(n_methods)
    width = 0.8 / n_metrics

    # Plot bars for each metric
    for i, (metric_key, metric_name) in enumerate(zip(metric_keys, metric_names, strict=False)):
        values = [metrics[m].get(metric_key, 0) for m in methods]
        offset = (i - n_metrics / 2 + 0.5) * width
        ax.bar(
            x + offset,
            values,
            width,
            label=metric_name,
            color=colors[i],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.8,
        )

        # Add value labels on bars
        for j, val in enumerate(values):
            if val > 0:
                ax.text(
                    x[j] + offset,
                    val,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    # Customize plot
    ax.set_xlabel("Imputation Method")
    ax.set_ylabel("Metric Value")
    ax.set_title("Imputation Performance Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.legend(loc="upper right")

    # Add grid
    ax.grid(True, alpha=0.3, axis="y")

    return ax


def plot_missing_pattern(
    container: ScpContainer,
    assay_name: str,
    layer_name: str,
    max_features: int = 100,
    max_samples: int = 100,
    figsize: tuple[float, float] = (12, 8),
) -> Axes:
    """Heatmap showing missing value patterns.

    Creates a binary heatmap showing missing (white) and observed (dark)
    value patterns across samples and features. Samples and features are
    clustered to reveal systematic missingness patterns.

    Parameters
    ----------
    container : ScpContainer
        Input container.
    assay_name : str
        Assay name.
    layer_name : str
        Layer name.
    max_features : int, default 100
        Maximum features to display. For larger datasets, features with
        highest missing rate are selected.
    max_samples : int, default 100
        Maximum samples to display. For larger datasets, samples are
        randomly sampled.
    figsize : tuple, default (12, 8)
        Figure size.

    Returns
    -------
    matplotlib.axes.Axes
        The plot axes with attached pattern statistics.

    Raises
    ------
    AssayNotFoundError
        If assay_name is not found in container.
    LayerNotFoundError
        If layer_name is not found in the assay.

    Examples
    --------
    >>> from scptensor import create_test_container
    >>> from scptensor.viz.recipes.impute import plot_missing_pattern
    >>> container = create_test_container(n_samples=50)
    >>> ax = plot_missing_pattern(container, "proteins", "raw")
    """
    # Apply style
    PlotStyle.apply_style(theme="science")

    # Validate assay and layer
    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]
    if layer_name not in assay.layers:
        raise LayerNotFoundError(layer_name, assay_name)

    matrix = assay.layers[layer_name]

    # Get mask matrix (True where missing/invalid)
    if matrix.M is not None:
        if sp.issparse(matrix.M):
            M = matrix.M.toarray()  # type: ignore[union-attr]
        else:
            M = matrix.M.copy()
        # Missing = non-zero mask code
        missing_mask = M != 0
    else:
        missing_mask = np.zeros(matrix.X.shape, dtype=bool)

    n_samples, n_features = missing_mask.shape

    # Subsample if needed
    sample_indices = np.arange(n_samples)
    feature_indices = np.arange(n_features)

    if n_samples > max_samples:
        np.random.seed(42)
        sample_indices = np.random.choice(n_samples, max_samples, replace=False)

    if n_features > max_features:
        # Select features with highest missing rate
        missing_rate = np.mean(missing_mask, axis=0)
        feature_indices = np.argsort(missing_rate)[-max_features:]

    # Subset data
    missing_subset = missing_mask[np.ix_(sample_indices, feature_indices)].astype(int)
    n_samples_sub, n_features_sub = missing_subset.shape

    # Apply hierarchical clustering
    sample_order = np.arange(n_samples_sub)
    feature_order = np.arange(n_features_sub)

    if n_samples_sub > 2:
        # Cluster samples by missing pattern
        sample_dist = np.zeros((n_samples_sub, n_samples_sub))
        for i in range(n_samples_sub):
            for j in range(i, n_samples_sub):
                # Hamming distance
                sample_dist[i, j] = np.mean(missing_subset[i] != missing_subset[j])
                sample_dist[j, i] = sample_dist[i, j]

        linkage_matrix = linkage(sample_dist[np.triu_indices(n_samples_sub, k=1)], method="average")
        sample_order = leaves_list(linkage_matrix)

    if n_features_sub > 2:
        # Cluster features by missing pattern
        feature_dist = np.zeros((n_features_sub, n_features_sub))
        for i in range(n_features_sub):
            for j in range(i, n_features_sub):
                feature_dist[i, j] = np.mean(missing_subset[:, i] != missing_subset[:, j])
                feature_dist[j, i] = feature_dist[i, j]

        linkage_matrix = linkage(
            feature_dist[np.triu_indices(n_features_sub, k=1)], method="average"
        )
        feature_order = leaves_list(linkage_matrix)

    # Reorder matrix
    missing_ordered = missing_subset[np.ix_(sample_order, feature_order)]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap (0=observed/dark, 1=missing/white)
    im = ax.imshow(
        missing_ordered,
        cmap="gray_r",
        aspect="auto",
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Missing Status")
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Observed", "Missing"])

    # Set labels
    ax.set_xlabel(f"Features (showing {n_features_sub} of {n_features})")
    ax.set_ylabel(f"Samples (showing {n_samples_sub} of {n_samples})")
    ax.set_title("Missing Value Pattern Heatmap")

    # Add statistics
    total_missing = np.sum(missing_subset)
    total_values = missing_subset.size
    missing_rate = total_missing / total_values

    # Count missing by samples
    missing_by_sample = np.sum(missing_subset, axis=1)
    # Count missing by features
    missing_by_feature = np.sum(missing_subset, axis=0)

    stats_text = (
        f"Overall Missing: {missing_rate:.1%}\n"
        f"Samples with >50% missing: {np.sum(missing_by_sample > n_features_sub / 2)}\n"
        f"Features with >50% missing: {np.sum(missing_by_feature > n_samples_sub / 2)}"
    )

    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
        fontsize=9,
    )

    # Attach statistics to axes
    ax.pattern_stats = {  # type: ignore[attr-defined]
        "missing_rate": missing_rate,
        "total_missing": total_missing,
        "total_values": total_values,
        "missing_by_sample": missing_by_sample,
        "missing_by_feature": missing_by_feature,
    }

    return ax


# ============================================================================
# Helper functions
# ============================================================================


def _compute_imputation_metrics(
    X_true: np.ndarray,
    X_imputed: np.ndarray,
    imputed_mask: np.ndarray,
    metric_names: list[str],
) -> dict[str, float]:
    """Compute imputation quality metrics.

    Parameters
    ----------
    X_true : np.ndarray
        True values.
    X_imputed : np.ndarray
        Imputed values.
    imputed_mask : np.ndarray
        Boolean mask indicating imputed positions.
    metric_names : list of str
        Metrics to compute.

    Returns
    -------
    dict
        Dictionary of metric values.
    """
    results = {}
    true_vals = X_true[imputed_mask]
    imp_vals = X_imputed[imputed_mask]

    if len(true_vals) == 0:
        return dict.fromkeys(metric_names, 0.0)

    for metric in metric_names:
        if metric == "nrmse":
            results[metric] = _compute_nrmse(true_vals, imp_vals)
        elif metric == "pcc":
            if len(true_vals) > 1:
                pcc = pearsonr(true_vals, imp_vals)[0]
                results[metric] = abs(pcc)
            else:
                results[metric] = 0.0
        elif metric == "cosine":
            results[metric] = _compute_cosine_similarity(true_vals, imp_vals)
        else:
            results[metric] = 0.0

    return results


def _compute_nrmse(true_vals: np.ndarray, imp_vals: np.ndarray) -> float:
    """Compute Normalized Root Mean Square Error.

    NRMSE = RMSE / (max(true) - min(true))
    """
    rmse = np.sqrt(np.mean((true_vals - imp_vals) ** 2))
    denom = np.max(true_vals) - np.min(true_vals)
    if denom > 0:
        return rmse / denom
    return 0.0


def _compute_cosine_similarity(true_vals: np.ndarray, imp_vals: np.ndarray) -> float:
    """Compute cosine similarity between true and imputed values."""
    dot_product = np.dot(true_vals, imp_vals)
    norm_true = np.linalg.norm(true_vals)
    norm_imp = np.linalg.norm(imp_vals)

    if norm_true > 0 and norm_imp > 0:
        return dot_product / (norm_true * norm_imp)
    return 0.0


__all__ = [
    "plot_imputation_comparison",
    "plot_imputation_scatter",
    "plot_imputation_metrics",
    "plot_missing_pattern",
]
