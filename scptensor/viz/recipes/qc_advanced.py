"""Advanced QC visualization recipes for single-cell proteomics data.

This module provides specialized visualization functions for advanced quality control
analysis, including sensitivity analysis, cumulative detection curves, and sample
similarity analysis.

Functions include:
- plot_sensitivity_summary: Sample-level detection sensitivity with violin plots
- plot_cumulative_sensitivity: Cumulative feature detection saturation analysis
- plot_jaccard_heatmap: Sample similarity heatmap using Jaccard index
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import seaborn as sns
from scipy.cluster.hierarchy import leaves_list, linkage

from scptensor.core.exceptions import AssayNotFoundError, LayerNotFoundError
from scptensor.core.structures import ScpContainer
from scptensor.viz.base.style import PlotStyle


def plot_sensitivity_summary(
    container: ScpContainer,
    assay_name: str = "proteins",
    layer_name: str = "raw",
    group_by: str | None = None,
    figsize: tuple[int, int] = (8, 6),
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Visualize sample detection sensitivity with violin and scatter plots.

    Creates a combined violin and strip plot showing the distribution of
    detected features per sample, grouped by a metadata column if specified.
    The violin shows the distribution shape, while individual points show
    sample-level values.

    Parameters
    ----------
    container : ScpContainer
        Input data container.
    assay_name : str, default "proteins"
        Name of the assay to analyze.
    layer_name : str, default "raw"
        Layer name within the assay.
    group_by : str or None, default None
        Column name in obs used for grouping samples. If None, all samples
        are treated as a single group.
    figsize : tuple[int, int], default (8, 6)
        Figure size. Only used if ax is None.
    ax : plt.Axes or None
        Matplotlib axes. If None, creates new figure.

    Returns
    -------
    plt.Axes
        The axes containing the plot.

    Raises
    ------
    AssayNotFoundError
        If assay_name is not found in container.
    LayerNotFoundError
        If layer_name is not found in the assay.

    Examples
    --------
    >>> from scptensor import create_test_container
    >>> from scptensor.viz.recipes.qc_advanced import plot_sensitivity_summary
    >>> container = create_test_container(n_samples=100)
    >>> ax = plot_sensitivity_summary(container, group_by="batch")
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

    # Calculate detected features per sample (M == 0 means valid/detected)
    if sp.issparse(matrix.M):
        n_detected = np.array(matrix.shape[1] - matrix.M.getnnz(axis=1)).flatten()  # type: ignore[union-attr, attr-defined]
    else:
        n_detected = np.sum(matrix.M == 0, axis=1)

    # Get grouping labels
    if group_by is None or group_by not in container.obs.columns:
        groups = np.full(container.n_samples, "All")
        group_labels = ["All"]
    else:
        groups = container.obs[group_by].to_numpy()
        group_labels = [str(g) for g in np.unique(groups)]

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Set up color palette (colorblind-friendly)
    colors = sns.color_palette("colorblind", n_colors=len(group_labels))

    # Prepare data for plotting
    data_by_group = []
    positions = []
    for i, label in enumerate(group_labels):
        mask = groups == label
        data_by_group.append(n_detected[mask])
        positions.append(i + 1)

    # Create violin plots
    parts = ax.violinplot(
        data_by_group,
        positions=positions,
        showmeans=True,
        showmedians=True,
        widths=0.6,
    )

    # Style violin bodies
    for i, pc in enumerate(parts["bodies"]):  # type: ignore[arg-type, var-annotated]
        pc.set_facecolor(colors[i])
        pc.set_edgecolor("black")
        pc.set_alpha(0.6)

    # Style violin statistics
    parts["cmeans"].set_color("black")
    parts["cmedians"].set_color("white")
    parts["cmedians"].set_linewidth(2)
    parts["cmaxes"].set_color("black")
    parts["cmins"].set_color("black")
    parts["cbars"].set_color("black")

    # Add strip plot (jittered scatter) on top
    for i, data in enumerate(data_by_group):
        x_jitter = np.random.normal(i + 1, 0.08, size=len(data))
        ax.scatter(
            x_jitter,
            data,
            color=colors[i],
            alpha=0.5,
            s=20,
            edgecolors="black",
            linewidth=0.5,
        )

    # Set labels and title
    ax.set_xticks(positions)
    ax.set_xticklabels(group_labels, rotation=45, ha="right")
    ax.set_xlabel("Group" if group_by else "All Samples")
    ax.set_ylabel("Number of Detected Features")
    title_suffix = f" by {group_by}" if group_by else ""
    ax.set_title(f"Sample Detection Sensitivity{title_suffix}")

    # Add total feature count as text
    n_features = assay.n_features
    ax.text(
        0.98,
        0.98,
        f"Total Features: {n_features}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    # Add grid
    ax.grid(True, alpha=0.3, axis="y")

    return ax


def plot_cumulative_sensitivity(
    container: ScpContainer,
    assay_name: str = "proteins",
    layer_name: str = "raw",
    group_by: str | None = None,
    figsize: tuple[int, int] = (8, 6),
    ax: plt.Axes | None = None,
    show_saturation: bool = True,
    saturation_threshold: float = 0.95,
) -> plt.Axes:
    """Visualize cumulative feature detection across samples.

    Creates a line plot showing how the cumulative number of unique detected
    features increases as more samples are included. This helps identify the
    point of diminishing returns (saturation) where adding more samples
    contributes few new features.

    Parameters
    ----------
    container : ScpContainer
        Input data container.
    assay_name : str, default "proteins"
        Name of the assay to analyze.
    layer_name : str, default "raw"
        Layer name within the assay.
    group_by : str or None, default None
        Column name in obs for grouping. If specified, plots separate curves
        for each group.
    figsize : tuple[int, int], default (8, 6)
        Figure size. Only used if ax is None.
    ax : plt.Axes or None
        Matplotlib axes. If None, creates new figure.
    show_saturation : bool, default True
        Whether to annotate the saturation point on the plot.
    saturation_threshold : float, default 0.95
        Proportion of total features at which saturation is considered reached.

    Returns
    -------
    plt.Axes
        The axes containing the plot.

    Raises
    ------
    AssayNotFoundError
        If assay_name is not found in container.
    LayerNotFoundError
        If layer_name is not found in the assay.

    Examples
    --------
    >>> from scptensor import create_test_container
    >>> from scptensor.viz.recipes.qc_advanced import plot_cumulative_sensitivity
    >>> container = create_test_container(n_samples=100)
    >>> ax = plot_cumulative_sensitivity(container)
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

    # Get detection matrix (True where detected/valid, M == 0)
    if sp.issparse(matrix.M):
        detected_mask = matrix.M == 0
    else:
        detected_mask = matrix.M == 0

    # Convert to dense if sparse for easier manipulation
    if sp.issparse(detected_mask):
        detected_mask = detected_mask.toarray()  # type: ignore[union-attr]

    n_features = assay.n_features

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Get groups
    if group_by is None or group_by not in container.obs.columns:
        groups = np.full(container.n_samples, "All")
        unique_groups = ["All"]
    else:
        groups = container.obs[group_by].to_numpy()
        unique_groups = [str(g) for g in np.unique(groups)]

    # Set up color palette (colorblind-friendly)
    colors = sns.color_palette("colorblind", n_colors=len(unique_groups))

    # Plot curve for each group
    for i, group_label in enumerate(unique_groups):
        group_mask = groups == group_label
        group_indices = np.where(group_mask)[0]

        if len(group_indices) == 0:
            continue

        # Get detection patterns for this group
        group_detected = detected_mask[group_indices, :]  # type: ignore[index]

        # Randomize order to avoid batch effects in ordering
        np.random.seed(42)
        perm = np.random.permutation(len(group_indices))
        group_detected = group_detected[perm, :]

        # Calculate cumulative unique detections
        cumulative_counts = []
        for j in range(1, len(group_indices) + 1):
            cumulative_detected = group_detected[:j, :].any(axis=0)
            cumulative_counts.append(np.sum(cumulative_detected))

        # Get x values (sample count)
        x_vals = np.arange(1, len(group_indices) + 1)

        # Plot the curve
        ax.plot(
            x_vals,
            cumulative_counts,
            color=colors[i],
            linewidth=2,
            label=group_label,
        )

        # Add saturation point if requested
        if show_saturation and len(cumulative_counts) > 0:
            target = saturation_threshold * n_features
            saturation_idx = None
            for j, count in enumerate(cumulative_counts):
                if count >= target:
                    saturation_idx = j
                    break

            if saturation_idx is not None:
                ax.scatter(
                    [saturation_idx + 1],
                    [cumulative_counts[saturation_idx]],
                    color=colors[i],
                    s=100,
                    marker="o",
                    edgecolors="black",
                    linewidth=1.5,
                    zorder=5,
                )
                ax.annotate(
                    f"{group_label}: {saturation_idx + 1} samples",
                    xy=(saturation_idx + 1, cumulative_counts[saturation_idx]),
                    xytext=(10, 10),
                    textcoords="offset points",
                    fontsize=9,
                    bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
                    arrowprops={"arrowstyle": "->", "color": colors[i]},
                )

    # Add saturation threshold line
    if show_saturation:
        ax.axhline(
            y=saturation_threshold * n_features,
            color="gray",
            linestyle="--",
            alpha=0.5,
            label=f"{saturation_threshold * 100:.0f}% saturation",
        )

    # Set labels and title
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Cumulative Unique Features Detected")
    title_suffix = f" by {group_by}" if group_by else ""
    ax.set_title(f"Cumulative Detection Sensitivity{title_suffix}")

    # Add legend
    if len(unique_groups) > 1 or show_saturation:
        ax.legend(loc="best", fontsize=9)

    # Add grid
    ax.grid(True, alpha=0.3)

    return ax


def plot_jaccard_heatmap(
    container: ScpContainer,
    assay_name: str = "proteins",
    layer_name: str = "raw",
    figsize: tuple[int, int] = (10, 8),
    ax: plt.Axes | None = None,
    cluster: bool = True,
    cmap: str = "viridis",
    show_low_similarity_only: bool = False,
    similarity_threshold: float = 0.3,
    annotation_threshold: int = 20,
) -> plt.Axes:
    """Visualize sample similarity using Jaccard index heatmap.

    The Jaccard index measures the similarity between two samples based on
    their detected feature overlap: J(A,B) = |A intersect B| / |A union B|.
    Values range from 0 (no overlap) to 1 (identical detection patterns).

    Parameters
    ----------
    container : ScpContainer
        Input data container.
    assay_name : str, default "proteins"
        Name of the assay to analyze.
    layer_name : str, default "raw"
        Layer name within the assay.
    figsize : tuple[int, int], default (10, 8)
        Figure size. Only used if ax is None.
    ax : plt.Axes or None
        Matplotlib axes. If None, creates new figure.
    cluster : bool, default True
        Whether to apply hierarchical clustering to reorder samples.
    cmap : str, default "viridis"
        Colormap for the heatmap.
    show_low_similarity_only : bool, default False
        If True, only show pairs with Jaccard index below threshold.
    similarity_threshold : float, default 0.3
        Threshold for low similarity filtering. Only used when
        show_low_similarity_only is True.
    annotation_threshold : int, default 20
        Maximum number of samples to show axis labels. For larger datasets,
        labels are omitted for clarity.

    Returns
    -------
    plt.Axes
        The axes containing the plot.

    Raises
    ------
    AssayNotFoundError
        If assay_name is not found in container.
    LayerNotFoundError
        If layer_name is not found in the assay.

    Examples
    --------
    >>> from scptensor import create_test_container
    >>> from scptensor.viz.recipes.qc_advanced import plot_jaccard_heatmap
    >>> container = create_test_container(n_samples=50)
    >>> ax = plot_jaccard_heatmap(container)
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

    # Get detection matrix (True where detected/valid, M == 0)
    if sp.issparse(matrix.M):
        detected_mask = matrix.M == 0
    else:
        detected_mask = matrix.M == 0

    # Convert to dense for Jaccard calculation
    if sp.issparse(detected_mask):
        detected_mask = detected_mask.toarray()  # type: ignore[union-attr]

    n_samples = container.n_samples

    # Calculate Jaccard similarity matrix
    # J(A,B) = |A intersect B| / |A union B|
    jaccard_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(i, n_samples):
            # Intersection: both detected
            intersection = np.sum(detected_mask[i] & detected_mask[j])  # type: ignore[index]
            # Union: either detected
            union = np.sum(detected_mask[i] | detected_mask[j])  # type: ignore[index]

            if union > 0:
                jaccard = intersection / union
            else:
                jaccard = 0.0

            jaccard_matrix[i, j] = jaccard
            jaccard_matrix[j, i] = jaccard

    # Apply clustering if requested
    if cluster and n_samples > 2:
        # Convert to distance (1 - similarity)
        distance_matrix = 1.0 - jaccard_matrix
        np.fill_diagonal(distance_matrix, 0)

        # Perform hierarchical clustering
        linkage_matrix = linkage(distance_matrix[np.triu_indices(n_samples, k=1)], method="average")
        leaf_order = leaves_list(linkage_matrix)

        # Reorder matrix
        jaccard_matrix = jaccard_matrix[np.ix_(leaf_order, leaf_order)]

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Show only low similarity if requested
    if show_low_similarity_only:
        # Create masked array for low similarity values
        display_matrix = np.ma.masked_where(jaccard_matrix >= similarity_threshold, jaccard_matrix)
    else:
        display_matrix = jaccard_matrix

    # Plot heatmap
    im = ax.imshow(
        display_matrix,
        cmap=cmap,
        aspect="auto",
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Jaccard Index")

    # Add labels for small datasets
    if n_samples <= annotation_threshold:
        # Get sample IDs if available
        if container.sample_id_col and container.sample_id_col in container.obs.columns:
            sample_ids = container.obs[container.sample_id_col].to_list()
        else:
            sample_ids = [f"S{i}" for i in range(n_samples)]

        # Reorder labels if clustered
        if cluster and n_samples > 2:
            sample_ids = [sample_ids[i] for i in leaf_order]

        ax.set_xticks(range(n_samples))
        ax.set_yticks(range(n_samples))
        ax.set_xticklabels(sample_ids, rotation=90, fontsize=8)
        ax.set_yticklabels(sample_ids, fontsize=8)
    else:
        # Show just tick marks for large datasets
        ax.set_xticks([])
        ax.set_yticks([])

    # Set title
    ax.set_title("Sample Similarity (Jaccard Index)")

    # Add statistics text
    mean_jaccard = np.mean(jaccard_matrix[np.triu_indices(n_samples, k=1)])
    min_jaccard = np.min(jaccard_matrix[np.triu_indices(n_samples, k=1)])
    ax.text(
        0.02,
        0.98,
        f"Mean: {mean_jaccard:.3f}\nMin: {min_jaccard:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        fontsize=9,
    )

    return ax


def plot_missing_type_heatmap(
    container: ScpContainer,
    assay_name: str = "proteins",
    layer_name: str = "raw",
    figsize: tuple[int, int] = (12, 8),
    ax: plt.Axes | None = None,
    cluster_samples: bool = True,
    cluster_features: bool = True,
    max_samples: int = 100,
    max_features: int = 100,
) -> plt.Axes:
    """Visualize mask value type distribution as a heatmap.

    Creates a heatmap showing the distribution of mask codes across samples
    and features. Each color represents a different mask code (value type):
    - Green (0): VALID - detected values
    - Yellow (1): MBR - missing between runs
    - Orange (2): LOD - below detection limit
    - Red (3): FILTERED - QC removed
    - Purple (5): IMPUTED - filled values

    Parameters
    ----------
    container : ScpContainer
        Input data container.
    assay_name : str, default "proteins"
        Name of the assay to analyze.
    layer_name : str, default "raw"
        Layer name within the assay.
    figsize : tuple[int, int], default (12, 8)
        Figure size. Only used if ax is None.
    ax : plt.Axes or None
        Matplotlib axes. If None, creates new figure.
    cluster_samples : bool, default True
        Whether to apply hierarchical clustering to reorder samples.
    cluster_features : bool, default True
        Whether to apply hierarchical clustering to reorder features.
    max_samples : int, default 100
        Maximum number of samples to display. For larger datasets,
        samples are randomly sampled.
    max_features : int, default 100
        Maximum number of features to display. For larger datasets,
        features with highest missing rate are selected.

    Returns
    -------
    plt.Axes
        The axes containing the plot.

    Raises
    ------
    AssayNotFoundError
        If assay_name is not found in container.
    LayerNotFoundError
        If layer_name is not found in the assay.

    Examples
    --------
    >>> from scptensor import create_test_container
    >>> from scptensor.viz.recipes.qc_advanced import plot_missing_type_heatmap
    >>> container = create_test_container(n_samples=50)
    >>> ax = plot_missing_type_heatmap(container)
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

    # Get mask matrix
    if sp.issparse(matrix.M):
        M = matrix.M.toarray()  # type: ignore[union-attr]
    else:
        M = matrix.M.copy()  # type: ignore[union-attr]

    n_samples, n_features = M.shape

    # Subsample if needed
    sample_indices = np.arange(n_samples)
    feature_indices = np.arange(n_features)

    if n_samples > max_samples:
        np.random.seed(42)
        sample_indices = np.random.choice(n_samples, max_samples, replace=False)

    if n_features > max_features:
        # Select features with highest variability in mask codes
        mask_variety = np.array([len(np.unique(M[:, i])) for i in range(n_features)])
        feature_indices = np.argsort(mask_variety)[-max_features:]

    # Subset data
    M_subset = M[np.ix_(sample_indices, feature_indices)]
    n_samples_sub, n_features_sub = M_subset.shape

    # Define mask code colors (colorblind-friendly palette)
    # 0: VALID (green), 1: MBR (yellow), 2: LOD (orange), 3: FILTERED (red), 5: IMPUTED (purple)
    mask_colors = {
        0: "#2ca02c",  # Green - VALID
        1: "#ffdd57",  # Yellow - MBR
        2: "#ff9f40",  # Orange - LOD
        3: "#d62728",  # Red - FILTERED
        5: "#9467bd",  # Purple - IMPUTED
    }

    # Create colormap from mask codes to integers for clustering
    # Map mask codes to consecutive integers for display
    unique_codes = sorted(set(M_subset.flatten()))
    code_to_int = {code: i for i, code in enumerate(unique_codes)}
    int_to_code = {i: code for code, i in code_to_int.items()}

    # Create integer matrix for display
    M_display = np.vectorize(code_to_int.get)(M_subset)

    # Apply clustering if requested
    sample_order = np.arange(n_samples_sub)
    feature_order = np.arange(n_features_sub)

    if cluster_samples and n_samples_sub > 2:
        # Compute distance between samples based on mask patterns
        sample_dist = np.zeros((n_samples_sub, n_samples_sub))
        for i in range(n_samples_sub):
            for j in range(i, n_samples_sub):
                # Use Hamming distance (proportion of differing positions)
                sample_dist[i, j] = np.mean(M_subset[i] != M_subset[j])
                sample_dist[j, i] = sample_dist[i, j]

        # Perform clustering
        linkage_matrix = linkage(sample_dist[np.triu_indices(n_samples_sub, k=1)], method="average")
        sample_order = leaves_list(linkage_matrix)

    if cluster_features and n_features_sub > 2:
        # Compute distance between features based on mask patterns
        feature_dist = np.zeros((n_features_sub, n_features_sub))
        for i in range(n_features_sub):
            for j in range(i, n_features_sub):
                feature_dist[i, j] = np.mean(M_subset[:, i] != M_subset[:, j])
                feature_dist[j, i] = feature_dist[i, j]

        # Perform clustering
        linkage_matrix = linkage(
            feature_dist[np.triu_indices(n_features_sub, k=1)], method="average"
        )
        feature_order = leaves_list(linkage_matrix)

    # Reorder display matrix
    M_display = M_display[np.ix_(sample_order, feature_order)]

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Create colormap for display
    display_colors = [mask_colors.get(int_to_code[i], "#gray") for i in range(len(unique_codes))]
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(display_colors)

    # Plot heatmap
    im = ax.imshow(
        M_display,
        cmap=cmap,
        aspect="auto",
        interpolation="nearest",
        vmin=-0.5,
        vmax=len(unique_codes) - 0.5,
    )

    # Create custom colorbar
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=mask_colors[code], label=f"{code}: {name}")
        for code, name in [
            (0, "VALID"),
            (1, "MBR"),
            (2, "LOD"),
            (3, "FILTERED"),
            (5, "IMPUTED"),
        ]
        if code in unique_codes
    ]

    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=True,
        title="Mask Code",
    )

    # Set labels
    ax.set_xlabel("Features")
    ax.set_ylabel("Samples")
    ax.set_title("Missing Value Type Distribution")

    # Add grid
    ax.grid(False)

    # Add statistics text
    code_counts = {code: np.sum(M_subset == code) for code in unique_codes}
    total_values = M_subset.size
    stats_text = "Mask Distribution:\n"
    for code in sorted(unique_codes):
        count = code_counts[code]
        pct = count / total_values * 100
        stats_text += f"{code}: {pct:.1f}%\n"

    ax.text(
        0.98,
        0.02,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
        fontsize=9,
        family="monospace",
    )

    return ax


def plot_missing_summary(
    container: ScpContainer,
    assay_name: str = "proteins",
    layer_name: str = "raw",
    figsize: tuple[int, int] = (14, 10),
    top_n_features: int = 20,
    show_sample_labels: bool = False,
) -> plt.Figure:
    """Create a comprehensive 4-panel missing value summary visualization.

    Creates a figure with four panels showing different aspects of missing data:
    1. Sample-wise missing rate (bar plot)
    2. Feature-wise missing rate (bar plot or histogram for many features)
    3. Missing value type distribution (pie chart)
    4. Missing value pattern analysis (bar plot of co-occurrence patterns)

    Parameters
    ----------
    container : ScpContainer
        Input data container.
    assay_name : str, default "proteins"
        Name of the assay to analyze.
    layer_name : str, default "raw"
        Layer name within the assay.
    figsize : tuple[int, int], default (14, 10)
        Figure size.
    top_n_features : int, default 20
        Number of top features to show in feature-wise plot. If number of
        features exceeds this, shows a histogram instead.
    show_sample_labels : bool, default False
        Whether to show sample labels on x-axis. Only practical for small datasets.

    Returns
    -------
    plt.Figure
        The figure object containing all four panels.

    Raises
    ------
    AssayNotFoundError
        If assay_name is not found in container.
    LayerNotFoundError
        If layer_name is not found in the assay.

    Examples
    --------
    >>> from scptensor import create_test_container
    >>> from scptensor.viz.recipes.qc_advanced import plot_missing_summary
    >>> container = create_test_container(n_samples=50)
    >>> fig = plot_missing_summary(container)
    >>> fig.savefig("missing_summary.png", dpi=300)
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

    # Get mask matrix
    if sp.issparse(matrix.M):
        M = matrix.M.toarray()  # type: ignore[union-attr]
    else:
        M = matrix.M.copy()  # type: ignore[union-attr]

    n_samples, n_features = M.shape

    # Create figure with 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax_sample, ax_feature, ax_pie, ax_pattern = axes.flatten()

    # Panel 1: Sample-wise missing rate
    is_missing = M != 0
    sample_missing_rate = np.mean(is_missing, axis=1)

    # Sort samples by missing rate
    sample_order = np.argsort(sample_missing_rate)
    sorted_sample_rates = sample_missing_rate[sample_order]

    # Color by missing rate (red = high missing, green = low missing)
    colors_sample = plt.cm.RdYlGn_r(sorted_sample_rates)  # type: ignore[attr-defined]

    ax_sample.bar(
        range(n_samples), sorted_sample_rates, color=colors_sample, edgecolor="black", linewidth=0.5
    )
    ax_sample.set_xlabel("Samples (sorted by missing rate)")
    ax_sample.set_ylabel("Missing Rate")
    ax_sample.set_title("Sample-wise Missing Rate")
    ax_sample.set_ylim(0, 1)

    # Add grid
    ax_sample.grid(True, alpha=0.3, axis="y")

    # Add mean line
    mean_missing = np.mean(sample_missing_rate)
    ax_sample.axhline(
        mean_missing, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_missing:.2%}"
    )
    ax_sample.legend(loc="upper right")

    # Panel 2: Feature-wise missing rate
    feature_missing_rate = np.mean(is_missing, axis=0)

    if n_features <= top_n_features:
        # Show bar plot for all features
        feature_order = np.argsort(feature_missing_rate)
        sorted_feature_rates = feature_missing_rate[feature_order]

        colors_feature = plt.cm.RdYlGn_r(sorted_feature_rates)  # type: ignore[attr-defined]
        ax_feature.bar(
            range(n_features),
            sorted_feature_rates,
            color=colors_feature,
            edgecolor="black",
            linewidth=0.5,
        )
        ax_feature.set_xlabel("Features (sorted by missing rate)")
        ax_feature.set_ylabel("Missing Rate")
        ax_feature.set_title("Feature-wise Missing Rate")
    else:
        # Show histogram for many features
        ax_feature.hist(
            feature_missing_rate, bins=30, color="steelblue", edgecolor="black", alpha=0.7
        )
        ax_feature.set_xlabel("Missing Rate")
        ax_feature.set_ylabel("Number of Features")
        ax_feature.set_title(f"Feature-wise Missing Rate Distribution (n={n_features})")

        # Add statistics
        ax_feature.axvline(
            mean_missing,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_missing:.2%}",
        )
        ax_feature.legend(loc="upper right")

    ax_feature.set_ylim(0, 1)
    ax_feature.grid(True, alpha=0.3, axis="y")

    # Panel 3: Missing value type distribution (pie chart)
    unique_codes, counts = np.unique(M, return_counts=True)

    # Define labels and colors for mask codes
    mask_labels = {
        0: "VALID (detected)",
        1: "MBR (missing)",
        2: "LOD (limit)",
        3: "FILTERED",
        5: "IMPUTED",
    }

    mask_colors_pie = {
        0: "#2ca02c",  # Green
        1: "#ffdd57",  # Yellow
        2: "#ff9f40",  # Orange
        3: "#d62728",  # Red
        5: "#9467bd",  # Purple
    }

    labels = [mask_labels.get(code, f"Code {code}") for code in unique_codes]
    colors = [mask_colors_pie.get(code, "#gray") for code in unique_codes]
    percentages = [count / M.size * 100 for count in counts]

    # Create pie chart
    wedges, texts, autotexts = ax_pie.pie(
        counts,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 9},
    )

    # Style percentage text
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_weight("bold")

    ax_pie.set_title("Missing Value Type Distribution")

    # Panel 4: Missing value pattern analysis
    # Count occurrences of each unique row pattern (sample-wise)
    # For efficiency with large datasets, use a simplified approach
    if n_samples <= 50:
        # Full pattern analysis for small datasets
        unique_patterns: list[tuple] = []
        pattern_counts: list[int] = []

        for i in range(n_samples):
            pattern = tuple(M[i, :])
            found = False
            for j, existing in enumerate(unique_patterns):
                if existing == pattern:
                    pattern_counts[j] += 1
                    found = True
                    break
            if not found:
                unique_patterns.append(pattern)
                pattern_counts.append(1)

        # Sort by count
        sorted_indices = np.argsort(pattern_counts)[::-1]
        pattern_counts = [pattern_counts[i] for i in sorted_indices]

        # Limit to top 10 patterns
        top_n = min(10, len(pattern_counts))
        x_pos = range(top_n)
        y_vals = pattern_counts[:top_n]

        ax_pattern.bar(x_pos, y_vals, color="steelblue", edgecolor="black", alpha=0.7)
        ax_pattern.set_xlabel("Missing Pattern (rank)")
        ax_pattern.set_ylabel("Number of Samples")
        ax_pattern.set_title("Sample Missing Pattern Frequency")

        # Add count labels
        for _, (x, y) in enumerate(zip(x_pos, y_vals, strict=False)):
            ax_pattern.text(x, y + 0.5, str(y), ha="center", fontsize=8)
    else:
        # Simplified pattern analysis for large datasets
        # Count number of missing value types per sample
        missing_complexity = []
        for i in range(n_samples):
            unique_in_sample = len(np.unique(M[i, :]))
            missing_complexity.append(unique_in_sample)

        unique_complexities, complexity_counts = np.unique(missing_complexity, return_counts=True)

        ax_pattern.bar(
            unique_complexities,
            complexity_counts,
            color="steelblue",
            edgecolor="black",
            alpha=0.7,
        )
        ax_pattern.set_xlabel("Number of Distinct Mask Codes")
        ax_pattern.set_ylabel("Number of Samples")
        ax_pattern.set_title("Sample Missing Pattern Complexity")

    ax_pattern.grid(True, alpha=0.3, axis="y")

    # Overall title
    fig.suptitle(
        f"Missing Value Summary: {assay_name} - {layer_name}", fontsize=14, fontweight="bold"
    )

    # Tight layout
    fig.tight_layout()

    return fig


def plot_cv_distribution(
    container: ScpContainer,
    assay_name: str = "proteins",
    layer_name: str = "raw",
    cv_threshold: float = 0.3,
    group_by: str | None = None,
    figsize: tuple[int, int] = (8, 6),
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Visualize coefficient of variation (CV) distribution across features.

    Creates a histogram showing the distribution of CV values for all features,
    with an optional threshold line to identify high-variance features.
    Supports grouping by metadata column for stratified analysis.

    Parameters
    ----------
    container : ScpContainer
        Input data container.
    assay_name : str, default "proteins"
        Name of the assay to analyze.
    layer_name : str, default "raw"
        Layer name within the assay.
    cv_threshold : float, default 0.3
        CV threshold line to display on the plot. Features above this
        threshold are considered highly variable.
    group_by : str or None, default None
        Column name in obs for grouping samples. If specified, CV is
        calculated separately for each group and overlaid on the plot.
    figsize : tuple[int, int], default (8, 6)
        Figure size. Only used if ax is None.
    ax : plt.Axes or None
        Matplotlib axes. If None, creates new figure.

    Returns
    -------
    plt.Axes
        The axes containing the plot.

    Raises
    ------
    AssayNotFoundError
        If assay_name is not found in container.
    LayerNotFoundError
        If layer_name is not found in the assay.
    ScpValueError
        If insufficient data for CV calculation.

    Examples
    --------
    >>> from scptensor import create_test_container
    >>> from scptensor.viz.recipes.qc_advanced import plot_cv_distribution
    >>> container = create_test_container(n_samples=50)
    >>> ax = plot_cv_distribution(container, cv_threshold=0.3)
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

    # Get data matrix
    if sp.issparse(matrix.X):
        X = matrix.X.toarray()  # type: ignore[union-attr]
    else:
        X = matrix.X.copy()

    # Get valid mask (M == 0 means detected/valid)
    if sp.issparse(matrix.M):
        valid_mask = matrix.M == 0
        if sp.issparse(valid_mask):
            valid_mask = valid_mask.toarray()  # type: ignore[union-attr]
    else:
        valid_mask = matrix.M == 0

    n_features = assay.n_features

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Get groups
    if group_by is None or group_by not in container.obs.columns:
        groups = np.full(container.n_samples, "All")
        unique_groups = ["All"]
    else:
        groups = container.obs[group_by].to_numpy()
        unique_groups = [str(g) for g in np.unique(groups)]

    # Set up color palette (colorblind-friendly)
    colors = sns.color_palette("colorblind", n_colors=len(unique_groups))

    # Calculate and plot CV for each group
    all_cv_values = []
    for i, group_label in enumerate(unique_groups):
        group_mask = groups == group_label
        group_X = X[group_mask, :]
        group_valid = valid_mask[group_mask, :]  # type: ignore[index]

        # Calculate CV per feature: std/mean for valid values only
        cv_values = np.zeros(n_features)
        for j in range(n_features):
            valid_vals = group_X[:, j][group_valid[:, j]]
            if len(valid_vals) > 1:
                mean_val = np.mean(valid_vals)
                if mean_val > 0:
                    cv_values[j] = np.std(valid_vals) / mean_val
                else:
                    cv_values[j] = np.nan
            else:
                cv_values[j] = np.nan

        # Remove NaN values
        cv_values = cv_values[~np.isnan(cv_values)]
        all_cv_values.extend(cv_values)

        # Plot histogram for this group
        if len(unique_groups) == 1:
            # Single group: use nicer bins
            bins = min(50, max(20, len(cv_values) // 10))
            ax.hist(
                cv_values,
                bins=bins,
                color=colors[0],
                edgecolor="black",
                alpha=0.7,
                label=f"All (n={len(cv_values)})",
            )
        else:
            # Multiple groups: use common bins
            if i == 0:
                bins = min(50, max(20, len(all_cv_values) // 10))
            ax.hist(
                cv_values,
                bins=bins,
                color=colors[i],
                edgecolor="black",
                alpha=0.5,
                label=f"{group_label} (n={len(cv_values)})",
            )

    # Calculate overall statistics
    all_cv_values = np.array(all_cv_values, dtype=np.float64)  # type: ignore[assignment]
    median_cv = np.median(all_cv_values)
    mean_cv = np.mean(all_cv_values)

    # Add threshold line
    ax.axvline(
        cv_threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Threshold: {cv_threshold:.2f}",
    )

    # Add median line
    ax.axvline(
        median_cv,
        color="green",
        linestyle="-.",
        linewidth=2,
        label=f"Median: {median_cv:.3f}",
    )

    # Set labels and title
    ax.set_xlabel("Coefficient of Variation (CV)")
    ax.set_ylabel("Number of Features")
    group_suffix = f" by {group_by}" if group_by else ""
    ax.set_title(f"Feature CV Distribution{group_suffix}")

    # Add legend
    ax.legend(loc="upper right", fontsize=9)

    # Add statistics box
    stats_text = f"Mean CV: {mean_cv:.3f}\nMedian CV: {median_cv:.3f}\n"
    stats_text += (
        f"Features > {cv_threshold}: {np.sum(all_cv_values > cv_threshold)} / {len(all_cv_values)}"  # type: ignore[operator]
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

    # Add grid
    ax.grid(True, alpha=0.3, axis="y")

    return ax


def plot_cv_by_feature(
    container: ScpContainer,
    assay_name: str = "proteins",
    layer_name: str = "raw",
    cv_threshold: float = 0.3,
    figsize: tuple[int, int] = (8, 6),
    ax: plt.Axes | None = None,
    use_log_scale: bool = False,
) -> plt.Axes:
    """Visualize CV vs mean expression for each feature.

    Creates a scatter plot with mean expression on x-axis and CV on y-axis.
    Features exceeding the CV threshold are highlighted in red. This plot
    helps identify highly variable features independent of their expression level.

    Parameters
    ----------
    container : ScpContainer
        Input data container.
    assay_name : str, default "proteins"
        Name of the assay to analyze.
    layer_name : str, default "raw"
        Layer name within the assay.
    cv_threshold : float, default 0.3
        CV threshold for highlighting high-variance features.
    figsize : tuple[int, int], default (8, 6)
        Figure size. Only used if ax is None.
    ax : plt.Axes or None
        Matplotlib axes. If None, creates new figure.
    use_log_scale : bool, default False
        If True, use log scale for x-axis (mean expression).

    Returns
    -------
    plt.Axes
        The axes containing the plot.

    Raises
    ------
    AssayNotFoundError
        If assay_name is not found in container.
    LayerNotFoundError
        If layer_name is not found in the assay.
    ScpValueError
        If insufficient data for CV calculation.

    Examples
    --------
    >>> from scptensor import create_test_container
    >>> from scptensor.viz.recipes.qc_advanced import plot_cv_by_feature
    >>> container = create_test_container(n_samples=50)
    >>> ax = plot_cv_by_feature(container, cv_threshold=0.3)
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

    # Get data matrix
    if sp.issparse(matrix.X):
        X = matrix.X.toarray()  # type: ignore[union-attr]
    else:
        X = matrix.X.copy()

    # Get valid mask (M == 0 means detected/valid)
    if sp.issparse(matrix.M):
        valid_mask = matrix.M == 0
        if sp.issparse(valid_mask):
            valid_mask = valid_mask.toarray()  # type: ignore[union-attr]
    else:
        valid_mask = matrix.M == 0

    n_features = assay.n_features

    # Calculate mean and CV per feature
    feature_means = np.zeros(n_features)
    feature_cvs = np.zeros(n_features)

    for j in range(n_features):
        valid_vals = X[:, j][valid_mask[:, j]]  # type: ignore[index]
        if len(valid_vals) > 1:
            feature_means[j] = np.mean(valid_vals)
            if feature_means[j] > 0:
                feature_cvs[j] = np.std(valid_vals) / feature_means[j]
            else:
                feature_cvs[j] = np.nan
                feature_means[j] = np.nan
        else:
            feature_cvs[j] = np.nan
            feature_means[j] = np.nan

    # Remove NaN values
    valid_features = ~np.isnan(feature_cvs) & ~np.isnan(feature_means)
    feature_means = feature_means[valid_features]
    feature_cvs = feature_cvs[valid_features]

    # Identify high CV features
    high_cv_mask = feature_cvs > cv_threshold
    n_high_cv = np.sum(high_cv_mask)

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot points
    ax.scatter(
        feature_means[~high_cv_mask],
        feature_cvs[~high_cv_mask],
        c="steelblue",
        alpha=0.6,
        s=30,
        edgecolors="black",
        linewidth=0.5,
        label=f"CV <= {cv_threshold}",
    )

    ax.scatter(
        feature_means[high_cv_mask],
        feature_cvs[high_cv_mask],
        c="red",
        alpha=0.7,
        s=40,
        edgecolors="black",
        linewidth=0.5,
        label=f"CV > {cv_threshold} (n={n_high_cv})",
        zorder=5,
    )

    # Add threshold line
    ax.axhline(
        cv_threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
    )

    # Set labels and title
    ax.set_xlabel("Mean Expression")
    ax.set_ylabel("Coefficient of Variation (CV)")
    ax.set_title("Feature CV vs Mean Expression")

    # Apply log scale if requested
    if use_log_scale:
        ax.set_xscale("log")

    # Add legend
    ax.legend(loc="upper right", fontsize=9)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Adjust layout
    ax.set_ylim(0, max(feature_cvs) * 1.1)

    return ax


def plot_cv_comparison(
    container: ScpContainer,
    assay_name: str = "proteins",
    layer_name: str = "raw",
    batch_col: str = "batch",
    figsize: tuple[int, int] = (10, 6),
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Compare within-batch vs between-batch CV for batch effect assessment.

    Creates a bar chart comparing the coefficient of variation within batches
    versus across all samples. A large difference indicates potential batch
    effects where samples vary more by batch than biological variation.

    Parameters
    ----------
    container : ScpContainer
        Input data container.
    assay_name : str, default "proteins"
        Name of the assay to analyze.
    layer_name : str, default "raw"
        Layer name within the assay.
    batch_col : str, default "batch"
        Column name in obs containing batch information.
    figsize : tuple[int, int], default (10, 6)
        Figure size. Only used if ax is None.
    ax : plt.Axes or None
        Matplotlib axes. If None, creates new figure.

    Returns
    -------
    plt.Axes
        The axes containing the plot. The axes also contains a
        cv_comparison attribute with the comparison results dict.

    Raises
    ------
    AssayNotFoundError
        If assay_name is not found in container.
    LayerNotFoundError
        If layer_name is not found in the assay.
    ScpValueError
        If batch_col is not found in obs or insufficient data.

    Examples
    --------
    >>> from scptensor import create_test_container
    >>> from scptensor.viz.recipes.qc_advanced import plot_cv_comparison
    >>> container = create_test_container(n_samples=100)
    >>> ax = plot_cv_comparison(container, batch_col="batch")
    >>> results = ax.cv_comparison  # Access comparison data
    """
    # Apply style
    PlotStyle.apply_style(theme="science")

    # Validate assay and layer
    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]
    if layer_name not in assay.layers:
        raise LayerNotFoundError(layer_name, assay_name)

    # Validate batch column
    if batch_col not in container.obs.columns:
        from scptensor.core.exceptions import ScpValueError

        raise ScpValueError(f"Batch column '{batch_col}' not found in obs")

    matrix = assay.layers[layer_name]

    # Get data matrix
    if sp.issparse(matrix.X):
        X = matrix.X.toarray()  # type: ignore[union-attr]
    else:
        X = matrix.X.copy()

    # Get valid mask (M == 0 means detected/valid)
    if sp.issparse(matrix.M):
        valid_mask = matrix.M == 0
        if sp.issparse(valid_mask):
            valid_mask = valid_mask.toarray()  # type: ignore[union-attr]
    else:
        valid_mask = matrix.M == 0

    n_features = assay.n_features

    # Get batch information
    batches = container.obs[batch_col].to_numpy()
    unique_batches = [str(b) for b in np.unique(batches)]

    # Calculate within-batch CV for each batch
    within_batch_cvs = {}
    for batch_label in unique_batches:
        batch_mask = batches == batch_label
        batch_X = X[batch_mask, :]
        batch_valid = valid_mask[batch_mask, :]  # type: ignore[index]

        batch_cv_values = []
        for j in range(n_features):
            valid_vals = batch_X[:, j][batch_valid[:, j]]
            if len(valid_vals) > 1:
                mean_val = np.mean(valid_vals)
                if mean_val > 0:
                    batch_cv_values.append(np.std(valid_vals) / mean_val)

        if batch_cv_values:
            within_batch_cvs[batch_label] = np.mean(batch_cv_values)
        else:
            within_batch_cvs[batch_label] = np.nan

    # Calculate between-batch CV (CV of batch means)
    between_batch_cvs = []
    for j in range(n_features):
        batch_means = []
        for batch_label in unique_batches:
            batch_mask = batches == batch_label
            batch_valid = valid_mask[batch_mask, j]  # type: ignore[index]
            valid_vals = X[batch_mask, j][batch_valid]

            if len(valid_vals) > 0:
                batch_means.append(np.mean(valid_vals))

        if len(batch_means) > 1:
            mean_of_means = np.mean(batch_means)
            if mean_of_means > 0:
                between_batch_cvs.append(np.std(batch_means) / mean_of_means)

    between_batch_cv = np.mean(between_batch_cvs) if between_batch_cvs else np.nan

    # Calculate overall CV (all samples together)
    overall_cv_values = []
    for j in range(n_features):
        valid_vals = X[:, j][valid_mask[:, j]]  # type: ignore[index]
        if len(valid_vals) > 1:
            mean_val = np.mean(valid_vals)
            if mean_val > 0:
                overall_cv_values.append(np.std(valid_vals) / mean_val)

    overall_cv = np.mean(overall_cv_values) if overall_cv_values else np.nan

    # Prepare comparison results
    comparison_results = {
        "within_batch": within_batch_cvs,
        "between_batch": between_batch_cv,
        "overall": overall_cv,
        "batch_effect_ratio": between_batch_cv / overall_cv if overall_cv > 0 else np.nan,
    }

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Prepare data for plotting
    batch_names = list(within_batch_cvs.keys())
    within_values = [within_batch_cvs[b] for b in batch_names]

    # Set up bar positions
    x = np.arange(len(batch_names))
    width = 0.35

    # Color palette
    colors = sns.color_palette("colorblind", n_colors=max(3, len(batch_names)))

    # Plot within-batch CV bars
    ax.bar(
        x - width / 2,
        within_values,
        width,
        label="Within-Batch CV",
        color=colors[0],
        edgecolor="black",
        linewidth=0.5,
        alpha=0.8,
    )

    # Plot between-batch CV as horizontal line
    ax.axhline(
        between_batch_cv,
        color=colors[1],
        linestyle="--",
        linewidth=2,
        label=f"Between-Batch CV: {between_batch_cv:.3f}",
    )

    # Plot overall CV as horizontal line
    ax.axhline(
        overall_cv,
        color=colors[2],
        linestyle=":",
        linewidth=2,
        label=f"Overall CV: {overall_cv:.3f}",
    )

    # Set labels and title
    ax.set_xlabel("Batch")
    ax.set_ylabel("Mean Coefficient of Variation (CV)")
    ax.set_title("Within-Batch vs Between-Batch CV Comparison")

    # Set x-axis ticks
    ax.set_xticks(x)
    ax.set_xticklabels(batch_names, rotation=45, ha="right")

    # Add legend
    ax.legend(loc="upper right", fontsize=9)

    # Add grid
    ax.grid(True, alpha=0.3, axis="y")

    # Add interpretation text
    ratio = comparison_results["batch_effect_ratio"]
    if not np.isnan(ratio):  # type: ignore[arg-type]
        if ratio > 1.5:  # type: ignore[operator]
            interpretation = "High batch effect detected"
            interpretation_color = "red"
        elif ratio > 1.1:  # type: ignore[operator]
            interpretation = "Moderate batch effect detected"
            interpretation_color = "orange"
        else:
            interpretation = "Low batch effect"
            interpretation_color = "green"

        ax.text(
            0.02,
            0.98,
            f"{interpretation}\nBetween/Within Ratio: {ratio:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
            fontsize=9,
            color=interpretation_color,
        )

    # Attach comparison results to axes
    ax.cv_comparison = comparison_results  # type: ignore[attr-defined]

    return ax


__all__ = [
    "plot_sensitivity_summary",
    "plot_cumulative_sensitivity",
    "plot_jaccard_heatmap",
    "plot_missing_type_heatmap",
    "plot_missing_summary",
    "plot_cv_distribution",
    "plot_cv_by_feature",
    "plot_cv_comparison",
]
