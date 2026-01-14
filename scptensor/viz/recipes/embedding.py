from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from scptensor.core.structures import ScpContainer
from scptensor.viz.base.scatter import MaskStyle, scatter

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


def embedding(
    container: ScpContainer,
    basis: str = "umap",
    color: str | None = None,
    layer: str = "imputed",
    show_missing: bool = False,
) -> "plt.Axes":
    """Plot 2D dimensional reduction embedding (PCA/UMAP/tSNE).

    Parameters
    ----------
    container : ScpContainer
        Input data container.
    basis : str, default "umap"
        Name of the assay containing 2D coordinates.
    color : str | None
        Either:
        - Column name in ``obs`` (metadata column)
        - Feature name found in assay var (expression values)
    layer : str, default "imputed"
        Layer to use when ``color`` is a feature name.
    show_missing : bool, default False
        If True, uses explicit marker coding for missing values.

    Returns
    -------
    plt.Axes
        The axes containing the plot.

    Raises
    ------
    ValueError
        If basis assay is not found or has < 2 dimensions.
        If color is not found in obs or feature var.

    """
    # Get coordinates
    basis_assay = container.assays.get(basis)
    if basis_assay is None:
        raise ValueError(f"Basis assay '{basis}' not found.")

    coords = _get_coordinates(basis_assay)
    X = coords[:, :2]

    # Resolve color and mask
    c, m = _resolve_color_and_mask(container, color, layer)

    mask_style: MaskStyle = "explicit" if show_missing else "subtle"
    title = f"{basis} colored by {color}" if color else basis
    cmap = "viridis" if (c is not None and np.issubdtype(c.dtype, np.number)) else "tab10"

    return scatter(
        X=X,
        c=c,
        m=m,
        mask_style=mask_style,
        title=title,
        xlabel=f"{basis}_1",
        ylabel=f"{basis}_2",
        cmap=cmap,
    )


def _get_coordinates(assay) -> np.ndarray:
    """Extract 2D coordinates from basis assay."""
    if "X" in assay.layers:
        coords = assay.layers["X"].X
    elif assay.layers:
        coords = next(iter(assay.layers.values())).X
    else:
        raise ValueError(f"Assay has no layers.")

    if coords.shape[1] < 2:
        raise ValueError(f"Basis has less than 2 dimensions.")

    return coords


def _resolve_color_and_mask(
    container: ScpContainer,
    color: str | None,
    layer: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Resolve color array and mask array from metadata or feature expression."""
    if color is None:
        return None, None

    # Metadata column
    if color in container.obs.columns:
        return _color_from_metadata(container, color)

    # Feature expression
    return _color_from_feature(container, color, layer)


def _color_from_metadata(
    container: ScpContainer,
    color: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract color values from obs metadata column."""
    col_data = container.obs[color]

    if col_data.dtype in (pl.String, pl.Categorical):
        # Factorize strings to integer codes for colormap
        _, c = np.unique(col_data.to_numpy(), return_inverse=True)
    else:
        c = col_data.to_numpy()

    # Metadata assumes all valid
    m = np.zeros(container.n_samples, dtype=np.int8)
    return c, m


def _color_from_feature(
    container: ScpContainer,
    color: str,
    layer: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract color values from feature expression layer."""
    assay_name = "protein"
    assay = container.assays.get(assay_name)

    if assay is None:
        raise ValueError(
            f"Color '{color}' not found in obs and assay '{assay_name}' missing."
        )

    # Find feature index
    feature_idx = _find_feature_index(assay, color)
    if feature_idx is None:
        raise ValueError(
            f"Color '{color}' not found in obs or {assay_name} features."
        )

    matrix = assay.layers.get(layer)
    if matrix is None:
        raise ValueError(f"Layer '{layer}' not found in assay '{assay_name}'.")

    c = matrix.X[:, feature_idx]
    m = matrix.M[:, feature_idx]
    return c, m


def _find_feature_index(assay, feature_name: str) -> int | None:
    """Find feature index in assay var by protein_id column."""
    if "protein_id" not in assay.var.columns:
        return None

    ids = assay.var["protein_id"].to_list()
    try:
        return ids.index(feature_name)
    except ValueError:
        return None
