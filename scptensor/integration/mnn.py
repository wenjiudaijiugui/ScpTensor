"""Full Mutual Nearest Neighbors (MNN) correction for exploratory integration.

This implementation follows the original MNN workflow more closely than the
previous pair-only heuristic:

1. optionally cosine-normalize the input for neighbor search;
2. merge batches progressively, starting from a reference batch;
3. identify mutual nearest neighbors between the current reference and the
   next target batch;
4. compute batch vectors from the paired cells;
5. Gaussian-smooth those batch vectors across *all* cells in the target batch;
6. apply the smoothed correction and expand the reference set.

The result remains an exploratory integration output rather than a stable
protein-level matrix for downstream DE.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from scptensor.core._structure_container import ScpContainer
from scptensor.core.exceptions import ScpValueError
from scptensor.integration.base import (
    add_integrated_layer,
    log_integration_operation,
    prepare_integration_input,
    preserve_sparsity,
    register_integrate_method,
    validate_batch_integration_params,
    validate_layer_context,
)


@register_integrate_method("mnn", integration_level="embedding", recommended_for_de=False)
def integrate_mnn(
    container: ScpContainer,
    batch_key: str,
    assay_name: str = "protein",
    base_layer: str = "raw",
    new_layer_name: str | None = "mnn_corrected",
    k: int = 20,
    sigma: float = 1.0,
    n_pcs: int | None = None,
    use_pca: bool = True,
    use_anchor_correction: bool = True,
    merge_order: Sequence[str] | None = None,
    cos_norm_in: bool = True,
    cos_norm_out: bool = True,
    svd_dim: int = 0,
    var_adj: bool = True,
) -> ScpContainer:
    """Correct batch effects with a progressive full-MNN workflow.

    Parameters
    ----------
    container : ScpContainer
        Input container with multiple batches.
    batch_key : str
        Column name in ``obs`` containing batch labels.
    assay_name : str, default="protein"
        Name of the assay to use.
    base_layer : str, default="raw"
        Layer to use as input.
    new_layer_name : str | None, default="mnn_corrected"
        Name for the corrected layer.
    k : int, default=20
        Number of nearest neighbors to use when identifying MNN pairs.
    sigma : float, default=1.0
        Gaussian kernel bandwidth used to smooth correction vectors across the
        target batch.
    n_pcs : int | None, default=None
        Number of principal components for MNN search when ``use_pca=True``.
        If None, up to 50 components are used subject to matrix dimensions.
    use_pca : bool, default=True
        Whether to perform PCA on the current reference+target merge step
        before nearest-neighbor search.
    use_anchor_correction : bool, default=True
        Backward-compatible flag. Full MNN always uses progressive merging for
        multi-batch inputs; for more than two batches this flag must remain
        True. Use ``merge_order`` to control merge order explicitly.
    merge_order : Sequence[str] | None, default=None
        Explicit progressive merge order of batch labels. By default, batches
        are merged in first-observed order.
    cos_norm_in : bool, default=True
        Whether to cosine-normalize the feature matrix used for MNN search and
        for the internal correction-state updates.
    cos_norm_out : bool, default=True
        Whether to cosine-normalize the matrix on which the correction vectors
        are applied and returned.
    svd_dim : int, default=0
        Number of biological subspace dimensions to remove from the correction
        vectors. When greater than zero, the correction component parallel to
        the leading singular vectors of the paired reference/target cells is
        subtracted.
    var_adj : bool, default=True
        Whether to perform batchelor-style variance adjustment to reduce
        under-correction caused by "kissing" effects.

    Returns
    -------
    ScpContainer
        Container with a new exploratory MNN-corrected layer.

    """
    if k <= 0:
        raise ScpValueError(f"k must be positive, got {k}.", parameter="k", value=k)
    if sigma <= 0:
        raise ScpValueError(f"sigma must be positive, got {sigma}.", parameter="sigma", value=sigma)
    if n_pcs is not None and n_pcs <= 0:
        raise ScpValueError(f"n_pcs must be positive or None, got {n_pcs}.", parameter="n_pcs")
    if svd_dim < 0:
        raise ScpValueError(f"svd_dim must be >= 0, got {svd_dim}.", parameter="svd_dim")

    ctx = validate_layer_context(container, assay_name, base_layer)
    assay = ctx.assay
    layer = ctx.layer
    _, batches, _, _ = validate_batch_integration_params(
        container,
        batch_key,
        ctx.resolved_assay_name,
        min_batches=2,
    )
    X, input_was_sparse = prepare_integration_input(layer, context="MNN integration")
    if not np.isfinite(X).all():
        raise ScpValueError(
            "MNN integration requires a complete matrix with only finite values "
            "(no NaN/Inf values). "
            "Please impute or filter missing values before batch integration.",
            parameter="X",
        )

    batch_labels = np.asarray([str(b) for b in batches], dtype=object)
    batch_order = _resolve_merge_order(batch_labels, merge_order)
    if len(batch_order) > 2 and not use_anchor_correction:
        raise ScpValueError(
            "Full MNN now uses progressive merging for multi-batch inputs. "
            "For more than two batches, keep use_anchor_correction=True and "
            "control the order with merge_order if needed.",
            parameter="use_anchor_correction",
            value=use_anchor_correction,
        )

    X_corrected = _run_progressive_mnn(
        X=X,
        batch_labels=batch_labels,
        batch_order=batch_order,
        k=k,
        sigma=sigma,
        n_pcs=n_pcs,
        use_pca=use_pca,
        cos_norm_in=cos_norm_in,
        cos_norm_out=cos_norm_out,
        svd_dim=svd_dim,
        var_adj=var_adj,
    )

    X_corrected = preserve_sparsity(X_corrected, input_was_sparse)
    add_integrated_layer(assay, new_layer_name or "mnn_corrected", X_corrected, layer)

    return log_integration_operation(
        container,
        action="integration_mnn",
        method_name="mnn",
        params={
            "batch_key": batch_key,
            "assay": ctx.resolved_assay_name,
            "k": k,
            "sigma": sigma,
            "use_pca": use_pca,
            "n_pcs": n_pcs,
            "merge_order": list(batch_order),
            "cos_norm_in": cos_norm_in,
            "cos_norm_out": cos_norm_out,
            "svd_dim": svd_dim,
            "var_adj": var_adj,
            "n_batches": len(batch_order),
        },
        description=(
            f"Full MNN correction (k={k}, sigma={sigma}, merge_order={list(batch_order)}, "
            f"svd_dim={svd_dim}, var_adj={var_adj}) "
            f"on assay '{ctx.resolved_assay_name}'."
        ),
    )


def _resolve_merge_order(
    batch_labels: np.ndarray,
    merge_order: Sequence[str] | None,
) -> list[str]:
    """Resolve progressive merge order using first-observed batch order by default."""
    observed_order = list(dict.fromkeys(batch_labels.tolist()))
    if merge_order is None:
        return observed_order

    requested = [str(label) for label in merge_order]
    if len(requested) != len(observed_order) or set(requested) != set(observed_order):
        raise ScpValueError(
            f"merge_order must contain each batch exactly once. "
            f"Observed batches: {observed_order}, got: {requested}",
            parameter="merge_order",
            value=list(merge_order),
        )
    return requested


def _run_progressive_mnn(
    *,
    X: np.ndarray,
    batch_labels: np.ndarray,
    batch_order: list[str],
    k: int,
    sigma: float,
    n_pcs: int | None,
    use_pca: bool,
    cos_norm_in: bool,
    cos_norm_out: bool,
    svd_dim: int,
    var_adj: bool,
) -> np.ndarray:
    """Run progressive MNN correction, smoothing batch vectors over all target cells."""
    batch_indices = {label: np.flatnonzero(batch_labels == label) for label in batch_order}
    X_corrected = np.zeros_like(X, dtype=np.float64)

    ref_label = batch_order[0]
    ref_idx = batch_indices[ref_label]
    ref_in = _prepare_batch_matrix(X[ref_idx], cosine_normalize=cos_norm_in)
    ref_out = _prepare_batch_matrix(X[ref_idx], cosine_normalize=cos_norm_out)
    X_corrected[ref_idx] = ref_out

    merged_labels = [ref_label]

    for target_label in batch_order[1:]:
        target_idx = batch_indices[target_label]
        target_in = _prepare_batch_matrix(X[target_idx], cosine_normalize=cos_norm_in)
        target_out = _prepare_batch_matrix(X[target_idx], cosine_normalize=cos_norm_out)

        ref_search, target_search = _compute_search_coordinates(
            ref_in=ref_in,
            target_in=target_in,
            use_pca=use_pca,
            n_pcs=n_pcs,
        )
        mnn_pairs = _find_mnn_pairs(ref_search, target_search, k=k)
        if not mnn_pairs:
            merged_str = ", ".join(merged_labels)
            raise ScpValueError(
                f"MNN correction found no mutual nearest neighbors while merging batch "
                f"'{target_label}' into reference [{merged_str}]. "
                "This usually means the batches do not share an overlapping cell "
                "population or the preprocessing spaces are incompatible.",
                parameter="batch_key",
                value=target_label,
            )

        correction_in = _compute_smoothed_correction(
            reference_data=ref_in,
            target_data=target_in,
            kernel_space=target_in,
            mnn_pairs=mnn_pairs,
            sigma=sigma,
        )
        correction_out: np.ndarray | None = None
        if cos_norm_in != cos_norm_out:
            correction_out = _compute_smoothed_correction(
                reference_data=ref_out,
                target_data=target_out,
                kernel_space=target_in,
                mnn_pairs=mnn_pairs,
                sigma=sigma,
            )

        if svd_dim > 0:
            ref_pair_ids = np.unique([pair[0] for pair in mnn_pairs])
            target_pair_ids = np.unique([pair[1] for pair in mnn_pairs])
            span_ref_in = _compute_biological_span(ref_in[ref_pair_ids], svd_dim)
            span_target_in = _compute_biological_span(target_in[target_pair_ids], svd_dim)
            correction_in = _subtract_biological_components(
                correction_in,
                span_ref_in,
                span_target_in,
            )
            if correction_out is not None:
                span_ref_out = _compute_biological_span(ref_out[ref_pair_ids], svd_dim)
                span_target_out = _compute_biological_span(target_out[target_pair_ids], svd_dim)
                correction_out = _subtract_biological_components(
                    correction_out,
                    span_ref_out,
                    span_target_out,
                )

        if var_adj:
            correction_in = _adjust_shift_variance(
                reference_data=ref_in,
                target_data=target_in,
                correction=correction_in,
                sigma=sigma,
            )
            if correction_out is not None:
                correction_out = _adjust_shift_variance(
                    reference_data=ref_out,
                    target_data=target_out,
                    correction=correction_out,
                    sigma=sigma,
                )

        corrected_target_in = target_in + correction_in

        if correction_out is None:
            corrected_target_out = corrected_target_in.copy()
        else:
            corrected_target_out = target_out + correction_out

        X_corrected[target_idx] = corrected_target_out
        ref_in = np.vstack([ref_in, corrected_target_in])
        ref_out = np.vstack([ref_out, corrected_target_out])
        merged_labels.append(target_label)

    return X_corrected


def _prepare_batch_matrix(
    X_batch: np.ndarray,
    *,
    cosine_normalize: bool,
) -> np.ndarray:
    """Prepare one batch for MNN correction."""
    X_batch = np.asarray(X_batch, dtype=np.float64)
    if not cosine_normalize:
        return np.array(X_batch, copy=True)
    return _cosine_normalize_rows(X_batch)


def _cosine_normalize_rows(X: np.ndarray) -> np.ndarray:
    """Apply row-wise cosine normalization, leaving zero rows unchanged."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    safe_norms = np.where(norms > 0, norms, 1.0)
    return X / safe_norms


def _compute_search_coordinates(
    *,
    ref_in: np.ndarray,
    target_in: np.ndarray,
    use_pca: bool,
    n_pcs: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the search-space coordinates for one progressive merge step."""
    if not use_pca:
        return ref_in, target_in

    combined = np.vstack([ref_in, target_in])
    n_components = min(n_pcs or 50, combined.shape[1], combined.shape[0] - 1)
    if n_components <= 0:
        return ref_in, target_in

    coords = PCA(n_components=n_components, random_state=42).fit_transform(combined)
    ref_n = ref_in.shape[0]
    return coords[:ref_n], coords[ref_n:]


def _find_mnn_pairs(
    reference_coords: np.ndarray,
    target_coords: np.ndarray,
    *,
    k: int,
) -> list[tuple[int, int]]:
    """Find mutual nearest-neighbor pairs between reference and target batches."""
    ref_to_target = NearestNeighbors(
        n_neighbors=min(k, target_coords.shape[0]),
        algorithm="auto",
    ).fit(target_coords)
    target_neighbors = ref_to_target.kneighbors(reference_coords, return_distance=False)

    target_to_ref = NearestNeighbors(
        n_neighbors=min(k, reference_coords.shape[0]),
        algorithm="auto",
    ).fit(reference_coords)
    ref_neighbors = target_to_ref.kneighbors(target_coords, return_distance=False)

    pairs: list[tuple[int, int]] = []
    for ref_idx, neighbors in enumerate(target_neighbors):
        for target_idx in neighbors:
            if ref_idx in ref_neighbors[target_idx]:
                pairs.append((ref_idx, int(target_idx)))
    return pairs


def _compute_smoothed_correction(
    *,
    reference_data: np.ndarray,
    target_data: np.ndarray,
    kernel_space: np.ndarray,
    mnn_pairs: list[tuple[int, int]],
    sigma: float,
) -> np.ndarray:
    """Compute Gaussian-smoothed correction vectors for all target cells."""
    ref_ids = np.array([pair[0] for pair in mnn_pairs], dtype=np.int64)
    target_ids = np.array([pair[1] for pair in mnn_pairs], dtype=np.int64)

    pair_vectors = reference_data[ref_ids] - target_data[target_ids]
    unique_targets, inverse = np.unique(target_ids, return_inverse=True)
    mean_vectors = _average_vectors_by_group(pair_vectors, inverse, n_groups=len(unique_targets))
    centers = kernel_space[unique_targets]
    return _gaussian_smooth_vectors(mean_vectors, centers, kernel_space, sigma=sigma)


def _compute_biological_span(X_cells_features: np.ndarray, svd_dim: int) -> np.ndarray | None:
    """Return leading feature-space singular vectors for the selected cells."""
    if svd_dim <= 0:
        return None

    centered = X_cells_features - np.mean(X_cells_features, axis=0, keepdims=True)
    max_rank = min(centered.shape[0], centered.shape[1])
    ndim = min(svd_dim, max_rank)
    if ndim <= 0:
        return None

    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return vt[:ndim].T


def _subtract_biological_components(
    correction: np.ndarray,
    span1: np.ndarray | None,
    span2: np.ndarray | None,
) -> np.ndarray:
    """Remove correction components parallel to either biological span."""
    corrected = correction.copy()
    for span in (span1, span2):
        if span is None or span.size == 0:
            continue
        corrected = corrected - (corrected @ span) @ span.T
    return corrected


def _adjust_shift_variance(
    *,
    reference_data: np.ndarray,
    target_data: np.ndarray,
    correction: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Apply batchelor-style variance adjustment to smoothed correction vectors."""
    adjusted = correction.copy()
    for cell_idx in range(correction.shape[0]):
        scale = _compute_variance_adjustment_scale(
            reference_data=reference_data,
            target_data=target_data,
            correction_vector=correction[cell_idx],
            target_cell_idx=cell_idx,
            sigma=sigma,
        )
        adjusted[cell_idx] *= max(scale, 1.0)
    return adjusted


def _compute_variance_adjustment_scale(
    *,
    reference_data: np.ndarray,
    target_data: np.ndarray,
    correction_vector: np.ndarray,
    target_cell_idx: int,
    sigma: float,
) -> float:
    """Compute the quantile-matching scale factor for one target-cell correction."""
    l2norm = float(np.linalg.norm(correction_vector))
    if l2norm == 0.0:
        return 1.0

    unit_vector = correction_vector / l2norm
    coords_target = target_data @ unit_vector
    coords_reference = reference_data @ unit_vector

    target_cell = target_data[target_cell_idx]
    dist_target = _orthogonal_distance_sq(target_data - target_cell, unit_vector)
    dist_reference = _orthogonal_distance_sq(reference_data - target_cell, unit_vector)

    weight_target = np.exp(-dist_target / sigma)
    weight_reference = np.exp(-dist_reference / sigma)
    if float(np.sum(weight_target)) == 0.0 or float(np.sum(weight_reference)) == 0.0:
        return 1.0

    rank_target = _rank_first(coords_target)
    target_rank = rank_target[target_cell_idx]
    prob_target = float(np.sum(weight_target[rank_target <= target_rank]) / np.sum(weight_target))

    ord_reference = np.argsort(coords_reference, kind="mergesort")
    ecdf_reference = np.cumsum(weight_reference[ord_reference]) / np.sum(weight_reference)
    quantile_idx = int(np.searchsorted(ecdf_reference, prob_target, side="left"))
    quantile_idx = min(quantile_idx, len(ord_reference) - 1)

    quantile_reference = float(coords_reference[ord_reference[quantile_idx]])
    quantile_target = float(coords_target[target_cell_idx])
    return (quantile_reference - quantile_target) / l2norm


def _orthogonal_distance_sq(deltas: np.ndarray, unit_vector: np.ndarray) -> np.ndarray:
    """Squared distances after removing the component along ``unit_vector``."""
    projection = deltas @ unit_vector
    residual = deltas - np.outer(projection, unit_vector)
    return np.sum(residual * residual, axis=1)


def _rank_first(values: np.ndarray) -> np.ndarray:
    """Return 1-based ranks with first-observation tie breaking."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.int64)
    ranks[order] = np.arange(1, values.shape[0] + 1, dtype=np.int64)
    return ranks


def _average_vectors_by_group(
    vectors: np.ndarray,
    group_ids: np.ndarray,
    *,
    n_groups: int,
) -> np.ndarray:
    """Average correction vectors for each matched target cell."""
    summed = np.zeros((n_groups, vectors.shape[1]), dtype=np.float64)
    np.add.at(summed, group_ids, vectors)

    counts = np.bincount(group_ids, minlength=n_groups).astype(np.float64)
    counts = np.where(counts > 0, counts, 1.0)
    return summed / counts[:, None]


def _gaussian_smooth_vectors(
    center_vectors: np.ndarray,
    centers: np.ndarray,
    query_points: np.ndarray,
    *,
    sigma: float,
) -> np.ndarray:
    """Smooth correction vectors from matched cells onto all target cells."""
    dist_sq = _pairwise_squared_distances(query_points, centers)
    weights = np.exp(-dist_sq / (2.0 * sigma * sigma))
    weight_sums = weights.sum(axis=1, keepdims=True)

    zero_weight_rows = np.where(weight_sums.ravel() == 0)[0]
    if zero_weight_rows.size > 0:
        nearest = np.argmin(dist_sq[zero_weight_rows], axis=1)
        weights[zero_weight_rows] = 0.0
        weights[zero_weight_rows, nearest] = 1.0
        weight_sums = weights.sum(axis=1, keepdims=True)

    return (weights @ center_vectors) / weight_sums


def _pairwise_squared_distances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Compute pairwise squared Euclidean distances between two matrices."""
    x_sq = np.sum(X * X, axis=1, keepdims=True)
    y_sq = np.sum(Y * Y, axis=1, keepdims=True).T
    dist_sq = x_sq + y_sq - 2.0 * (X @ Y.T)
    return np.maximum(dist_sq, 0.0)


__all__ = ["integrate_mnn"]
