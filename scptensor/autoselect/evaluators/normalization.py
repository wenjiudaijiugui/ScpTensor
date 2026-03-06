"""Normalization evaluator for automatic method selection.

This module provides an evaluator for testing and comparing different
normalization methods.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from sklearn.cluster import KMeans

from scptensor.autoselect.evaluators.base import BaseEvaluator

if TYPE_CHECKING:
    from scptensor.core.structures import ScpContainer


# Numerical stability constant
_EPS = 1e-10


class NormalizationEvaluator(BaseEvaluator):
    """Evaluator for normalization methods.

    This evaluator tests various normalization methods and evaluates their
    performance using metrics such as intragroup variation, intergroup
    preservation, technical variance reduction, and clustering quality.

    Attributes
    ----------
    stage_name : str
        Name of the analysis stage ("normalization")
    methods : dict[str, Callable]
        Dictionary of normalization methods to test
    metric_weights : dict[str, float]
        Weights for evaluation metrics

    Examples
    --------
    >>> evaluator = NormalizationEvaluator()
    >>> result_container, report = evaluator.run_all(
    ...     container=data,
    ...     assay_name="proteins",
    ...     source_layer="raw"
    ... )
    """

    @property
    def stage_name(self) -> str:
        """Return the name of the analysis stage.

        Returns
        -------
        str
            Stage name ("normalize")
        """
        return "normalize"

    @property
    def methods(self) -> dict[str, Callable]:
        """Return dictionary of available normalization methods.

        Returns
        -------
        dict[str, Callable]
            Dictionary mapping method names to their implementation functions.
            Only includes true normalization methods (not log_transform, which is
            a preprocessing step).
        """
        from scptensor.autoselect.evaluators.base import create_wrapper
        from scptensor.normalization import norm_mean, norm_median, norm_quantile, norm_trqn

        return {
            "norm_mean": create_wrapper(norm_mean, layer_namer="auto"),
            "norm_median": create_wrapper(norm_median, layer_namer="auto"),
            "norm_quantile": create_wrapper(norm_quantile, layer_namer="auto"),
            "norm_trqn": create_wrapper(norm_trqn, layer_namer="auto"),
        }

    @property
    def metric_weights(self) -> dict[str, float]:
        """Return weights for each evaluation metric.

        Returns
        -------
        dict[str, float]
            Dictionary mapping metric names to their weights
        """
        return {
            "intragroup_variation": 0.25,
            "intergroup_preservation": 0.25,
            "technical_variance": 0.25,
            "clustering_quality": 0.25,
        }

    def compute_metrics(
        self,
        container: ScpContainer,
        original_container: ScpContainer,
        layer_name: str,
    ) -> dict[str, float]:
        """Compute evaluation metrics for a normalized layer.

        Parameters
        ----------
        container : ScpContainer
            Container with the normalized data layer
        original_container : ScpContainer
            Original container before normalization (for comparison)
        layer_name : str
            Name of the layer to evaluate

        Returns
        -------
        dict[str, float]
            Dictionary mapping metric names to their scores (0.0 to 1.0)
        """
        # Check if layer exists
        if "proteins" not in container.assays:
            return dict.fromkeys(self.metric_weights, 0.0)

        assay = container.assays["proteins"]
        if layer_name not in assay.layers:
            return dict.fromkeys(self.metric_weights, 0.0)

        # Get normalized data
        matrix = assay.layers[layer_name]
        X_normalized = matrix.X  # noqa: N806

        # Convert sparse matrix to dense if needed
        if hasattr(X_normalized, "toarray"):
            X_normalized = X_normalized.toarray()  # noqa: N806

        # Get source data for comparison.
        original_assay = original_container.assays.get("proteins")
        source_layer = (
            self._infer_source_layer(original_assay, layer_name) if original_assay else None
        )

        if source_layer is not None and original_assay is not None:
            X_original = original_assay.layers[source_layer].X  # noqa: N806
            if hasattr(X_original, "toarray"):
                X_original = X_original.toarray()  # noqa: N806
        else:
            # Fall back to current layer if source cannot be inferred.
            X_original = X_normalized  # noqa: N806

        # Compute metrics
        intragroup_score = self._compute_intragroup_variation(X_normalized, container)
        intergroup_score = self._compute_intergroup_preservation(
            X_normalized, X_original, container
        )
        technical_score = self._compute_technical_variance(X_normalized, X_original, container)
        clustering_score = self._compute_clustering_quality(X_normalized, container)

        return {
            "intragroup_variation": intragroup_score,
            "intergroup_preservation": intergroup_score,
            "technical_variance": technical_score,
            "clustering_quality": clustering_score,
        }

    def _infer_source_layer(self, assay, layer_name: str) -> str | None:
        """Infer source layer name from output layer naming convention."""
        if assay is None:
            return None
        candidates = [name for name in assay.layers if layer_name.startswith(f"{name}_")]
        if not candidates:
            return None
        return max(candidates, key=len)

    def _compute_intragroup_variation(
        self,
        X: np.ndarray,  # noqa: N803
        container: ScpContainer,
    ) -> float:
        """Compute intragroup variation reduction score.

        Measures how well normalization reduces variation within biological groups.
        Lower intragroup variation indicates better normalization.

        Parameters
        ----------
        X : np.ndarray
            Normalized data matrix (samples x features)
        container : ScpContainer
            Container with sample metadata

        Returns
        -------
        float
            Score between 0.0 and 1.0 (higher is better)
        """
        # Check if biological group information is available
        group_col = None
        for col in ["true_cluster", "cluster", "cell_type", "group", "label", "bio_group"]:
            if col in container.obs.columns:
                group_col = col
                break

        if group_col is None:
            # No group info: use CV stability as fallback
            return self._compute_cv_stability(X)

        # Get group assignments
        groups = container.obs[group_col].to_numpy()
        unique_groups = np.unique(groups)

        if len(unique_groups) < 2:
            return self._compute_cv_stability(X)

        # Compute within-group CV for each group
        group_cvs = []
        for group_id in unique_groups:
            group_mask = groups == group_id
            if np.sum(group_mask) < 2:
                continue

            X_group = X[group_mask]  # noqa: N806

            # Compute CV for each feature
            means = np.nanmean(X_group, axis=0)
            stds = np.nanstd(X_group, axis=0, ddof=1)

            # Avoid division by zero
            valid_mask = means > _EPS
            if not np.any(valid_mask):
                continue

            cvs = np.zeros_like(means)
            cvs[valid_mask] = stds[valid_mask] / means[valid_mask]

            # Median CV for this group
            group_cv = np.nanmedian(cvs)
            if not np.isnan(group_cv):
                group_cvs.append(group_cv)

        if len(group_cvs) == 0:
            return 0.5

        # Average CV across groups
        mean_cv = np.mean(group_cvs)

        # Transform to score: lower CV -> higher score
        # CV typically ranges from 0.1 to 2.0 in biological data
        # Use exponential decay: score = exp(-cv)
        score = np.exp(-mean_cv)

        return float(np.clip(score, 0.0, 1.0))

    def _compute_intergroup_preservation(
        self,
        X_normalized: np.ndarray,  # noqa: N803
        X_original: np.ndarray,  # noqa: N803
        container: ScpContainer,
    ) -> float:
        """Compute intergroup preservation score.

        Measures how well biological differences between groups are preserved
        after normalization. Good normalization should preserve or enhance
        intergroup separation.

        Parameters
        ----------
        X_normalized : np.ndarray
            Normalized data matrix (samples x features)
        X_original : np.ndarray
            Original data matrix before normalization
        container : ScpContainer
            Container with sample metadata

        Returns
        -------
        float
            Score between 0.0 and 1.0 (higher is better)
        """
        # Check if biological group information is available
        group_col = None
        for col in ["true_cluster", "cluster", "cell_type", "group", "label", "bio_group"]:
            if col in container.obs.columns:
                group_col = col
                break

        if group_col is None:
            # No group info: use signal preservation as fallback
            return self._compute_signal_preservation(X_normalized, X_original)

        # Get group assignments
        groups = container.obs[group_col].to_numpy()
        unique_groups = np.unique(groups)

        if len(unique_groups) < 2:
            return self._compute_signal_preservation(X_normalized, X_original)

        def compute_group_separation(X: np.ndarray, groups: np.ndarray) -> float:  # noqa: N803
            """Compute average pairwise distance between group centroids."""
            group_means = []
            for group_id in unique_groups:
                group_mask = groups == group_id
                if np.sum(group_mask) > 0:
                    group_mean = np.nanmean(X[group_mask], axis=0)
                    group_means.append(group_mean)

            if len(group_means) < 2:
                return 0.0

            # Compute pairwise distances between group centroids
            distances = []
            for i in range(len(group_means)):
                for j in range(i + 1, len(group_means)):
                    dist = np.nanmean(np.abs(group_means[i] - group_means[j]))
                    distances.append(dist)

            return np.mean(distances) if distances else 0.0

        # Compute group separation in original and normalized data
        original_separation = compute_group_separation(X_original, groups)
        normalized_separation = compute_group_separation(X_normalized, groups)

        # Score based on preservation or enhancement of group separation
        if original_separation > _EPS:
            # Ratio of separations
            ratio = normalized_separation / original_separation

            # Score: ratio >= 1 (enhanced) -> high score
            # ratio < 1 (reduced) -> lower score
            if ratio >= 1.0:
                # Enhanced separation
                score = 0.8 + 0.2 * min(1.0, (ratio - 1.0) / 2.0)
            else:
                # Reduced separation, but still some preservation
                score = ratio * 0.8
        else:
            # No original separation to compare
            score = 0.9 if normalized_separation > 0.1 else 0.7

        return float(np.clip(score, 0.0, 1.0))

    def _compute_technical_variance(
        self,
        X_normalized: np.ndarray,  # noqa: N803
        X_original: np.ndarray,  # noqa: N803
        container: ScpContainer,
    ) -> float:
        """Compute technical variance reduction score.

        Measures how much batch-related variance was reduced.
        Uses batch information from container.obs if available.

        Parameters
        ----------
        X_normalized : np.ndarray
            Normalized data matrix (samples x features)
        X_original : np.ndarray
            Original data matrix before normalization
        container : ScpContainer
            Container with sample metadata (including batch info)

        Returns
        -------
        float
            Score between 0.0 and 1.0 (higher is better)
        """
        # Check if batch information is available
        batch_col = None
        for col in ["batch", "run", "experiment", "plate"]:
            if col in container.obs.columns:
                batch_col = col
                break

        if batch_col is None:
            # No batch info: use variance stability as fallback
            return self._compute_variance_stability(X_normalized, X_original)

        # Get batch assignments
        batches = container.obs[batch_col].to_numpy()
        unique_batches = np.unique(batches)

        if len(unique_batches) < 2:
            return self._compute_variance_stability(X_normalized, X_original)

        # Compute batch centroid distances before and after normalization
        def compute_batch_separation(X: np.ndarray, batches: np.ndarray) -> float:  # noqa: N803
            batch_means = []
            for batch_id in unique_batches:
                mask = batches == batch_id
                if np.sum(mask) > 0:
                    batch_mean = np.nanmean(X[mask], axis=0)
                    batch_means.append(batch_mean)

            if len(batch_means) < 2:
                return 0.0

            # Compute pairwise distances between batch centroids
            total_dist = 0.0
            count = 0
            for i in range(len(batch_means)):
                for j in range(i + 1, len(batch_means)):
                    dist = np.nanmean(np.abs(batch_means[i] - batch_means[j]))
                    total_dist += dist
                    count += 1

            return total_dist / count if count > 0 else 0.0

        original_separation = compute_batch_separation(X_original, batches)
        normalized_separation = compute_batch_separation(X_normalized, batches)

        if original_separation < _EPS:
            return 0.5  # No batch effect to begin with

        # Score based on reduction in batch separation
        # Lower separation after normalization = better batch effect removal
        reduction_ratio = (original_separation - normalized_separation) / original_separation

        # Map reduction ratio to score
        # reduction_ratio > 0 means batch effect reduced (good)
        # reduction_ratio < 0 means batch effect increased (bad)
        if reduction_ratio >= 0:
            score = 0.5 + 0.5 * min(1.0, reduction_ratio)
        else:
            score = 0.5 + 0.5 * max(-1.0, reduction_ratio)  # Will be < 0.5

        return float(np.clip(score, 0.0, 1.0))

    def _compute_clustering_quality(
        self,
        X: np.ndarray,  # noqa: N803
        container: ScpContainer,
    ) -> float:
        """Compute clustering quality score.

        Evaluates how well the normalized data supports meaningful clustering
        using silhouette score and Calinski-Harabasz index.

        Parameters
        ----------
        X : np.ndarray
            Normalized data matrix (samples x features)
        container : ScpContainer
            Container with sample metadata

        Returns
        -------
        float
            Score between 0.0 and 1.0 (higher is better)
        """
        # Handle edge cases
        if X.shape[0] < 4 or X.shape[1] < 2:
            return 0.5

        # Replace NaN with 0 for clustering
        X_clean = np.nan_to_num(X, nan=0.0)  # noqa: N806

        # Determine number of clusters
        # Check if biological group information is available
        group_col = None
        for col in ["true_cluster", "cluster", "cell_type", "group", "label", "bio_group"]:
            if col in container.obs.columns:
                group_col = col
                break

        if group_col is not None:
            # Use known number of groups
            groups = container.obs[group_col].to_numpy()
            n_clusters = len(np.unique(groups))
        else:
            # Estimate number of clusters using sqrt rule
            n_clusters = max(2, int(np.sqrt(X.shape[0] / 2)))

        n_clusters = min(n_clusters, X.shape[0] - 1)

        if n_clusters < 2:
            return 0.5

        try:
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_clean)

            # Check if clustering produced valid labels
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                return 0.5

            # Compute silhouette score
            from sklearn.metrics import silhouette_score

            sil_score = silhouette_score(X_clean, labels)

            # Clip to [0, 1]
            sil_score = float(np.clip(sil_score, 0.0, 1.0))

            # Compute Calinski-Harabasz score
            from sklearn.metrics import calinski_harabasz_score as ch_score

            ch_raw = ch_score(X_clean, labels)

            # Normalize CH score to [0, 1]
            # CH scores can vary widely, use sigmoid-like transformation
            ch_normalized = ch_raw / (ch_raw + 100.0)

            # Combine scores with equal weights
            combined_score = 0.5 * sil_score + 0.5 * ch_normalized

            return float(np.clip(combined_score, 0.0, 1.0))

        except Exception:
            # Fallback to variance-based score
            sample_variances = np.nanvar(X, axis=1)
            mean_variance = np.nanmean(sample_variances)

            if mean_variance > 0:
                score = 1.0 / (1.0 + np.log1p(mean_variance))
            else:
                score = 1.0

            return float(np.clip(score, 0.0, 1.0))

    def _compute_cv_stability(self, X: np.ndarray) -> float:  # noqa: N803
        """Compute coefficient of variation stability.

        Fallback metric when group information is not available.

        Parameters
        ----------
        X : np.ndarray
            Data matrix (samples x features)

        Returns
        -------
        float
            Score between 0.0 and 1.0 (higher is better)
        """
        if X.size == 0 or X.shape[0] < 2:
            return 0.5

        # Compute CV for each feature
        means = np.nanmean(X, axis=0)
        stds = np.nanstd(X, axis=0, ddof=1)

        # Avoid division by zero
        valid_mask = means > _EPS
        if not np.any(valid_mask):
            return 0.5

        cvs = np.zeros_like(means)
        cvs[valid_mask] = stds[valid_mask] / means[valid_mask]

        # Compute stability: low std(CVs) / mean(CVs) = high stability
        cv_mean = np.nanmean(cvs[valid_mask])
        cv_std = np.nanstd(cvs[valid_mask], ddof=1)

        if cv_mean < _EPS:
            return 0.5

        stability = 1.0 - min(cv_std / cv_mean, 1.0)
        return float(np.clip(stability, 0.0, 1.0))

    def _compute_signal_preservation(
        self,
        X_normalized: np.ndarray,  # noqa: N803
        X_original: np.ndarray,  # noqa: N803
    ) -> float:
        """Compute signal preservation score.

        Fallback metric when group information is not available.
        Measures how well the relative relationships between samples
        are preserved after normalization using correlation matrices.

        Parameters
        ----------
        X_normalized : np.ndarray
            Normalized data matrix (samples x features)
        X_original : np.ndarray
            Original data matrix before normalization

        Returns
        -------
        float
            Score between 0.0 and 1.0 (higher is better)
        """
        # Handle edge cases
        n_samples = X_normalized.shape[0]
        if n_samples < 2:
            return 0.5  # Need at least 2 samples for correlation

        # Replace NaN with 0 for correlation computation
        X_norm_clean = np.nan_to_num(X_normalized, nan=0.0)  # noqa: N806
        X_orig_clean = np.nan_to_num(X_original, nan=0.0)  # noqa: N806

        try:
            # Compute sample correlation matrices
            # Shape: (n_samples, n_samples)
            corr_normalized = np.corrcoef(X_norm_clean)
            corr_original = np.corrcoef(X_orig_clean)

            # Handle cases where correlation is NaN (constant features)
            if np.any(np.isnan(corr_normalized)) or np.any(np.isnan(corr_original)):
                # Use variance ratio as fallback
                var_norm = np.nanvar(X_norm_clean)
                var_orig = np.nanvar(X_orig_clean)

                if var_orig < _EPS:
                    return 0.5

                ratio = min(var_norm / var_orig, var_orig / var_norm)
                return float(np.clip(ratio, 0.0, 1.0))

            # Extract upper triangle (excluding diagonal)
            # to avoid redundant comparisons
            triu_indices = np.triu_indices(n_samples, k=1)

            corr_norm_upper = corr_normalized[triu_indices]
            corr_orig_upper = corr_original[triu_indices]

            # Compute correlation between the two correlation vectors
            if len(corr_norm_upper) < 2:
                return 0.5

            # Pearson correlation between the correlation structures
            correlation = np.corrcoef(corr_norm_upper, corr_orig_upper)[0, 1]

            if np.isnan(correlation):
                return 0.5

            # Map to [0, 1]: negative correlation -> low score
            score = (correlation + 1.0) / 2.0

            return float(np.clip(score, 0.0, 1.0))

        except (ValueError, RuntimeWarning):
            # Handle numerical issues
            return 0.5

    def _compute_variance_stability(
        self,
        X_normalized: np.ndarray,  # noqa: N803
        X_original: np.ndarray,  # noqa: N803
    ) -> float:
        """Compute variance stability score.

        Fallback metric when batch information is not available.
        Measures how much variance was reduced while maintaining stability.

        Parameters
        ----------
        X_normalized : np.ndarray
            Normalized data matrix (samples x features)
        X_original : np.ndarray
            Original data matrix before normalization

        Returns
        -------
        float
            Score between 0.0 and 1.0 (higher is better)
        """
        # Handle edge cases
        if X_normalized.shape[0] < 2 or X_normalized.shape[1] < 2:
            return 0.5

        # Compute total variance before and after normalization
        var_original = np.nanvar(X_original)
        var_normalized = np.nanvar(X_normalized)

        if var_original < _EPS:
            return 0.5

        # Variance reduction ratio
        reduction_ratio = (var_original - var_normalized) / var_original

        # Moderate reduction is good (removes technical noise)
        # Too much reduction is bad (removes biological signal)
        # Too little reduction is bad (doesn't normalize well)

        # Ideal reduction: 20-50%
        if 0.2 <= reduction_ratio <= 0.5:
            score = 0.9 + 0.1 * (1.0 - abs(reduction_ratio - 0.35) / 0.15)
        elif 0.1 <= reduction_ratio < 0.2:
            # Slight under-normalization
            score = 0.7 + 0.2 * (reduction_ratio / 0.1)
        elif 0.5 < reduction_ratio <= 0.7:
            # Slight over-normalization
            score = 0.7 + 0.2 * ((0.7 - reduction_ratio) / 0.2)
        elif reduction_ratio < 0.1:
            # Under-normalized
            score = reduction_ratio / 0.1 * 0.7
        elif reduction_ratio > 0.7:
            # Over-normalized
            score = max(0.0, 0.7 - (reduction_ratio - 0.7) * 2.0)
        else:
            score = 0.7

        return float(np.clip(score, 0.0, 1.0))
