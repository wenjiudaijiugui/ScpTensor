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
from scptensor.core._log_scale_detection import detect_logged_source_layer
from scptensor.core._rank_normalization import rank_invariance_frequency
from scptensor.core.assay_alias import resolve_assay_name

if TYPE_CHECKING:
    from scptensor.autoselect.core import StageReport
    from scptensor.core.structures import ScpContainer


# Numerical stability constant
_EPS = 1e-10
_SELECTION_METRIC_WEIGHTS = {
    "batch_removal": 0.30,
    "bio_conservation": 0.30,
    "technical_quality": 0.20,
    "balance_score": 0.20,
}
_REPORT_METRIC_NAMES = (
    "loading_bias_reduction",
    "intragroup_variation",
    "intergroup_preservation",
    "technical_variance",
    "clustering_quality",
    "batch_asw",
    "batch_mixing",
    "bio_asw",
    "signal_preservation",
)


class NormalizationEvaluator(BaseEvaluator):
    """Evaluator for normalization methods.

    This evaluator tests various normalization methods and evaluates their
    performance using literature-aligned composite axes:
    batch removal, biological conservation, technical quality, and balance
    between batch removal and biological conservation.
    Legacy fine-grained metrics are also computed for interpretability.

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
    ...     source_layer="log"
    ... )

    """

    def __init__(self) -> None:
        """Initialize the normalization evaluator."""
        self._group_columns = (
            "true_cluster",
            "cluster",
            "cell_type",
            "group",
            "label",
            "bio_group",
        )
        self._batch_columns = ("batch", "run", "experiment", "plate")
        self._trqn_freq_quantile = 0.95
        self._trqn_min_match_count = 2
        self._available_methods: dict[str, Callable] | None = None
        self._source_layer_logged: bool | None = None
        self._source_layer_log_reason = "no scale context"
        self._scale_sensitive_methods = frozenset({"norm_quantile", "norm_trqn"})
        self._metric_cache_key: tuple[int, int, str] | None = None
        self._cached_selection_metrics: dict[str, float] | None = None
        self._cached_report_metrics: dict[str, float] | None = None

    def _detect_logged_source_layer(
        self,
        container: ScpContainer,
        assay_name: str,
        source_layer: str,
    ) -> tuple[bool, str]:
        """Return whether the source layer has explicit log provenance."""
        resolved_assay_name = resolve_assay_name(container, assay_name)
        assay = container.assays.get(resolved_assay_name)
        if assay is None or source_layer not in assay.layers:
            return False, "source layer unavailable for log-scale detection"

        return detect_logged_source_layer(
            container=container,
            assay_name=resolved_assay_name,
            source_layer=source_layer,
            x=assay.layers[source_layer].X,
            detect_logged_by_distribution=False,
        )

    def _build_candidate_methods(self) -> dict[str, Callable]:
        """Build candidate normalization methods under the current scale gate."""
        from scptensor.autoselect.evaluators.base import create_wrapper
        from scptensor.normalization import (
            norm_mean,
            norm_median,
            norm_none,
            norm_quantile,
        )

        methods: dict[str, Callable] = {
            "norm_none": create_wrapper(norm_none, layer_namer="auto"),
            "norm_mean": create_wrapper(
                norm_mean,
                layer_namer="auto",
                add_global_mean=True,
            ),
            "norm_median": create_wrapper(
                norm_median,
                layer_namer="auto",
                add_global_median=True,
            ),
        }

        # Quantile-family methods are only compared automatically on layers
        # with explicit log provenance (layer naming/history).
        if self._source_layer_logged is not False:
            methods["norm_quantile"] = create_wrapper(norm_quantile, layer_namer="auto")
            methods["norm_trqn"] = create_wrapper(
                self._run_trqn_adaptive,
                layer_namer=lambda src, _: f"{src}_norm_trqn",
            )

        return methods

    def _build_method_contracts(self, method_names: list[str]) -> dict[str, dict[str, object]]:
        """Attach scale-gating metadata for normalization candidates."""
        comparison_scale = "logged" if self._source_layer_logged else "raw_or_unknown"
        source_logged = bool(self._source_layer_logged)
        contracts: dict[str, dict[str, object]] = {}
        for method_name in method_names:
            contracts[method_name] = {
                "input_scale_requirement": (
                    "logged" if method_name in self._scale_sensitive_methods else "any"
                ),
                "source_layer_logged": source_logged,
                "comparison_scale": comparison_scale,
                "candidate_scope": "stable",
            }
        return contracts

    def _attach_result_contracts(self, report) -> None:
        """Attach normalization method contracts to the stage report."""
        report.method_contracts = self._build_method_contracts(
            [result.method_name for result in report.results],
        )
        for result in report.results:
            result.method_contract = report.method_contracts.get(result.method_name)

    def _build_scale_gate_message(self, source_layer: str) -> str:
        """Summarize the normalization scale gate in report text."""
        if self._source_layer_logged:
            return (
                f"Normalization candidates were compared on logged source layer "
                f"'{source_layer}' ({self._source_layer_log_reason})."
            )
        return (
            f"Source layer '{source_layer}' has no explicit log provenance "
            f"({self._source_layer_log_reason}); AutoSelect excluded "
            "scale-sensitive methods norm_quantile and norm_trqn."
        )

    def _run_trqn_adaptive(
        self,
        container: ScpContainer,
        assay_name: str = "protein",
        source_layer: str = "raw",
        new_layer_name: str = "trqn_norm",
    ) -> ScpContainer:
        """Run TRQN with adaptive RI threshold to avoid QN-degenerate runs."""
        from scptensor.normalization import norm_trqn

        resolved_assay_name = resolve_assay_name(container, assay_name)
        assay = container.assays.get(resolved_assay_name)
        if assay is None or source_layer not in assay.layers:
            return norm_trqn(
                container=container,
                assay_name=assay_name,
                source_layer=source_layer,
                new_layer_name=new_layer_name,
            )

        x = assay.layers[source_layer].X
        if hasattr(x, "toarray"):
            x = x.toarray()
        x = np.asarray(x, dtype=float)

        if x.size == 0 or x.shape[0] < 2:
            adaptive_low_thr = 0.5
        else:
            ri_freq = rank_invariance_frequency(x.T)
            finite = ri_freq[np.isfinite(ri_freq)]
            if finite.size == 0:
                adaptive_low_thr = 0.5
            else:
                q_thr = float(np.quantile(finite, self._trqn_freq_quantile))
                # Require at least a small amount of rank co-occurrence support.
                floor = self._trqn_min_match_count / float(max(2, x.shape[0]))
                adaptive_low_thr = float(np.clip(max(q_thr, floor), 0.01, 0.5))

        return norm_trqn(
            container=container,
            assay_name=assay_name,
            source_layer=source_layer,
            new_layer_name=new_layer_name,
            low_thr=adaptive_low_thr,
        )

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
        if self._available_methods is None:
            self._available_methods = self._build_candidate_methods()
        return self._available_methods

    def run_all(
        self,
        container: ScpContainer,
        assay_name: str = "proteins",
        source_layer: str = "raw",
        keep_all: bool = False,
        **kwargs,
    ) -> tuple[ScpContainer, StageReport]:
        """Run normalization selection with explicit source-scale gating."""
        previous_logged = self._source_layer_logged
        previous_reason = self._source_layer_log_reason
        previous_methods = self._available_methods

        self._source_layer_logged, self._source_layer_log_reason = self._detect_logged_source_layer(
            container=container,
            assay_name=assay_name,
            source_layer=source_layer,
        )
        self._available_methods = None

        try:
            result_container, report = super().run_all(
                container=container,
                assay_name=assay_name,
                source_layer=source_layer,
                keep_all=keep_all,
                **kwargs,
            )
            self._attach_result_contracts(report)
            scale_message = self._build_scale_gate_message(source_layer)
            if report.recommendation_reason:
                report.recommendation_reason = f"{scale_message} {report.recommendation_reason}"
            else:
                report.recommendation_reason = scale_message
            return result_container, report
        finally:
            self._source_layer_logged = previous_logged
            self._source_layer_log_reason = previous_reason
            self._available_methods = previous_methods

    @property
    def metric_weights(self) -> dict[str, float]:
        """Return weights for each evaluation metric.

        Returns
        -------
        dict[str, float]
            Dictionary mapping metric names to their weights

        """
        return dict(_SELECTION_METRIC_WEIGHTS)

    def _zero_report_metrics(self) -> dict[str, float]:
        """Return the fail-closed reporting metric payload."""
        return dict.fromkeys(_REPORT_METRIC_NAMES, 0.0)

    def _get_cached_metric_channels(
        self,
        container: ScpContainer,
        original_container: ScpContainer,
        layer_name: str,
    ) -> tuple[dict[str, float], dict[str, float]] | None:
        """Return cached metric channels for the current evaluation call."""
        cache_key = (id(container), id(original_container), layer_name)
        if cache_key != self._metric_cache_key:
            return None
        if self._cached_selection_metrics is None or self._cached_report_metrics is None:
            return None
        return (
            self._cached_selection_metrics.copy(),
            self._cached_report_metrics.copy(),
        )

    def _set_cached_metric_channels(
        self,
        container: ScpContainer,
        original_container: ScpContainer,
        layer_name: str,
        selection_metrics: dict[str, float],
        report_metrics: dict[str, float],
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Store and return metric channels for one evaluation call."""
        self._metric_cache_key = (id(container), id(original_container), layer_name)
        self._cached_selection_metrics = selection_metrics.copy()
        self._cached_report_metrics = report_metrics.copy()
        return selection_metrics.copy(), report_metrics.copy()

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
        selection_metrics, _ = self._compute_metric_channels(
            container=container,
            original_container=original_container,
            layer_name=layer_name,
        )
        return selection_metrics

    def compute_report_metrics(
        self,
        container: ScpContainer,
        original_container: ScpContainer,
        layer_name: str,
        scores: dict[str, float],
    ) -> dict[str, float]:
        """Return normalization diagnostics that are not part of ranking."""
        del scores
        _, report_metrics = self._compute_metric_channels(
            container=container,
            original_container=original_container,
            layer_name=layer_name,
        )
        return report_metrics

    def _compute_metric_channels(
        self,
        container: ScpContainer,
        original_container: ScpContainer,
        layer_name: str,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Compute ranking metrics and reporting diagnostics together."""
        cached = self._get_cached_metric_channels(container, original_container, layer_name)
        if cached is not None:
            return cached

        zero_selection = dict.fromkeys(self.metric_weights, 0.0)
        zero_report = self._zero_report_metrics()

        # Check if layer exists
        assay = self._get_metric_assay(container)
        if assay is None:
            return self._set_cached_metric_channels(
                container,
                original_container,
                layer_name,
                zero_selection,
                zero_report,
            )
        if layer_name not in assay.layers:
            return self._set_cached_metric_channels(
                container,
                original_container,
                layer_name,
                zero_selection,
                zero_report,
            )

        # Get normalized data
        matrix = assay.layers[layer_name]
        x_normalized = matrix.X

        # Convert sparse matrix to dense if needed
        if hasattr(x_normalized, "toarray"):
            x_normalized = x_normalized.toarray()

        # Get source data for comparison.
        original_assay = self._get_metric_assay(original_container)
        source_layer = (
            self._infer_source_layer(original_assay, layer_name) if original_assay else None
        )

        if (
            original_assay is None
            or source_layer is None
            or source_layer not in original_assay.layers
        ):
            # Refuse optimistic self-comparison when the true source layer
            # cannot be reconstructed from evaluator context + naming contract.
            return self._set_cached_metric_channels(
                container,
                original_container,
                layer_name,
                zero_selection,
                zero_report,
            )

        x_original = original_assay.layers[source_layer].X
        if hasattr(x_original, "toarray"):
            x_original = x_original.toarray()

        groups = self._get_group_labels(container)
        batches = self._get_batch_labels(container)

        # Compute legacy metrics (kept for compatibility/reporting)
        intragroup_score = self._compute_intragroup_variation(x_normalized, container)
        intergroup_score = self._compute_intergroup_preservation(
            x_normalized,
            x_original,
            container,
        )
        technical_score = self._compute_technical_variance(x_normalized, x_original, container)
        clustering_score = self._compute_clustering_quality(x_normalized, container, groups=groups)
        signal_score = self._compute_signal_preservation(x_normalized, x_original)
        loading_bias_score = self._compute_loading_bias_reduction(x_normalized, x_original)
        batch_asw_score = self._compute_batch_asw(x_normalized, batches)
        batch_mixing_score = self._compute_batch_mixing(x_normalized, batches)
        bio_asw_score = self._compute_bio_asw(x_normalized, groups)

        # Composite metrics based on published benchmark practices:
        # - batch removal and bio conservation are jointly optimized
        # - balance term penalizes one-sided optimization.
        if batches is None:
            # Without batch labels, the batch-removal axis is undefined.
            # Fail closed instead of assigning a neutral default.
            batch_removal = 0.0
        else:
            batch_removal = self._safe_mean(
                [technical_score, batch_asw_score, batch_mixing_score],
                default=0.0,
            )
        bio_conservation = self._safe_mean(
            [intergroup_score, bio_asw_score, clustering_score],
            default=0.0,
        )
        technical_quality = self._safe_mean(
            [intragroup_score, signal_score, loading_bias_score],
            default=0.0,
        )
        balance_score = float(np.clip(1.0 - abs(batch_removal - bio_conservation), 0.0, 1.0))

        selection_metrics = {
            "batch_removal": batch_removal,
            "bio_conservation": bio_conservation,
            "technical_quality": technical_quality,
            "balance_score": balance_score,
        }
        report_metrics = {
            "loading_bias_reduction": loading_bias_score,
            "intragroup_variation": intragroup_score,
            "intergroup_preservation": intergroup_score,
            "technical_variance": technical_score,
            "clustering_quality": clustering_score,
            "batch_asw": batch_asw_score,
            "batch_mixing": batch_mixing_score,
            "bio_asw": bio_asw_score,
            "signal_preservation": signal_score,
        }
        return self._set_cached_metric_channels(
            container,
            original_container,
            layer_name,
            selection_metrics,
            report_metrics,
        )

    def _clear_evaluation_context(self) -> None:
        """Clear evaluator-local bookkeeping and metric caches."""
        super()._clear_evaluation_context()
        self._metric_cache_key = None
        self._cached_selection_metrics = None
        self._cached_report_metrics = None

    def _safe_mean(self, values: list[float], default: float = 0.0) -> float:
        """Compute a finite mean with a fail-closed fallback."""
        arr = np.asarray(values, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return default
        return float(np.clip(np.mean(finite), 0.0, 1.0))

    def _finite_mean(self, values: np.ndarray, default: float = 0.0) -> float:
        """Compute mean over finite values only."""
        arr = np.asarray(values, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return default
        return float(np.mean(finite))

    def _nanmean_axis(self, x: np.ndarray, axis: int) -> np.ndarray:
        """Compute axis-wise mean without empty-slice warnings."""
        finite = np.isfinite(x)
        counts = np.sum(finite, axis=axis)
        sums = np.sum(np.where(finite, x, 0.0), axis=axis)
        return np.divide(
            sums,
            counts,
            out=np.zeros_like(sums, dtype=float),
            where=counts > 0,
        )

    def _nanstd_axis(self, x: np.ndarray, axis: int) -> np.ndarray:
        """Compute axis-wise std without empty-slice warnings."""
        means = self._nanmean_axis(x, axis=axis)
        means_expanded = np.expand_dims(means, axis=axis)
        finite = np.isfinite(x)
        sq = np.where(finite, (x - means_expanded) ** 2, 0.0)
        counts = np.sum(finite, axis=axis)
        var = np.divide(
            np.sum(sq, axis=axis),
            counts,
            out=np.zeros_like(means, dtype=float),
            where=counts > 0,
        )
        return np.sqrt(var)

    def _nanmedian_axis(self, x: np.ndarray, axis: int) -> np.ndarray:
        """Compute axis-wise median over finite values only."""
        arr = np.asarray(x, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got ndim={arr.ndim}")

        if axis == 0:
            out = np.zeros(arr.shape[1], dtype=float)
            for j in range(arr.shape[1]):
                col = arr[:, j]
                finite = col[np.isfinite(col)]
                out[j] = float(np.median(finite)) if finite.size > 0 else 0.0
            return out

        if axis == 1:
            out = np.zeros(arr.shape[0], dtype=float)
            for i in range(arr.shape[0]):
                row = arr[i, :]
                finite = row[np.isfinite(row)]
                out[i] = float(np.median(finite)) if finite.size > 0 else 0.0
            return out

        raise ValueError(f"axis must be 0 or 1, got {axis}")

    def _prepare_metric_matrix(
        self,
        x: np.ndarray,
        min_samples: int = 4,
        min_features: int = 2,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Prepare dense matrix for distance-based metrics under high missingness.

        Returns a tuple of:
        - matrix with finite values (NaN imputed by per-feature medians),
        - boolean mask indicating retained samples from the original matrix.
        """
        arr = np.asarray(x, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < min_samples or arr.shape[1] < min_features:
            return None

        finite = np.isfinite(arr)
        feature_valid_counts = np.sum(finite, axis=0)
        feature_mask = feature_valid_counts >= 2
        if int(np.sum(feature_mask)) < min_features:
            return None

        arr_feat = arr[:, feature_mask]
        finite_feat = np.isfinite(arr_feat)

        sample_valid_counts = np.sum(finite_feat, axis=1)
        sample_mask = sample_valid_counts >= 2
        if int(np.sum(sample_mask)) < min_samples:
            return None

        arr_eval = arr_feat[sample_mask]
        finite_eval = np.isfinite(arr_eval)
        medians = self._nanmedian_axis(arr_eval, axis=0)
        filled = np.where(finite_eval, arr_eval, medians[None, :])
        x_eval = np.nan_to_num(filled, nan=0.0, posinf=0.0, neginf=0.0)
        return x_eval, sample_mask

    def _get_group_labels(self, container: ScpContainer) -> np.ndarray | None:
        """Return biological group labels if available."""
        for col in self._group_columns:
            if col in container.obs.columns:
                groups = container.obs[col].to_numpy()
                if len(np.unique(groups)) >= 2:
                    return groups
        return None

    def _get_batch_labels(self, container: ScpContainer) -> np.ndarray | None:
        """Return batch labels if available."""
        for col in self._batch_columns:
            if col in container.obs.columns:
                batches = container.obs[col].to_numpy()
                if len(np.unique(batches)) >= 2:
                    return batches
        return None

    def _encode_labels(self, labels: np.ndarray) -> np.ndarray:
        """Encode arbitrary labels to contiguous integer labels."""
        _, encoded = np.unique(labels, return_inverse=True)
        return encoded.astype(np.int64, copy=False)

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
        x: np.ndarray,
        container: ScpContainer,
    ) -> float:
        """Compute intragroup variation reduction score.

        Measures how well normalization reduces variation within biological groups.
        Lower intragroup variation indicates better normalization.

        Parameters
        ----------
        x : np.ndarray
            Normalized data matrix (samples x features)
        container : ScpContainer
            Container with sample metadata

        Returns
        -------
        float
            Score between 0.0 and 1.0 (higher is better)

        """
        groups = self._get_group_labels(container)
        if groups is None:
            # No group info: use CV stability as fallback
            return self._compute_cv_stability(x)

        unique_groups = np.unique(groups)

        if len(unique_groups) < 2:
            return self._compute_cv_stability(x)

        # Compute within-group CV for each group
        group_cvs = []
        for group_id in unique_groups:
            group_mask = groups == group_id
            if np.sum(group_mask) < 2:
                continue

            x_group = x[group_mask]

            # Compute CV for each feature
            means = self._nanmean_axis(x_group, axis=0)
            stds = self._nanstd_axis(x_group, axis=0)
            valid_counts = np.sum(np.isfinite(x_group), axis=0)

            # Avoid division by zero
            valid_mask = (means > _EPS) & (valid_counts >= 2)
            if not np.any(valid_mask):
                continue

            cvs = np.zeros_like(means)
            cvs[valid_mask] = stds[valid_mask] / means[valid_mask]

            # Median CV for this group
            finite_cvs = cvs[np.isfinite(cvs) & valid_mask]
            if finite_cvs.size > 0:
                group_cvs.append(float(np.median(finite_cvs)))

        if len(group_cvs) == 0:
            return 0.0

        # Average CV across groups
        mean_cv = np.mean(group_cvs)

        # Transform to score: lower CV -> higher score
        # CV typically ranges from 0.1 to 2.0 in biological data
        # Use exponential decay: score = exp(-cv)
        score = np.exp(-mean_cv)

        return float(np.clip(score, 0.0, 1.0))

    def _compute_intergroup_preservation(
        self,
        x_normalized: np.ndarray,
        x_original: np.ndarray,
        container: ScpContainer,
    ) -> float:
        """Compute intergroup preservation score.

        Measures how well biological differences between groups are preserved
        after normalization. Good normalization should preserve or enhance
        intergroup separation.

        Parameters
        ----------
        x_normalized : np.ndarray
            Normalized data matrix (samples x features)
        x_original : np.ndarray
            Original data matrix before normalization
        container : ScpContainer
            Container with sample metadata

        Returns
        -------
        float
            Score between 0.0 and 1.0 (higher is better)

        """
        groups = self._get_group_labels(container)
        if groups is None:
            # No group info: use signal preservation as fallback
            return self._compute_signal_preservation(x_normalized, x_original)

        unique_groups = np.unique(groups)

        if len(unique_groups) < 2:
            return self._compute_signal_preservation(x_normalized, x_original)

        # Weak-separation groups are not reliable for preservation scoring.
        # Skip non-informative comparisons and fail closed if none remain.
        feature_stds = self._nanstd_axis(x_original, axis=0)
        global_scale = self._finite_mean(feature_stds, default=0.0)
        min_effect = 0.08 * max(global_scale, _EPS)

        pair_scores: list[float] = []
        for i in range(len(unique_groups)):
            for j in range(i + 1, len(unique_groups)):
                g_i = unique_groups[i]
                g_j = unique_groups[j]
                mask_i = groups == g_i
                mask_j = groups == g_j
                if int(np.sum(mask_i)) < 2 or int(np.sum(mask_j)) < 2:
                    continue

                orig_i = self._nanmean_axis(x_original[mask_i], axis=0)
                orig_j = self._nanmean_axis(x_original[mask_j], axis=0)
                norm_i = self._nanmean_axis(x_normalized[mask_i], axis=0)
                norm_j = self._nanmean_axis(x_normalized[mask_j], axis=0)

                effect_orig = orig_i - orig_j
                effect_norm = norm_i - norm_j
                valid = np.isfinite(effect_orig) & np.isfinite(effect_norm)
                if int(np.sum(valid)) < 10:
                    continue

                eo = effect_orig[valid]
                en = effect_norm[valid]

                effect_strength = float(np.mean(np.abs(eo)))
                if effect_strength < min_effect:
                    continue

                if np.std(eo, ddof=0) < _EPS or np.std(en, ddof=0) < _EPS:
                    continue

                corr = float(np.corrcoef(eo, en)[0, 1])
                if not np.isfinite(corr):
                    continue
                corr_score = float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))

                norm_eo = float(np.linalg.norm(eo))
                norm_en = float(np.linalg.norm(en))
                if norm_eo < _EPS:
                    continue
                ratio = max(norm_en / (norm_eo + _EPS), _EPS)
                mag_score = float(np.exp(-abs(np.log(ratio))))

                pair_scores.append(float(np.clip(0.6 * corr_score + 0.4 * mag_score, 0.0, 1.0)))

        if not pair_scores:
            return 0.0
        return float(np.clip(np.mean(pair_scores), 0.0, 1.0))

    def _compute_technical_variance(
        self,
        x_normalized: np.ndarray,
        x_original: np.ndarray,
        container: ScpContainer,
    ) -> float:
        """Compute technical variance reduction score.

        Measures how much batch-related variance was reduced.
        Uses batch information from container.obs if available.

        Parameters
        ----------
        x_normalized : np.ndarray
            Normalized data matrix (samples x features)
        x_original : np.ndarray
            Original data matrix before normalization
        container : ScpContainer
            Container with sample metadata (including batch info)

        Returns
        -------
        float
            Score between 0.0 and 1.0 (higher is better)

        """
        batches = self._get_batch_labels(container)
        if batches is None:
            # Technical batch-variance reduction is undefined without
            # explicit batch labels; refuse surrogate scoring.
            return 0.0

        unique_batches = np.unique(batches)

        if len(unique_batches) < 2:
            return 0.0

        # Compute batch centroid distances before and after normalization
        def compute_batch_separation(x: np.ndarray, batches: np.ndarray) -> float:
            batch_means = []
            for batch_id in unique_batches:
                mask = batches == batch_id
                if np.sum(mask) > 0:
                    batch_mean = self._nanmean_axis(x[mask], axis=0)
                    if np.any(np.isfinite(batch_mean)):
                        batch_means.append(batch_mean)

            if len(batch_means) < 2:
                return 0.0

            # Compute pairwise distances between batch centroids
            total_dist = 0.0
            count = 0
            for i in range(len(batch_means)):
                for j in range(i + 1, len(batch_means)):
                    delta = np.abs(batch_means[i] - batch_means[j])
                    finite = delta[np.isfinite(delta)]
                    if finite.size == 0:
                        continue
                    total_dist += float(np.mean(finite))
                    count += 1

            return total_dist / count if count > 0 else 0.0

        original_separation = compute_batch_separation(x_original, batches)
        normalized_separation = compute_batch_separation(x_normalized, batches)

        if original_separation < _EPS:
            return 0.0

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
        x: np.ndarray,
        container: ScpContainer,
        groups: np.ndarray | None = None,
    ) -> float:
        """Compute clustering quality score.

        Evaluates how well the normalized data supports meaningful clustering
        using silhouette score and Calinski-Harabasz index.

        Parameters
        ----------
        x : np.ndarray
            Normalized data matrix (samples x features)
        container : ScpContainer
            Container with sample metadata

        Returns
        -------
        float
            Score between 0.0 and 1.0 (higher is better)

        """
        prepared = self._prepare_metric_matrix(x, min_samples=4, min_features=2)
        if prepared is None:
            return 0.0
        x_eval, sample_mask = prepared

        if groups is None:
            groups = self._get_group_labels(container)

        # Supervised branch: when biological labels exist, evaluate direct
        # separation quality for known groups.
        if groups is not None and len(np.unique(groups)) >= 2:
            try:
                from sklearn.metrics import calinski_harabasz_score as ch_score
                from sklearn.metrics import silhouette_score

                y_eval = groups[sample_mask]
                if len(np.unique(y_eval)) < 2:
                    return 0.0

                _, counts = np.unique(y_eval, return_counts=True)
                if int(np.min(counts)) < 2:
                    return 0.0

                encoded = self._encode_labels(y_eval)
                sil_raw = float(silhouette_score(x_eval, encoded))
                sil_score = float(np.clip((sil_raw + 1.0) / 2.0, 0.0, 1.0))
                ch_raw = float(ch_score(x_eval, encoded))
                ch_normalized = float(np.clip(ch_raw / (ch_raw + 100.0), 0.0, 1.0))
                return float(np.clip(0.6 * sil_score + 0.4 * ch_normalized, 0.0, 1.0))
            except Exception:
                return 0.0

        # Unsupervised fallback branch when no labels are available.
        n_clusters = max(2, int(np.sqrt(x_eval.shape[0] / 2)))
        n_clusters = min(n_clusters, x_eval.shape[0] - 1)
        if n_clusters < 2:
            return 0.0

        try:
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(x_eval)

            # Check if clustering produced valid labels
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                return 0.0

            # Compute silhouette score
            from sklearn.metrics import silhouette_score

            sil_raw = float(silhouette_score(x_eval, labels))
            # Silhouette is in [-1, 1], map to [0, 1].
            sil_score = float(np.clip((sil_raw + 1.0) / 2.0, 0.0, 1.0))

            # Compute Calinski-Harabasz score
            from sklearn.metrics import calinski_harabasz_score as ch_score

            ch_raw = ch_score(x_eval, labels)

            # Normalize CH score to [0, 1]
            # CH scores can vary widely, use sigmoid-like transformation
            ch_normalized = ch_raw / (ch_raw + 100.0)

            # Combine scores with equal weights
            combined_score = 0.5 * sil_score + 0.5 * ch_normalized

            return float(np.clip(combined_score, 0.0, 1.0))

        except Exception:
            return 0.0

    def _compute_batch_asw(
        self,
        x: np.ndarray,
        batches: np.ndarray | None,
    ) -> float:
        """Compute 1-ASW batch mixing score (higher is better)."""
        if batches is None:
            return 0.0

        try:
            from sklearn.metrics import silhouette_score

            prepared = self._prepare_metric_matrix(x, min_samples=4, min_features=2)
            if prepared is None:
                return 0.0

            x_eval, sample_mask = prepared
            y_eval = self._encode_labels(np.asarray(batches)[sample_mask])
            if len(np.unique(y_eval)) < 2:
                return 0.0
            _, counts = np.unique(y_eval, return_counts=True)
            if int(np.min(counts)) < 2:
                return 0.0

            asw = float(silhouette_score(x_eval, y_eval))
            return float(np.clip(1.0 - ((asw + 1.0) / 2.0), 0.0, 1.0))
        except Exception:
            return 0.0

    def _compute_bio_asw(
        self,
        x: np.ndarray,
        groups: np.ndarray | None,
    ) -> float:
        """Compute biological ASW score (higher is better)."""
        if groups is None:
            return 0.0

        try:
            from sklearn.metrics import silhouette_score

            prepared = self._prepare_metric_matrix(x, min_samples=4, min_features=2)
            if prepared is None:
                return 0.0

            x_eval, sample_mask = prepared
            y_eval = self._encode_labels(np.asarray(groups)[sample_mask])
            if len(np.unique(y_eval)) < 2:
                return 0.0
            _, counts = np.unique(y_eval, return_counts=True)
            if int(np.min(counts)) < 2:
                return 0.0

            asw = float(silhouette_score(x_eval, y_eval))
            return float(np.clip((asw + 1.0) / 2.0, 0.0, 1.0))
        except Exception:
            return 0.0

    def _compute_batch_mixing(
        self,
        x: np.ndarray,
        batches: np.ndarray | None,
        n_neighbors: int = 30,
    ) -> float:
        """Compute local batch mixing (simplified LISI proxy)."""
        if batches is None or x.shape[0] < 3:
            return 0.0

        try:
            from sklearn.neighbors import NearestNeighbors

            prepared = self._prepare_metric_matrix(x, min_samples=4, min_features=2)
            if prepared is None:
                return 0.0

            x_eval, sample_mask = prepared
            y_eval = np.asarray(batches)[sample_mask]
            unique_batches = np.unique(y_eval)
            n_batches = len(unique_batches)
            if n_batches < 2:
                return 0.0

            k = min(n_neighbors, x_eval.shape[0] - 1)
            if k < 2:
                return 0.0

            knn = NearestNeighbors(n_neighbors=k + 1)
            knn.fit(x_eval)
            _, indices = knn.kneighbors(x_eval)
            indices = indices[:, 1:]

            scores: list[float] = []
            for i in range(x_eval.shape[0]):
                neighbor_labels = y_eval[indices[i]]
                unique_neighbor = len(np.unique(neighbor_labels))
                scores.append(unique_neighbor / n_batches)

            return float(np.clip(np.mean(scores), 0.0, 1.0))
        except Exception:
            return 0.0

    def _compute_cv_stability(self, x: np.ndarray) -> float:
        """Compute coefficient of variation stability.

        Fallback metric when group information is not available.

        Parameters
        ----------
        x : np.ndarray
            Data matrix (samples x features)

        Returns
        -------
        float
            Score between 0.0 and 1.0 (higher is better)

        """
        if x.size == 0 or x.shape[0] < 2:
            return 0.0

        finite_counts = np.sum(np.isfinite(x), axis=0)
        evaluable_features = finite_counts >= 2
        if not np.any(evaluable_features):
            return 0.0

        x_eval = x[:, evaluable_features]

        # Compute CV for each feature
        means = self._nanmean_axis(x_eval, axis=0)
        stds = self._nanstd_axis(x_eval, axis=0)

        # Avoid division by zero
        valid_mask = means > _EPS
        if not np.any(valid_mask):
            return 0.0

        cvs = np.zeros_like(means)
        cvs[valid_mask] = stds[valid_mask] / means[valid_mask]

        # Compute stability: low std(CVs) / mean(CVs) = high stability
        cv_mean = self._finite_mean(cvs[valid_mask], default=0.0)
        valid_cvs = cvs[valid_mask]
        finite_cvs = valid_cvs[np.isfinite(valid_cvs)]
        if finite_cvs.size < 2:
            return 1.0

        cv_std = float(np.std(finite_cvs, ddof=0))

        if cv_mean < _EPS:
            return 0.0

        stability = 1.0 - min(cv_std / cv_mean, 1.0)
        return float(np.clip(stability, 0.0, 1.0))

    def _compute_loading_bias_reduction(
        self,
        x_normalized: np.ndarray,
        x_original: np.ndarray,
    ) -> float:
        """Score reduction of sample-level global intensity offsets.

        This metric follows a common proteomics normalization intuition:
        good normalization should reduce spread of per-sample global location
        (sample medians) while avoiding unstable behavior.
        """
        if x_normalized.shape[0] < 2 or x_normalized.shape[1] < 2:
            return 0.0

        sample_med_orig = self._nanmedian_axis(x_original, axis=1)
        sample_med_norm = self._nanmedian_axis(x_normalized, axis=1)

        def robust_dispersion(values: np.ndarray) -> float:
            arr = np.asarray(values, dtype=float)
            finite = arr[np.isfinite(arr)]
            if finite.size < 2:
                return 0.0
            center = float(np.median(finite))
            return float(np.median(np.abs(finite - center)))

        disp_orig = robust_dispersion(sample_med_orig)
        disp_norm = robust_dispersion(sample_med_norm)

        if disp_orig < _EPS:
            return 0.0

        reduction = (disp_orig - disp_norm) / disp_orig
        if reduction >= 0:
            score = 0.5 + 0.5 * min(1.0, reduction)
        else:
            score = 0.5 + 0.5 * max(-1.0, reduction)
        return float(np.clip(score, 0.0, 1.0))

    def _compute_signal_preservation(
        self,
        x_normalized: np.ndarray,
        x_original: np.ndarray,
    ) -> float:
        """Compute signal preservation score.

        Fallback metric when group information is not available.
        Measures how well the relative relationships between samples
        are preserved after normalization using correlation matrices.

        Parameters
        ----------
        x_normalized : np.ndarray
            Normalized data matrix (samples x features)
        x_original : np.ndarray
            Original data matrix before normalization

        Returns
        -------
        float
            Score between 0.0 and 1.0 (higher is better)

        """
        # Handle edge cases
        n_samples = x_normalized.shape[0]
        if n_samples < 2:
            return 0.0  # Need at least 2 samples for correlation

        # Replace NaN with 0 for correlation computation
        x_norm_clean = np.nan_to_num(x_normalized, nan=0.0)
        x_orig_clean = np.nan_to_num(x_original, nan=0.0)

        try:
            # Guard against degenerate rows that can trigger invalid warnings
            # inside np.corrcoef (zero-variance sample profiles).
            if (
                np.sum(np.nanvar(x_norm_clean, axis=1) > _EPS) < 2
                or np.sum(np.nanvar(x_orig_clean, axis=1) > _EPS) < 2
            ):
                var_norm = np.nanvar(x_norm_clean)
                var_orig = np.nanvar(x_orig_clean)
                if var_orig < _EPS or var_norm < _EPS:
                    return 0.0
                ratio = min(var_norm / var_orig, var_orig / var_norm)
                return float(np.clip(ratio, 0.0, 1.0))

            # Compute sample correlation matrices
            # Shape: (n_samples, n_samples)
            corr_normalized = np.corrcoef(x_norm_clean)
            corr_original = np.corrcoef(x_orig_clean)

            # Handle cases where correlation is NaN (constant features)
            if np.any(np.isnan(corr_normalized)) or np.any(np.isnan(corr_original)):
                # Use variance ratio as fallback
                var_norm = np.nanvar(x_norm_clean)
                var_orig = np.nanvar(x_orig_clean)

                if var_orig < _EPS or var_norm < _EPS:
                    return 0.0

                ratio = min(var_norm / var_orig, var_orig / var_norm)
                return float(np.clip(ratio, 0.0, 1.0))

            # Extract upper triangle (excluding diagonal)
            # to avoid redundant comparisons
            triu_indices = np.triu_indices(n_samples, k=1)

            corr_norm_upper = corr_normalized[triu_indices]
            corr_orig_upper = corr_original[triu_indices]

            # Compute correlation between the two correlation vectors
            if len(corr_norm_upper) < 2:
                return 0.0

            # Pearson correlation between the correlation structures
            correlation = np.corrcoef(corr_norm_upper, corr_orig_upper)[0, 1]

            if np.isnan(correlation):
                return 0.0

            # Map to [0, 1]: negative correlation -> low score
            score = (correlation + 1.0) / 2.0

            return float(np.clip(score, 0.0, 1.0))

        except (ValueError, RuntimeWarning):
            # Handle numerical issues
            return 0.0

    def _compute_variance_stability(
        self,
        x_normalized: np.ndarray,
        x_original: np.ndarray,
    ) -> float:
        """Compute variance stability score.

        Fallback metric when batch information is not available.
        Measures how much variance was reduced while maintaining stability.

        Parameters
        ----------
        x_normalized : np.ndarray
            Normalized data matrix (samples x features)
        x_original : np.ndarray
            Original data matrix before normalization

        Returns
        -------
        float
            Score between 0.0 and 1.0 (higher is better)

        """
        # Handle edge cases
        if x_normalized.shape[0] < 2 or x_normalized.shape[1] < 2:
            return 0.0

        # Compute total variance before and after normalization
        var_original = np.nanvar(x_original)
        var_normalized = np.nanvar(x_normalized)

        if var_original < _EPS:
            return 0.0

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
