"""Integration (batch correction) evaluator for automatic method selection.

This module provides an evaluator for testing and comparing different
batch correction/integration methods.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from scptensor.autoselect.evaluators.base import BaseEvaluator

if TYPE_CHECKING:
    from scptensor.autoselect.core import StageReport
    from scptensor.core.structures import ScpContainer


class IntegrationEvaluator(BaseEvaluator):
    """Evaluator for batch correction/integration methods.

    This evaluator tests various integration methods and evaluates their
    performance using metrics such as batch mixing, biological signal
    preservation, and variance preservation.

    Attributes
    ----------
    stage_name : str
        Name of the analysis stage ("integrate")
    methods : dict[str, Callable]
        Dictionary of integration methods to test
    metric_weights : dict[str, float]
        Weights for evaluation metrics

    Examples
    --------
    >>> evaluator = IntegrationEvaluator(batch_key="batch")
    >>> result_container, report = evaluator.run_all(
    ...     container=data,
    ...     assay_name="proteins",
    ...     source_layer="imputed"
    ... )
    """

    def __init__(
        self,
        batch_key: str = "batch",
        bio_key: str | None = None,
        include_embedding_methods: bool = False,
    ):
        """Initialize the integration evaluator.

        Parameters
        ----------
        batch_key : str, optional
            Column name in obs containing batch labels, by default "batch"
        bio_key : str | None, optional
            Column name in obs containing biological group labels (e.g., cell_type)
            for computing biological ASW. If None, biological ASW is skipped.
        include_embedding_methods : bool, optional
            Whether to include embedding-only integration methods such as
            MNN/Harmony/Scanorama. Disabled by default so stable integration
            only compares matrix-level methods recommended for downstream
            differential analysis.
        """
        self._batch_key = batch_key
        self._bio_key = bio_key
        self._include_embedding_methods = include_embedding_methods
        self._available_methods: dict[str, Callable] | None = None

    def _method_is_allowed(self, method_name: str) -> bool:
        """Return whether a method satisfies the current integration contract."""
        from scptensor.integration import get_integrate_method_info

        method_info = get_integrate_method_info(method_name)
        if method_info.integration_level == "matrix" and method_info.recommended_for_de:
            return True
        return self._include_embedding_methods and method_info.integration_level == "embedding"

    def _build_method_contracts(self, method_names: list[str]) -> dict[str, dict[str, object]]:
        """Collect per-method contract metadata for the stage report."""
        from scptensor.integration import get_integrate_method_info

        contracts: dict[str, dict[str, object]] = {}
        for method_name in method_names:
            method_info = get_integrate_method_info(method_name)
            is_stable = method_info.integration_level == "matrix" and method_info.recommended_for_de
            contracts[method_name] = {
                "integration_level": method_info.integration_level,
                "recommended_for_de": method_info.recommended_for_de,
                "candidate_scope": "stable" if is_stable else "exploratory",
                "selection_batch_metric": "batch_mixing",
                "selection_batch_metric_kind": "heuristic_proxy",
                "standardized_batch_metrics": ["batch_kbet", "batch_ilisi"],
            }
        return contracts

    def _attach_result_contracts(
        self,
        report: StageReport,
    ) -> None:
        """Attach contract metadata to the stage report and each method result."""
        report.method_contracts = self._build_method_contracts(
            [result.method_name for result in report.results]
        )
        for result in report.results:
            result.method_contract = report.method_contracts.get(result.method_name)

    def _get_available_methods(self) -> dict[str, Callable]:
        """Get available integration methods, checking for optional dependencies.

        Returns
        -------
        dict[str, Callable]
            Dictionary of available methods
        """
        from scptensor.autoselect.evaluators.base import create_wrapper

        if self._available_methods is not None:
            return self._available_methods

        methods: dict[str, Callable] = {}

        # Explicit no-batch-correction baseline
        if self._method_is_allowed("none"):
            try:
                from scptensor.integration import integrate_none

                methods["none"] = create_wrapper(
                    integrate_none,
                    source_layer_param="base_layer",
                    layer_namer="clean",
                    batch_key=self._batch_key,
                )
            except ImportError:
                pass

        # ComBat remains available as a direct API, but only enters AutoSelect
        # when its method contract marks it as stable for DE-oriented matrix use.
        if self._method_is_allowed("combat"):
            try:
                from scptensor.integration import integrate_combat

                methods["combat"] = create_wrapper(
                    integrate_combat,
                    source_layer_param="base_layer",
                    layer_namer="clean",
                    batch_key=self._batch_key,
                )
            except ImportError:
                pass

        # limma-style linear correction is built-in
        if self._method_is_allowed("limma"):
            try:
                from scptensor.integration import integrate_limma

                methods["limma"] = create_wrapper(
                    integrate_limma,
                    source_layer_param="base_layer",
                    layer_namer="clean",
                    batch_key=self._batch_key,
                )
            except ImportError:
                pass

        # MNN is exploratory because it operates at embedding level
        if self._method_is_allowed("mnn"):
            try:
                from scptensor.integration import integrate_mnn

                methods["mnn"] = create_wrapper(
                    integrate_mnn,
                    source_layer_param="base_layer",
                    layer_namer="clean",
                    batch_key=self._batch_key,
                )
            except ImportError:
                pass

        # Harmony requires harmonypy and embedding-like inputs
        if self._method_is_allowed("harmony"):
            try:
                from scptensor.integration import integrate_harmony

                methods["harmony"] = create_wrapper(
                    integrate_harmony,
                    source_layer_param="base_layer",
                    layer_namer=lambda src, _: f"{src}_harmony",
                    batch_key=self._batch_key,
                )
            except ImportError:
                pass

        # Scanorama requires scanorama package
        if self._method_is_allowed("scanorama"):
            try:
                from scptensor.integration import integrate_scanorama

                methods["scanorama"] = create_wrapper(
                    integrate_scanorama,
                    source_layer_param="base_layer",
                    layer_namer="clean",
                    batch_key=self._batch_key,
                )
            except ImportError:
                pass

        self._available_methods = methods
        return methods

    @property
    def stage_name(self) -> str:
        """Return the name of the analysis stage.

        Returns
        -------
        str
            Stage name ("integrate")
        """
        return "integrate"

    @property
    def methods(self) -> dict[str, Callable]:
        """Return dictionary of available integration methods.

        Returns
        -------
        dict[str, Callable]
            Dictionary mapping method names to their implementation functions.
            Only methods with installed dependencies are included.
        """
        return self._get_available_methods()

    def run_all(
        self,
        container: ScpContainer,
        assay_name: str = "proteins",
        source_layer: str = "raw",
        keep_all: bool = False,
        **kwargs,
    ) -> tuple[ScpContainer, StageReport]:
        """Run integration selection with contract-aware candidate filtering."""
        include_embedding_methods = kwargs.pop(
            "include_embedding_methods",
            self._include_embedding_methods,
        )
        batch_key = kwargs.get("batch_key", self._batch_key)
        bio_key = kwargs.get("bio_key", self._bio_key)

        previous_batch_key = self._batch_key
        previous_bio_key = self._bio_key
        previous_include_embedding = self._include_embedding_methods
        self._batch_key = batch_key
        self._bio_key = bio_key
        self._include_embedding_methods = include_embedding_methods
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
            if report.recommendation_reason:
                if include_embedding_methods:
                    report.recommendation_reason = (
                        "Exploratory embedding-level integration methods were included. "
                        + report.recommendation_reason
                    )
                else:
                    report.recommendation_reason = (
                        "Candidate set restricted to matrix-level methods "
                        "recommended for downstream differential analysis. "
                        + report.recommendation_reason
                    )
            return result_container, report
        finally:
            self._batch_key = previous_batch_key
            self._bio_key = previous_bio_key
            self._include_embedding_methods = previous_include_embedding
            self._available_methods = None

    @property
    def metric_weights(self) -> dict[str, float]:
        """Return weights for each evaluation metric.

        Returns
        -------
        dict[str, float]
            Dictionary mapping metric names to their weights
        """
        weights = {
            "batch_asw": 0.25,  # Batch ASW (lower is better, returns 1-asw)
            "batch_mixing": 0.25,  # Current heuristic proxy used for selection
            "variance_preserved": 0.25,  # Variance preservation
        }
        if self._bio_key is not None:
            weights["bio_asw"] = 0.25
        else:
            # Redistribute weight to variance_preserved
            weights["variance_preserved"] = 0.50
        return weights

    def _zero_metric_scores(self) -> dict[str, float]:
        """Return a zero-filled metric payload for early exits."""
        scores = dict.fromkeys(self.metric_weights, 0.0)
        scores["batch_kbet"] = 0.0
        scores["batch_ilisi"] = 0.0
        return scores

    def compute_metrics(
        self,
        container: ScpContainer,
        original_container: ScpContainer,
        layer_name: str,
    ) -> dict[str, float]:
        """Compute evaluation metrics for an integrated layer.

        Parameters
        ----------
        container : ScpContainer
            Container with the integrated data layer
        original_container : ScpContainer
            Original container before integration (for comparison)
        layer_name : str
            Name of the layer to evaluate

        Returns
        -------
        dict[str, float]
            Dictionary mapping metric names to their scores (0.0 to 1.0)
        """
        import numpy as np

        # Check if layer exists
        assay = self._get_metric_assay(container)
        if assay is None:
            return self._zero_metric_scores()
        if layer_name not in assay.layers:
            return self._zero_metric_scores()

        # Get batch labels
        if self._batch_key not in container.obs.columns:
            return self._zero_metric_scores()

        batches = container.obs[self._batch_key].to_numpy()

        # Check for multiple batches
        if len(np.unique(batches)) < 2:
            return self._zero_metric_scores()

        # Get data matrix
        x_matrix = assay.layers[layer_name].X
        if hasattr(x_matrix, "toarray"):
            x_matrix = x_matrix.toarray()

        # Compute metrics
        scores: dict[str, float] = {}

        source_layer = self._infer_source_layer(
            self._get_metric_assay(original_container), layer_name
        )

        # Batch ASW (1 - ASW so higher is better)
        scores["batch_asw"] = self._compute_batch_asw(x_matrix, batches)

        # Batch mixing score
        scores["batch_mixing"] = self._compute_batch_mixing(x_matrix, batches)
        scores["batch_kbet"] = self._compute_batch_kbet(x_matrix, batches)
        scores["batch_ilisi"] = self._compute_batch_ilisi(x_matrix, batches)

        # Variance preservation
        scores["variance_preserved"] = self._compute_variance_preserved(
            container, original_container, layer_name, source_layer=source_layer
        )

        # Biological ASW (if bio_key is provided)
        if self._bio_key is not None:
            if self._bio_key in container.obs.columns:
                bio_labels = container.obs[self._bio_key].to_numpy()
                scores["bio_asw"] = self._compute_bio_asw(x_matrix, bio_labels)
            else:
                scores["bio_asw"] = 0.0

        return scores

    def _compute_batch_asw(self, x_data: np.ndarray, batches: np.ndarray) -> float:
        """Compute batch average silhouette width (1 - ASW)."""
        from scptensor.autoselect.metrics.batch import batch_asw

        try:
            if x_data.shape[0] > 5000:
                idx = np.random.choice(x_data.shape[0], 5000, replace=False)
                x_sub = x_data[idx]
                batches_sub = batches[idx]
            else:
                x_sub = x_data
                batches_sub = batches

            return batch_asw(x_sub, batches_sub)
        except Exception:
            return 0.0

    def _compute_bio_asw(self, x_data: np.ndarray, bio_labels: np.ndarray) -> float:
        """Compute biological group average silhouette width."""
        from scptensor.autoselect.metrics.batch import bio_asw

        try:
            if x_data.shape[0] > 5000:
                idx = np.random.choice(x_data.shape[0], 5000, replace=False)
                x_sub = x_data[idx]
                bio_sub = bio_labels[idx]
            else:
                x_sub = x_data
                bio_sub = bio_labels

            return bio_asw(x_sub, bio_sub)
        except Exception:
            return 0.0

    def _compute_batch_mixing(
        self, x_data: np.ndarray, batches: np.ndarray, n_neighbors: int = 30
    ) -> float:
        """Compute the current heuristic batch-mixing proxy used for selection."""
        from scptensor.autoselect.metrics.batch import batch_mixing_score

        try:
            return batch_mixing_score(x_data, batches, n_neighbors=n_neighbors)
        except Exception:
            return 0.0

    def _compute_batch_kbet(
        self, x_data: np.ndarray, batches: np.ndarray, n_neighbors: int = 30
    ) -> float:
        """Compute standardized fixed-k kBET acceptance rate."""
        from scptensor.autoselect.metrics.batch import kbet_score

        try:
            return kbet_score(x_data, batches, n_neighbors=n_neighbors)
        except Exception:
            return 0.0

    def _compute_batch_ilisi(
        self,
        x_data: np.ndarray,
        batches: np.ndarray,
        n_neighbors: int = 60,
        perplexity: float = 30.0,
    ) -> float:
        """Compute standardized iLISI summary for reporting."""
        from scptensor.autoselect.metrics.batch import ilisi_score

        try:
            return ilisi_score(
                x_data,
                batches,
                n_neighbors=n_neighbors,
                perplexity=perplexity,
                scale=True,
            )
        except Exception:
            return 0.0

    def _compute_variance_preserved(
        self,
        container: ScpContainer,
        original_container: ScpContainer,
        layer_name: str,
        source_layer: str | None = None,
    ) -> float:
        """Compute variance preservation score."""
        import numpy as np

        try:
            # Get original variance
            original_assay = self._get_metric_assay(original_container)
            if original_assay is None:
                return 0.0

            # Prefer inferred source layer from naming convention.
            if source_layer not in original_assay.layers:
                source_layer = None
                for ln in ["imputed", "normalized", "raw"]:
                    if ln in original_assay.layers:
                        source_layer = ln
                        break

            if source_layer is None:
                return 0.5  # Default score if no source layer

            x_orig = original_assay.layers[source_layer].X
            if hasattr(x_orig, "toarray"):
                x_orig = x_orig.toarray()

            # Get integrated variance
            assay = self._get_metric_assay(container)
            if assay is None:
                return 0.0
            x_int = assay.layers[layer_name].X
            if hasattr(x_int, "toarray"):
                x_int = x_int.toarray()

            # Compute variance per feature
            var_orig = np.nanvar(x_orig, axis=0, ddof=1)
            var_int = np.nanvar(x_int, axis=0, ddof=1)

            # Compute correlation of variances
            valid_mask = ~(np.isnan(var_orig) | np.isnan(var_int) | (var_orig == 0))
            if not np.any(valid_mask):
                return 0.5

            var_orig_valid = var_orig[valid_mask]
            var_int_valid = var_int[valid_mask]

            if len(var_orig_valid) < 2:
                return 0.5

            # Pearson correlation
            corr = np.corrcoef(var_orig_valid, var_int_valid)[0, 1]
            if np.isnan(corr):
                return 0.5

            # Map correlation [-1, 1] to [0, 1]
            return float(np.clip((corr + 1) / 2, 0.0, 1.0))
        except Exception:
            return 0.5

    def _infer_source_layer(self, assay, layer_name: str) -> str | None:
        """Infer source layer from output layer naming convention."""
        if assay is None:
            return None
        candidates = [name for name in assay.layers if layer_name.startswith(f"{name}_")]
        if not candidates:
            return None
        return max(candidates, key=len)


__all__ = ["IntegrationEvaluator"]
