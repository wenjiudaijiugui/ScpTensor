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

    def __init__(self, batch_key: str = "batch", bio_key: str | None = None):
        """Initialize the integration evaluator.

        Parameters
        ----------
        batch_key : str, optional
            Column name in obs containing batch labels, by default "batch"
        bio_key : str | None, optional
            Column name in obs containing biological group labels (e.g., cell_type)
            for computing biological ASW. If None, biological ASW is skipped.
        """
        self._batch_key = batch_key
        self._bio_key = bio_key
        self._available_methods: dict[str, Callable] | None = None

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

        # ComBat is always available (built-in)
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

        # MNN is always available (built-in)
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

        # Harmony requires harmonypy
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
            "batch_mixing": 0.25,  # Batch mixing score
            "variance_preserved": 0.25,  # Variance preservation
        }
        if self._bio_key is not None:
            weights["bio_asw"] = 0.25
        else:
            # Redistribute weight to variance_preserved
            weights["variance_preserved"] = 0.50
        return weights

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
        if "proteins" not in container.assays:
            return dict.fromkeys(self.metric_weights, 0.0)

        assay = container.assays["proteins"]
        if layer_name not in assay.layers:
            return dict.fromkeys(self.metric_weights, 0.0)

        # Get batch labels
        if self._batch_key not in container.obs.columns:
            return dict.fromkeys(self.metric_weights, 0.0)

        batches = container.obs[self._batch_key].to_numpy()

        # Check for multiple batches
        if len(np.unique(batches)) < 2:
            return dict.fromkeys(self.metric_weights, 0.0)

        # Get data matrix
        x_matrix = assay.layers[layer_name].X
        if hasattr(x_matrix, "toarray"):
            x_matrix = x_matrix.toarray()

        # Compute metrics
        scores: dict[str, float] = {}

        # Batch ASW (1 - ASW so higher is better)
        scores["batch_asw"] = self._compute_batch_asw(x_matrix, batches)

        # Batch mixing score
        scores["batch_mixing"] = self._compute_batch_mixing(x_matrix, batches)

        # Variance preservation
        scores["variance_preserved"] = self._compute_variance_preserved(
            container, original_container, layer_name
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
        from sklearn.metrics import silhouette_score

        try:
            # Subsample if too large
            if x_data.shape[0] > 5000:
                idx = np.random.choice(x_data.shape[0], 5000, replace=False)
                x_sub = x_data[idx]
                batches_sub = batches[idx]
            else:
                x_sub = x_data
                batches_sub = batches

            # Handle NaN
            valid_mask = ~np.isnan(x_sub).any(axis=1)
            if not np.any(valid_mask):
                return 0.0

            x_clean = x_sub[valid_mask]
            batches_clean = batches_sub[valid_mask]

            if len(np.unique(batches_clean)) < 2:
                return 0.0

            asw = silhouette_score(x_clean, batches_clean)
            # Return 1 - ASW so higher is better
            return float(np.clip(1.0 - asw, 0.0, 1.0))
        except Exception:
            return 0.0

    def _compute_bio_asw(self, x_data: np.ndarray, bio_labels: np.ndarray) -> float:
        """Compute biological group average silhouette width."""
        from sklearn.metrics import silhouette_score

        try:
            # Subsample if too large
            if x_data.shape[0] > 5000:
                idx = np.random.choice(x_data.shape[0], 5000, replace=False)
                x_sub = x_data[idx]
                bio_sub = bio_labels[idx]
            else:
                x_sub = x_data
                bio_sub = bio_labels

            # Handle NaN
            valid_mask = ~np.isnan(x_sub).any(axis=1)
            if not np.any(valid_mask):
                return 0.0

            x_clean = x_sub[valid_mask]
            bio_clean = bio_sub[valid_mask]

            if len(np.unique(bio_clean)) < 2:
                return 0.0

            asw = silhouette_score(x_clean, bio_clean)
            return float(np.clip(asw, 0.0, 1.0))
        except Exception:
            return 0.0

    def _compute_batch_mixing(
        self, x_data: np.ndarray, batches: np.ndarray, n_neighbors: int = 30
    ) -> float:
        """Compute batch mixing score using simplified LISI."""
        from sklearn.neighbors import NearestNeighbors

        try:
            unique_batches = np.unique(batches)
            n_batches = len(unique_batches)

            if n_batches < 2:
                return 0.0

            n_neighbors = min(n_neighbors, x_data.shape[0] - 1)
            if n_neighbors < 1:
                return 0.0

            # Handle NaN
            valid_mask = ~np.isnan(x_data).any(axis=1)
            if not np.any(valid_mask):
                return 0.0

            x_clean = x_data[valid_mask]
            batches_clean = batches[valid_mask]

            # Find nearest neighbors
            nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
            nn.fit(x_clean)
            _, indices = nn.kneighbors(x_clean)

            # Compute mixing score
            scores = []
            for i in range(len(x_clean)):
                neighbor_batches = batches_clean[indices[i, 1:]]
                unique_in_neighborhood = len(np.unique(neighbor_batches))
                scores.append(unique_in_neighborhood / n_batches)

            return float(np.mean(scores))
        except Exception:
            return 0.0

    def _compute_variance_preserved(
        self,
        container: ScpContainer,
        original_container: ScpContainer,
        layer_name: str,
    ) -> float:
        """Compute variance preservation score."""
        import numpy as np

        try:
            # Get original variance
            if "proteins" not in original_container.assays:
                return 0.0

            original_assay = original_container.assays["proteins"]
            # Find the source layer (assume it's the first layer or "imputed")
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
            assay = container.assays["proteins"]
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


__all__ = ["IntegrationEvaluator"]
