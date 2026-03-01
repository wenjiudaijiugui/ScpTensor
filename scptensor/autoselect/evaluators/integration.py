"""Integration (batch correction) evaluator for automatic method selection.

This module provides an evaluator for testing and comparing different
batch correction/integration methods.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

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
        if self._available_methods is not None:
            return self._available_methods

        methods: dict[str, Callable] = {}

        # ComBat is always available (built-in)
        try:
            from scptensor.integration import integrate_combat

            methods["combat"] = self._wrap_integrate(integrate_combat)
        except ImportError:
            pass

        # MNN is always available (built-in)
        try:
            from scptensor.integration import integrate_mnn

            methods["mnn"] = self._wrap_integrate(integrate_mnn)
        except ImportError:
            pass

        # Harmony requires harmonypy
        try:
            from scptensor.integration import integrate_harmony

            methods["harmony"] = self._wrap_integrate_harmony(integrate_harmony)
        except ImportError:
            pass

        # Scanorama requires scanorama package
        try:
            from scptensor.integration import integrate_scanorama

            methods["scanorama"] = self._wrap_integrate(integrate_scanorama)
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
        X = assay.layers[layer_name].X
        if hasattr(X, "toarray"):
            X = X.toarray()

        # Compute metrics
        scores: dict[str, float] = {}

        # Batch ASW (1 - ASW so higher is better)
        scores["batch_asw"] = self._compute_batch_asw(X, batches)

        # Batch mixing score
        scores["batch_mixing"] = self._compute_batch_mixing(X, batches)

        # Variance preservation
        scores["variance_preserved"] = self._compute_variance_preserved(
            container, original_container, layer_name
        )

        # Biological ASW (if bio_key is provided)
        if self._bio_key is not None:
            if self._bio_key in container.obs.columns:
                bio_labels = container.obs[self._bio_key].to_numpy()
                scores["bio_asw"] = self._compute_bio_asw(X, bio_labels)
            else:
                scores["bio_asw"] = 0.0

        return scores

    def _compute_batch_asw(self, X: np.ndarray, batches: np.ndarray) -> float:
        """Compute batch average silhouette width (1 - ASW)."""
        from sklearn.metrics import silhouette_score

        try:
            # Subsample if too large
            if X.shape[0] > 5000:
                idx = np.random.choice(X.shape[0], 5000, replace=False)
                X_sub = X[idx]
                batches_sub = batches[idx]
            else:
                X_sub = X
                batches_sub = batches

            # Handle NaN
            valid_mask = ~np.isnan(X_sub).any(axis=1)
            if not np.any(valid_mask):
                return 0.0

            X_clean = X_sub[valid_mask]
            batches_clean = batches_sub[valid_mask]

            if len(np.unique(batches_clean)) < 2:
                return 0.0

            asw = silhouette_score(X_clean, batches_clean)
            # Return 1 - ASW so higher is better
            return float(np.clip(1.0 - asw, 0.0, 1.0))
        except Exception:
            return 0.0

    def _compute_bio_asw(self, X: np.ndarray, bio_labels: np.ndarray) -> float:
        """Compute biological group average silhouette width."""
        from sklearn.metrics import silhouette_score

        try:
            # Subsample if too large
            if X.shape[0] > 5000:
                idx = np.random.choice(X.shape[0], 5000, replace=False)
                X_sub = X[idx]
                bio_sub = bio_labels[idx]
            else:
                X_sub = X
                bio_sub = bio_labels

            # Handle NaN
            valid_mask = ~np.isnan(X_sub).any(axis=1)
            if not np.any(valid_mask):
                return 0.0

            X_clean = X_sub[valid_mask]
            bio_clean = bio_sub[valid_mask]

            if len(np.unique(bio_clean)) < 2:
                return 0.0

            asw = silhouette_score(X_clean, bio_clean)
            return float(np.clip(asw, 0.0, 1.0))
        except Exception:
            return 0.0

    def _compute_batch_mixing(
        self, X: np.ndarray, batches: np.ndarray, n_neighbors: int = 30
    ) -> float:
        """Compute batch mixing score using simplified LISI."""
        from sklearn.neighbors import NearestNeighbors

        try:
            unique_batches = np.unique(batches)
            n_batches = len(unique_batches)

            if n_batches < 2:
                return 0.0

            n_neighbors = min(n_neighbors, X.shape[0] - 1)
            if n_neighbors < 1:
                return 0.0

            # Handle NaN
            valid_mask = ~np.isnan(X).any(axis=1)
            if not np.any(valid_mask):
                return 0.0

            X_clean = X[valid_mask]
            batches_clean = batches[valid_mask]

            # Find nearest neighbors
            nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
            nn.fit(X_clean)
            _, indices = nn.kneighbors(X_clean)

            # Compute mixing score
            scores = []
            for i in range(len(X_clean)):
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

            X_orig = original_assay.layers[source_layer].X
            if hasattr(X_orig, "toarray"):
                X_orig = X_orig.toarray()

            # Get integrated variance
            assay = container.assays["proteins"]
            X_int = assay.layers[layer_name].X
            if hasattr(X_int, "toarray"):
                X_int = X_int.toarray()

            # Compute variance per feature
            var_orig = np.nanvar(X_orig, axis=0, ddof=1)
            var_int = np.nanvar(X_int, axis=0, ddof=1)

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

    def _wrap_integrate(self, func: Callable) -> Callable:
        """Wrap integration function to match expected signature.

        Parameters
        ----------
        func : Callable
            Original integration function

        Returns
        -------
        Callable
            Wrapped function with signature:
            (container, assay_name, source_layer, **kwargs) -> ScpContainer
        """

        def wrapper(
            container: ScpContainer,
            assay_name: str,
            source_layer: str,
            **kwargs,
        ) -> ScpContainer:
            """Wrapper for integration functions."""
            return func(
                container=container,
                batch_key=self._batch_key,
                assay_name=assay_name,
                base_layer=source_layer,
                new_layer_name=f"{source_layer}_{func.__name__.replace('integrate_', '')}",
                **kwargs,
            )

        return wrapper

    def _wrap_integrate_harmony(self, func: Callable) -> Callable:
        """Wrap Harmony integration function (requires PCA layer).

        Parameters
        ----------
        func : Callable
            Harmony integration function

        Returns
        -------
        Callable
            Wrapped function
        """

        def wrapper(
            container: ScpContainer,
            assay_name: str,
            source_layer: str,
            **kwargs,
        ) -> ScpContainer:
            """Wrapper for Harmony integration."""
            # Harmony expects PCA input, so we need to check if we have it
            # For now, use the source layer directly
            return func(
                container=container,
                batch_key=self._batch_key,
                assay_name=assay_name,
                base_layer=source_layer,
                new_layer_name=f"{source_layer}_harmony",
                **kwargs,
            )

        return wrapper


__all__ = ["IntegrationEvaluator"]
