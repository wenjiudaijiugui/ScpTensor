"""Imputation evaluator for automatic method selection.

This module provides an evaluator for testing and comparing different
imputation methods.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from scptensor.autoselect._heuristics import IMPUTATION_HOLDOUT_HEURISTICS
from scptensor.autoselect.core import EvaluationResult
from scptensor.autoselect.evaluators.base import BaseEvaluator
from scptensor.core._structure_matrix import MaskCode
from scptensor.core.assay_alias import resolve_assay_name

if TYPE_CHECKING:
    from scptensor.core.structures import ScpContainer


class ImputationEvaluator(BaseEvaluator):
    """Evaluator for imputation methods.

    This evaluator tests various imputation methods and evaluates their
    performance using metrics such as RMSE, correlation preservation, and
    imputation quality.

    Attributes
    ----------
    stage_name : str
        Name of the analysis stage ("imputation")
    methods : dict[str, Callable]
        Dictionary of imputation methods to test
    metric_weights : dict[str, float]
        Weights for evaluation metrics

    Examples
    --------
    >>> evaluator = ImputationEvaluator()
    >>> result_container, report = evaluator.run_all(
    ...     container=data,
    ...     assay_name="proteins",
    ...     source_layer="log"
    ... )

    """

    def __init__(self) -> None:
        """Initialize the imputation evaluator."""
        self._available_methods: dict[str, Callable] | None = None
        self._eps = 1e-10
        self._holdout_heuristics = IMPUTATION_HOLDOUT_HEURISTICS

    def _get_available_methods(self) -> dict[str, Callable]:
        """Get available imputation methods.

        Returns
        -------
        dict[str, Callable]
            Dictionary of available methods

        """
        from scptensor.autoselect.evaluators.base import create_wrapper

        if self._available_methods is not None:
            return self._available_methods

        methods: dict[str, Callable] = {}

        # Core imputation methods (always available)
        # Use "clean" layer_namer to remove "impute_" prefix from function names
        # This ensures layer names match what evaluate_method expects:
        # {source_layer}_{method_name} instead of {source_layer}_{func.__name__}
        try:
            from scptensor.impute import impute_none

            methods["none"] = create_wrapper(impute_none, layer_namer="clean")
        except ImportError:
            pass

        try:
            from scptensor.impute import impute_zero

            methods["zero"] = create_wrapper(impute_zero, layer_namer="clean")
        except ImportError:
            pass

        try:
            from scptensor.impute import impute_row_mean

            methods["row_mean"] = create_wrapper(impute_row_mean, layer_namer="clean")
        except ImportError:
            pass

        try:
            from scptensor.impute import impute_row_median

            methods["row_median"] = create_wrapper(impute_row_median, layer_namer="clean")
        except ImportError:
            pass

        try:
            from scptensor.impute import impute_half_row_min

            methods["half_row_min"] = create_wrapper(impute_half_row_min, layer_namer="clean")
        except ImportError:
            pass

        try:
            from scptensor.impute import impute_knn

            methods["knn"] = create_wrapper(impute_knn, layer_namer="clean")
        except ImportError:
            pass

        try:
            from scptensor.impute import impute_bpca

            methods["bpca"] = create_wrapper(impute_bpca, layer_namer="clean")
        except ImportError:
            pass

        try:
            from scptensor.impute import impute_mf

            methods["mf"] = create_wrapper(impute_mf, layer_namer="clean")
        except ImportError:
            pass

        try:
            from scptensor.impute import impute_minprob

            methods["minprob"] = create_wrapper(impute_minprob, layer_namer="clean")
        except ImportError:
            pass

        try:
            from scptensor.impute import impute_qrilc

            methods["qrilc"] = create_wrapper(impute_qrilc, layer_namer="clean")
        except ImportError:
            pass

        try:
            from scptensor.impute import impute_lls

            methods["lls"] = create_wrapper(impute_lls, layer_namer="clean")
        except ImportError:
            pass

        try:
            from scptensor.impute import impute_iterative_svd

            methods["iterative_svd"] = create_wrapper(impute_iterative_svd, layer_namer="clean")
        except ImportError:
            pass

        try:
            from scptensor.impute import impute_softimpute

            methods["softimpute"] = create_wrapper(impute_softimpute, layer_namer="clean")
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
            Stage name ("impute")

        """
        return "impute"

    @property
    def methods(self) -> dict[str, Callable]:
        """Return dictionary of available imputation methods.

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
        return {
            "rmse": 0.4,
            "correlation": 0.3,
            "completeness": 0.3,
        }

    def compute_metrics(
        self,
        container: ScpContainer,
        original_container: ScpContainer,
        layer_name: str,
    ) -> dict[str, float]:
        """Compute evaluation metrics for an imputed layer.

        Parameters
        ----------
        container : ScpContainer
            Container with the imputed data layer
        original_container : ScpContainer
            Original container before imputation (for comparison)
        layer_name : str
            Name of the layer to evaluate

        Returns
        -------
        dict[str, float]
            Dictionary mapping metric names to their scores (0.0 to 1.0)

        """
        assay = self._get_metric_assay(container)
        if assay is None or layer_name not in assay.layers:
            return dict.fromkeys(self.metric_weights, 0.0)

        original_assay = self._get_metric_assay(original_container)
        if original_assay is None:
            return dict.fromkeys(self.metric_weights, 0.0)

        source_layer = self._infer_source_layer(original_assay, layer_name)
        if source_layer is None or source_layer not in original_assay.layers:
            return dict.fromkeys(self.metric_weights, 0.0)

        imputed_matrix = assay.layers[layer_name]
        original_matrix = original_assay.layers[source_layer]

        original_x = self._to_dense_array(original_matrix.X)
        imputed_x = self._to_dense_array(imputed_matrix.X)

        if original_x.shape != imputed_x.shape:
            return dict.fromkeys(self.metric_weights, 0.0)

        missing_mask = self._get_missing_mask(original_matrix, original_x)
        observed_mask = ~missing_mask

        rmse_score = self._compute_rmse_score(original_x, imputed_x, observed_mask)
        correlation_score = self._compute_correlation_score(original_x, imputed_x, observed_mask)
        completeness_score = self._compute_completeness_score(imputed_x, missing_mask)

        return {
            "rmse": rmse_score,
            "correlation": correlation_score,
            "completeness": completeness_score,
        }

    def _infer_source_layer(self, assay: Any, layer_name: str) -> str | None:
        candidates = [name for name in assay.layers if layer_name.startswith(f"{name}_")]
        if not candidates:
            return None
        return max(candidates, key=len)

    def _to_dense_array(self, data: Any) -> np.ndarray:
        if hasattr(data, "toarray"):
            return data.toarray()
        return np.asarray(data)

    def _get_missing_mask(self, matrix, x_data: np.ndarray) -> np.ndarray:
        mask = np.isnan(x_data)
        m_matrix = matrix.get_m()
        if hasattr(m_matrix, "toarray"):
            m_matrix = m_matrix.toarray()
        return mask | (m_matrix != MaskCode.VALID)

    def _compute_rmse_score(
        self,
        original_x: np.ndarray,
        imputed_x: np.ndarray,
        observed_mask: np.ndarray,
    ) -> float:
        if not np.any(observed_mask):
            return 0.0

        original_vals = original_x[observed_mask]
        imputed_vals = imputed_x[observed_mask]
        valid = np.isfinite(original_vals) & np.isfinite(imputed_vals)
        if not np.any(valid):
            return 0.0

        diff = imputed_vals[valid] - original_vals[valid]
        rmse = float(np.sqrt(np.mean(diff**2)))
        scale = float(np.nanstd(original_vals[valid]))
        if scale < self._eps:
            return 1.0 if rmse < self._eps else 0.0

        score = float(np.exp(-rmse / (scale + self._eps)))
        return float(np.clip(score, 0.0, 1.0))

    def _compute_correlation_score(
        self,
        original_x: np.ndarray,
        imputed_x: np.ndarray,
        observed_mask: np.ndarray,
    ) -> float:
        if not np.any(observed_mask):
            return 0.0

        original_vals = original_x[observed_mask]
        imputed_vals = imputed_x[observed_mask]
        valid = np.isfinite(original_vals) & np.isfinite(imputed_vals)
        if np.sum(valid) < 2:
            return 0.0

        if (
            np.nanstd(original_vals[valid]) < self._eps
            or np.nanstd(imputed_vals[valid]) < self._eps
        ):
            return 0.0

        corr = float(np.corrcoef(original_vals[valid], imputed_vals[valid])[0, 1])
        if not np.isfinite(corr):
            return 0.0

        score = (corr + 1.0) / 2.0
        return float(np.clip(score, 0.0, 1.0))

    def _compute_completeness_score(
        self,
        imputed_x: np.ndarray,
        missing_mask: np.ndarray,
    ) -> float:
        missing_count = int(np.sum(missing_mask))
        if missing_count == 0:
            return 1.0

        filled = np.isfinite(imputed_x[missing_mask])
        return float(np.clip(np.mean(filled), 0.0, 1.0))

    def _build_holdout_mask(self, observed_mask: np.ndarray) -> np.ndarray | None:
        """Sample deterministic holdout entries from observed values."""
        observed_idx = np.flatnonzero(observed_mask)
        if observed_idx.size < 2:
            return None

        target = int(observed_idx.size * self._holdout_heuristics.fraction)
        n_holdout = max(self._holdout_heuristics.min_entries, target)
        n_holdout = min(
            n_holdout,
            self._holdout_heuristics.max_entries,
            observed_idx.size,
        )
        if n_holdout < 2:
            return None

        rng = np.random.default_rng(self._holdout_heuristics.random_state)
        chosen = rng.choice(observed_idx, size=n_holdout, replace=False)

        holdout_mask = np.zeros(observed_mask.size, dtype=bool)
        holdout_mask[chosen] = True
        return holdout_mask.reshape(observed_mask.shape)

    def _mask_holdout_entries(
        self,
        container: ScpContainer,
        assay_name: str,
        source_layer: str,
        holdout_mask: np.ndarray,
    ) -> ScpContainer:
        """Create a container copy with holdout entries masked as missing."""
        masked_container = container.copy()
        resolved_assay_name = resolve_assay_name(masked_container, assay_name)
        source_matrix = masked_container.assays[resolved_assay_name].layers[source_layer]

        x_data = self._to_dense_array(source_matrix.X).copy()
        x_data[holdout_mask] = np.nan
        source_matrix.X = x_data

        m_data = source_matrix.get_m()
        if hasattr(m_data, "toarray"):
            m_data = m_data.toarray()
        m_data = np.asarray(m_data, dtype=np.int8).copy()
        m_data[holdout_mask] = MaskCode.LOD.value
        source_matrix.M = m_data

        return masked_container

    def _restore_holdout_entries(
        self,
        result_container: ScpContainer,
        assay_name: str,
        layer_name: str,
        holdout_mask: np.ndarray,
        original_x: np.ndarray,
        original_m: np.ndarray,
    ) -> None:
        """Restore holdout entries so output layer preserves original observed values."""
        resolved_assay_name = resolve_assay_name(result_container, assay_name)
        out_matrix = result_container.assays[resolved_assay_name].layers[layer_name]

        out_x = self._to_dense_array(out_matrix.X).copy()
        out_x[holdout_mask] = original_x[holdout_mask]
        out_matrix.X = out_x

        out_m = out_matrix.get_m()
        if hasattr(out_m, "toarray"):
            out_m = out_m.toarray()
        out_m = np.asarray(out_m, dtype=np.int8).copy()
        out_m[holdout_mask] = original_m[holdout_mask]
        out_matrix.M = out_m

    def _compute_holdout_rmse_score(
        self,
        truth: np.ndarray,
        pred: np.ndarray,
    ) -> float:
        """RMSE-derived score on held-out entries."""
        valid = np.isfinite(truth) & np.isfinite(pred)
        if not np.any(valid):
            return 0.0

        diff = pred[valid] - truth[valid]
        rmse = float(np.sqrt(np.mean(diff**2)))
        scale = float(np.nanstd(truth[valid]))
        if scale < self._eps:
            return 1.0 if rmse < self._eps else 0.0

        score = float(np.exp(-rmse / (scale + self._eps)))
        return float(np.clip(score, 0.0, 1.0))

    def _compute_holdout_correlation_score(
        self,
        truth: np.ndarray,
        pred: np.ndarray,
    ) -> float:
        """Correlation-derived score on held-out entries."""
        valid = np.isfinite(truth) & np.isfinite(pred)
        if np.sum(valid) < 2:
            return 0.0

        if np.nanstd(truth[valid]) < self._eps or np.nanstd(pred[valid]) < self._eps:
            return 0.0

        corr = float(np.corrcoef(truth[valid], pred[valid])[0, 1])
        if not np.isfinite(corr):
            return 0.0

        score = (corr + 1.0) / 2.0
        return float(np.clip(score, 0.0, 1.0))

    def evaluate_method(
        self,
        container: ScpContainer,
        method_name: str,
        method_func: Callable,
        assay_name: str = "proteins",
        source_layer: str = "raw",
        **kwargs,
    ) -> tuple[ScpContainer | None, EvaluationResult]:
        """Evaluate a method using holdout-based scoring on observed entries.

        Notes
        -----
        A subset of observed values is masked before imputation, then scored
        against the ground truth on the held-out subset. This avoids score
        saturation when methods leave observed values unchanged.

        """
        assay = container.assays.get(assay_name)
        if assay is None or source_layer not in assay.layers:
            return super().evaluate_method(
                container=container,
                method_name=method_name,
                method_func=method_func,
                assay_name=assay_name,
                source_layer=source_layer,
                **kwargs,
            )

        source_matrix = assay.layers[source_layer]
        original_x = self._to_dense_array(source_matrix.X)
        missing_mask = self._get_missing_mask(source_matrix, original_x)
        observed_mask = np.isfinite(original_x) & ~missing_mask
        holdout_mask = self._build_holdout_mask(observed_mask)
        if holdout_mask is None:
            return super().evaluate_method(
                container=container,
                method_name=method_name,
                method_func=method_func,
                assay_name=assay_name,
                source_layer=source_layer,
                **kwargs,
            )

        original_m = source_matrix.get_m()
        if hasattr(original_m, "toarray"):
            original_m = original_m.toarray()
        original_m = np.asarray(original_m, dtype=np.int8)

        new_layer_name = f"{source_layer}_{method_name}"
        start_time = time.perf_counter()
        result_container: ScpContainer | None = None
        error_msg: str | None = None
        scores: dict[str, float] = {}

        try:
            masked_container = self._mask_holdout_entries(
                container=container,
                assay_name=assay_name,
                source_layer=source_layer,
                holdout_mask=holdout_mask,
            )
            result_container = method_func(
                container=masked_container,
                assay_name=assay_name,
                source_layer=source_layer,
                **kwargs,
            )

            if result_container is None:
                raise RuntimeError("Method returned None")

            resolved_assay_name = resolve_assay_name(result_container, assay_name)
            if new_layer_name not in result_container.assays[resolved_assay_name].layers:
                raise KeyError(
                    f"Expected output layer '{new_layer_name}' not found in assay '{assay_name}'",
                )

            imputed_matrix = result_container.assays[resolved_assay_name].layers[new_layer_name]
            imputed_x = self._to_dense_array(imputed_matrix.X)

            if imputed_x.shape != original_x.shape:
                raise ValueError(
                    "Shape mismatch after imputation: "
                    f"expected {original_x.shape}, got {imputed_x.shape}",
                )

            holdout_true = original_x[holdout_mask]
            holdout_pred = imputed_x[holdout_mask]

            scores = {
                "rmse": self._compute_holdout_rmse_score(holdout_true, holdout_pred),
                "correlation": self._compute_holdout_correlation_score(holdout_true, holdout_pred),
                "completeness": self._compute_completeness_score(imputed_x, missing_mask),
            }

            # Restore held-out observed values in returned layer.
            self._restore_holdout_entries(
                result_container=result_container,
                assay_name=assay_name,
                layer_name=new_layer_name,
                holdout_mask=holdout_mask,
                original_x=original_x,
                original_m=original_m,
            )
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            result_container = None
            scores = dict.fromkeys(self.get_metric_weights(), 0.0)

        execution_time = time.perf_counter() - start_time
        overall_score = 0.0 if error_msg else self.compute_overall_score(scores)

        eval_result = EvaluationResult(
            method_name=method_name,
            scores=scores,
            overall_score=overall_score,
            execution_time=execution_time,
            layer_name=new_layer_name,
            error=error_msg,
        )

        return result_container, eval_result
