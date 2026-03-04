"""Base utilities for imputation modules.

Provides unified interface for missing value imputation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import scipy.sparse as sp

from scptensor.core.exceptions import AssayNotFoundError, LayerNotFoundError, ScpValueError

if TYPE_CHECKING:
    from scptensor.core.structures import ScpContainer

# Registry for imputation methods
_IMPUTE_METHODS: dict[str, ImputeMethod] = {}

MissingMechanism = Literal["auto", "mcar", "mar", "mnar"]
_VALID_MISSING_MECHANISMS: set[str] = {"auto", "mcar", "mar", "mnar"}
_MECHANISM_DEFAULT_METHOD: dict[str, str] = {
    "mcar": "knn",
    "mar": "missforest",
    "mnar": "qrilc",
}
_MECHANISM_COMPATIBILITY: dict[str, set[str]] = {
    "mcar": {"knn", "lls", "bpca", "missforest", "iterative_svd", "softimpute"},
    "mar": {"missforest", "knn", "lls", "bpca", "iterative_svd", "softimpute"},
    "mnar": {"qrilc", "minprob", "half_row_min"},
}


@dataclass
class ImputeMethod:
    """Registration entry for an imputation method.

    Attributes
    ----------
    name : str
        Method name.
    supports_sparse : bool
        Whether method supports sparse matrices.
    validate : Callable
        Validation function.
    apply : Callable
        Application function.
    """

    name: str
    supports_sparse: bool = True
    validate: Callable[..., Any] | None = None
    apply: Callable[..., ScpContainer] | None = None


def register_impute_method(method: ImputeMethod) -> ImputeMethod:
    """Register an imputation method.

    Parameters
    ----------
    method : ImputeMethod
        Method to register.

    Returns
    -------
    ImputeMethod
        The registered method.
    """
    _IMPUTE_METHODS[method.name] = method
    return method


def get_impute_method(name: str) -> ImputeMethod:
    """Get a registered imputation method by name.

    Parameters
    ----------
    name : str
        Method name.

    Returns
    -------
    ImputeMethod
        The registered method.

    Raises
    ------
    ScpValueError
        If method not found.
    """
    if name not in _IMPUTE_METHODS:
        available = list(_IMPUTE_METHODS.keys())
        raise ScpValueError(
            f"Imputation method '{name}' not found. Available methods: {available}",
            parameter="method",
            value=name,
        )
    return _IMPUTE_METHODS[name]


def list_impute_methods() -> list[str]:
    """List all registered imputation methods.

    Returns
    -------
    list[str]
        List of method names.
    """
    return list(_IMPUTE_METHODS.keys())


def _validate_missing_mechanism(missing_mechanism: str) -> None:
    if missing_mechanism not in _VALID_MISSING_MECHANISMS:
        raise ScpValueError(
            "missing_mechanism must be one of {'auto', 'mcar', 'mar', 'mnar'}.",
            parameter="missing_mechanism",
            value=missing_mechanism,
        )


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Return average-free ranks for 1D values (sufficient for selection heuristics)."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman correlation with safe fallback to NaN."""
    if x.size < 3 or y.size < 3:
        return np.nan
    xr = _rankdata(x)
    yr = _rankdata(y)
    x_std = np.std(xr)
    y_std = np.std(yr)
    if x_std <= 0 or y_std <= 0:
        return np.nan
    return float(np.corrcoef(xr, yr)[0, 1])


def _to_dense_float(x_data: Any) -> np.ndarray:
    if sp.issparse(x_data):
        return np.asarray(x_data.toarray(), dtype=np.float64)
    return np.asarray(x_data, dtype=np.float64)


def infer_missing_mechanism(
    container: ScpContainer,
    assay_name: str,
    source_layer: str,
) -> tuple[Literal["mcar", "mar", "mnar"], str]:
    """Infer missingness mechanism using transparent heuristics.

    Heuristics:
    1. Feature-level correlation between observed intensity and missing rate.
       Strong negative correlation implies left-censoring (MNAR).
    2. Sample-level heterogeneity of missingness.
       Strongly uneven sample missingness suggests MAR.
    3. Low correlation + relatively uniform missingness falls back to MCAR.
    """
    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name, available_assays=container.assays.keys())
    assay = container.assays[assay_name]
    if source_layer not in assay.layers:
        raise LayerNotFoundError(source_layer, assay_name, available_layers=assay.layers.keys())

    x = _to_dense_float(assay.layers[source_layer].X)
    missing_mask = np.isnan(x)
    missing_rate = float(np.mean(missing_mask))

    if missing_rate <= 0:
        return "mcar", "No missing values detected; defaulting mechanism to MCAR."

    sample_missing_rate = np.mean(missing_mask, axis=1)
    sample_mean = float(np.mean(sample_missing_rate))
    sample_cv = (
        0.0 if sample_mean <= 0 else float(np.std(sample_missing_rate) / (sample_mean + 1e-12))
    )

    feature_missing_rate = np.mean(missing_mask, axis=0)
    feature_mean = np.full(x.shape[1], np.nan, dtype=np.float64)
    for feat_idx in range(x.shape[1]):
        observed = x[~missing_mask[:, feat_idx], feat_idx]
        if observed.size >= 5:
            feature_mean[feat_idx] = float(np.mean(observed))

    valid = np.isfinite(feature_mean) & (feature_missing_rate > 0)
    corr = _safe_spearman(feature_mean[valid], feature_missing_rate[valid])

    corr_text = "nan" if np.isnan(corr) else f"{corr:.3f}"
    stats_prefix = (
        f"missing_rate={missing_rate:.3f}, "
        f"feature_intensity_vs_missing_spearman={corr_text}, "
        f"sample_missing_cv={sample_cv:.3f}"
    )

    if np.isfinite(corr) and corr <= -0.45:
        return "mnar", (
            f"{stats_prefix}. Strong negative intensity-missing correlation suggests "
            "left-censoring (MNAR)."
        )
    if sample_cv >= 0.6 and missing_rate >= 0.05:
        return "mar", f"{stats_prefix}. Heterogeneous sample missingness suggests MAR."
    if (np.isfinite(corr) and abs(corr) <= 0.2) or missing_rate <= 0.08:
        return "mcar", f"{stats_prefix}. Weak intensity dependence suggests MCAR."
    if np.isfinite(corr) and corr < -0.2:
        return "mnar", f"{stats_prefix}. Moderate negative intensity dependence suggests MNAR."
    return "mar", f"{stats_prefix}. Remaining pattern treated as MAR."


def recommend_impute_method(
    missing_mechanism: Literal["mcar", "mar", "mnar"],
) -> str:
    """Recommend default imputation method for a given mechanism."""
    return _MECHANISM_DEFAULT_METHOD[missing_mechanism]


def _resolve_missing_mechanism(
    container: ScpContainer,
    missing_mechanism: MissingMechanism,
    kwargs: dict[str, Any],
) -> tuple[Literal["mcar", "mar", "mnar"], str]:
    if missing_mechanism != "auto":
        return missing_mechanism, f"Missing mechanism explicitly set to '{missing_mechanism}'."

    assay_name = kwargs.get("assay_name")
    source_layer = kwargs.get("source_layer")
    if not isinstance(assay_name, str) or not assay_name:
        raise ScpValueError(
            "impute(..., missing_mechanism='auto') requires 'assay_name' in kwargs.",
            parameter="assay_name",
            value=assay_name,
        )
    if not isinstance(source_layer, str) or not source_layer:
        raise ScpValueError(
            "impute(..., missing_mechanism='auto') requires 'source_layer' in kwargs.",
            parameter="source_layer",
            value=source_layer,
        )
    mechanism, reason = infer_missing_mechanism(container, assay_name, source_layer)
    return mechanism, f"Inferred mechanism='{mechanism}': {reason}"


def impute(
    container: ScpContainer,
    method: str = "knn",
    missing_mechanism: MissingMechanism | None = None,
    **kwargs,
) -> ScpContainer:
    """Unified interface for missing value imputation.

    Parameters
    ----------
    container : ScpContainer
        Input container.
    method : str, default="knn"
        Imputation method name.
        Use ``method="auto"`` for mechanism-aware automatic selection.
    missing_mechanism : {"auto", "mcar", "mar", "mnar"} or None, default=None
        Missingness mechanism hint. Used when ``method="auto"`` and optional
        compatibility checking when an explicit method is provided.
    **kwargs
        Additional arguments passed to the method.

    Returns
    -------
    ScpContainer
        Container with imputed data.

    Examples
    --------
    >>> container = impute(container, method='knn', n_neighbors=5)
    >>> container = impute(container, method='bpca', n_components=10)
    """
    mechanism: Literal["mcar", "mar", "mnar"] | None = None
    mechanism_reason: str | None = None

    if missing_mechanism is not None:
        _validate_missing_mechanism(missing_mechanism)
        mechanism, mechanism_reason = _resolve_missing_mechanism(
            container,
            missing_mechanism,
            kwargs,
        )

    selected_method = method
    if method == "auto":
        mechanism_for_auto = mechanism
        reason_for_auto = mechanism_reason
        if mechanism_for_auto is None:
            mechanism_for_auto, reason_for_auto = _resolve_missing_mechanism(
                container,
                "auto",
                kwargs,
            )

        selected_method = recommend_impute_method(mechanism_for_auto)
        if selected_method not in _IMPUTE_METHODS:
            fallback_order = ("knn", "row_mean", "zero")
            fallback = next((name for name in fallback_order if name in _IMPUTE_METHODS), None)
            if fallback is None:
                raise ScpValueError(
                    "No registered imputation methods available for auto selection.",
                    parameter="method",
                    value=method,
                )
            selected_method = fallback
            reason_for_auto = (
                f"{reason_for_auto} Preferred method unavailable; fallback='{selected_method}'."
            )

        kwargs.setdefault("new_layer_name", f"imputed_{selected_method}")
        container.log_operation(
            action="impute_method_selection",
            params={
                "selected_method": selected_method,
                "requested_method": method,
                "missing_mechanism": mechanism_for_auto,
                "selection_reason": reason_for_auto,
            },
            description=f"Auto-selected imputation method '{selected_method}'.",
        )
    elif mechanism is not None:
        compatible = _MECHANISM_COMPATIBILITY.get(mechanism, set())
        if selected_method not in compatible:
            recommended = recommend_impute_method(mechanism)
            warning_msg = (
                f"Method '{selected_method}' may not match missing_mechanism='{mechanism}'. "
                f"Recommended default is '{recommended}'. {mechanism_reason}"
            )
            container.log_operation(
                action="impute_mechanism_warning",
                params={
                    "requested_method": selected_method,
                    "missing_mechanism": mechanism,
                    "recommended_method": recommended,
                    "reason": mechanism_reason,
                },
                description=warning_msg,
            )

    entry = get_impute_method(selected_method)
    if entry.apply is None:
        raise ScpValueError(
            f"Method '{selected_method}' has no apply function.",
            parameter="method",
            value=selected_method,
        )
    return entry.apply(container, **kwargs)


__all__ = [
    "ImputeMethod",
    "register_impute_method",
    "get_impute_method",
    "list_impute_methods",
    "infer_missing_mechanism",
    "recommend_impute_method",
    "impute",
]
