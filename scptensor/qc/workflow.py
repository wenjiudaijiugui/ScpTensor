"""High-level QC preflight workflow for DIA single-cell protein matrices."""

from __future__ import annotations

import numpy as np

from scptensor.core._structure_container import ScpContainer
from scptensor.core.exceptions import ScpValueError
from scptensor.qc._utils import (
    compute_detection_stats,
    compute_sample_qc_vectors,
    normalize_detected_codes,
    resolve_assay,
    resolve_layer,
    validate_threshold,
)
from scptensor.qc.metrics import compute_cv
from scptensor.qc.qc_feature import (
    calculate_feature_qc_metrics,
    filter_features_by_cv,
    filter_features_by_missingness,
)
from scptensor.qc.qc_sample import (
    calculate_sample_qc_metrics,
    filter_doublets_mad,
    filter_low_quality_samples,
)


def _quantile(values: np.ndarray, q: float, fallback: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return fallback
    return float(np.quantile(finite, q))


def _resolve_sample_min_features(
    *,
    explicit_min_features: int | None,
    n_features_per_sample: np.ndarray,
    sample_quantile: float,
    floor: int,
    n_total_features: int,
) -> int:
    if explicit_min_features is not None:
        if explicit_min_features < 0:
            raise ScpValueError(
                f"min_features must be >= 0, got {explicit_min_features}.",
                parameter="min_features",
                value=explicit_min_features,
            )
        return int(explicit_min_features)

    validate_threshold(sample_quantile, "sample_min_features_quantile", min_val=0.0, max_val=1.0)
    if floor < 0:
        raise ScpValueError(
            f"sample_min_features_floor must be >= 0, got {floor}.",
            parameter="sample_min_features_floor",
            value=floor,
        )

    adaptive = int(
        np.floor(_quantile(n_features_per_sample, sample_quantile, fallback=float(floor)))
    )
    bounded = max(floor, adaptive)
    bounded = min(bounded, max(n_total_features, 0))
    return int(max(0, bounded))


def _resolve_feature_missingness_threshold(
    *,
    explicit_max_missing_rate: float | None,
    feature_missing_rate: np.ndarray,
    missing_rate_quantile: float,
) -> float:
    if explicit_max_missing_rate is not None:
        validate_threshold(
            explicit_max_missing_rate,
            "max_missing_rate",
            min_val=0.0,
            max_val=1.0,
        )
        return float(explicit_max_missing_rate)

    validate_threshold(
        missing_rate_quantile,
        "feature_missing_rate_quantile",
        min_val=0.0,
        max_val=1.0,
    )
    adaptive = _quantile(feature_missing_rate, missing_rate_quantile, fallback=1.0)
    return float(np.clip(adaptive, 0.0, 1.0))


def _resolve_feature_cv_threshold(
    *,
    explicit_max_cv: float | None,
    feature_cv: np.ndarray,
    cv_quantile: float,
) -> float:
    if explicit_max_cv is not None:
        if explicit_max_cv <= 0:
            raise ScpValueError(
                f"max_cv must be positive, got {explicit_max_cv}.",
                parameter="max_cv",
                value=explicit_max_cv,
            )
        return float(explicit_max_cv)

    validate_threshold(cv_quantile, "feature_cv_quantile", min_val=0.0, max_val=1.0)
    adaptive = _quantile(feature_cv, cv_quantile, fallback=1.0)
    if not np.isfinite(adaptive) or adaptive <= 0:
        return 1.0
    return float(adaptive)


def qc_preflight(
    container: ScpContainer,
    assay_name: str = "proteins",
    layer_name: str = "raw",
    *,
    detected_codes: tuple[int, ...] | list[int] | None = None,
    min_features: int | None = None,
    sample_min_features_quantile: float = 0.05,
    sample_min_features_floor: int = 10,
    sample_nmads: float = 3.0,
    use_mad: bool = True,
    doublet_nmads: float = 3.0,
    max_missing_rate: float | None = None,
    feature_missing_rate_quantile: float = 0.90,
    max_cv: float | None = None,
    feature_cv_quantile: float = 0.90,
    min_mean_for_cv: float = 1e-6,
) -> ScpContainer:
    """Run a full QC preflight with adaptive defaults.

    Workflow:
    1. sample metrics
    2. low-quality sample filtering
    3. doublet filtering
    4. feature metrics
    5. missingness filtering
    6. CV filtering
    """
    resolved_detected_codes = normalize_detected_codes(detected_codes)

    resolved_assay_name, assay = resolve_assay(container, assay_name)
    resolved_layer_name, layer = resolve_layer(
        assay,
        assay_name=resolved_assay_name,
        layer_name=layer_name,
    )
    n_features_per_sample, _ = compute_sample_qc_vectors(
        layer,
        detected_codes=resolved_detected_codes,
    )
    resolved_min_features = _resolve_sample_min_features(
        explicit_min_features=min_features,
        n_features_per_sample=n_features_per_sample,
        sample_quantile=sample_min_features_quantile,
        floor=sample_min_features_floor,
        n_total_features=layer.X.shape[1],
    )

    qc_container = calculate_sample_qc_metrics(
        container,
        assay_name=resolved_assay_name,
        layer_name=resolved_layer_name,
        detected_codes=resolved_detected_codes,
    )
    qc_container = filter_low_quality_samples(
        qc_container,
        assay_name=resolved_assay_name,
        layer_name=resolved_layer_name,
        min_features=resolved_min_features,
        nmads=sample_nmads,
        use_mad=use_mad,
        detected_codes=resolved_detected_codes,
    )
    qc_container = filter_doublets_mad(
        qc_container,
        assay_name=resolved_assay_name,
        layer_name=resolved_layer_name,
        nmads=doublet_nmads,
        detected_codes=resolved_detected_codes,
    )
    qc_container = calculate_feature_qc_metrics(
        qc_container,
        assay_name=resolved_assay_name,
        layer_name=resolved_layer_name,
        detected_codes=resolved_detected_codes,
    )

    filtered_assay = qc_container.assays[resolved_assay_name]
    feature_layer_name, feature_layer = resolve_layer(
        filtered_assay,
        assay_name=resolved_assay_name,
        layer_name=resolved_layer_name,
    )
    _, detection_rate, _ = compute_detection_stats(
        feature_layer.X,
        M=feature_layer.M,
        detected_codes=resolved_detected_codes,
    )
    feature_missing_rate = 1.0 - detection_rate
    feature_cv = compute_cv(feature_layer.X, axis=0, min_mean=min_mean_for_cv)

    resolved_max_missing_rate = _resolve_feature_missingness_threshold(
        explicit_max_missing_rate=max_missing_rate,
        feature_missing_rate=feature_missing_rate,
        missing_rate_quantile=feature_missing_rate_quantile,
    )
    resolved_max_cv = _resolve_feature_cv_threshold(
        explicit_max_cv=max_cv,
        feature_cv=feature_cv,
        cv_quantile=feature_cv_quantile,
    )

    qc_container = filter_features_by_missingness(
        qc_container,
        assay_name=resolved_assay_name,
        layer_name=feature_layer_name,
        max_missing_rate=resolved_max_missing_rate,
        detected_codes=resolved_detected_codes,
    )
    qc_container = filter_features_by_cv(
        qc_container,
        assay_name=resolved_assay_name,
        layer_name=feature_layer_name,
        max_cv=resolved_max_cv,
        min_mean=min_mean_for_cv,
    )
    qc_container.log_operation(
        action="qc_preflight",
        params={
            "assay": resolved_assay_name,
            "layer": feature_layer_name,
            "detected_codes": list(resolved_detected_codes),
            "resolved_min_features": resolved_min_features,
            "resolved_max_missing_rate": resolved_max_missing_rate,
            "resolved_max_cv": resolved_max_cv,
            "sample_nmads": sample_nmads,
            "doublet_nmads": doublet_nmads,
            "use_mad": use_mad,
            "min_mean_for_cv": min_mean_for_cv,
        },
        description=(
            "Ran qc_preflight (sample metrics/filter, doublet filter, feature metrics/filter) "
            f"on assay '{resolved_assay_name}/{feature_layer_name}'."
        ),
    )
    return qc_container


__all__ = ["qc_preflight"]
