"""Tests for stable QC namespace exports."""

from __future__ import annotations

import scptensor as scp
import scptensor.qc as stable_qc
import scptensor.qc.qc_feature as qc_feature_core
import scptensor.qc.qc_sample as qc_sample_core
from scptensor.qc import (
    assess_batch_effects,
    calculate_feature_qc_metrics,
    calculate_sample_qc_metrics,
    filter_doublets_mad,
    filter_features_by_cv,
    filter_features_by_missingness,
    filter_low_quality_samples,
    qc_feature,
    qc_sample,
)
from scptensor.qc.qc_feature import (
    calculate_feature_qc_metrics as calculate_feature_qc_metrics_core,
)
from scptensor.qc.qc_feature import filter_features_by_cv as filter_features_by_cv_core
from scptensor.qc.qc_feature import (
    filter_features_by_missingness as filter_features_by_missingness_core,
)
from scptensor.qc.qc_sample import assess_batch_effects as assess_batch_effects_core
from scptensor.qc.qc_sample import (
    calculate_sample_qc_metrics as calculate_sample_qc_metrics_core,
)
from scptensor.qc.qc_sample import filter_doublets_mad as filter_doublets_mad_core
from scptensor.qc.qc_sample import (
    filter_low_quality_samples as filter_low_quality_samples_core,
)


def test_stable_qc_namespace_all_freezes_package_surface() -> None:
    assert stable_qc.__all__ == [
        "qc_sample",
        "qc_feature",
        "calculate_sample_qc_metrics",
        "filter_low_quality_samples",
        "filter_doublets_mad",
        "assess_batch_effects",
        "calculate_feature_qc_metrics",
        "filter_features_by_missingness",
        "filter_features_by_cv",
    ]


def test_stable_qc_namespace_reexports_stable_implementations() -> None:
    assert qc_sample is qc_sample_core
    assert qc_feature is qc_feature_core
    assert calculate_sample_qc_metrics is calculate_sample_qc_metrics_core
    assert filter_low_quality_samples is filter_low_quality_samples_core
    assert filter_doublets_mad is filter_doublets_mad_core
    assert assess_batch_effects is assess_batch_effects_core
    assert calculate_feature_qc_metrics is calculate_feature_qc_metrics_core
    assert filter_features_by_missingness is filter_features_by_missingness_core
    assert filter_features_by_cv is filter_features_by_cv_core


def test_root_package_does_not_reexport_qc_surface_or_experimental_helpers() -> None:
    for name in stable_qc.__all__:
        assert name not in scp.__all__
        assert not hasattr(scp, name)

    for name in ("qc_psm", "compute_mad", "is_outlier_mad", "compute_cv"):
        assert name not in scp.__all__
        assert not hasattr(scp, name)


def test_metrics_and_experimental_helpers_not_exported_from_stable_qc_namespace() -> None:
    for name in ("qc_psm", "compute_mad", "is_outlier_mad", "compute_cv"):
        assert name not in stable_qc.__all__
