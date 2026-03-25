"""Real-data robustness tests for QC, imputation, and integration."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from scptensor.core import MaskCode
from scptensor.impute import impute_half_row_min, impute_none, impute_row_median
from scptensor.integration import integrate_limma, integrate_none
from scptensor.integration.diagnostics import (
    compute_batch_asw,
    compute_batch_mixing_metric,
    compute_ilisi,
    compute_kbet,
    compute_lisi_approx,
    integration_quality_report,
)
from scptensor.io import aggregate_to_protein, load_diann, load_peptide_pivot, load_spectronaut
from scptensor.qc import (
    assess_batch_effects,
    calculate_feature_qc_metrics,
    calculate_sample_qc_metrics,
    filter_features_by_cv,
    filter_features_by_missingness,
    filter_low_quality_samples,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

MINIMAL_DIR = Path("data/public_minimal_training_set")
DIANN_LONG = MINIMAL_DIR / "01_diann_long_report.tsv"
SPECTRONAUT_PROTEIN = MINIMAL_DIR / "02_spectronaut_protein_matrix.tsv"
SPECTRONAUT_PEPTIDE = MINIMAL_DIR / "03_spectronaut_peptide_matrix.tsv"


def _require_real_training_set() -> None:
    required = [DIANN_LONG, SPECTRONAUT_PROTEIN, SPECTRONAUT_PEPTIDE]
    missing = [path for path in required if not path.exists()]
    if missing:
        pytest.skip(
            "Real public minimal training set is not available locally. Missing: "
            + ", ".join(str(path) for path in missing)
        )


def _load_real_protein_container(case_name: str):
    _require_real_training_set()

    if case_name == "diann_protein":
        return load_diann(DIANN_LONG, level="protein", assay_name="proteins")

    if case_name == "spectronaut_protein":
        return load_spectronaut(
            SPECTRONAUT_PROTEIN,
            level="protein",
            assay_name="proteins",
        )

    if case_name == "spectronaut_peptide_aggregated":
        peptides = load_peptide_pivot(
            SPECTRONAUT_PEPTIDE,
            software="spectronaut",
            assay_name="peptides",
        )
        return aggregate_to_protein(
            peptides,
            source_assay="peptides",
            target_assay="proteins",
            method="sum",
        )

    raise ValueError(f"Unsupported real-data case '{case_name}'.")


def _annotate_real_diann_batches(container):
    obs = container.obs.with_columns(
        [
            pl.col("_index").str.extract(r"MGE_HStdia_(S\d)_", 1).alias("batch"),
            pl.when(pl.col("_index").str.contains("SCamount300ng"))
            .then(pl.lit("300ng"))
            .otherwise(pl.lit("600ng"))
            .alias("condition"),
        ],
    )

    annotated = container.copy()
    annotated.obs = obs
    return annotated


@pytest.mark.parametrize(
    ("case_name", "expected_shape"),
    [
        pytest.param("diann_protein", (24, 2719), id="diann-protein"),
        pytest.param("spectronaut_protein", (3, 5683), id="spectronaut-protein"),
        pytest.param(
            "spectronaut_peptide_aggregated",
            (3, 5683),
            id="spectronaut-peptide-aggregated",
        ),
    ],
)
def test_real_protein_matrices_support_qc_metric_calculation(
    case_name: str,
    expected_shape: tuple[int, int],
) -> None:
    container = _load_real_protein_container(case_name)

    sample_qc = calculate_sample_qc_metrics(container, assay_name="proteins")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        warnings.filterwarnings(
            "ignore",
            message="Degrees of freedom <= 0 for slice.",
            category=RuntimeWarning,
        )
        feature_qc = calculate_feature_qc_metrics(container, assay_name="proteins")

    sample_cols = [
        "n_features_proteins",
        "total_intensity_proteins",
        "log1p_total_intensity_proteins",
    ]
    feature_cols = ["missing_rate", "detection_rate", "mean_expression", "cv"]

    assert sample_qc.assays["proteins"].layers["raw"].X.shape == expected_shape
    assert feature_qc.assays["proteins"].layers["raw"].X.shape == expected_shape
    assert all(col in sample_qc.obs.columns for col in sample_cols)
    assert all(col in feature_qc.assays["proteins"].var.columns for col in feature_cols)

    n_features = sample_qc.obs["n_features_proteins"].to_numpy()
    total_intensity = sample_qc.obs["total_intensity_proteins"].to_numpy()
    missing_rate = feature_qc.assays["proteins"].var["missing_rate"].to_numpy()
    detection_rate = feature_qc.assays["proteins"].var["detection_rate"].to_numpy()
    mean_expression = feature_qc.assays["proteins"].var["mean_expression"].to_numpy()
    cv = feature_qc.assays["proteins"].var["cv"].to_numpy()

    assert np.all(n_features > 0)
    assert np.all(np.isfinite(total_intensity))
    assert np.all((missing_rate >= 0.0) & (missing_rate <= 1.0))
    assert np.all((detection_rate >= 0.0) & (detection_rate <= 1.0))
    np.testing.assert_allclose(missing_rate + detection_rate, np.ones_like(missing_rate))
    assert np.isfinite(mean_expression).any()
    assert np.isfinite(cv).any()


def test_real_diann_protein_supports_qc_filtering_and_batch_summary() -> None:
    container = _annotate_real_diann_batches(_load_real_protein_container("diann_protein"))

    batch_condition_counts = (
        container.obs.group_by("batch", "condition").len().sort("batch", "condition")
    )
    sample_filtered = filter_low_quality_samples(
        container,
        assay_name="proteins",
        min_features=1800,
        use_mad=False,
    )
    missingness_filtered = filter_features_by_missingness(
        container,
        assay_name="proteins",
        max_missing_rate=0.5,
    )
    cv_filtered = filter_features_by_cv(
        container,
        assay_name="proteins",
        max_cv=0.5,
    )
    batch_summary = assess_batch_effects(
        container,
        batch_col="batch",
        assay_name="proteins",
    )

    assert batch_condition_counts["len"].to_list() == [4, 4, 4, 4, 4, 4]
    assert sample_filtered.n_samples == 17
    assert missingness_filtered.assays["proteins"].n_features == 2177
    assert cv_filtered.assays["proteins"].n_features == 2498
    assert batch_summary.shape == (3, 5)
    assert batch_summary["batch"].to_list() == ["S1", "S2", "S3"]
    assert batch_summary["n_cells"].to_list() == [8, 8, 8]
    assert sample_filtered.history[-1].action == "filter_low_quality_samples"
    assert missingness_filtered.history[-1].action == "filter_features_by_missingness"
    assert cv_filtered.history[-1].action == "filter_features_by_cv"


@pytest.mark.parametrize(
    ("case_name", "expected_shape", "expected_missing_count"),
    [
        pytest.param("diann_protein", (24, 2719), 16970, id="diann-protein"),
        pytest.param("spectronaut_protein", (3, 5683), 189, id="spectronaut-protein"),
        pytest.param(
            "spectronaut_peptide_aggregated",
            (3, 5683),
            122,
            id="spectronaut-peptide-aggregated",
        ),
    ],
)
def test_real_protein_matrices_support_baseline_imputation(
    case_name: str,
    expected_shape: tuple[int, int],
    expected_missing_count: int,
) -> None:
    base = _load_real_protein_container(case_name)
    raw = base.assays["proteins"].layers["raw"]
    raw_x = raw.X
    raw_mask = raw.get_m()
    missing_mask = np.isnan(raw_x)

    none_result = impute_none(
        base.copy(),
        assay_name="proteins",
        source_layer="raw",
        new_layer_name="none_real",
    )
    row_result = impute_row_median(
        base.copy(),
        assay_name="proteins",
        source_layer="raw",
        new_layer_name="row_median_real",
    )
    half_result = impute_half_row_min(
        base.copy(),
        assay_name="proteins",
        source_layer="raw",
        new_layer_name="half_row_min_real",
    )

    none_layer = none_result.assays["proteins"].layers["none_real"]
    row_layer = row_result.assays["proteins"].layers["row_median_real"]
    half_layer = half_result.assays["proteins"].layers["half_row_min_real"]

    assert none_layer.X.shape == expected_shape
    assert row_layer.X.shape == expected_shape
    assert half_layer.X.shape == expected_shape
    assert int(missing_mask.sum()) == expected_missing_count
    assert int(np.isnan(none_layer.X).sum()) == expected_missing_count
    assert not np.isnan(row_layer.X).any()
    assert not np.isnan(half_layer.X).any()
    np.testing.assert_allclose(none_layer.X, raw_x, equal_nan=True)
    np.testing.assert_array_equal(none_layer.get_m(), raw_mask)
    np.testing.assert_allclose(row_layer.X[~missing_mask], raw_x[~missing_mask], equal_nan=True)
    np.testing.assert_allclose(half_layer.X[~missing_mask], raw_x[~missing_mask], equal_nan=True)
    assert (
        int(np.count_nonzero(row_layer.get_m() == MaskCode.IMPUTED.value)) == expected_missing_count
    )
    assert (
        int(np.count_nonzero(half_layer.get_m() == MaskCode.IMPUTED.value))
        == expected_missing_count
    )


def test_real_diann_protein_supports_limma_integration_and_diagnostics_after_imputation() -> None:
    container = _annotate_real_diann_batches(_load_real_protein_container("diann_protein"))

    container = impute_row_median(
        container,
        assay_name="proteins",
        source_layer="raw",
        new_layer_name="imputed_real",
    )
    container = integrate_none(
        container,
        batch_key="batch",
        assay_name="proteins",
        base_layer="imputed_real",
        new_layer_name="none_real",
    )
    container = integrate_limma(
        container,
        batch_key="batch",
        assay_name="proteins",
        base_layer="imputed_real",
        new_layer_name="limma_real",
        covariates=["condition"],
    )

    assay = container.assays["proteins"]
    imputed = assay.layers["imputed_real"].X
    baseline = assay.layers["none_real"].X
    limma = assay.layers["limma_real"].X

    none_asw = compute_batch_asw(
        container,
        assay_name="proteins",
        layer_name="none_real",
        batch_key="batch",
    )
    limma_asw = compute_batch_asw(
        container,
        assay_name="proteins",
        layer_name="limma_real",
        batch_key="batch",
    )
    none_batch_mixing = compute_batch_mixing_metric(
        container,
        assay_name="proteins",
        layer_name="none_real",
        batch_key="batch",
        n_neighbors=5,
    )
    limma_batch_mixing = compute_batch_mixing_metric(
        container,
        assay_name="proteins",
        layer_name="limma_real",
        batch_key="batch",
        n_neighbors=5,
    )
    none_lisi = compute_lisi_approx(
        container,
        assay_name="proteins",
        layer_name="none_real",
        batch_key="batch",
        n_neighbors=5,
    )
    limma_lisi = compute_lisi_approx(
        container,
        assay_name="proteins",
        layer_name="limma_real",
        batch_key="batch",
        n_neighbors=5,
    )
    none_kbet = compute_kbet(
        container,
        assay_name="proteins",
        layer_name="none_real",
        batch_key="batch",
        n_neighbors=5,
        alpha=0.05,
    )
    limma_kbet = compute_kbet(
        container,
        assay_name="proteins",
        layer_name="limma_real",
        batch_key="batch",
        n_neighbors=5,
        alpha=0.05,
    )
    none_ilisi = compute_ilisi(
        container,
        assay_name="proteins",
        layer_name="none_real",
        batch_key="batch",
        n_neighbors=5,
        perplexity=4.0,
    )
    limma_ilisi = compute_ilisi(
        container,
        assay_name="proteins",
        layer_name="limma_real",
        batch_key="batch",
        n_neighbors=5,
        perplexity=4.0,
    )
    report = integration_quality_report(
        container,
        assay_name="proteins",
        layer_name="limma_real",
        batch_key="batch",
    )

    assert imputed.shape == (24, 2719)
    assert baseline.shape == (24, 2719)
    assert limma.shape == (24, 2719)
    assert np.isfinite(imputed).all()
    assert np.isfinite(baseline).all()
    assert np.isfinite(limma).all()
    np.testing.assert_allclose(baseline, imputed)
    assert not np.allclose(limma, baseline)

    assert -1.0 <= none_asw <= 1.0
    assert -1.0 <= limma_asw <= 1.0
    assert 0.0 <= none_batch_mixing <= 1.0
    assert 0.0 <= limma_batch_mixing <= 1.0
    assert 1.0 <= none_lisi <= 3.0
    assert 1.0 <= limma_lisi <= 3.0
    assert 0.0 <= none_kbet <= 1.0
    assert 0.0 <= limma_kbet <= 1.0
    assert 0.0 <= none_ilisi <= 1.0
    assert 0.0 <= limma_ilisi <= 1.0
    assert limma_asw < none_asw
    assert limma_batch_mixing > none_batch_mixing
    assert limma_lisi > none_lisi
    assert limma_kbet >= none_kbet
    assert limma_ilisi >= none_ilisi

    assert set(report) == {"batch_asw", "batch_mixing", "lisi_approx", "interpretation"}
    assert report["batch_asw"] == pytest.approx(limma_asw)
    assert (
        report["interpretation"]["lisi_approx"]
        == "Higher is better (max = n_valid_batches, here: 3)"
    )
    assert [entry.action for entry in container.history[-3:]] == [
        "impute_row_median",
        "integration_none",
        "integration_limma",
    ]
