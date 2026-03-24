"""Tests for shared state-aware metrics and layer-lineage helpers."""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import sparse

from scptensor.core import (
    Assay,
    MaskCode,
    ScpContainer,
    ScpMatrix,
    compute_direct_observation_rate,
    compute_layer_state_metrics,
    compute_state_counts,
    compute_state_rates,
    compute_state_summary,
    compute_state_transition_metrics,
    compute_supported_observation_rate,
    compute_uncertainty_burden,
    resolve_layer_lineage,
    resolve_origin_layer,
    resolve_source_layer,
)
from scptensor.impute import impute_row_mean
from scptensor.integration import integrate_none
from scptensor.normalization import normalize
from scptensor.transformation import log_transform


def _make_container(*, assay_key: str = "proteins", with_nan: bool = False) -> ScpContainer:
    obs = pl.DataFrame({"_index": ["S1", "S2", "S3"], "batch": ["B1", "B1", "B2"]})
    var = pl.DataFrame({"_index": ["P1", "P2", "P3"]})
    x = np.array(
        [
            [10.0, 20.0, 30.0],
            [12.0, 18.0, 28.0],
            [11.0, 19.0, 31.0],
        ],
        dtype=np.float64,
    )
    if with_nan:
        x[1, 1] = np.nan
    m = np.zeros(x.shape, dtype=np.int8)
    assay = Assay(
        var=var,
        layers={"raw": ScpMatrix(X=x, M=m)},
        feature_id_col="_index",
    )
    return ScpContainer(obs=obs, assays={assay_key: assay}, sample_id_col="_index")


def test_compute_state_counts_rates_and_summary_dense() -> None:
    matrix = ScpMatrix(
        X=np.ones((2, 4), dtype=np.float64),
        M=np.array(
            [
                [
                    MaskCode.VALID.value,
                    MaskCode.MBR.value,
                    MaskCode.LOD.value,
                    MaskCode.FILTERED.value,
                ],
                [
                    MaskCode.OUTLIER.value,
                    MaskCode.IMPUTED.value,
                    MaskCode.UNCERTAIN.value,
                    MaskCode.VALID.value,
                ],
            ],
            dtype=np.int8,
        ),
    )

    counts = compute_state_counts(matrix)
    rates = compute_state_rates(matrix)
    summary = compute_state_summary(matrix)

    assert counts == {
        "valid_count": 2,
        "mbr_count": 1,
        "lod_count": 1,
        "filtered_count": 1,
        "outlier_count": 1,
        "imputed_count": 1,
        "uncertain_count": 1,
    }
    assert rates["valid_rate"] == 0.25
    assert rates["mbr_rate"] == 0.125
    assert rates["imputed_rate"] == 0.125
    assert compute_direct_observation_rate(matrix) == 0.25
    assert compute_supported_observation_rate(matrix) == 0.375
    assert compute_uncertainty_burden(matrix) == 0.5
    assert summary["supported_observation_rate"] == 0.375
    assert summary["uncertainty_burden"] == 0.5


def test_compute_state_rates_respects_sparse_implicit_valid_entries() -> None:
    matrix = ScpMatrix(
        X=np.ones((2, 3), dtype=np.float64),
        M=sparse.csr_matrix(
            np.array(
                [
                    [MaskCode.MBR.value, 0, 0],
                    [0, 0, MaskCode.IMPUTED.value],
                ],
                dtype=np.int8,
            ),
        ),
    )

    counts = compute_state_counts(matrix)
    rates = compute_state_rates(matrix)

    assert counts["valid_count"] == 4
    assert counts["mbr_count"] == 1
    assert counts["imputed_count"] == 1
    assert rates["valid_rate"] == 4.0 / 6.0
    assert rates["mbr_rate"] == 1.0 / 6.0
    assert rates["imputed_rate"] == 1.0 / 6.0


def test_resolve_layer_lineage_tracks_log_then_normalization_chain() -> None:
    container = _make_container(assay_key="proteins")

    log_transform(
        container,
        assay_name="protein",
        source_layer="raw",
        new_layer_name="log2",
        detect_logged=False,
    )
    normalize(
        container,
        method="mean",
        assay_name="protein",
        source_layer="log2",
        new_layer_name="norm_mean",
    )

    lineage = resolve_layer_lineage(container, assay_name="protein", layer_name="norm_mean")
    origin = resolve_origin_layer(container, assay_name="protein", layer_name="norm_mean")
    creation_info = container.assays["proteins"].layers["norm_mean"].metadata.creation_info

    assert len(lineage) == 2
    assert lineage[0].action == "normalization_sample_mean"
    assert lineage[0].source_assay == "proteins"
    assert lineage[0].source_layer == "log2"
    assert lineage[1].action == "log_transform"
    assert lineage[1].source_assay == "proteins"
    assert lineage[1].source_layer == "raw"
    assert origin == ("proteins", "raw")
    assert creation_info["source_assay"] == "proteins"
    assert creation_info["source_layer"] == "log2"
    assert creation_info["action"] == "normalization_sample_mean"


def test_resolve_layer_lineage_falls_back_to_history_for_imputed_layer() -> None:
    container = _make_container(assay_key="proteins", with_nan=True)

    impute_row_mean(
        container,
        assay_name="protein",
        source_layer="raw",
        new_layer_name="imputed_row_mean_test",
    )

    lineage = resolve_layer_lineage(
        container,
        assay_name="protein",
        layer_name="imputed_row_mean_test",
    )
    origin = resolve_origin_layer(
        container,
        assay_name="protein",
        layer_name="imputed_row_mean_test",
    )

    assert len(lineage) == 1
    assert lineage[0].action == "impute_row_mean"
    assert lineage[0].source_assay == "proteins"
    assert lineage[0].source_layer == "raw"
    creation_info = (
        container.assays["proteins"].layers["imputed_row_mean_test"].metadata.creation_info
    )
    assert creation_info["source_assay"] == "proteins"
    assert creation_info["source_layer"] == "raw"
    assert creation_info["action"] == "impute_row_mean"
    assert origin == ("proteins", "raw")


def test_resolve_source_layer_returns_immediate_parent() -> None:
    container = _make_container(assay_key="proteins")

    log_transform(
        container,
        assay_name="protein",
        source_layer="raw",
        new_layer_name="log2",
        detect_logged=False,
    )
    normalize(
        container,
        method="mean",
        assay_name="protein",
        source_layer="log2",
        new_layer_name="norm_mean",
    )

    assert resolve_source_layer(container, assay_name="protein", layer_name="norm_mean") == (
        "proteins",
        "log2",
    )


def test_compute_layer_state_transition_metrics_tracks_source_delta() -> None:
    container = _make_container(assay_key="proteins", with_nan=True)
    raw_mask = container.assays["proteins"].layers["raw"].M
    assert raw_mask is not None
    raw_mask[1, 1] = MaskCode.LOD.value

    impute_row_mean(
        container,
        assay_name="protein",
        source_layer="raw",
        new_layer_name="imputed_row_mean_test",
    )

    current_metrics = compute_layer_state_metrics(
        container,
        assay_name="protein",
        layer_name="imputed_row_mean_test",
    )
    transition = compute_state_transition_metrics(
        container,
        assay_name="protein",
        layer_name="imputed_row_mean_test",
    )

    assert current_metrics["imputed_rate"] == 1.0 / 9.0
    assert current_metrics["uncertainty_burden"] == 1.0 / 9.0
    assert transition["reference_lod_rate"] == 1.0 / 9.0
    assert transition["reference_imputed_rate"] == 0.0
    assert transition["delta_imputed_rate"] == 1.0 / 9.0
    assert transition["delta_lod_rate"] == -(1.0 / 9.0)
    assert transition["delta_uncertainty_burden"] == 1.0 / 9.0


def test_resolve_layer_lineage_tracks_integration_none_output() -> None:
    container = _make_container(assay_key="proteins")

    integrate_none(
        container,
        assay_name="protein",
        base_layer="raw",
        new_layer_name="none_copy",
    )

    lineage = resolve_layer_lineage(container, assay_name="protein", layer_name="none_copy")
    origin = resolve_origin_layer(container, assay_name="protein", layer_name="none_copy")
    creation_info = container.assays["proteins"].layers["none_copy"].metadata.creation_info

    assert len(lineage) == 1
    assert lineage[0].action == "integration_none"
    assert lineage[0].source_assay == "proteins"
    assert lineage[0].source_layer == "raw"
    assert creation_info["source_assay"] == "proteins"
    assert creation_info["source_layer"] == "raw"
    assert creation_info["action"] == "integration_none"
    assert origin == ("proteins", "raw")
