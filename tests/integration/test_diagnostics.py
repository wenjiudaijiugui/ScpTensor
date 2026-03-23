"""Tests for integration diagnostics metrics."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp

from scptensor.core.structures import Assay, ScpContainer, ScpMatrix
from scptensor.integration.diagnostics import (
    compute_batch_asw,
    compute_batch_mixing_metric,
    compute_ilisi,
    compute_kbet,
    compute_lisi_approx,
    integration_quality_report,
)


def _make_container(
    X: np.ndarray,
    batches: list[str],
    *,
    assay_name: str = "pca",
    layer_name: str = "X",
) -> ScpContainer:
    obs = pl.DataFrame(
        {
            "_index": [f"S{i:03d}" for i in range(len(batches))],
            "batch": batches,
        }
    )
    var = pl.DataFrame({"_index": [f"F{i:03d}" for i in range(X.shape[1])]})
    assay = Assay(var=var, layers={layer_name: ScpMatrix(X=X)})
    return ScpContainer(obs=obs, assays={assay_name: assay}, sample_id_col="_index")


def test_compute_batch_asw_returns_float_in_range() -> None:
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [5.0, 5.0],
            [5.1, 5.0],
            [5.0, 5.1],
        ]
    )
    container = _make_container(X, ["A", "A", "A", "B", "B", "B"])
    score = compute_batch_asw(container)
    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0


def test_compute_batch_asw_small_valid_set_returns_zero() -> None:
    X = np.array(
        [
            [0.0, 0.0],
            [np.nan, np.nan],
            [1.0, 1.0],
        ]
    )
    container = _make_container(X, ["A", "A", "B"])
    assert compute_batch_asw(container) == 0.0


def test_compute_batch_mixing_metric_handles_sparse_and_nan_rows() -> None:
    dense = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [1.0, 1.0],
            [1.1, 1.0],
            [np.nan, 1.1],
            [np.inf, 1.2],
        ]
    )
    X = sp.csr_matrix(np.nan_to_num(dense, nan=0.0))
    container = _make_container(X, ["A", "A", "A", "B", "B", "B", "B"])
    # Put NaN row back in dense path via layer replacement to test filtering
    container.assays["pca"].layers["X"] = ScpMatrix(X=dense)

    score = compute_batch_mixing_metric(container, n_neighbors=3)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_compute_batch_mixing_metric_single_sample_returns_one() -> None:
    X = np.array([[1.0, 2.0]])
    container = _make_container(X, ["A"])
    assert compute_batch_mixing_metric(container) == 1.0


def test_compute_lisi_approx_range_two_batches() -> None:
    X = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.0],
            [0.0, 0.2],
            [1.0, 1.0],
            [1.2, 1.0],
            [1.0, 1.2],
        ]
    )
    container = _make_container(X, ["A", "A", "A", "B", "B", "B"])
    score = compute_lisi_approx(container, n_neighbors=3)
    assert isinstance(score, float)
    assert 1.0 <= score <= 2.0


def test_compute_lisi_approx_single_batch_returns_one() -> None:
    X = np.array([[0.0, 1.0], [0.5, 1.5], [1.0, 2.0]])
    container = _make_container(X, ["A", "A", "A"])
    assert compute_lisi_approx(container) == 1.0


def test_compute_kbet_distinguishes_mixed_from_segregated_batches() -> None:
    mixed_x = np.column_stack([np.arange(12, dtype=float), np.zeros(12, dtype=float)])
    mixed_container = _make_container(mixed_x, ["A", "B"] * 6)

    segregated_x = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [0.3, 0.0],
            [0.4, 0.0],
            [0.5, 0.0],
            [10.0, 0.0],
            [10.1, 0.0],
            [10.2, 0.0],
            [10.3, 0.0],
            [10.4, 0.0],
            [10.5, 0.0],
        ]
    )
    segregated_container = _make_container(segregated_x, ["A"] * 6 + ["B"] * 6)

    mixed_score = compute_kbet(mixed_container, n_neighbors=5, alpha=0.05)
    segregated_score = compute_kbet(segregated_container, n_neighbors=5, alpha=0.05)

    assert 0.0 <= mixed_score <= 1.0
    assert 0.0 <= segregated_score <= 1.0
    assert mixed_score > segregated_score
    assert mixed_score > 0.8
    assert segregated_score < 0.2


def test_compute_ilisi_scaled_and_raw_distinguish_mixing_quality() -> None:
    mixed_x = np.array([[i // 2, 0.0] for i in range(12)], dtype=float)
    mixed_container = _make_container(mixed_x, ["A", "B"] * 6)

    segregated_x = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [0.3, 0.0],
            [0.4, 0.0],
            [0.5, 0.0],
            [10.0, 0.0],
            [10.1, 0.0],
            [10.2, 0.0],
            [10.3, 0.0],
            [10.4, 0.0],
            [10.5, 0.0],
        ]
    )
    segregated_container = _make_container(segregated_x, ["A"] * 6 + ["B"] * 6)

    mixed_scaled = compute_ilisi(mixed_container, n_neighbors=5, perplexity=4.0)
    mixed_raw = compute_ilisi(mixed_container, n_neighbors=5, perplexity=4.0, scale=False)
    segregated_scaled = compute_ilisi(segregated_container, n_neighbors=5, perplexity=4.0)
    segregated_raw = compute_ilisi(
        segregated_container,
        n_neighbors=5,
        perplexity=4.0,
        scale=False,
    )

    assert 0.0 <= mixed_scaled <= 1.0
    assert 0.0 <= segregated_scaled <= 1.0
    assert 1.0 <= mixed_raw <= 2.0
    assert 1.0 <= segregated_raw <= 2.0
    assert mixed_scaled == pytest.approx(mixed_raw - 1.0)
    assert segregated_scaled == pytest.approx(segregated_raw - 1.0)
    assert mixed_scaled > segregated_scaled
    assert mixed_scaled > 0.55
    assert segregated_scaled < 0.25


def test_compute_ilisi_single_batch_returns_one() -> None:
    X = np.array([[0.0, 1.0], [0.5, 1.5], [1.0, 2.0]])
    container = _make_container(X, ["A", "A", "A"])
    assert compute_ilisi(container) == 1.0


@pytest.mark.parametrize(
    "fn", [compute_batch_mixing_metric, compute_lisi_approx, compute_kbet, compute_ilisi]
)
def test_neighbor_based_metrics_reject_nonpositive_neighbor_count(fn) -> None:
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    container = _make_container(X, ["A", "B"])

    with pytest.raises(ValueError, match="n_neighbors must be positive"):
        fn(container, n_neighbors=0)


def test_standardized_metrics_validate_alpha_and_perplexity() -> None:
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    container = _make_container(X, ["A", "B", "A"])

    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        compute_kbet(container, alpha=0.0)

    with pytest.raises(ValueError, match="perplexity must be positive"):
        compute_ilisi(container, perplexity=0.0)


@pytest.mark.parametrize(
    "fn",
    [
        compute_batch_asw,
        compute_batch_mixing_metric,
        compute_ilisi,
        compute_kbet,
        compute_lisi_approx,
        integration_quality_report,
    ],
)
def test_diagnostics_raise_for_missing_assay_layer_and_batch(fn) -> None:
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    container = _make_container(X, ["A", "B"])

    with pytest.raises(ValueError, match="Assay 'missing' not found"):
        fn(container, assay_name="missing")

    with pytest.raises(ValueError, match="Layer 'missing' not found"):
        fn(container, layer_name="missing")

    with pytest.raises(ValueError, match="Batch key 'missing' not found in obs"):
        fn(container, batch_key="missing")


def test_integration_quality_report_structure() -> None:
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [1.0, 1.0],
            [1.1, 1.0],
        ]
    )
    container = _make_container(X, ["A", "A", "B", "B"])
    report = integration_quality_report(container)

    assert set(report.keys()) == {"batch_asw", "batch_mixing", "lisi_approx", "interpretation"}
    assert isinstance(report["batch_asw"], float)
    assert isinstance(report["batch_mixing"], float)
    assert isinstance(report["lisi_approx"], float)
    assert isinstance(report["interpretation"], dict)


def test_integration_quality_report_uses_valid_batch_count_in_lisi_interpretation() -> None:
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [np.nan, np.nan],
            [np.nan, np.nan],
        ]
    )
    container = _make_container(X, ["A", "A", "B", "B"])

    report = integration_quality_report(container)

    assert (
        report["interpretation"]["lisi_approx"]
        == "Higher is better (max = n_valid_batches, here: 1)"
    )
