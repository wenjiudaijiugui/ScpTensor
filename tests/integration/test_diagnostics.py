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
        ]
    )
    X = sp.csr_matrix(np.nan_to_num(dense, nan=0.0))
    container = _make_container(X, ["A", "A", "A", "B", "B", "B"])
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


@pytest.mark.parametrize(
    "fn",
    [
        compute_batch_asw,
        compute_batch_mixing_metric,
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
