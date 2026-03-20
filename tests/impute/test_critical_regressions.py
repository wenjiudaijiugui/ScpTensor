"""Critical regression tests for imputation math/engineering fixes."""

from __future__ import annotations

import numpy as np
import polars as pl
import scipy.sparse as sp_sparse

from scptensor.core import Assay, ScpContainer, ScpMatrix
from scptensor.impute.bpca import bpca_impute
from scptensor.impute.knn import impute_knn, knn_impute
from scptensor.impute.minprob import minprob_impute
from scptensor.impute.qrilc import impute_qrilc, qrilc_impute


def test_minprob_handles_negative_log_values_without_crash() -> None:
    """Regression: old implementation crashed when min_detected < 0 (negative scale)."""
    x = np.array(
        [
            [-2.0, np.nan, -1.0],
            [-1.8, -0.5, np.nan],
            [np.nan, -0.3, -0.8],
            [-1.5, -0.4, -0.7],
        ],
        dtype=np.float64,
    )

    x_out = minprob_impute(x, sigma=2.0, random_state=7)
    missing_mask = np.isnan(x)

    assert not np.any(np.isnan(x_out))
    # For log-scale negative data, imputations should stay in the same domain.
    assert np.any(x_out[missing_mask] < 0)


def test_qrilc_does_not_force_nonnegative_in_log_space() -> None:
    """Regression: old implementation clipped all imputed values to >=0."""
    x = np.array(
        [
            [-5.1, np.nan],
            [-4.8, -3.2],
            [np.nan, -3.1],
            [-4.9, -3.0],
            [-5.0, -3.3],
        ],
        dtype=np.float64,
    )
    x_out = qrilc_impute(x, q=0.2, random_state=11)
    missing_mask = np.isnan(x)

    assert not np.any(np.isnan(x_out))
    assert np.any(x_out[missing_mask] < 0)


def test_knn_distance_weighting_normalizes_over_used_neighbors_only() -> None:
    """Regression: old distance mode normalized by all valid neighbors, not top-k."""
    x = np.array(
        [
            [0.0, np.nan],
            [0.1, 10.0],
            [0.2, 10.0],
            [0.3, 0.0],
            [0.4, 0.0],
            [0.5, 0.0],
            [0.6, 0.0],
            [0.7, 0.0],
            [0.8, 0.0],
        ],
        dtype=np.float64,
    )

    x_out = knn_impute(x, n_neighbors=2, weights="distance", oversample_factor=10)
    # Correct top-2 weighted value should be very close to 10.
    assert x_out[0, 1] > 9.0


def test_bpca_handles_single_informative_row_gracefully() -> None:
    """Regression: old BPCA could crash on degenerate covariance (ARPACK error)."""
    x = np.full((10, 5), np.nan, dtype=np.float64)
    x[0] = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    x_out = bpca_impute(x, n_components=4, max_iter=30, random_state=3)
    assert not np.any(np.isnan(x_out))
    assert x_out.shape == x.shape


def test_knn_wrapper_handles_sparse_input_without_type_error() -> None:
    """Regression: old wrapper passed sparse matrix to np.isnan and crashed."""
    x = np.array(
        [
            [1.0, np.nan, 3.0],
            [2.0, 5.0, np.nan],
            [4.0, 6.0, 7.0],
        ],
        dtype=np.float64,
    )
    assay = Assay(var=pl.DataFrame({"_index": ["p1", "p2", "p3"]}))
    assay.add_layer("raw", ScpMatrix(X=sp_sparse.csr_matrix(x), M=None))
    container = ScpContainer(
        obs=pl.DataFrame({"_index": ["s1", "s2", "s3"]}),
        assays={"protein": assay},
    )

    out = impute_knn(container, assay_name="protein", source_layer="raw", k=2)
    x_out = out.assays["protein"].layers["imputed_knn"].X
    assert not np.any(np.isnan(x_out))


def test_knn_no_missing_fast_path_does_not_append_history() -> None:
    """Regression: KNN no-missing fast path currently returns without logging."""
    x = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float64,
    )
    assay = Assay(var=pl.DataFrame({"_index": ["p1", "p2"]}))
    assay.add_layer("raw", ScpMatrix(X=x, M=None))
    container = ScpContainer(
        obs=pl.DataFrame({"_index": ["s1", "s2"]}),
        assays={"protein": assay},
    )

    initial_len = len(container.history)
    out = impute_knn(container, assay_name="protein", source_layer="raw", k=1)

    assert len(out.history) == initial_len
    np.testing.assert_allclose(out.assays["protein"].layers["imputed_knn"].X, x)


def test_qrilc_no_missing_fast_path_still_logs_history() -> None:
    """Regression: QRILC no-missing path keeps its current logging behavior."""
    x = np.array(
        [
            [8.0, 7.5],
            [6.2, 5.9],
        ],
        dtype=np.float64,
    )
    assay = Assay(var=pl.DataFrame({"_index": ["p1", "p2"]}))
    assay.add_layer("raw", ScpMatrix(X=x, M=None))
    container = ScpContainer(
        obs=pl.DataFrame({"_index": ["s1", "s2"]}),
        assays={"protein": assay},
    )

    initial_len = len(container.history)
    out = impute_qrilc(container, assay_name="protein", source_layer="raw", q=0.01)

    assert len(out.history) == initial_len + 1
    assert out.history[-1].action == "impute_qrilc"
