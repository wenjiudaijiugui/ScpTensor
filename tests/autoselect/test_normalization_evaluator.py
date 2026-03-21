"""Tests for literature-driven normalization scoring in AutoSelect."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from scptensor.autoselect.evaluators.normalization import NormalizationEvaluator
from scptensor.core import Assay, ScpContainer, ScpMatrix
from scptensor.transformation import log_transform


def _build_structured_container(
    *,
    include_batch: bool = True,
    include_group: bool = True,
    n_samples: int = 80,
    n_features: int = 120,
    batch_scale: float = 3.0,
    group_effect: float = 1.6,
    missing_rate: float = 0.10,
    sample_shift_sigma: float = 0.0,
    seed: int = 42,
) -> ScpContainer:
    """Build a structured proteomics-like matrix with optional batch/group labels."""
    rng = np.random.default_rng(seed)

    cell_types = np.array([0, 1] * (n_samples // 2))
    batches = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    rng.shuffle(cell_types)
    rng.shuffle(batches)

    x = rng.lognormal(mean=10.0, sigma=0.35, size=(n_samples, n_features))
    # Biological signal on marker proteins.
    marker_idx = np.arange(min(25, n_features))
    x[cell_types == 1, : len(marker_idx)] *= group_effect

    if include_batch:
        x[batches == 1] *= batch_scale

    if sample_shift_sigma > 0:
        # Sample-level global intensity offsets (e.g., loading/injection bias).
        sample_shift = rng.lognormal(mean=0.0, sigma=sample_shift_sigma, size=(n_samples, 1))
        x *= sample_shift

    # Measurement noise + moderate missingness.
    x *= rng.lognormal(mean=0.0, sigma=0.08, size=x.shape)
    missing = rng.random(size=x.shape) < missing_rate
    x[missing] = np.nan

    obs_dict: dict[str, list[str]] = {"_index": [f"C{i:03d}" for i in range(n_samples)]}
    if include_batch:
        obs_dict["batch"] = [f"B{b}" for b in batches]
    if include_group:
        obs_dict["cell_type"] = [f"T{g}" for g in cell_types]

    obs = pl.DataFrame(obs_dict)
    var = pl.DataFrame(
        {
            "_index": [f"P{j:04d}" for j in range(n_features)],
            "protein": [f"Protein_{j:04d}" for j in range(n_features)],
        }
    )

    assay = Assay(var=var)
    assay.add_layer("raw", ScpMatrix(X=x))
    container = ScpContainer(obs=obs)
    container.add_assay("proteins", assay)
    return container


def test_methods_include_norm_none() -> None:
    """No-normalization baseline should be available as candidate."""
    evaluator = NormalizationEvaluator()
    assert "norm_none" in evaluator.methods


def test_composite_score_uses_balance_axis() -> None:
    """Balanced batch/bio tradeoff should score higher than imbalanced results."""
    evaluator = NormalizationEvaluator()
    base = {
        "batch_removal": 0.80,
        "bio_conservation": 0.80,
        "technical_quality": 0.70,
        "balance_score": 1.00,
    }
    balanced = evaluator.compute_overall_score(base)
    imbalanced = evaluator.compute_overall_score({**base, "balance_score": 0.10})
    assert balanced > imbalanced


def test_batch_aware_scores_improve_over_none_baseline() -> None:
    """On strong batch-effect data, at least one normalization should beat none."""
    evaluator = NormalizationEvaluator()
    container = _build_structured_container(include_batch=True, include_group=True, batch_scale=4.0)

    _, report = evaluator.run_all(
        container=container,
        assay_name="proteins",
        source_layer="raw",
        keep_all=True,
        selection_strategy="quality",
    )

    scores_by_method = {
        result.method_name: result.scores
        for result in report.results
        if result.error is None and "batch_removal" in result.scores
    }
    assert "norm_none" in scores_by_method

    none_batch_score = scores_by_method["norm_none"]["batch_removal"]
    improved = [
        metrics["batch_removal"]
        for method, metrics in scores_by_method.items()
        if method != "norm_none"
    ]
    assert improved
    assert max(improved) > none_batch_score


def test_raw_layer_excludes_scale_sensitive_methods_from_autoselect() -> None:
    """AutoSelect should not compare quantile-family methods on raw layers."""
    evaluator = NormalizationEvaluator()
    container = _build_structured_container(include_batch=True, include_group=True)

    _, report = evaluator.run_all(
        container=container,
        assay_name="proteins",
        source_layer="raw",
        keep_all=True,
        selection_strategy="quality",
    )

    assert {result.method_name for result in report.results} == {
        "norm_none",
        "norm_mean",
        "norm_median",
    }
    assert set(report.method_contracts) == {"norm_none", "norm_mean", "norm_median"}
    assert all(
        contract["source_layer_logged"] is False for contract in report.method_contracts.values()
    )
    assert all(
        contract["comparison_scale"] == "raw_or_unknown"
        for contract in report.method_contracts.values()
    )
    assert (
        "excluded scale-sensitive methods norm_quantile and norm_trqn"
        in report.recommendation_reason
    )


def test_trqn_and_quantile_scores_are_not_identical() -> None:
    """Adaptive TRQN path should avoid degenerating to exact quantile scores."""
    evaluator = NormalizationEvaluator()
    container = _build_structured_container(
        include_batch=True,
        include_group=True,
        n_samples=120,
        n_features=180,
        batch_scale=2.5,
        seed=123,
    )
    container = log_transform(
        container,
        assay_name="proteins",
        source_layer="raw",
        new_layer_name="log",
    )

    _, report = evaluator.run_all(
        container=container,
        assay_name="proteins",
        source_layer="log",
        keep_all=True,
        selection_strategy="quality",
    )

    by_method = {r.method_name: r for r in report.results if r.error is None}
    assert "norm_quantile" in by_method
    assert "norm_trqn" in by_method

    q = by_method["norm_quantile"].scores
    t = by_method["norm_trqn"].scores
    composite_keys = (
        "batch_removal",
        "bio_conservation",
        "technical_quality",
        "balance_score",
    )
    assert any(abs(q[k] - t[k]) > 1e-9 for k in composite_keys)
    assert report.method_contracts["norm_quantile"]["input_scale_requirement"] == "logged"
    assert report.method_contracts["norm_trqn"]["input_scale_requirement"] == "logged"
    assert report.method_contracts["norm_quantile"]["source_layer_logged"] is True
    assert report.method_contracts["norm_trqn"]["comparison_scale"] == "logged"
    assert "compared on logged source layer 'log'" in report.recommendation_reason


def test_log_transform_skipped_passthrough_still_counts_as_logged_provenance() -> None:
    """Skipped passthrough layers should still reopen quantile-family candidates."""
    evaluator = NormalizationEvaluator()
    container = _build_structured_container(
        include_batch=True,
        include_group=True,
        n_samples=120,
        n_features=180,
        batch_scale=2.5,
        seed=321,
    )
    container = log_transform(
        container,
        assay_name="proteins",
        source_layer="raw",
        new_layer_name="log",
    )

    with pytest.warns(UserWarning, match="already log-transformed"):
        container = log_transform(
            container,
            assay_name="proteins",
            source_layer="log",
            new_layer_name="logged_checked",
        )

    _, report = evaluator.run_all(
        container=container,
        assay_name="proteins",
        source_layer="logged_checked",
        keep_all=True,
        selection_strategy="quality",
    )

    assert "norm_quantile" in report.method_contracts
    assert "norm_trqn" in report.method_contracts
    assert report.method_contracts["norm_quantile"]["source_layer_logged"] is True
    assert report.method_contracts["norm_trqn"]["comparison_scale"] == "logged"
    assert "compared on logged source layer 'logged_checked'" in report.recommendation_reason


def test_missing_batch_metadata_uses_neutral_batch_scores() -> None:
    """When batch labels are absent, batch-specific metrics should be neutral."""
    evaluator = NormalizationEvaluator()
    container = _build_structured_container(include_batch=False, include_group=True)

    method = evaluator.methods["norm_median"]
    _, result = evaluator.evaluate_method(
        container=container,
        method_name="norm_median",
        method_func=method,
        assay_name="proteins",
        source_layer="raw",
    )

    assert result.error is None
    assert result.scores["batch_asw"] == pytest.approx(0.5)
    assert result.scores["batch_mixing"] == pytest.approx(0.5)
    assert 0.0 <= result.overall_score <= 1.0


def test_sparse_matrix_still_produces_informative_asw_scores() -> None:
    """High-missing matrices should not force ASW scores to neutral defaults."""
    evaluator = NormalizationEvaluator()
    container = _build_structured_container(
        include_batch=True,
        include_group=True,
        n_samples=120,
        n_features=600,
        batch_scale=4.0,
        group_effect=2.0,
        missing_rate=0.45,
        seed=7,
    )

    _, report = evaluator.run_all(
        container=container,
        assay_name="proteins",
        source_layer="raw",
        keep_all=True,
        selection_strategy="quality",
    )

    successful = [r for r in report.results if r.error is None]
    assert successful
    assert any(abs(r.scores["batch_asw"] - 0.5) > 1e-3 for r in successful)
    assert any(abs(r.scores["bio_asw"] - 0.5) > 1e-3 for r in successful)


def test_loading_bias_metric_rewards_global_shift_correction() -> None:
    """Mean/median scaling should improve loading-bias reduction over none."""
    evaluator = NormalizationEvaluator()
    container = _build_structured_container(
        include_batch=False,
        include_group=True,
        n_samples=100,
        n_features=500,
        group_effect=1.3,
        missing_rate=0.15,
        sample_shift_sigma=0.8,
        seed=19,
    )

    _, none_result = evaluator.evaluate_method(
        container=container,
        method_name="norm_none",
        method_func=evaluator.methods["norm_none"],
        assay_name="proteins",
        source_layer="raw",
    )
    _, median_result = evaluator.evaluate_method(
        container=container,
        method_name="norm_median",
        method_func=evaluator.methods["norm_median"],
        assay_name="proteins",
        source_layer="raw",
    )

    assert none_result.error is None
    assert median_result.error is None
    assert (
        median_result.scores["loading_bias_reduction"]
        > none_result.scores["loading_bias_reduction"]
    )
    assert median_result.scores["technical_quality"] > none_result.scores["technical_quality"]


def test_run_all_accepts_protein_alias() -> None:
    """Normalization evaluator should resolve proteins/protein assay aliases."""
    container = _build_structured_container()
    assay = container.assays.pop("proteins")
    container.assays["protein"] = assay

    evaluator = NormalizationEvaluator()
    result_container, report = evaluator.run_all(
        container=container,
        assay_name="proteins",
        source_layer="raw",
    )

    assert report.best_result is not None
    assert report.best_result.error is None
    assert "protein" in result_container.assays


def test_run_all_accepts_protein_assay_alias() -> None:
    """Normalization evaluator should accept assay_name='protein' for proteins assay."""
    evaluator = NormalizationEvaluator()
    container = _build_structured_container(include_batch=True, include_group=True)
    assay = container.assays.pop("proteins")
    container.assays["protein"] = assay

    result_container, report = evaluator.run_all(
        container=container,
        assay_name="proteins",
        source_layer="raw",
        keep_all=False,
        selection_strategy="quality",
    )

    assert report.best_result is not None
    assert report.best_result.error is None
    assert "protein" in result_container.assays
