"""Regression tests for AutoSelect imputation evaluator behavior."""

from __future__ import annotations

import numpy as np
import polars as pl

from scptensor.autoselect import AutoSelector
from scptensor.core import Assay, ScpContainer, ScpMatrix


def _build_container() -> ScpContainer:
    """Create a small protein container with realistic missingness."""
    rng = np.random.default_rng(7)
    n_samples = 36
    n_features = 48

    # Create structured signal so methods are not trivially tied.
    row_effect = rng.normal(0.0, 0.4, size=(n_samples, 1))
    col_effect = rng.normal(0.0, 0.6, size=(1, n_features))
    noise = rng.normal(0.0, 0.2, size=(n_samples, n_features))
    x = row_effect + col_effect + noise

    missing_mask = rng.random(size=x.shape) < 0.2
    x_missing = x.copy()
    x_missing[missing_mask] = np.nan

    obs = pl.DataFrame({"_index": [f"S{i}" for i in range(n_samples)]})
    var = pl.DataFrame({"_index": [f"P{j}" for j in range(n_features)]})

    assay = Assay(var=var)
    assay.add_layer("raw", ScpMatrix(X=x_missing))

    container = ScpContainer(obs=obs)
    container.add_assay("proteins", assay)
    return container


def test_impute_scores_not_all_saturated_to_one() -> None:
    """Imputation scores should not collapse to all 1.0."""
    container = _build_container()
    selector = AutoSelector(stages=["impute"], keep_all=False, selection_strategy="quality")

    _, report = selector.run_stage(
        container=container,
        stage="impute",
        assay_name="proteins",
        source_layer="raw",
    )

    successful = [result for result in report.results if result.error is None]
    assert successful, "Expected at least one successful imputation method."

    rounded_scores = {round(result.overall_score, 4) for result in successful}
    assert len(rounded_scores) > 1, "Expected varied overall scores across methods."

    none_result = next(result for result in successful if result.method_name == "none")
    assert none_result.overall_score < 0.95


def test_impute_none_not_selected_as_best_when_missing_present() -> None:
    """`none` imputation should not win when there are missing entries."""
    container = _build_container()
    selector = AutoSelector(stages=["impute"], keep_all=False, selection_strategy="quality")

    _, report = selector.run_stage(
        container=container,
        stage="impute",
        assay_name="proteins",
        source_layer="raw",
    )

    assert report.best_method != "none"
