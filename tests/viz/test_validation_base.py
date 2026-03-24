"""Tests for visualization validation helpers."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from scptensor.core import Assay, ScpContainer, ScpMatrix
from scptensor.core.exceptions import LayerNotFoundError, VisualizationError
from scptensor.viz.base.validation import (
    validate_container,
    validate_features,
    validate_groupby,
    validate_layer,
    validate_plot_data,
)


@pytest.fixture
def validation_container() -> ScpContainer:
    obs = pl.DataFrame({"_index": ["S1", "S2"], "grp": ["A", "B"]})
    var = pl.DataFrame({"_index": ["P1", "P2"]})
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    return ScpContainer(
        obs=obs,
        assays={"proteins": Assay(var=var, layers={"raw": ScpMatrix(X=x)})},
    )


def test_validate_container_errors(validation_container: ScpContainer) -> None:
    """Container validator should reject None and wrong types."""
    validate_container(validation_container)
    with pytest.raises(VisualizationError, match="cannot be None"):
        validate_container(None)  # type: ignore[arg-type]
    with pytest.raises(VisualizationError, match="Expected ScpContainer"):
        validate_container("not_a_container")  # type: ignore[arg-type]


def test_validate_layer_errors(validation_container: ScpContainer) -> None:
    """Layer validator should distinguish assay and layer errors."""
    validate_layer(validation_container, "proteins", "raw")
    with pytest.raises(VisualizationError, match="Assay 'missing' not found"):
        validate_layer(validation_container, "missing", "raw")
    with pytest.raises(LayerNotFoundError, match="Layer 'norm' not found"):
        validate_layer(validation_container, "proteins", "norm")


def test_validate_features_errors(validation_container: ScpContainer) -> None:
    """Feature validator should fail on missing assay/features and empty var."""
    validate_features(validation_container, "proteins", ["P1"])

    with pytest.raises(VisualizationError, match="Assay 'missing' not found"):
        validate_features(validation_container, "missing", ["P1"])

    with pytest.raises(VisualizationError, match="Features not found"):
        validate_features(validation_container, "proteins", ["PX"])

    empty_var_container = ScpContainer(
        obs=pl.DataFrame({"_index": ["S1"]}),
        assays={
            "proteins": Assay(
                var=pl.DataFrame({"_index": []}),
                layers={"raw": ScpMatrix(X=np.zeros((1, 0)))},
            ),
        },
    )
    with pytest.raises(VisualizationError, match="has no features"):
        validate_features(empty_var_container, "proteins", ["P1"])


def test_validate_groupby_and_plot_data(validation_container: ScpContainer) -> None:
    """Grouping/data validators should report explicit validation errors."""
    validate_groupby(validation_container, "grp")
    with pytest.raises(VisualizationError, match="Column 'missing' not found"):
        validate_groupby(validation_container, "missing")

    validate_plot_data(np.array([1.0]), n_min=1)
    with pytest.raises(VisualizationError, match="Insufficient data for plotting"):
        validate_plot_data(np.array([]), n_min=1)
