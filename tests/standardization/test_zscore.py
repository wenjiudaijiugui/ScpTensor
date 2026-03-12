"""Tests for scptensor.standardization.zscore."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from scptensor import zscore
from scptensor.core.exceptions import ScpValueError, ValidationError
from scptensor.core.structures import Assay, ScpContainer, ScpMatrix


def _make_container(x: np.ndarray, layer_name: str = "imputed") -> ScpContainer:
    obs = pl.DataFrame({"_index": [f"s{i}" for i in range(x.shape[0])]})
    var = pl.DataFrame({"_index": [f"p{j}" for j in range(x.shape[1])]})
    assay = Assay(var=var, layers={layer_name: ScpMatrix(X=x)})
    return ScpContainer(obs=obs, assays={"protein": assay})


def test_zscore_feature_wise_standardization() -> None:
    x = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
        ]
    )
    container = _make_container(x)

    result = zscore(container, assay_name="protein", source_layer="imputed", new_layer_name="z")
    z = result.assays["protein"].layers["z"].X

    assert np.allclose(np.mean(z, axis=0), 0.0, atol=1e-12)
    assert np.allclose(np.std(z, axis=0, ddof=1), 1.0, atol=1e-12)


def test_zscore_rejects_nan_input() -> None:
    x = np.array([[1.0, np.nan], [2.0, 3.0]])
    container = _make_container(x)

    with pytest.raises(ValidationError, match="requires a complete matrix"):
        zscore(container)


def test_zscore_rejects_invalid_axis() -> None:
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    container = _make_container(x)

    with pytest.raises(ScpValueError, match="Axis must be 0 or 1"):
        zscore(container, axis=2)


def test_zscore_exported_from_top_level() -> None:
    assert callable(zscore)
