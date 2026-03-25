"""Tests for advanced QC recipe visualizations."""

from __future__ import annotations

import matplotlib
import numpy as np
import polars as pl
import pytest

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from scptensor.core.exceptions import VisualizationError
from scptensor.core.structures import Assay, ScpContainer, ScpMatrix
from scptensor.viz.recipes.qc import (
    missing_value_patterns,
    pca_overview,
    plot_qc_missing_value_patterns,
    plot_qc_pca_overview,
)


@pytest.fixture
def qc_overview_container() -> ScpContainer:
    rng = np.random.default_rng(42)
    n_samples = 12
    n_features = 6
    n_pcs = 3

    obs = pl.DataFrame(
        {
            "_index": [f"S{i:02d}" for i in range(n_samples)],
            "batch": np.array(["A"] * 6 + ["B"] * 6),
        },
    )

    protein_var = pl.DataFrame(
        {
            "_index": [f"P{i:02d}" for i in range(n_features)],
            "pca_PC1_loading": rng.normal(size=n_features),
            "pca_PC2_loading": rng.normal(size=n_features),
            "pca_PC3_loading": rng.normal(size=n_features),
        },
    )
    protein_x = rng.normal(size=(n_samples, n_features))
    protein_m = rng.choice([0, 1, 2, 5], size=(n_samples, n_features), p=[0.7, 0.1, 0.1, 0.1])
    proteins = Assay(var=protein_var, layers={"norm": ScpMatrix(X=protein_x, M=protein_m)})

    pca_var = pl.DataFrame(
        {
            "_index": [f"PC{i + 1}" for i in range(n_pcs)],
            "explained_variance_ratio": [0.5, 0.3, 0.2],
        },
    )
    scores = rng.normal(size=(n_samples, n_pcs))
    pca_assay = Assay(var=pca_var, layers={"scores": ScpMatrix(X=scores)})

    return ScpContainer(obs=obs, assays={"proteins": proteins, "pca": pca_assay})


def test_pca_overview_runs_with_color_and_alias(qc_overview_container: ScpContainer) -> None:
    plt.close("all")
    fig = pca_overview(
        qc_overview_container,
        layer="scores",
        assay_name="proteins",
        pca_assay_name="pca",
        n_pcs=3,
        color="batch",
        show=False,
    )
    assert fig is not None
    assert len(fig.axes) >= 4

    fig_alias = plot_qc_pca_overview(
        qc_overview_container,
        layer="scores",
        assay_name="proteins",
        pca_assay_name="pca",
        n_pcs=3,
        show=False,
    )
    assert fig_alias is not None
    plt.close("all")


def test_pca_overview_missing_variance_column_raises(
    qc_overview_container: ScpContainer,
) -> None:
    pca_var = pl.DataFrame({"_index": ["PC1", "PC2", "PC3"]})
    scores = qc_overview_container.assays["pca"].layers["scores"].X.copy()
    qc_overview_container.assays["pca"] = Assay(var=pca_var, layers={"scores": ScpMatrix(X=scores)})

    with pytest.raises(ValueError, match="does not contain explained_variance_ratio column"):
        pca_overview(
            qc_overview_container,
            layer="scores",
            assay_name="proteins",
            pca_assay_name="pca",
            show=False,
        )


def test_missing_value_patterns_runs_and_alias(qc_overview_container: ScpContainer) -> None:
    plt.close("all")
    fig = missing_value_patterns(
        qc_overview_container,
        layer="norm",
        assay_name="proteins",
        groupby="batch",
        show=False,
    )
    assert fig is not None
    assert len(fig.axes) == 4

    fig_alias = plot_qc_missing_value_patterns(
        qc_overview_container,
        layer="norm",
        assay_name="proteins",
        show=False,
    )
    assert fig_alias is not None
    plt.close("all")


def test_missing_value_patterns_no_mask_shows_message(
    qc_overview_container: ScpContainer,
) -> None:
    plt.close("all")
    x = qc_overview_container.assays["proteins"].layers["norm"].X.copy()
    qc_overview_container.assays["proteins"].layers["norm"] = ScpMatrix(X=x, M=None)

    fig = missing_value_patterns(
        qc_overview_container,
        layer="norm",
        assay_name="proteins",
        show=False,
    )
    texts = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert any("No missing values found" in t for t in texts)
    plt.close("all")


def test_missing_value_patterns_invalid_groupby_raises(
    qc_overview_container: ScpContainer,
) -> None:
    with pytest.raises(VisualizationError, match="Column 'missing_group' not found in obs"):
        missing_value_patterns(
            qc_overview_container,
            layer="norm",
            assay_name="proteins",
            groupby="missing_group",
            show=False,
        )
