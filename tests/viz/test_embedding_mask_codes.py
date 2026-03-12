"""Regression tests for embedding visualization mask-code handling."""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from scptensor.core import Assay, ScpContainer, ScpMatrix
from scptensor.viz.recipes.embedding import embedding


def test_embedding_renders_imputed_mask_code_points() -> None:
    """mask code 5 (IMPUTED) should still be rendered when showing missing values."""
    plt.close("all")
    obs = pl.DataFrame(
        {
            "_index": ["S1", "S2", "S3", "S4"],
            "reduce_umap_1": [0.0, 1.0, 2.0, 3.0],
            "reduce_umap_2": [1.0, 2.0, 3.0, 4.0],
        }
    )
    var = pl.DataFrame({"_index": ["P1"]})
    x = np.array([[1.0], [2.0], [3.0], [4.0]])
    m = np.array([[0], [5], [0], [0]], dtype=np.int8)
    container = ScpContainer(
        obs=obs,
        assays={"proteins": Assay(var=var, layers={"raw": ScpMatrix(X=x, M=m)})},
    )

    ax = embedding(
        container,
        basis="reduce_umap",
        color="P1",
        layer="raw",
        show_missing_values=True,
    )

    total_points = sum(collection.get_offsets().shape[0] for collection in ax.collections)
    assert total_points == container.n_samples
    assert len(ax.collections) >= 2  # detected + mask-code overlay
    plt.close("all")
