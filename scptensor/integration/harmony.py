"""Harmony integration wrapper for single-cell proteomics data.

This module re-exports the Harmony implementation from nonlinear.py
with the new integrate_* naming convention.

Reference
---------
Korsunsky I, et al. Fast, sensitive and accurate integration of
single-cell data with Harmony. Nature Methods (2019).
"""

from typing import Final

import numpy as np
import polars as pl

from scptensor.core.structures import Assay, ScpContainer, ScpMatrix
from scptensor.integration.nonlinear import integrate_harmony as _integrate_harmony_impl

# Public API wrapper - delegates to the actual implementation
integrate_harmony: Final = _integrate_harmony_impl


def harmony(
    container: ScpContainer,
    batch_key: str,
    assay_name: str = "protein",
    base_layer: str = "pca",
    new_layer_name: str | None = "harmony",
    theta: float | None = None,
    lamb: float | None = None,
    sigma: float = 0.1,
    nclust: int | None = None,
    max_iter_harmony: int = 10,
    max_iter_cluster: int = 20,
    epsilon_cluster: float = 1e-5,
    epsilon_harmony: float = 1e-4,
) -> ScpContainer:
    """Harmony integration for batch effect correction.

    .. deprecated:: 0.1.0
        Use :func:`integrate_harmony` instead. This function will be removed in a future version.

    Examples
    --------
    >>> container = harmony(container, batch_key='batch')
    """
    import warnings

    warnings.warn(
        "'harmony' is deprecated, use 'integrate_harmony' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return integrate_harmony(
        container=container,
        batch_key=batch_key,
        assay_name=assay_name,
        base_layer=base_layer,
        new_layer_name=new_layer_name,
        theta=theta,
        lamb=lamb,
        sigma=sigma,
        nclust=nclust,
        max_iter_harmony=max_iter_harmony,
        max_iter_cluster=max_iter_cluster,
        epsilon_cluster=epsilon_cluster,
        epsilon_harmony=epsilon_harmony,
    )


__all__ = ["integrate_harmony", "harmony"]


if __name__ == "__main__":
    # Test: Basic functionality
    print("Testing Harmony integration wrapper...")

    np.random.seed(42)
    n_samples_per_batch = 50
    n_features = 30

    # Generate PCA-like data with batch effects
    X_batch1 = np.random.randn(n_samples_per_batch, n_features) * 0.5
    X_batch2 = np.random.randn(n_samples_per_batch, n_features) * 0.5 + 2.0
    X = np.vstack([X_batch1, X_batch2])

    # Create container
    var = pl.DataFrame({"_index": [f"pc_{i}" for i in range(n_features)]})
    obs = pl.DataFrame(
        {
            "_index": [f"cell_{i}" for i in range(2 * n_samples_per_batch)],
            "batch": ["batch1"] * n_samples_per_batch + ["batch2"] * n_samples_per_batch,
        }
    )

    assay = Assay(var=var)
    assay.add_layer("pca", ScpMatrix(X=X, M=None))
    container = ScpContainer(obs=obs, assays={"protein": assay})

    try:
        result = integrate_harmony(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="pca",
            theta=2.0,
            max_iter_harmony=5,
        )

        assert "harmony" in result.assays["protein"].layers
        X_corrected = result.assays["protein"].layers["harmony"].X
        print(f"  Shape: {X_corrected.shape}")
        print("  All tests passed")
    except ImportError:
        print("  harmonypy not installed, skipping test")
        print("  Install with: pip install harmonypy")
