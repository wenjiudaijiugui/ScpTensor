"""Harmony integration wrapper for single-cell proteomics data.

This module re-exports the Harmony implementation from nonlinear.py.

Reference
---------
Korsunsky I, et al. Fast, sensitive and accurate integration of
single-cell data with Harmony. Nature Methods (2019).
"""

from typing import Final

import numpy as np
import polars as pl

from scptensor.core.structures import Assay, ScpContainer, ScpMatrix
from scptensor.integration.nonlinear import harmony as _harmony_impl

# Public API wrapper - delegates to the actual implementation
harmony: Final = _harmony_impl

__all__ = ["harmony"]


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
    obs = pl.DataFrame({
        "_index": [f"cell_{i}" for i in range(2 * n_samples_per_batch)],
        "batch": ["batch1"] * n_samples_per_batch + ["batch2"] * n_samples_per_batch,
    })

    assay = Assay(var=var)
    assay.add_layer("pca", ScpMatrix(X=X, M=None))
    container = ScpContainer(obs=obs, assays={"protein": assay})

    try:
        result = harmony(
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
