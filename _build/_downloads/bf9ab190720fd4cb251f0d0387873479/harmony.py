"""
Harmony integration algorithm for single-cell proteomics data.

This module provides a wrapper for the Harmony algorithm for batch effect correction.
The actual implementation uses the harmonypy package via the nonlinear.harmony() function.

Reference:
    Korsunsky I, et al. Fast, sensitive and accurate integration of
    single-cell data with Harmony. Nature Methods (2019).

Note
----
This module re-exports the harmony function from scptensor.integration.nonlinear.
The implementation requires the harmonypy package to be installed.

Installation:
    pip install harmonypy

Usage:
    >>> from scptensor.integration import harmony
    >>> container = harmony(container, batch_key='batch', base_layer='pca')
"""

from typing import Optional
from scptensor.core.structures import ScpContainer

# Import the actual implementation
from scptensor.integration.nonlinear import harmony as _harmony_impl


def harmony(
    container: ScpContainer,
    batch_key: str,
    assay_name: str = 'protein',
    base_layer: str = 'pca',
    new_layer_name: Optional[str] = 'harmony',
    **kwargs
) -> ScpContainer:
    """
    Harmony integration for batch effect correction.

    This function is a wrapper around the harmony implementation in
    scptensor.integration.nonlinear. It uses the harmonypy package
    to perform iterative clustering and correction of batch effects.

    Parameters
    ----------
    container : ScpContainer
        Input container with multiple batches
    batch_key : str
        Column name in obs containing batch labels
    assay_name : str, optional
        Assay to use for integration (default: 'protein')
    base_layer : str, optional
        Layer to use as input (default: 'pca')
        Harmony typically works best on PCA-reduced data
    new_layer_name : str, optional
        Name for the corrected layer (default: 'harmony')
    **kwargs
        Additional parameters passed to harmonypy.run_harmony()
        Common parameters include:
        - theta: Clustering penalty parameter (default: 2)
        - lamb: Ridge regularization penalty (default: 1)
        - sigma: Bandwidth parameter (default: 0.1)
        - nclust: Number of clusters (default: None, auto-detect)
        - max_iter_harmony: Maximum iterations (default: 10)

    Returns
    -------
    ScpContainer
        Container with batch-corrected data

    Raises
    ------
    ValueError
        If harmonypy is not installed or if assay/layer not found

    Notes
    -----
    Harmony algorithm:
        1. Assigns cells to clusters using current cell embeddings
        2. Computes cluster-specific batch effects
        3. Corrects cell embeddings to maximize diversity
        4. Iterates until convergence

    Harmony is particularly effective for:
        - Large datasets with many batches
        - Complex batch structures
        - When biological variation should be preserved

    Examples
    --------
    >>> from scptensor.integration import harmony
    >>> # Basic usage
    >>> container = harmony(container, batch_key='batch')
    >>> # With custom parameters
    >>> container = harmony(container, batch_key='batch',
    ...                      theta=3, nclust=15)
    """
    # Delegate to the actual implementation
    return _harmony_impl(
        container=container,
        batch_key=batch_key,
        assay_name=assay_name,
        base_layer=base_layer,
        new_layer_name=new_layer_name,
        **kwargs
    )


if __name__ == "__main__":
    # Test: Basic functionality
    print("Testing Harmony integration wrapper...")

    # Create simple test data
    import numpy as np
    np.random.seed(42)
    n_samples_per_batch = 50
    n_features = 30

    # Generate PCA-like data
    X_batch1 = np.random.randn(n_samples_per_batch, n_features) * 0.5
    X_batch2 = np.random.randn(n_samples_per_batch, n_features) * 0.5 + 2.0

    X = np.vstack([X_batch1, X_batch2])

    # Create container
    import polars as pl
    from scptensor.core.structures import ScpContainer, Assay, ScpMatrix

    var = pl.DataFrame({"_index": [f"pc_{i}" for i in range(n_features)]})
    obs = pl.DataFrame({
        "_index": [f"cell_{i}" for i in range(2 * n_samples_per_batch)],
        "batch": ["batch1"] * n_samples_per_batch + ["batch2"] * n_samples_per_batch
    })

    assay = Assay(var=var)
    assay.add_layer("pca", ScpMatrix(X=X, M=None))

    container = ScpContainer(obs=obs, assays={"protein": assay})

    try:
        # Test Harmony wrapper (requires harmonypy)
        result = harmony(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="pca",
            theta=2.0,
            max_iter_harmony=5
        )

        # Check results
        assert "harmony" in result.assays["protein"].layers
        X_corrected = result.assays["protein"].layers["harmony"].X

        print(f"  Shape: {X_corrected.shape}")
        print("✅ All tests passed")
    except ImportError:
        print("  ⚠️  harmonypy not installed, skipping test")
        print("  Install with: pip install harmonypy")
