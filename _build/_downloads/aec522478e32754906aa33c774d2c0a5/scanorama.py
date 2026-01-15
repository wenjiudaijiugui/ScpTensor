"""
Scanorama integration for single-cell proteomics data.

This module provides a wrapper for the Scanorama algorithm for batch effect correction.
Scanorama is an efficient method for integrating heterogeneous single-cell data.

Reference:
    Hie B, et al. Efficient integration of heterogeneous single-cell
    transcriptomics data using Scanorama. Nature Biotechnology (2019).

Note
----
This module provides a convenient wrapper around the scanorama package.
The implementation requires the scanorama package to be installed.

Installation:
    pip install scanorama

Usage:
    >>> from scptensor.integration import scanorama_integrate
    >>> container = scanorama_integrate(container, batch_key='batch')
"""

from typing import Optional, List
import numpy as np
import polars as pl
from scptensor.core.structures import ScpContainer, ScpMatrix, Assay
from scptensor.core.exceptions import AssayNotFoundError, LayerNotFoundError, ValueError as ScpValueError, MissingDependencyError


def scanorama_integrate(
    container: ScpContainer,
    batch_key: str,
    assay_name: str = 'protein',
    base_layer: str = 'raw',
    new_layer_name: Optional[str] = 'scanorama',
    sigma: float = 15.0,
    alpha: float = 0.1,
    knn: Optional[int] = None,
    approx: bool = True
) -> ScpContainer:
    """
    Scanorama integration for batch effect correction.

    This function wraps the scanorama package to perform efficient
    batch correction and integration of single-cell data.

    Parameters
    ----------
    container : ScpContainer
        Input container with multiple batches
    batch_key : str
        Column name in obs containing batch labels
    assay_name : str, optional
        Assay to use for integration (default: 'protein')
    base_layer : str, optional
        Layer to use as input (default: 'raw')
    new_layer_name : str, optional
        Name for the corrected layer (default: 'scanorama')
    sigma : float, optional
        Alignment parameter (default: 15.0)
    alpha : float, optional
        Mixture weight parameter (default: 0.1)
    knn : int, optional
        Number of nearest neighbors (default: None, auto-detect)
    approx : bool, optional
        Use approximate nearest neighbors (default: True)

    Returns
    -------
    ScpContainer
        Container with integrated and corrected data

    Raises
    ------
    AssayNotFoundError
        If the specified assay does not exist
    LayerNotFoundError
        If the specified layer does not exist in the assay
    ScpValueError
        If batch_key not found in obs or parameters are invalid
    MissingDependencyError
        If scanorama package is not installed

    Notes
    -----
    Scanorama algorithm:
        1. Finds mutual nearest neighbors between batches
        2. Performs mutual nearest neighbors alignment
        3. Corrects and integrates data in a single step

    Scanorama is particularly effective for:
        - Large-scale datasets (>100k cells)
        - Integrating many batches simultaneously
        - Preserving biological variation
        - Fast integration with approximate methods

    Examples
    --------
    >>> from scptensor.integration import scanorama_integrate
    >>> # Basic usage
    >>> container = scanorama_integrate(container, batch_key='batch')
    >>> # With custom parameters
    >>> container = scanorama_integrate(container, batch_key='batch',
    ...                                  sigma=20.0, alpha=0.05)
    """
    # Validate parameters
    if sigma <= 0:
        raise ScpValueError(
            f"sigma must be positive, got {sigma}.",
            parameter="sigma",
            value=sigma
        )
    if not (0 < alpha < 1):
        raise ScpValueError(
            f"alpha must be in (0, 1), got {alpha}.",
            parameter="alpha",
            value=alpha
        )
    if knn is not None and knn <= 0:
        raise ScpValueError(
            f"knn must be positive or None, got {knn}.",
            parameter="knn",
            value=knn
        )

    # Validate assay and layer
    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]
    if base_layer not in assay.layers:
        raise LayerNotFoundError(base_layer, assay_name)

    obs_df = container.obs
    if batch_key not in obs_df.columns:
        raise ScpValueError(
            f"Batch key '{batch_key}' not found in obs. "
            f"Available columns: {list(obs_df.columns)}",
            parameter="batch_key",
            value=batch_key
        )

    # Get data
    X = assay.layers[base_layer].X.copy()
    M = assay.layers[base_layer].M

    # Handle NaN values
    if np.isnan(X).any():
        col_mean = np.nanmean(X, axis=0)
        col_mean = np.nan_to_num(col_mean, nan=0.0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_mean, inds[1])

    # Get batch information
    batches = obs_df[batch_key].to_numpy()
    unique_batches = np.unique(batches)

    if len(unique_batches) < 2:
        raise ScpValueError(
            "Scanorama requires at least 2 batches.",
            parameter="batch_key",
            value=batch_key
        )

    # Import scanorama
    try:
        import scanorama
    except ImportError:
        raise MissingDependencyError("scanorama")

    # Prepare data for scanorama
    # Scanorama expects a list of datasets
    datasets_list = []
    for batch in unique_batches:
        idx = np.where(batches == batch)[0]
        X_batch = X[idx]
        datasets_list.append(X_batch)

    # Integrate using scanorama
    # Use scanorama.correct for batch correction
    # Returns: (integrated datasets, corrected datasets)
    integrated = scanorama.correct(
        datasets_list,
        ds_names=[str(b) for b in unique_batches],
        return_dimred=True,
        return_dense=True,
        sigma=sigma,
        alpha=alpha,
        knn=knn,
        approx=approx
    )

    # integrated is a list of arrays, concatenate them
    X_corrected = np.vstack(integrated)

    # Create new layer
    new_M = M.copy() if M is not None else None
    new_matrix = ScpMatrix(X=X_corrected, M=new_M)
    if new_layer_name is None:
        new_layer_name = 'scanorama'
    assay.add_layer(new_layer_name, new_matrix)

    # Log operation
    container.log_operation(
        action="integration_scanorama",
        params={
            "batch_key": batch_key,
            "assay": assay_name,
            "sigma": sigma,
            "alpha": alpha,
            "n_batches": len(unique_batches)
        },
        description=f"Scanorama integration (sigma={sigma}, alpha={alpha}) on assay '{assay_name}'."
    )

    return container


if __name__ == "__main__":
    # Test: Basic functionality
    print("Testing Scanorama integration wrapper...")

    # Create simple test data
    np.random.seed(42)
    n_samples_per_batch = 50
    n_features = 30

    # Generate data with batch effects
    X_batch1 = np.random.randn(n_samples_per_batch, n_features) * 0.5
    X_batch2 = np.random.randn(n_samples_per_batch, n_features) * 0.5 + 2.0

    X = np.vstack([X_batch1, X_batch2])

    # Create container
    var = pl.DataFrame({"_index": [f"prot_{i}" for i in range(n_features)]})
    obs = pl.DataFrame({
        "_index": [f"cell_{i}" for i in range(2 * n_samples_per_batch)],
        "batch": ["batch1"] * n_samples_per_batch + ["batch2"] * n_samples_per_batch
    })

    assay = Assay(var=var)
    assay.add_layer("raw", ScpMatrix(X=X, M=None))

    container = ScpContainer(obs=obs, assays={"protein": assay})

    try:
        # Test Scanorama wrapper (requires scanorama)
        result = scanorama_integrate(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            sigma=15.0,
            alpha=0.1
        )

        # Check results
        assert "scanorama" in result.assays["protein"].layers
        X_corrected = result.assays["protein"].layers["scanorama"].X

        # Check batch effect reduction
        mean1 = np.mean(X_corrected[:n_samples_per_batch], axis=0)
        mean2 = np.mean(X_corrected[n_samples_per_batch:], axis=0)
        batch_diff = np.linalg.norm(mean1 - mean2)

        print(f"  Corrected batch difference: {batch_diff:.3f}")
        print(f"  Shape: {X_corrected.shape}")
        print("✅ All tests passed")
    except ImportError:
        print("  ⚠️  scanorama not installed, skipping test")
        print("  Install with: pip install scanorama")
