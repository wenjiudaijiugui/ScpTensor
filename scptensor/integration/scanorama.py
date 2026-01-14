"""Scanorama integration wrapper for single-cell proteomics data.

Reference
---------
Hie B, et al. Efficient integration of heterogeneous single-cell
transcriptomics data using Scanorama. Nature Biotechnology (2019).
"""

import numpy as np
import polars as pl
import scipy.sparse as sp

from scptensor.core.exceptions import AssayNotFoundError, LayerNotFoundError, MissingDependencyError
from scptensor.core.exceptions import ScpValueError
from scptensor.core.sparse_utils import is_sparse_matrix
from scptensor.core.structures import Assay, ScpContainer, ScpMatrix
from scptensor.core.utils import requires_dependency


@requires_dependency("scanorama", "pip install scanorama")
def scanorama_integrate(
    container: ScpContainer,
    batch_key: str,
    assay_name: str = "protein",
    base_layer: str = "raw",
    new_layer_name: str | None = "scanorama",
    sigma: float = 15.0,
    alpha: float = 0.1,
    knn: int | None = None,
    approx: bool = True,
    return_dimred: bool = False,
    dimred: int | None = None,
) -> ScpContainer:
    """Scanorama integration for batch effect correction.

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
        Higher values = more aggressive alignment
    alpha : float, optional
        Mixture weight parameter (default: 0.1)
        Controls balance between mutual nearest neighbors and similarity
    knn : int, optional
        Number of nearest neighbors (default: None, auto-detect)
    approx : bool, optional
        Use approximate nearest neighbors (default: True)
        Much faster but slightly less accurate
    return_dimred : bool, optional
        Return dimensionality-reduced data (default: False)
    dimred : int, optional
        Number of dimensions for reduction (default: 100)

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

    Examples
    --------
    >>> from scptensor.integration import scanorama_integrate
    >>> container = scanorama_integrate(container, batch_key='batch')
    >>> container = scanorama_integrate(container, batch_key='batch', sigma=20.0)
    """
    import scanorama

    # Validate parameters
    if sigma <= 0:
        raise ScpValueError(f"sigma must be positive, got {sigma}.", parameter="sigma", value=sigma)
    if not (0 < alpha < 1):
        raise ScpValueError(f"alpha must be in (0, 1), got {alpha}.", parameter="alpha", value=alpha)
    if knn is not None and knn <= 0:
        raise ScpValueError(f"knn must be positive or None, got {knn}.", parameter="knn", value=knn)

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
            value=batch_key,
        )

    # Get and prepare data
    X = assay.layers[base_layer].X.copy()
    M = assay.layers[base_layer].M
    input_was_sparse = is_sparse_matrix(X)

    # Convert to dense and impute NaNs
    X = _prepare_scanorama_data(X)

    # Get batch information
    batches = obs_df[batch_key].to_numpy()
    unique_batches = np.unique(batches)

    if len(unique_batches) < 2:
        raise ScpValueError(
            "Scanorama requires at least 2 batches.",
            parameter="batch_key",
            value=batch_key,
        )

    # Prepare data list for scanorama
    datasets_list = [X[batches == b] for b in unique_batches]

    # Set default knn based on data size
    knn = knn or max(5, min(20, X.shape[0] // len(unique_batches) - 1))

    # Integrate using scanorama
    integrated = scanorama.correct(
        datasets_list,
        ds_names=[str(b) for b in unique_batches],
        return_dimred=return_dimred,
        return_dense=True,
        sigma=sigma,
        alpha=alpha,
        knn=knn,
        approx=approx,
        dimred=dimred,
    )

    X_corrected = np.vstack(integrated)

    # Preserve sparsity if appropriate
    if input_was_sparse and not return_dimred:
        sparsity_ratio = 1.0 - (np.count_nonzero(X_corrected) / X_corrected.size)
        if sparsity_ratio > 0.5:
            X_corrected = sp.csr_matrix(X_corrected)

    # Create new layer
    new_matrix = ScpMatrix(
        X=X_corrected,
        M=M.copy() if M is not None else None,
    )
    assay.add_layer(new_layer_name or "scanorama", new_matrix)

    # Log operation
    container.log_operation(
        action="integration_scanorama",
        params={
            "batch_key": batch_key,
            "assay": assay_name,
            "sigma": sigma,
            "alpha": alpha,
            "knn": knn,
            "approx": approx,
            "return_dimred": return_dimred,
            "n_batches": len(unique_batches),
        },
        description=f"Scanorama integration (sigma={sigma}, alpha={alpha}) on assay '{assay_name}'.",
    )

    return container


def _prepare_scanorama_data(X: np.ndarray | sp.spmatrix) -> np.ndarray:
    """Convert sparse to dense and impute NaN values."""
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X)

    if np.isnan(X).any():
        col_mean = np.nan_to_num(np.nanmean(X, axis=0), nan=0.0)
        nan_idx = np.where(np.isnan(X))
        X[nan_idx] = np.take(col_mean, nan_idx[1])

    return X


if __name__ == "__main__":
    # Test: Basic functionality
    print("Testing Scanorama integration wrapper...")

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
        "batch": ["batch1"] * n_samples_per_batch + ["batch2"] * n_samples_per_batch,
    })

    assay = Assay(var=var)
    assay.add_layer("raw", ScpMatrix(X=X, M=None))
    container = ScpContainer(obs=obs, assays={"protein": assay})

    try:
        result = scanorama_integrate(
            container,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            sigma=15.0,
            alpha=0.1,
        )

        assert "scanorama" in result.assays["protein"].layers
        X_corrected = result.assays["protein"].layers["scanorama"].X

        mean1 = np.mean(X_corrected[:n_samples_per_batch], axis=0)
        mean2 = np.mean(X_corrected[n_samples_per_batch:], axis=0)
        batch_diff = np.linalg.norm(mean1 - mean2)
        original_diff = np.linalg.norm(np.mean(X_batch1, axis=0) - np.mean(X_batch2, axis=0))

        print(f"  Original batch difference: {original_diff:.3f}")
        print(f"  Corrected batch difference: {batch_diff:.3f}")
        print(f"  Reduction ratio: {batch_diff / original_diff:.3f}")
        print(f"  Shape: {X_corrected.shape}")

        # Test with sparse input
        print("  Testing with sparse input...")
        X_sparse = sp.csr_matrix(X)
        assay2 = Assay(var=var)
        assay2.add_layer("raw", ScpMatrix(X=X_sparse, M=None))
        container2 = ScpContainer(obs=obs, assays={"protein": assay2})

        result2 = scanorama_integrate(
            container2,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            sigma=15.0,
            alpha=0.1,
        )

        assert "scanorama" in result2.assays["protein"].layers
        print("  Sparse input test passed")

        # Test with dimensionality reduction
        print("  Testing with dimensionality reduction...")
        assay3 = Assay(var=var)
        assay3.add_layer("raw", ScpMatrix(X=X.copy(), M=None))
        container3 = ScpContainer(obs=obs, assays={"protein": assay3})

        result3 = scanorama_integrate(
            container3,
            batch_key="batch",
            assay_name="protein",
            base_layer="raw",
            sigma=15.0,
            alpha=0.1,
            return_dimred=True,
            dimred=15,
        )

        assert "scanorama" in result3.assays["protein"].layers
        X_dimred = result3.assays["protein"].layers["scanorama"].X
        print(f"  Dimension-reduced shape: {X_dimred.shape}")
        print("  All tests passed")
    except ImportError:
        print("  scanorama not installed, skipping test")
        print("  Install with: pip install scanorama")
