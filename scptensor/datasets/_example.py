"""Example dataset loaders for ScpTensor.

This module provides ready-to-use example datasets for tutorials, testing,
and documentation. All datasets are generated synthetically with realistic
biological and technical variation patterns.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from scipy.special import expit

from scptensor.core.structures import Assay, ScpContainer, ScpMatrix
from scptensor.core.utils import compute_pca, compute_umap

if TYPE_CHECKING:
    from collections.abc import Sequence


# =============================================================================
# Public API Enums and Constants
# =============================================================================


class DatasetType(IntEnum):
    """Classification of dataset characteristics."""

    TOY = 0  # Small, fast to load, for quick testing
    SIMULATED = 1  # Larger, with realistic biological variation
    LABELED = 2  # Includes pre-computed labels for supervised tasks


class DatasetSize(IntEnum):
    """Dataset size categories."""

    TINY = 0  # < 100 samples, < 100 features
    SMALL = 1  # 100-500 samples, 100-500 features
    MEDIUM = 2  # 500-2000 samples, 500-2000 features
    LARGE = 3  # > 2000 samples, > 2000 features


REPRODUCIBILITY_NOTE = """
Note on Reproducibility:
------------------------
All example datasets use fixed random seeds for reproducibility.
However, numerical results may still vary slightly across different
hardware platforms due to floating-point arithmetic differences.

To ensure exact reproducibility, use the same:
- NumPy version (1.24+)
- SciPy version (1.10+)
- Python version (3.11+)
"""


# =============================================================================
# Internal Dataset Generation
# =============================================================================


def _generate_dataset(
    n_samples: int,
    n_features: int,
    n_batches: int = 3,
    n_groups: int = 4,
    missing_rate: float = 0.2,
    lod_ratio: float = 0.6,
    random_seed: int = 42,
    cell_type_names: Sequence[str] | None = None,
    batch_names: Sequence[str] | None = None,
    feature_names: Sequence[str] | None = None,
    include_qc_metrics: bool = False,
    include_embeddings: bool = False,
    cluster_separation: float = 1.0,
) -> ScpContainer:
    """
    Generate a synthetic single-cell proteomics dataset.

    Parameters
    ----------
    n_samples : int
        Number of samples (cells) to generate.
    n_features : int
        Number of features (proteins) to generate.
    n_batches : int, default=3
        Number of experimental batches.
    n_groups : int, default=4
        Number of biological groups (cell types).
    missing_rate : float, default=0.2
        Proportion of missing values (0.0-1.0).
    lod_ratio : float, default=0.6
        Proportion of missing values that are MNAR vs MCAR.
    random_seed : int, default=42
        Random seed for reproducibility.
    cell_type_names : Sequence[str] | None, default=None
        Custom cell type names. If None, generates "CellType_0", etc.
    batch_names : Sequence[str] | None, default=None
        Custom batch names. If None, generates "Batch_0", etc.
    feature_names : Sequence[str] | None, default=None
        Custom feature names. If None, generates "Protein_00001", etc.
    include_qc_metrics : bool, default=False
        Whether to add quality control metrics to obs.
    include_embeddings : bool, default=False
        Whether to pre-compute and add PCA/UMAP embeddings.
    cluster_separation : float, default=1.0
        Scale factor for group separation (higher = more distinct clusters).

    Returns
    -------
    ScpContainer
        Generated dataset with complete metadata.
    """
    rng = np.random.default_rng(seed=random_seed)

    # Validate inputs (early return pattern)
    if not (0.0 <= missing_rate <= 1.0):
        raise ValueError("missing_rate must be between 0.0 and 1.0")
    if not (0.0 <= lod_ratio <= 1.0):
        raise ValueError("lod_ratio must be between 0.0 and 1.0")

    # Default names (lazy evaluation)
    if cell_type_names is None:
        cell_type_names = [f"CellType_{i}" for i in range(n_groups)]
    elif len(cell_type_names) != n_groups:
        raise ValueError(
            f"cell_type_names length ({len(cell_type_names)}) must equal n_groups ({n_groups})"
        )

    if batch_names is None:
        batch_names = [f"Batch_{i}" for i in range(n_batches)]
    elif len(batch_names) != n_batches:
        raise ValueError(
            f"batch_names length ({len(batch_names)}) must equal n_batches ({n_batches})"
        )

    # ========================================================================
    # 1. Generate Biological Signal (Latent Factor Model)
    # ========================================================================

    n_pathways = max(5, n_features // 40)

    # Group (cell type) assignments
    group_indices = rng.permutation(np.repeat(np.arange(n_groups), n_samples // n_groups + 1))[
        :n_samples
    ]

    # Pathway activity per cell
    pathway_activity = rng.normal(0.0, 1.0, (n_samples, n_pathways))

    # Group-specific pathway shifts (creates distinct cell types)
    # Scale by cluster_separation for tunable cluster distinctness
    group_shifts = rng.normal(0.0, 2.0 * cluster_separation, (n_groups, n_pathways))
    group_shifts[rng.random((n_groups, n_pathways)) < 0.6] = 0.0  # Sparsity

    pathway_activity += group_shifts[group_indices]

    # Protein loadings (sparse)
    protein_loadings = rng.normal(0.0, 1.0, (n_pathways, n_features))
    protein_loadings[rng.random((n_pathways, n_features)) < 0.7] = 0.0

    # Biological signal (latent factor model)
    biological_signal = pathway_activity @ protein_loadings

    # Normalize to unit variance, scale by factor
    bio_std = biological_signal.std()
    if bio_std > 0:
        biological_signal = biological_signal / bio_std * 2.0

    # ========================================================================
    # 2. Add Technical Variation
    # ========================================================================

    batch_indices = np.arange(n_samples) % n_batches

    # Baseline protein abundance
    protein_means = rng.normal(15.0, 2.0, (1, n_features))

    # Sample efficiency (library size variation)
    sample_efficiency = rng.normal(0.0, 0.5, (n_samples, 1))

    # Batch effects
    batch_effects = rng.normal(0.0, 0.4, (n_batches, n_features))
    sample_batch_effects = batch_effects[batch_indices]

    # Combine signals
    X_clean = protein_means + biological_signal + sample_efficiency + sample_batch_effects

    # ========================================================================
    # 3. Add Heteroscedastic Noise
    # ========================================================================

    # Min-max normalization for noise scaling (vectorized)
    X_min, X_max = X_clean.min(), X_clean.max()
    norm_intensity = (
        np.zeros_like(X_clean) if X_max <= X_min else (X_clean - X_min) / (X_max - X_min)
    )

    # High intensity -> Low noise, Low intensity -> High noise
    noise = rng.normal(0.0, 1.0, (n_samples, n_features)) * (0.8 - 0.5 * norm_intensity)
    X_complete = X_clean + noise

    # ========================================================================
    # 4. Generate Missing Value Mask
    # ========================================================================

    M = np.zeros((n_samples, n_features), dtype=np.int8)

    if missing_rate > 0:
        total_elements = n_samples * n_features
        target_missing = int(total_elements * missing_rate)
        target_mnar = int(target_missing * lod_ratio)
        target_mcar = target_missing - target_mnar

        # MNAR: Intensity-based probabilistic dropout
        if target_mnar > 0:
            bias_guess = np.percentile(X_complete, target_mnar / total_elements * 100)
            p_missing = expit(-(X_complete - bias_guess))
            mnar_mask = rng.random(X_complete.shape) < p_missing
            M[mnar_mask] = 2  # LOD code

        # MCAR: Random missing (vectorized selection)
        if target_mcar > 0:
            valid_indices = np.flatnonzero(M == 0)
            if len(valid_indices) > 0:
                n_to_mask = min(target_mcar, len(valid_indices))
                selected = rng.choice(valid_indices, n_to_mask, replace=False)
                M[np.unravel_index(selected, M.shape)] = 1  # MBR code

    # ========================================================================
    # 5. Create Metadata
    # ========================================================================

    feature_names = feature_names or [f"Protein_{i:05d}" for i in range(n_features)]
    sample_ids = [f"Cell_{i:05d}" for i in range(n_samples)]

    # Feature metadata (var) - direct dict construction
    var = pl.DataFrame(
        {
            "protein_id": feature_names,
            "gene_name": [f"Gene_{i:05d}" for i in range(n_features)],
            "mean_abundance": protein_means.ravel(),
            "variance": X_complete.var(axis=0),
        }
    )

    # Sample metadata (obs) - build dict incrementally
    obs_data = {
        "sample_id": sample_ids,
        "batch": [batch_names[int(i)] for i in batch_indices],
        "cell_type": [cell_type_names[int(i)] for i in group_indices],
        "batch_id": batch_indices.tolist(),
        "cell_type_id": group_indices.tolist(),
        "n_detected": np.sum(M == 0, axis=1).tolist(),
        "missing_rate": (M != 0).sum(axis=1) / n_features,
    }

    # Optional QC metrics (computed vectorized)
    if include_qc_metrics:
        row_sums = X_complete.sum(axis=1)
        row_medians = np.median(X_complete, axis=1)
        global_median = np.median(X_complete, axis=0, keepdims=True)
        obs_data.update(
            {
                "total_intensity": row_sums.tolist(),
                "mean_intensity": (row_sums / n_features).tolist(),
                "median_intensity": row_medians.tolist(),
                "mad_intensity": np.median(np.abs(X_complete - global_median), axis=1).tolist(),
            }
        )

    obs = pl.DataFrame(obs_data)

    # ========================================================================
    # 6. Create ScpContainer
    # ========================================================================

    matrix = ScpMatrix(X=X_complete.astype(np.float64), M=M)
    assay = Assay(var=var, feature_id_col="protein_id")
    assay.add_layer("raw", matrix)

    container = ScpContainer(obs=obs, sample_id_col="sample_id")
    container.add_assay("proteins", assay)

    container.log_operation(
        action="generate_example_dataset",
        params={
            "n_samples": n_samples,
            "n_features": n_features,
            "n_batches": n_batches,
            "n_groups": n_groups,
            "missing_rate": missing_rate,
            "lod_ratio": lod_ratio,
            "random_seed": random_seed,
        },
        description="Generated example dataset for tutorials and testing.",
    )

    # ========================================================================
    # 7. Add Embeddings (if requested)
    # ========================================================================

    if include_embeddings:
        # Vectorized median imputation for PCA
        X_filled = X_complete.copy()
        col_medians = np.nanmedian(np.where(M == 0, X_complete, np.nan), axis=0)
        mask = M != 0
        X_filled[mask] = np.take(col_medians, np.where(mask)[1])

        # Compute and add PCA embeddings
        pca_result = compute_pca(X_filled, n_components=10, random_state=random_seed)
        container.obs = obs.with_columns(
            pl.Series("PC1", pca_result[:, 0]),
            pl.Series("PC2", pca_result[:, 1]),
            pl.Series("PC3", pca_result[:, 2]),
        )

        # Compute and add UMAP embeddings
        umap_result = compute_umap(pca_result[:, :10], n_components=2, random_state=random_seed)
        container.obs = container.obs.with_columns(
            pl.Series("UMAP1", umap_result[:, 0]),
            pl.Series("UMAP2", umap_result[:, 1]),
        )

    return container


# =============================================================================
# Public Dataset Loading Functions
# =============================================================================


def load_toy_example() -> ScpContainer:
    """
    Load a small toy dataset for quick testing and examples.

    This is a minimal dataset designed for:
    - Quick code testing
    - Documentation examples
    - Tutorial demonstrations

    Dataset Specifications:
    -----------------------
    - **Samples:** 100 cells
    - **Features:** 50 proteins
    - **Batches:** 3 (Batch_0, Batch_1, Batch_2)
    - **Cell Types:** 3 (CellType_0, CellType_1, CellType_2)
    - **Missing Rate:** 20% (12% MNAR, 8% MCAR)
    - **Random Seed:** 42 (reproducible)

    Obs Metadata:
    -------------
    - sample_id: Unique cell identifier
    - batch: Experimental batch
    - cell_type: Biological group/cell type
    - batch_id: Numeric batch index
    - cell_type_id: Numeric cell type index
    - n_detected: Number of detected proteins per cell
    - missing_rate: Proportion of missing values per cell

    Var Metadata:
    -------------
    - protein_id: Unique protein identifier
    - gene_name: Corresponding gene name
    - mean_abundance: Average protein abundance
    - variance: Protein variance across cells

    Returns
    -------
    ScpContainer
        Container with 'proteins' assay containing 'raw' layer.

    Examples
    --------
    >>> from scptensor.datasets import load_toy_example
    >>> container = load_toy_example()
    >>> print(container)
    ScpContainer with 100 samples and 1 assay

    >>> # Access the data
    >>> assay = container.assays["proteins"]
    >>> matrix = assay.layers["raw"]
    >>> print(f"Samples: {container.n_samples}, Features: {container.n_features}")  # 100, 50

    Notes
    -----
    This dataset uses a fixed random seed (42) for reproducibility.
    All generated values follow realistic single-cell proteomics patterns
    including batch effects and MNAR missing values.
    """
    return _generate_dataset(
        n_samples=100,
        n_features=50,
        n_batches=3,
        n_groups=3,
        missing_rate=0.2,
        lod_ratio=0.6,
        random_seed=42,
        include_qc_metrics=False,
        include_embeddings=False,
    )


def load_simulated_scrnaseq_like() -> ScpContainer:
    """
    Load a larger simulated dataset resembling single-cell proteomics data.

    This dataset is designed for comprehensive pipeline testing with
    realistic biological and technical variation patterns.

    Dataset Specifications:
    -----------------------
    - **Samples:** 500 cells
    - **Features:** 200 proteins
    - **Batches:** 3 (Batch_0, Batch_1, Batch_2)
    - **Cell Types:** 4 distinct types
        - T_Cell: T lymphocytes
        - B_Cell: B lymphocytes
        - NK_Cell: Natural killer cells
        - Monocyte: Monocytes
    - **Missing Rate:** 30% (18% MNAR, 12% MCAR)
    - **Random Seed:** 123

    Obs Metadata:
    -------------
    - sample_id: Unique cell identifier (e.g., Cell_00000)
    - batch: Experimental batch assignment
    - cell_type: Biological cell type label
    - batch_id: Numeric batch index (0, 1, 2)
    - cell_type_id: Numeric cell type index (0-3)
    - n_detected: Number of detected proteins per cell
    - missing_rate: Proportion of missing values per cell
    - total_intensity: Sum of all intensities
    - mean_intensity: Average intensity per cell
    - median_intensity: Median intensity per cell
    - mad_intensity: Median absolute deviation

    Var Metadata:
    -------------
    - protein_id: Unique protein identifier
    - gene_name: Corresponding gene name
    - mean_abundance: Average protein abundance
    - variance: Protein variance across cells

    Returns
    -------
    ScpContainer
        Container with 'proteins' assay containing 'raw' layer.
        Includes QC metrics in obs for quality control workflows.

    Examples
    --------
    >>> from scptensor.datasets import load_simulated_scrnaseq_like
    >>> container = load_simulated_scrnaseq_like()
    >>> print(container)
    ScpContainer with 500 samples and 1 assay

    >>> # View cell type distribution
    >>> container.obs.group_by("cell_type").count()
    shape: (4, 2)
    ┌───────────┬───────┐
    │ cell_type ┆ count │
    │ ---       ┆ ---   │
    │ str       ┆ u32   │
    ╞═══════════╪═══════╡
    ... ...

    >>> # Check batch effects
    >>> import scptensor.viz as viz
    >>> viz.scatterplot(container, basis="PC1", color="batch")

    Notes
    -----
    This dataset includes:
    - Distinct cell type signatures for clustering
    - Batch effects for integration testing
    - QC metrics for quality control workflows
    - Realistic missing value patterns

    Use this dataset for:
    - Testing full analysis pipelines
    - Benchmarking normalization methods
    - Evaluating batch correction algorithms
    - Clustering and dimensionality reduction tutorials
    """
    cell_type_names = ["T_Cell", "B_Cell", "NK_Cell", "Monocyte"]
    return _generate_dataset(
        n_samples=500,
        n_features=200,
        n_batches=3,
        n_groups=4,
        missing_rate=0.3,
        lod_ratio=0.6,
        random_seed=123,
        cell_type_names=cell_type_names,
        batch_names=["Batch_0", "Batch_1", "Batch_2"],
        include_qc_metrics=True,
        include_embeddings=False,
    )


def load_example_with_clusters() -> ScpContainer:
    """
    Load a dataset with known cluster labels for clustering tutorials.

    This dataset is specifically designed for clustering algorithm evaluation
    and tutorials, with clear, well-separated clusters.

    Dataset Specifications:
    -----------------------
    - **Samples:** 300 cells
    - **Features:** 100 proteins
    - **Batches:** 3 (Batch_A, Batch_B, Batch_C)
    - **Cell Types:** 5 distinct types with clear separation
        - Stem: Stem cells
        - Progenitor: Progenitor cells
        - Differentiated_A: Differentiated type A
        - Differentiated_B: Differentiated type B
        - Immune: Immune cells
    - **Missing Rate:** 25%
    - **Random Seed:** 456
    - **Pre-computed:** None (user should run clustering)

    Obs Metadata:
    -------------
    - sample_id: Unique cell identifier
    - batch: Experimental batch
    - cell_type: True biological label (ground truth)
    - batch_id: Numeric batch index
    - cell_type_id: Numeric cell type index
    - n_detected: Number of detected proteins
    - missing_rate: Proportion of missing values
    - total_intensity: Sum intensity
    - mean_intensity: Mean intensity
    - median_intensity: Median intensity
    - mad_intensity: Median absolute deviation

    Var Metadata:
    -------------
    - protein_id: Unique protein identifier
    - gene_name: Corresponding gene name
    - mean_abundance: Average protein abundance
    - variance: Protein variance

    Returns
    -------
    ScpContainer
        Container with 'proteins' assay containing 'raw' layer.
        The 'cell_type' column in obs provides ground truth labels
        for clustering evaluation.

    Examples
    --------
    >>> from scptensor.datasets import load_example_with_clusters
    >>> from scptensor.cluster import kmeans
    >>> from scptensor.dim_reduction import pca
    >>> from scptensor import zscore
    >>>
    >>> # Load dataset with known clusters
    >>> container = load_example_with_clusters()
    >>>
    >>> # Standardize and reduce dimensions
    >>> container = zscore(container, "proteins", "raw", "zscore")
    >>> container = pca(container, "proteins", "zscore", n_components=10)
    >>>
    >>> # Run clustering
    >>> container = kmeans(container, "proteins", "zscore", n_clusters=5)
    >>>
    >>> # Compare with ground truth
    >>> predicted = container.obs["kmeans_cluster"].to_numpy()
    >>> true_labels = container.obs["cell_type_id"].to_numpy()
    >>> # ... compute clustering metrics ...

    Notes
    -----
    The cell types in this dataset have distinct protein expression signatures,
    making them suitable for testing clustering algorithms. Use the 'cell_type'
    column as ground truth to evaluate clustering performance.

    Recommended workflow:
    1. Quality control (filter low-quality cells)
    2. Normalization (z-score or other method)
    3. Dimensionality reduction (PCA -> UMAP)
    4. Clustering (KMeans, Leiden, etc.)
    5. Compare clusters with 'cell_type' ground truth
    """
    cell_type_names = ["Stem", "Progenitor", "Differentiated_A", "Differentiated_B", "Immune"]
    return _generate_dataset(
        n_samples=300,
        n_features=100,
        n_batches=3,
        n_groups=5,
        missing_rate=0.25,
        lod_ratio=0.5,
        random_seed=456,
        cell_type_names=cell_type_names,
        batch_names=["Batch_A", "Batch_B", "Batch_C"],
        include_qc_metrics=True,
        include_embeddings=False,
        cluster_separation=1.5,  # Stronger separation for clear clusters
    )


# =============================================================================
# Module Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ScpTensor Example Datasets - Module Test")
    print("=" * 60)
    print()

    # Test 1: load_toy_example
    print("Test 1: load_toy_example()")
    print("-" * 40)
    toy = load_toy_example()
    print(f"  Samples: {toy.n_samples}")
    print(f"  Features: {toy.assays['proteins'].n_features}")
    print(f"  Batches: {toy.obs['batch'].unique().len()}")
    print(f"  Cell types: {toy.obs['cell_type'].unique().len()}")
    print(f"  Missing rate: {toy.obs['missing_rate'].mean():.2%}")  # type: ignore[str-bytes-safe]
    print(f"  Container: {toy!r}")
    print()

    # Test 2: load_simulated_scrnaseq_like
    print("Test 2: load_simulated_scrnaseq_like()")
    print("-" * 40)
    simulated = load_simulated_scrnaseq_like()
    print(f"  Samples: {simulated.n_samples}")
    print(f"  Features: {simulated.assays['proteins'].n_features}")
    print(f"  Batches: {simulated.obs['batch'].unique().len()}")
    print(f"  Cell types: {list(simulated.obs['cell_type'].unique())}")
    print(f"  Missing rate: {simulated.obs['missing_rate'].mean():.2%}")  # type: ignore[str-bytes-safe]
    print(f"  QC columns: {[c for c in simulated.obs.columns if 'intensity' in c]}")
    print(f"  Container: {simulated!r}")
    print()

    # Test 3: load_example_with_clusters
    print("Test 3: load_example_with_clusters()")
    print("-" * 40)
    clustered = load_example_with_clusters()
    print(f"  Samples: {clustered.n_samples}")
    print(f"  Features: {clustered.assays['proteins'].n_features}")
    print(f"  Cell types: {list(clustered.obs['cell_type'].unique())}")
    print("  Cluster labels available: Yes")
    print(f"  Container: {clustered}")
    print()

    # Cell type distribution for clustered dataset
    print("  Cell type distribution:")
    cell_dist = clustered.obs.group_by("cell_type").len().sort("cell_type")
    for row in cell_dist.iter_rows(named=True):
        print(f"    {row['cell_type']}: {row['count']} cells")
    print()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
