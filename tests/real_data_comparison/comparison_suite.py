"""
Comparison test suite for ScpTensor vs Scanpy.

This module provides a comprehensive testing framework to compare
ScpTensor and Scanpy outputs for numerical equivalence and performance.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp

# Optional scanpy import
try:
    import scanpy as sc
    from anndata import AnnData

    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False
    sc = None
    AnnData = None

from scptensor.core.io import from_scanpy, to_scanpy

# Import ScpTensor
# Import with aliases to avoid conflicts with scanpy's pca/umap
from scptensor.dim_reduction import pca as scptensor_pca
from scptensor.dim_reduction import umap as scptensor_umap

# Import HVG selection
from scptensor.feature_selection.hvg import select_hvg
from scptensor.normalization import log_normalize
from tests.real_data_comparison.metrics import (
    ComparisonResult,
    SimilarityMetrics,
    format_time,
    speedup_factor,
)


class ComparisonSuite:
    """Test suite for comparing ScpTensor and Scanpy outputs.

    This class manages the comparison workflow, including data loading,
    method comparison, and result aggregation.

    Parameters
    ----------
    h5ad_path : str | Path
        Path to h5ad file containing the test data
    assay_name : str
        Name of the assay to use (default: "proteins")

    Attributes
    ----------
    adata_scanpy : AnnData
        Original Scanpy data (reference)
    container_scptensor : ScpContainer
        ScpTensor copy of the same data
    results : List[ComparisonResult]
        Accumulated test results

    Examples
    --------
    >>> suite = ComparisonSuite("tests/data/PXD061065.h5ad")
    >>> result = suite.compare_normalization("log1p")
    >>> print(result)
    >>> suite.summary()
    """

    # Thresholds for UMAP comparison (more lenient due to stochastic nature)
    UMAP_THRESHOLD_COSINE = 0.95  # Lower threshold for UMAP
    UMAP_THRESHOLD_PEARSON = 0.95

    def __init__(
        self,
        h5ad_path: str | Path,
        assay_name: str = "proteins",
    ):
        if not SCANPY_AVAILABLE:
            raise ImportError("Scanpy is required. Install with: pip install scanpy")

        self.h5ad_path = Path(h5ad_path)
        self.assay_name = assay_name

        # Load data
        self.adata_scanpy = sc.read_h5ad(h5ad_path)
        self.container_scptensor = from_scanpy(self.adata_scanpy, assay_name=assay_name)

        # Store original data for each test
        self.results: list[ComparisonResult] = []

        # Verification that data loaded correctly
        assert self.adata_scanpy.shape == self.container_scptensor.shape, (
            f"Shape mismatch: Scanpy {self.adata_scanpy.shape} vs "
            f"ScpTensor {self.container_scptensor.shape}"
        )

    # ==================== Normalization Comparison ====================

    def compare_normalization(
        self,
        method: Literal["log1p", "log", "sqrt"] = "log1p",
        base: float = 2.0,
        offset: float = 1.0,
        target_sum: float | None = None,
    ) -> ComparisonResult:
        """Compare normalization methods.

        Parameters
        ----------
        method : str
            Normalization method: "log1p", "log", "sqrt"
        base : float
            Base for log transformation (used when method="log")
        offset : float
            Offset for log transformation (used when method="log")
        target_sum : float, optional
            Target sum for total count normalization (before log)

        Returns
        -------
        ComparisonResult
            Comparison metrics and timing
        """
        # Create fresh copies
        adata = self.adata_scanpy.copy()
        container = from_scanpy(self.adata_scanpy, assay_name=self.assay_name)

        # Get base layer name
        base_layer = "X"
        if base_layer not in container.assays[self.assay_name].layers:
            # Try to find a suitable layer
            for layer_name in ["counts", "raw", "data"]:
                if layer_name in container.assays[self.assay_name].layers:
                    base_layer = layer_name
                    break

        # Scanpy normalization
        t0 = time.perf_counter()
        if method == "log1p":
            if target_sum:
                sc.pp.normalize_total(adata, target_sum=target_sum)
            sc.pp.log1p(adata)
        elif method == "log":
            # Custom log transformation: log2(X + offset)
            X = adata.X if not sp.issparse(adata.X) else adata.X.toarray()
            adata.X = np.log2(X + offset)
        elif method == "sqrt":
            sc.pp.sqrt(adata)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        time_scanpy = time.perf_counter() - t0

        # ScpTensor normalization
        t0 = time.perf_counter()
        if method == "log1p":
            # log1p in Scanpy is log(1 + x), which is natural log
            # We need to match this with log_normalize using base=e and offset=1
            container = log_normalize(
                container,
                assay_name=self.assay_name,
                source_layer=base_layer,
                new_layer_name="log1p",
                base=np.e,  # Natural log
                offset=1.0,
            )
            scptensor_result = container.assays[self.assay_name].layers["log1p"].X
        elif method == "log":
            container = log_normalize(
                container,
                assay_name=self.assay_name,
                source_layer=base_layer,
                new_layer_name="log",
                base=base,
                offset=offset,
            )
            scptensor_result = container.assays[self.assay_name].layers["log"].X
        elif method == "sqrt":
            # Apply sqrt transformation
            X = container.assays[self.assay_name].layers[base_layer].X
            if sp.issparse(X):
                X_sqrt = X.copy()
                X_sqrt.data = np.sqrt(X_sqrt.data)
            else:
                X_sqrt = np.sqrt(X)
            # Create a new layer with sqrt result
            from scptensor.core.structures import ScpMatrix

            container.assays[self.assay_name].layers["sqrt"] = ScpMatrix(
                X=X_sqrt,
                M=container.assays[self.assay_name].layers[base_layer].M,
            )
            scptensor_result = X_sqrt
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        time_scptensor = time.perf_counter() - t0

        # Ensure both results are dense for comparison
        if sp.issparse(adata.X):
            scanpy_result = adata.X.toarray()
        else:
            scanpy_result = adata.X

        if sp.issparse(scptensor_result):
            scptensor_result = scptensor_result.toarray()

        # Compare results
        result = SimilarityMetrics.compare_results(
            scanpy_result=scanpy_result,
            scptensor_result=scptensor_result,
            module=f"normalization_{method}",
            time_scanpy=time_scanpy,
            time_scptensor=time_scptensor,
        )

        self.results.append(result)
        return result

    # ==================== PCA Comparison ====================

    def compare_pca(
        self,
        n_components: int = 50,
        method: Literal["truncated", "arpack", "full"] = "arpack",
        use_highly_variable: bool = False,
        svd_solver: str = "arpack",
    ) -> ComparisonResult:
        """Compare PCA dimensionality reduction.

        Parameters
        ----------
        n_components : int
            Number of principal components
        method : str
            SVD solver method (truncated, arpack, full)
        use_highly_variable : bool
            Whether to use highly variable features
        svd_solver : str
            Scanpy svd_solver parameter

        Returns
        -------
        ComparisonResult
            Comparison metrics and timing
        """
        # Create fresh copies
        adata = self.adata_scanpy.copy()
        container = from_scanpy(self.adata_scanpy, assay_name=self.assay_name)

        # Get base layer name
        base_layer = "X"
        for layer_name in ["log1p", "log", "X"]:
            if layer_name in container.assays[self.assay_name].layers:
                base_layer = layer_name
                break

        # Scanpy PCA
        t0 = time.perf_counter()
        sc.tl.pca(adata, n_comps=n_components, svd_solver=svd_solver)
        time_scanpy = time.perf_counter() - t0

        # ScpTensor PCA
        t0 = time.perf_counter()
        container = scptensor_pca(
            container,
            assay_name=self.assay_name,
            base_layer_name=base_layer,
            new_assay_name="pca",
            n_components=n_components,
            center=True,
            scale=False,
            random_state=42,  # Fixed seed for reproducibility
        )
        time_scptensor = time.perf_counter() - t0

        # Get results for comparison
        # Scanpy stores results in adata.obsm['X_pca']
        scanpy_pca = adata.obsm["X_pca"]

        # ScpTensor stores results in a new assay
        scptensor_pca_result = container.assays["pca"].layers["scores"].X

        # Compare results
        # Note: PCA signs can be flipped, so we use absolute correlation
        result = SimilarityMetrics.compare_results(
            scanpy_result=scanpy_pca,
            scptensor_result=scptensor_pca_result,
            module=f"pca_{n_components}pc",
            time_scanpy=time_scanpy,
            time_scptensor=time_scptensor,
        )

        # Add explained variance comparison
        if "variance" in adata.uns.get("pca", {}):
            scanpy_var = np.array(adata.uns["pca"]["variance"])
            scptensor_var = container.assays["pca"].var["explained_variance"].to_numpy()
            result.details["variance_correlation"] = float(
                np.corrcoef(scanpy_var[:n_components], scptensor_var[:n_components])[0, 1]
            )

        self.results.append(result)
        return result

    # ==================== UMAP Comparison ====================

    def compare_umap(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        random_state: int = 0,
    ) -> ComparisonResult:
        """Compare UMAP dimensionality reduction.

        Parameters
        ----------
        n_components : int
            Number of UMAP dimensions
        n_neighbors : int
            Number of neighbors for UMAP
        min_dist : float
            Minimum distance for UMAP
        metric : str
            Distance metric
        random_state : int
            Random seed for reproducibility

        Returns
        -------
        ComparisonResult
            Comparison metrics and timing

        Notes
        -----
        UMAP is stochastic, so results may vary slightly between runs.
        We use a more lenient threshold for UMAP comparisons.
        """
        # Create fresh copies
        adata = self.adata_scanpy.copy()
        container = from_scanpy(self.adata_scanpy, assay_name=self.assay_name)

        # Check if PCA is needed (UMAP typically works on PCA space)
        use_pca = adata.n_vars > 50

        # Scanpy UMAP
        t0 = time.perf_counter()
        if use_pca:
            sc.tl.pca(adata, n_comps=50, svd_solver="arpack")
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, metric=metric)
        sc.tl.umap(adata, n_components=n_components, min_dist=min_dist, random_state=random_state)
        time_scanpy = time.perf_counter() - t0

        # ScpTensor UMAP
        t0 = time.perf_counter()

        # First run PCA if needed (using ScpTensor's PCA)
        if use_pca:
            base_layer = "X"
            for layer_name in ["log1p", "log", "X"]:
                if layer_name in container.assays[self.assay_name].layers:
                    base_layer = layer_name
                    break
            container = scptensor_pca(
                container,
                assay_name=self.assay_name,
                base_layer_name=base_layer,
                new_assay_name="pca_umap",
                n_components=50,
                center=True,
                scale=False,
                random_state=42,
            )
            umap_input_layer = "scores"
            umap_assay = "pca_umap"
        else:
            umap_assay = self.assay_name
            umap_input_layer = "X"

        # Run UMAP
        container = scptensor_umap(
            container,
            assay_name=umap_assay,
            source_layer=umap_input_layer,
            new_assay_name="umap",
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )
        time_scptensor = time.perf_counter() - t0

        # Get results for comparison
        scanpy_umap = adata.obsm["X_umap"]
        scptensor_umap_result = container.assays["umap"].layers["embedding"].X

        # For UMAP, we need to use custom thresholds due to stochasticity
        result = SimilarityMetrics.compare_results(
            scanpy_result=scanpy_umap,
            scptensor_result=scptensor_umap_result,
            module=f"umap_{n_neighbors}n_{n_components}d",
            time_scanpy=time_scanpy,
            time_scptensor=time_scptensor,
        )

        # Override passed check with UMAP-specific thresholds
        cos_sim = result.metrics.get("cosine_similarity", 0.0)
        pearson = result.metrics.get("pearson_r", 0.0)
        result.passed = (
            cos_sim >= self.UMAP_THRESHOLD_COSINE and pearson >= self.UMAP_THRESHOLD_PEARSON
        )
        result.details["umap_mode"] = "pca_space" if use_pca else "direct"
        result.details["cosine_threshold"] = self.UMAP_THRESHOLD_COSINE
        result.details["pearson_threshold"] = self.UMAP_THRESHOLD_PEARSON

        self.results.append(result)
        return result

    # ==================== QC Metrics Comparison ====================

    def compare_qc_metrics(
        self,
        percent_top: list[int] | None = None,
        log1p: bool = False,
        detection_threshold: float = 0.0,
    ) -> ComparisonResult:
        """Compare QC metrics calculation between Scanpy and ScpTensor.

        Compares the calculation of quality control metrics including:
        - n_proteins (number of detected proteins per sample)
        - total_counts (total intensity per sample)
        - pct_counts (percentage statistics)

        Parameters
        ----------
        percent_top : List[int], optional
            Percentage of highest values to compute, e.g., [50, 100, 200, 500]
        log1p : bool
            Whether to log1p transform the metrics
        detection_threshold : float
            Threshold for a value to be considered detected (default: 0.0)

        Returns
        -------
        ComparisonResult
            Comparison metrics for QC metrics calculation

        Notes
        -----
        Scanpy stores QC metrics in adata.obs with the following keys:
        - n_genes_by_counts: number of genes with non-zero counts
        - total_counts: total counts per cell
        - pct_counts_*by_counts: percentage of counts in top genes

        ScpTensor computes equivalent metrics and stores them in container.obs.
        """
        # Create fresh copies
        adata = self.adata_scanpy.copy()
        container = from_scanpy(self.adata_scanpy, assay_name=self.assay_name)

        # Get base layer
        base_layer = "X"
        for layer_name in ["counts", "raw", "X"]:
            if layer_name in container.assays[self.assay_name].layers:
                base_layer = layer_name
                break

        # Get data matrix
        layer = container.assays[self.assay_name].layers[base_layer]
        X_scptensor = layer.X

        # Scanpy QC metrics calculation
        t0 = time.perf_counter()
        sc.pp.calculate_qc_metrics(
            adata,
            percent_top=percent_top,
            log1p=log1p,
            inplace=True,
        )
        time_scanpy = time.perf_counter() - t0

        # ScpTensor QC metrics calculation
        t0 = time.perf_counter()

        # Convert to dense if sparse for calculation
        if sp.issparse(X_scptensor):
            X_dense = X_scptensor.toarray()
        else:
            X_dense = X_scptensor

        # Calculate n_proteins (number of proteins with counts > threshold per sample)
        n_proteins = np.sum(X_dense > detection_threshold, axis=1).flatten()

        # Calculate total_counts (sum of all counts per sample)
        total_counts = np.sum(X_dense, axis=1).flatten()

        # Initialize results dictionary
        qc_metrics: dict[str, np.ndarray] = {
            "n_proteins": n_proteins,
            "total_counts": total_counts,
        }

        # Calculate percentage metrics if requested
        if percent_top:
            n_samples, n_features = X_dense.shape
            for n_top in percent_top:
                # Number of top proteins to consider
                n = min(n_top, n_features)

                # For each sample, calculate percentage of counts in top n proteins
                pct_counts_top = np.zeros(n_samples, dtype=float)
                for i in range(n_samples):
                    # Get indices of top n values for this sample
                    sample_values = X_dense[i, :]
                    top_indices = np.argpartition(-sample_values, n - 1)[:n]
                    top_sum = np.sum(sample_values[top_indices])

                    # Calculate percentage
                    if total_counts[i] > 0:
                        pct_counts_top[i] = (top_sum / total_counts[i]) * 100
                    else:
                        pct_counts_top[i] = 0.0

                qc_metrics[f"pct_counts_{n_top}"] = pct_counts_top

        # Apply log1p if requested
        if log1p:
            for key in qc_metrics:
                qc_metrics[key] = np.log1p(qc_metrics[key])

        time_scptensor = time.perf_counter() - t0

        # Extract Scanpy results for comparison
        scanpy_metrics: dict[str, np.ndarray] = {}
        scanpy_metrics["n_proteins"] = adata.obs["n_genes_by_counts"].to_numpy().flatten()
        scanpy_metrics["total_counts"] = adata.obs["total_counts"].to_numpy().flatten()

        if percent_top:
            for n_top in percent_top:
                key = f"pct_counts_in_top_{n_top}_genes"
                if key in adata.obs.columns:
                    scanpy_metrics[f"pct_counts_{n_top}"] = adata.obs[key].to_numpy().flatten()

        # Calculate Pearson correlation for each metric
        correlations: dict[str, float] = {}
        all_passed = True

        for key in ["n_proteins", "total_counts"]:
            if key in scanpy_metrics and key in qc_metrics:
                # Ensure both are 1D arrays
                scanpy_arr = scanpy_metrics[key].flatten()
                scptensor_arr = qc_metrics[key].flatten()

                # Calculate Pearson correlation
                if scanpy_arr.shape == scptensor_arr.shape and scanpy_arr.size > 1:
                    corr = np.corrcoef(scanpy_arr, scptensor_arr)[0, 1]
                    if np.isfinite(corr):
                        correlations[f"{key}_pearson"] = float(corr)
                        # QC metrics should have very high correlation (> 0.99)
                        if corr < 0.99:
                            all_passed = False
                    else:
                        correlations[f"{key}_pearson"] = 0.0
                        all_passed = False

        # Compare percentage metrics
        if percent_top:
            for n_top in percent_top:
                key = f"pct_counts_{n_top}"
                if key in scanpy_metrics and key in qc_metrics:
                    scanpy_arr = scanpy_metrics[key].flatten()
                    scptensor_arr = qc_metrics[key].flatten()

                    if scanpy_arr.shape == scptensor_arr.shape and scanpy_arr.size > 1:
                        corr = np.corrcoef(scanpy_arr, scptensor_arr)[0, 1]
                        if np.isfinite(corr):
                            correlations[f"{key}_pearson"] = float(corr)
                            if corr < 0.99:
                                all_passed = False
                        else:
                            correlations[f"{key}_pearson"] = 0.0
                            all_passed = False

        # Create result
        result = ComparisonResult(
            module="qc_metrics",
            metrics=correlations,
            time_scanpy=time_scanpy,
            time_scptensor=time_scptensor,
            passed=all_passed,
            details={
                "percent_top": percent_top,
                "log1p": log1p,
                "detection_threshold": detection_threshold,
                "n_samples": adata.n_obs,
                "n_features": adata.n_vars,
            },
        )

        self.results.append(result)
        return result

    # ==================== Clustering Comparison ====================

    def compare_clustering(
        self,
        method: Literal["leiden", "louvain", "kmeans"] = "leiden",
        resolution: float = 1.0,
        n_clusters: int = 8,
        random_state: int = 0,
    ) -> ComparisonResult:
        """Compare clustering results.

        Parameters
        ----------
        method : str
            Clustering method
        resolution : float
            Resolution for Leiden/Louvain
        n_clusters : int
            Number of clusters for KMeans
        random_state : int
            Random seed for reproducibility

        Returns
        -------
        ComparisonResult
            Adjusted Rand Index for cluster similarity

        Notes
        -----
        Clustering results can vary due to the stochastic nature of the algorithms.
        We use Adjusted Rand Index (ARI) to measure similarity.
        """
        # Create fresh copies
        adata = self.adata_scanpy.copy()
        container = from_scanpy(self.adata_scanpy, assay_name=self.assay_name)

        # Need PCA and neighbors first for Leiden/Louvain
        if method in ["leiden", "louvain"]:
            # Scanpy pipeline
            t0 = time.perf_counter()
            sc.tl.pca(adata, n_comps=50)
            sc.pp.neighbors(adata)
            if method == "leiden":
                sc.tl.leiden(adata, resolution=resolution)
                scanpy_labels = adata.obs["leiden"].to_numpy()
            else:
                sc.tl.louvain(adata, resolution=resolution)
                scanpy_labels = adata.obs["louvain"].to_numpy()
            time_scanpy = time.perf_counter() - t0

            # Convert string labels to integers
            unique_labels = np.unique(scanpy_labels)
            label_map = {label: i for i, label in enumerate(unique_labels)}
            scanpy_labels_int = np.array([label_map[label] for label in scanpy_labels])

        else:  # KMeans
            from sklearn.cluster import KMeans

            t0 = time.perf_counter()
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            scanpy_labels_int = kmeans.fit_predict(adata.X)
            time_scanpy = time.perf_counter() - t0

        # ScpTensor clustering
        # For now, we use sklearn directly since ScpTensor doesn't have clustering yet
        t0 = time.perf_counter()

        # Get data matrix
        base_layer = "X"
        for layer_name in ["log1p", "log", "X"]:
            if layer_name in container.assays[self.assay_name].layers:
                base_layer = layer_name
                break

        X = container.assays[self.assay_name].layers[base_layer].X
        if sp.issparse(X):
            X = X.toarray()

        if method in ["leiden", "louvain"]:
            # Use scanpy's clustering on ScpTensor data for comparison
            # (ScpTensor doesn't have graph-based clustering yet)
            adata_scptensor = to_scanpy(container, assay_name=self.assay_name)
            sc.tl.pca(adata_scptensor, n_components=50)
            sc.pp.neighbors(adata_scptensor)
            if method == "leiden":
                sc.tl.leiden(adata_scptensor, resolution=resolution)
                scptensor_labels = adata_scptensor.obs["leiden"].to_numpy()
            else:
                sc.tl.louvain(adata_scptensor, resolution=resolution)
                scptensor_labels = adata_scptensor.obs["louvain"].to_numpy()

            unique_labels = np.unique(scptensor_labels)
            label_map = {label: i for i, label in enumerate(unique_labels)}
            scptensor_labels_int = np.array([label_map[label] for label in scptensor_labels])
        else:  # KMeans
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            scptensor_labels_int = kmeans.fit_predict(X)

        time_scptensor = time.perf_counter() - t0

        # Calculate Adjusted Rand Index
        from sklearn.metrics import adjusted_rand_score

        ari = adjusted_rand_score(scanpy_labels_int, scptensor_labels_int)

        # Create result
        result = ComparisonResult(
            module=f"clustering_{method}",
            metrics={"ari": ari},
            time_scanpy=time_scanpy,
            time_scptensor=time_scptensor,
            passed=ari >= 0.8,  # Good clustering similarity
            details={
                "n_clusters_scanpy": len(np.unique(scanpy_labels_int)),
                "n_clusters_scptensor": len(np.unique(scptensor_labels_int)),
            },
        )

        self.results.append(result)
        return result

    # ==================== HVG Comparison ====================

    def compare_hvg(
        self,
        n_top_genes: int = 2000,
        flavor: Literal["seurat_v3", "seurat", "cell_ranger"] = "seurat_v3",
        method: Literal["cv", "dispersion"] = "cv",
    ) -> ComparisonResult:
        """Compare Highly Variable Genes/Proteins (HVG) selection.

        Compares the feature selection results between Scanpy's
        highly_variable_genes and ScpTensor's select_hvg.

        Since feature selection is discrete (selected/not selected),
        Jaccard similarity is used instead of Pearson correlation.

        Parameters
        ----------
        n_top_genes : int
            Number of top variable features to select
        flavor : str
            Scanpy flavor for HVG calculation ('seurat_v3', 'seurat', 'cell_ranger')
            Note: ScpTensor uses CV-based method which approximates seurat_v3
        method : str
            ScpTensor method: 'cv' for Coefficient of Variation,
            'dispersion' for variance-to-mean ratio

        Returns
        -------
        ComparisonResult
            Comparison metrics with Jaccard similarity

        Notes
        -----
        Scanpy stores HVG results in adata.var['highly_variable'] (boolean array).
        ScpTensor stores results in assay.var['highly_variable'] when subset=False.

        The Jaccard similarity measures the overlap between selected features:
        J(A, B) = |A n B| / |A u B|

        A value of 1.0 means identical feature selection, 0.0 means no overlap.
        """
        # Import here to avoid issues when Scanpy is not available
        from tests.real_data_comparison.metrics import jaccard_similarity

        # Create fresh copies
        adata = self.adata_scanpy.copy()
        container = from_scanpy(self.adata_scanpy, assay_name=self.assay_name)

        # Get base layer name
        base_layer = "X"
        for layer_name in ["counts", "raw", "X"]:
            if layer_name in container.assays[self.assay_name].layers:
                base_layer = layer_name
                break

        # Scanpy HVG selection
        t0 = time.perf_counter()
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top_genes,
            flavor=flavor,
            inplace=True,
        )
        time_scanpy = time.perf_counter() - t0

        # Get Scanpy selected features (indices where highly_variable is True)
        scanpy_hvg_mask = adata.var["highly_variable"].values
        scanpy_selected = set(np.where(scanpy_hvg_mask)[0])
        n_scanpy_selected = len(scanpy_selected)

        # ScpTensor HVG selection
        t0 = time.perf_counter()
        container_hvg = select_hvg(
            container,
            assay_name=self.assay_name,
            layer=base_layer,
            n_top_features=n_top_genes,
            method=method,
            subset=False,  # Keep all features, add annotation
        )
        time_scptensor = time.perf_counter() - t0

        # Get ScpTensor selected features
        scptensor_hvg_mask = container_hvg.assays[self.assay_name].var["highly_variable"].to_numpy()
        scptensor_selected = set(np.where(scptensor_hvg_mask)[0])
        n_scptensor_selected = len(scptensor_selected)

        # Calculate Jaccard similarity
        jaccard = jaccard_similarity(scanpy_selected, scptensor_selected)

        # Calculate overlap statistics
        intersection = len(scanpy_selected & scptensor_selected)
        union = len(scanpy_selected | scptensor_selected)
        only_scanpy = len(scanpy_selected - scptensor_selected)
        only_scptensor = len(scptensor_selected - scanpy_selected)

        # Determine pass threshold
        # Jaccard > 0.7 is considered good for feature selection
        # since different algorithms may select slightly different features
        passed = jaccard >= 0.7

        # Create result
        result = ComparisonResult(
            module=f"hvg_{n_top_genes}",
            metrics={"jaccard": jaccard},
            time_scanpy=time_scanpy,
            time_scptensor=time_scptensor,
            passed=passed,
            details={
                "n_top_genes": n_top_genes,
                "scanpy_flavor": flavor,
                "scptensor_method": method,
                "n_scanpy_selected": n_scanpy_selected,
                "n_scptensor_selected": n_scptensor_selected,
                "intersection": intersection,
                "union": union,
                "only_scanpy": only_scanpy,
                "only_scptensor": only_scptensor,
                "jaccard_threshold": 0.7,
            },
        )

        self.results.append(result)
        return result

    # ==================== Reporting ====================

    def summary(self) -> dict[str, Any]:
        """Generate summary of all comparison results.

        Returns
        -------
        Dict with summary statistics
        """
        if not self.results:
            return {"status": "No tests run yet"}

        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        # Calculate aggregate metrics
        avg_speedup = np.mean(
            [speedup_factor(r.time_scanpy, r.time_scptensor) for r in self.results]
        )
        max_speedup = max([speedup_factor(r.time_scanpy, r.time_scptensor) for r in self.results])
        min_speedup = min([speedup_factor(r.time_scanpy, r.time_scptensor) for r in self.results])

        return {
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0,
            "avg_speedup": avg_speedup,
            "max_speedup": max_speedup,
            "min_speedup": min_speedup,
            "results": [r.to_dict() for r in self.results],
        }

    def print_summary(self) -> None:
        """Print formatted summary of results."""
        summary = self.summary()
        print("\n" + "=" * 70)
        print(f"Comparison Summary: {summary['passed']}/{summary['total_tests']} tests passed")
        print(f"Pass Rate: {summary['pass_rate']:.1%}")
        print(f"Average Speedup: {summary['avg_speedup']:.2f}x")
        print("=" * 70)

        for r in self.results:
            status = "PASS" if r.passed else "FAIL"

            # Get the appropriate metric based on module type
            if r.module.startswith("hvg"):
                metric_name = "Jaccard"
                metric_value = r.metrics.get("jaccard", "N/A")
                if isinstance(metric_value, str):
                    metric_str = metric_value
                else:
                    metric_str = f"{metric_value:.4f}"
            else:
                metric_value = r.metrics.get("pearson_r", r.metrics.get("ari", "N/A"))
                if isinstance(metric_value, str):
                    metric_str = metric_value
                else:
                    metric_str = f"{metric_value:.4f}"
                metric_name = "Pearson/ARI"

            print(f"\n[{status}] {r.module}:")
            print(f"  {metric_name}: {metric_str}")
            print(
                f"  Time: Scanpy {format_time(r.time_scanpy)} | "
                f"ScpTensor {format_time(r.time_scptensor)} | "
                f"Speedup: {speedup_factor(r.time_scanpy, r.time_scptensor):.2f}x"
            )

            if r.details:
                for key, value in r.details.items():
                    if "threshold" not in key:  # Skip threshold details
                        print(f"  {key}: {value}")

    def save_report(self, path: str | Path) -> None:
        """Save detailed report to file.

        Parameters
        ----------
        path : str | Path
            Output path for report (markdown format)
        """
        path = Path(path)
        summary = self.summary()

        lines = [
            "# ScpTensor vs Scanpy Comparison Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Data:** {self.h5ad_path.name}",
            "",
            "## Summary",
            "",
            f"- **Total Tests:** {summary['total_tests']}",
            f"- **Passed:** {summary['passed']}",
            f"- **Failed:** {summary['failed']}",
            f"- **Pass Rate:** {summary['pass_rate']:.1%}",
            f"- **Average Speedup:** {summary['avg_speedup']:.2f}x",
            "",
            "## Detailed Results",
            "",
        ]

        for i, r in enumerate(self.results, 1):
            status_icon = "✓" if r.passed else "✗"
            lines.append(f"### {i}. {r.module} {status_icon}")
            lines.append("")
            lines.append("**Metrics:**")
            for name, value in r.metrics.items():
                if name in ["cosine_similarity", "pearson_r", "spearman_r", "ari", "jaccard"]:
                    lines.append(f"- {name}: {value:.6f}")
                else:
                    lines.append(f"- {name}: {value:.6e}")
            lines.append("")
            lines.append("**Timing:**")
            lines.append(f"- Scanpy: {format_time(r.time_scanpy)}")
            lines.append(f"- ScpTensor: {format_time(r.time_scptensor)}")
            spd = speedup_factor(r.time_scanpy, r.time_scptensor)
            if spd > 0:
                lines.append(f"- Speedup: {spd:.2f}x")
            else:
                lines.append(f"- Slowdown: {abs(spd):.2f}x")
            lines.append("")

            if r.details:
                lines.append("**Details:**")
                for key, value in r.details.items():
                    lines.append(f"- {key}: {value}")
                lines.append("")

        lines.append("---")
        lines.append("*Report generated by ScpTensor Comparison Suite*")

        path.write_text("\n".join(lines))
        print(f"\nReport saved to: {path}")


def run_phase1_tests(
    h5ad_path: str | Path,
    verbose: bool = True,
) -> ComparisonSuite:
    """Run Phase 1 comparison tests (Normalization + PCA + UMAP).

    Parameters
    ----------
    h5ad_path : str | Path
        Path to test data
    verbose : bool
        Whether to print progress

    Returns
    -------
    ComparisonSuite
        Test suite with all results
    """
    if not SCANPY_AVAILABLE:
        raise ImportError("Scanpy is required. Install with: pip install scanpy")

    if verbose:
        print("Starting Phase 1 Comparison Tests")
        print("=" * 70)

    suite = ComparisonSuite(h5ad_path)

    # Test 1: Log1p normalization
    if verbose:
        print("\n1. Testing log1p normalization...")
    result = suite.compare_normalization("log1p")
    if verbose:
        print(f"   Result: {'PASS' if result.passed else 'FAIL'}")
        print(f"   Pearson: {result.metrics.get('pearson_r', 'N/A'):.4f}")
        print(f"   Speedup: {speedup_factor(result.time_scanpy, result.time_scptensor):.2f}x")

    if not result.passed:
        if verbose:
            print("\n[ERROR] Normalization failed, stopping Phase 1")
        return suite

    # Test 2: PCA
    if verbose:
        print("\n2. Testing PCA (50 components)...")
    try:
        result = suite.compare_pca(n_components=50)
        if verbose:
            print(f"   Result: {'PASS' if result.passed else 'FAIL'}")
            print(f"   Pearson: {result.metrics.get('pearson_r', 'N/A'):.4f}")
            print(f"   Speedup: {speedup_factor(result.time_scanpy, result.time_scptensor):.2f}x")
    except Exception as e:
        if verbose:
            print(f"   Result: ERROR - {e}")
        # Create a failure result
        from tests.real_data_comparison.metrics import ComparisonResult

        result = ComparisonResult(
            module="pca_50pc",
            metrics={},
            time_scanpy=0,
            time_scptensor=0,
            passed=False,
            details={"error": str(e)},
        )
        suite.results.append(result)

    if not result.passed and verbose:
        print("\n[WARNING] PCA failed, continuing to UMAP")

    # Test 3: UMAP
    if verbose:
        print("\n3. Testing UMAP (n_neighbors=15)...")
    try:
        result = suite.compare_umap(n_neighbors=15, random_state=42)
        if verbose:
            print(f"   Result: {'PASS' if result.passed else 'FAIL'}")
            print(f"   Pearson: {result.metrics.get('pearson_r', 'N/A'):.4f}")
            print(f"   Speedup: {speedup_factor(result.time_scanpy, result.time_scptensor):.2f}x")
    except Exception as e:
        if verbose:
            print(f"   Result: ERROR - {e}")
        # Create a failure result
        from tests.real_data_comparison.metrics import ComparisonResult

        result = ComparisonResult(
            module="umap_15n_2d",
            metrics={},
            time_scanpy=0,
            time_scptensor=0,
            passed=False,
            details={"error": str(e)},
        )
        suite.results.append(result)

    suite.print_summary()
    return suite


if __name__ == "__main__":
    print("ComparisonSuite module for ScpTensor vs Scanpy testing")
    print()
    print("Usage:")
    print("  from tests.real_data_comparison.comparison_suite import run_phase1_tests")
    print("  suite = run_phase1_tests('path/to/data.h5ad')")
    print()
    print("Available comparison methods:")
    print("  - compare_qc_metrics(percent_top=[50, 100, 200, 500])")
    print("  - compare_normalization(method='log1p|log|sqrt')")
    print("  - compare_pca(n_components=50)")
    print("  - compare_umap(n_neighbors=15)")
    print("  - compare_clustering(method='leiden|louvain|kmeans')")
    print("  - compare_hvg(n_top_genes=2000)")
    print()
    print("Example:")
    print("  suite = ComparisonSuite('path/to/data.h5ad')")
    print("  result = suite.compare_hvg(n_top_genes=2000)")
    print("  print(f'Jaccard similarity: {result.metrics[\"jaccard\"]:.3f}')")
