"""Scanpy adapter for ScpTensor vs Scanpy comparison benchmarks.

This module provides wrappers around Scanpy methods to allow
fair comparison with ScpTensor implementations.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

warnings.filterwarnings("ignore", message=".*observation names.*")
warnings.filterwarnings("ignore", message=".*variable names.*")

if TYPE_CHECKING:
    from collections.abc import Callable

try:
    import scanpy as sc
    import anndata as ad
    from scipy.sparse import issparse
    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False
    sc = None
    ad = None


def _container_to_anndata(
    X: np.ndarray,
    M: np.ndarray | None = None,
    batch_labels: np.ndarray | None = None,
    group_labels: np.ndarray | None = None,
) -> Any:
    if not SCANPY_AVAILABLE:
        raise ImportError("Scanpy is not installed. Install with: pip install scanpy")

    X_nan = X.copy().astype(float)
    if M is not None:
        X_nan[M > 0] = np.nan

    adata = ad.AnnData(X=X_nan)

    if batch_labels is not None or group_labels is not None:
        import pandas as pd
        obs_dict = {}
        if batch_labels is not None:
            obs_dict["batch"] = batch_labels.astype(str)
        if group_labels is not None:
            obs_dict["group"] = group_labels.astype(str)
        adata.obs = pd.DataFrame(obs_dict, index=adata.obs_names)

    return adata


def _anndata_to_arrays(adata: Any) -> tuple[np.ndarray, np.ndarray]:
    X = adata.X

    if issparse(X):
        X = X.toarray()

    X = np.asarray(X, dtype=float)
    M = np.isnan(X).astype(np.uint8)
    X = np.nan_to_num(X, nan=0.0)

    return X, M


class ScanpyMethods:
    def __init__(self) -> None:
        if not SCANPY_AVAILABLE:
            raise ImportError("Scanpy is not installed. Install with: pip install scanpy")
        sc.settings.verbosity = 0
        sc.settings.n_jobs = -1

    def log_normalize(
        self,
        X: np.ndarray,
        M: np.ndarray | None = None,
        target_sum: float = 1e4,
        log_base: float = 2.0,
    ) -> tuple[np.ndarray, float]:
        import time

        adata = _container_to_anndata(X, M)

        start = time.time()
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata, base=np.log(log_base) if log_base != np.e else None)
        runtime = time.time() - start

        X_norm, _ = _anndata_to_arrays(adata)
        return X_norm, runtime

    def z_score_normalize(
        self,
        X: np.ndarray,
        M: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        import time

        X_for_norm = X.copy().astype(float)
        if M is not None:
            X_for_norm[M > 0] = np.nan

        start = time.time()

        mean = np.nanmean(X_for_norm, axis=0, keepdims=True)
        std = np.nanstd(X_for_norm, axis=0, keepdims=True, ddof=1)
        std[std == 0] = 1.0

        X_norm = (X_for_norm - mean) / std
        X_norm = np.nan_to_num(X_norm, nan=0.0)

        runtime = time.time() - start
        return X_norm, runtime

    def knn_impute(
        self,
        X: np.ndarray,
        M: np.ndarray,
        k: int = 15,
    ) -> tuple[np.ndarray, float]:
        import time
        from sklearn.neighbors import NearestNeighbors

        valid_mask = M == 0
        missing_mask = M > 0

        start = time.time()

        X_valid_only = X.copy()
        X_valid_only[missing_mask] = 0

        k_adj = min(k + 1, X.shape[0])
        nbrs = NearestNeighbors(n_neighbors=k_adj, n_jobs=-1)
        nbrs.fit(X_valid_only)
        distances, indices = nbrs.kneighbors(X_valid_only)

        X_imp = X.copy().astype(float)

        missing_rows, missing_cols = np.where(missing_mask)
        for i, j in zip(missing_rows, missing_cols):
            neighbor_idxs = indices[i, 1:k_adj]
            valid_mask_neighbors = M[neighbor_idxs, j] == 0
            if valid_mask_neighbors.any():
                X_imp[i, j] = X[neighbor_idxs[valid_mask_neighbors], j].mean()

        runtime = time.time() - start
        return X_imp, runtime

    def pca(
        self,
        X: np.ndarray,
        M: np.ndarray | None = None,
        n_components: int = 50,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        import time

        X_pca_input = X.copy().astype(float)
        if M is not None and M.sum() > 0:
            for j in range(X.shape[1]):
                valid_mask = M[:, j] == 0
                if valid_mask.any():
                    X_pca_input[~valid_mask, j] = X[valid_mask, j].mean()

        adata = _container_to_anndata(X_pca_input, None)

        start = time.time()
        sc.pp.pca(adata, n_comps=n_components, svd_solver="arpack")
        runtime = time.time() - start

        X_pca = adata.obsm["X_pca"]
        variance_ratio = adata.uns["pca"]["variance_ratio"].copy()

        return X_pca, variance_ratio, runtime

    def umap(
        self,
        X: np.ndarray,
        M: np.ndarray | None = None,
        n_components: int = 2,
        n_neighbors: int = 15,
    ) -> tuple[np.ndarray, float]:
        import time

        adata = _container_to_anndata(X, M)

        if X.shape[1] > 50:
            sc.pp.pca(adata, n_comps=50)

        start = time.time()
        sc.tl.umap(adata, n_components=n_components, n_neighbors=n_neighbors)
        runtime = time.time() - start

        X_umap = adata.obsm["X_umap"]
        return X_umap, runtime

    def kmeans(
        self,
        X: np.ndarray,
        M: np.ndarray | None = None,
        n_clusters: int = 5,
        random_state: int = 42,
    ) -> tuple[np.ndarray, float]:
        import time
        from sklearn.cluster import KMeans

        start = time.time()
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        labels = kmeans.fit_predict(X)
        runtime = time.time() - start

        return labels, runtime

    def highly_variable_genes(
        self,
        X: np.ndarray,
        M: np.ndarray | None = None,
        n_top_genes: int = 2000,
        flavor: str = "seurat_v3",
    ) -> tuple[np.ndarray, float]:
        import time

        adata = _container_to_anndata(X, M)

        start = time.time()

        batch_key = "batch" if "batch" in adata.obs else None
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor=flavor, batch_key=batch_key)

        runtime = time.time() - start

        hvg_mask = adata.var["highly_variable"].values
        hvg_indices = np.where(hvg_mask)[0]

        return hvg_indices, runtime


def get_scanpy_methods() -> ScanpyMethods:
    return ScanpyMethods()
