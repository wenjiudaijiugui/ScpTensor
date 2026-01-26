"""Simplified adapters for Scanpy vs ScpTensor comparison."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import polars as pl

try:
    import anndata as ad
    import scanpy as sc
    from scipy.sparse import issparse

    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False
    sc = None
    ad = None

from scptensor.core.structures import Assay, ScpContainer, ScpMatrix
from scptensor.dim_reduction.pca import reduce_pca
from scptensor.dim_reduction.umap import reduce_umap
from scptensor.normalization.log_transform import log_transform


def to_anndata(container: ScpContainer) -> Any:
    """Convert ScpContainer to AnnData."""
    if not SCANPY_AVAILABLE:
        raise ImportError("Scanpy not installed")

    assay = list(container.assays.values())[0]
    matrix = list(assay.layers.values())[0]

    x = matrix.X.copy().astype(float)
    if matrix.M is not None:
        x[matrix.M > 0] = np.nan

    adata = ad.AnnData(X=x)

    obs_dict = {}
    if "batch" in container.obs.columns:
        obs_dict["batch"] = container.obs["batch"].to_numpy().astype(str)
    if "group" in container.obs.columns:
        obs_dict["group"] = container.obs["group"].to_numpy().astype(str)

    if obs_dict:
        import pandas as pd

        adata.obs = pd.DataFrame(obs_dict, index=adata.obs_names)

    return adata


def from_anndata(adata: Any) -> ScpContainer:
    """Convert AnnData to ScpContainer."""
    x = adata.X

    if issparse(x):
        x = x.toarray()

    x = np.asarray(x, dtype=float)
    m = np.isnan(x).astype(np.uint8)
    x = np.nan_to_num(x, nan=0.0)

    n_samples, n_features = x.shape
    obs_data = {"_index": [f"S{i:04d}" for i in range(n_samples)]}

    if hasattr(adata, "obs") and "batch" in adata.obs.columns:
        obs_data["batch"] = adata.obs["batch"].values.astype(str)
    if hasattr(adata, "obs") and "group" in adata.obs.columns:
        obs_data["group"] = adata.obs["group"].values.astype(str)

    obs = pl.DataFrame(obs_data)
    var = pl.DataFrame({"_index": [f"P{i:04d}" for i in range(n_features)]})

    matrix = ScpMatrix(X=x, M=m)
    assay = Assay(var=var, layers={"raw": matrix})

    return ScpContainer(assays={"protein": assay}, obs=obs)


def _create_container(
    x: np.ndarray,
    m: np.ndarray | None = None,
    batch_labels: np.ndarray | None = None,
    group_labels: np.ndarray | None = None,
) -> ScpContainer:
    """Create ScpContainer from arrays."""
    n_samples, n_features = x.shape

    obs_data = {"_index": [f"S{i:04d}" for i in range(n_samples)]}
    if batch_labels is not None:
        obs_data["batch"] = batch_labels.astype(str)
    if group_labels is not None:
        obs_data["group"] = group_labels.astype(str)

    obs = pl.DataFrame(obs_data)
    var = pl.DataFrame({"_index": [f"P{i:04d}" for i in range(n_features)]})

    m_copy = m.copy().astype(np.uint8) if m is not None else None
    matrix = ScpMatrix(X=x.copy(), M=m_copy)
    assay = Assay(var=var, layers={"raw": matrix})

    return ScpContainer(assays={"protein": assay}, obs=obs)


def _extract_result(container: ScpContainer, layer: str = "processed") -> np.ndarray:
    """Extract result matrix from container."""
    assay = container.assays["protein"]
    if layer in assay.layers:
        return assay.layers[layer].X.copy()
    last_layer = list(assay.layers.keys())[-1]
    return assay.layers[last_layer].X.copy()


def run_scanpy(
    container: ScpContainer,
    methods: list[str],
) -> dict[str, Any]:
    """Run Scanpy analysis pipeline."""
    if not SCANPY_AVAILABLE:
        raise ImportError("Scanpy not installed")

    sc.settings.verbosity = 0
    sc.settings.n_jobs = -1

    adata = to_anndata(container)
    results = {"adata": adata}

    for method in methods:
        if method == "normalize":
            start = time.time()
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            results["normalize_time"] = time.time() - start

        elif method == "scale":
            start = time.time()
            sc.pp.scale(adata, max_value=10)
            results["scale_time"] = time.time() - start

        elif method == "pca":
            start = time.time()
            sc.pp.pca(adata, n_comps=50, svd_solver="arpack")
            results["pca_time"] = time.time() - start
            results["X_pca"] = adata.obsm["X_pca"].copy()
            results["variance_ratio"] = adata.uns["pca"]["variance_ratio"].copy()

        elif method == "umap":
            if adata.shape[1] > 50:
                sc.pp.pca(adata, n_comps=50)

            start = time.time()
            sc.tl.umap(adata)
            results["umap_time"] = time.time() - start
            results["X_umap"] = adata.obsm["X_umap"].copy()

        elif method == "kmeans":
            from sklearn.cluster import KMeans

            x = adata.X if not issparse(adata.X) else adata.X.toarray()

            start = time.time()
            kmeans = KMeans(n_clusters=5, random_state=42, n_init="auto")
            labels = kmeans.fit_predict(x)
            results["kmeans_time"] = time.time() - start
            results["labels"] = labels

        elif method == "hvg":
            start = time.time()

            batch_key = "batch" if "batch" in adata.obs else None
            sc.pp.highly_variable_genes(
                adata, n_top_genes=2000, flavor="seurat_v3", batch_key=batch_key
            )

            results["hvg_time"] = time.time() - start
            results["hvg_indices"] = np.where(adata.var["highly_variable"].values)[0]

    return results


def run_scptensor(
    container: ScpContainer,
    methods: list[str],
) -> dict[str, Any]:
    """Run ScpTensor analysis pipeline."""
    results = {"container": container}

    for method in methods:
        if method == "normalize":
            start = time.time()
            container = log_transform(container, "protein", "raw", "log")
            results["normalize_time"] = time.time() - start
            results["container"] = container

        elif method == "scale":
            x = _extract_result(container).copy().astype(float)

            start = time.time()
            mean = x.mean(axis=0, keepdims=True)
            std = x.std(axis=0, keepdims=True, ddof=1)
            std[std == 0] = 1.0
            x_scaled = (x - mean) / std

            assay = container.assays["protein"]
            matrix = ScpMatrix(
                X=x_scaled, M=assay.layers["raw"].M.copy() if "raw" in assay.layers else None
            )
            assay.layers["scaled"] = matrix

            results["scale_time"] = time.time() - start
            results["container"] = container

        elif method == "pca":
            start = time.time()
            container = reduce_pca(
                container,
                "protein",
                "log" if "log" in container.assays["protein"].layers else "raw",
                "pca",
                n_components=50,
                svd_solver="arpack",
                random_state=0,
            )
            results["pca_time"] = time.time() - start
            results["container"] = container

            pca_assay = container.assays["pca"]
            results["X_pca"] = pca_assay.layers["scores"].X.copy()

            explained_var = pca_assay.var["explained_variance"].to_numpy()
            results["variance_ratio"] = explained_var / explained_var.sum()

        elif method == "umap":
            start = time.time()
            container = reduce_umap(
                container,
                "protein",
                "log" if "log" in container.assays["protein"].layers else "raw",
                "umap",
                n_components=2,
            )
            results["umap_time"] = time.time() - start
            results["container"] = container
            results["X_umap"] = _extract_result(container, "umap")

        elif method == "kmeans":
            from sklearn.cluster import KMeans

            x = _extract_result(container)

            start = time.time()
            kmeans_obj = KMeans(n_clusters=5, random_state=42, n_init="auto")
            labels = kmeans_obj.fit_predict(x)
            results["kmeans_time"] = time.time() - start
            results["labels"] = labels

    return results


def compare_frameworks(
    container: ScpContainer,
    scanpy_methods: list[str],
    scptensor_methods: list[str],
) -> dict[str, dict[str, Any]]:
    """Compare Scanpy and ScpTensor frameworks."""
    scanpy_results = run_scanpy(container, scanpy_methods)
    scptensor_results = run_scptensor(container, scptensor_methods)

    return {
        "scanpy": scanpy_results,
        "scptensor": scptensor_results,
    }
