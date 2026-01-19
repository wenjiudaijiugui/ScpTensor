"""ScpTensor method adapters for comparison benchmarks.

This module provides optimized wrappers around ScpTensor methods
for fair comparison with Scanpy implementations.
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl

from scptensor.core.structures import Assay, ScpContainer, ScpMatrix
from scptensor.dim_reduction.pca import reduce_pca
from scptensor.dim_reduction.umap import reduce_umap
from scptensor.feature_selection.hvg import select_hvg
from scptensor.impute.knn import impute_knn
from scptensor.impute.svd import impute_svd
from scptensor.normalization.log import norm_log


def _create_container(
    X: np.ndarray,
    M: np.ndarray | None = None,
    batch_labels: np.ndarray | None = None,
    group_labels: np.ndarray | None = None,
) -> ScpContainer:
    from scptensor.core.structures import Assay

    n_samples, n_features = X.shape

    obs_data = {"_index": [f"S{i:04d}" for i in range(n_samples)]}
    if batch_labels is not None:
        obs_data["batch"] = batch_labels.astype(str)
    if group_labels is not None:
        obs_data["group"] = group_labels.astype(str)

    obs = pl.DataFrame(obs_data)
    var = pl.DataFrame({"_index": [f"P{i:04d}" for i in range(n_features)]})

    M_copy = M.copy().astype(np.uint8) if M is not None else None
    matrix = ScpMatrix(X=X.copy(), M=M_copy)
    assay = Assay(var=var, layers={"raw": matrix})

    return ScpContainer(assays={"protein": assay}, obs=obs)


def _extract_result(container: ScpContainer, layer: str = "processed") -> np.ndarray:
    assay = container.assays["protein"]
    if layer in assay.layers:
        return assay.layers[layer].X.copy()
    last_layer = list(assay.layers.keys())[-1]
    return assay.layers[last_layer].X.copy()


class ScpTensorMethods:
    @staticmethod
    def log_normalize(
        X: np.ndarray,
        M: np.ndarray | None = None,
        base: float = 2.0,
        offset: float = 1.0,
    ) -> tuple[np.ndarray, float]:
        container = _create_container(X, M)
        start = time.time()
        result = norm_log(container, "protein", "raw", "log", base=base, offset=offset)
        runtime = time.time() - start
        return _extract_result(result, "log"), runtime

    @staticmethod
    def z_score_normalize(
        X: np.ndarray,
        M: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
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

    @staticmethod
    def knn_impute(
        X: np.ndarray,
        M: np.ndarray,
        k: int = 15,
    ) -> tuple[np.ndarray, float]:
        container = _create_container(X, M)
        start = time.time()
        result = impute_knn(container, "protein", "raw", "imputed", k=k)
        runtime = time.time() - start
        return _extract_result(result, "imputed"), runtime

    @staticmethod
    def svd_impute(
        X: np.ndarray,
        M: np.ndarray,
        rank: int = 50,
    ) -> tuple[np.ndarray, float]:
        container = _create_container(X, M)
        start = time.time()
        result = impute_svd(container, "protein", "raw", "imputed", rank=rank)
        runtime = time.time() - start
        return _extract_result(result, "imputed"), runtime

    @staticmethod
    def pca(
        X: np.ndarray,
        M: np.ndarray | None = None,
        n_components: int = 50,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        X_pca_input = X.copy().astype(float)
        if M is not None and M.sum() > 0:
            for j in range(X.shape[1]):
                valid_mask = M[:, j] == 0
                if valid_mask.any():
                    X_pca_input[~valid_mask, j] = X[valid_mask, j].mean()

        container = _create_container(X_pca_input, None)

        start = time.time()
        result = reduce_pca(container, "protein", "raw", "pca", n_components=n_components, svd_solver="arpack", random_state=0)
        runtime = time.time() - start

        pca_assay = result.assays["pca"]
        X_pca = pca_assay.layers["scores"].X.copy()
        explained_var = pca_assay.var["explained_variance"].to_numpy()
        variance_ratio = explained_var / explained_var.sum()

        return X_pca, variance_ratio, runtime

    @staticmethod
    def umap(
        X: np.ndarray,
        M: np.ndarray | None = None,
        n_components: int = 2,
        n_neighbors: int = 15,
    ) -> tuple[np.ndarray, float]:
        container = _create_container(X, M)
        start = time.time()
        result = reduce_umap(container, "protein", "raw", "umap", n_components=n_components, n_neighbors=n_neighbors)
        runtime = time.time() - start
        return _extract_result(result, "umap"), runtime

    @staticmethod
    def kmeans(
        X: np.ndarray,
        M: np.ndarray | None = None,
        n_clusters: int = 5,
        random_state: int = 42,
    ) -> tuple[np.ndarray, float]:
        from sklearn.cluster import KMeans

        start = time.time()
        kmeans_obj = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        labels = kmeans_obj.fit_predict(X)
        runtime = time.time() - start

        return labels, runtime

    @staticmethod
    def highly_variable_genes(
        X: np.ndarray,
        M: np.ndarray | None = None,
        n_top_genes: int = 2000,
    ) -> tuple[np.ndarray, float]:
        container = _create_container(X, M)
        start = time.time()
        result = select_hvg(container, "protein", "raw", n_top_features=n_top_genes, subset=False)
        runtime = time.time() - start

        var = result.assays["protein"].var
        hvg_mask = var["highly_variable"].to_numpy()
        hvg_indices = np.where(hvg_mask)[0]

        return hvg_indices, runtime


def get_scptensor_methods() -> type[ScpTensorMethods]:
    return ScpTensorMethods
