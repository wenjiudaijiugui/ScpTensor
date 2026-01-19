"""Batch correction performance test module."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from scptensor.benchmark.data_provider import COMPARISON_DATASETS, ComparisonDataset, get_provider
from scptensor.benchmark.modules.base import BaseModule, ModuleConfig, ModuleResult
from scptensor.benchmark.scptensor_adapter import _create_container, _extract_result
from scptensor.benchmark.utils.registry import register_module
from scptensor.integration import integrate_combat, integrate_mnn

if TYPE_CHECKING:
    from collections.abc import Callable

try:
    from scptensor.integration import integrate_harmony
    HARMONY_AVAILABLE = True
except ImportError:
    HARMONY_AVAILABLE = False

try:
    from scptensor.integration import integrate_scanorama
    SCANORAMA_AVAILABLE = True
except ImportError:
    SCANORAMA_AVAILABLE = False

try:
    import scib_metrics
    SCIB_METRICS_AVAILABLE = True
except ImportError:
    SCIB_METRICS_AVAILABLE = False


def _time_function(func: Callable[[], tuple[np.ndarray, float]]) -> tuple[np.ndarray, float]:
    return func()


@register_module("batch_correction_test")
class BatchCorrectionTestModule(BaseModule):
    """Benchmark module for batch correction performance testing."""

    def __init__(self, config: ModuleConfig) -> None:
        super().__init__(config)
        self._methods: list[str] = config.params.get("methods", ["combat", "mnn"])
        self._use_scib_metrics: bool = config.params.get("use_scib_metrics", True)
        self._n_pcs: int = config.params.get("n_pcs", 50)
        self._k_kbet: int = config.params.get("k_kbet", 25)
        self._random_state: int = config.params.get("random_state", 42)
        self._provider = get_provider()

    def run(self, dataset_name: str) -> list[ModuleResult]:
        if not self.should_process_dataset(dataset_name):
            return []

        results: list[ModuleResult] = []

        dataset_config = next((d for d in COMPARISON_DATASETS if d.name == dataset_name), None)
        if dataset_config is None:
            results.append(ModuleResult(
                module_name="batch_correction_test",
                dataset_name=dataset_name,
                method_name="load_data",
                success=False,
                error_message=f"Dataset '{dataset_name}' not found",
            ))
            return results

        X, M, batches, groups = self._provider.get_dataset(dataset_config)

        if len(np.unique(batches)) < 2:
            error_result = ModuleResult(
                module_name="batch_correction_test",
                dataset_name=dataset_name,
                method_name="validation",
                success=False,
                error_message="Dataset must have at least 2 batches",
            )
            results.append(error_result)
            self._add_result(error_result)
            return results

        method_map = {
            "combat": (self._test_combat, True),
            "harmony": (self._test_harmony, HARMONY_AVAILABLE),
            "mnn": (self._test_mnn, True),
            "scanorama": (self._test_scanorama, SCANORAMA_AVAILABLE),
        }

        for method in self._methods:
            if not self.should_process_method(method):
                continue
            if method not in method_map:
                results.append(ModuleResult(
                    module_name="batch_correction_test",
                    dataset_name=dataset_name,
                    method_name=method,
                    success=False,
                    error_message=f"Unknown method: {method}",
                ))
                continue
            test_fn, available = method_map[method]
            if not available:
                results.append(ModuleResult(
                    module_name="batch_correction_test",
                    dataset_name=dataset_name,
                    method_name=f"scptensor_{method}",
                    success=False,
                    error_message=f"{method} not installed",
                ))
                continue
            results.append(test_fn(dataset_name, X, M, batches, groups))

        for result in results:
            self._add_result(result)
        return results

    def _test_combat(self, dataset_name: str, X: np.ndarray, M: np.ndarray,
                     batches: np.ndarray, groups: np.ndarray) -> ModuleResult:
        try:
            container = _create_container(X, M, batch_labels=batches, group_labels=groups)
            start_time = time.time()
            result_container = integrate_combat(
                container, batch_key="batch", assay_name="protein",
                base_layer="raw", new_layer_name="combat",
            )
            runtime = time.time() - start_time
            X_corrected = _extract_result(result_container, "combat")
            metrics = self._compute_metrics(X, X_corrected, batches, groups)
            return ModuleResult(
                module_name="batch_correction_test", dataset_name=dataset_name,
                method_name="scptensor_combat", output=X_corrected,
                metrics=metrics, runtime_seconds=runtime, success=True,
            )
        except Exception as e:
            return ModuleResult(
                module_name="batch_correction_test", dataset_name=dataset_name,
                method_name="scptensor_combat", success=False, error_message=str(e),
            )

    def _test_harmony(self, dataset_name: str, X: np.ndarray, M: np.ndarray,
                      batches: np.ndarray, groups: np.ndarray) -> ModuleResult:
        if not HARMONY_AVAILABLE:
            return ModuleResult(
                module_name="batch_correction_test", dataset_name=dataset_name,
                method_name="scptensor_harmony", success=False,
                error_message="harmonypy not installed",
            )
        try:
            from scptensor.dim_reduction.pca import reduce_pca

            container = _create_container(X, M, batch_labels=batches, group_labels=groups)
            n_pcs_harmony = min(self._n_pcs, X.shape[1], X.shape[0] - 1)
            container = reduce_pca(
                container, "protein", "raw", "pca",
                n_components=n_pcs_harmony, random_state=self._random_state,
            )
            start_time = time.time()
            result_container = integrate_harmony(
                container, batch_key="batch", assay_name="protein",
                base_layer="pca", new_layer_name="harmony", theta=2.0, max_iter_harmony=10,
            )
            runtime = time.time() - start_time
            X_corrected = _extract_result(result_container, "harmony")
            metrics = self._compute_metrics(X[:, :n_pcs_harmony], X_corrected, batches, groups)
            return ModuleResult(
                module_name="batch_correction_test", dataset_name=dataset_name,
                method_name="scptensor_harmony", output=X_corrected,
                metrics=metrics, runtime_seconds=runtime, success=True,
            )
        except Exception as e:
            return ModuleResult(
                module_name="batch_correction_test", dataset_name=dataset_name,
                method_name="scptensor_harmony", success=False, error_message=str(e),
            )

    def _test_mnn(self, dataset_name: str, X: np.ndarray, M: np.ndarray,
                  batches: np.ndarray, groups: np.ndarray) -> ModuleResult:
        try:
            container = _create_container(X, M, batch_labels=batches, group_labels=groups)
            start_time = time.time()
            result_container = integrate_mnn(
                container, batch_key="batch", assay_name="protein",
                base_layer="raw", new_layer_name="mnn_corrected",
                k=20, sigma=1.0, n_pcs=self._n_pcs, use_pca=True,
            )
            runtime = time.time() - start_time
            X_corrected = _extract_result(result_container, "mnn_corrected")
            metrics = self._compute_metrics(X, X_corrected, batches, groups)
            return ModuleResult(
                module_name="batch_correction_test", dataset_name=dataset_name,
                method_name="scptensor_mnn", output=X_corrected,
                metrics=metrics, runtime_seconds=runtime, success=True,
            )
        except Exception as e:
            return ModuleResult(
                module_name="batch_correction_test", dataset_name=dataset_name,
                method_name="scptensor_mnn", success=False, error_message=str(e),
            )

    def _test_scanorama(self, dataset_name: str, X: np.ndarray, M: np.ndarray,
                        batches: np.ndarray, groups: np.ndarray) -> ModuleResult:
        if not SCANORAMA_AVAILABLE:
            return ModuleResult(
                module_name="batch_correction_test", dataset_name=dataset_name,
                method_name="scptensor_scanorama", success=False,
                error_message="scanorama not installed",
            )
        try:
            container = _create_container(X, M, batch_labels=batches, group_labels=groups)
            start_time = time.time()
            result_container = integrate_scanorama(
                container, batch_key="batch", assay_name="protein",
                base_layer="raw", new_layer_name="scanorama", sigma=15.0, alpha=0.1,
            )
            runtime = time.time() - start_time
            X_corrected = _extract_result(result_container, "scanorama")
            metrics = self._compute_metrics(X, X_corrected, batches, groups)
            return ModuleResult(
                module_name="batch_correction_test", dataset_name=dataset_name,
                method_name="scptensor_scanorama", output=X_corrected,
                metrics=metrics, runtime_seconds=runtime, success=True,
            )
        except Exception as e:
            return ModuleResult(
                module_name="batch_correction_test", dataset_name=dataset_name,
                method_name="scptensor_scanorama", success=False, error_message=str(e),
            )

    def _compute_metrics(self, X_orig: np.ndarray, X_corr: np.ndarray,
                         batches: np.ndarray, groups: np.ndarray) -> dict[str, float]:
        if self._use_scib_metrics and SCIB_METRICS_AVAILABLE:
            return self._scib_metrics(X_corr, batches, groups)
        return self._simple_metrics(X_orig, X_corr, batches, groups)

    def _scib_metrics(self, X: np.ndarray, batches: np.ndarray, groups: np.ndarray) -> dict[str, float]:
        metrics: dict[str, float] = {}
        try:
            import anndata as ad
            import pandas as pd

            obs_df = pd.DataFrame({"batch": batches.astype(str), "label": groups.astype(str)})
            adata = ad.AnnData(X=X, obs=obs_df)

            try:
                metrics["kbet"] = float(scib_metrics.metrics.kBET(
                    adata, batch_key="batch", label_key="label", type_="embed",
                    embed=None, k0=self._k_kbet, replicates=100, alpha=0.05, subsample=10000,
                ))
            except Exception:
                metrics["kbet"] = np.nan
            try:
                metrics["ilisi"] = float(scib_metrics.metrics.ilisi_graph(
                    adata, batch_key="batch", type_="embed", k0=self._k_kbet,
                    prng=np.random.default_rng(self._random_state), subsample=10000, scale=True,
                ))
            except Exception:
                metrics["ilisi"] = np.nan
            try:
                metrics["clisi"] = float(scib_metrics.metrics.clisi_graph(
                    adata, label_key="label", type_="embed", k0=self._k_kbet,
                    prng=np.random.default_rng(self._random_state), subsample=10000, scale=True,
                ))
            except Exception:
                metrics["clisi"] = np.nan
        except ImportError:
            pass
        return metrics

    def _simple_metrics(self, X_orig: np.ndarray, X_corr: np.ndarray,
                        batches: np.ndarray, groups: np.ndarray) -> dict[str, float]:
        metrics: dict[str, float] = {}
        limit = min(10000, len(X_corr))

        try:
            from sklearn.metrics import silhouette_score
            metrics["asw_batch"] = float(silhouette_score(X_corr[:limit], batches[:limit]))
            metrics["asw_label"] = float(silhouette_score(X_corr[:limit], groups[:limit]))
        except Exception:
            metrics["asw_batch"] = metrics["asw_label"] = np.nan

        try:
            metrics["pcr"] = self._pcr(X_corr, batches)
            metrics["batch_r2_reduction"] = self._batch_r2(X_orig, X_corr, batches)
        except Exception:
            metrics["pcr"] = metrics["batch_r2_reduction"] = np.nan

        return metrics

    def _pcr(self, X: np.ndarray, batches: np.ndarray) -> float:
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder

        n_pcs = min(self._n_pcs, X.shape[1], X.shape[0] - 1)
        X_pca = PCA(n_components=n_pcs, random_state=self._random_state).fit_transform(X)
        y = LabelEncoder().fit_transform(batches)
        return float(LogisticRegression(max_iter=1000, random_state=self._random_state).fit(X_pca, y).score(X_pca, y))

    def _batch_r2(self, X_orig: np.ndarray, X_corr: np.ndarray, batches: np.ndarray) -> float:
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder

        n_pcs = min(self._n_pcs, X_orig.shape[1], X_orig.shape[0] - 1)
        le = LabelEncoder()
        y = le.fit_transform(batches)
        clf = LogisticRegression(max_iter=1000, random_state=self._random_state)

        X_pca_orig = PCA(n_components=n_pcs, random_state=self._random_state).fit_transform(X_orig)
        r2_orig = clf.fit(X_pca_orig, y).score(X_pca_orig, y)

        X_pca_corr = PCA(n_components=n_pcs, random_state=self._random_state).fit_transform(X_corr)
        r2_corr = clf.fit(X_pca_corr, y).score(X_pca_corr, y)

        return (r2_orig - r2_corr) / r2_orig if r2_orig > 0 else 0.0


__all__ = ["BatchCorrectionTestModule"]
