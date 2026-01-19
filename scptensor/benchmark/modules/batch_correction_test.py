"""Batch correction performance test module for benchmarking.

This module provides comprehensive benchmarking of batch correction methods
including ComBat, Harmony, MNN, and Scanorama, evaluating their effectiveness
using biological and batch-specific metrics.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from scptensor.benchmark.data_provider import ComparisonDataset, get_provider
from scptensor.benchmark.modules.base import BaseModule, ModuleConfig, ModuleResult
from scptensor.benchmark.scptensor_adapter import _create_container, _extract_result
from scptensor.benchmark.utils.registry import register_module

if TYPE_CHECKING:
    from collections.abc import Callable


# Check for scib-metrics availability
try:
    import scib_metrics

    SCIB_METRICS_AVAILABLE = True
except ImportError:
    SCIB_METRICS_AVAILABLE = False

# Check for optional batch correction methods
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

from scptensor.integration import integrate_combat, integrate_mnn


def _time_function(func: Callable[[], tuple[np.ndarray, float]]) -> tuple[np.ndarray, float]:
    """Time a function execution.

    Parameters
    ----------
    func : Callable[[], tuple[np.ndarray, float]]
        Function to time that returns (result, runtime).

    Returns
    -------
    tuple[np.ndarray, float]
        (result, runtime) - Result and elapsed time in seconds.
    """
    return func()


@register_module("batch_correction_test")
class BatchCorrectionTestModule(BaseModule):
    """Benchmark module for batch correction performance testing.

    This module evaluates batch correction methods including:
    - ComBat (empirical Bayes correction)
    - Harmony (iterative clustering-based correction, if available)
    - MNN (Mutual Nearest Neighbors correction)
    - Scanorama (efficient large-scale integration, if available)

    For each method, it computes:
    - kBET (k-nearest neighbour batch effect test) - measures batch mixing
    - iLISI (inverse label Simpson's diversity) - measures batch mixing quality
    - cLISI (cell label Simpson's diversity) - measures biological cluster preservation
    - ASW (Average Silhouette Width) for batch separation
    - PCR (Principal Component Regression) for batch effect strength

    Examples
    --------
    >>> from scptensor.benchmark.modules.base import ModuleConfig
    >>> from scptensor.benchmark.modules.batch_correction_test import BatchCorrectionTestModule
    >>> config = ModuleConfig(
    ...     name="batch_correction_test",
    ...     datasets=["synthetic_batch"],
    ...     params={"methods": ["combat", "mnn"], "n_pcs": 50}
    ... )
    >>> module = BatchCorrectionTestModule(config)
    >>> results = module.run("synthetic_batch")
    """

    def __init__(self, config: ModuleConfig) -> None:
        """Initialize the batch correction test module.

        Parameters
        ----------
        config : ModuleConfig
            Configuration object with module parameters:
            - methods: list[str] - Methods to test ["combat", "harmony", "mnn", "scanorama"]
            - use_scib_metrics: bool - Use scib-metrics for evaluation (default True)
            - n_pcs: int - Number of PCs for evaluation (default 50)
            - k_kbet: int - k parameter for kBET (default 25)
            - random_state: int - Random seed (default 42)
        """
        super().__init__(config)

        # Extract parameters with defaults
        self._methods: list[str] = config.params.get("methods", ["combat", "mnn"])
        self._use_scib_metrics: bool = config.params.get("use_scib_metrics", True)
        self._n_pcs: int = config.params.get("n_pcs", 50)
        self._k_kbet: int = config.params.get("k_kbet", 25)
        self._random_state: int = config.params.get("random_state", 42)

        # Data provider for dataset access
        self._provider = get_provider()

    def run(self, dataset_name: str) -> list[ModuleResult]:
        """Execute batch correction benchmark tests on a dataset.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset to process.

        Returns
        -------
        list[ModuleResult]
            List of benchmark results, one for each method tested.
        """
        if not self.should_process_dataset(dataset_name):
            return []

        results: list[ModuleResult] = []

        # Get dataset configuration
        dataset_config = self._get_dataset_config(dataset_name)
        if dataset_config is None:
            results.append(
                ModuleResult(
                    module_name="batch_correction_test",
                    dataset_name=dataset_name,
                    method_name="load_data",
                    success=False,
                    error_message=f"Dataset '{dataset_name}' not found",
                )
            )
            return results

        # Load data
        X, M, batches, groups = self._provider.get_dataset(dataset_config)

        # Skip if no batches
        if len(np.unique(batches)) < 2:
            error_result = ModuleResult(
                module_name="batch_correction_test",
                dataset_name=dataset_name,
                method_name="validation",
                success=False,
                error_message="Dataset must have at least 2 batches for batch correction testing",
            )
            results.append(error_result)
            self._add_result(error_result)
            return results

        # Run tests for each method
        for method in self._methods:
            if not self.should_process_method(method):
                continue

            if method == "combat":
                result = self._test_combat(dataset_name, X, M, batches, groups)
            elif method == "harmony":
                result = self._test_harmony(dataset_name, X, M, batches, groups)
            elif method == "mnn":
                result = self._test_mnn(dataset_name, X, M, batches, groups)
            elif method == "scanorama":
                result = self._test_scanorama(dataset_name, X, M, batches, groups)
            else:
                results.append(
                    ModuleResult(
                        module_name="batch_correction_test",
                        dataset_name=dataset_name,
                        method_name=method,
                        success=False,
                        error_message=f"Unknown method: {method}",
                    )
                )
                continue

            results.append(result)

        # Store results
        for result in results:
            self._add_result(result)

        return results

    def _get_dataset_config(self, dataset_name: str) -> ComparisonDataset | None:
        """Get dataset configuration by name.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset.

        Returns
        -------
        ComparisonDataset | None
            Dataset configuration or None if not found.
        """
        from scptensor.benchmark.data_provider import COMPARISON_DATASETS

        for dataset in COMPARISON_DATASETS:
            if dataset.name == dataset_name:
                return dataset
        return None

    def _test_combat(
        self,
        dataset_name: str,
        X: np.ndarray,
        M: np.ndarray,
        batches: np.ndarray,
        groups: np.ndarray,
    ) -> ModuleResult:
        """Test ComBat batch correction.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset.
        X : np.ndarray
            Data matrix.
        M : np.ndarray
            Mask matrix.
        batches : np.ndarray
            Batch labels.
        groups : np.ndarray
            True group labels.

        Returns
        -------
        ModuleResult
            Benchmark result.
        """
        try:
            # Create container
            container = _create_container(X, M, batch_labels=batches, group_labels=groups)

            # Apply ComBat correction
            start_time = time.time()
            result_container = integrate_combat(
                container,
                batch_key="batch",
                assay_name="protein",
                base_layer="raw",
                new_layer_name="combat",
            )
            runtime = time.time() - start_time

            # Extract corrected data
            X_corrected = _extract_result(result_container, "combat")

            # Compute metrics
            metrics = self._compute_batch_correction_metrics(
                X, X_corrected, batches, groups, dataset_name
            )

            return ModuleResult(
                module_name="batch_correction_test",
                dataset_name=dataset_name,
                method_name="scptensor_combat",
                output=X_corrected,
                metrics=metrics,
                runtime_seconds=runtime,
                success=True,
            )

        except Exception as e:
            return ModuleResult(
                module_name="batch_correction_test",
                dataset_name=dataset_name,
                method_name="scptensor_combat",
                success=False,
                error_message=str(e),
            )

    def _test_harmony(
        self,
        dataset_name: str,
        X: np.ndarray,
        M: np.ndarray,
        batches: np.ndarray,
        groups: np.ndarray,
    ) -> ModuleResult:
        """Test Harmony batch correction.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset.
        X : np.ndarray
            Data matrix.
        M : np.ndarray
            Mask matrix.
        batches : np.ndarray
            Batch labels.
        groups : np.ndarray
            True group labels.

        Returns
        -------
        ModuleResult
            Benchmark result.
        """
        if not HARMONY_AVAILABLE:
            return ModuleResult(
                module_name="batch_correction_test",
                dataset_name=dataset_name,
                method_name="scptensor_harmony",
                success=False,
                error_message="harmonypy not installed. Install with: pip install harmonypy",
            )

        try:
            from scptensor.dim_reduction.pca import reduce_pca

            # Create container
            container = _create_container(X, M, batch_labels=batches, group_labels=groups)

            # First compute PCA for Harmony
            n_pcs_harmony = min(self._n_pcs, X.shape[1], X.shape[0] - 1)
            container = reduce_pca(
                container, "protein", "raw", "pca", n_components=n_pcs_harmony, random_state=self._random_state
            )

            # Apply Harmony correction
            start_time = time.time()
            result_container = integrate_harmony(
                container,
                batch_key="batch",
                assay_name="protein",
                base_layer="pca",
                new_layer_name="harmony",
                theta=2.0,
                max_iter_harmony=10,
            )
            runtime = time.time() - start_time

            # Extract corrected data
            X_corrected = _extract_result(result_container, "harmony")

            # Compute metrics
            metrics = self._compute_batch_correction_metrics(
                X[:, :n_pcs_harmony], X_corrected, batches, groups, dataset_name
            )

            return ModuleResult(
                module_name="batch_correction_test",
                dataset_name=dataset_name,
                method_name="scptensor_harmony",
                output=X_corrected,
                metrics=metrics,
                runtime_seconds=runtime,
                success=True,
            )

        except Exception as e:
            return ModuleResult(
                module_name="batch_correction_test",
                dataset_name=dataset_name,
                method_name="scptensor_harmony",
                success=False,
                error_message=str(e),
            )

    def _test_mnn(
        self,
        dataset_name: str,
        X: np.ndarray,
        M: np.ndarray,
        batches: np.ndarray,
        groups: np.ndarray,
    ) -> ModuleResult:
        """Test MNN batch correction.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset.
        X : np.ndarray
            Data matrix.
        M : np.ndarray
            Mask matrix.
        batches : np.ndarray
            Batch labels.
        groups : np.ndarray
            True group labels.

        Returns
        -------
        ModuleResult
            Benchmark result.
        """
        try:
            # Create container
            container = _create_container(X, M, batch_labels=batches, group_labels=groups)

            # Apply MNN correction
            start_time = time.time()
            result_container = integrate_mnn(
                container,
                batch_key="batch",
                assay_name="protein",
                base_layer="raw",
                new_layer_name="mnn_corrected",
                k=20,
                sigma=1.0,
                n_pcs=self._n_pcs,
                use_pca=True,
            )
            runtime = time.time() - start_time

            # Extract corrected data
            X_corrected = _extract_result(result_container, "mnn_corrected")

            # Compute metrics
            metrics = self._compute_batch_correction_metrics(
                X, X_corrected, batches, groups, dataset_name
            )

            return ModuleResult(
                module_name="batch_correction_test",
                dataset_name=dataset_name,
                method_name="scptensor_mnn",
                output=X_corrected,
                metrics=metrics,
                runtime_seconds=runtime,
                success=True,
            )

        except Exception as e:
            return ModuleResult(
                module_name="batch_correction_test",
                dataset_name=dataset_name,
                method_name="scptensor_mnn",
                success=False,
                error_message=str(e),
            )

    def _test_scanorama(
        self,
        dataset_name: str,
        X: np.ndarray,
        M: np.ndarray,
        batches: np.ndarray,
        groups: np.ndarray,
    ) -> ModuleResult:
        """Test Scanorama batch correction.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset.
        X : np.ndarray
            Data matrix.
        M : np.ndarray
            Mask matrix.
        batches : np.ndarray
            Batch labels.
        groups : np.ndarray
            True group labels.

        Returns
        -------
        ModuleResult
            Benchmark result.
        """
        if not SCANORAMA_AVAILABLE:
            return ModuleResult(
                module_name="batch_correction_test",
                dataset_name=dataset_name,
                method_name="scptensor_scanorama",
                success=False,
                error_message="scanorama not installed. Install with: pip install scanorama",
            )

        try:
            # Create container
            container = _create_container(X, M, batch_labels=batches, group_labels=groups)

            # Apply Scanorama correction
            start_time = time.time()
            result_container = integrate_scanorama(
                container,
                batch_key="batch",
                assay_name="protein",
                base_layer="raw",
                new_layer_name="scanorama",
                sigma=15.0,
                alpha=0.1,
            )
            runtime = time.time() - start_time

            # Extract corrected data
            X_corrected = _extract_result(result_container, "scanorama")

            # Compute metrics
            metrics = self._compute_batch_correction_metrics(
                X, X_corrected, batches, groups, dataset_name
            )

            return ModuleResult(
                module_name="batch_correction_test",
                dataset_name=dataset_name,
                method_name="scptensor_scanorama",
                output=X_corrected,
                metrics=metrics,
                runtime_seconds=runtime,
                success=True,
            )

        except Exception as e:
            return ModuleResult(
                module_name="batch_correction_test",
                dataset_name=dataset_name,
                method_name="scptensor_scanorama",
                success=False,
                error_message=str(e),
            )

    def _compute_batch_correction_metrics(
        self,
        X_original: np.ndarray,
        X_corrected: np.ndarray,
        batches: np.ndarray,
        groups: np.ndarray,
        dataset_name: str,
    ) -> dict[str, float]:
        """Compute batch correction evaluation metrics.

        Parameters
        ----------
        X_original : np.ndarray
            Original data matrix.
        X_corrected : np.ndarray
            Corrected data matrix.
        batches : np.ndarray
            Batch labels.
        groups : np.ndarray
            True group labels.
        dataset_name : str
            Dataset name for logging.

        Returns
        -------
        dict[str, float]
            Dictionary of metric names and values.
        """
        metrics: dict[str, float] = {}

        # Use scib-metrics if available and requested
        if self._use_scib_metrics and SCIB_METRICS_AVAILABLE:
            scib_metrics = self._compute_scib_metrics(X_corrected, batches, groups)
            metrics.update(scib_metrics)
        else:
            # Fallback to simple metrics
            simple_metrics = self._compute_simple_metrics(X_original, X_corrected, batches, groups)
            metrics.update(simple_metrics)

        return metrics

    def _compute_scib_metrics(
        self,
        X_corrected: np.ndarray,
        batches: np.ndarray,
        groups: np.ndarray,
    ) -> dict[str, float]:
        """Compute metrics using scib-metrics.

        Parameters
        ----------
        X_corrected : np.ndarray
            Corrected data matrix.
        batches : np.ndarray
            Batch labels.
        groups : np.ndarray
            True group labels.

        Returns
        -------
        dict[str, float]
            Dictionary of scib-metrics values.
        """
        result_metrics: dict[str, float] = {}

        try:
            import anndata as ad
            import pandas as pd

            # Create AnnData object for scib-metrics
            obs_df = pd.DataFrame({"batch": batches.astype(str), "label": groups.astype(str)})
            adata = ad.AnnData(X=X_corrected, obs=obs_df)

            # Compute kBET (higher is better, measures batch mixing)
            try:
                kbet_result = scib_metrics.metrics.kBET(
                    adata,
                    batch_key="batch",
                    label_key="label",
                    type_="embed",
                    embed=None,
                    k0=self._k_kbet,
                    replicates=100,
                    alpha=0.05,
                    subsample=10000,
                )
                result_metrics["kbet"] = float(kbet_result)
            except Exception:
                result_metrics["kbet"] = np.nan

            # Compute iLISI (higher is better, measures batch mixing quality)
            try:
                ilisi_result = scib_metrics.metrics.ilisi_graph(
                    adata,
                    batch_key="batch",
                    type_="embed",
                    k0=self._k_kbet,
                    prng=np.random.default_rng(self._random_state),
                    subsample=10000,
                    scale=True,
                )
                result_metrics["ilisi"] = float(ilisi_result)
            except Exception:
                result_metrics["ilisi"] = np.nan

            # Compute cLISI (higher is better, measures biological cluster preservation)
            try:
                clisi_result = scib_metrics.metrics.clisi_graph(
                    adata,
                    label_key="label",
                    type_="embed",
                    k0=self._k_kbet,
                    prng=np.random.default_rng(self._random_state),
                    subsample=10000,
                    scale=True,
                )
                result_metrics["clisi"] = float(clisi_result)
            except Exception:
                result_metrics["clisi"] = np.nan

        except ImportError:
            # Fall back to simple metrics if anndata not available
            pass

        return result_metrics

    def _compute_simple_metrics(
        self,
        X_original: np.ndarray,
        X_corrected: np.ndarray,
        batches: np.ndarray,
        groups: np.ndarray,
    ) -> dict[str, float]:
        """Compute simple batch correction metrics without scib-metrics.

        Parameters
        ----------
        X_original : np.ndarray
            Original data matrix.
        X_corrected : np.ndarray
            Corrected data matrix.
        batches : np.ndarray
            Batch labels.
        groups : np.ndarray
            True group labels.

        Returns
        -------
        dict[str, float]
            Dictionary of metric values.
        """
        metrics: dict[str, float] = {}

        # Compute ASW for batch separation (lower is better - want batches mixed)
        try:
            from sklearn.metrics import silhouette_score

            asw_batch = silhouette_score(
                X_corrected[:10000] if X_corrected.shape[0] > 10000 else X_corrected,
                batches[:10000] if X_corrected.shape[0] > 10000 else batches,
            )
            metrics["asw_batch"] = float(asw_batch)
        except Exception:
            metrics["asw_batch"] = np.nan

        # Compute ASW for biological groups (higher is better - want groups separated)
        try:
            from sklearn.metrics import silhouette_score

            asw_label = silhouette_score(
                X_corrected[:10000] if X_corrected.shape[0] > 10000 else X_corrected,
                groups[:10000] if X_corrected.shape[0] > 10000 else groups,
            )
            metrics["asw_label"] = float(asw_label)
        except Exception:
            metrics["asw_label"] = np.nan

        # Compute PCR (Principal Component Regression) for batch effect
        try:
            pcr_score = self._compute_pcr(X_corrected, batches)
            metrics["pcr"] = float(pcr_score)
        except Exception:
            metrics["pcr"] = np.nan

        # Compute batch effect strength (R2)
        try:
            batch_r2 = self._compute_batch_r2(X_original, X_corrected, batches)
            metrics["batch_r2_reduction"] = float(batch_r2)
        except Exception:
            metrics["batch_r2_reduction"] = np.nan

        return metrics

    def _compute_pcr(self, X: np.ndarray, batches: np.ndarray) -> float:
        """Compute Principal Component Regression for batch effect.

        Lower values indicate less batch effect.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.
        batches : np.ndarray
            Batch labels.

        Returns
        -------
        float
            PCR score (R2 of batch labels on PCs).
        """
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder

        # Compute PCA
        n_pcs = min(self._n_pcs, X.shape[1], X.shape[0] - 1)
        pca = PCA(n_components=n_pcs, random_state=self._random_state)
        X_pca = pca.fit_transform(X)

        # Predict batch from PCs
        le = LabelEncoder()
        y = le.fit_transform(batches)

        clf = LogisticRegression(max_iter=1000, random_state=self._random_state)
        clf.fit(X_pca, y)
        score = clf.score(X_pca, y)

        return float(score)

    def _compute_batch_r2(
        self,
        X_original: np.ndarray,
        X_corrected: np.ndarray,
        batches: np.ndarray,
    ) -> float:
        """Compute reduction in batch effect R2.

        Higher values indicate more batch effect removed.

        Parameters
        ----------
        X_original : np.ndarray
            Original data.
        X_corrected : np.ndarray
            Corrected data.
        batches : np.ndarray
            Batch labels.

        Returns
        -------
        float
            Proportion of batch R2 removed.
        """
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder

        n_pcs = min(self._n_pcs, X_original.shape[1], X_original.shape[0] - 1)

        # Original R2
        pca_orig = PCA(n_components=n_pcs, random_state=self._random_state)
        X_pca_orig = pca_orig.fit_transform(X_original)

        le = LabelEncoder()
        y = le.fit_transform(batches)

        clf = LogisticRegression(max_iter=1000, random_state=self._random_state)
        clf.fit(X_pca_orig, y)
        r2_original = clf.score(X_pca_orig, y)

        # Corrected R2
        pca_corr = PCA(n_components=n_pcs, random_state=self._random_state)
        X_pca_corr = pca_corr.fit_transform(X_corrected)

        clf_corr = LogisticRegression(max_iter=1000, random_state=self._random_state)
        clf_corr.fit(X_pca_corr, y)
        r2_corrected = clf_corr.score(X_pca_corr, y)

        # Proportion of R2 removed
        if r2_original > 0:
            return (r2_original - r2_corrected) / r2_original
        return 0.0


__all__ = ["BatchCorrectionTestModule"]
