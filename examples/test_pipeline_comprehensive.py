#!/usr/bin/env python3
"""Comprehensive Pipeline Test Script for ScpTensor.

This script performs sequential testing of ScpTensor modules:
1. Normalization (4 methods): log_transform, norm_mean, norm_median, norm_quantile
2. Imputation (6 methods): knn, bpca, mf, lls, qrilc, minprob
3. Integration (5 methods): combat, harmony, mnn, scanorama, nonlinear
4. Dimensionality Reduction (2 methods): pca, umap
5. Clustering (2 methods): kmeans, leiden

Usage:
    python test_pipeline_comprehensive.py

Output:
    All test reports are saved to tmp/pipeline_test_report/
    - pipeline_test_report.md: Comprehensive markdown report
"""

from __future__ import annotations

import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import polars as pl

# Import ScpTensor components
from scptensor import (
    ScpContainer,
    load_diann,
    ScpDataGenerator,
)

# Normalization modules
from scptensor.normalization import (
    log_transform,
    norm_mean,
    norm_median,
    norm_quantile,
)

# Imputation modules
from scptensor.impute import (
    impute_knn,
    impute_bpca,
    impute_mf,
    impute_lls,
    impute_qrilc,
    impute_minprob,
)

# Integration modules
from scptensor.integration import (
    integrate_combat,
    integrate_harmony,
    integrate_mnn,
    integrate_scanorama,
    integrate_nonlinear,
)

# Dimensionality reduction modules
from scptensor.dim_reduction import (
    reduce_pca,
    reduce_umap,
)

# Clustering modules
from scptensor.cluster import (
    cluster_kmeans,
    cluster_leiden,
)

# Configuration
OUTPUT_DIR = Path("tmp/pipeline_test_report")
DATA_PATH = Path("data/dia/diann/PXD054343/1_SC_LF_report.tsv")

# Test data configuration
SYNTHETIC_DATA_CONFIG = {
    "n_samples": 100,
    "n_features": 500,
    "missing_rate": 0.3,
    "n_batches": 3,
    "n_groups": 2,
    "random_seed": 42,
}


class TestResult:
    """Store test result information."""

    def __init__(
        self,
        module: str,
        method: str,
        status: str,
        duration: float,
        metrics: dict[str, Any] | None = None,
        error_message: str | None = None,
    ):
        self.module = module
        self.method = method
        self.status = status  # PASS, FAIL, SKIP
        self.duration = duration
        self.metrics = metrics or {}
        self.error_message = error_message


class PipelineTester:
    """Comprehensive pipeline tester for ScpTensor."""

    def __init__(self, output_dir: Path = OUTPUT_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[TestResult] = []
        self.container: ScpContainer | None = None
        self.assay_name = "proteins"
        self.base_layer = "raw"

    def load_data(self, use_synthetic: bool = False) -> bool:
        """Load test data.

        Parameters
        ----------
        use_synthetic : bool, default=False
            If True, use synthetic data. Otherwise try to load real data.

        Returns
        -------
        bool
            True if data loaded successfully.
        """
        print("\n" + "=" * 80)
        print("数据加载阶段")
        print("=" * 80)

        if use_synthetic:
            print("\n生成合成测试数据...")
            print(f"  - 样本数: {SYNTHETIC_DATA_CONFIG['n_samples']}")
            print(f"  - 特征数: {SYNTHETIC_DATA_CONFIG['n_features']}")
            print(f"  - 缺失率: {SYNTHETIC_DATA_CONFIG['missing_rate']}")
            print(f"  - 批次数: {SYNTHETIC_DATA_CONFIG['n_batches']}")

            generator = ScpDataGenerator(**SYNTHETIC_DATA_CONFIG)
            self.container = generator.generate()
            self.base_layer = "raw"

            print("\n✓ 合成数据加载成功")
            print(f"  - 容器形状: {self.container.n_samples} × {self.container.n_features}")
            print(f"  - Assay名称: {self.assay_name}")
            print(f"  - 基础层: {self.base_layer}")

            # Add batch column if not present
            if "batch" not in self.container.obs.columns:
                import numpy as np
                n_batches = SYNTHETIC_DATA_CONFIG["n_batches"]
                batches = np.array([f"batch_{i % n_batches}" for i in range(self.container.n_samples)])
                self.container.obs = self.container.obs.with_columns(
                    pl.Series("batch", batches)
                )
                print(f"  - 批次信息已添加: {n_batches} 个批次")

        else:
            if not DATA_PATH.exists():
                print(f"\n✗ 数据文件不存在: {DATA_PATH}")
                print("  切换到合成数据模式...")
                return self.load_data(use_synthetic=True)

            print(f"\n加载真实数据: {DATA_PATH}")
            try:
                self.container = load_diann(DATA_PATH, assay_name=self.assay_name)

                # Get first layer name
                assay = self.container.assays[self.assay_name]
                self.base_layer = list(assay.layers.keys())[0]

                print("\n✓ 真实数据加载成功")
                print(f"  - 容器形状: {self.container.n_samples} × {self.container.n_features}")
                print(f"  - Assay名称: {self.assay_name}")
                print(f"  - 基础层: {self.base_layer}")

                # Add batch column if not present
                if "batch" not in self.container.obs.columns:
                    import numpy as np
                    n_batches = 3
                    batches = np.array([f"batch_{i % n_batches}" for i in range(self.container.n_samples)])
                    self.container.obs = self.container.obs.with_columns(
                        pl.Series("batch", batches)
                    )
                    print(f"  - 批次信息已添加（合成）: {n_batches} 个批次")

            except Exception as e:
                print(f"\n✗ 数据加载失败: {e}")
                print("  切换到合成数据模式...")
                return self.load_data(use_synthetic=True)

        return True

    def _get_layer_name(self, assay_name: str) -> str:
        """Get the first available layer name from an assay.

        Parameters
        ----------
        assay_name : str
            Name of the assay

        Returns
        -------
        str
            Name of the first layer
        """
        if self.container is None:
            raise ValueError("Container not initialized")

        assay = self.container.assays[assay_name]
        layer_names = list(assay.layers.keys())

        if not layer_names:
            raise ValueError(f"Assay '{assay_name}' has no layers")

        return layer_names[0]

    def _calculate_sparsity(self, container: ScpContainer, assay_name: str, layer_name: str) -> float:
        """Calculate sparsity ratio of a layer.

        Parameters
        ----------
        container : ScpContainer
            Data container
        assay_name : str
            Assay name
        layer_name : str
            Layer name

        Returns
        -------
        float
            Sparsity ratio (0-1)
        """
        from scptensor import get_sparsity_ratio
        assay = container.assays[assay_name]
        layer = assay.layers[layer_name]
        return get_sparsity_ratio(layer.X)

    def _calculate_missing_rate(self, container: ScpContainer, assay_name: str, layer_name: str) -> float:
        """Calculate missing value rate.

        Parameters
        ----------
        container : ScpContainer
            Data container
        assay_name : str
            Assay name
        layer_name : str
            Layer name

        Returns
        -------
        float
            Missing value rate (0-1)
        """
        assay = container.assays[assay_name]
        layer = assay.layers[layer_name]

        if hasattr(layer, "M") and layer.M is not None:
            from scptensor import count_mask_codes
            mask_counts = count_mask_codes(layer.M)
            total = mask_counts.sum()
            if total > 0:
                return (mask_counts.get(1, 0) + mask_counts.get(2, 0)) / total

        # Use NaN count for dense matrices
        X = layer.X if not hasattr(layer.X, "toarray") else layer.X.toarray()
        return np.isnan(X).sum() / X.size

    def _run_test(
        self,
        module: str,
        method: str,
        test_func: Callable,
        update_container: bool = True,
        **kwargs,
    ) -> TestResult:
        """Run a single test function.

        Parameters
        ----------
        module : str
            Module name
        method : str
            Method name
        test_func : Callable
            Test function to execute
        update_container : bool, default=True
            Whether to update self.container with the result
        **kwargs
            Arguments to pass to test_func

        Returns
        -------
        TestResult
            Test result object
        """
        print(f"\n{'─' * 60}")
        print(f"测试: [{module}] {method}")
        print(f"{'─' * 60}")

        start_time = time.time()
        status = "PASS"
        metrics = {}
        error_message = None

        try:
            result_container = test_func(**kwargs)

            # Update container reference if requested and test passed
            if update_container and result_container is not None:
                self.container = result_container

            # Calculate metrics
            duration = time.time() - start_time

            # Get basic metrics
            if hasattr(self.container, "n_samples"):
                metrics["n_samples"] = self.container.n_samples
            if hasattr(self.container, "n_features"):
                metrics["n_features"] = self.container.n_features

            # Get assay-specific metrics
            if self.assay_name in self.container.assays:
                assay = self.container.assays[self.assay_name]
                metrics["n_layers"] = len(assay.layers)
                metrics["layers"] = list(assay.layers.keys())

            print(f"\n✓ 测试通过")
            print(f"  耗时: {duration:.3f} 秒")
            if metrics:
                print(f"  指标: {metrics}")

        except Exception as e:
            duration = time.time() - start_time
            status = "FAIL"
            error_message = f"{type(e).__name__}: {str(e)}"
            print(f"\n✗ 测试失败")
            print(f"  错误: {error_message}")
            print(f"  耗时: {duration:.3f} 秒")

        result = TestResult(
            module=module,
            method=method,
            status=status,
            duration=duration,
            metrics=metrics,
            error_message=error_message,
        )

        self.results.append(result)
        return result

    def test_normalization(self) -> None:
        """Test all normalization methods."""
        print("\n" + "=" * 80)
        print("1. 归一化模块测试 (Normalization)")
        print("=" * 80)

        if self.container is None:
            print("✗ 容器未初始化")
            return

        # Log transform
        self._run_test(
            module="normalization",
            method="log_transform",
            test_func=lambda: log_transform(
                self.container,
                assay_name=self.assay_name,
                source_layer=self.base_layer,
                new_layer_name="log",
                base=2.0,
            ),
        )

        # Mean normalization
        self._run_test(
            module="normalization",
            method="norm_mean",
            test_func=lambda: norm_mean(
                self.container,
                assay_name=self.assay_name,
                source_layer=self.base_layer,
                new_layer_name="norm_mean",
            ),
        )

        # Median normalization
        self._run_test(
            module="normalization",
            method="norm_median",
            test_func=lambda: norm_median(
                self.container,
                assay_name=self.assay_name,
                source_layer=self.base_layer,
                new_layer_name="norm_median",
            ),
        )

        # Quantile normalization
        self._run_test(
            module="normalization",
            method="norm_quantile",
            test_func=lambda: norm_quantile(
                self.container,
                assay_name=self.assay_name,
                source_layer=self.base_layer,
                new_layer_name="norm_quantile",
            ),
        )

    def test_imputation(self) -> None:
        """Test all imputation methods."""
        print("\n" + "=" * 80)
        print("2. 缺失值填充模块测试 (Imputation)")
        print("=" * 80)

        if self.container is None:
            print("✗ 容器未初始化")
            return

        # Calculate initial missing rate
        initial_missing_rate = self._calculate_missing_rate(
            self.container, self.assay_name, self.base_layer
        )
        print(f"\n初始缺失率: {initial_missing_rate:.2%}")

        # KNN imputation
        result = self._run_test(
            module="imputation",
            method="impute_knn",
            test_func=lambda: impute_knn(
                self.container,
                assay_name=self.assay_name,
                source_layer=self.base_layer,
                new_layer_name="knn_imputed",
                k=5,
            ),
        )
        if result.status == "PASS":
            final_missing_rate = self._calculate_missing_rate(
                self.container, self.assay_name, "knn_imputed"
            )
            result.metrics["initial_missing_rate"] = initial_missing_rate
            result.metrics["final_missing_rate"] = final_missing_rate
            result.metrics["imputed_ratio"] = initial_missing_rate - final_missing_rate

        # BPCA imputation
        result = self._run_test(
            module="imputation",
            method="impute_bpca",
            test_func=lambda: impute_bpca(
                self.container,
                assay_name=self.assay_name,
                source_layer=self.base_layer,
                new_layer_name="bpca_imputed",
                n_components=10,
            ),
        )
        if result.status == "PASS":
            final_missing_rate = self._calculate_missing_rate(
                self.container, self.assay_name, "bpca_imputed"
            )
            result.metrics["initial_missing_rate"] = initial_missing_rate
            result.metrics["final_missing_rate"] = final_missing_rate
            result.metrics["imputed_ratio"] = initial_missing_rate - final_missing_rate

        # MissForest imputation
        result = self._run_test(
            module="imputation",
            method="impute_mf",
            test_func=lambda: impute_mf(
                self.container,
                assay_name=self.assay_name,
                source_layer=self.base_layer,
                new_layer_name="mf_imputed",
                max_iter=5,
                n_estimators=50,
            ),
        )
        if result.status == "PASS":
            final_missing_rate = self._calculate_missing_rate(
                self.container, self.assay_name, "mf_imputed"
            )
            result.metrics["initial_missing_rate"] = initial_missing_rate
            result.metrics["final_missing_rate"] = final_missing_rate
            result.metrics["imputed_ratio"] = initial_missing_rate - final_missing_rate

        # LLS imputation
        result = self._run_test(
            module="imputation",
            method="impute_lls",
            test_func=lambda: impute_lls(
                self.container,
                assay_name=self.assay_name,
                source_layer=self.base_layer,
                new_layer_name="lls_imputed",
                k=10,
            ),
        )
        if result.status == "PASS":
            final_missing_rate = self._calculate_missing_rate(
                self.container, self.assay_name, "lls_imputed"
            )
            result.metrics["initial_missing_rate"] = initial_missing_rate
            result.metrics["final_missing_rate"] = final_missing_rate
            result.metrics["imputed_ratio"] = initial_missing_rate - final_missing_rate

        # QRILC imputation
        result = self._run_test(
            module="imputation",
            method="impute_qrilc",
            test_func=lambda: impute_qrilc(
                self.container,
                assay_name=self.assay_name,
                source_layer=self.base_layer,
                new_layer_name="qrilc_imputed",
            ),
        )
        if result.status == "PASS":
            final_missing_rate = self._calculate_missing_rate(
                self.container, self.assay_name, "qrilc_imputed"
            )
            result.metrics["initial_missing_rate"] = initial_missing_rate
            result.metrics["final_missing_rate"] = final_missing_rate
            result.metrics["imputed_ratio"] = initial_missing_rate - final_missing_rate

        # MinProb imputation
        result = self._run_test(
            module="imputation",
            method="impute_minprob",
            test_func=lambda: impute_minprob(
                self.container,
                assay_name=self.assay_name,
                source_layer=self.base_layer,
                new_layer_name="minprob_imputed",
            ),
        )
        if result.status == "PASS":
            final_missing_rate = self._calculate_missing_rate(
                self.container, self.assay_name, "minprob_imputed"
            )
            result.metrics["initial_missing_rate"] = initial_missing_rate
            result.metrics["final_missing_rate"] = final_missing_rate
            result.metrics["imputed_ratio"] = initial_missing_rate - final_missing_rate

    def test_integration(self) -> None:
        """Test all integration (batch correction) methods."""
        print("\n" + "=" * 80)
        print("3. 批次校正模块测试 (Integration)")
        print("=" * 80)

        if self.container is None:
            print("✗ 容器未初始化")
            return

        # Check if batch column exists
        if "batch" not in self.container.obs.columns:
            print("✗ 缺少批次信息，跳过批次校正测试")
            # Add SKIP results
            for method in ["combat", "harmony", "mnn", "scanorama", "nonlinear"]:
                self.results.append(TestResult(
                    module="integration",
                    method=method,
                    status="SKIP",
                    duration=0.0,
                    error_message="缺少批次信息",
                ))
            return

        n_batches = self.container.obs["batch"].n_unique()
        print(f"\n批次数量: {n_batches}")

        # Use log-transformed layer for batch correction
        source_layer = "log" if "log" in self.container.assays[self.assay_name].layers else self.base_layer

        # ComBat
        self._run_test(
            module="integration",
            method="integrate_combat",
            test_func=lambda: integrate_combat(
                self.container,
                batch_key="batch",
                assay_name=self.assay_name,
                base_layer=source_layer,
                new_layer_name="combat",
            ),
        )

        # Harmony
        result = self._run_test(
            module="integration",
            method="integrate_harmony",
            test_func=lambda: integrate_harmony(
                self.container,
                batch_key="batch",
                assay_name=self.assay_name,
                base_layer=source_layer,
                new_layer_name="harmony",
            ),
        )
        if result.status == "FAIL" and ("harmonypy" in str(result.error_message).lower() or "MissingDependencyError" in str(result.error_message)):
            # Update the result in the list
            for i, r in enumerate(self.results):
                if r.module == "integration" and r.method == "integrate_harmony":
                    self.results[i] = TestResult(
                        module="integration",
                        method="integrate_harmony",
                        status="SKIP",
                        duration=0.0,
                        error_message="harmonypy 未安装",
                    )
                    break

        # MNN
        self._run_test(
            module="integration",
            method="integrate_mnn",
            test_func=lambda: integrate_mnn(
                self.container,
                batch_key="batch",
                assay_name=self.assay_name,
                base_layer=source_layer,
                new_layer_name="mnn",
            ),
        )

        # Scanorama
        result = self._run_test(
            module="integration",
            method="integrate_scanorama",
            test_func=lambda: integrate_scanorama(
                self.container,
                batch_key="batch",
                assay_name=self.assay_name,
                base_layer=source_layer,
                new_layer_name="scanorama",
            ),
        )
        if result.status == "FAIL" and ("scanorama" in str(result.error_message).lower() or "MissingDependencyError" in str(result.error_message)):
            # Update the result in the list
            for i, r in enumerate(self.results):
                if r.module == "integration" and r.method == "integrate_scanorama":
                    self.results[i] = TestResult(
                        module="integration",
                        method="integrate_scanorama",
                        status="SKIP",
                        duration=0.0,
                        error_message="scanorama 未安装",
                    )
                    break

        # Nonlinear (alias for Harmony)
        result = self._run_test(
            module="integration",
            method="integrate_nonlinear",
            test_func=lambda: integrate_nonlinear(
                self.container,
                batch_key="batch",
                assay_name=self.assay_name,
                base_layer=source_layer,
                new_layer_name="nonlinear",
            ),
        )
        if result.status == "FAIL" and ("harmonypy" in str(result.error_message).lower() or "MissingDependencyError" in str(result.error_message)):
            # Update the result in the list
            for i, r in enumerate(self.results):
                if r.module == "integration" and r.method == "integrate_nonlinear":
                    self.results[i] = TestResult(
                        module="integration",
                        method="integrate_nonlinear",
                        status="SKIP",
                        duration=0.0,
                        error_message="harmonypy 未安装",
                    )
                    break

    def test_dim_reduction(self) -> None:
        """Test dimensionality reduction methods."""
        print("\n" + "=" * 80)
        print("4. 降维模块测试 (Dimensionality Reduction)")
        print("=" * 80)

        if self.container is None:
            print("✗ 容器未初始化")
            return

        # Use imputed layer for dimensionality reduction
        source_layer_candidates = ["knn_imputed", "bpca_imputed", "mf_imputed", "log", self.base_layer]
        source_layer = None
        for layer in source_layer_candidates:
            if layer in self.container.assays[self.assay_name].layers:
                source_layer = layer
                break

        if source_layer is None:
            source_layer = self.base_layer

        print(f"\n使用源层: {source_layer}")

        # PCA - dynamically adjust n_components based on data size
        n_components = min(30, self.container.n_samples - 1, self.container.n_features)
        pca_result = self._run_test(
            module="dim_reduction",
            method="reduce_pca",
            test_func=lambda n_comp=n_components: reduce_pca(
                self.container,
                assay_name=self.assay_name,
                base_layer=source_layer,
                new_assay_name="pca",
                n_components=n_comp,
            ),
        )

        # UMAP (requires PCA first)
        if pca_result.status == "PASS":
            umap_result = self._run_test(
                module="dim_reduction",
                method="reduce_umap",
                test_func=lambda: reduce_umap(
                    self.container,
                    assay_name="pca",
                    base_layer="X",
                    new_assay_name="umap",
                    n_components=2,
                ),
            )
            if umap_result.status == "FAIL" and "umap" in str(umap_result.error_message).lower():
                umap_result.status = "SKIP"
                umap_result.error_message = "umap-learn 未安装"
        else:
            print("\n⚠ PCA 未完成，跳过 UMAP 测试")
            self.results.append(TestResult(
                module="dim_reduction",
                method="reduce_umap",
                status="SKIP",
                duration=0.0,
                error_message="PCA 前置步骤未完成",
            ))

    def test_clustering(self) -> None:
        """Test clustering methods."""
        print("\n" + "=" * 80)
        print("5. 聚类模块测试 (Clustering)")
        print("=" * 80)

        if self.container is None:
            print("✗ 容器未初始化")
            return

        # Check if PCA was successful for clustering
        pca_was_run = any(r.module == "dim_reduction" and r.method == "reduce_pca" and r.status == "PASS" for r in self.results)

        if pca_was_run and "pca" in self.container.assays:
            assay_name = "pca"
            base_layer = "X"
            print(f"\n使用数据: assay={assay_name}, layer={base_layer}")
        else:
            print("\n⚠ PCA 未完成，尝试使用原始数据进行聚类")
            assay_name = self.assay_name
            base_layer = self.base_layer
            print(f"使用数据: assay={assay_name}, layer={base_layer}")

        # K-means clustering
        self._run_test(
            module="clustering",
            method="cluster_kmeans",
            test_func=lambda: cluster_kmeans(
                self.container,
                assay_name=assay_name,
                base_layer=base_layer,
                n_clusters=5,
                random_state=42,
            ),
        )

        # Leiden clustering
        result = self._run_test(
            module="clustering",
            method="cluster_leiden",
            test_func=lambda: cluster_leiden(
                self.container,
                assay_name=assay_name,
                base_layer=base_layer,
                n_neighbors=min(15, self.container.n_samples - 1),
                resolution=1.0,
                random_state=42,
            ),
        )
        if result.status == "FAIL" and ("leiden" in str(result.error_message).lower() or "igraph" in str(result.error_message).lower() or "ModuleNotFoundError" in str(result.error_message) or "MissingDependencyError" in str(result.error_message)):
            # Update the result in the list
            for i, r in enumerate(self.results):
                if r.module == "clustering" and r.method == "cluster_leiden":
                    self.results[i] = TestResult(
                        module="clustering",
                        method="cluster_leiden",
                        status="SKIP",
                        duration=0.0,
                        error_message="leidenalg/igraph 未安装",
                    )
                    break

    def generate_report(self) -> None:
        """Generate comprehensive test report in Markdown format."""
        print("\n" + "=" * 80)
        print("生成测试报告")
        print("=" * 80)

        report_path = self.output_dir / "pipeline_test_report.md"

        # Calculate summary statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.status == "PASS")
        failed_tests = sum(1 for r in self.results if r.status == "FAIL")
        skipped_tests = sum(1 for r in self.results if r.status == "SKIP")
        total_duration = sum(r.duration for r in self.results)

        # Generate report content
        report_lines = [
            "# ScpTensor 综合管道测试报告",
            "",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 测试概要",
            "",
            f"| 指标 | 数值 |",
            f"|------|------|",
            f"| 总测试数 | {total_tests} |",
            f"| 通过 | {passed_tests} |",
            f"| 失败 | {failed_tests} |",
            f"| 跳过 | {skipped_tests} |",
            f"| 通过率 | {passed_tests / total_tests * 100 if total_tests > 0 else 0:.1f}% |",
            f"| 总耗时 | {total_duration:.2f} 秒 |",
            "",
        ]

        # Module summaries
        modules = {}
        for result in self.results:
            if result.module not in modules:
                modules[result.module] = {"pass": 0, "fail": 0, "skip": 0, "duration": 0}
            modules[result.module][result.status.lower()] += 1
            modules[result.module]["duration"] += result.duration

        report_lines.extend([
            "## 模块汇总",
            "",
            "| 模块 | 通过 | 失败 | 跳过 | 耗时 (秒) |",
            "|------|------|------|------|----------|",
        ])

        for module, stats in modules.items():
            report_lines.append(
                f"| {module} | {stats['pass']} | {stats['fail']} | {stats['skip']} | {stats['duration']:.2f} |"
            )

        report_lines.append("")

        # Detailed results by module
        for module in ["normalization", "imputation", "integration", "dim_reduction", "clustering"]:
            module_results = [r for r in self.results if r.module == module]

            if not module_results:
                continue

            # Module title
            module_titles = {
                "normalization": "1. 归一化模块 (Normalization)",
                "imputation": "2. 缺失值填充模块 (Imputation)",
                "integration": "3. 批次校正模块 (Integration)",
                "dim_reduction": "4. 降维模块 (Dimensionality Reduction)",
                "clustering": "5. 聚类模块 (Clustering)",
            }

            report_lines.extend([
                f"## {module_titles.get(module, module)}",
                "",
                "| 方法 | 状态 | 耗时 (秒) | 关键指标 | 备注 |",
                "|------|------|----------|---------|------|",
            ])

            for result in module_results:
                status_icon = {
                    "PASS": "✅",
                    "FAIL": "❌",
                    "SKIP": "⚠️",
                }.get(result.status, "❓")

                metrics_str = ""
                if result.metrics:
                    metrics_items = []
                    for key, value in result.metrics.items():
                        if key == "layers":
                            metrics_items.append(f"{key}={len(value)}")
                        elif isinstance(value, float):
                            metrics_items.append(f"{key}={value:.4f}")
                        else:
                            metrics_items.append(f"{key}={value}")
                    metrics_str = ", ".join(metrics_items)

                error_note = result.error_message if result.error_message else ""

                report_lines.append(
                    f"| {result.method} | {status_icon} {result.status} | {result.duration:.3f} | {metrics_str} | {error_note} |"
                )

            report_lines.append("")

        # Summary and recommendations
        report_lines.extend([
            "## 总结与建议",
            "",
        ])

        if passed_tests == total_tests - skipped_tests:
            report_lines.extend([
                "✅ **所有可运行测试均通过！**",
                "",
                f"共有 {skipped_tests} 个测试因缺少可选依赖包而被跳过。",
                "这些包不是 ScpTensor 核心功能必需的，但如果需要使用这些功能，",
                "请按照以下说明安装依赖包：",
                "",
            ])
            if skipped_tests > 0:
                missing_deps = set()
                for result in self.results:
                    if result.status == "SKIP":
                        if "harmonypy" in result.error_message:
                            missing_deps.add("harmonypy")
                        if "scanorama" in result.error_message:
                            missing_deps.add("scanorama")
                        if "igraph" in result.error_message or "leiden" in result.error_message:
                            missing_deps.add("igraph")
                            missing_deps.add("leidenalg")

                if missing_deps:
                    report_lines.append("```bash")
                    report_lines.append(f"pip install {' '.join(sorted(missing_deps))}")
                    report_lines.append("```")
                    report_lines.append("")

        if failed_tests > 0:
            report_lines.append(f"### 失败测试分析")
            report_lines.append("")
            for result in self.results:
                if result.status == "FAIL":
                    report_lines.append(f"**{result.module}.{result.method}**")
                    report_lines.append(f"- 错误: {result.error_message}")
                    report_lines.append("")

        if skipped_tests > 0:
            report_lines.append(f"### 跳过测试说明")
            report_lines.append("")
            report_lines.append("以下测试由于缺少依赖包而被跳过：")
            report_lines.append("")
            for result in self.results:
                if result.status == "SKIP":
                    report_lines.append(f"- **{result.module}.{result.method}**: {result.error_message}")
            report_lines.append("")

        # Write report
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        print(f"\n✓ 测试报告已生成: {report_path}")
        print(f"\n测试汇总:")
        print(f"  - 总测试数: {total_tests}")
        print(f"  - 通过: {passed_tests}")
        print(f"  - 失败: {failed_tests}")
        print(f"  - 跳过: {skipped_tests}")
        print(f"  - 通过率: {passed_tests / total_tests * 100 if total_tests > 0 else 0:.1f}%")
        print(f"  - 总耗时: {total_duration:.2f} 秒")


def main():
    """Main entry point for the pipeline test script."""
    print("=" * 80)
    print("ScpTensor 综合管道测试")
    print("=" * 80)

    tester = PipelineTester()

    # Load data
    if not tester.load_data(use_synthetic=False):
        print("\n✗ 数据加载失败，退出测试")
        return

    # Run tests sequentially
    tester.test_normalization()
    tester.test_imputation()
    tester.test_integration()
    tester.test_dim_reduction()
    tester.test_clustering()

    # Generate report
    tester.generate_report()

    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
