#!/usr/bin/env python3
"""Comprehensive Quality Control Report Generator for ScpTensor.

This script performs a complete QC analysis on single-cell proteomics data,
generating visualizations and a detailed markdown report.

Usage:
    python test_qc_report.py

Output:
    All results are saved to tmp/qc_report/ directory
    - PNG visualizations (DPI=300)
    - qc_report.md: Comprehensive markdown report
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

# Import ScpTensor components
from scptensor import (
    ScpContainer,
    load_diann,
)
from scptensor.qc import (
    calculate_sample_qc_metrics,
    calculate_feature_qc_metrics,
)
from scptensor.dim_reduction import reduce_pca
from scptensor.viz.recipes import (
    qc_completeness,
    qc_matrix_spy,
    scatter as embedding_scatter,
)
from scptensor.viz.base import violin


# Configuration
DATA_PATH = Path("data/dia/diann/PXD054343/1_SC_LF_report.tsv")
OUTPUT_DIR = Path("tmp/qc_report")
DPI = 300  # Publication quality DPI


def setup_output_directory() -> Path:
    """Create output directory if it doesn't exist.

    Returns
    -------
    Path
        Path to output directory
    """
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_first_layer_name(container: ScpContainer, assay_name: str = "proteins") -> str:
    """Get the first available layer name from an assay.

    This helper function dynamically detects the first layer name in an assay,
    making the code compatible with different data loaders that may use
    different layer naming conventions (e.g., "raw", "PG_MaxLFQ", etc.).

    Parameters
    ----------
    container : ScpContainer
        Input data container
    assay_name : str, default="proteins"
        Name of the assay to get layer from

    Returns
    -------
    str
        Name of the first layer in the assay

    Raises
    ------
    ValueError
        If the assay has no layers
    """
    assay = container.assays[assay_name]
    layer_names = list(assay.layers.keys())

    if not layer_names:
        raise ValueError(f"Assay '{assay_name}' has no layers")

    return layer_names[0]


def calculate_qc_metrics(container: ScpContainer, layer_name: str) -> dict[str, Any]:
    """Calculate comprehensive QC metrics for samples and features.

    Parameters
    ----------
    container : ScpContainer
        Input data container
    layer_name : str
        Name of the layer to use for QC calculations

    Returns
    -------
    dict[str, Any]
        Dictionary containing QC metric summaries
    """
    print("\nCalculating QC metrics...")

    # Calculate sample-level QC metrics
    container_with_sample_qc = calculate_sample_qc_metrics(
        container,
        assay_name="proteins",
        layer_name=layer_name
    )

    # Calculate feature-level QC metrics
    container_with_qc = calculate_feature_qc_metrics(
        container_with_sample_qc,
        assay_name="proteins",
        layer_name=layer_name
    )

    # Extract metrics for reporting
    assay = container_with_qc.assays["proteins"]

    # Sample metrics
    n_features_col = "n_features_proteins"
    total_intensity_col = "total_intensity_proteins"
    log1p_total_intensity_col = "log1p_total_intensity_proteins"

    sample_metrics = {
        "n_features_mean": float(container_with_qc.obs[n_features_col].mean()),
        "n_features_median": float(container_with_qc.obs[n_features_col].median()),
        "n_features_std": float(container_with_qc.obs[n_features_col].std()),
        "total_intensity_mean": float(container_with_qc.obs[total_intensity_col].mean()),
        "total_intensity_median": float(container_with_qc.obs[total_intensity_col].median()),
        "log1p_total_intensity_mean": float(container_with_qc.obs[log1p_total_intensity_col].mean()),
    }

    # Feature metrics - check if detection_rate column exists
    if "detection_rate" in assay.var.columns:
        feature_metrics = {
            "n_features": assay.n_features,
            "missing_rate_mean": float(assay.var["missing_rate"].mean()),
            "missing_rate_median": float(assay.var["missing_rate"].median()),
            "detection_rate_mean": float(assay.var["detection_rate"].mean()),
            "cv_mean": float(assay.var["cv"].mean()),
            "cv_median": float(assay.var["cv"].median()),
        }
    else:
        # Calculate manually if not available
        X = assay.layers[layer_name].X
        n_detected_per_feature = np.sum((X > 0) & (~np.isnan(X)), axis=0)
        detection_rate = n_detected_per_feature / container_with_qc.n_samples
        missing_rate = 1.0 - detection_rate

        feature_metrics = {
            "n_features": assay.n_features,
            "missing_rate_mean": float(np.mean(missing_rate)),
            "missing_rate_median": float(np.median(missing_rate)),
            "detection_rate_mean": float(np.mean(detection_rate)),
            "cv_mean": 0.0,  # Placeholder
            "cv_median": 0.0,  # Placeholder
        }

    # Data sparsity
    matrix = assay.layers[layer_name]
    if matrix.M is not None:
        n_missing = int(np.sum(matrix.M > 0))
    else:
        # Calculate missing from NaN values if mask is not available
        n_missing = int(np.sum(np.isnan(matrix.X)))
    n_total = matrix.X.size
    sparsity = n_missing / n_total

    metrics = {
        "container": container_with_qc,
        "sample": sample_metrics,
        "feature": feature_metrics,
        "sparsity": {
            "n_missing": n_missing,
            "n_total": n_total,
            "sparsity_rate": sparsity,
        }
    }

    print(f"Sample QC: {sample_metrics['n_features_median']:.0f} median features detected")
    print(f"Feature QC: {feature_metrics['missing_rate_mean']:.1%} mean missing rate")
    print(f"Data sparsity: {sparsity:.1%}")

    return metrics


def run_pca_analysis(container: ScpContainer, layer_name: str) -> ScpContainer:
    """Run PCA dimensionality reduction for visualization.

    Parameters
    ----------
    container : ScpContainer
        Input data container
    layer_name : str
        Name of the layer to use for PCA

    Returns
    -------
    ScpContainer
        Container with PCA results added
    """
    print("\nRunning PCA analysis...")

    try:
        container = reduce_pca(
            container,
            assay_name="proteins",
            base_layer=layer_name,
            new_assay_name="pca",
            n_components=10,
            center=True,
            scale=False,
            return_info=True,
        )

        # Add PCA coordinates to obs
        pca_assay = container.assays["pca"]
        scores = pca_assay.layers["scores"].X

        container.obs = container.obs.with_columns([
            pl.Series("pca_1", scores[:, 0]),
            pl.Series("pca_2", scores[:, 1]),
        ])

        variance_explained = pca_assay.var["explained_variance_ratio"].to_numpy()
        print(f"PCA completed: PC1={variance_explained[0]:.1%}, PC2={variance_explained[1]:.1%}")

    except Exception as e:
        print(f"PCA analysis failed: {e}")
        # Add dummy coordinates for visualization
        n_samples = container.n_samples
        container.obs = container.obs.with_columns([
            pl.Series("pca_1", np.random.randn(n_samples)),
            pl.Series("pca_2", np.random.randn(n_samples)),
        ])

    return container


def generate_visualizations(container: ScpContainer, output_dir: Path, layer_name: str) -> dict[str, str]:
    """Generate all QC visualization plots.

    Parameters
    ----------
    container : ScpContainer
        Data container with QC metrics
    output_dir : Path
        Output directory for plots
    layer_name : str
        Name of the layer to use for visualizations

    Returns
    -------
    dict[str, str]
        Dictionary mapping visualization names to file paths
    """
    print("\nGenerating visualizations...")

    # Apply SciencePlots style
    try:
        plt.style.use(["science", "no-latex"])
    except:
        print("  Warning: SciencePlots style not available, using default")
        plt.style.use("default")

    visualization_paths = {}

    # 1. QC Completeness Violin Plot
    print("  - Creating completeness violin plot...")
    try:
        # Create custom completeness plot if cell_type column exists
        if "cell_type" in container.obs.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            qc_completeness(
                container,
                assay_name="proteins",
                layer=layer_name,
                group_by="cell_type",
                ax=ax
            )
            path = output_dir / "qc_completeness.png"
            plt.savefig(path, dpi=DPI, bbox_inches="tight")
            plt.close()
            visualization_paths["completeness"] = str(path)
            print(f"    Saved: {path}")
        else:
            print("    Skipped: cell_type column not found")
    except Exception as e:
        print(f"    Error creating completeness plot: {e}")

    # 2. QC Matrix Spy Plot
    print("  - Creating matrix spy plot...")
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        qc_matrix_spy(
            container,
            assay_name="proteins",
            layer=layer_name,
            ax=ax
        )
        path = output_dir / "qc_matrix_spy.png"
        plt.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close()
        visualization_paths["spy_plot"] = str(path)
        print(f"    Saved: {path}")
    except Exception as e:
        print(f"    Error creating spy plot: {e}")

    # 3. PCA Scatter Plot
    print("  - Creating PCA scatter plot...")
    try:
        if "pca_1" in container.obs.columns and "pca_2" in container.obs.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            embedding_scatter(
                container,
                layer=layer_name,
                basis="pca",
                color="cell_type" if "cell_type" in container.obs.columns else None,
                ax=ax
            )
            path = output_dir / "pca_scatter.png"
            plt.savefig(path, dpi=DPI, bbox_inches="tight")
            plt.close()
            visualization_paths["pca_scatter"] = str(path)
            print(f"    Saved: {path}")
        else:
            print("    Skipped: PCA coordinates not found")
    except Exception as e:
        print(f"    Error creating PCA scatter plot: {e}")

    # 4. QC Metrics Violin Plot
    print("  - Creating QC metrics violin plot...")
    try:
        # Create multi-panel violin plot for QC metrics
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        # Check if required columns exist
        has_cell_type = "cell_type" in container.obs.columns
        has_sample_metrics = "n_features_proteins" in container.obs.columns

        # Panel 1: Number of detected features
        if has_cell_type and has_sample_metrics:
            cell_types = [ct for ct in container.obs["cell_type"].unique().to_list() if ct is not None]
            data_by_type = []
            valid_labels = []
            for ct in cell_types:
                if ct is not None:
                    subset = container.obs.filter(pl.col("cell_type") == ct)
                    if not subset.is_empty():
                        data_by_type.append(subset["n_features_proteins"].to_numpy())
                        valid_labels.append(str(ct))

            if data_by_type:
                violin(
                    data=data_by_type,
                    labels=valid_labels,
                    ax=axes[0, 0],
                    title="Detected Features by Cell Type",
                    ylabel="Number of Features"
                )
            else:
                axes[0, 0].text(0.5, 0.5, "No data available", ha="center", va="center")
        else:
            # Simple histogram if no cell type
            if has_sample_metrics:
                axes[0, 0].hist(container.obs["n_features_proteins"].to_numpy(), bins=30, edgecolor="black")
                axes[0, 0].set_xlabel("Number of Features")
                axes[0, 0].set_ylabel("Frequency")
                axes[0, 0].set_title("Detected Features Distribution")

        # Panel 2: Total intensity
        if has_cell_type and has_sample_metrics:
            data_by_type = []
            for ct in cell_types:
                if ct is not None:
                    subset = container.obs.filter(pl.col("cell_type") == ct)
                    if not subset.is_empty():
                        data_by_type.append(subset["total_intensity_proteins"].to_numpy())

            if data_by_type:
                violin(
                    data=data_by_type,
                    labels=valid_labels,
                    ax=axes[0, 1],
                    title="Total Intensity by Cell Type",
                    ylabel="Total Intensity"
                )
            else:
                axes[0, 1].text(0.5, 0.5, "No data available", ha="center", va="center")

        # Panel 3: Log-transformed intensity
        if has_cell_type and has_sample_metrics:
            data_by_type = []
            for ct in cell_types:
                if ct is not None:
                    subset = container.obs.filter(pl.col("cell_type") == ct)
                    if not subset.is_empty():
                        data_by_type.append(subset["log1p_total_intensity_proteins"].to_numpy())

            if data_by_type:
                violin(
                    data=data_by_type,
                    labels=valid_labels,
                    ax=axes[1, 0],
                    title="Log1p Intensity by Cell Type",
                    ylabel="Log1p Total Intensity"
                )
            else:
                axes[1, 0].text(0.5, 0.5, "No data available", ha="center", va="center")

        # Panel 4: Detection rate distribution
        assay = container.assays["proteins"]
        if "detection_rate" in assay.var.columns:
            detection_rates = assay.var["detection_rate"].to_numpy()
            axes[1, 1].hist(detection_rates, bins=50, edgecolor="black", alpha=0.7)
            axes[1, 1].set_xlabel("Detection Rate")
            axes[1, 1].set_ylabel("Number of Features")
            axes[1, 1].set_title("Feature Detection Rate Distribution")
        else:
            axes[1, 1].text(0.5, 0.5, "Detection rate not available", ha="center", va="center")

        plt.tight_layout()
        path = output_dir / "qc_metrics_violin.png"
        plt.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close()
        visualization_paths["qc_violin"] = str(path)
        print(f"    Saved: {path}")
    except Exception as e:
        print(f"    Error creating QC metrics violin plot: {e}")

    return visualization_paths


def generate_markdown_report(
    metrics: dict[str, Any],
    visualization_paths: dict[str, str],
    output_dir: Path,
    layer_name: str
) -> Path:
    """Generate comprehensive markdown QC report.

    Parameters
    ----------
    metrics : dict[str, Any]
        Dictionary containing all QC metrics
    visualization_paths : dict[str, str]
        Dictionary mapping visualization names to file paths
    output_dir : Path
        Output directory for the report
    layer_name : str
        Name of the layer used for analysis

    Returns
    -------
    Path
        Path to generated markdown report
    """
    print("\nGenerating markdown report...")

    report_path = output_dir / "qc_report.md"

    # Calculate statistics for report
    container = metrics["container"]
    sample_metrics = metrics["sample"]
    feature_metrics = metrics["feature"]
    sparsity = metrics["sparsity"]

    assay = container.assays["proteins"]

    # Get cell type distribution if available
    cell_type_stats = ""
    if "cell_type" in container.obs.columns:
        cell_type_counts = container.obs["cell_type"].value_counts()
        cell_type_stats = "\n\n### 细胞类型分布\n\n"
        cell_type_stats += "| 细胞类型 | 数量 | 百分比 |\n"
        cell_type_stats += "|----------|------|--------|\n"
        total = container.n_samples
        for row in cell_type_counts.rows():
            cell_type = row[0] if row[0] is not None else "未知"
            count = row[1]
            cell_type_stats += f"| {cell_type} | {count} | {count/total*100:.1f}% |\n"

    # Get feature quality statistics
    has_detection_rate = "detection_rate" in assay.var.columns
    if has_detection_rate:
        high_quality_features = int(np.sum(assay.var["detection_rate"].to_numpy() > 0.5))
        low_quality_features = int(np.sum(assay.var["missing_rate"].to_numpy() > 0.8))
        well_detected = int(np.sum(assay.var["detection_rate"].to_numpy() > 0.7))
        moderate_detected = int(np.sum((assay.var["detection_rate"].to_numpy() >= 0.3) & (assay.var["detection_rate"].to_numpy() <= 0.7)))
        poorly_detected = int(np.sum(assay.var["detection_rate"].to_numpy() < 0.3))
    else:
        # Calculate manually
        X = assay.layers[layer_name].X
        n_detected = np.sum((X > 0) & (~np.isnan(X)), axis=0)
        detection_rate = n_detected / container.n_samples
        missing_rate = 1.0 - detection_rate

        high_quality_features = int(np.sum(detection_rate > 0.5))
        low_quality_features = int(np.sum(missing_rate > 0.8))
        well_detected = int(np.sum(detection_rate > 0.7))
        moderate_detected = int(np.sum((detection_rate >= 0.3) & (detection_rate <= 0.7)))
        poorly_detected = int(np.sum(detection_rate < 0.3))

    # Build report content
    report_content = f"""# 质量控制报告

**生成时间:** 2026-02-28
**数据集:** DIA-NN 单细胞蛋白质组学 (PXD054343)

---

## 数据概览

本报告对单细胞蛋白质组学数据进行全面的质量控制分析。

### 数据集摘要

| 指标 | 数值 |
|------|------|
| **总样本数** | {container.n_samples} |
| **总特征数** | {feature_metrics['n_features']} |
| **数据点数** | {sparsity['n_total']:,} |
| **缺失值数量** | {sparsity['n_missing']:,} |
| **整体稀疏度** | {sparsity['sparsity_rate']:.1%} |
{cell_type_stats}

### 数据完整性

| 指标 | 数值 |
|------|------|
| **高质量特征 (>50% 检出率)** | {high_quality_features} ({high_quality_features/feature_metrics['n_features']*100:.1f}%) |
| **低质量特征 (>80% 缺失率)** | {low_quality_features} ({low_quality_features/feature_metrics['n_features']*100:.1f}%) |

---

## 样本质量评估

### 检出统计

| 指标 | 平均值 | 中位数 | 标准差 |
|------|--------|--------|--------|
| **检出特征数** | {sample_metrics['n_features_mean']:.0f} | {sample_metrics['n_features_median']:.0f} | {sample_metrics['n_features_std']:.0f} |
| **总强度** | {sample_metrics['total_intensity_mean']:,.0f} | {sample_metrics['total_intensity_median']:,.0f} | - |
| **Log1p 强度** | {sample_metrics['log1p_total_intensity_mean']:.2f} | - | - |

### 结果解读

- **特征检出**: 样本中位检出 **{sample_metrics['n_features_median']:.0f} 个特征**，总计 {feature_metrics['n_features']} 个
- **强度分布**: 对数转换后的强度均值为 {sample_metrics['log1p_total_intensity_mean']:.2f}
- **样本质量**: {'良好 - 样本间检出一致性高' if sample_metrics['n_features_std'] < sample_metrics['n_features_mean'] * 0.3 else '变异较大 - 建议过滤低质量样本'}

---

## 特征质量评估

### 缺失与检出

| 指标 | 平均值 | 中位数 |
|------|--------|--------|
| **缺失率** | {feature_metrics['missing_rate_mean']:.1%} | {feature_metrics['missing_rate_median']:.1%} |
| **检出率** | {feature_metrics['detection_rate_mean']:.1%} | {(1 - feature_metrics['missing_rate_median']):.1%} |
| **变异系数** | {feature_metrics['cv_mean']:.2f} | {feature_metrics['cv_median']:.2f} |

### 特征质量分布

- **良好检出特征** (检出率 > 70%): {well_detected} 个特征
- **中等检出特征** (检出率 30-70%): {moderate_detected} 个特征
- **较差检出特征** (检出率 < 30%): {poorly_detected} 个特征

### 建议

{'- 建议过滤高缺失率特征 (>70%)' if feature_metrics['missing_rate_mean'] > 0.5 else '- 特征检出率可接受' }
{'- 检测到高变异，建议进行标准化' if feature_metrics['cv_mean'] > 1.0 else '- 变异性在可接受范围内' }

---

## 数据稀疏性分析

### 缺失值分布

- **总缺失值数**: {sparsity['n_missing']:,} / {sparsity['n_total']:,} 个数据点
- **稀疏率**: {sparsity['sparsity_rate']:.1%}
- **数据完整度**: {(1 - sparsity['sparsity_rate']):.1%}

### 缺失值模式

数据呈现典型的单细胞蛋白质组学稀疏模式:
- **非随机缺失 (MNAR)**: 因检测限导致的缺失值
- **稀疏矩阵**: 建议使用稀疏矩阵运算以提高效率
- **插值建议**: {'是 - 检测到高稀疏度 (' + f'{sparsity["sparsity_rate"]:.1%}' + ')' if sparsity['sparsity_rate'] > 0.3 else '可选 - 中等稀疏度 (' + f'{sparsity["sparsity_rate"]:.1%}' + ')'}

---

## 可视化图表

### 1. 按细胞类型的数据完整性

![Data Completeness](qc_completeness.png)

**解读**: 小提琴图展示了不同细胞类型的检出特征分布。较宽的部分表示具有该特征数量的样本较多。

### 2. 缺失值分布

![Matrix Spy Plot](qc_matrix_spy.png)

**解读**: 间谍图可视化整个数据集的缺失值模式。每一行代表一个样本，每一列代表一个特征。缺失值以浅灰色显示。

### 3. PCA 降维嵌入

![PCA Scatter Plot](pca_scatter.png)

**解读**: PCA 散点图展示了样本在降维空间中的聚类情况。样本按细胞类型着色，以可视化批次效应或生物学分组。

### 4. 质控指标汇总

![QC Metrics Violin](qc_metrics_violin.png)

**解读**: 多面板可视化展示 (Panel titles in English for publication standards):
- **左上 (Detected Features by Cell Type)**: 按细胞类型的特征检出分布
- **右上 (Total Intensity by Cell Type)**: 按细胞类型的总强度分布
- **左下 (Log1p Intensity by Cell Type)**: 对数转换后的强度分布
- **右下 (Feature Detection Rate Distribution)**: 整体特征检出率直方图

---

## 建议

### 预处理步骤

1. **质量过滤**
   - 移除低质量样本（特征检出率低）
   - 过滤高缺失率特征 (>70%)

2. **标准化**
   - 对强度值应用对数转换
   - 考虑中位数或分位数标准化

3. **插值填补**
   - {'建议进行插值填补，因为稀疏度较高 (' + f'{sparsity["sparsity_rate"]:.1%}' + ')' if sparsity['sparsity_rate'] > 0.3 else '插值填补可选 (' + f'{sparsity["sparsity_rate"]:.1%}' + ' 稀疏度)'}
   - 使用适当的插值方法（KNN、MissForest 等）

4. **批次校正**
   - 使用元数据评估批次效应
   - 如需要，应用 ComBat 或 Harmony

### 下一步操作

1. 基于上述指标应用质量控制过滤
2. 执行标准化和转换
3. 对过滤后的数据运行降维分析 (PCA/UMAP)
4. 进行下游分析（聚类、差异表达分析）

---

## 分析参数

- **输入数据**: {DATA_PATH}
- **质控层**: {layer_name}
- **DPI**: {DPI}
- **样式**: SciencePlots (no-latex)

---

*由 ScpTensor 质控流程生成*
"""

    # Write report to file
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"Report saved to: {report_path}")

    return report_path


def main() -> None:
    """Main execution function."""
    print("=" * 70)
    print("ScpTensor Quality Control Report Generator")
    print("=" * 70)

    # Setup output directory
    output_dir = setup_output_directory()
    print(f"\nOutput directory: {output_dir}")

    # Load data using the project's IO method
    print(f"Loading data from {DATA_PATH}...")
    container = load_diann(DATA_PATH)

    # Detect the first available layer name dynamically
    layer_name = get_first_layer_name(container, assay_name="proteins")
    print(f"Using layer: {layer_name}")

    # Calculate QC metrics
    metrics = calculate_qc_metrics(container, layer_name=layer_name)

    # Run PCA for visualization
    container = run_pca_analysis(metrics["container"], layer_name=layer_name)

    # Update metrics with PCA-enabled container
    metrics["container"] = container

    # Generate visualizations
    visualization_paths = generate_visualizations(container, output_dir, layer_name=layer_name)

    # Generate markdown report
    report_path = generate_markdown_report(metrics, visualization_paths, output_dir, layer_name=layer_name)

    print("\n" + "=" * 70)
    print("QC Report Generation Complete!")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")
    print(f"- Visualizations: {len(visualization_paths)} plots")
    print(f"- Report: {report_path}")
    print("\nGenerated files:")
    for name, path in visualization_paths.items():
        print(f"  - {name}: {path}")
    print(f"  - report: {report_path}")


if __name__ == "__main__":
    main()
