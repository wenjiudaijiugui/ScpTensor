# ScpTensor Research Report: A High-Performance Single-Cell Proteomics Analysis Framework
# ScpTensor 研究报告：高性能单细胞蛋白质组学分析框架

**Author:** Shenshang (ScpTensor Team)  
**Date:** 2025-12-01  
**Version:** 0.1.0  

---

## Abstract / 摘要

**English:**  
ScpTensor is a cutting-edge Python library specifically designed for the rigorous analysis of single-cell proteomics (SCP) data. It introduces a structured data container, `ScpContainer`, optimized for multi-assay experiments, enabling efficient handling of complex datasets including peptides and proteins. The framework integrates a comprehensive suite of tools for quality control, imputation, normalization, batch effect correction, feature selection, dimensionality reduction, and clustering. By leveraging high-performance libraries such as Polars, NumPy, SciPy, and Numba, ScpTensor ensures scalability and speed, making it an essential tool for modern computational biology research.

**中文:**  
ScpTensor 是一个专为单细胞蛋白质组学（SCP）数据严谨分析而设计的尖端 Python 库。它引入了一种结构化的数据容器 `ScpContainer`，针对多检测实验进行了优化，能够高效处理包括肽段和蛋白质在内的复杂数据集。该框架集成了质量控制、插补、归一化、批次效应校正、特征选择、降维和聚类等全套工具。通过利用 Polars、NumPy、SciPy 和 Numba 等高性能库，ScpTensor 确保了可扩展性和速度，使其成为现代计算生物学研究的重要工具。

---

## 1. Introduction / 引言

### 1.1 Background / 背景
Single-cell proteomics offers unprecedented insights into cellular heterogeneity but presents unique computational challenges due to data sparsity (missing values), high noise levels, and complex batch effects. Existing tools often lack a unified structure to handle the hierarchical nature of SCP data (e.g., peptide-to-protein aggregation).

单细胞蛋白质组学为细胞异质性提供了前所未有的洞察力，但由于数据稀疏性（缺失值）、高噪声水平和复杂的批次效应，也带来了独特的计算挑战。现有工具往往缺乏统一的结构来处理 SCP 数据（例如肽段到蛋白质的聚合）的层级特性。

### 1.2 Objectives / 目标
ScpTensor aims to provide:
*   **Unified Data Structure:** A robust `ScpContainer` to manage multi-modal data and metadata.
*   **Reproducibility:** A `ProvenanceLog` system to track all analytical operations.
*   **Performance:** Optimized algorithms for large-scale datasets.
*   **Flexibility:** Modular components for custom analysis pipelines.

ScpTensor 旨在提供：
*   **统一数据结构：** 一个强大的 `ScpContainer` 来管理多模态数据和元数据。
*   **可复现性：** 一个 `ProvenanceLog` 系统来追踪所有分析操作。
*   **高性能：** 针对大规模数据集优化的算法。
*   **灵活性：** 用于自定义分析流程的模块化组件。

---

## 2. Methodology & Architecture / 方法论与架构

### 2.1 Core Data Structures / 核心数据结构
At the heart of ScpTensor is the `ScpContainer`, which orchestrates data management across samples and features.
ScpTensor 的核心是 `ScpContainer`，它协调样本和特征的数据管理。

*   **ScpContainer:** Top-level manager for global sample metadata (`obs`) and multiple `Assay` objects.
    *   **ScpContainer:** 全局样本元数据 (`obs`) 和多个 `Assay` 对象的顶层管理者。
*   **Assay:** Manages feature-specific data (e.g., Peptides, Proteins) and their metadata (`var`). It contains multiple `Layers`.
    *   **Assay:** 管理特定特征的数据（如肽段、蛋白质）及其元数据 (`var`)。它包含多个 `Layers`。
*   **ScpMatrix:** The physical storage unit supporting both dense (`numpy`) and sparse (`scipy.sparse`) matrices, along with a mask matrix (`M`) for tracking data quality (e.g., valid, missing, filtered).
    *   **ScpMatrix:** 物理存储单元，支持稠密 (`numpy`) 和稀疏 (`scipy.sparse`) 矩阵，以及用于追踪数据质量（如有效、缺失、过滤）的掩码矩阵 (`M`)。
*   **AggregationLink:** Defines relationships between assays (e.g., mapping peptides to proteins).
    *   **AggregationLink:** 定义检测之间的关系（例如，将肽段映射到蛋白质）。

### 2.2 Analytical Modules / 分析模块

ScpTensor is organized into specialized modules:
ScpTensor 被组织成专门的模块：

| Module / 模块 | Description / 描述 | Key Methods / 关键方法 |
| :--- | :--- | :--- |
| **`scptensor.core`** | Data structures and exceptions. <br> 数据结构和异常。 | `ScpContainer`, `Assay`, `ScpMatrix` |
| **`scptensor.qc`** | Quality control for samples and features. <br> 样本和特征的质量控制。 | `basic`, `outlier` |
| **`scptensor.normalization`** | Data normalization techniques. <br> 数据归一化技术。 | `median_centering`, `median_scaling`, `zscore`, `log` |
| **`scptensor.impute`** | Handling missing values. <br> 处理缺失值。 | `knn`, `missforest`, `ppca`, `svd` |
| **`scptensor.integration`** | Batch effect correction and integration. <br> 批次效应校正和整合。 | `combat`, `harmony`, `mnn`, `scanorama` |
| **`scptensor.dim_reduction`** | Dimensionality reduction. <br> 降维。 | `pca`, `umap` |
| **`scptensor.cluster`** | Clustering algorithms. <br> 聚类算法。 | `kmeans`, `graph` (Leiden/Louvain) |
| **`scptensor.diff_expr`** | Differential expression analysis. <br> 差异表达分析。 | *(In development / 开发中)* |
| **`scptensor.viz`** | Visualization tools (SciencePlots style). <br> 可视化工具（SciencePlots 风格）。 | `heatmap`, `scatter`, `violin` |

---

## 3. Installation & Usage / 安装与使用

### 3.1 Environment Setup / 环境设置
ScpTensor strictly adheres to modern Python practices using `uv` for dependency management.
ScpTensor 严格遵守现代 Python 实践，使用 `uv` 进行依赖管理。

```bash
# Clone the repository
git clone https://github.com/yourusername/ScpTensor.git
cd ScpTensor

# Create a virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e .
```

### 3.2 Quick Start Example / 快速入门示例

```python
import numpy as np
import polars as pl
from scptensor.core.structures import ScpContainer, Assay, ScpMatrix

# 1. Create Dummy Data / 创建模拟数据
n_samples, n_features = 100, 500
X = np.random.rand(n_samples, n_features)
obs = pl.DataFrame({"_index": [f"S{i}" for i in range(n_samples)], "group": ["A"]*50 + ["B"]*50})
var = pl.DataFrame({"_index": [f"P{i}" for i in range(n_features)]})

# 2. Initialize Container / 初始化容器
matrix = ScpMatrix(X=X)
assay = Assay(var=var, layers={"raw": matrix})
container = ScpContainer(obs=obs, assays={"proteins": assay})

# 3. Basic Analysis Pipeline / 基础分析流程
# (Conceptual example, exact API calls depend on module implementation)
# from scptensor.normalization import median_centering
# median_centering(container, assay_name="proteins", ... )
```

---

## 4. Development Standards / 开发标准

### 4.1 Coding Conventions / 代码规范
*   **Type Safety:** All code must use explicit type hints (Python 3.12+).
    *   **类型安全：** 所有代码必须使用显式类型提示 (Python 3.12+)。
*   **Dependency Management:** `uv` is mandatory for environment management.
    *   **依赖管理：** 强制使用 `uv` 进行环境管理。
*   **Visualization:** Plots must use `scienceplots` and support English-only text.
    *   **可视化：** 绘图必须使用 `scienceplots` 并仅支持英文文本。
    ```python
    import matplotlib.pyplot as plt
    import scienceplots
    plt.style.use(["science", "no-latex"])
    ```

### 4.2 Testing / 测试
The project maintains a comprehensive test suite using `pytest`.
项目使用 `pytest` 维护了一套全面的测试套件。

```bash
# Run all tests
pytest tests/
```

---

## 5. Future Directions / 未来方向
*   Implementation of advanced differential expression methods.
    *   实现高级差异表达方法。
*   Enhanced integration with `scanpy` ecosystem.
    *   增强与 `scanpy` 生态系统的整合。
*   GPU acceleration support for intensive computations.
    *   针对密集计算的 GPU 加速支持。

---

## 6. Contact / 联系方式
For issues and contributions, please refer to the GitHub repository.
如有问题和贡献，请参考 GitHub 仓库。

---
*Generated by ScpTensor Team*
