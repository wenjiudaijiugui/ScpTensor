# ScpTensor vs Scanpy 性能对比设计文档

**创建日期**: 2026-01-18
**状态**: 设计阶段
**目标**: 为 ScpTensor 开源发布提供与 Scanpy 的全面性能对比

---

## 1. 项目概述

### 1.1 目标

构建 ScpTensor 与 Scanpy 的性能对比框架，为开源发布提供可验证的性能基准数据。展示 ScpTensor 在单细胞蛋白质组学（SCP）数据分析中的性能特点。

### 1.2 对比策略

采用三层对比架构：

| 层级 | 描述 | 内容 |
|------|------|------|
| **Layer 1** | 共有方法直接对比 | ScpTensor 与 Scanpy 功能对应方法 |
| **Layer 2** | ScpTensor 内部对比 | 展示 ScpTensor 独有高级方法 |
| **Layer 3** | Scanpy 内部对比 | 完整性补充，标记未实现功能 |

---

## 2. 模块映射

### 2.1 Layer 1: 共有方法对比

| 模块 | ScpTensor 方法 | Scanpy 方法 | 评估指标 |
|------|----------------|-------------|----------|
| **Normalization** | `log_normalize()` | `pp.normalize_total() + pp.log1p()` | 数值精度、运行时间、内存使用 |
| **Normalization** | `z_score_normalize()` | `pp.scale()` | 标准化效果、分布特征、性能 |
| **Imputation** | `knn_impute()` | `pp.neighbors()` + 填充 | 插补精度、分布保真度、K值敏感性 |
| **Dim Reduction** | `pca()` | `tl.pca()` | 方差解释比、重构误差、可扩展性 |
| **Dim Reduction** | `umap()` | `tl.umap()` | 局部/全局结构保持、运行时间 |
| **Clustering** | `kmeans()` | `external.pp.kmeans()` | 聚类一致性、Silhouette Score |
| **Feature Selection** | `hvg()` | `pp.highly_variable_genes()` | 特征重叠度、排名相关性 |

### 2.2 Layer 2: ScpTensor 内部对比

#### 高级插补方法

| 方法 | 对比基线 | 关键指标 |
|------|----------|----------|
| SVD Impute | KNN | 不同缺失率下的表现 |
| BPCA Impute | KNN + SVD | 收敛速度、插补精度 |
| MissForest Impute | 全部方法 | 复杂缺失模式处理 |
| LLS Impute | KNN | 局部结构利用能力 |
| MinProb Impute | 均值插补 | 零值膨胀数据处理 |
| QRILC Impute | 随机插补 | 检测限以下数据处理 |

#### 批次校正方法

| 方法 | 对比基线 | 关键指标 |
|------|----------|----------|
| ComBat | 无校正 | 批次混合评分、生物学信号保持 |
| Harmony | ComBat | 复杂批次结构处理 |
| MNN Correct | ComBat | 跨批次对应关系 |
| Scanorama | MNN | 大规模数据整合 |

### 2.3 Layer 3: Scanpy 内部对比

标记为 ScpTensor 未实现的功能：

| 方法 | 用途 | 优先级 |
|------|------|--------|
| Magic Impute | 扩散插补 | P2 |
| Diffusion Maps | 非线性降维 | P3 |
| Paga | 轨迹推断 | P2 |
| TSNE | 降维 | P3 |

---

## 3. 数据集策略

### 3.1 合成数据集

| 名称 | 样本数 | 特征数 | 缺失率 | 批次数 | 噪声 | 用途 |
|------|--------|--------|--------|--------|------|------|
| `synthetic_small` | 500 | 100 | 10%-30% | 1 | 低 | 快速验证 |
| `synthetic_medium` | 2,000 | 500 | 15%-25% | 2-3 | 中 | 标准测试 |
| `synthetic_large` | 10,000 | 1,000 | 20% | 3-5 | 中+高 | 可扩展性 |
| `synthetic_batch` | 3,000 | 500 | 20% | 5 | 中 | 批次校正 |

### 3.2 真实 SCP 数据集

从 `scptensor.datasets` 加载真实单细胞蛋白质组数据用于验证。

### 3.3 参数扫描

| 参数 | 扫描范围 | 步长 |
|------|----------|------|
| 数据规模 | 500 - 10,000 | 对数 |
| 缺失率 | 5% - 40% | 5% |
| 批次数 | 1 - 5 | 1 |
| K (KNN) | 5 - 50 | 5 |
| n_components (PCA) | 10 - 100 | 10 |

---

## 4. 评估指标体系

### 4.1 技术指标

| 指标 | 计算 | 单位 | 目标 |
|------|------|------|------|
| 运行时间 | `time.perf_counter()` | 秒 | 越小越好 |
| 内存峰值 | Resource tracking | MB | 越小越好 |
| 加速比 | baseline / method | 倍 | >1 优势 |

### 4.2 生物学指标

| 指标 | 计算 | 范围 | 目标 |
|------|------|------|------|
| 数据恢复度 | 1 - MSE / var | [0,1] | 越高越好 |
| MAE | mean(abs(x - x_true)) | [0,∞) | 越小越好 |
| ARI | adjusted_rand_score | [0,1] | 越高越好 |
| NMI | normalized_mutual_info_score | [0,1] | 越高越好 |
| Silhouette | silhouette_score | [-1,1] | 越高越好 |
| 批次混合 | batch_mixing_score | [0,1] | 越高越好 |

### 4.3 统计指标

| 指标 | 计算 | 范围 | 目标 |
|------|------|------|------|
| 方差保留率 | explained_var / total_var | [0,1] | 越高越好 |
| 重构误差 | MSE(X, X_reconstructed) | [0,∞) | 越小越好 |
| 信噪比 | signal_var / noise_var | dB | 越高越好 |

### 4.4 综合评分

```
总体得分 = 0.3 × 技术得分 + 0.4 × 生物学得分 + 0.3 × 统计得分
```

---

## 5. 可视化方案

### 5.1 配置标准

```python
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "no-latex"])
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
})
```

### 5.2 图表类型

| 类型 | 用途 | 示例 |
|------|------|------|
| 柱状图 | 运行时间、内存对比 | ScpTensor vs Scanpy |
| 折线图 | 可扩展性 | 数据规模 vs 时间 |
| 箱线图 | 稳定性分析 | 多次运行分布 |
| 散点图 | 插补准确性 | X_true vs X_imputed |
| 热图 | 加速比矩阵 | 方法间性能对比 |
| 雷达图 | 多维度评分 | 技术/生物/统计 |

### 5.3 文件结构

```
figures/
├── 01_performance/
├── 02_normalization/
├── 03_imputation/
├── 04_dim_reduction/
├── 05_clustering/
├── 06_feature_selection/
└── summary/
```

---

## 6. 报告结构

### 6.1 Markdown 模板

```markdown
# ScpTensor vs Scanpy: 性能对比报告

## 执行摘要
### 核心发现
### 总体评分

## 1. 测试配置
### 1.1 数据集
### 1.2 评估指标说明

## 2. 共有方法直接对比
### 2.1 Normalization
### 2.2 Imputation
### 2.3 Dimensionality Reduction
### 2.4 Clustering
### 2.5 Feature Selection

## 3. ScpTensor 内部对比
### 3.1 高级插补方法
### 3.2 批次校正方法

## 4. Scanpy 内部对比

## 5. 综合分析
### 5.1 性能汇总
### 5.2 推荐使用场景

## 6. 附录
### 6.1 完整指标数据
### 6.2 系统环境
### 6.3 复现命令
```

### 6.2 输出目录

```
benchmark_results/
├── scanpy_comparison_report.md
├── figures/
├── data/
├── results/
└── config.yaml
```

---

## 7. 实施计划

### 7.1 新增文件

| 文件 | 功能 | 优先级 |
|------|------|--------|
| `data_provider.py` | 数据集管理 | P0 |
| `method_registry.py` | 方法注册 | P0 |
| `scanpy_adapter.py` | Scanpy 适配 | P0 |
| `comparison_engine.py` | 对比引擎 | P0 |
| `comparison_viz.py` | 可视化 | P0 |
| `report_generator.py` | 报告生成 | P0 |
| `run_scanpy_comparison.py` | 主入口 | P0 |

### 7.2 执行步骤

```bash
# Step 0: 清理现有结果
rm -rf benchmark_results/
rm -rf benchmark_results_*.json

# Step 1: 安装依赖
uv pip install scanpy

# Step 2: 运行对比
python -m scptensor.benchmark.run_scanpy_comparison

# Step 3: 生成报告
# 自动生成
```

---

## 8. 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                   Benchmark Controller                   │
└─────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│Data Provider │  │Method Registry│  │Metrics Engine│
└──────────────┘  └──────────────┘  └──────────────┘
        │                 │                 │
        └─────────────────┼─────────────────┘
                          ▼
                ┌─────────────────────┐
                │  Execution Engine    │
                └─────────────────────┘
                          ▼
                ┌─────────────────────┐
                │  Visualization Layer │
                └─────────────────────┘
                          ▼
                ┌─────────────────────┐
                │   Report Generator   │
                └─────────────────────┘
```

---

**文档版本**: 1.0
**最后更新**: 2026-01-18
