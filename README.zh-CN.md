# ScpTensor：基于 DIA 的单细胞蛋白质组学预处理工具包

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[English](README.md) | [简体中文](README.zh-CN.md)

ScpTensor 是一个面向基于 DIA 的单细胞蛋白质组学预处理的 Python 包。
它聚焦于稳健的 DIA 定量表导入，以及以蛋白层为中心的预处理工作流。

项目范围契约：[AGENTS.md](AGENTS.md)

## 范围

当前支持范围：
- DIA-NN 定量结果导入
- Spectronaut 定量结果导入
- 肽段/前体到蛋白的聚合
- 蛋白层预处理：变换、归一化、插补、批次整合
- 面向预处理阶段的可视化

当前包范围内明确不做的内容：
- 差异表达分析
- 特征选择模块
- 默认支持非 DIA 软件输入

发布边界说明：
- 降维（`reduce_*`）和聚类（`cluster_*`）当前被视为**实验性的下游分析辅助工具**，
  不属于发布验收意义上的核心预处理交付内容。
- 它们通过 `scptensor.experimental` 提供。

## 安装

```bash
git clone https://github.com/wenjiudaijiugui/ScpTensor.git
cd ScpTensor

# 使用 uv 管理环境
uv venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

# 稳定预处理运行时
uv pip install -e .

# 可选：开发工具
uv pip install -e ".[dev]"

# 可选：JIT 加速
uv pip install -e ".[accel]"

# 可选：可视化美化 / 高级配色
uv pip install -e ".[viz]"

# 可选：实验性下游辅助模块（例如 reduce_umap）
uv pip install -e ".[experimental]"

# 可选：实验性图聚类辅助模块（cluster_leiden）
uv pip install -e ".[graph]"

# 可选：性能基线工具
uv pip install -e ".[perf]"

# 可选：benchmark 回放 / 下载工具
uv pip install -e ".[benchmark]"
```

依赖边界：
- 默认安装有意保持稳定 DIA 预处理运行时尽可能精简
- `.[accel]` 仅添加 `numba` JIT 加速
- `.[viz]` 添加可选的 `seaborn` / `scienceplots` 样式和高级配色
- `.[experimental]` 添加 `umap-learn`，用于实验性下游 `reduce_umap`
- `.[graph]` 添加 `igraph` / `leidenalg`，用于实验性 `cluster_leiden`
- `.[perf]` 添加 `psutil`，用于 `scripts/perf/run_runtime_baseline.py`
- `.[benchmark]` 添加仅用于 benchmark 的 dataframe / 下载 / 绘图工具

## 用户工作流总览

规范用户指南：
- [稳定用户工作流](docs/user_workflows.md)

请选择与你输入相匹配的主线：
- 已经有 DIA-NN / Spectronaut 蛋白表：直接加载到 `proteins`，然后继续执行 `log -> norm -> imputed`
- 只有肽段/前体输出：先导入到 `peptides`，运行 `aggregate_to_protein()`，然后在 `proteins` assay 上继续后续流程

当前稳定的用户入口是 Python API。项目暂时还没有包级 CLI。

## 快速开始（DIA-NN）

两条规范工作流都见：[docs/user_workflows.md](docs/user_workflows.md)。

```python
from pathlib import Path

from scptensor.io import aggregate_to_protein, load_diann
from scptensor.normalization import norm_median
from scptensor.transformation import log_transform
from scptensor.viz import plot_data_overview

# 1) 加载 DIA-NN long-format report（肽段层）
report = Path("data/dia/diann/PXD054343/1_SC_LF_report.tsv")
container = load_diann(report, level="peptide", table_format="long", assay_name="peptides")

# 2) 肽段 -> 蛋白聚合
container = aggregate_to_protein(
    container,
    source_assay="peptides",
    source_layer="raw",
    target_assay="proteins",
    method="top_n",
)

# 3) 变换 + 归一化
container = log_transform(
    container,
    assay_name="proteins",
    source_layer="raw",
    new_layer_name="log",
    base=2.0,
)
container = norm_median(
    container,
    assay_name="proteins",
    source_layer="log",
    new_layer_name="norm",
)

# 4) 预处理层面的可视化
_ = plot_data_overview(container, assay_name="proteins", layer="norm")
```

## 实验性模块

实验性的下游降维与聚类 API 位于：

```python
from scptensor.experimental import cluster_kmeans, reduce_pca
```

这些 API 被有意排除在核心预处理发布标准之外。
边界契约：[docs/experimental_downstream_contract.md](docs/experimental_downstream_contract.md)

实验性的肽段/PSM QC 辅助工具仍然保留在同一命名空间中，以维持边界清晰，
但它**不属于**下游辅助模块。它的角色是一个实验性的聚合前 / 肽段-PSM QC 模块：

```python
from scptensor.experimental import qc_psm
```

辅助工具契约：[docs/qc_psm_contract.md](docs/qc_psm_contract.md)

## 支持的输入类型（I/O）

ScpTensor 当前的 I/O 仅面向 DIA-NN 与 Spectronaut。

| 软件 | 定量层级 | 文件形态 |
| --- | --- | --- |
| DIA-NN | 蛋白 | long + matrix |
| DIA-NN | 肽段/前体 | long + matrix |
| Spectronaut | 蛋白 | long + matrix |
| Spectronaut | 肽段/前体 | long + matrix |

主要 API：
- `scptensor.io.load_quant_table`
- `scptensor.io.load_diann`
- `scptensor.io.load_spectronaut`
- `scptensor.io.load_peptide_pivot`
- `scptensor.io.aggregate_to_protein`

## 文档

索引：
- [文档索引](docs/README.md)
- [教程索引](tutorial/README.md)

契约索引：
- 完整的文档级契约索引：[docs/README.md#contract](docs/README.md#contract)

范围 / 基础：
- [项目范围契约](AGENTS.md)
- [核心数据契约](docs/core_data_contract.md)
- [核心计算契约](docs/core_compute_contract.md)
- [DIA-NN / Spectronaut I/O 契约](docs/io_diann_spectronaut.md)

稳定预处理模块：
- [Aggregation 契约](docs/aggregation_contract.md)
- [Transformation 契约](docs/transformation_contract.md)
- [Normalization 契约](docs/normalization_contract.md)
- [Standardization 契约](docs/standardization_contract.md)
- [Imputation 契约](docs/imputation_contract.md)
- [Integration 契约](docs/integration_contract.md)
- [QC 契约](docs/qc_contract.md)

稳定支持包：
- [AutoSelect 契约](docs/autoselect_contract.md)
- [Utils 契约](docs/utils_contract.md)
- [Visualization 契约](docs/viz_contract.md)

发布边界 / 实验性：
- [实验性下游边界契约](docs/experimental_downstream_contract.md)
- [实验性 PSM QC 辅助工具契约](docs/qc_psm_contract.md)

综述注册表：
- [综述清单](docs/review_manifest_20260312.json)
- [引文注册表](docs/references/citations.json)
- [引文使用映射](docs/references/citation_usage.json)

说明：
- `review_manifest_20260312.json` 是面向 `review_*.md` 文件的综述清单，不是契约清单。
- 冻结的实现契约索引位于 [docs/README.md#contract](docs/README.md#contract)，并已按上面的分类列出。

背景 / 收敛：
- [Aggregation 文献背景](docs/aggregation_literature.md)
- [实验性辅助模块对齐计划](docs/experimental_downstream_alignment_plan.md)
- [优化执行清单](docs/optimization_checklist.md)
- [运行时基线规范](docs/runtime_baseline.md)

教程：
- [主教程 Notebook](tutorial/tutorial.ipynb)
- [AutoSelect 教程](tutorial/autoselect_tutorial.ipynb)
- [稳定用户工作流](docs/user_workflows.md)

## Benchmark 资产

- [Benchmark 索引](benchmark/README.md)
- [Aggregation benchmark README](benchmark/aggregation/README.md)
- [Normalization benchmark README](benchmark/normalization/README.md)
- [Imputation benchmark README](benchmark/imputation/README.md)
- [Integration benchmark README](benchmark/integration/README.md)
- [AutoSelect benchmark 资产](benchmark/autoselect/README.md)

## 开发

```bash
# 与 CI 对齐的核心矩阵（不包含 graph clustering extras）
uv sync --extra io --extra accel --extra integration --extra viz --extra experimental --extra perf --extra benchmark
uv sync --group dev

# 开发 cluster_leiden 时再补充 graph extras
uv sync --extra graph

# 另一种方式：可编辑安装并包含全部可选 extras
uv pip install -e ".[all,dev]"

# Lint
uv run ruff check scptensor tests

# 测试
uv run pytest -q
```

## 许可证

MIT License。详见 [LICENSE](LICENSE)。
