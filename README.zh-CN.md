# ScpTensor：DIA 单细胞蛋白质组学预处理工具箱

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[English](README.md) | [简体中文](README.zh-CN.md)

ScpTensor 是一个专为基于 DIA（数据独立采集）的单细胞蛋白质组学数据设计的 Python 预处理工具箱。
其核心目标是提供稳健的 DIA 定量表格解析以及蛋白质水平的预处理工作流。

## 核心特性

- **稳健的 I/O 接口：** 原生支持导入 DIA-NN 与 Spectronaut 的定量输出（支持 protein 与 peptide 双层级）。
- **完整的预处理管线：** 提供包含对数变换 (log transform)、归一化 (normalization)、插值 (imputation) 和批次整合 (integration) 的端到端蛋白质层级预处理。
- **数据聚合：** 提供稳健的 peptide/precursor 到 protein 的聚合方法。
- **契约驱动设计：** 严格定义的数据边界和计算契约，保证结果的可审计性与可重复性。

**注意：** ScpTensor 明确不支持默认解析非 DIA 软件数据，也不原生提供差异表达分析或特征选择模块。下游的降维 (`reduce_*`) 和聚类 (`cluster_*`) 功能仅作为实验性辅助工具（位于 `scptensor.experimental` 下）提供。

## 安装指南

推荐使用 [uv](https://github.com/astral-sh/uv) 进行快速、可靠的环境管理。

```bash
git clone https://github.com/wenjiudaijiugui/ScpTensor.git
cd ScpTensor

uv venv
source .venv/bin/activate  # Windows 下使用: .venv\Scripts\activate

# 安装稳定版核心预处理运行时
uv pip install -e .

# 可选增强安装：
uv pip install -e ".[viz]"          # 可视化美化增强
uv pip install -e ".[accel]"        # Numba JIT 加速
uv pip install -e ".[experimental]" # 下游实验性工具（如 UMAP）
uv pip install -e ".[all,dev]"      # 完整的开发与测试环境
```

## 快速开始

ScpTensor 目前通过 Python API 进行交互。下面是处理 DIA-NN 报告的极简示例：

```python
from pathlib import Path
from scptensor.io import aggregate_to_protein, load_diann
from scptensor.normalization import norm_median
from scptensor.transformation import log_transform
from scptensor.viz import plot_data_overview

# 1. 读取 DIA-NN 长表格格式报告（peptide 层级）
report = Path("data/dia/diann/PXD054343/1_SC_LF_report.tsv")
container = load_diann(report, level="peptide", table_format="long", assay_name="peptides")

# 2. 从 peptide 聚合至 protein 层级
container = aggregate_to_protein(
    container, source_assay="peptides", source_layer="raw", target_assay="proteins", method="top_n"
)

# 3. 对数变换与归一化
container = log_transform(container, assay_name="proteins", source_layer="raw", new_layer_name="log", base=2.0)
container = norm_median(container, assay_name="proteins", source_layer="log", new_layer_name="norm")

# 4. 可视化概览
_ = plot_data_overview(container, assay_name="proteins", layer="norm")
```

更多详细指南，请参阅 [稳定用户工作流](docs/user_workflows.md) 和 [主教程 Notebook](tutorial/tutorial.ipynb)。

## 文档指引

- **[完整文档站点](docs/index.md)：** （可在本地运行 `uv run mkdocs serve` 预览网页文档）
- **[用户工作流](docs/user_workflows.md)：** 推荐的标准处理流程指南。
- **[API 参考](docs/api.md)：** 完整的模块与函数说明。
- **[架构契约](docs/README.md#contract)：** 核心数据模型、计算语义及 I/O 规范。

## 社区与参与贡献

我们非常欢迎社区参与！在提交 Pull Request 或建立 Issue 之前，请先阅读我们的社区规范：

- **[贡献指南](CONTRIBUTING.md)：** 环境配置、代码规范及 PR 流程说明。
- **[行为准则](CODE_OF_CONDUCT.md)：** 我们的社区期望与互动准则。
- **[安全策略](SECURITY.md)：** 如何负责任地报告安全漏洞。

*有关项目内部治理、架构评审纪要及性能基准测试等内容，请查阅 `docs/internal/` 与 `benchmark/` 目录。*

## 许可证

本项目基于 [MIT 许可证](LICENSE) 开源。
