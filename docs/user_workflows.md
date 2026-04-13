# Stable User Workflows

本文档面向“第一次实际使用 ScpTensor 的用户”，只覆盖当前稳定主线：

- 上游输入仅限 `DIA-NN` 与 `Spectronaut`
- 最终交付物是可审计的 `protein-level quantitative matrix`
- peptide/precursor 到 protein 的转换只能通过 `aggregate_to_protein()`
- 当前稳定入口是 Python API，不提供 package CLI

如果你想看实现边界与长期合同，继续读：

- `AGENTS.md`（repo root project scope contract; not published in MkDocs site）
- [I/O contract](io_diann_spectronaut.md)
- [Aggregation contract](aggregation_contract.md)

## 1. 先判断你的输入在哪一层

使用 ScpTensor 前，先判断你手里的 quant table 已经处于哪一层：

| 你的输入 | 推荐入口 | 后续步骤 |
| --- | --- | --- |
| DIA-NN / Spectronaut protein table | `load_diann(..., level="protein")` / `load_spectronaut(..., level="protein")` | `log_transform -> norm_* -> impute_*` |
| DIA-NN / Spectronaut peptide/precursor long table | `load_diann(..., level="peptide")` / `load_spectronaut(..., level="peptide")` | `aggregate_to_protein()` 后再进入 protein 主线 |
| peptide/precursor pivot matrix | `load_peptide_pivot()` | `aggregate_to_protein()` 后再进入 protein 主线 |

判断原则：

- 如果上游已经给出 protein quant table，不要再次做 peptide -> protein 聚合
- 如果输入仍是 peptide/precursor 层，先聚合，再做后续 preprocessing
- `aggregation` 是唯一合法的 peptide/precursor -> protein 交接阶段

## 2. 安装最小运行环境

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

常见可选增强：

- 需要更多绘图风格：`uv pip install -e ".[viz]"`
- 需要 Harmony / Scanorama 等额外 integration backend：`uv pip install -e ".[integration]"`
- 需要 tutorial / 开发检查：`uv pip install -e ".[dev]"`

## 3. 工作流 A: 已经有 protein table

这是最短主线。适合：

- DIA-NN 已直接导出 protein table
- Spectronaut 已直接导出 protein table
- 你当前目标是尽快得到完整 protein matrix

下面示例使用 `Spectronaut`，`DIA-NN` 只需把加载入口替换成 `load_diann()`。

```python
from scptensor.impute import impute_row_median
from scptensor.io import load_spectronaut
from scptensor.normalization import norm_median
from scptensor.transformation import log_transform

container = load_spectronaut(
    "spectronaut_protein_report.tsv",
    level="protein",
    table_format="long",
    assay_name="proteins",
)

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
container = impute_row_median(
    container,
    assay_name="proteins",
    source_layer="norm",
    new_layer_name="imputed",
)
```

如果你暂时不想做填补，也可以把 `norm` 作为当前分析层；但项目稳定主线要求的“完整 protein-level matrix”通常意味着还需要显式 imputation。

## 4. 工作流 B: 从 peptide/precursor 聚合到 protein

这是另一条 canonical path。适合：

- 你手上是 DIA-NN peptide long table
- 你手上是 Spectronaut precursor / peptide pivot table
- 你希望先保留 peptide 层导入，再显式完成 protein 聚合

```python
from scptensor.impute import impute_row_median
from scptensor.io import aggregate_to_protein, load_diann
from scptensor.normalization import norm_median
from scptensor.transformation import log_transform

container = load_diann(
    "diann_report.tsv",
    level="peptide",
    table_format="long",
    assay_name="peptides",
)

container = aggregate_to_protein(
    container,
    source_assay="peptides",
    source_layer="raw",
    target_assay="proteins",
    method="top_n",
)

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
container = impute_row_median(
    container,
    assay_name="proteins",
    source_layer="norm",
    new_layer_name="imputed",
)
```

如果你的输入已经是 peptide/precursor pivot matrix，可以把第一步换成：

```python
from scptensor.io import load_peptide_pivot

container = load_peptide_pivot(
    "spectronaut_precursor_matrix.tsv",
    software="spectronaut",
    assay_name="peptides",
)
```

然后继续同样的 `aggregate_to_protein()` 和 protein 主线步骤。

## 5. 导出稳定 protein matrix

如果你想把当前主线结果导出成可直接交给下游脚本或其他工具的文件，不需要自己再手拆 `obs / var / X`：

```python
from scptensor.utils import protein_matrix_to_table, write_protein_matrix_bundle

matrix_table = protein_matrix_to_table(
    container,
    assay_name="proteins",
    layer="imputed",
    feature_id_column="protein_id",
)

paths = write_protein_matrix_bundle(
    container,
    "exports/protein_mainline",
    assay_name="proteins",
    layer="imputed",
    feature_id_column="protein_id",
)
```

这会得到三份文件：

- `protein_matrix.tsv`
- `sample_metadata.tsv`
- `protein_metadata.tsv`

其中 `protein_matrix.tsv` 的方向是：

- 行 = protein
- 列 = sample
- 首列 = protein ID

如果你当前只想导出未填补版本，可以把 `layer="imputed"` 改成 `layer="norm"` 或其他明确存在的 protein layer。

## 6. 如何确认你已经走在稳定主线上

最小自检建议：

```python
print(container.summary())
print(container.list_assays())
print(container.list_layers("proteins"))
print([record.action for record in container.history])
```

你通常应看到：

- `proteins` assay 存在
- `raw`, `log`, `norm`, `imputed` 等层按顺序生成
- 如果从 peptide/precursor 起步，history 中应出现 `aggregate_to_protein`

## 7. 何时使用 `load_quant_table()`

如果你在写更通用的项目封装，而不想在外层代码里分支 vendor-specific 入口，可以直接使用统一入口：

```python
from scptensor.io import load_quant_table

container = load_quant_table(
    "input.tsv",
    software="diann",
    level="protein",
    table_format="auto",
)
```

推荐规则：

- 面向最终用户脚本时，优先 `load_diann()` / `load_spectronaut()`，更直观
- 面向你自己的封装层或批处理逻辑时，再考虑 `load_quant_table()`

## 8. 当前不要把这些能力当成稳定主线

下列能力可以存在于仓库中，但不应被当成当前 release acceptance 的主目标：

- `scptensor.experimental` 下的降维 / 聚类 exploratory helper
- benchmark 脚本与 benchmark 输出
- 面向算法研究的额外 stress path

当前 repo 的核心判断标准是：

> 一个用户能否安装 ScpTensor，导入 DIA-NN 或 Spectronaut 定量表，最终得到可解释、可追踪的 protein-level quantitative matrix。
