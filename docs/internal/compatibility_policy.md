# Compatibility Policy

本文档只回答一件事：哪些兼容值得保留，哪些历史行为应直接收口。

## Scope

适用稳定区域：

- `scptensor.core`
- `scptensor.io`
- `scptensor.aggregation`
- `scptensor.transformation`
- `scptensor.normalization`
- `scptensor.impute`
- `scptensor.integration`
- `scptensor.standardization`
- `scptensor.qc`

`experimental` downstream helpers 不在稳定兼容承诺内。

## Default Rule

默认不保留兼容。只有同时满足下列条件才保留：

1. 已被 `AGENTS.md`、contract 或稳定文档明示。
2. 不把隐式推断包装成默认真相。
3. 边界明确、有限、可测试。
4. 删除它会破坏真实支持的用户路径，而不是历史偶然行为。

“可能有人在用”本身不是保留理由。

## Allowed Compatibility

当前允许的有界兼容：

- assay 别名：
  - `protein / proteins`
  - `peptide / peptides`
- normalization 方法别名：
  - `median / norm_median`
  - `mean / norm_mean`
  - `quantile / norm_quantile`
  - `trqn / norm_trqn`
  - `none / norm_none`

规则：

- alias 必须语义等价。
- alias 必须集中单点解析，不能到处散落 fallback。
- 新增 alias 需要文档和测试理由。

## Disallowed Compatibility

明确禁止：

- 容器级 shortcut 伪装多 assay 语义
- 依赖“第一个 assay”之类插入顺序
- 保留已不支持的 stub API
- 用 `__module__`、伪导出身份制造错误实现表象
- 跨 stage convenience 混淆 I/O、aggregation、preprocessing 边界
- undocumented heuristic / silent / best-effort fallback
- 为兼容继续扩大根包平铺导出

## Stable Surface Rules

根包 `scptensor` 只保留少量高频入口：

- `ScpContainer`
- `Assay`
- `ScpMatrix`
- `load_diann`
- `load_spectronaut`
- `load_peptide_pivot`
- `aggregate_to_protein`

其余稳定能力应走明确子包：

- `scptensor.transformation`
- `scptensor.normalization`
- `scptensor.impute`
- `scptensor.integration`
- `scptensor.standardization`
- `scptensor.qc`
- `scptensor.viz`
- `scptensor.utils`

子包根命名空间也应收敛：

- `scptensor.impute`: 保留 `impute` 与 `impute_*`
- `scptensor.integration`: 保留 `integrate` 与 `integrate_*`
- `scptensor.viz`: 保留 canonical `plot_*`

registry、diagnostics、heuristic helper 默认走显式子模块。

## Core And I/O Boundary

- `ScpContainer` 不承诺容器级 `shape` 或 `n_features`
- feature 维度必须经 assay 显式访问
- 不提供通用 `ScpContainer.save()` / `load()`
- 稳定 I/O 入口只限：
  - `scptensor.io.load_quant_table`
  - `scptensor.io.load_diann`
  - `scptensor.io.load_spectronaut`
  - `scptensor.io.load_peptide_pivot`
  - `scptensor.io.aggregate_to_protein`
- I/O 只负责 vendor quant-table 读取、profile 解释、组装 `ScpContainer`
- I/O 不自动触发聚合，不隐式扩展到非 DIA-NN / Spectronaut

## Documentation And Tests

文档优先写 canonical path：

- 写 `proteins / peptides`
- 写显式 `assay_name + source_layer`
- 不把已移除或兼容路径当推荐入口

测试优先锁定：

- canonical public import path
- alias 是否集中且边界清晰
- I/O、aggregation、preprocessing 边界
- 显式 assay/layer 访问

不要锁定：

- `container.shape` 之类错误 shortcut
- `__module__` 伪装
- 已移除 stub API 的继续存在
