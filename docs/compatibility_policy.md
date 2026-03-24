# Compatibility Policy

## Scope

本文档定义 ScpTensor 稳定表面的兼容策略。默认立场不是“尽量保留旧行为”，而是“只保留有边界、可解释、契约化的兼容”。

该策略适用于以下稳定区域:

- `scptensor.core`
- `scptensor.io`
- `scptensor.aggregation`
- `scptensor.transformation`
- `scptensor.normalization`
- `scptensor.impute`
- `scptensor.integration`
- `scptensor.standardization`
- `scptensor.qc`

experimental downstream helpers 不在本策略的稳定兼容承诺范围内。

## Default Rule

除非某行为同时满足以下条件，否则默认不保留兼容:

1. 它属于项目契约或稳定文档中明示的公共行为。
2. 它不会把隐式推断变成默认真相。
3. 它不会让核心领域模型为兼容层让路。
4. 它有明确、有限、可测试的边界。
5. 删除它时，会破坏真实支持的用户路径，而不是历史偶然行为。

“已有用户可能在用”本身不构成保留理由。若一个行为没有契约依据，且继续保留只会扩大维护面，应优先删除。

## Allowed Compatibility

当前允许保留的兼容仅包括以下有界别名:

- assay 名称别名:
  - `protein` / `proteins`
  - `peptide` / `peptides`
- normalization dispatcher 的方法名别名:
  - `median` / `norm_median`
  - `mean` / `norm_mean`
  - `quantile` / `norm_quantile`
  - `trqn` / `norm_trqn`
  - `none` / `norm_none`

保留原则:

- alias 必须语义等价，不能引入新语义。
- alias 必须集中在单点解析，不允许到处散落 fallback。
- alias 集合默认冻结，新增 alias 需要明确文档和测试理由。

## Disallowed Compatibility

以下兼容模式明确禁止:

- 用容器级 shortcut 伪装多 assay 语义
- 依赖“第一个 assay”之类的插入顺序约定
- 在公共类上保留已不支持的 stub API
- 用 `__module__` 伪装、导出伪装等方式制造错误的实现身份
- 跨 stage convenience 混淆 I/O、aggregation、preprocessing 边界
- 无文档的 heuristic fallback、silent fallback、best-effort fallback
- 为了少改旧代码而继续扩大根包平铺导出

## Root Package Rule

顶层 `scptensor` 只保留少量高频、契约明确的稳定入口。

当前根包稳定表面应限制在：

- 核心数据模型：
  - `ScpContainer`
  - `Assay`
  - `ScpMatrix`
- vendor-specific I/O convenience：
  - `load_diann`
  - `load_spectronaut`
  - `load_peptide_pivot`
- aggregation：
  - `aggregate_to_protein`

其余稳定能力应通过明确子包导入，例如：

- `scptensor.normalization`
- `scptensor.transformation`
- `scptensor.impute`
- `scptensor.integration`
- `scptensor.standardization`
- `scptensor.qc`
- `scptensor.viz`
- `scptensor.utils`

## Subpackage Root Rule

稳定子包的根命名空间也必须收敛，只保留 stage 的主操作入口；registry、diagnostics、selection heuristic 等辅助能力必须走显式子模块路径。

当前应遵循：

- `scptensor.impute`
  - 保留：`impute` 与各 `impute_*` 方法
  - 走显式子模块：`scptensor.impute.base.list_impute_methods`、`infer_missing_mechanism`、`recommend_impute_method`
- `scptensor.integration`
  - 保留：`integrate` 与各 `integrate_*` 方法
  - 走显式子模块：`scptensor.integration.base` 中的 registry / metadata helper
  - 走显式子模块：`scptensor.integration.diagnostics` 中的 batch metric / report helper
- `scptensor.viz`
  - 保留：canonical `plot_*` API 与少量无歧义 base primitive
  - 不再把 recipe alias 或兼容 scatter 名继续上浮到顶层

如果一个子包根表面的价值主要来自“少写一个子模块名”，但代价是混淆主操作、诊断工具与兼容 alias 的边界，默认拒绝。

## Core Boundary Rules

### `ScpContainer`

`ScpContainer` 是多 assay 容器，不承诺单一 feature 维度。因此:

- 不提供容器级 `shape`
- 不提供容器级 `n_features`
- feature 维度必须通过显式 assay 访问

推荐访问方式:

- `container.assay_shape("proteins")`
- `container.assays["proteins"].n_features`
- `container.assays["proteins"].layers["raw"].X.shape`

### Persistence

`ScpContainer` 不承诺通用对象持久化接口。因此:

- 不提供 `ScpContainer.save()`
- 不提供 `ScpContainer.load()`

稳定 I/O 边界是:

- `scptensor.io.load_quant_table`
- `scptensor.io.load_diann`
- `scptensor.io.load_spectronaut`
- `scptensor.io.load_peptide_pivot`
- `scptensor.io.aggregate_to_protein`

### Public Import Identity

`scptensor.core.structures` 仍是稳定导入路径，但它是 public facade，不是实现身份伪装层。因此:

- 可以从 `scptensor.core.structures` 导入公共类
- 不保证实例类型的 `__module__` 显示为 `scptensor.core.structures`
- 回归测试应锁定导入可用性，不应锁定 `__module__` 伪装

## I/O Boundary Rules

I/O 层只负责:

- vendor quant-table 读取
- vendor profile 解释
- 组装 `ScpContainer`

I/O 层不负责:

- 自动触发 peptide -> protein 聚合
- 用 convenience 参数跨越 stage 边界
- 隐式扩展到非 DIA-NN / Spectronaut 软件

## Documentation Rule

仓库文档必须优先描述 canonical path，而不是兼容 path。

例如:

- 写 `proteins / peptides`，而不是混写多个别名集合
- 写 `assay_name + source_layer` 的显式访问方式
- 不再把不存在或已移除的兼容接口当作“兼容 helper”推荐给读者

## Test Rule

兼容测试只能锁定稳定契约，不能锁定历史包袱。以下测试方向应避免:

- 锁定 `container.shape` 之类错误 shortcut
- 锁定 `__module__ == "scptensor.core.structures"`
- 锁定已移除 stub API 的继续存在

应优先测试:

- canonical public import path 是否可用
- alias 是否集中且有边界
- I/O、aggregation、preprocessing 的边界是否清晰
- 显式 assay/layer 访问是否正常工作
