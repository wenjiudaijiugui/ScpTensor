# ScpTensor 文档驱动优化执行清单（2026-03-17）

## 1. 文档定位

本文档是 **execution checklist**，用于约束后续代码优化、重构、性能改进与测试补强的执行方式。

它的作用不是替代合同文档，而是把“应先读什么、按什么边界动手、每个 PR 必须交付什么”写成统一执行规范。

本文档适用于：

- `core`
- `io`
- `aggregation`
- `transformation`
- `normalization`
- `impute`
- `integration`
- `qc`
- `autoselect`
- `utils`
- `viz`
- `experimental` 边界收口

本文档不替代：

- 仓库总合同 `AGENTS.md`
- `docs/*_contract.md`
- `docs/review_*.md`

若本文档与上述文档冲突，以上述文档为准。

## 2. Source-of-Truth 优先级

后续所有代码优化必须按以下优先级读取和执行：

1. `AGENTS.md`
2. 对应模块的 `docs/*_contract.md`
3. 对应主题的 `docs/review_*.md`
4. `benchmark/README.md` 与各子目录 README
5. `tutorial/README.md` 与 notebook
6. 当前源码实现

执行规则：

1. 若 `AGENTS.md` 与其他文档冲突，遵循 `AGENTS.md`。
2. 若 contract 与 review 冲突，遵循 contract。
3. 若 benchmark / tutorial 与 contract 冲突，遵循 contract，并同步修正文档导航或说明。
4. 若源码实现与 contract 冲突，不得直接把“当前代码行为”继续扩散成新优化基线；必须先确认是代码漂移还是合同过时。

## 3. 冲突处理规则

发现以下任一情况时，停止继续做“性能优化”，先做对齐判断：

1. 代码当前行为与 contract 不一致。
2. 同一模块的 history action / params 键名与 contract 不一致。
3. benchmark README 对外叙述与 contract 边界不一致。
4. tutorial 把 experimental helper 叙述成 stable mainline。
5. 优化方案要求更改 stable layer naming、missingness semantics、detection semantics、copy/mutation semantics、overwrite semantics 之一。

处理顺序：

1. 先确认冲突来源。
2. 若是代码偏离合同，优先修代码。
3. 若是合同明确写成“当前实现事实”，且目标是改变该事实，则先改 contract，再改代码和测试。
4. 若只是 benchmark / tutorial 漂移，则同步修正文档，不反向定义稳定实现。

## 4. 进入任何优化 PR 之前的必做检查

每个优化 PR 在开工前都必须完成以下清单：

- [ ] 明确本 PR 触及的模块范围。
- [ ] 列出本 PR 的 authority docs。
- [ ] 从 contract 中摘出本 PR 不得破坏的 frozen invariants。
- [ ] 标注本 PR 处理的是 stable 模块还是 experimental 模块。
- [ ] 判断本 PR 是否涉及文档同步。
- [ ] 判断本 PR 是否需要新增或扩充 targeted regression tests。
- [ ] 判断本 PR 是否需要 runtime / memory 基线对比。
- [ ] 判断本 PR 是否会碰到 history / provenance / layer naming / overwrite semantics。

若以上任一项无法明确，不得直接进入实现。

## 5. 每个优化 PR 的固定模板

后续建议把每个优化 PR 说明统一为以下结构：

### 5.1 Authority Docs

必须列出：

- `AGENTS.md`
- 对应 `docs/*_contract.md`
- 必要时列出相关 `docs/review_*.md`
- 若涉及 benchmark / tutorial 对齐，再列出对应 README

### 5.2 Frozen Invariants

只写本 PR 绝对不能破坏的行为，例如：

- 层级结构
- shape/dtype 语义
- `MaskCode` 语义
- history action / params
- assay / layer naming
- stable / experimental 边界

### 5.3 Planned Scope

只列：

- 触及文件
- 计划新增测试
- 计划新增 benchmark / profiling

### 5.4 Verification Gates

至少包括：

- targeted regression tests
- runtime baseline 前后对比
- memory / densify 风险检查
- docs sync 检查
- changed files 的 link / navigation 检查

## 6. 阶段化执行路线

### 6.1 `PR-0` 运行时基线

目标：

- 为后续优化建立 runtime benchmark、peak memory、densify 路径、copy 路径的可比较基线

Authority docs：

- `AGENTS.md`
- `docs/README.md`
- `benchmark/README.md`

允许做的事：

- 新增运行时 profiling / perf 脚本
- 为稳定主链设计最小可重复基线数据流

不得做的事：

- 不把 runtime benchmark 混写成科学方法 benchmark
- 不借此改变任何 stable 语义

### 6.2 `PR-1` Core 数据对象层

目标：

- 收敛 `ScpContainer -> Assay -> ScpMatrix` 的实现组织
- 降低 `structures.py` 的单文件复杂度

Authority docs：

- `AGENTS.md`
- `docs/core_data_contract.md`

冻结约束：

- 保持三层结构不变
- 保持显式 assay / layer 访问为 stable 主路径
- 不把 `Assay.X`、`ScpContainer.shape`、`n_features` 升格为核心接口
- 不引入 AnnData 式 view/backed 合同

### 6.3 `PR-2` Core 低层计算面

目标：

- 优化 `matrix_ops / sparse_utils / jit_ops`
- 降低 dense/sparse 分支重复
- 提升稀疏 mask 相关热点性能

Authority docs：

- `AGENTS.md`
- `docs/core_compute_contract.md`

冻结约束：

- `X.shape == (n_samples, n_features)`
- `M.shape == X.shape`
- `M is None` 的默认语义仍是全 `VALID`
- 稀疏未存储位置仍解释为 `VALID`
- `MatrixOps` 仍保持非原地语义
- `MaskCode` 数值不变

### 6.4 `PR-3` I/O 与 Aggregation 主入口

目标：

- 拆分 `load_quant_table()` 的大函数结构
- 优化 long / matrix 解析路径
- 优化 peptide/precursor -> protein 聚合主链

Authority docs：

- `AGENTS.md`
- `docs/io_diann_spectronaut.md`
- `docs/aggregation_contract.md`
- `docs/review_io_state_mapping_20260312.md`

冻结约束：

- 支持输入仍限于 DIA-NN / Spectronaut
- `aggregation` 仍是唯一 stable 的 feature-universe change stage
- importer provenance 不能丢
- vendor-normalized 语义不能漂移
- `AggregationLink` 语义不能被隐式缩减

### 6.5 `PR-4` Transformation 与 Normalization 公共底座

目标：

- 统一校验、history、warning、layer 写入的 shared helper
- 收敛 transform / normalization 的共用执行框架

Authority docs：

- `AGENTS.md`
- `docs/transformation_contract.md`
- `docs/normalization_contract.md`
- `docs/review_log_scale_20260312.md`
- `docs/review_normalization_20260307.md`

冻结约束：

- `log_transform` 与 `normalization` 仍是两个阶段
- 不得在 normalization 中隐式做 log
- `quantile / trqn` 的 logged-layer gate 不得无声放宽
- importer 与 normalization 之间的 vendor-normalized provenance 协作不得破坏
- 当前 overwrite / history 语义若改变，必须显式升级文档

### 6.6 `PR-5` Imputation 与 Integration

目标：

- 优化 registry / dispatch / data prep
- 优化完整矩阵校验与方法级分流
- 降低重复 densify / copy

Authority docs：

- `AGENTS.md`
- `docs/imputation_contract.md`
- `docs/integration_contract.md`
- `docs/review_imputation_20260304.md`
- `docs/review_masked_imputation_20260312.md`
- `docs/review_batch_correction_20260305.md`
- `docs/review_batch_diagnostics_20260312.md`

冻结约束：

- 缺失判定仍以 `np.isnan(X)` 为准
- 不得把 `0` 偷偷并入缺失语义
- 原始 finite 值逐元素不变
- `M` 更新规则不变
- stable integration 默认候选集不得静默扩大
- embedding-level 方法不得伪装成 stable protein-level output

### 6.7 `PR-6` QC 与 Utils

目标：

- 收敛 QC 统计工具、过滤路径与 provenance 路径
- 收敛 `utils` 中已形成 public surface 的 helper 组织

Authority docs：

- `AGENTS.md`
- `docs/qc_contract.md`
- `docs/utils_contract.md`
- `docs/review_qc_filtering_20260312.md`
- `docs/review_batch_diagnostics_20260312.md`

冻结约束：

- protein-level QC 主线边界不变
- detection semantics 不变
- `VALID` zero 仍按当前合同解释
- filter provenance 不得丢
- `utils` 不得反向替代阶段模块语义

### 6.8 `PR-7` AutoSelect

目标：

- 先做缓存、结果复用、重复评估聚合
- 再考虑真正的并行执行

Authority docs：

- `AGENTS.md`
- `docs/autoselect_contract.md`
- `docs/review_autoselect_scoring_20260312.md`

冻结约束：

- `overall_score` 与 `selection_score` 必须语义分离
- runtime 只能影响 `selection_score`
- `StageReport.results` 必须保留成功与失败方法
- `parallel / n_jobs` 在真正实现前不得被视为已生效保证
- `reduce / cluster` 不得被上升为 stable preprocessing core

### 6.9 `PR-8` Experimental Helper 收口

目标：

- 先修正 experimental helper 的不一致点
- 再考虑其性能和结构优化

Authority docs：

- `AGENTS.md`
- `docs/experimental_downstream_contract.md`
- `docs/experimental_downstream_alignment_plan.md`
- `docs/qc_psm_contract.md`

冻结约束：

- experimental 结果不得反向定义 stable preprocessing 完成判据
- `reduce / cluster` 继续留在 experimental namespace 边界
- `qc_psm` 继续视为 experimental pre-aggregation helper

## 7. 模块到 authority docs 的快速映射

| 模块 | 首要合同 | 必要 review / 辅助文档 |
|---|---|---|
| `core` 数据对象 | `docs/core_data_contract.md` | `docs/io_diann_spectronaut.md` |
| `core` 低层计算 | `docs/core_compute_contract.md` | `tests/core/*` 对应回归 |
| `io` | `docs/io_diann_spectronaut.md` | `docs/review_io_state_mapping_20260312.md` |
| `aggregation` | `docs/aggregation_contract.md` | `docs/aggregation_literature.md`, `docs/review_aggregation_benchmark_20260312.md` |
| `transformation` | `docs/transformation_contract.md` | `docs/review_log_scale_20260312.md` |
| `normalization` | `docs/normalization_contract.md` | `docs/review_normalization_20260307.md`, `docs/review_log_scale_20260312.md` |
| `standardization` | `docs/standardization_contract.md` | `docs/review_zscore_standardization_20260313.md` |
| `impute` | `docs/imputation_contract.md` | `docs/review_imputation_20260304.md`, `docs/review_masked_imputation_20260312.md` |
| `integration` | `docs/integration_contract.md` | `docs/review_batch_correction_20260305.md`, `docs/review_batch_diagnostics_20260312.md`, `docs/review_batch_confounding_20260312.md` |
| `qc` | `docs/qc_contract.md` | `docs/review_qc_filtering_20260312.md`, `docs/review_batch_diagnostics_20260312.md` |
| `qc_psm` | `docs/qc_psm_contract.md` | `docs/experimental_downstream_alignment_plan.md` |
| `autoselect` | `docs/autoselect_contract.md` | `docs/review_autoselect_scoring_20260312.md` |
| `utils` | `docs/utils_contract.md` | `tests/utils/*` |
| `viz` | `docs/viz_contract.md` | 相关 stage contract + `tests/viz/*` |
| `dim_reduction / cluster` | `docs/experimental_downstream_contract.md` | `docs/experimental_downstream_alignment_plan.md` |

## 8. 禁止直接越过文档边界的事项

后续优化中，以下事项禁止“顺手修改”：

1. 改 stable 主线终点定义。
2. 扩支持输入到 DIA-NN / Spectronaut 之外。
3. 修改 canonical assay / layer naming。
4. 修改 history action names 或最小 params 键名。
5. 修改 overwrite 语义但不更新合同。
6. 修改 detection semantics / missingness semantics / mask semantics 但不更新合同。
7. 把 experimental helper 提升为 stable contract。
8. 把 benchmark README 当作稳定实现合同来覆盖 contract。

## 9. 文档同步规则

若某次优化只改变内部组织、性能、局部 helper，而不改变稳定语义：

- 可以不更新 contract 正文
- 但 PR 说明必须显式声明“contract unchanged”

若某次优化改变了任何稳定可观察行为：

- 必须同步更新对应 `docs/*_contract.md`
- 若变更理由依赖外部证据，也应同步更新相关 `review_*.md`
- 若 benchmark / tutorial 叙述会因此漂移，也必须同步修正

若某次优化只修正 benchmark 或 tutorial 漂移：

- 不得反向修改 contract 去迎合漂移

## 10. 统一验收门禁

每个优化 PR 完成前，至少要交付以下证据：

- [ ] 对应 authority docs 已列出
- [ ] frozen invariants 已逐条核对
- [ ] targeted regression tests 已执行
- [ ] 新增或修改的性能基线已记录
- [ ] docs navigation 已同步
- [ ] 修改过的 Markdown 链接已本地检查
- [ ] 若行为变化影响 contract，文档已同步更新

## 11. 一句话执行结论

ScpTensor 后续代码优化必须坚持：

1. 先读合同，再改代码。
2. 先锁不变量，再做性能优化。
3. benchmark 与 tutorial 只做补充，不得反向定义 stable 语义。
4. 遇到语义漂移先对齐，再优化，不得带着冲突继续重构。
