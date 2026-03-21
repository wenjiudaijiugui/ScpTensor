# ScpTensor Experimental Helper 对齐计划（2026-03-17）

## 1. 文档目标

本文档不是合同，而是 experimental helper 收口过程的 convergence record。

基于当前仓库状态：

- 其中原有 P0/P1/P2 任务已完成或已定案
- 本文档不再代表 active backlog
- 但在仓库出现 archive / internal-plan 分组前，仍保留它作为收口过程记录

它把当前 experimental helper 里已经识别出的实现不对称点，拆成后续可逐项收口的任务，覆盖：

- `scptensor.dim_reduction`
- `scptensor.cluster`
- `scptensor.experimental`
- `scptensor.experimental.qc_psm`

目标不是立即统一一切，而是区分：

- 哪些差异是需要修复的工程不一致
- 哪些差异是应保留的边界设计

## 1.1 状态更新（2026-03-20）

本计划中的以下任务已完成并已同步进入正式合同：

- assay alias 解析在 `dim_reduction / cluster / qc_psm` 的验证与执行路径中对齐
- `reduce_* / cluster_*` 的 provenance `params` 命名统一为 source/output schema
- copy / source-mutation 语义已完成第一阶段冻结：合同已明确共享粒度，测试可直接观察对象身份
- `scptensor.experimental` facade 的完整 re-export 测试已补齐
- `qc_psm` 的 PIF / q-value 阈值已落实为真实 `[0, 1]` 校验
- `qc_psm` 的 namespace 位置已定案：保留在 `scptensor.experimental`，并明确解释为 experimental pre-aggregation helper

当前没有必须继续收口的 experimental helper 合同缺口。
后续若继续推进，只剩可选重构议题，例如是否统一 copy 策略。

## 2. 非目标

本文档当前不推动：

- 把 experimental helper 升格为 stable top-level API
- 把 reduction / clustering 纳入 stable preprocessing release 验收
- 改变 protein-level preprocessing 主线边界

## 3. 当前问题分型

### 3.1 需要收口的工程不一致

- 当前无必须继续收口的工程不一致

### 3.2 应保留的边界差异

- reduction 结果写入 new assay / layer `X`
- clustering 结果写入 `obs`
- experimental API 不上浮到顶层 `scptensor`

这些是边界设计，不是当前计划中的“收口对象”。

## 4. 优先级任务

### P0. 统一 assay 解析语义（已完成，2026-03-20）

完成状态：

- `dim_reduction.base._validate_assay_layer()` 与 `cluster.base._validate_assay_layer()`
  现在都使用 `resolve_assay_name()`
- `qc_psm` 在验证、过滤执行与 provenance 记录阶段都统一使用 resolved assay name
- 不再存在“验证通过、执行失败”的 alias 半打通路径

涉及文件：

- `scptensor/dim_reduction/base.py`
- `scptensor/cluster/base.py`
- `scptensor/qc/_utils.py`
- `scptensor/qc/qc_psm.py`
- `tests/dim_reduction/*`
- `tests/cluster/*`
- `tests/qc/*`

验收标准：

- `proteins/protein`、`peptides/peptide` 等 alias 在 experimental helper 中行为一致
- 不再出现“验证通过、执行失败”的 alias 半打通情况

### P0. 统一 provenance 参数命名（已完成，2026-03-20）

完成状态：

- 输入 assay -> `source_assay`
- 输入 layer -> `source_layer`
- 输出 assay -> `target_assay`（若有）
- 输出 obs key -> `output_key`（若有）

涉及文件：

- `scptensor/dim_reduction/pca.py`
- `scptensor/dim_reduction/tsne.py`
- `scptensor/dim_reduction/umap.py`
- `scptensor/cluster/kmeans.py`
- `scptensor/cluster/graph.py`

验收标准：

- action 名可以保持不变
- 但同类参数键名不再漂移

### P1. 收口 copy / source-mutation 语义（第一阶段已完成，2026-03-20）

完成状态：

- `reduce_pca / reduce_tsne / reduce_umap / cluster_*` 的对象共享粒度
  已在正式合同中逐项写明
- 相关测试已直接断言：
  - container 是否新建
  - `obs` 是否共享
  - `history` 是否共享
  - assay mapping 是否共享
  - source assay / source layer matrix 是否共享

后续可选目标：

- 明确选择一种统一策略，并写入合同与测试：
  - 方案 A：全 experimental helper 统一 shallow-copy container + explicit source mutation
  - 方案 B：全 experimental helper 统一 no-source-mutation unless declared
  - 方案 C：保留差异，但通过命名和测试明确“哪些 helper 会写回 source assay”

当前状态：

- 方案 C 已完成
- 是否继续推进到 A/B，属于新的行为变更议题

涉及文件：

- `scptensor/dim_reduction/pca.py`
- `scptensor/dim_reduction/tsne.py`
- `scptensor/dim_reduction/umap.py`
- `scptensor/cluster/base.py`
- `tests/dim_reduction/*`
- `tests/cluster/*`

验收标准：

- 所有 helper 的 copy / mutation 语义都能被测试直接观察

### P1. 扩展 `scptensor.experimental` facade 测试（已完成，2026-03-20）

完成状态：

- `tests/core/test_experimental_api.py` 已覆盖 `experimental.__all__` 的完整 exported names
- `SolverType`、`reduce_tsne`、`reduce_umap`、`cluster_leiden` 等此前缺失项已补入
- 顶层 `scptensor` 不导出 experimental API 的负向测试也已完整

涉及文件：

- `tests/core/test_experimental_api.py`

验收标准：

- `experimental.__all__` 中所有条目都有正向导出测试
- 顶层 `scptensor` 不导出 experimental API 的负向测试也完整

### P1. 修正 `qc_psm` 阈值校验与真实行为不一致（已完成，2026-03-20）

完成状态：

- `filter_psms_by_pif()` / `filter_psms_by_qvalue()` 现在都显式传入
  `min_val=0.0, max_val=1.0`
- 文档、错误语义、实际校验已对齐
- 非法阈值已有专项回归测试

涉及文件：

- `scptensor/qc/qc_psm.py`
- `tests/qc/test_qc.py`

验收标准：

- 非法阈值会稳定抛错
- 错误类型与错误信息可测试

### P2. 决定 `qc_psm` 在 experimental namespace 内的长期位置（已完成，2026-03-20）

完成状态：

- `qc_psm` 继续保留在 `scptensor.experimental`
- 用户级 canonical 入口继续是 `from scptensor.experimental import qc_psm`
- 文档与模块说明已明确：它是 experimental pre-aggregation helper
- 不把它迁移到 stable `scptensor.qc`
- 当前也不继续拆出新的 experimental 子命名空间

涉及文件：

- `scptensor/experimental/__init__.py`
- `README.md`
- `docs/qc_psm_contract.md`
- `docs/experimental_downstream_contract.md`

## 5. 明确保留、不做收口的差异

以下差异当前应保留，不列入修复目标：

### 5.1 reduction -> assay，cluster -> obs

这是当前正确的结果槽位边界：

- reduction 坐标是 assay-level表示
- cluster labels 是 sample-level annotation

### 5.2 experimental API 不上浮到顶层 `scptensor`

这是仓库的发布边界，不是实现缺口。

### 5.3 stable preprocessing release 不依赖 experimental helper

这是总合同的一部分，不应因 helper 完善而被改变。

## 6. 建议执行顺序

建议后续按以下顺序推进：

1. 若确有工程收益，再单独立项讨论是否统一 experimental helper 的 copy 策略
2. 若 `qc_psm` 将来扩张为更大的一组 peptide-PSM experimental API，再重新评估是否需要独立子命名空间

## 7. 与现有合同的关系

本计划文档是后续整改路线，不替代：

- `docs/experimental_downstream_contract.md`
- `docs/qc_psm_contract.md`

如果计划实施后代码行为改变，应先更新对应合同，再改示例与教程。
