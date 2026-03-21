# ScpTensor Experimental PSM QC Helper 合同（`scptensor.experimental.qc_psm`，2026-03-17）

## 状态更新（2026-03-20）

`qc_psm` 在 experimental namespace 内的长期位置现已定案：

- 继续保留在 `scptensor.experimental`
- 继续以模块对象 `qc_psm` 的形式暴露
- 明确解释为 experimental pre-aggregation / peptide-PSM QC helper
- 不迁移到 stable `scptensor.qc`
- 不与 downstream `reduce_* / cluster_*` 混写成同类 helper
- 默认 assay naming 已统一到 canonical `peptides`，`peptide` 保留为兼容 alias

## 1. 文档目标

本文档冻结 `qc_psm` 当前实现边界，服务于后续：

- peptide / PSM 级 experimental helper 的文档收口
- 与 stable protein-level QC 主线彻底分界
- 后续是否提升、迁移或重命名的判断依据

它回答的是：

- `qc_psm` 当前通过什么命名空间对用户可见；
- 它与 `scptensor.qc` stable contract 的边界在哪里；
- 当前有哪些 module-level helper 已经形成可观察公共面；
- 哪些实现不对称点需要如实记录，而不能被误写成“已统一”。

本文档基于以下仓库内事实：

- `scptensor/qc/__init__.py`
- `scptensor/qc/qc_psm.py`
- `scptensor/qc/_utils.py`
- `scptensor/qc/metrics.py`
- `scptensor/experimental/__init__.py`
- `scptensor/__init__.py`
- `tests/qc/test_qc.py`
- `tests/core/test_experimental_api.py`
- `docs/qc_contract.md`
- `docs/experimental_downstream_contract.md`

## 2. 范围与非范围

### 2.1 范围

本文档覆盖 `qc_psm.py` 当前定义的 helper：

- `filter_psms_by_pif`
- `filter_contaminants`
- `pep_to_qvalue`
- `filter_psms_by_qvalue`
- `compute_sample_cv`
- `compute_median_cv`

### 2.2 非范围

本文档不覆盖：

- protein-level stable QC 主线
- sample / feature QC 的稳定实现合同
- peptide -> protein aggregation
- PSM 级 benchmark 结论

## 3. 发布边界与命名空间合同

### 3.1 `qc_psm` 不属于 stable `scptensor.qc` 主线

`scptensor.qc.__all__` 当前不导出：

- `qc_psm`

并且 `scptensor.qc.__init__` 的模块说明已明确：

- peptide / PSM-level QC helper 保留在源码中
- 但不属于 stable preprocessing contract

因此 `qc_psm` 的发布边界必须冻结为：

- experimental helper
- pre-aggregation / peptide-PSM-level helper
- 不是 stable protein-level QC 主线的一部分

### 3.2 用户可见入口

当前用户级 canonical 入口应解释为：

```python
from scptensor.experimental import qc_psm
```

这里导入到的是模块对象，而不是函数集合。

### 3.3 非入口路径

当前以下路径不应被写成用户文档的 canonical 入口：

- `import scptensor as scp; scp.qc_psm`
- `from scptensor.qc import qc_psm`

原因：

- 顶层 `scptensor` 不导出它
- stable `scptensor.qc` 也不导出它

### 3.4 `experimental` namespace 的语义

`qc_psm` 被放在 `scptensor.experimental` 下，并不意味着它是“downstream embedding/clustering helper”。

更准确的当前边界是：

- 它属于 experimental namespace
- 但语义位置是 pre-aggregation / peptide-PSM-level helper
- 不是 protein-level downstream display helper

### 3.5 当前 namespace 决策

当前正式决策是：

- 保留 `qc_psm` 在 `scptensor.experimental`
- 不把它迁移到 `scptensor.qc`
- 当前也不再拆出新的 experimental 子命名空间

这样做的理由是：

- 它不属于 stable protein-level QC 主线
- 但它仍然是用户可见的 experimental helper
- 与 `reduce_* / cluster_*` 共处于 `experimental`，可以统一表达“非 stable 主线”的发布边界
- 同时通过合同明确它的语义是 pre-aggregation，而不是 downstream

## 4. 当前公开面与测试覆盖

### 4.1 模块级公共面

`qc_psm.py` 现在有显式 module-level `__all__`，当前冻结的公共面是：

- `filter_psms_by_pif`
- `filter_contaminants`
- `pep_to_qvalue`
- `filter_psms_by_qvalue`
- `compute_sample_cv`
- `compute_median_cv`

这意味着当前公开面不再依赖“源文件里恰好有哪些顶层名字”，而是依赖显式 export list。
当前内置 contaminant regex 列表不再属于 module public surface。

### 4.2 已直接测试的部分

当前直接有测试覆盖的主要是：

- `filter_contaminants`
- `filter_psms_by_pif`
- `filter_psms_by_qvalue`
- `pep_to_qvalue` 的 method / `lambda_param` 失败路径、NaN 保留与 `[0, 1]` 边界
- `compute_sample_cv` 的 alias 解析、canonical `obs` 输出列、history action，以及 layer 缺失时报 `ScpValueError`
- `compute_median_cv` 作为 legacy compatibility alias 的 warning、旧列名与旧 action 保留
- `scptensor.experimental` 对 `qc_psm` 模块对象的重导出

### 4.3 已补基础专项回归，但仍不等于 stable 主线级锁定

以下函数现在已经具备基础专项回归：

- `pep_to_qvalue`
- `compute_sample_cv`
- `compute_median_cv`

但它们仍应被写成“当前 experimental public helper”，而不是“已像 stable 主线那样完全测试锁定的接口”。
原因是：

- 其当前语义仍以“冻结实现事实”为主，而不是长期抽象后的 stable API
- `compute_median_cv()` 本身现在只是 compatibility alias，而不是推荐新入口

## 5. 当前输入与输出合同

### 5.1 assay 语义

大多数 `qc_psm` helper 默认：

- `assay_name="peptides"`

当前实现通过 `resolve_assay()` 调用 `resolve_assay_name()`，并在真正执行过滤时也使用
同一个 resolved assay name。

因此当前稳定解释是：

- 默认参数已经使用仓库 canonical naming `peptides`
- 历史兼容写法 `peptide` 仍被接受
- `peptide / peptides` alias 会在验证、过滤执行与 provenance 记录三个阶段保持一致

### 5.2 过滤类函数

以下函数都是“按 feature 过滤 assay”的容器级 helper：

- `filter_psms_by_pif`
- `filter_contaminants`
- `filter_psms_by_qvalue`

它们的当前稳定行为是：

- 根据 `assay.var` 中某列构造 keep mask
- 转成 feature keep-indices
- 对 resolved assay name 调用 `container.filter_features(...)`
- 返回新的 `ScpContainer`
- 追加 provenance log

对 `filter_contaminants()` 还需额外冻结一条边界：

- 稳定入口是 `patterns=None` 时使用模块内置默认污染物模式
- 但“内置默认 regex 列表本身”不是当前 public contract
- 后续若需要自定义或审计，调用方应显式传入自己的 `patterns`

也就是说，函数默认行为是合同的一部分，但 `_DEFAULT_CONTAMINANT_PATTERNS` 的精确列表不是。

### 5.3 纯数组 helper

`pep_to_qvalue()` 当前是纯 NumPy helper：

- 输入一维 `pep` 数组
- 输出一维 `qvalue` 数组
- 不读写 `ScpContainer`

因此它在 `qc_psm` 里属于“同模块下的纯统计工具”，不是容器级 QC pipeline step。

### 5.4 `compute_sample_cv()`

`compute_sample_cv()` 是当前推荐的 canonical helper。它的行为与前面三类不同：

- 不过滤 feature
- 不新建 assay
- 对输入 `container.copy()`
- 在 `obs` 中新增：
  - `sample_cv`
  - `is_high_sample_cv`

它实际计算的是：

- 每个样本在 feature 维度上的 CV 摘要

因此它是 peptide/PSM-level样本 QC 摘要 helper，而不是 feature filter。

### 5.5 `compute_median_cv()` compatibility alias

`compute_median_cv()` 现在保留为 legacy compatibility alias：

- canonical 推荐入口不再是它
- 它会发出 `FutureWarning`
- 它仍返回新 `ScpContainer`
- 它仍保留历史列名：
  - `median_cv`
  - `is_high_cv`
- 它仍保留历史 provenance action：
  - `compute_median_cv`

这样做的目的不是继续扩大旧命名，而是给已有调用提供有边界的迁移路径。

### 5.6 `compute_median_cv()` 迁移策略

当前迁移策略已经定案，不再留作开放问题：

#### 阶段 A：当前仓库状态

- `compute_sample_cv` 是唯一 canonical helper
- `compute_median_cv` 继续保留在 `qc_psm.__all__`
- 旧 alias 每次调用都发 `FutureWarning`
- warning 必须明确提示：
  - 新代码应改用 `compute_sample_cv`
  - alias 的移除不能静默发生
  - 只能在 future contract update 之后移除

#### 阶段 B：仓内迁移门禁

在仓库仍处于迁移期时，允许保留 alias，但必须满足：

- 新增文档、示例、教程、测试一律优先使用 `compute_sample_cv`
- 旧 alias 只允许出现在：
  - compatibility 测试
  - 迁移说明
  - 明确标注 legacy 的文档段落
- canonical sample-level输出列只写：
  - `sample_cv`
  - `is_high_sample_cv`

#### 阶段 C：最终移除门禁

`compute_median_cv` 只有在以下条件同时满足时才允许删除：

1. `docs/qc_psm_contract.md` 已先更新，明确它不再属于当前公共面
2. `qc_psm.__all__` 已同步移除该名称
3. 仓库内 README / docs / tutorial / tests 不再把它当作可调用入口
4. compatibility 测试已从“warning + 旧列名保留”切换为“名称已移除或调用失败”的负向测试
5. 若仍需兼容旧 `obs` 列名或旧 provenance action，必须在移除前给出单独迁移说明；不得一边删 alias，一边静默改变消费路径

也就是说：

- 允许未来删除
- 但禁止无文档、无测试切换、无合同更新的静默删除

## 6. 当前失败合同

### 6.1 assay / column 失败路径

`qc_psm` 当前主要复用 `qc._utils`：

- `validate_assay()`：
  - assay 不存在 -> `AssayNotFoundError`
- `validate_column_exists()`：
  - 列不存在 -> `ScpValueError`

### 6.2 `pep_to_qvalue()` 的失败路径

当前显式抛错：

- `method` 非 `storey / bh` -> `ValueError`
- `lambda_param` 不在 `[0, 1)` -> `ValueError`

### 6.3 `compute_sample_cv()` / `compute_median_cv()` 的 layer 失败路径

当前这两个 helper 共享同一段 layer 检查逻辑，没有复用 `qc._utils.validate_layer()`，而是在函数内部手写检查。

因此 layer 缺失时当前抛的是：

- `ScpValueError`

而不是 `LayerNotFoundError`。

这是一条必须冻结记录的实现事实。

## 7. provenance 合同

当前会记录的操作名包括：

- `filter_psms_by_pif`
- `filter_contaminants`
- `filter_psms_by_qvalue`
- `compute_sample_cv`
- `compute_median_cv`

其中：

- `compute_sample_cv` 是 canonical action
- `compute_median_cv` 是 compatibility alias 的 legacy action

这些 action 名已经是可观察接口的一部分。后续若改动实现，不应无迁移地更改这些 action 名。

## 8. 当前已知实现不对称点

### 8.1 `compute_median_cv` 仍作为 compatibility alias 暂时保留

当前 canonical 命名已经收口到：

- `compute_sample_cv`

但旧名：

- `compute_median_cv`

仍然保留，用于兼容已有调用。这不是新的 canonical 推荐，而是显式保留的迁移层。其迁移规则见 `5.6`。

### 8.2 内置 contaminant regex 集合仍是实现细节

当前 `filter_contaminants(patterns=None)` 的默认行为会继续使用模块内置污染物模式。

但以下内容当前不被冻结为 public API：

- 内置模式的变量名
- 内置模式的精确顺序
- 内置模式的完整 regex 集合

因此，后续若要微调默认模式集合，应优先检查：

- `filter_contaminants()` 的默认过滤行为是否仍合理
- 相关测试与合同说明是否需要同步

而不是把“模块内部默认 regex 常量”当作稳定入口来维护。

## 9. 优化时不得破坏的高优先级边界

后续若要重构 `qc_psm`，必须优先保留：

1. 它属于 experimental helper，而不是 stable `scptensor.qc` 主线
2. canonical 用户入口是 `from scptensor.experimental import qc_psm`
3. 过滤类函数返回新 `ScpContainer`
4. `pep_to_qvalue()` 保持纯数组 helper 定位
5. provenance action 名的基本稳定性
6. 默认 assay naming 保持 `peptides`，同时继续接受 `peptide` 兼容写法
7. `compute_sample_cv` 作为 canonical helper 写入 `sample_cv / is_high_sample_cv`
8. `compute_median_cv` 只作为 compatibility alias 保留旧列名与旧 action

## 10. 对后续重构的直接指导

基于当前实现，最合理的下一步不是把 `qc_psm` 悄悄塞回 stable `scptensor.qc`，而是先做以下收口：

- 若将来继续清理旧命名，按 `5.6` 的门禁执行 `compute_median_cv` alias 移除

这意味着：`qc_psm` 当前值得单独立约，但它的正确定位仍然是 experimental helper，而不是 stable preprocessing contract 的一部分。
