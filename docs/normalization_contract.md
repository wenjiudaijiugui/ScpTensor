# ScpTensor 归一化冻结实现合同（当前代码基线）

## 1. 文档目的

本文档冻结 `scptensor.normalization` 的当前实现合同，服务于后续代码优化、重构、测试补强与接口收敛。

它回答的是下面这些实现问题：

- 归一化阶段现在到底覆盖哪些方法。
- 输入和输出 layer 的稳定语义是什么。
- 哪些尺度约束已经被代码或上游文档收紧。
- provenance 至少要留下什么。
- 哪些当前行为即使不完美，也不能在重构时被悄悄改掉。

本文档基于以下仓库内现状收束，不额外扩展文献层结论：

- `scptensor/core/_layer_processing.py`
- `scptensor/normalization/__init__.py`
- `scptensor/normalization/api.py`
- `scptensor/normalization/_context.py`
- `scptensor/normalization/mean_normalization.py`
- `scptensor/normalization/median_normalization.py`
- `scptensor/normalization/quantile_normalization.py`
- `scptensor/normalization/trqn_normalization.py`
- `scptensor/__init__.py`
- `scptensor/autoselect/evaluators/normalization.py`
- `tests/core/test_normalization_api.py`
- `tests/normalization/test_normalize_api.py`
- `tests/normalization/test_normalization.py`
- `tests/normalization/test_quantile_normalization.py`
- `tests/normalization/test_trqn_normalization.py`
- `docs/core_data_contract.md`
- `docs/review_normalization_20260307.md`
- `docs/review_log_scale_20260312.md`
- `docs/review_io_state_mapping_20260312.md`

## 2. 范围与非范围

### 2.1 范围

本文档只覆盖 `normalization` 阶段本身。

当前冻结范围包括：

- `norm_none`
- `norm_mean`
- `norm_median`
- `norm_quantile`
- `norm_trqn`
- `normalize()` 统一分发入口

### 2.2 非范围

以下内容不属于本合同：

- `log_transform` 的实现细节
- peptide/precursor -> protein 聚合
- imputation
- batch correction / integration
- benchmark 评分逻辑本身

尤其要强调：

- `log_transform` 归属于 `scptensor.transformation`，不是 `scptensor.normalization` 的一部分。
- 归一化阶段不是 assay 转换阶段，不得在内部隐式做 `peptides -> proteins` 聚合。

### 2.3 稳定场景边界

从项目总合同看，稳定主线是 `protein-level` 预处理。因此：

- 当前 frozen contract 的主目标 assay 是 `proteins`。
- 代码实现本身在技术上对 assay 名基本是通用的。
- 但 `peptides` 上的归一化不应作为 stable core refactor 的驱动场景。
- 若上游输入是 peptide/precursor 定量，稳定主线仍应先经 `scptensor.aggregation` 聚合到 protein level，再进入默认归一化流程。

### 2.4 当前公开 API

`scptensor.normalization.__all__` 当前只导出 user-facing normalization API：

- `norm_none`
- `norm_mean`
- `norm_median`
- `norm_quantile`
- `norm_trqn`
- `normalize`

`scptensor.__all__` 当前不再从 normalization 顶层重导出这些 API。

不会重导出：

- user-facing normalization methods
- stage-plumbing helpers
- vendor-normalized input warning helpers
- `scptensor.core._layer_processing` 中的内部实现细节

因此当前稳定边界是：

- `scptensor.normalization` 子包只承载用户方法入口；
- preprocessing 共用的 layer plumbing 收敛到 `scptensor.core._layer_processing`；
- normalization 特有 warning 逻辑保留在内部模块 `scptensor.normalization._context`，但不作为公共 API 导出；
- 顶层 `scptensor` 不再承担 normalization facade，调用方应显式从 `scptensor.normalization` 导入。

## 3. 当前已实现方法池

截至当前代码基线，`scptensor.normalization` 真实已实现且公开导出的只有下表这些方法：

| 公开函数 / method key | 当前默认输出层名 | 直接实现事实 | 当前尺度合同 |
|---|---|---|---|
| `norm_none` / `none` | `no_norm` | 复制源层 `X`，并沿用当前 `M` 引用形成 passthrough layer | `any` |
| `norm_mean` / `mean` | `sample_mean_norm` | 按样本行做 `nanmean` 居中；可选加回全局均值 | `any` |
| `norm_median` / `median` | `median_centered` | 按样本行做 `nanmedian` 居中；可选加回全局中位数 | `any` |
| `norm_quantile` / `quantile` | `quantile_norm` | 样本分布对齐；NaN 保位；平均秩处理 ties | `logged` 作为稳定合同 |
| `norm_trqn` / `trqn` | `trqn_norm` | 先做 quantile baseline，再对 rank-invariant features 做 tail-robust 重平衡 | `logged` 作为稳定合同 |

补充说明：

- 当前没有实现 `sum`、`tmm`、`vsn`、`cyclic loess` 或泛化的 `robust_quantile` 公共 API。
- `normalize()` 统一入口当前只接受 `none / mean / median / quantile / trqn` 及其 `norm_*` 别名。
- `norm_mean` 与 `norm_median` 的直接函数默认是“只居中”的模式；`AutoSelect` 当前包装这两种方法时会显式开启 `add_global_mean=True` 和 `add_global_median=True`，即使用加回全局中心的 scaling mode。
- `AutoSelect` 当前只把 `norm_none / norm_mean / norm_median` 视作原始或未知尺度上的稳定候选；`norm_quantile / norm_trqn` 被视为 scale-sensitive 方法。

## 4. 输入与输出 assay-layer 合同

### 4.1 归一化是 assay 内部变换，不改变轴定义

归一化阶段必须保持：

- 同一 assay 内处理。
- 同一 `obs` 样本轴。
- 同一 `var` 特征轴。
- 同一矩阵 shape。

禁止把以下行为混入 `normalization`：

- 改变 feature 集合。
- 新建另一种 feature universe。
- 在归一化内部做 assay 聚合或 assay 间映射。

### 4.2 当前 API 是“原位修改 container，再返回该 container”

当前实现不是纯函数式 copy-return API，而是：

1. 校验 `container + assay + source_layer`
2. 计算新矩阵
3. 在原 `Assay.layers` 上新增或覆盖一个 layer
4. 向 `container.history` append 一条记录
5. 返回同一个 `container`

因此，后续重构不得悄悄改成“默认深拷贝再返回新对象”，除非同步修改测试、文档和调用约定。

### 4.3 输入 assay 名解析

当前 normalization 在内部通过 `resolve_layer_context()` 做 assay 别名解析。

当前支持的 assay alias 组只有：

- `protein` <-> `proteins`
- `peptide` <-> `peptides`

因此：

- 直接调用时传 `protein` 或 `proteins` 都可能工作。
- 仓库文档和合同层的 canonical assay naming 仍然应写 `proteins / peptides`。

### 4.4 输入 layer 前提

归一化函数当前要求：

- 目标 assay 存在。
- `source_layer` 在该 assay 中存在。
- `source_layer.X.shape[1] == assay.n_features`，这由 `Assay` 本身保证。

当前 normalization 模块不会额外强制检查：

- 是否是 protein assay
- 是否已经 log
- 是否存在 batch metadata
- 是否存在 group metadata

这些约束目前是在更高层合同或 `AutoSelect` 入口中收紧，而不是在每个归一化函数里强行拦截。

### 4.5 输出 layer 的当前稳定事实

所有已实现归一化方法都遵守下面这些输出事实：

- 输出 layer 与输入 layer shape 完全一致。
- 输出 layer 的样本轴和特征轴顺序不变。
- 输入 `source_layer.X` 不被原地数值修改。
- 新 layer 默认通过 `assay.add_layer(name, matrix)` 写入。

### 4.6 同名目标层的当前行为

`Assay.add_layer()` 当前是直接赋值：

- 若 `new_layer_name` 已存在，则静默覆盖。
- 不会发 warning。
- 不会做版本化保存。

这属于当前实现事实。若未来要引入“禁止覆盖”或“自动重命名”，属于显式行为变更，不能悄悄引入。

对公开方法而言，当前真实写回语义还要再分两类：

- `norm_mean / norm_median / norm_quantile / norm_trqn`
  - 若 `new_layer_name` 与已有 layer 同名，包含 `new_layer_name == source_layer`
  - 会在写回阶段替换该 layer registry entry
- `norm_none`
  - 若 `new_layer_name == source_layer`
  - 不会创建新层，也不会替换 source layer registry entry
  - 只会 append 一条 `normalization_none` history

### 4.7 `norm_none` 的特殊行为

`norm_none()` 当前是显式 no-op baseline：

- 若 `new_layer_name != source_layer`，则复制 source `X` 到新层，并沿用 source `M` 引用。
- 若 `new_layer_name == source_layer`，则不复制新层，只记录 history。

因此它的职责是“形成可审计的无归一化对照层”，而不是简单返回不做记录。

## 5. 数值、mask、metadata 合同

### 5.1 行列方向合同

当前归一化实现统一把矩阵解释为：

- 行：samples
- 列：features

这条约束不能被优化破坏。

具体来说：

- `norm_mean` 和 `norm_median` 使用 `axis=1` 做样本级居中。
- `norm_quantile` 以“每一行是一个 sample distribution”的方式做 row-wise quantile normalization。
- `norm_trqn` 在实现中会先把 `(samples, features)` 转为 `(features, samples)` 来评估 rank-invariant features，但其外部合同仍然建立在“输入 X 是 sample-by-feature”上。

### 5.2 mask 继承合同

当前实现对 `ScpMatrix.M` 的合同是：

- 若源层有 `M`，则输出层当前沿用同一个 `M` 引用，不做 copy。
- 不会因为归一化而重编码 `MaskCode`。
- 不会把 `MBR / LOD / FILTERED / IMPUTED` 等状态折叠成布尔 missing mask。

这意味着归一化阶段只变换 `X`，不改变状态码语义。

### 5.3 metadata 的当前事实

当前 `create_result_layer()` 只沿用 `source_layer.M` 的现有引用，不复制 `ScpMatrix.metadata`。

也就是说：

- `norm_mean`
- `norm_median`
- `norm_quantile`
- `norm_trqn`

在当前实现下都会创建一个 `metadata=None` 的新 `ScpMatrix`，除非未来显式修改这部分逻辑。

对 `norm_none` 而言：

- 若 `new_layer_name != source_layer`，同样会创建 `metadata=None` 的新 `ScpMatrix`
- 且该新层会沿用 source `M` 的同一引用
- 若 `new_layer_name == source_layer`，则不会创建新层，只记录 history

这是一个需要被文档化的当前事实。后续若要改为保留 metadata，必须视为显式合同更新，而不是“顺手优化”。

### 5.4 dense / sparse 当前行为

当前实现对稀疏矩阵的处理并不一致，这一点必须明确写死：

- `norm_none`：
  - dense 输入保留 dense
  - sparse 输入保留 sparse copy
- `norm_mean / norm_median / norm_quantile / norm_trqn`：
  - 先通过内部 dense helper 转成 dense `numpy.ndarray`
  - 结果层 `X` 当前是 dense

因此，当前合同不是“所有归一化方法都保留输入存储格式”，而是：

- `norm_none` 保留
- 其余方法 densify

若后续引入 sparse-aware kernel，需要保证数值等价、mask/history 等价，并把“输出类型是否仍为 dense”当作明确的兼容性决策处理。

## 6. 尺度合同

### 6.1 归一化不等于 log transform

当前仓库合同明确要求：

- `raw`、`log`、`norm` 是不同 layer 语义。
- `log_transform` 单独归属 `scptensor.transformation`。
- `normalization` 模块不得偷偷做 log。

因此：

- 不得把 `quantile` 或 `trqn` 的前置 log 变成归一化函数内部隐式步骤。
- 不得在归一化函数中通过启发式把 `raw` 自动转成 `log` 再比较。

### 6.2 方法级尺度要求

当前稳定合同应这样解释：

- `norm_none`：可用于任意 quantitative layer，只做 passthrough。
- `norm_mean`：当前实现可运行于 `raw` 或 `log`，作为 scale-weaker baseline。
- `norm_median`：当前实现可运行于 `raw` 或 `log`，作为 scale-weaker baseline。
- `norm_quantile`：底层函数当前对任意数值矩阵都能执行，但稳定合同要求输入层必须具有显式 log provenance。
- `norm_trqn`：同上，稳定合同要求输入层必须具有显式 log provenance。

这里要把“代码能跑”与“稳定工作流允许默认比较”分开。

### 6.3 `AutoSelect` 的当前尺度门控

`scptensor.autoselect.evaluators.normalization.NormalizationEvaluator` 当前已经把尺度合同部分冻结为工程规则：

- 若 `source_layer` 没有显式 log provenance，则只比较：
  - `norm_none`
  - `norm_mean`
  - `norm_median`
- 若 `source_layer` 具有显式 log provenance，则额外纳入：
  - `norm_quantile`
  - `norm_trqn`

因此安全重构必须满足：

- 不能把 `quantile` / `trqn` 自动放宽到 raw/unknown-scale AutoSelect 候选。
- 若未来要放宽，必须同步修改 docs、tests 与 benchmark 解释边界。

这里还要补一个实现级事实：

- `NormalizationEvaluator` 当前直接调用共享内部 detector
  `scptensor.core._log_scale_detection.detect_logged_source_layer(...)`
- `scptensor.transformation.log_transform._detect_already_logged(...)` 现在只是兼容包装层，
  同样委托到这套共享 detector
- evaluator 固定传入 `detect_logged_by_distribution=False`

因此对 normalization 候选门控而言，当前“logged source layer”只依赖：

1. layer naming 显式像 log
2. history 中存在匹配该 layer 的：
   - `log_transform`
   - `log_transform_skipped`

而不会因为 source 数值分布“看起来像 log”就自动开放 `norm_quantile` / `norm_trqn`。

### 6.4 vendor-normalized `raw` 输入的当前合同

当前 normalization 模块会检查历史记录，识别下面这种情况：

- 当前要处理的 `source_layer == "raw"`
- 上游 `load_quant_table` 记录显示该层来自 vendor-normalized quantity

满足条件时：

- 会发出 `UserWarning`
- 不会阻止继续归一化

warning 语义是：

- 你正在对已经 vendor-normalized 的线性输入再做归一化
- 应考虑比较 `norm_none`
- 或重新载入 unnormalized vendor 列

这条 warning 依赖 importer 在 history 中留下的键位，包括但不限于：

- `action == "load_quant_table"`
- `assay_name`
- `layer_name`
- `input_quantity_is_vendor_normalized`
- `resolved_quantity_column`

因此 normalization 与 importer 之间存在明确的 provenance 协作边界，不能单边重构。

## 7. Provenance 合同

### 7.1 每次调用都必须 append 一条 history

当前公开归一化 API 的稳定行为是：

- 每次成功调用 append 一条 `ProvenanceLog`
- 不覆盖旧记录
- 不静默跳过 provenance

### 7.2 当前 action 名称

当前 action 名称已经被测试和文档事实部分固定为：

| 方法 | 当前 history action |
|---|---|
| `norm_none` | `normalization_none` |
| `norm_mean` | `normalization_sample_mean` |
| `norm_median` | `normalization_median_centering` |
| `norm_quantile` | `normalization_quantile` |
| `norm_trqn` | `normalization_trqn` |

这些 action 名若要变更，应视为兼容性变更。

### 7.3 当前 params 最小集合

当前公开方法写入的 history 参数至少包含：

- `assay`
- `source_layer`
- `new_layer_name`

此外还有方法特异参数：

- `norm_mean`
  - `add_global_mean`
- `norm_median`
  - `add_global_median`
- `norm_trqn`
  - `low_thr`
  - `balance_stat`
  - `selected_features`
  - `feature_indices_provided`

### 7.4 当前 assay 名记录合同

当前 normalization 方法在写 history 时，`params["assay"]` 记录的是
**解析后的 assay key**，也就是容器里实际被使用的 assay 名。

这意味着：

- normalization 内部会先做 alias 解析来找到 assay
- history 中的 `params["assay"]` 与真正读写的 assay key 一致
- 若容器里 assay 名是 `protein`，传入 `proteins` 后 history 仍写 `protein`
- 若容器里 assay 名是 `proteins`，传入 `protein` 后 history 仍写 `proteins`

这一点现在与 `log_transform()` 的 resolved assay name 语义保持一致。

### 7.5 不要让内部 stage helper 改写公开 history schema

公开方法当前已经稳定下来的 history 参数命名是：

- `source_layer`
- `new_layer_name`

因此，即使底层 layer-plumbing helper 收敛到 `scptensor.core._layer_processing`，公开方法仍必须显式保留这套 history schema，不能因为内部 helper 更换而无声改名。

## 8. 失败合同

### 8.1 必须抛出的错误

当前实现下，下面这些错误属于稳定失败合同：

- assay 不存在：
  - 抛 `AssayNotFoundError`
- source layer 不存在：
  - 抛 `LayerNotFoundError`
- `normalize(method=...)` 传入未支持方法：
  - 抛 `ScpValueError`
- `norm_trqn(low_thr=...)` 不在 `(0, 1]`：
  - 抛 `ScpValueError`
- `norm_trqn(balance_stat=...)` 不是 `median` 或 `mean`：
  - 抛 `ScpValueError`
- `norm_trqn(feature_indices=...)` 越界：
  - 抛 `ScpValueError`

### 8.2 当前不会硬失败的情况

以下情况当前不会被 normalization 层主动拦截为 hard error：

- sparse 输入
- NaN 存在
- `Inf` 以外的有限异常分布
- 单样本矩阵
- 全零行
- quantile 场景中的全 NaN 行
- vendor-normalized `raw` 被再次归一化

对应当前行为分别是：

- sparse：除 `norm_none` 外会 densify 后继续
- NaN：`mean` / `median` 使用 `np.nanmean` / `np.nanmedian`，`quantile` 保持 NaN 位置
- `Inf`：`mean` / `median` / `quantile` / `trqn` 当前会显式报错，不再静默继续
- vendor-normalized `raw`：发 warning 但继续

如果未来要把这些场景升级成 error，属于显式合同变化。

### 8.3 覆盖行为不是错误

当前 `new_layer_name` 与已有层同名时：

- 会覆盖已有层
- 不抛错
- 不告警

这不是理想行为与否的问题，而是当前实现事实。任何改变都需要配套迁移说明。

## 9. 文档命名规范与实现兼容名

### 9.1 文档 canonical naming

当前仓库文档层的首选命名是：

- assays:
  - `proteins`
  - `peptides`
- quantitative layers:
  - `raw`
  - `log`
  - `norm`
  - `imputed`
- representation layer:
  - `zscore`

### 9.2 normalization 当前实现级兼容名

当前 normalization 代码默认生成的层名并不等于文档规范名：

| 方法 | 当前默认层名 | 在文档中的定位 |
|---|---|---|
| `norm_none` | `no_norm` | 比较层 / 对照层，非 canonical |
| `norm_mean` | `sample_mean_norm` | 实现级兼容名，非 canonical |
| `norm_median` | `median_centered` | 实现级兼容名，非 canonical |
| `norm_quantile` | `quantile_norm` | 实现级兼容名，非 canonical |
| `norm_trqn` | `trqn_norm` | 实现级兼容名，非 canonical |

因此：

- 对外教程、合同、benchmark 边界应优先写 `norm`，表示“被选中的主线 normalized quantitative layer”。
- 这些默认产物层名仍可保留在代码、测试和候选比较层中。
- 但它们不应继续扩散成仓库级默认示例命名体系。

### 9.3 method key 兼容别名

`normalize()` 当前接受：

- `none`
- `norm_none`
- `mean`
- `norm_mean`
- `median`
- `norm_median`
- `quantile`
- `norm_quantile`
- `trqn`
- `norm_trqn`

method key alias 可以保留，但对外文档叙述应优先使用 canonical method family 名：

- `none`
- `mean`
- `median`
- `quantile`
- `trqn`

## 10. 面向重构的直接约束

下面这些约束应视为“安全优化不能破坏”的硬边界。

### 10.1 不能扩大阶段边界

不要把这些逻辑偷偷并入 `normalization`：

- log transform
- aggregation
- imputation
- batch correction

### 10.2 不能改变轴方向假设

所有优化都必须保留：

- `X.shape == (n_samples, n_features)`
- `mean` / `median` 以样本行为单位
- `quantile` / `trqn` 的样本分布比较仍以当前轴解释为前提

### 10.3 不能丢失或重写 mask 语义

归一化层当前不会改变 `MaskCode`。这条性质必须保留。

### 10.4 不能取消 provenance append

即使引入更快的 kernel、更底层的 numba/jit 路径，也必须保留：

- 一次调用一条 history
- 当前 action 名
- 当前最小 params 结构

### 10.5 不能把 scale gate 从实现文档里抹掉

即使底层 `norm_quantile` / `norm_trqn` 在 raw 上技术上能跑，稳定合同仍要求：

- 默认工作流把它们视为 logged-layer 方法
- `AutoSelect` 原始层比较中不自动纳入这两类方法

### 10.6 不能假设 metadata 已被保留

当前 normalized layer 不复制 `ScpMatrix.metadata`。重构时如果引入 metadata 复制，必须：

- 明确写入文档
- 明确更新测试
- 明确评估序列化和内存后果

### 10.7 不要无声切换到 helper 的另一套 history schema

如前所述，内部 stage helper 不得无声替换公开 API 当前的 `source_layer/new_layer_name` 命名。

### 10.8 默认产物层名可以收敛，但不能偷偷改

当前默认层名仍是：

- `no_norm`
- `sample_mean_norm`
- `median_centered`
- `quantile_norm`
- `trqn_norm`

若未来决定把公开默认输出统一收敛到 `norm`，这属于用户可见行为变化，需要显式迁移。

## 11. 推荐的稳定使用方式

当前在实现与文档之间最稳妥的用法是：

1. importer 后把主 quantitative 输入层统一视为 `raw`
2. 若需要 log，显式生成 `log`
3. 在 `log` 上比较 `quantile` / `trqn`
4. 将最终选中的归一化结果落到主线层名 `norm`
5. 保留候选层与 provenance，避免覆盖原始输入

一个稳定示意是：

`raw -> log -> norm -> imputed -> zscore`

而不是：

- 在 `normalization` 里隐式做 log
- 让 `sample_mean_norm` / `median_centered` 等实现级命名直接替代主线文档命名
- 在 `peptides` 上长期堆叠下游预处理而不先聚合到 `proteins`
