# ScpTensor Core 数据对象合同（参考 AnnData / MuData，2026-03-16）

## 1. 文档目标

本文档定义 ScpTensor 核心数据对象的稳定合同，服务于后续 `core` 代码优化、API 收敛与测试补强。它回答的不是“算法应如何实现”，而是“哪些结构性约束不能被优化破坏”。

本文档覆盖的稳定对象：

- `ScpContainer`
- `Assay`
- `ScpMatrix`
- `MaskCode`
- `ProvenanceLog`
- `AggregationLink`
- `FilterCriteria`

本文档不覆盖：

- 具体归一化 / 填补 / 去批次算法细节
- benchmark 设计
- 下游降维 / 聚类实验接口
- 通用序列化格式扩展

核心范围受项目合同约束：ScpTensor 的主线目标是从 `DIA-NN / Spectronaut` 定量表出发，得到可审计的 **protein-level quantitative matrix**。

## 2. 外部设计参考与 ScpTensor 的取舍

### 2.1 AnnData 提供的可借鉴原则

按官方文档与 GitHub README，`AnnData` 的核心是“二维带注释矩阵”：

- `X` 是二维主矩阵。
- `obs` 与 `var` 分别沿样本轴和特征轴对齐。
- `layers` 与 `X` 共享相同 shape。
- 还有 `obsm`、`varm`、`obsp`、`varp`、`uns`、`raw` 等扩展槽位。
- 子集操作默认返回 view，并采用 copy-on-modify 语义。
- 支持 backed / on-disk 访问与 `to_memory()`。

这些设计说明，单细胞对象最重要的不是“字段越多越好”，而是：

1. 轴语义必须稳定。
2. 注释必须与矩阵严格对齐。
3. 变换后的矩阵必须保留来源关系。

### 2.2 MuData 提供的可借鉴原则

`MuData` 的 GitHub README 强调的是“分层多模态容器”：

- 顶层对象统一管理多个 modality。
- 每个 modality 自己维护一个独立的 `AnnData`。
- 顶层仍保留全局样本轴语义。

这对 ScpTensor 很重要，因为 ScpTensor 不是单一 feature space：

- `peptides` 与 `proteins` 可以共存。
- 两者共享同一批样本，但 feature universe 不同。
- 它们之间需要显式 linkage，而不是假设同索引可对齐。

### 2.3 ScpTensor 的设计决策

ScpTensor 借鉴 `AnnData + MuData` 的地方是：

- 借鉴 `AnnData` 的“二维矩阵 + 轴对齐注释”原则。
- 借鉴 `MuData` 的“顶层容器管理多个 feature space”原则。

ScpTensor 刻意不照搬的地方是：

- 不把 `Assay` 做成一个完整 `AnnData` 克隆。
- 不在 stable core 中引入 `obsm` / `varm` / `uns` / `raw` 等整套槽位。
- 不承诺 AnnData 式 view/backed 行为。
- 不让 peptide/protein 关系依赖隐式 feature index 对齐，而要求显式 `AggregationLink`。

原因很直接：ScpTensor 的主线任务是 DIA 单细胞蛋白组预处理，不是构建一个通用单细胞对象宇宙。

## 3. 稳定层级结构

| 参考设计 | ScpTensor 对象 | 稳定职责 |
|---|---|---|
| MuData top-level container | `ScpContainer` | 管理全局样本轴、assay 注册表、跨 assay links、history |
| AnnData per-modality object | `Assay` | 管理单一 feature space 的 `var` 与多层 `layers` |
| AnnData `X` / `layers[k]` payload | `ScpMatrix` | 管理数值矩阵 `X`、状态矩阵 `M` 与可选 layer metadata |
| 无直接等价 | `AggregationLink` | 显式表达 peptide/precursor -> protein 的映射关系 |
| 无直接等价 | `MaskCode` | 用分类状态码记录数值来源与可信度语义 |
| 无直接等价 | `ProvenanceLog` | 记录 append-only 处理历史 |

从已有代码看，这一层级已经成立：

- `ScpContainer` 拥有 `obs`、`assays`、`links`、`history`。
- `Assay` 拥有 `var` 与 `layers`。
- `ScpMatrix` 拥有 `X`、`M`、`metadata`。

因此后续优化应围绕“强化此三层合同”展开，而不是重新发明第四层对象。

## 4. 轴与 shape 合同

### 4.1 统一方向

ScpTensor 当前稳定方向是：

- `X.shape == (n_samples, n_features)`
- 行轴是样本轴
- 列轴是特征轴

这里的“样本”在 DIA 单细胞蛋白组主线中，通常对应下游分析样本单元，例如单细胞样本、well 级样本或与其等价的定量样本单元；它不是谱图、PSM 或 fragment 级对象。

### 4.2 `obs` 与样本轴

`ScpContainer.obs` 的合同：

- 必须是 `polars.DataFrame`
- `sample_id_col` 必须存在
- `sample_id_col` 必须唯一
- `obs.height == n_samples`

所有 assay 的所有 layer，都必须满足：

- `matrix.X.shape[0] == obs.height`

也就是说，`obs` 是全容器共享的样本轴 source of truth。

### 4.3 `var` 与特征轴

`Assay.var` 的合同：

- 必须是 `polars.DataFrame`
- `feature_id_col` 必须存在
- `feature_id_col` 必须唯一
- `var.height == n_features`

同一个 assay 下的所有 layer，都必须满足：

- `matrix.X.shape[1] == var.height`

也就是说，`var` 是该 assay feature space 的 source of truth。

### 4.4 同 assay 内的严格对齐

同一个 `Assay` 内：

- 所有 layer 共享完全相同的样本轴顺序
- 所有 layer 共享完全相同的特征轴顺序
- layer 之间只允许“值空间不同”，不允许“轴定义不同”

因此，`raw`、`log`、`norm`、`imputed`、`zscore` 之类 layer 的差异应只体现在数值与 provenance 上，而不是 feature 集合变化。

### 4.5 多 assay 之间不要求 feature 对齐

不同 assay 之间：

- 共享同一 `obs`
- 不要求共享同一 `var`
- 不要求 feature 数量相同
- 不要求 feature 顺序相同

例如：

- `proteins` assay 与 `peptides` assay 可以合法地拥有不同 feature universe。
- 它们的关系必须通过 `AggregationLink` 表达，而不是通过同列位置或同列名隐式猜测。

### 4.6 显式 assay 维度访问边界

`ScpContainer` 是多 assay 容器，不承诺单一 feature 维度。因此当前稳定合同明确：

- `ScpContainer` 不提供容器级 `shape`
- `ScpContainer` 不提供容器级 `n_features`
- 多 assay 代码必须显式按 assay 查询 feature 维度

兼容性 shortcut 只剩一处仍可存在：

- `Assay.X` 是 layer `'X'` 的快捷访问，而不是所有 layer 的统一别名

因此稳定合同中应明确：

- feature 维度访问应通过 `container.assay_shape(assay_name)`、`container.assays[assay_name].n_features` 或显式 layer shape 完成
- 稳定 preprocessing API 不应依赖 `Assay.X`，而应显式要求 `assay_name + source_layer`

### 4.7 稳定访问模式

对 stable preprocessing 文档、教程和后续重构而言，推荐的标准访问模式是：

- `container.assays[assay_name]`
- `container.assays[assay_name].layers[source_layer]`
- `container.assays[assay_name].var`
- `container.obs`

不推荐把以下接口当作 stable 主路径：

- `assay.X`

原因是这类 shortcut 会弱化“显式 assay / layer 选择”这一核心合同。

## 5. 元数据合同

### 5.1 `obs`

`obs` 只承载样本级元数据，例如：

- batch
- condition
- run / file name
- QC summary
- 样本来源与分组信息

稳定代码不应把 feature 级信息塞进 `obs`。

### 5.2 `var`

`var` 只承载 assay 内 feature 级元数据，例如：

- protein ID / gene name
- precursor ID / modified sequence
- feature-level QC summary
- aggregation 相关注释

稳定代码不应把全局样本信息塞进 `var`。

### 5.3 不扩展为 AnnData 全槽位对象

当前 stable core **不承诺** 以下等价槽位：

- `obsm`
- `varm`
- `obsp`
- `varp`
- `uns`
- 独立 `raw` slot

这不是缺陷，而是当前项目边界：

- 主线预处理更关心可审计 quantitative layers
- 下游 embedding / clustering 已被项目合同界定为 experimental downstream helpers
- 若未来确实需要引入额外容器槽位，应以明确用例驱动，而不是为了“更像 AnnData”而增加

### 5.4 `ScpMatrix.metadata`

`ScpMatrix.metadata` 的稳定定位是：

- layer 级、矩阵同 shape 或近矩阵级的质量信息
- 例如 confidence、detection limit、imputation quality、outlier score、creation info

它不是通用 `uns` 垃圾桶。

### 5.5 仓库级 assay / layer 命名规范

为减少文档漂移，仓库级文档应优先使用以下 canonical naming：

- stable quantitative assay:
  - `proteins`
  - `peptides`
- compatibility aliases:
  - `protein`
  - `peptide`
- stable quantitative layers:
  - `raw`
  - `log`
  - `norm`
  - `imputed`
- representation / downstream helper layer:
  - `zscore`

补充说明：

- `proteins` / `peptides` 是文档首选命名；`protein` / `peptide` 仅作为历史兼容别名处理。
- `raw` 是仓库文档首选的“导入后主 quantitative layer”名称；它表示当前预处理主线的输入层，不自动等同于“未做 vendor normalization”。是否经过上游归一化应由 provenance 字段例如 `is_vendor_normalized`、`source_column` 说明，而不是再引入第二套默认 layer 名。
- `norm` 是仓库文档首选的“ScpTensor-owned 已归一化 quantitative layer”名称；`normalized`、`trqn_norm`、`sample_mean_norm` 等可作为实现级自定义 layer 存在，但不应继续作为仓库文档的默认示例命名。
- 方法比较、兼容输出或评测 artifact 允许继续使用实现级 layer 名，例如 `sample_mean_norm`、`quantile_norm`、`imputed_knn`、`limma`、`combat`、`mnn_corrected`、`scanorama`；但这些名字不属于仓库级 canonical 主线层命名，若要成为工作流主结果层，应通过显式 promote/rename 步骤落到对应阶段的 canonical layer，并保留 provenance。
- `zscore` 明确属于 representation layer，而不是 stable quantitative endpoint。
- layer `'X'` 仅保留给兼容性场景或 experimental/downstream assay（如 `pca`、`umap`、`tsne`）的坐标载荷；不应作为 stable protein/peptide quantitative assay 的默认 layer 名。

## 6. Layer 合同

### 6.1 `ScpMatrix` 是稳定 payload

`ScpMatrix` 的合同：

- `X`: 数值矩阵，`float` dense 或 `scipy.sparse` 矩阵
- `M`: 与 `X` 同 shape 的状态矩阵，`int8` dense 或 sparse，可为空
- `metadata`: 可选 layer metadata

若 `M is None`，稳定语义是“默认全部视为 `VALID`”，而不是“缺失语义未知”。

### 6.2 `layers` 是同轴多表示，不是多 feature universe

`Assay.layers` 中的每个元素都必须表示：

- 同一组样本
- 同一组特征
- 不同数值表示或处理阶段

因此以下是合法的 layer 变换：

- `raw -> log`
- `log -> norm`
- `norm -> imputed`
- `imputed -> zscore`

而以下不是 layer 应承担的语义：

- `peptide -> protein`
- `fragment -> precursor`
- feature 集合改变后的对象替换

这类跨 feature universe 的变化必须通过新的 assay 加 `AggregationLink` 表达。

### 6.3 不设 AnnData 式独立 `raw` slot

AnnData 有专门的 `raw` 槽位；ScpTensor 当前不应复制这一设计。稳定合同是：

- `raw` 只是一个命名 layer
- 是否是 `raw`、`log`、`norm`，以及 `raw` 是否承载 vendor-normalized input，不只由 layer 名字决定
- source of truth 应由 `history` + 明确参数共同界定

这与现有 `log_transform`、normalization、I/O provenance 设计一致。

## 7. `MaskCode` 与 provenance 合同

### 7.1 `MaskCode` 是分类状态码，不是布尔缺失掩码

稳定状态语义如下：

| 代码 | 名称 | 含义 |
|---|---|---|
| `0` | `VALID` | 明确可用的原始/处理后有效值 |
| `1` | `MBR` | run 间匹配或 transfer 语义 |
| `2` | `LOD` | 低于检测下限 / 未检出 / 保守缺失 |
| `3` | `FILTERED` | 被显式过滤掉 |
| `4` | `OUTLIER` | 被标记为异常值 |
| `5` | `IMPUTED` | 后验填补得到 |
| `6` | `UNCERTAIN` | 来源语义不能可靠恢复 |

因此：

- `M` 不是简单的 `is_missing`。
- 后续优化不得把它退化为布尔 NA mask。
- 若新增状态，必须先论证是否真的需要扩 enum，而不是偷塞进 `metadata` 或负数编码。

### 7.2 provenance 是 append-only 全局历史

`ScpContainer.history` 的稳定合同：

- 类型为 `list[ProvenanceLog]`
- 新操作应 append，而不是重写旧记录
- `params` 应保持 JSON-serializable
- 用户可见变换应尽量记录 `assay/layer/source/target/key params`

ScpTensor 当前已经依赖这条合同来区分：

- importer 来源
- vendor-normalized 输入
- transformation / normalization / filtering 顺序

后续核心优化不得破坏 `history` 的可读性与可序列化性。

### 7.3 `MaskCode` 的当前实现差距与收敛方向

当前文档语义与实现之间，仍有一个需要显式记录的差距：

- 对 long-format importer，部分状态仍有机会结合列语义、FDR 与 vendor 文档做较强解释。
- 对 matrix/pivot importer，当前实现采用更保守的基线映射：
  - 有限值 -> `VALID`
  - 非有限值 -> `UNCERTAIN`

这条基线映射可以接受为当前工程起点，但它不是“完整恢复上游状态语义”的证明。文档层必须继续明确：

- matrix-only 输入中的 `UNCERTAIN`，表示“缺失单元格存在，但来源语义无法可靠恢复”。
- 仅凭最终 pivot matrix，不应把缺失单元格自动解释成 `MBR`、真实 `LOD` 或 vendor-filtered failure。
- 当前 matrix importer 已把这类缺失单元格优先落到 `UNCERTAIN`；后续若要进一步细分 `LOD` / `FILTERED` / `MBR`，必须依赖更强 source evidence，而不是仅凭 pivot matrix 猜测。

因此，后续 core / io 优化的收敛顺序应是：

1. 优先保留更多 source evidence 与 provenance。
2. 在 evidence 足够时再细分 `FILTERED`、`MBR`、`UNCERTAIN`。
3. 在 evidence 不足时，坚持保守映射，不做伪精确声明。

## 8. 跨 assay 合同

### 8.1 `AggregationLink` 是正式结构，不是附带信息

`AggregationLink` 的稳定合同：

- 必须包含 `source_assay`
- 必须包含 `target_assay`
- `linkage` 必须至少有 `source_id` 与 `target_id`
- link 中引用的 feature ID 必须能在对应 assay 中找到

对 ScpTensor 而言，这个 link 不是可有可无的注释，而是：

- peptide/precursor -> protein 聚合 provenance
- 跨层级定量解释的桥梁
- 防止 feature-space 混淆的结构性约束

### 8.2 跨 assay 转换的唯一主线

根据项目合同：

- `scptensor.aggregation` 是唯一正式的 peptide/precursor -> protein 转换阶段
- 其余 stable downstream 模块默认工作在 protein-level data

因此后续核心代码不应：

- 在 normalization / imputation / integration 内部隐式做 peptide -> protein 聚合
- 通过字符串列名猜测来构建 assay 间关系
- 把不同 feature level 挤进同一个 assay 里规避 link 机制

## 9. copy / view / backed / serialization 合同

### 9.1 不承诺 AnnData view 语义

AnnData 的 slicing 默认返回 view，这对大对象高效，但也会带来隐式共享状态与 copy-on-modify 复杂性。

ScpTensor 当前稳定合同应更保守：

- `filter_samples()` / `filter_features()` 返回新的 `ScpContainer`
- `copy=True` 是默认路径，应视为稳定、安全语义
- `copy=False` 与 `shallow_copy()` 仅是性能优化入口，不应被表述成 AnnData 风格 lazy view

换句话说：

- 可以保留轻量 aliasing 机制
- 但不应让核心正确性依赖“对象可能是 view”这种隐式状态

这更适合当前以可解释预处理为主的 ScpTensor 主线。

### 9.2 不承诺通用 backed / on-disk container

AnnData 官方支持 backed mode 与 on-disk arrays。ScpTensor 当前 stable contract **不承诺**：

- 通用 backed `ScpContainer`
- 通用 `.save()` / `.load()` 对象持久化格式
- 一个类似 `.h5ad` / `.zarr` 的主线对象序列化标准

稳定 I/O 边界明确限定为：

- `scptensor.io.load_quant_table`
- `scptensor.io.load_diann`
- `scptensor.io.load_spectronaut`
- `scptensor.io.load_peptide_pivot`
- `scptensor.io.aggregate_to_protein`

因此，若未来需要 out-of-core 或 collection 设计，更合理的路线是：

- 单列新的 collection/backed API
- 而不是把当前 `ScpContainer` 静默扩成半兼容 AnnData backed object

### 9.3 模块家族的 mutability 语义冻结

当前仓库已经形成三类不同的对象写回语义，后续文档与重构都应按家族区分，而不是混写成“所有 API 都 copy-return”。

第一类，stable preprocessing 写回式 API：

- `transformation`
- `normalization`
- `impute`
- `integration`
- `standardization`

它们当前的共同合同是：

- 读取已有 `assay + source/base layer`
- 在同一个 `container` / `assay` 上新增或覆盖结果 layer
- 追加 `history`
- 返回同一个逻辑容器对象

第二类，stable filtering / selection API：

- 例如 `filter_samples()` / `filter_features()`

这类 API 当前以“返回新容器”作为稳定、安全语义，不应被表述成原位写回族。

第三类，experimental downstream helper：

- `dim_reduction`
- `cluster`
- `experimental` namespace 下的相关 facade

这类 API 当前没有统一的 copy 语义；有的重建 container mapping，有的共享 assay 对象，有的直接把结果写进 `obs`。其边界以 [experimental_downstream_contract.md](experimental_downstream_contract.md) 为准。

因此后续若做 API 收敛，至少要先明确“改变的是哪一个模块家族”，不能把：

- preprocessing 的原位 layer 写回
- filtering 的新容器返回
- experimental downstream 的混合共享语义

混成同一条笼统规则。

## 10. 面向核心代码优化的直接指导

后续 `core` 优化应优先守住以下不变量：

1. `obs` 与所有 assay 样本轴严格对齐。
2. `var` 与 assay 内所有 layer 特征轴严格对齐。
3. layer 只表示同 feature space 的不同数值表示。
4. peptide/precursor -> protein 必须保留显式 `AggregationLink`。
5. `MaskCode` 仍是分类状态矩阵，而不是简化布尔 mask。
6. `history` 仍是 append-only、JSON-friendly provenance。
7. public preprocessing API 继续显式要求 `assay_name` 与 `source_layer`。
8. 多 assay 场景下，feature 维度必须显式从 assay 查询，而不是从容器级 shortcut 推断。
9. 仓库文档继续优先使用 `proteins / peptides` 与 `raw / log / norm / imputed / zscore` 这套 canonical naming。

相应地，以下方向可以大胆优化，而不会破坏稳定合同：

- 稀疏 / 稠密矩阵内部实现
- validation 性能
- 复制策略与内存占用
- `MatrixOps` / `jit_ops` / `sparse_utils` 内部实现
- 更清晰的错误消息与类型注解

而以下方向不应轻易推进为 stable core 变更：

- 引入大量 AnnData 风格新增槽位
- 把 experimental embedding/clustering 状态塞进 core 结构
- 重建通用对象序列化系统
- 让跨 assay 关系依赖隐式列名或位置约定

## 11. 与当前代码的对齐结论（2026-03-16 核查）

当前代码已经与本合同大体一致：

- `ScpContainer` 校验全 assay 的样本轴对齐。
- `Assay` 校验 layer 与 `var` 的 feature 轴对齐。
- `ScpMatrix` 校验 `X/M` shape 与 mask code 合法性。
- `AggregationLink` 已做 assay 与 feature ID 校验。
- `filter_*` 默认返回新容器，而不是 in-place 修改。
- `history` 已广泛用于 importer 与后续处理 provenance。
- 通用 `save/load` 已从 `ScpContainer` 公共表面移除。

当前需要在文档层进一步强调的细节有一点：

1. `Assay.X` 只是 layer `'X'` 的兼容捷径，不应成为 stable preprocessing 主接口。

## 12. 外部来源（访问日期：2026-03-16）

### 12.1 官方文档 / GitHub

- AnnData official docs, API overview and object fields:
  - https://anndata.readthedocs.io/en/stable/generated/anndata.AnnData.html
- AnnData official docs, on-disk / backed APIs:
  - https://anndata.readthedocs.io/en/stable/generated/anndata.AnnData.to_memory.html
- AnnData GitHub repository README:
  - https://github.com/scverse/anndata
- MuData GitHub repository README:
  - https://github.com/scverse/mudata

### 12.2 本地实现锚点

- `scptensor/core/structures.py`
- `scptensor/core/types.py`
- `scptensor/core/filtering.py`
- `scptensor/core/exceptions.py`
- `AGENTS.md`
