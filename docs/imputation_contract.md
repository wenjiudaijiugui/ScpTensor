# ScpTensor 填补实现冻结合同（`scptensor.impute`，2026-03-16）

## 1. 文档目标

本文档冻结 `scptensor.impute` 当前实现边界，服务于后续代码优化、API 收敛与测试补强。它回答的是：

- 当前已经实现了哪些填补方法；
- 这些方法面向什么缺失语义；
- 输入层、输出层、状态矩阵与 provenance 目前遵守什么约束；
- 后续重构时，哪些行为可以优化，哪些行为不能被悄悄改变。

本文档是 **implementation-facing contract**，不是文献综述，也不是 benchmark 排名结论。

本文档基于以下仓库内事实：

- `scptensor/impute/__init__.py`
- `scptensor/impute/base.py`
- `scptensor/impute/_utils.py`
- `scptensor/impute/baseline.py`
- `scptensor/impute/knn.py`
- `scptensor/impute/lls.py`
- `scptensor/impute/bpca.py`
- `scptensor/impute/missforest.py`
- `scptensor/impute/svd.py`
- `scptensor/impute/qrilc.py`
- `scptensor/impute/minprob.py`
- `scptensor/__init__.py`
- `tests/core/test_impute_api.py`
- `tests/impute/test_baseline_methods.py`
- `tests/impute/test_svd_and_selection.py`
- `tests/impute/test_critical_regressions.py`

## 2. 作用域与非作用域

### 2.1 作用域

本文档覆盖：

- `scptensor.impute.base`
- `scptensor.impute.baseline`
- `scptensor.impute.knn`
- `scptensor.impute.lls`
- `scptensor.impute.bpca`
- `scptensor.impute.missforest`
- `scptensor.impute.svd`
- `scptensor.impute.qrilc`
- `scptensor.impute.minprob`

### 2.2 非作用域

本文档不覆盖：

- 文献优先级与方法推荐强度
- masked-value benchmark 设计
- AutoSelect 排名逻辑
- peptide/precursor -> protein 聚合
- batch correction / integration
- downstream clustering 或 DE 任务定义

这些边界仍应分别以后续 benchmark 文档、`review_*.md` 综述和对应模块合同为准。

### 2.3 当前公开 API

`scptensor.impute.__all__` 当前导出两类公共入口：

- individual method wrappers：
  - `impute_none`
  - `impute_zero`
  - `impute_row_mean`
  - `impute_row_median`
  - `impute_half_row_min`
  - `impute_knn`
  - `impute_bpca`
  - `impute_mf`
  - `impute_lls`
  - `impute_iterative_svd`
  - `impute_softimpute`
  - `impute_qrilc`
  - `impute_minprob`
- unified interface helpers：
  - `impute`
  - `list_impute_methods`
  - `infer_missing_mechanism`
  - `recommend_impute_method`

`scptensor.__all__` 当前只从 imputation 模块顶层重导出 individual wrappers：

- `impute_none`
- `impute_zero`
- `impute_row_mean`
- `impute_row_median`
- `impute_half_row_min`
- `impute_knn`
- `impute_lls`
- `impute_iterative_svd`
- `impute_softimpute`
- `impute_bpca`
- `impute_mf`
- `impute_qrilc`
- `impute_minprob`

不会重导出：

- `impute`
- `list_impute_methods`
- `infer_missing_mechanism`
- `recommend_impute_method`

因此当前稳定边界是：

- `scptensor.impute` 子包是 unified dispatch / mechanism helper 的公共入口；
- 顶层 `scptensor` 只暴露 individual method wrappers，不把统一分发入口升格为顶层主 API。

## 3. 稳定场景边界

### 3.1 稳定默认场景

按项目合同，`scptensor.impute` 的稳定默认场景是：

- assay：`proteins`
- 输入对象：protein-level quantitative matrix
- 推荐工作流位置：`raw`、`log` 或 `norm` 之后的缺失修复
- 文档级 canonical 输出层：`imputed`

这里要明确两件事：

1. 当前代码技术上接受任意 `assay_name` 字符串，而不强制只能处理 `proteins`。
2. 但稳定预处理主线仍应把 `proteins` 视为默认目标；`peptides` 不是下游 stable preprocessing 的默认合同入口。

### 3.2 与聚合模块的边界

- `scptensor.impute` 不负责 peptide/precursor -> protein 转换。
- 若输入是 `peptides`，那只是当前实现允许的技术路径，不应被文档表述成仓库默认主线。
- `protein-level complete matrix` 仍是项目要求的最终稳定终点。

## 4. 当前已实现的方法全集

### 4.1 统一注册名

当前 `list_impute_methods()` 注册的稳定方法名为：

- `none`
- `zero`
- `row_mean`
- `row_median`
- `half_row_min`
- `knn`
- `lls`
- `bpca`
- `missforest`
- `iterative_svd`
- `softimpute`
- `qrilc`
- `minprob`

这些注册名是 `impute(container, method=...)` 的统一分发入口。

### 4.2 方法家族与当前实现映射

| 方法家族 | 注册名 | 对外包装函数 | 当前默认 `new_layer_name` | 主要定位 |
|---|---|---|---|---|
| baseline | `none` | `impute_none` | `imputed_none` | 显式 no-op 基线 |
| baseline | `zero` | `impute_zero` | `imputed_zero` | 简单常数填补 |
| baseline | `row_mean` | `impute_row_mean` | `imputed_row_mean` | 样本内均值填补 |
| baseline | `row_median` | `impute_row_median` | `imputed_row_median` | 样本内中位数填补 |
| baseline / MNAR-like fallback | `half_row_min` | `impute_half_row_min` | `imputed_half_row_min` | 样本内最小值比例填补 |
| neighborhood | `knn` | `impute_knn` | `imputed_knn` | KNN 邻域填补 |
| local regression | `lls` | `impute_lls` | `imputed_lls` | Local Least Squares |
| latent factor | `bpca` | `impute_bpca` | `imputed_bpca` | Bayesian PCA |
| tree / iterative | `missforest` | `impute_mf` | `imputed_missforest` | RF-based iterative imputation |
| low-rank completion | `iterative_svd` | `impute_iterative_svd` | `imputed_iterative_svd` | 低秩迭代 SVD |
| low-rank completion | `softimpute` | `impute_softimpute` | `imputed_softimpute` | nuclear-norm style completion |
| left-censored MNAR | `qrilc` | `impute_qrilc` | `qrilc` | 左删失分布填补 |
| left-censored MNAR | `minprob` | `impute_minprob` | `imputed_minprob` | 概率最小值填补 |

实现注意：

- `method="missforest"` 对应的包装函数名是 `impute_mf`，不是 `impute_missforest`。
- `qrilc` 当前默认输出层名是 `qrilc`，而不是 `imputed_qrilc`。
- 这说明当前实现的默认层名并不统一；它们是 **implementation-local compatibility names**，不是仓库级 canonical layer naming。

## 5. 文档级 canonical 命名与当前实现的关系

仓库文档当前统一采用：

- assay：`proteins` / `peptides`
- layer：`raw` / `log` / `norm` / `imputed` / `zscore`

因此对 `scptensor.impute` 的冻结解释应为：

1. 文档级、教程级、后续稳定工作流中，填补后的主结果层应优先命名为 `imputed`。
2. 当前代码里各方法自带的 `imputed_knn`、`imputed_lls`、`qrilc` 等默认层名，只是当前实现与评测、可视化、AutoSelect 的兼容输出。
3. 后续若做 API 收敛，允许把默认主输出进一步规范到 `imputed`，但不能在不说明迁移影响的情况下直接破坏现有方法名或统一入口名。

换句话说：

- **代码当前默认层名**：方法特异
- **仓库稳定文档层名**：`imputed`

## 6. 缺失值边界与状态假设

### 6.1 当前实现认定“需要填补”的唯一直接依据

当前 `scptensor.impute` 只依据以下条件决定哪些位置会被填补：

- `source_layer.X` 中该位置为 `NaN`

当前实现 **不会** 自动把以下值当成缺失：

- `0.0`
- 负值
- `MaskCode.LOD`
- `MaskCode.FILTERED`
- `MaskCode.OUTLIER`
- `MaskCode.UNCERTAIN`
- `MaskCode.MBR`

只要数值在 `X` 里是 finite，它就会被视为 observed，并在输出层中被原样保留。

### 6.2 与 state-aware missingness 的边界

结合当前 missingness 文档，应把这一点冻结得很明确：

- `scptensor.impute` 当前不是 state-aware target selector。
- 它不会读取 `M` 来决定“只填补 `LOD`、不填补 `FILTERED`”。
- 它只读取 `X` 中的 `NaN`。

因此稳定使用前提是：

- 如果调用方只希望填补 `LOD-like missing`，就必须在进入 `scptensor.impute` 之前，把目标位置编码成 `NaN`，并避免把 `FILTERED / OUTLIER / UNCERTAIN / MBR` 混成同一批待填补空洞。

### 6.3 `0` 的边界

当前实现不会把 `0` 自动转成 `NaN`。因此：

- vendor `0` 是否表示低浓度占位，属于 I/O 与状态映射问题；
- 一旦保留成有限值进入 `source_layer.X`，imputation 阶段就不会再把它当作缺失。

## 7. 输入合同

### 7.1 容器与轴方向

当前实现固定遵守 ScpTensor 核心矩阵方向：

- `X.shape == (n_samples, n_features)`
- 行轴是样本轴
- 列轴是特征轴

与此对应的方法语义也已固定：

- `row_mean` / `row_median` / `half_row_min` 是 **按样本行** 计算；
- `knn` / `lls` 也是以样本行为邻域对象；
- `qrilc` / `minprob` 以特征列为单位估计低丰度分布；
- `bpca` / `iterative_svd` / `softimpute` 作用于整张样本 × 特征矩阵。

后续优化不能通过隐式转置改变这些轴语义。

### 7.2 必要输入条件

所有当前实现共同要求：

- `container.assays[assay_name]` 存在
- `container.assays[assay_name].layers[source_layer]` 存在
- `source_layer.X` 可转为浮点矩阵
- 缺失值用 `NaN` 表示

### 7.3 对输入状态完整性的当前假设

当前实现不强制检查：

- 输入是否已经 log 化
- 输入是否已经归一化
- 输入是否只包含 protein-level 数据
- 输入是否只对某些 mask code 的位置开放填补

因此这些都是 **caller responsibility**。稳定建议是：

- 通用方法族优先用于可比尺度层，通常是 `log` 或 `norm`
- `qrilc` / `minprob` / `half_row_min` 更适合“较低值更接近低丰度”的场景
- 但当前代码不会主动验证这些前提

### 7.4 稀疏输入边界

当前注册表中，只有 `none` 标记为 `supports_sparse=True`。其余方法都被注册为 `supports_sparse=False`，并在包装层普遍执行 dense conversion。

因此冻结合同是：

- 输入可以是 sparse matrix
- 除 `none` 外，大多数方法会先 densify
- 输出是否保持 sparse 不是稳定承诺
- 后续优化可以改进内存表现，但不能改变数值语义、shape 或 mask/provenance 合同

## 8. 自动机制推断与方法选择合同

### 8.1 统一入口

`impute(container, method=..., missing_mechanism=..., **kwargs)` 是统一入口。

支持的 `missing_mechanism`：

- `auto`
- `mcar`
- `mar`
- `mnar`

### 8.2 当前 auto 选择映射

当前默认映射固定为：

- `mcar -> knn`
- `mar -> missforest`
- `mnar -> qrilc`

若 `method="auto"` 但推荐方法未注册，则当前回退顺序为：

1. `knn`
2. `row_mean`
3. `zero`

### 8.3 当前缺失机制推断规则

`infer_missing_mechanism()` 当前使用透明启发式，而不是模型训练：

1. 计算 feature 平均强度与 feature 缺失率的 Spearman 相关。
2. 计算 sample-level missingness CV。
3. 根据当前阈值作以下判断：
   - 无缺失：`mcar`
   - 强负相关 `corr <= -0.45`：`mnar`
   - `sample_missing_cv >= 0.6` 且 `missing_rate >= 0.05`：`mar`
   - `abs(corr) <= 0.2` 或 `missing_rate <= 0.08`：`mcar`
   - `corr < -0.2`：`mnar`
   - 其他：`mar`

这是当前实现行为的一部分。后续若更改阈值或判定顺序，应视为合同变更，而不是“纯优化”。

### 8.4 机制与方法不匹配时的当前行为

若调用方显式指定了 `missing_mechanism`，但所选方法不在当前 compatibility map 中：

- 不会报错
- 会向 `container.history` 追加 `impute_mechanism_warning`

也就是说，当前机制兼容性是 **warning-level provenance**，不是 hard validation。

## 9. 输出层合同

### 9.1 返回对象语义

当前所有方法都：

- 在传入的 `container` 上 **原地追加新 layer**
- 返回同一个 `container`

它不是 pure functional copy API。

因此：

- 若调用方需要隔离副作用，应在调用前自行 `container.copy()`
- 后续优化不能把它悄悄改成“默认返回新容器”而不做明确迁移说明

### 9.2 输出 shape 与位置不变性

所有方法都必须保持：

- 输出层与输入层 shape 完全相同
- 所属 assay 不变
- `obs` 与 `var` 不变

### 9.3 已观测值保持不变

当前除 `none` 外的所有主包装函数在填补后都会显式执行：

- `X_imputed[~missing_mask] = X_original[~missing_mask]`

因此冻结合同是：

- 原本 finite 的 observed 值必须逐元素保持不变
- 填补只能发生在原始 `NaN` 位置

这是后续优化绝不能破坏的首要数值不变量之一。

### 9.4 canonical 输出层与兼容输出层

对稳定文档与后续核心流程，推荐输出层口径是：

- `imputed`

但当前实现层依然保留方法特异默认名。故冻结合同中必须同时接受两层表述：

- 文档主层：`imputed`
- 兼容/评测层：`imputed_knn`、`imputed_lls`、`qrilc` 等

### 9.5 layer 名冲突的当前行为

当前 `Assay.add_layer()` 对同名 layer 不报错，而是直接覆盖：

- `self.layers[name] = matrix`

因此当前冻结行为是：

- `new_layer_name` 冲突不会抛出 LayerExists 异常
- 结果会覆盖 assay 中原有同名层

这属于当前实现行为，不是推荐最佳实践。调用方若要保留多份结果，必须显式使用不同的 `new_layer_name`。

## 10. Mask 与 provenance 合同

### 10.1 `M` 的更新规则

当前 `_update_imputed_mask()` 规则固定如下：

1. 若原始没有缺失：
   - 有 `M_original`：返回其拷贝
   - 无 `M_original`：返回 `None`
2. 若原始存在缺失且已有 `M_original`：
   - 复制原 `M`
   - 将原始 `NaN` 位置写为 `MaskCode.IMPUTED`
3. 若原始存在缺失且没有 `M_original`：
   - 新建 `int8` mask
   - observed 位置为 `MaskCode.VALID`
   - 原始 `NaN` 位置为 `MaskCode.IMPUTED`

这里的关键边界是：

- 当前实现只会把“原始 `NaN` 位置”标成 `IMPUTED`
- 不会根据方法内部迭代、置信度或 mechanism 进一步细分状态

### 10.2 `history` 记录

当前成功路径通常会写入 `container.history`，例如：

- `impute_knn`
- `impute_lls`
- `impute_bpca`
- `impute_missforest`
- `impute_qrilc`
- `impute_minprob`
- `impute_iterative_svd`
- `impute_softimpute`

自动选择额外会写入：

- `impute_method_selection`

显式机制不匹配额外会写入：

- `impute_mechanism_warning`

但需要冻结一个当前现实：

- 部分方法在“无缺失 fast path”下不会统一追加 history。

因此当前下游代码 **不能假设** “只要调用过 imputation，就一定新增一条 history”。后续可以统一这一行为，但那应被视为明确的行为收敛，而不是静默优化。

## 11. 失败合同

### 11.1 结构类失败

当前统一失败类型如下：

- assay 不存在：`AssayNotFoundError`
- layer 不存在：`LayerNotFoundError`

### 11.2 参数类失败

当前参数非法主要抛：

- `ScpValueError`

典型场景包括：

- `k <= 0`
- `max_iter <= 0`
- `tol <= 0`
- `q` 不在 `(0, 1)`
- `sigma <= 0`
- `rank <= 0`
- `oversample_factor < 1`
- `weights` 不是 `uniform` 或 `distance`
- `missing_mechanism` 不在允许集合中

### 11.3 维度与依赖失败

- `bpca` 中 `n_components >= min(n_samples, n_features)`：`DimensionError`
- `softimpute` 缺少 `fancyimpute`：`MissingDependencyError`

### 11.4 底层算法失败包装

当前部分包装器会把底层第三方异常包装成：

- `missforest`: `ScpValueError("IterativeImputer failed: ...")`
- `softimpute`: `ScpValueError("SoftImpute failed: ...")`

因此冻结合同是：

- 对外优先暴露 ScpTensor 自有异常体系
- 不应把原始 sklearn / fancyimpute 异常直接泄露成稳定 API

### 11.5 退化输入的当前行为

当前实现对退化输入通常更偏向“给出退化输出”而不是报错：

- 无缺失：复制原层并直接返回
- 全矩阵全缺失：除 `none` 外，多数方法会退化到全零或零主导输出
- 极小矩阵 / 少量检测值：各方法有内部 fallback，而不一定拒绝执行

这意味着：

- 当前 `scptensor.impute` 不是强防御式 gatekeeper
- 数据是否值得插补，更多仍由上游 QC 和 benchmark 设计负责

## 12. 方法家族的当前工程语义

### 12.1 baseline 家族

- `none`：显式 no-op，对照基线；保留 `NaN`
- `zero`：把缺失直接补成 `0`
- `row_mean` / `row_median`：按样本行统计
- `half_row_min`：按样本行最小值乘比例

当前冻结语义：

- baseline 方法不是“仅供测试”的私有路径
- 它们是 stable registry 的正式成员
- `none` 必须继续保留，不能被当成可删除冗余

### 12.2 邻域 / 回归家族

- `knn`
- `lls`
- `missforest`

共同特点：

- 主要利用样本间相似性或回归关系
- 当前实现都以样本行为主要建模对象
- 适合 `MCAR / MAR-like` 场景，但当前代码不会强制阻止它们用于其他机制

### 12.3 低秩 / 潜变量家族

- `bpca`
- `iterative_svd`
- `softimpute`

共同特点：

- 依赖整矩阵低秩结构
- 当前通常会 densify
- 适合完整矩阵补全类场景

### 12.4 左删失 MNAR 家族

- `qrilc`
- `minprob`
- `half_row_min` 可作粗糙低值 fallback

冻结语义：

- 这些方法假设“较小值更接近低丰度/检出下限”
- 它们不是 generic all-purpose missingness solver
- 但当前代码不会强制验证输入是否满足该假设

### 12.5 负值与 log-space 边界

结合当前代码与现有回归测试，MNAR 方法必须保留以下语义：

- `qrilc` 和 `minprob` 在数据中存在负值时，不得无条件强制裁成非负
- 只有当当前观测值整体非负时，才允许把随机填补结果裁到 `>= 0`

这保证了：

- 线性非负强度空间不会被填出负值
- log-space 负值场景不会被错误夹断

## 13. 优化安全不变量

后续若对 `scptensor.impute` 做性能优化、代码去重、底层替换或 API 收敛，以下不变量必须保留，除非文档和测试同时明确升级：

1. 矩阵方向不变：始终是 `n_samples x n_features`。
2. 原始 finite 值逐元素不变；只允许修改原始 `NaN` 位置。
3. 输出层 shape 不变，所属 assay 不变，`obs` / `var` 不变。
4. 当前缺失判定依据仍是 `np.isnan(X)`，不能悄悄改成“`0` 也算缺失”。
5. `M` 更新规则不变：原始缺失位置标记为 `MaskCode.IMPUTED`。
6. 统一注册名不变：`none / zero / row_mean / row_median / half_row_min / knn / lls / bpca / missforest / iterative_svd / softimpute / qrilc / minprob`。
7. `method="auto"` 的机制映射不变：`mcar -> knn`，`mar -> missforest`，`mnar -> qrilc`。
8. `impute(...)` 与各包装函数继续返回被原地追加新层的同一个 `container`。
9. `softimpute` 继续是 optional dependency 边界，不能在无说明情况下变成硬依赖。
10. 不在 imputation 阶段隐式做 aggregation、normalization、batch correction 或 state remapping。
11. 当前 layer collision 的覆盖语义若要修改，必须作为显式 API 变更处理，不能在优化中静默改变。

## 14. 对后续代码优化的直接约束

如果后续要把 `scptensor.impute` 继续工程化收敛，最安全的方向是：

1. 保留 registry 名与方法家族边界不变。
2. 在不改数值合同的前提下统一 no-missing fast path 的 history 行为。
3. 把文档级主输出层逐步收敛到 `imputed`，但保留当前方法特异层名作为兼容层。
4. 若要新增 state-aware imputation，应作为新能力显式引入，而不是悄悄改变“所有 `NaN` 都会被填补”的现有语义。

在此之前，`docs`、benchmark 与核心实现都应继续把本文件视为当前 imputation 模块的冻结实现边界。
