# ScpTensor Integration 实现冻结合同（`scptensor.integration`，2026-03-16）

## 1. 文档目标

本文档冻结 `scptensor.integration` 当前实现边界，服务于后续代码优化、API 收敛与测试补强。它回答的是：

- 当前已经实现了哪些 integration / batch-correction 方法；
- 哪些方法属于稳定的 protein-level 矩阵合同，哪些只是 exploratory embedding-level 路径；
- matrix input 与 embedding input 在当前代码里分别如何被接受与校验；
- assay / layer / covariate / diagnostics / failure / provenance 目前各自遵守什么约束；
- 后续重构时，哪些行为可以优化，哪些行为不能被悄悄改变。

本文档是 **implementation-facing contract**，不是文献综述，也不是 benchmark 排名结论。

本文档基于以下仓库内事实：

- `scptensor/integration/__init__.py`
- `scptensor/integration/base.py`
- `scptensor/integration/none.py`
- `scptensor/integration/limma.py`
- `scptensor/integration/combat.py`
- `scptensor/integration/mnn.py`
- `scptensor/integration/nonlinear.py`
- `scptensor/integration/harmony.py`
- `scptensor/integration/scanorama.py`
- `scptensor/integration/diagnostics.py`
- `scptensor/__init__.py`
- `tests/core/test_integration_api.py`
- `tests/integration/test_integration.py`
- `tests/integration/test_diagnostics.py`

## 2. 作用域与非作用域

### 2.1 作用域

本文档覆盖：

- `scptensor.integration.__init__`
- `scptensor.integration.base`
- `scptensor.integration.none`
- `scptensor.integration.limma`
- `scptensor.integration.combat`
- `scptensor.integration.mnn`
- `scptensor.integration.nonlinear`
- `scptensor.integration.harmony`
- `scptensor.integration.scanorama`
- `scptensor.integration.diagnostics`

### 2.2 非作用域

本文档不覆盖：

- 文献优先级、方法优劣结论与综述层推荐
- AutoSelect 最终评分权重的研究合理性
- peptide / precursor -> protein 聚合
- downstream dim reduction / clustering 的实现合同
- 外部包 `harmonypy` / `scanorama` 的完整 API 语义

其中项目边界仍以仓库总合同为准：ScpTensor 的稳定预处理主线是 **DIA 单细胞蛋白组的 protein-level quantitative matrix**；`dim_reduction` / `cluster` 等下游帮助模块属于 experimental downstream helpers。

### 2.3 当前公开 API

`scptensor.integration.__all__` 当前导出三类公共入口：

- unified interface / registry helpers：
  - `integrate`
  - `list_integrate_methods`
  - `get_integrate_method`
  - `IntegrateMethod`
  - `IntegrationMethodInfo`
  - `register_integrate_method`
  - `get_integrate_method_info`
  - `list_integrate_method_info`
- individual integration methods：
  - `integrate_none`
  - `integrate_combat`
  - `integrate_limma`
  - `integrate_harmony`
  - `integrate_mnn`
  - `integrate_scanorama`
- diagnostics helpers：
  - `compute_batch_mixing_metric`
  - `compute_batch_asw`
  - `compute_lisi_approx`
  - `compute_kbet`
  - `compute_ilisi`
  - `integration_quality_report`

`scptensor.__all__` 当前只从 integration 模块顶层重导出 individual methods：

- `integrate_none`
- `integrate_combat`
- `integrate_limma`
- `integrate_harmony`
- `integrate_mnn`
- `integrate_scanorama`

不会重导出：

- `integrate`
- `list_integrate_methods`
- `get_integrate_method`
- `get_integrate_method_info`
- `list_integrate_method_info`
- `register_integrate_method`
- diagnostics helpers

因此当前稳定边界是：

- `scptensor.integration` 子包是 unified dispatch、registry metadata 与 diagnostics 的公共入口；
- 顶层 `scptensor` 只暴露 direct integration methods，不把 unified dispatch 或 diagnostics 升格为顶层 stable API。

## 3. 稳定场景边界

### 3.1 稳定默认终点

`scptensor.integration` 的稳定默认终点不是“任意 integrated representation”，而是：

- 目标 assay：`proteins`
- 目标数据层级：protein-level quantitative matrix
- 目标用途：继续作为稳定 preprocessing 主线的输入，或作为最终交付矩阵的一部分

这里要明确两件事：

1. 当前大多数函数技术上接受任意存在的 `assay_name`。
2. 但按项目合同，稳定主线仍应把 `proteins` 视为默认目标；integration 不是 peptide/precursor 主线的终点定义。

### 3.2 稳定 vs exploratory

当前实现中，integration 方法的稳定性不是靠文档口头描述决定，而是由 `IntegrationMethodInfo` 元数据显式声明：

- `integration_level="matrix"` 且 `recommended_for_de=True`：稳定矩阵级方法
- `integration_level="matrix"` 且 `recommended_for_de=False`：矩阵级，但不进入稳定 DE-oriented candidate set
- `integration_level="embedding"`：exploratory embedding-level 路径

据当前代码，矩阵级 / embedding-level 的冻结边界如下：

- 稳定矩阵级：`none`、`limma`
- 手动可用但不属于稳定 DE-oriented 默认集合：`combat`
- exploratory embedding-level：`mnn`、`harmony`、`scanorama`

注意：这里的 `embedding` 是 **方法合同与输出用途定位**，不是“所有方法都已经严格做了 embedding 输入校验”。当前只有 `harmony` 显式强制 embedding-like 输入；`mnn` 与 `scanorama` 目前仍可直接接受 complete matrix。

### 3.3 文档命名与实现默认值

仓库文档的 canonical naming 已统一为：

- assay：`proteins` / `peptides`
- layers：`raw` / `log` / `norm` / `imputed` / `zscore`

但 integration 当前函数签名默认仍是：

- `assay_name="protein"`
- `base_layer="raw"`

这一差异通过 `resolve_assay_name()` 兼容：`protein` / `proteins` 会互相解析。冻结合同应视为：

- 文档与教程层：优先写 `proteins`
- 当前实现兼容层：允许 `protein`

## 4. 当前已实现的方法全集

### 4.1 统一入口与注册名

统一入口是：

- `integrate(container, method=..., **kwargs)`

当前注册表中的方法名包括：

- `none`
- `combat`
- `limma`
- `mnn`
- `harmony`
- `nonlinear`
- `scanorama`

其中：

- `nonlinear` 是 `integrate_harmony` 的兼容别名
- 对外推荐函数名是 `integrate_harmony`

### 4.2 方法元数据与当前角色

| 注册名 | 对外函数 | integration_level | recommended_for_de | 当前角色 | 默认 `new_layer_name` | 依赖 |
|---|---|---|---|---|---|---|
| `none` | `integrate_none` | `matrix` | `True` | 稳定 no-op baseline | `none` | 内置 |
| `limma` | `integrate_limma` | `matrix` | `True` | 稳定矩阵级批次校正 | `limma` | 内置 |
| `combat` | `integrate_combat` | `matrix` | `False` | 手动矩阵级校正；不进稳定 AutoSelect 默认集合 | `combat` | 内置 |
| `mnn` | `integrate_mnn` | `embedding` | `False` | exploratory integration；当前实现为 full MNN progressive merge，仍可直接接受 complete matrix | `mnn_corrected` | 内置 |
| `harmony` | `integrate_harmony` | `embedding` | `False` | exploratory embedding integration；显式要求 embedding-like 输入 | `harmony` | `harmonypy` |
| `nonlinear` | `integrate_harmony` | `embedding` | `False` | `harmony` 兼容别名 | `harmony` | `harmonypy` |
| `scanorama` | `integrate_scanorama` | `embedding` | `False` | exploratory integration；当前代码要求 complete-finite matrix，并只接受可写回同一 assay 的 corrected matrix | `scanorama` | `scanorama` |

冻结解释：

- `integration_level` 是方法用途合同，不是“结果一定存到独立 embedding 容器”的硬编码结构。
- `recommended_for_de` 直接影响当前 AutoSelect 对 stable candidate set 的过滤；后续不能随意改含义。

## 5. 输入合同

### 5.1 容器与轴方向

integration 模块当前固定遵守 ScpTensor 核心矩阵方向：

- `X.shape == (n_samples, n_features)`
- 行轴是样本轴
- 列轴是特征轴

后续优化不能通过隐式转置改变这些轴语义。

### 5.2 assay / layer 访问合同

所有 integration 方法都依赖现有 assay / layer，而不是新建 assay 作为默认行为：

- 输入 assay 必须已存在于 `container.assays`
- 输入 layer 必须已存在于 `container.assays[assay_name].layers`
- 输出默认添加到 **同一个 assay** 下

因此当前结构上，integration 不是“跨 assay 搬运器”，而是“对现有 assay 增加新 layer 的处理器”。

### 5.3 protein alias 合同

`validate_layer_params()` 会调用 `resolve_assay_name()`，因此：

- 若容器中只有 `proteins`，传入 `assay_name="protein"` 仍可工作
- 若容器中只有 `protein`，传入 `assay_name="proteins"` 也可工作

冻结合同里，这种 singular/plural alias 兼容应保留。

### 5.4 matrix-level 方法输入

当前矩阵级方法包括：

- `integrate_none`
- `integrate_limma`
- `integrate_combat`

它们的共同点：

- 默认从 `base_layer="raw"` 读取
- 输出 layer 加回原 assay
- 语义上面向 quantitative matrix，而不是低维 embedding

稳定文档应把它们解释为 protein-level 主线方法；即便代码技术上可对任意 assay 调用，也不应把这件事扩写成“稳定支持任意 feature space 的最终矩阵校正”。

### 5.5 embedding-level 方法输入

当前 embedding-level 方法包括：

- `integrate_mnn`
- `integrate_harmony`
- `integrate_scanorama`

但三者的输入门禁并不一致。

#### `integrate_harmony`

`integrate_harmony` 通过 `validate_embedding_input()` 做显式校验。当前接受两类输入：

1. embedding assay + layer `X`
   - 典型形式：`assay_name="pca"`, `base_layer="X"`
2. protein assay 上的 PCA-like layer
   - 典型形式：`assay_name="protein"`, `base_layer="pca"`
   - 还兼容 `pc` / `pcs` / `pca_*` / `*_pca`

当前 `harmony` 会显式拒绝：

- `protein/raw`
- 任何看起来不是 embedding-like 的 layer

#### `integrate_mnn`

`integrate_mnn` 当前 **不做** embedding-like 输入校验。它的真实实现合同是：

- 接受任意存在的 assay/layer
- 把输入转为 dense array
- 要求 complete finite matrix（无 `NaN / Inf`）
- 默认对输入做 cosine normalization（`cos_norm_in=True`）
- 默认对输出也做 cosine normalization（`cos_norm_out=True`）
- 对多批次输入执行 **progressive merge**
  - 默认按 batch 首次出现顺序合并
  - 可通过 `merge_order` 显式指定顺序
- 可选先对当前 reference+target merge step 做 PCA (`use_pca=True`) 再做 MNN 搜索
- correction vector 会对 **target batch 的全部细胞** 做 Gaussian smoothing，而不是只改直接参与 MNN pair 的细胞
- `svd_dim > 0` 时，会从 correction vector 中减去平行于 paired-cell biological subspace 的分量
- `var_adj=True` 时，会执行 batchelor 风格的 variance adjustment，以缓解 kissing effects

因此：

- `mnn` 的 `integration_level="embedding"` 表示它被归为 exploratory integration 方法
- 但这不等于它今天只接受 embedding assay
- 当前代码里，`protein/raw` 也是可调用输入
- 当前实现已不再保留旧的 pairwise-only multi-batch heuristic；多批次必须走 progressive merge

#### `integrate_scanorama`

`integrate_scanorama` 与 `mnn` 类似，当前 **不做** embedding-like 输入校验。其真实实现合同是：

- 接受任意存在的 assay/layer
- 先转换为 dense array
- 当前代码要求 complete finite matrix（无 `NaN / Inf`）
- 按 batch 切分后调用 `scanorama.correct(datasets, genes_list, ...)`
- 若 `dimred` 未显式提供，wrapper 会保留 Scanorama 上游默认降维维数，而不是把 `None` 透传给外部库
- wrapper 会把 `corrected_genes` 对齐回 assay 当前 feature ID 顺序
- wrapper 会在写回 assay 前恢复 **原始 sample 顺序**；不会因为内部按 batch 分组调用而重排 `obs`
- 当前 wrapper 只接受“纠正后 feature set 与 assay 原 feature set 完全一致”的情况
- `return_dimred=True` 当前被显式拒绝，因为低维 embedding 不能安全写回同一 assay layer

因此，`scanorama` 的 exploratory 性质来自方法元数据与用途定位，而不是输入校验。

### 5.6 低维表示的结构边界

ScpTensor 当前 stable core 没有 `obsm` / `uns` 这类槽位。因此低维表示只能通过以下两种方式承载：

1. 单独 assay
   - 例如 `pca` assay + layer `X`
2. 同 assay 下的兼容 layer
   - 但该 layer 仍受 `Assay.add_layer()` 的 feature-width 约束

这意味着：

- 对真正降维后的结果，**更稳妥的结构方式是单独 embedding assay**
- 在 `proteins` assay 内塞一个低维 layer，不属于稳定主路径

## 6. 完整矩阵、缺失值与稀疏输入合同

### 6.1 各方法对 non-finite 值的当前行为

当前实现对 `NaN / Inf` 的处理不是统一的：

- `none`：直接复制，不检查 `NaN`
- `limma`：允许 `NaN`；按 feature 仅在 observed samples 上拟合，并保留原缺失位置为 `NaN`；但 `Inf / -Inf` 会显式失败
- `combat`：要求 complete finite matrix；遇到 `NaN / Inf` 抛出 `ValidationError`
- `mnn`：要求 complete finite matrix；遇到 `NaN / Inf` 抛出 `ScpValueError`
- `harmony`：拒绝 `NaN`，抛出 `ScpValueError`
- `scanorama`：要求 complete finite matrix；遇到 `NaN / Inf` 抛出 `ScpValueError`

冻结合同应以 **当前实现代码** 为准，而不是以未来想支持的行为为准。

### 6.2 complete-matrix vs missing-preserving

按当前实现：

- `limma` 是唯一明确支持“missing-preserving batch correction”的矩阵级方法
- `combat` 仍处于 complete-finite-matrix contract
- `mnn` 现在也处于 complete-finite-matrix contract
- `harmony` 仍是 complete-matrix contract
- `scanorama` 现在也是 complete-finite-matrix contract

这也是为什么当前 stable protein-level 默认矩阵方法应优先理解为：

- baseline：`none`
- correction：`limma`

### 6.3 稀疏输入与输出存储类型

当前大多数方法都允许 sparse input，但是否保留 sparse output 不是稳定承诺：

- `none`：直接复制输入矩阵类型
- `combat`：内部 densify；结果若输入原本 sparse 且稀疏率足够高，可能转回 CSR
- `limma`：内部 densify；结果通过 `preserve_sparsity()` 决定是否转回 sparse
- `mnn` / `harmony` / `scanorama`：内部 densify；结果再按稀疏阈值决定是否转回 sparse

冻结不变量是：

- shape 与数值语义不能变
- 稀疏/稠密的具体存储类型可以优化

## 7. batch、covariate 与 confounding guardrails

### 7.1 batch_key 合同

所有 integration 方法都要求 `batch_key` 能在 `container.obs` 中解析，但校验强度不同：

- `none`：只要求 `batch_key` 存在，用于 API 一致性与 downstream evaluator 兼容
- `combat` / `limma` / `mnn` / `harmony` / `scanorama`：要求
  - `batch_key` 存在
  - 至少 2 个 batch
  - 每个 batch 至少 2 个样本

当前这些失败统一走 `ScpValueError`。

### 7.2 covariate 支持范围

当前只有以下方法支持显式 biological covariates：

- `integrate_combat(covariates=...)`
- `integrate_limma(covariates=...)`

其余方法当前没有 covariate 建模入口；冻结合同里不能把它们描述成“会自动保留生物信号”。

### 7.3 covariate 编码规则

当前线性模型路径的 covariate 设计矩阵规则是：

- 总是带 `intercept`
- categorical covariates 用 dummy encoding
- 对分类变量 `drop_first=True`
- `combat` 与 `limma` 的 covariates 都必须 fully observed / finite；缺失或非有限值会直接失败
- limma 的 batch terms 以 **sum-coded contrasts** 追加到 covariate 设计之后

这套编码顺序影响 rank-deficiency 判断，后续若优化实现，不能改变“confounding 应显式失败”的语义。

### 7.4 `reference_batch` 合同

仅 `integrate_limma()` 支持 `reference_batch`：

- 默认是 `None`
- `reference_batch=None` 时，按 limma `removeBatchEffect()` 的默认语义把 batch effect 去到中心，而不是对齐到第一个 batch
- 若显式提供 `reference_batch`，这是 ScpTensor 的扩展语义：会把校正结果平移到“指定 batch 保持不变”
- 若传入值不在实际 batch labels 中，抛 `ScpValueError`

### 7.5 confounding 失败策略

`combat` 与 `limma` 当前都遵守同一 guardrail：

- 若设计矩阵 rank deficient，则直接失败
- 错误类型是 `ValueError`
- 错误信息会包含 design columns，帮助定位 batch 与 covariate 的共线问题

冻结合同要求：

- 不得静默删除 covariate 列
- 不得自动把 fully confounded 设计降级成可运行模型
- 不得把结构性共线问题伪装成“成功但结果不可信”

## 8. diagnostics 合同

### 8.1 diagnostics 的定位

`scptensor.integration.diagnostics` 当前提供的是 **post-hoc quality metrics**，不是 integration 方法本身：

- `compute_batch_asw`
- `compute_batch_mixing_metric`
- `compute_lisi_approx`
- `compute_kbet`
- `compute_ilisi`
- `integration_quality_report`

这些函数当前都：

- 不修改容器
- 不写 history
- 只读取指定 assay/layer 与 `obs[batch_key]`

### 8.2 默认输入空间

diagnostics 当前默认参数是：

- `assay_name="pca"`
- `layer_name="X"`
- `batch_key="batch"`

这说明其默认语义是 **embedding/PCA representation 上的评估**，而不是 `proteins/raw` 上的默认解释。

### 8.3 当前各指标的冻结语义

- `compute_batch_asw`
  - 返回 **原始 ASW**
  - 数值范围 `[-1, 1]`
  - 解释方向：**越低越好**
  - 若有效样本太少或 batch 数不足，返回 `0.0`

- `compute_batch_mixing_metric`
  - 返回局部邻域 batch composition 与全局 composition 接近程度的启发式分数
  - 它是 ScpTensor 自定义 proxy，不等同于 `kBET` 或原始 `LISI`
  - 数值范围大致在 `[0, 1]`
  - 解释方向：**越高越好**
  - 样本或 batch 过少时，返回 `1.0`

- `compute_lisi_approx`
  - 返回基于固定 kNN + inverse Simpson 的近似值
  - 解释方向：**越高越好**
  - 单 batch 时返回 `1.0`
  - 双 batch 场景常落在 `[1, 2]`

- `compute_kbet`
  - 返回 fixed-k 邻域上的 `kBET acceptance rate`
  - 对每个样本，把局部 batch 组成与全局 batch 组成做卡方检验
  - 返回 `p >= alpha` 的样本比例，范围 `[0, 1]`
  - 解释方向：**越高越好**
  - 它实现了 kBET 的核心局部检验逻辑，但当前**不**复刻原始 R 包里的自适应邻域 / outsider 处理等全部启发式

- `compute_ilisi`
  - 返回更标准化的 `iLISI` 汇总值
  - 当前实现先在 fixed-k 邻域上做 perplexity-matched 局部概率，再计算 inverse Simpson
  - 默认 `scale=True`，因此默认返回缩放后的中位数 summary，范围 `[0, 1]`
  - 若 `scale=False`，返回 raw median iLISI，理论范围 `[1, n_valid_batches]`
  - 解释方向：**越高越好**
  - 小样本场景下，effective perplexity 会裁剪到可用邻居数

- `integration_quality_report`
  - 当前为了兼容既有 API，仍只返回旧的 3 个冻结指标：
    - `batch_asw`
    - `batch_mixing`
    - `lisi_approx`
  - 不会静默切换成 `compute_kbet` / `compute_ilisi`

### 8.4 diagnostics 对缺失值的当前行为

这些函数对输入矩阵都会先转 dense，然后：

- 对任何含 non-finite 值（`NaN / Inf`）的样本行做过滤
- 再在剩余样本上计算指标

因此 diagnostics 的当前合同是：

- 它们不会修复 non-finite 值
- 也不会因为少量 non-finite 行直接报错
- 但有效样本数会因此变化

### 8.5 diagnostics 的失败合同

integration diagnostics 当前不复用 `AssayNotFoundError` / `LayerNotFoundError`，而是直接抛：

- `ValueError("Assay '...' not found")`
- `ValueError("Layer '...' not found")`
- `ValueError("Batch key '...' not found in obs")`

冻结合同里，这一差异应视为当前实现事实。

## 9. 输出语义、layer 合同与 provenance

### 9.1 容器是否原地修改

当前 integration 方法都会：

- 在已有容器对象上新增 layer
- 返回同一个 `container`

也就是说：

- 它们不是 immutable transform
- 也不是 deep-copy transform

这一点对后续优化非常重要，不能在不说明迁移影响的情况下悄悄改成“返回新容器”。

### 9.2 source layer 的当前真实写回边界

当前方法实现的共同语义是：

- 先从 `base_layer` 读取输入
- 构造新的 `ScpMatrix`
- 再把结果写入 `new_layer_name`

因此当前真实行为应拆成两层理解：

1. 当 `new_layer_name` 与现有 layer 名不同：
   - 不会原地改写 `base_layer.X`
   - `base_layer.M` 也不会被原地改写
2. 当 `new_layer_name` 与某个现有 layer 同名，尤其是 `new_layer_name == base_layer`：
   - `Assay.add_layer()` 的覆盖语义会在写回阶段替换该 layer
   - 这意味着 source layer registry entry 也可能被最终覆盖

也就是说：

- 当前 integration 不会在“计算过程中”就地修改源层数组；
- 但最终写回仍受 layer collision 规则支配，不能把它写成“无条件保护 source layer”。

### 9.3 `new_layer_name` 与默认层名

当前默认输出层名是方法特异的：

| 方法 | 默认输出层 | history action |
|---|---|---|
| `none` | `none` | `integration_none` |
| `combat` | `combat` | `integration_combat` |
| `limma` | `limma` | `integration_limma` |
| `mnn` | `mnn_corrected` | `integration_mnn` |
| `harmony` / `nonlinear` | `harmony` | `integration_harmony` |
| `scanorama` | `scanorama` | `integration_scanorama` |

若 `new_layer_name=None`，则各方法会回退到自己的默认层名。

这里必须再区分两层命名语义：

- 仓库级 canonical quantitative layer taxonomy 仍是 `raw / log / norm / imputed / zscore`
- `none / limma / combat / mnn_corrected / harmony / scanorama` 这些 integration 默认层名是 **method-specific compatibility / comparison artifact**

因此冻结解释应为：

- direct API、benchmark 与 AutoSelect 可以继续保留这些方法特异层名
- 它们不应被误写成仓库级 canonical final layer naming
- 若某次 integration 结果要晋升为后续工作流主线层，应通过显式 rename/promote 步骤完成，而不是默认把方法名层当成 canonical 主结果层

### 9.4 layer 名冲突的当前行为

`Assay.add_layer()` 当前实现是直接赋值：

- 若 `new_layer_name` 已存在，会被 **静默覆盖**
- 当前不会抛冲突错误

这属于当前实现事实。后续如果要改成“禁止覆盖”，应视为显式 API 变更，而不是内部优化。

### 9.5 `M` 与 metadata 的当前行为

当前 integration 方法都会尽量传播 `M`：

- 若输入层 `M is not None`，则输出层复制 `M`
- 若输入层 `M is None`，则输出层也保持 `None`

当前 integration 方法 **不会** 生成新的 `ScpMatrix.metadata` 内容；输出层的 metadata 仍为空，除非未来显式扩展。

### 9.6 shape 合同

在稳定主路径里，输出层应与输入层保持相同 shape。

对当前代码的冻结解释是：

- `none` / `combat` / `limma` / `mnn` / `harmony`：当前都按“输入 shape == 输出 shape”实现
- `scanorama`：当 `return_dimred=False` 时保持原 shape 与原始 sample 顺序；`return_dimred=True` 当前显式失败，不属于稳定 protein-level layer 合同

特别是：

- 当前 ScpTensor `Assay.add_layer()` 要求 layer feature 数与 assay `var.height` 一致
- 因此任何真正改变 feature/dimension 数的输出，都不应被视为稳定 protein assay 合同的一部分
- 当前 `scanorama(return_dimred=True)` 选择显式失败，而不是把低维结果伪装成普通 assay layer

### 9.7 provenance 中的 assay 记录

当前 integration 方法里，只要某个 history params 包含 `assay` 字段，
该字段应记录 **解析后的 assay key**，也就是容器里实际被访问的 assay 名。

这意味着：

- 若容器里 assay 名是 `protein`，调用时传 `proteins`，history 仍写 `protein`
- 若容器里 assay 名是 `proteins`，调用时传 `protein`，history 仍写 `proteins`

对当前实现的冻结解释是：

- provenance 应描述真正执行所用的 assay，而不是保留调用方原样别名
- 这一规则当前适用于 `integrate_none`、`integrate_mnn`、`integrate_scanorama`
  等写入 `params["assay"]` 的路径

## 10. failure 合同

### 10.1 主要失败类型总表

| 失败条件 | 当前涉及方法 | 当前异常类型 | 说明 |
|---|---|---|---|
| assay 不存在 | 全部 integration 方法 | `AssayNotFoundError` | 通过 `validate_layer_params()` / `validate_embedding_input()` 抛出 |
| layer 不存在 | 全部 integration 方法 | `LayerNotFoundError` | 同上 |
| `batch_key` 不存在 | 全部 integration 方法 | `ScpValueError` | `none` 也会校验 |
| batch 数不足 / singleton batch | `combat` / `limma` / `mnn` / `harmony` / `scanorama` | `ScpValueError` | 当前至少要求 2 batches 且每批至少 2 个样本 |
| covariate 列缺失 | `combat` / `limma` | `ScpValueError` | 错误信息包含缺失列名 |
| covariates 含缺失或非有限值 | `combat` / `limma` | `ScpValueError` | 线性模型设计矩阵要求 fully observed / finite covariates |
| `reference_batch` 非法 | `limma` | `ScpValueError` | 必须存在于 batch labels |
| 设计矩阵秩亏 | `combat` / `limma` | `ValueError` | 视为 batch/covariate confounding guardrail |
| `Inf / -Inf` 出现在 limma 输入矩阵 | `limma` | `ScpValueError` | `NaN` 允许保留；无限值必须显式失败 |
| `svd_dim < 0` | `mnn` | `ScpValueError` | `svd_dim` 只能是非负整数 |
| complete finite matrix 违约 | `combat` | `ValidationError` | 明确提示改用 `integrate_limma()` 或先显式处理缺失 |
| complete finite matrix 违约 | `mnn` | `ScpValueError` | `NaN` 来自 `prepare_integration_data()`；`Inf` 由 `integrate_mnn()` 显式拒绝 |
| complete matrix 违约 | `harmony` | `ScpValueError` | 来自 `prepare_integration_data()` |
| complete finite matrix 违约 | `scanorama` | `ScpValueError` | `NaN` 来自 `prepare_integration_data()`；`Inf` 由 `integrate_scanorama()` 显式拒绝 |
| Scanorama 返回 feature set 与 assay 不一致 | `scanorama` | `ScpValueError` | 当前 wrapper 只支持可写回同一 assay 的 corrected matrix |
| `return_dimred=True` | `scanorama` | `ScpValueError` | 低维 embedding 不能安全写回当前 assay layer 结构 |
| 缺少外部依赖 | `harmony` / `scanorama` | `MissingDependencyError` | `scanorama` 由 decorator 触发；`harmony` 在函数体内导入 |
| Harmony 输入不是 embedding-like | `harmony` | `ScpValueError` | 明确拒绝 `protein/raw` |
| 输出 layer shape 与 assay 不匹配 | 任意方法 | `ValueError` | 由 `Assay.add_layer()` / `ScpMatrix` 结构校验触发 |
| diagnostics 输入缺失 | diagnostics | `ValueError` | 不走 ScpTensor 自定义异常 |

### 10.2 失败合同的冻结原则

后续优化允许改进：

- 错误消息可读性
- 异常上下文
- 失败前的参数检查顺序

但不能改变以下原则：

- 缺失结构性前提时必须失败，而不是静默修正
- confounded design 必须显式暴露
- method-specific 的 complete-matrix 约束必须保留为显式契约

## 11. 可安全优化的范围与不可变不变量

### 11.1 可安全优化的范围

后续可以优化：

- 内部线性代数实现
- 稀疏 / 稠密转换策略
- PCA / kNN / nearest-neighbor 的性能实现
- 公共校验函数与日志逻辑的重构
- diagnostics 的数值稳定性和性能

前提是下面这些冻结不变量不被破坏。

### 11.2 optimization-safe invariants

1. **项目主线边界不变**
   - 稳定预处理主线仍以 protein-level matrix 为终点。
   - integration 不能暗中扩写成 peptide/precursor 主线终点定义。

2. **方法元数据是权威合同**
   - `integration_level` 与 `recommended_for_de` 的语义不能漂移。
   - 依赖这些元数据的 AutoSelect / report 逻辑不能在文档不更新的情况下被悄悄改变。

3. **稳定默认候选集不应被静默扩大**
   - 以当前合同，稳定 DE-oriented matrix candidate set 仍是 `none` + `limma`。
   - `combat` 继续是 direct API 支持，但不属于默认稳定候选。
   - `mnn` / `harmony` / `scanorama` 继续属于 exploratory。

4. **source layer 不原地改写**
   - integration 必须新增输出 layer，而不是修改输入 layer。
   - `M` 的传播仍应保持 copy 语义。

5. **confounding 必须显式失败**
   - 不得静默删除 covariate
   - 不得把 rank-deficient design 自动修成“能跑”

6. **method-specific missingness contract 不得模糊化**
   - `limma` 的 missing-preserving 语义要保留
   - `combat` / `mnn` / `scanorama` 的 complete-finite-matrix 语义要保留，除非明确做版本化扩展
   - `harmony` 的 complete-matrix 语义也要保留，除非明确做版本化扩展

7. **diagnostics 保持只读**
   - diagnostics 只能读取容器，不写 layer、不写 provenance、不改 obs/var
   - 评分方向必须明确且稳定

8. **低维输出不应伪装成稳定 protein assay layer**
   - 当前 stable core 没有 `obsm`
   - 真正低维结果更适合独立 embedding assay
   - 任何会改变 feature/dimension 数的输出都不应被伪装成稳定 protein-level matrix path

9. **layer 覆盖语义若要变更，必须显式宣布**
   - 当前已有 layer 名冲突会被覆盖
   - 如果未来改成“禁止覆盖”，这应是明确 API 变更，而不是内部重构副作用

## 12. 面向后续实现的简短结论

按当前代码，`scptensor.integration` 的冻结实现边界可以简化为：

- 稳定 protein-level matrix integration：`none`、`limma`
- 手动矩阵级补充方法：`combat`
- exploratory integration：`mnn`、`harmony`、`scanorama`
- `harmony` 是唯一显式要求 embedding-like 输入的方法
- `limma` 是当前唯一明确保留 `NaN` 语义的 batch-correction 方法
- diagnostics 默认以 `pca/X` 为评估空间，且保持只读

后续任何核心代码优化，都应围绕这组合同推进，而不是在未更新文档与测试的前提下改变其边界。
