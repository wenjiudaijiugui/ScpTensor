# ScpTensor Experimental Downstream 边界合同（`dim_reduction / cluster`，2026-03-17）

## 1. 文档目标

本文档冻结 ScpTensor 中 experimental downstream helper 的发布边界，重点覆盖：

- `scptensor.dim_reduction`
- `scptensor.cluster`
- `scptensor.experimental` 对上述模块的重导出

它回答的是：

- `reduce_*` / `cluster_*` 在仓库中到底处于什么发布层级；
- 它们与 stable preprocessing release 的边界在哪里；
- 哪些输出可以存在于容器中，但不能被误写成核心交付物；
- 后续如果继续扩写 downstream helper，哪些边界不能被悄悄突破。

本文档基于以下仓库内事实：

- `AGENTS.md`
- `README.md`
- `scptensor/__init__.py`
- `scptensor/experimental/__init__.py`
- `scptensor/dim_reduction/__init__.py`
- `scptensor/dim_reduction/base.py`
- `scptensor/dim_reduction/pca.py`
- `scptensor/dim_reduction/tsne.py`
- `scptensor/dim_reduction/umap.py`
- `scptensor/cluster/__init__.py`
- `scptensor/cluster/base.py`
- `scptensor/cluster/graph.py`
- `scptensor/cluster/kmeans.py`
- `tests/dim_reduction/test_pca.py`
- `tests/dim_reduction/test_tsne.py`
- `tests/dim_reduction/test_umap.py`
- `tests/cluster/test_cluster.py`

## 2. 范围与非范围

### 2.1 范围

本文档覆盖：

- `reduce_pca`
- `reduce_tsne`
- `reduce_umap`
- `cluster_kmeans`
- `cluster_leiden`
- `scptensor.experimental` 对这些 API 的命名空间边界

### 2.2 非范围

本文档不覆盖：

- stable preprocessing 主线的 I/O / aggregation / transformation / normalization / imputation / integration / qc 合同
- differential expression
- feature selection
- 下游 biological interpretation
- `qc_psm` 的具体实现合同

说明：

- `scptensor.experimental` 当前还暴露 `qc_psm`，但本文档只用它来说明 namespace 边界；其独立实现边界见 `docs/qc_psm_contract.md`。

## 3. 发布边界结论

### 3.1 稳定主线交付物不变

按项目总合同，ScpTensor 当前稳定主线交付物仍然是：

- 从 DIA-NN / Spectronaut 定量表出发
- 形成可审计的 protein-level quantitative matrix

这意味着 stable preprocessing release 的验收边界停在：

- protein-level quantitative assay
- 及其 preprocessing provenance

### 3.2 `dim_reduction / cluster` 不属于 stable preprocessing release

`reduce_*` 与 `cluster_*` 当前在仓库中允许存在，也有测试，也可以被文档提及；
但它们的发布层级必须冻结为：

- experimental downstream analysis helpers
- 不属于核心 preprocessing release contract
- 不作为 release acceptance 的必选交付物

### 3.3 downstream 结果不能反向定义主线合同

因此后续任何文档、教程、报告、benchmark 叙述都必须遵守：

- `pca` / `tsne` / `umap` assay 不是主线 quantitative assay
- `obs` 里的 cluster labels 不是稳定 preprocessing deliverable
- downstream score 或图形结果不能用来倒推“主线 preprocessing 一定正确”

## 4. 导入边界合同

### 4.1 用户文档的 canonical import 路径

用户文档、README、教程中的 canonical downstream import 路径应冻结为：

```python
from scptensor.experimental import (
    reduce_pca,
    reduce_tsne,
    reduce_umap,
    cluster_kmeans,
    cluster_leiden,
)
```

原因不是这些函数只能从这里导入，而是该 namespace 明确表达了发布边界。

### 4.2 真实模块路径仍然存在

当前源码中这些函数也可以直接从模块导入：

- `scptensor.dim_reduction`
- `scptensor.cluster`

但这些路径的稳定解释应为：

- 模块级 public surface 存在
- 发布边界仍然是 experimental
- 不应因为能直接导入，就把它们误写成顶层 stable mainline API

### 4.3 顶层 `scptensor` 不重导出这些 API

`scptensor.__all__` 当前不导出：

- `reduce_pca`
- `reduce_tsne`
- `reduce_umap`
- `cluster_kmeans`
- `cluster_leiden`

这不是遗漏，而是当前发布边界的一部分。

### 4.4 `scptensor.experimental` 当前是边界命名空间，而不是独立实现层

`scptensor.experimental.__init__` 当前本质上是重导出层：

- 从 `scptensor.dim_reduction` 导出 `reduce_*` 与 `SolverType`
- 从 `scptensor.cluster` 导出 `cluster_*`
- 额外挂出 `qc_psm`

仓库当前没有 `tests/experimental/` 目录，但 `tests/core/test_experimental_api.py`
已经直接覆盖 `experimental.__all__` 的完整 re-export 面，因此它的角色应解释为：

- 边界声明层
- 薄包装命名空间
- 不是另一套独立实现

## 5. 与 stable preprocessing release 的硬边界

### 5.1 允许存在，但不计入主线验收

`dim_reduction / cluster` 可以：

- 作为 exploratory helper 存在于仓库
- 作为教程的后续附加步骤出现
- 被 `viz` 消费进行 downstream display

但它们不应：

- 成为 stable preprocessing release 的必测验收条件
- 进入“主线 quantitative matrix 已完成”的定义
- 替代 protein-level matrix 的最终交付语义

### 5.2 stable 模块不应依赖 experimental 输出作为默认输入

稳定预处理模块的默认输入不应依赖：

- `pca/X`
- `umap/X`
- `tsne/X`
- `obs` 中的 cluster labels

也就是说：

- downstream helper 可以消费 stable preprocessing 产物
- stable preprocessing 不应反向依赖 downstream helper 才算完成

### 5.3 benchmark 边界

若后续对 preprocessing 模块做 benchmark：

- 可以把 downstream helper 作为补充可视化或辅助分析
- 不应把 `reduce_*` / `cluster_*` 的存在与否写成 preprocessing benchmark 的必须构件

## 6. `dim_reduction` 当前实现合同

### 6.1 当前公开面

`scptensor.dim_reduction.__all__` 当前导出：

- `reduce_pca`
- `reduce_tsne`
- `reduce_umap`
- `SolverType`

模块头注释已明确其状态是 experimental。

### 6.2 输入合同

三类 reduction 当前都要求：

- 一个已有 assay
- 一个已有 layer
- 输入矩阵不含 `NaN` / `Inf`

`dim_reduction.base._check_no_nan_inf()` 当前显式拒绝：

- `NaN`
- `Inf`

因此 reduction 的当前真实前提是：

- complete finite matrix

### 6.3 assay 解析语义

`dim_reduction.base._validate_assay_layer()` 当前会调用：

- `resolve_assay_name(container, assay_name)`

因此 reduction 路径在 assay 名上带有 alias-friendly 行为。

当前 `cluster.base._validate_assay_layer()` 也已对齐为同一套解析逻辑，
因此 experimental helper 在 assay alias 处理上当前的稳定解释是：

- validation 阶段使用 resolved assay name
- 真正执行与 provenance 记录也使用同一 resolved assay name

### 6.4 输出位置合同

当前 reduction 输出统一写成：

- 一个新的 assay
- assay 内 layer 名固定为 `X`
- `ScpMatrix.M` 为全零 `int8` mask

默认 assay 名分别是：

- `pca`
- `tsne`
- `umap`

因此当前稳定解释是：

- downstream coordinate payload 放在 assay-level `X`
- `layer='X'` 在这里是 experimental coordinate assay 的标准用法
- 它不应迁移成 stable quantitative assay 的默认 layer 命名

### 6.5 history / provenance

当前 reduction 会追加历史记录：

- `reduce_pca`
- `reduce_tsne`
- `reduce_umap`

并记录统一命名的主要参数：

- `source_assay`
- `source_layer`
- `target_assay`

后续若重构实现，不能删除这些 action 名或把它们改成模糊无区分的日志。

### 6.6 `reduce_pca` 的额外副作用

`reduce_pca` 与 `reduce_tsne` / `reduce_umap` 当前存在一个重要差异：

- `reduce_pca` 会把 loading 列写回原 assay 的 `var`
- `reduce_tsne` / `reduce_umap` 不会改写原 assay 的 `var`

因此：

- PCA 不是纯“只加一个新 assay”这么简单
- 它还会更新 source assay 的 feature metadata

### 6.7 结果容器复制语义

当前实现差异如下：

- `reduce_pca`
  - 返回新的 `ScpContainer`
  - `obs` 与输入 container 共享同一对象引用
  - `history` 使用新的列表对象
  - `assays` 使用新的 mapping
  - source assay 会被替换成新的 assay 对象
  - source assay 的 `var` 会被重建并追加 loading 列
  - source assay 中既有 layer dict 会被重建，但底层 source `ScpMatrix` 仍沿用原对象引用
  - 新增 target assay（默认 `pca`）
- `reduce_tsne` / `reduce_umap`
  - 返回新的 `ScpContainer`
  - `obs` 与输入 container 共享同一对象引用
  - `history` 使用新的列表对象
  - `assays` 使用新的 mapping
  - 原有 source assay 对象直接复用
  - 新增 target assay（默认 `tsne` / `umap`）

因此当前 reduction 家族并不是统一的 deep-copy 语义。
更准确地说，当前冻结的是：

- 它们都返回“新 container”
- 但 source assay / source layer / obs 的共享粒度并不一致
- 后续重构若要改变这些对象身份语义，必须先更新合同与测试

## 7. `cluster` 当前实现合同

### 7.1 当前公开面

`scptensor.cluster.__all__` 当前导出：

- `cluster_kmeans`
- `cluster_leiden`

模块头注释已明确其状态是 experimental。

### 7.2 默认输入位置

cluster 当前默认使用：

- `assay_name="pca"`
- `base_layer="X"`

这进一步说明它们是 downstream helper：

- 默认假设 reduction 已经先发生
- 不属于主线 preprocessing 的默认终点

### 7.3 输出位置合同

cluster 当前不新建 assay，而是：

- 把 labels 写进 `container.obs`

默认 key 格式由 `_get_default_key()` 生成，例如：

- `kmeans_k3`
- `leiden_r1.0`

因此当前稳定解释是：

- reduction 输出放 assay
- clustering 输出放 obs

两者输出槽位不同，不能混写。

### 7.4 label dtype

`cluster.base._add_labels_to_obs()` 当前会把 labels 转成字符串再写入 `obs`。

因此测试已锁定：

- cluster 结果列 dtype 为字符串类

不应在无迁移的情况下静默改成整数标签列。

### 7.5 结果容器与 assay 引用

cluster 当前通过 `_add_labels_to_obs()` 新建 container，但直接沿用：

- `assays=container.assays`

因此当前稳定事实是：

- 返回新的 `ScpContainer`
- `obs` 是新的 DataFrame 对象
- `history` 使用新的列表对象
- `assays` mapping 与输入 container 共享同一对象引用
- mapping 内的 assay 对象也与输入 container 共享

这意味着 cluster 当前不是“assay dict copy + assay object share”，而是更强的共享：

- 调用后新增的 cluster labels 只写入结果 container 的 `obs`
- 但结果 container 与输入 container 仍共用同一套 assay mapping / assay objects

### 7.6 history / provenance

cluster 当前会追加：

- `cluster_kmeans`
- `cluster_leiden`

并记录统一命名的关键参数：

- `source_assay`
- `source_layer`
- `output_key`

以及方法级参数，例如：

- `n_clusters`
- `backend`
- `n_neighbors`
- `resolution`

### 7.7 backend 与 optional dependency 边界

`cluster_kmeans` 当前支持：

- `backend="sklearn"`
- `backend="numba"`

其中：

- `numba` backend 依赖 `core.jit_ops` 中的 k-means helper
- 若 `NUMBA_AVAILABLE=False` 且要求 `numba` backend，会抛 `ImportError`

`cluster_leiden` 当前依赖：

- `igraph`
- `leidenalg`

测试也把它们视作 optional dependency。

因此当前稳定合同是：

- clustering helper 允许存在 optional backend / optional dependency
- 但 stable preprocessing release 不应以这些 optional downstream dependency 的存在作为必备条件

## 8. 当前已知实现不对称点

### 8.1 reduction 与 cluster 的输出槽位不同

当前：

- reduction -> 新 assay / layer `X`
- cluster -> `obs` 新列

后续若要统一 downstream result model，需要另立方案；当前不能把二者文档化成同一输出结构。

### 8.2 copy 语义不一致

当前：

- `reduce_pca` 会重建 source assay，并只在 source layer matrix 级别复用既有对象
- `reduce_tsne` / `reduce_umap` 复用原 assay 对象，但 assay mapping 不共享
- `cluster_*` 同时复用 assay mapping 与 assay 对象

因此 downstream helper 当前不是统一的 immutable deep-copy API 家族。

### 8.3 `experimental` namespace 是边界层，不是独立实现层

当前虽然已有 `tests/core/test_experimental_api.py` 直接锁定 facade re-export，
但这仍不意味着 `experimental` 成为独立实现层。它的语义仍然是：

- `scptensor.experimental` 主要承担 import 边界表达
- 其稳定性来自下层模块

这意味着如果未来要把 `experimental` 进一步做成稳定 facade，需要额外测试与合同。

## 9. 优化时不得破坏的高优先级边界

后续若要扩展 downstream helper，必须优先保留：

1. `reduce_*` / `cluster_*` 属于 experimental downstream，而非 stable preprocessing release
2. 用户文档优先使用 `scptensor.experimental` import 路径
3. 顶层 `scptensor` 不把这些 API 升格为 stable top-level export
4. reduction 结果默认写入 new assay 的 layer `X`
5. clustering 结果默认写入 `obs`
6. downstream helper 不得反向成为 stable preprocessing 的默认依赖

## 10. 对后续重构的直接指导

基于当前仓库状态，后续如果要继续推进 downstream helper，最合理的路径是：

- 继续把 `scptensor.experimental` 作为用户文档的统一入口
- 把当前 reduction / clustering 的 copy 语义视为已冻结的实现合同
- 若未来要统一成单一 copy 策略，必须先显式改 contract 和回归测试
- 若未来要把某一类 downstream helper 升为 stable API，必须单独立约，而不能顺手越界
- 不要把 `pca` / `umap` / `tsne` assay 或 cluster obs 列写进 stable preprocessing release 的核心交付定义

这意味着：ScpTensor 目前允许 exploratory downstream 能力存在，但 stable release 的完成判据仍然必须停在 protein-level preprocessing 主线。
