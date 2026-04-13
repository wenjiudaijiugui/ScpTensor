# ScpTensor Experimental Downstream Contract

本文档只冻结 `dim_reduction / cluster / scptensor.experimental` 的发布边界。
它们允许存在，但不进入 stable preprocessing release 验收。

## Scope

覆盖：

- `reduce_pca`
- `reduce_tsne`
- `reduce_umap`
- `cluster_kmeans`
- `cluster_leiden`
- `scptensor.experimental` 对这些 API 的重导出边界

不覆盖：

- stable preprocessing 主线
- `qc_psm` 的独立 helper 合同
- differential expression、feature selection、biological interpretation

## Release Boundary

稳定主线终点仍然是：

- DIA-NN / Spectronaut 输入
- 可审计的 protein-level quantitative matrix

因此：

- `reduce_*` / `cluster_*` 是 experimental downstream helper
- 它们不是 release acceptance 必选交付物
- downstream 结果不能反向定义主线 preprocessing 是否完成

## Canonical Import Path

用户文档优先写：

```python
from scptensor.experimental import (
    reduce_pca,
    reduce_tsne,
    reduce_umap,
    cluster_kmeans,
    cluster_leiden,
)
```

虽然真实模块路径仍可用：

- `scptensor.dim_reduction`
- `scptensor.cluster`

但这些路径不应被写成 stable mainline API。

顶层 `scptensor` 也不应重导出这些 helper。

## Output Slots

当前冻结的输出槽位：

- reduction -> 新 assay，layer 固定为 `X`
- clustering -> 写入 `obs`

因此：

- `pca` / `tsne` / `umap` assay 不是主线 quantitative assay
- cluster labels 不是稳定 preprocessing deliverable
- reduction 和 clustering 不能被文档化成同一种输出结构

## Input Boundary

当前 downstream helper 的真实前提：

- 需要已有 assay
- 需要已有 layer
- 输入矩阵必须是 complete finite matrix

cluster 默认输入仍是 downstream-oriented：

- `cluster_kmeans`: `assay_name="pca"`, `base_layer="X"`
- `cluster_leiden`: `assay_name="reduce_pca"`, `base_layer="X"`

这进一步说明它们依赖主线产物，而不是主线本身。

## Copy / Mutation Facts

当前实现没有统一 deep-copy 语义。只冻结这些事实：

- 所有 helper 都返回新 `ScpContainer`
- `reduce_pca` 会重建 source assay，并向 source `var` 写入 loading 列
- `reduce_tsne` / `reduce_umap` 复用原 assay 对象，但新增 target assay
- `cluster_*` 新建 `obs`，但共享 assay mapping / assay objects

这些对象身份语义若要改，必须先改合同与测试，不能在重构里静默漂移。

## Provenance

当前 action 名必须继续稳定可见：

- `reduce_pca`
- `reduce_tsne`
- `reduce_umap`
- `cluster_kmeans`
- `cluster_leiden`

当前关键参数命名也应继续保持：

- reduction: `source_assay`, `source_layer`, `target_assay`
- clustering: `source_assay`, `source_layer`, `output_key`

## Do Not Cross

- 不要把 `reduce_*` / `cluster_*` 升格为 stable preprocessing contract
- 不要让 stable preprocessing 默认依赖 `pca/X`、`umap/X`、`tsne/X` 或 cluster labels
- 不要把 optional downstream dependencies 当成 stable runtime 必备项
- 不要把 exploratory score 或 embedding 图写成主线完成判据
