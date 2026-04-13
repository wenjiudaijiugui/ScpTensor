# ScpTensor Experimental PSM QC Helper Contract

本文档只冻结 `scptensor.experimental.qc_psm` 的边界与当前可观察行为。
它是 experimental pre-aggregation helper，不属于 stable protein-level QC 主线。

## Status

`qc_psm` 当前正式定位：

- 保留在 `scptensor.experimental`
- 以模块对象 `qc_psm` 暴露
- 解释为 experimental peptide/PSM QC helper
- 不迁移到 stable `scptensor.qc`

## Canonical Import

```python
from scptensor.experimental import qc_psm
```

不应把这些写成 canonical user entry：

- `from scptensor.qc import qc_psm`
- `import scptensor as scp; scp.qc_psm`

## Scope

当前模块级公开面：

- `filter_psms_by_pif`
- `filter_contaminants`
- `pep_to_qvalue`
- `filter_psms_by_qvalue`
- `compute_sample_cv`

旧别名 `compute_median_cv` 已退场，不再保留兼容层。

## Boundary

`qc_psm` 不属于：

- stable protein-level QC 主线
- peptide -> protein aggregation
- downstream embedding / clustering helper

它属于：

- experimental namespace
- pre-aggregation / peptide-PSM helper

## Assay Naming

当前默认 assay naming 已收口到：

- `assay_name="peptides"`

兼容 alias 仍接受：

- `peptide`

并且 alias 解析在验证、执行、provenance 记录阶段保持一致。

## Current Behavior

### Filtering helpers

- `filter_psms_by_pif`
- `filter_contaminants`
- `filter_psms_by_qvalue`

当前都属于按 feature 过滤 assay 的容器级 helper：

- 从 `assay.var` 读列
- 生成 keep mask
- 调用 `container.filter_features(...)`
- 返回新的 `ScpContainer`
- 追加 provenance

`filter_contaminants(patterns=None)` 的默认行为属于合同；
但内置 regex 集合本身不是 public API。

### Array helper

`pep_to_qvalue()` 当前是纯 NumPy helper：

- 输入一维 `pep`
- 输出一维 `qvalue`
- 不读写 `ScpContainer`

### Sample summary helper

`compute_sample_cv()` 是当前唯一 canonical sample-CV helper：

- 不过滤 feature
- 不新建 assay
- 对输入容器做 copy
- 在 `obs` 中写入：
  - `sample_cv`
  - `is_high_sample_cv`

## Failure Contract

当前显式失败边界：

- assay 不存在 -> `AssayNotFoundError`
- 所需列不存在 -> `ScpValueError`
- `pep_to_qvalue(method=...)` 非 `storey / bh` -> `ValueError`
- `lambda_param` 不在 `[0, 1)` -> `ValueError`
- `compute_sample_cv()` layer 缺失 -> `ScpValueError`

这里特别冻结一条实现事实：

- `compute_sample_cv()` 的 layer 缺失当前不是 `LayerNotFoundError`
- 文档不能把它误写成与 stable QC 完全一致的错误族

## Provenance

当前 action 名必须继续稳定可见：

- `filter_psms_by_pif`
- `filter_contaminants`
- `filter_psms_by_qvalue`
- `compute_sample_cv`

尤其：

- `compute_sample_cv` 是 canonical action
- 不应恢复 `compute_median_cv` 或旧列名 `median_cv / is_high_cv`

## Do Not Cross

- 不要把 `qc_psm` 塞回 stable `scptensor.qc`
- 不要恢复 `compute_median_cv` 兼容层
- 不要把 peptide/PSM helper 误写成 downstream display helper
- 不要把模块内部 contaminant regex 常量当成 public surface
