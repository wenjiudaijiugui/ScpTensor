# ScpTensor I/O Contract (`DIA-NN / Spectronaut`)

本文档只冻结当前稳定 I/O 合同。更细的 importer 语义、缺失状态讨论与
vendor-normalized 边界，继续看 `docs/internal/review_io_state_mapping_20260312.md`。

## Stable Entrypoints

当前公开入口与 `AGENTS.md` 一致：

- `scptensor.io.load_quant_table`
- `scptensor.io.load_diann`
- `scptensor.io.load_spectronaut`
- `scptensor.io.load_peptide_pivot`
- `scptensor.io.aggregate_to_protein`

边界：

- `load_peptide_pivot` 只负责 peptide/precursor matrix 导入
- peptide/precursor -> protein 转换必须由 `aggregate_to_protein()` 单独触发

## Supported Scope

- 软件：`DIA-NN`、`Spectronaut`
- 层级：`protein`、`peptide`
- 表形：`long`、`matrix`

四类稳定输入原型：

1. `protein-long`
2. `protein-matrix`
3. `peptide-long`
4. `peptide-matrix`

超出这个范围，不属于当前稳定 I/O 合同。

## Canonical Loader Surface

```python
from scptensor.io import (
    load_quant_table,
    load_diann,
    load_spectronaut,
    load_peptide_pivot,
)
```

默认 assay naming：

- `protein` level -> `proteins`
- `peptide` level -> `peptides`

默认导入 layer 名是：

- `raw`

这里的 `raw` 是 canonical imported layer name，不等于“必然未经 vendor normalization”。
若输入列本身是 vendor-normalized，解释应通过 provenance 字段如
`input_quantity_is_vendor_normalized` 表达，而不是再引入第二套默认 layer 命名。

## Vendor Field Families

稳定文档只保留当前 importer 真正依赖的主字段家族，不再追踪长尾模板变体。

| 语义 | DIA-NN 主列 | Spectronaut 主列 |
| --- | --- | --- |
| sample id | `Run` | `R.FileName` |
| protein id | `Protein.Group`, `Protein.Ids` | `PG.ProteinGroups` |
| peptide/precursor id | `Precursor.Id`, `Modified.Sequence` | `EG.PrecursorId`, `PEP.GroupingKey` |
| protein quantity | `PG.MaxLFQ`, `PG.TopN` | `PG.Quantity` |
| peptide quantity | `Precursor.Quantity`, `Precursor.Normalised` | `EG.TotalQuantity`, `PEP.Quantity` |
| protein q/FDR | `PG.Q.Value`, `Global.PG.Q.Value` | `PG.Qvalue` |
| precursor q/FDR | `Q.Value`, `Global.Q.Value` | `EG.Qvalue` |

说明：

- DIA-NN `PG.MaxLFQ`、`PG.TopN` 已带方法学语义，不应被写成“未处理 raw”
- `normalized` / `normalised` 类列通常仍是线性尺度，不等于 logged layer
- Spectronaut 导出模板可变；若只有最终 pivot matrix，解释必须更保守

## Minimal Column Contract

### `protein-long`

必需：

- sample column
- protein feature column
- protein quantity column

推荐：

- protein q/FDR column
- gene / protein-name annotation

### `protein-matrix`

必需：

- protein row-id column
- one column per sample with protein quantity

推荐：

- annotation columns kept into `var`

### `peptide-long`

必需：

- sample column
- peptide/precursor feature column
- peptide/precursor quantity column

推荐：

- q/FDR column
- protein mapping column for later aggregation

### `peptide-matrix`

必需：

- peptide/precursor row-id column
- one column per sample with peptide/precursor quantity

推荐：

- protein mapping column
- quality columns preserved into feature metadata

## Loader Behavior

当前 `load_quant_table()` 的稳定行为：

- 先解析软件：`DIA-NN` / `Spectronaut`
- 再解析 level：`protein` / `peptide`
- 再解析表形：`long` / `matrix`
- `fdr_threshold` 若提供，必须在 `[0, 1]`
- 空输入直接报错
- assay 默认为 `proteins` 或 `peptides`
- 最终在 `container.history` 中记录导入 provenance

`load_diann()` 与 `load_spectronaut()` 只是 vendor-fixed 包装层。

`load_peptide_pivot()` 只是：

- `level="peptide"`
- matrix-oriented import convenience entry

它不会替代后续 aggregation。

## Provenance Contract

importer 层最小应记录这些信息：

- `path`
- `software`
- `level`
- `assay_name`
- `format`
- `resolved_feature_column`
- `resolved_quantity_column`
- `resolved_sample_column`
- `used_fdr_column`
- `input_quantity_is_vendor_normalized`
- `fdr_threshold`
- `layer_name`

文档、错误信息和 benchmark 解释都应继续围绕这组 provenance 字段，而不是依赖手工记忆导出模板。

## State Mapping Boundary

当前文档只冻结保守边界：

- 明确可用定量值 -> `VALID`
- 明确 run 间匹配/transfer -> `MBR`
- 明确低丰度/未检出/类似 vendor `0` 占位 -> `LOD`
- 明确 q/FDR 或规则过滤 -> `FILTERED`
- 后验填补 -> `IMPUTED`
- 只有最终 matrix、无法恢复来源 -> `UNCERTAIN`

当前 matrix/pivot 导入的工程基线仍应解释为：

- finite cell -> `VALID`
- non-finite cell -> `UNCERTAIN`

这只是 importer 的保守工程基线，不是“vendor 缺失机制已完全恢复”的声明。

## Naming Contract

I/O 示例与文档继续优先使用：

- assay: `proteins`, `peptides`
- layers: `raw`, `log`, `norm`, `imputed`

不要在 I/O 文档里把这些实现局部名字写成仓库级默认：

- `protein`, `peptide`
- `normalized`, `normalised`
- `log2`
- `trqn_norm`

## Read Order

I/O 相关变更时，按这个顺序读：

1. `AGENTS.md`
2. 本文档
3. `docs/internal/review_io_state_mapping_20260312.md`
4. `scptensor/io/api.py`
5. `benchmark/README.md`
