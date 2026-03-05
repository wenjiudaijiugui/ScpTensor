# ScpTensor I/O 输入规范整理（DIA-NN / Spectronaut）

## 1. 文档目标

本文档用于统一 `scptensor/io` 对 DIA-NN 与 Spectronaut 定量输出的理解，对齐当前 I/O 实现与测试。重点覆盖：

- 软件输出形态：长表（long）与透视矩阵（pivot/matrix）
- 定量层级：蛋白级与肽段/前体级
- 字段映射：样本、特征、定量值、q/FDR
- 导入最小必需列与推荐列
- 过滤与清洗建议

当前公开入口（与 `AGENTS.md` 一致）：

- `scptensor.io.load_quant_table`
- `scptensor.io.load_diann`
- `scptensor.io.load_spectronaut`
- `scptensor.io.load_peptide_pivot`
- `scptensor.io.aggregate_to_protein`


## 2. 适用范围与版本说明

- DIA-NN：基于官方 README 中 Output / Main output reference 描述（访问日期：2026-03-03）
- Spectronaut：基于 Spectronaut 19 Manual（v4）中的 Report 章节与 Most Relevant Report Headers（访问日期：2026-03-03）
- 生态补充：MSstats / protti 对 Spectronaut 导出列名的实践约定（作为补充，不替代官方）

说明：
- 不同版本、模板、自定义导出会造成列名差异。本文将区分“官方主列”和“实践常见别名”。
- 本文聚焦定量矩阵导入，不覆盖鉴定谱图层面细节。


## 3. 四类输入原型（对应当前 I/O 统一接口）

1. `protein-long`：蛋白级长表（每行 = 某蛋白在某 run 的定量）
2. `protein-matrix`：蛋白级矩阵（行 = 蛋白，列 = 样本）
3. `peptide-long`：肽段/前体级长表（每行 = 某前体在某 run 的定量）
4. `peptide-matrix`：肽段/前体级矩阵（行 = 肽段/前体，列 = 样本）

这四类可覆盖 DIA-NN 与 Spectronaut 的主要下游分析入口。


## 4. DIA-NN 输出信息整理

### 4.1 常见输出文件

- 主报告：`*.parquet`（长表，信息最全；也可导出为文本）
- 蛋白矩阵：`*.pg_matrix.tsv`
- 前体矩阵：`*.pr_matrix.tsv`
- 基因矩阵：`*.gg_matrix.tsv`、`*.unique_genes_matrix.tsv`

### 4.2 DIA-NN 长表关键列（官方主列）

- 样本维度：`Run`
- 前体/肽段标识：`Precursor.Id`, `Modified.Sequence`, `Stripped.Sequence`, `Precursor.Charge`
- 蛋白映射：`Protein.Group`, `Protein.Ids`, `Protein.Names`, `Genes`
- 定量值：
  - 前体级：`Precursor.Quantity`, `Precursor.Normalised`
  - 蛋白级：`PG.MaxLFQ`, `PG.TopN`
- 质量控制：
  - 前体相关：`Q.Value`, `Global.Q.Value`, `Lib.Q.Value`
  - 蛋白相关：`PG.Q.Value`, `Global.PG.Q.Value`, `Lib.PG.Q.Value`

### 4.3 DIA-NN 矩阵要点

- `pg_matrix` 适合直接导入蛋白矩阵。
- `pr_matrix` 适合直接导入前体矩阵。
- 矩阵导出受 DIA-NN 内部 q-value 过滤参数影响；导入时仍建议显式记录过滤阈值（便于复现）。


## 5. Spectronaut 输出信息整理

### 5.1 报告形态（官方术语）

- Normal Report：长表（行式），保留更多上下文列
- Run Pivot Report：宽表/矩阵（列式），适合直接构建表达矩阵

### 5.2 Spectronaut 长表关键列（官方主列）

- 样本维度：`R.FileName`（部分导出模板可见同义写法）
- 蛋白级：
  - 标识：`PG.ProteinGroups`
  - 定量：`PG.Quantity`
  - 质量：`PG.Qvalue`
  - 注释：`PG.Genes`, `PG.ProteinNames`
- 肽段/前体级：
  - 肽段组：`PEP.GroupingKey`, `PEP.GroupingKeyType`, `PEP.Quantity`
  - 前体组：`EG.PrecursorId`, `EG.TotalQuantity`, `EG.Qvalue`, `EG.Identified`
- 片段级（如做更细粒度分析）：`FG.Id`, `FG.Quantity`

### 5.3 Spectronaut Pivot 报告要点

- 通常会把 run 相关信息编码进列名，并拼接定量类型后缀。
- 常见后缀示例（实践中常见）：`_Quantity`, `_PeptideQuantity`, `_PrecursorQuantity`, `_Intensity`, `_PeakArea`
- 导入时建议采用“后缀集合 + 列名前缀回收样本名”的规则进行解析。


## 6. 统一导入最小列规范（ScpTensor 建议）

### 6.1 `protein-long`

必需列：
- `sample_id`：DIA-NN `Run` | Spectronaut `R.FileName`
- `feature_id`：DIA-NN `Protein.Group` 或 `Protein.Ids` | Spectronaut `PG.ProteinGroups`
- `quantity`：DIA-NN `PG.MaxLFQ`（优先）/ `PG.TopN` | Spectronaut `PG.Quantity`

推荐列：
- `q_value`：DIA-NN `PG.Q.Value`（或 `Global.PG.Q.Value`）| Spectronaut `PG.Qvalue`
- 注释：`Genes` / `PG.Genes`, `Protein.Names` / `PG.ProteinNames`

### 6.2 `protein-matrix`

必需列：
- 行标识（蛋白）：`Protein.Group` / `Protein.Ids` / `PG.ProteinGroups`
- 样本列：每列一个样本，单元格为蛋白定量值

推荐列：
- 蛋白注释列（基因、名称）
- 若有全局质量列，可作为 `var` 元数据保留

### 6.3 `peptide-long`

必需列：
- `sample_id`：DIA-NN `Run` | Spectronaut `R.FileName`
- `feature_id`：DIA-NN `Precursor.Id`（优先）或 `Modified.Sequence`
  | Spectronaut `EG.PrecursorId`（优先）或 `PEP.GroupingKey`
- `quantity`：DIA-NN `Precursor.Quantity`（或 `Precursor.Normalised`）
  | Spectronaut `EG.TotalQuantity`（或 `PEP.Quantity`）

推荐列：
- `q_value`：DIA-NN `Q.Value` / `Global.Q.Value` | Spectronaut `EG.Qvalue`
- 蛋白映射：`Protein.Group` / `Protein.Ids` / `PG.ProteinGroups`

### 6.4 `peptide-matrix`

必需列：
- 行标识（前体/肽段）：`Precursor.Id` / `EG.PrecursorId` / `PEP.GroupingKey`
- 样本列：每列一个样本，单元格为肽段/前体定量值

推荐列：
- 与蛋白的映射列（用于后续聚合）
- 质量列（如 q-value）并作为 feature 元数据保留


## 7. DIA-NN 与 Spectronaut 字段映射（统一语义层）

| 统一语义 | DIA-NN 常用列 | Spectronaut 常用列 |
|---|---|---|
| 软件识别 | `Run` | `R.FileName` |
| 样本 ID | `Run` | `R.FileName` |
| 蛋白 ID | `Protein.Group`, `Protein.Ids` | `PG.ProteinGroups` |
| 肽段/前体 ID | `Precursor.Id`, `Modified.Sequence` | `EG.PrecursorId`, `PEP.GroupingKey` |
| 蛋白定量 | `PG.MaxLFQ`, `PG.TopN` | `PG.Quantity` |
| 肽段/前体定量 | `Precursor.Quantity`, `Precursor.Normalised` | `EG.TotalQuantity`, `PEP.Quantity` |
| 蛋白质量 | `PG.Q.Value`, `Global.PG.Q.Value` | `PG.Qvalue` |
| 前体质量 | `Q.Value`, `Global.Q.Value` | `EG.Qvalue` |
| 蛋白注释 | `Genes`, `Protein.Names` | `PG.Genes`, `PG.ProteinNames` |


## 8. 过滤与清洗建议（导入阶段）

1. q/FDR 过滤
   蛋白级和前体级分开处理，建议默认阈值从 `0.01` 起（可配置）。
2. 定量列优先级
   蛋白级优先更稳定的归一化聚合列（如 DIA-NN `PG.MaxLFQ`，Spectronaut `PG.Quantity`）。
3. 重复键处理
   若同一 `sample_id + feature_id` 出现多行，需定义聚合规则（`max` / `sum` / `mean`）。
4. 缺失值与 0 值
   建议导入时统一转为缺失语义（而非混用 0 与 NA）。
5. 元数据保留
   把关键注释与质量列保留到 `var` 元数据，避免二次读取原文件。


## 9. 与当前生态的兼容补充（非官方主规范）

在 MSstats / protti 的 Spectronaut 工作流中，常见额外列包括：

- `R.Condition`, `R.Replicate`（实验设计信息）
- `F.PeakArea`, `F.NormalizedPeakArea`（片段级强度）
- `F.ExcludedFromQuantification`（定量排除标记）

建议：
- 这些列可作为“可选兼容列”处理。
- 不应作为核心必需字段，以避免对导出模板产生过强耦合。


## 10. 与当前 `scptensor/io` 实现对齐建议

1. 接口只暴露四类输入原型（protein/peptide × long/matrix）。
2. 先识别软件，再识别格式，再选 level，最后解析列。
3. 建立“主列 + 别名”字典，并在解析日志中记录命中情况。
4. 把过滤阈值、聚合策略、定量列选择写入 provenance，保证可复现。


## 11. 参考来源

- DIA-NN 官方 README
  https://raw.githubusercontent.com/vdemichev/DiaNN/master/README.md
- Spectronaut 19 Manual v4（Biognosys）
  https://biognosys.com/content/uploads/2024/09/Spectronaut-19-manual-v4.pdf
- MSstats: Spectronaut 格式转换文档
  https://rdrr.io/bioc/MSstats/man/SpectronauttoMSstatsFormat.html
- MSstats: 对应实现源码
  https://rdrr.io/bioc/MSstats/src/R/SpectronauttoMSstatsFormat.R
- protti 输入准备工作流（Rmd）
  https://raw.githubusercontent.com/jpquast/protti/master/vignettes/input_preparation_workflow.Rmd
