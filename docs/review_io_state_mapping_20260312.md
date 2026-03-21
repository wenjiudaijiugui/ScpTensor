# DIA-NN / Spectronaut 定量表到 ScpTensor importer 的状态映射合同：优先级文献综述（截至 2026-03-12）

## 1. 研究范围

- 研究问题：ScpTensor 应如何把 `DIA-NN` 与 `Spectronaut` 的 quantitative outputs 映射到稳定、可解释的 importer contract，尤其是列解析、FDR/filtering、normalized quantity、`MBR` 与缺失状态语义？
- 目标输出：为以下实现与文档提供证据化合同：
  - `scptensor.io.load_quant_table`
  - `scptensor.io.load_diann`
  - `scptensor.io.load_spectronaut`
  - `scptensor.io.load_peptide_pivot`
  - `docs/io_diann_spectronaut.md`
  - `scptensor.core.structures.MaskCode`
- 补充边界：除 vendor long/matrix importer 外，本综述也约束 `peptide-matrix -> protein aggregation` 交接前必须保留的 provenance，避免 `load_peptide_pivot` 导入后丢失 level/format/normalized-state 语义。
- 核心边界：遵循项目合同，支持的软件仅限 `DIA-NN` 与 `Spectronaut`；最终交付物是 **完整 protein-level quantitative matrix**。
- 检索日期：`2026-03-12`

### 1.1 关键词优先级（P1 / P2 / P3）

| 优先级 | 目的 | 关键词 |
|---|---|---|
| P1 | 方法核心 | output columns, q value, normalized quantity, match-between-runs, report semantics |
| P2 | 模态与软件 | DIA-NN, Spectronaut |
| P3 | 应用边界 | import contract, preprocessing, protein matrix, missingness semantics |

## 2. 检索策略与筛选标准

### 2.1 Query 梯度（P1 -> P2 -> P3）

1. 先检索 DIA-NN 与 Spectronaut 的官方输出/手册说明。
2. 再结合原始方法论文确认 `MBR`、cross-run normalization 与 protein quantification 的语义边界。
3. 最后对照 ScpTensor 当前 importer 代码与 I/O 规范文档，收束成可执行合同。

### 2.2 纳入标准

- 一手来源：官方文档、官方手册 PDF、原始方法论文。
- 直接涉及以下至少一项：
  - 主报告/透视矩阵中的列语义
  - q-value / FDR 与 quantitative columns 的关系
  - normalized quantity 与 raw quantity 的区分
  - `MBR` 或 cross-run matching 对结果状态语义的影响

### 2.3 排除标准

- 二手博客、论坛、非官方列名整理。
- 与当前 importer 合同边界无关的谱图级细节。

### 2.4 本轮纳入

- 初筛候选：`10+`
- 深读纳入：`6`
- 官方实现/手册来源：`3`
- 方法学论文来源：`3`

## 3. 逐篇证据摘要（Per-paper Summaries）

说明：本节统一沿用全仓库资源分型。除特别标注外，单篇文献条目默认记为 `论文证据`；官方软件/手册页记为 `模块规范 / 软件文档`；具体 accession 或 dataset page 记为 `数据入口`；可脚本化分发包记为 `资源包`。
共享高频条目的规范元数据统一以 `docs/references/citations.json` 为准；若本文件历史写法与 registry 在作者简称、发布日期、期刊、DOI 或 canonical URL 上不一致，以 registry 为准，本文件仅保留 importer contract 语义解释。

### 3.1 DIA-NN 官方文档（访问于 2026-03-12）

- 资源类型：`模块规范 / 软件文档`
- 链接：https://vdemichev.github.io/DiaNN/
- 文档直接信息：
  - 主报告包含 `Precursor.Id`、`Protein.Group`、`Precursor.Normalised`、`PG.MaxLFQ` 等列。
  - `Quantity` 表示非归一化量；`Normalised` 表示归一化量。
  - `Precursor.Quantity` 是 non-normalised precursor quantity，`Precursor.Normalised`、`Ms1.Normalised` 是归一化后的线性值。
  - `Normalisation.Factor` 满足 `normalised quantity = factor × non-normalised quantity`。
  - `MBR` 是两遍流程，会提高 identification numbers、data completeness 与 quantification accuracy。
  - `Zero quantities` 更合理的解释是低浓度 analyte；若要 log-transform，可将其替换为 `NA`。
- 对 ScpTensor 的意义：
  - importer 不能把 `Normalised` 列当作未处理 raw。
  - `MBR`、`0`、未报告、过滤失效、后验插补必须分开建模。

### 3.2 DIA-NN 原始论文，Nature Methods, 2020

- 题目：DIA-NN: neural networks and interference correction enable deep proteome coverage in high throughput
- 链接：https://doi.org/10.1038/s41592-019-0638-x
- 主要价值：
  - 提供 DIA-NN 作为现代 DIA 工作流核心软件的算法基础。
  - 其高 throughput / interference correction 背景解释了为什么 output semantics 不能脱离软件模式理解。
- 对 ScpTensor 的意义：
  - importer 合同应围绕 DIA-NN 的主输出列，而不是泛化到任意 DIA 软件导出。

### 3.3 Spectronaut 官方手册入口与 PDF（访问于 2026-03-12）

- 资源类型：`模块规范 / 软件文档`
- 手册入口：https://biognosys.com/resources/spectronaut-manual/
- PDF：https://biognosys.com/content/uploads/2024/09/Spectronaut-19-manual-v4.pdf
- 官方手册直接信息：
  - Spectronaut 提供 `Normal Report` 与 `Run Pivot Report` 两类输出。
  - 支持 `Automatic / Global / Local` 的 cross-run normalization。
  - `PG.Quantity`、`EG.TotalQuantity`、`Qvalue`、`NormalizedPeakArea` 等字段的解释依赖导出上下文。
- 对 ScpTensor 的意义：
  - Spectronaut importer 需要同时识别 `long` 与 `matrix/pivot` 两种形态。
  - `Quantity` 类列不能脱离导出模板和 normalization 设置解释。

### 3.4 Cox et al., Molecular & Cellular Proteomics, 2014

- 题目：Accurate Proteome-wide Label-free Quantification by Delayed Normalization and Maximal Peptide Ratio Extraction, Termed MaxLFQ
- 链接：https://doi.org/10.1074/mcp.M113.031591
- 主要发现：
  - `MaxLFQ` 是有特定 delayed normalization 与 peptide ratio extraction 语义的 protein quantification。
- 对 ScpTensor 的意义：
  - `PG.MaxLFQ` 不是普通 raw intensity。
  - 对这类列再次做 normalization 时，文档必须显式说明是“downstream task-specific normalization”，而不是 pretend 它仍是未处理原始量。

### 3.5 Wang et al., Nature Communications, 2025

- 题目：Benchmarking informatics workflows for data-independent acquisition single-cell proteomics
- 链接：https://doi.org/10.1038/s41467-025-65174-4
- 主要发现：
  - 在 DIA-sc 流程中，sparsity reduction、normalization、batch correction 都强依赖 upstream software output 的实际语义。
  - 论文明确比较了 `DIA-NN` 与 `Spectronaut` 路线。
- 对 ScpTensor 的意义：
  - importer contract 不是纯技术细节，它直接决定 benchmark 与 preprocessing 解释边界。

### 3.6 ScpTensor 当前 I/O 文档与实现（本地实现上下文）

- 资源类型：`本地实现上下文`
- 文档：[io_diann_spectronaut.md](io_diann_spectronaut.md)
- 代码：[mass_spec.py](../scptensor/io/mass_spec.py)
- 当前实现直接体现的事实：
  - `DIA-NN` 与 `Spectronaut` 的 `protein / peptide` 两层 profile 已分开。
  - 代码已经识别 vendor-normalized 列名，并为不同软件维护字段候选集合。
  - 当前 `MaskCode` 已定义 `VALID / MBR / LOD / FILTERED / OUTLIER / IMPUTED / UNCERTAIN` 七态。
- 对当前问题的意义：
  - 文档合同下一步最重要的是把现有解析逻辑的“语义假设”写清楚，而不是先扩展更多兼容层。

### 3.7 二次核查补充（资源分型、稳定入口与边界）

- `DIA-NN 官方文档` 与 `Spectronaut 官方手册/PDF`（accessed: `2026-03-12`）在本综述中都应固定记为 `模块规范 / 软件文档`；它们约束的是 importer 语义和列解释，而不是 benchmark 数据或公共资源包。
- `DIA-NN 2020`、`MaxLFQ 2014`、`Wang 2025` 统一属于 `论文证据`；这些来源为 importer contract 提供方法学边界，但不应被写成稳定数据入口。
- `ScpTensor 当前 I/O 文档与实现` 属于 `本地实现上下文`，它用于校对“当前代码是否已经体现这些合同”，不能替代外部证据来源。
- 由于本综述目标是 importer contract，本文件不单列 `数据入口` 或 `资源包`；若后续需要公共数据或 package 角色，应由公共 benchmark 数据综述单独承担。

## 4. 横向比较与证据分级

### 4.1 importer 最小状态映射合同

| importer 合同维度 | DIA-NN | Spectronaut | ScpTensor 建议 |
|---|---|---|---|
| 软件识别 | `Run`、`Protein.Group`、`Precursor.Normalised` 等主列 | `R.FileName`、`PG.Quantity`、`EG.TotalQuantity` 等主列 | 先识别软件，再识别 level/format |
| 格式识别 | 主报告长表、`pg_matrix`/`pr_matrix` | Normal Report、Run Pivot Report | `long` 与 `matrix` 必须拆开解析 |
| 定量尺度 | `Quantity` vs `Normalised` 明确区分 | `Quantity` 与 normalized/pivot 依导出设置解释 | 强制记录 `source_column` 和 `is_vendor_normalized` |
| FDR 语义 | `Q.Value` / `PG.Q.Value` / global q-values | `PG.Qvalue` / `EG.Qvalue` | q/FDR 字段不应静默丢弃 |
| MBR / 匹配 | DIA-NN 官方直接定义两遍 `MBR` | Spectronaut 需依模板/输出语义保守解释 | 保守映射到 `MBR` 或 `UNCERTAIN` |
| 零值 | 官方说明更接近低浓度占位 | 需依导出上下文解释 | 不应自动并入 `VALID` |

### 4.2 一致结论（facts）

1. vendor 输出列本身就带有方法学语义，尤其是 `MaxLFQ`、`Normalised`、`NormalizedPeakArea`、`Qvalue`。
2. importer 必须先区分 `software × level × format`，否则同一列名在不同上下文下可能被误解。
3. q/FDR 字段不只是附属元数据，它直接决定某个 quantitative value 是否可被视为稳定观测。
4. vendor-normalized quantity 仍通常是线性尺度，不应与 logged layer 混淆。
5. `MBR` / match-like completion 与直接观测不是同一状态，应在 `MaskCode` 或 provenance 中保留差异。

### 4.3 分歧与解释（inference）

- [推断] 对 Spectronaut 而言，若用户只提供最终 pivot 矩阵而不附带导出模板信息，ScpTensor 无法总是唯一判定其 normalized / matched 状态，此时应保守标成 `UNCERTAIN` 或仅保留 provenance 字段，不应过度推断。
- [推断] 对 DIA-NN 而言，若输入来自 `PG.MaxLFQ` 或 `Precursor.Normalised`，更稳妥的文档策略是继续使用 canonical `raw` 作为导入后主 layer 名，但在 provenance 中强制记录 `is_vendor_normalized=true` 与 `source_column`，避免把 layer 名本身再拆成第二套默认体系。

### 4.4 证据强度

- 高：DIA-NN 官方文档、Spectronaut 官方手册、Wang 2025
- 中高：MaxLFQ 原始论文、DIA-NN 2020 原始论文
- 中：当前本地实现与 I/O 文档的工程收束

## 5. 面向 ScpTensor 的实践建议

### 5.1 importer 应强制保留的最小 provenance

- `source_software`
- `source_column`
- `data_level`：`protein` / `peptide`
- `table_format`：`long` / `matrix`
- `is_vendor_normalized`
- `fdr_column_used`
- `sample_column_used`
- `feature_column_used`

### 5.2 对 `MaskCode` 的最小映射建议

- `VALID`
  - 当前 run / sample 中可解释为可用观测值
- `MBR`
  - 明确的 run 间匹配/第二遍转移量
- `LOD`
  - vendor `0` 或缺失更接近低丰度/检测极限
- `FILTERED`
  - 明确被 q/FDR 或 QC 规则排除
- `UNCERTAIN`
  - 只有最终矩阵、无法恢复上游状态来源时的保守默认

### 5.3 `docs/io_diann_spectronaut.md` 最值得补的内容

1. `source_column -> layer semantics` 对照表
2. `q/FDR column -> filtering state` 对照表
3. `zero / missing / MBR / normalized` 的状态映射边界
4. 对 `PG.MaxLFQ`、`Precursor.Normalised`、`PG.Quantity`、`EG.TotalQuantity` 的默认 importer 解释

### 5.4 与当前代码的对齐建议

- 当前 `_ImportProfile` 与 `_is_vendor_normalized_column()` 已经具备 contract 雏形。
- 下一步不应先加更多软件兼容，而应把现有 DIA-NN / Spectronaut 语义写进用户文档与错误信息。

## 6. 风险边界

1. Spectronaut 导出模板高度可配置，部分字段语义只有结合导出参数才能完全解释。
2. 若上游只提供 pivot matrix，ScpTensor 无法完全恢复 MBR / filtering 细节，必须保守。
3. `PG.MaxLFQ` 等聚合列本身已经包含方法学假设，二次处理时必须在文档中说明边界。
4. importer contract 若不显式记录 provenance，很容易在 normalization、log transform、imputation 之间产生 silent double-processing。

## 7. 对后续实现/文档的优先建议

1. 在 [io_diann_spectronaut.md](io_diann_spectronaut.md) 中新增 `state mapping` 章节。
2. 在 importer 返回对象的 provenance 中显式记录 `is_vendor_normalized` 与 `source_column_used`。
3. 在用户文档中统一使用 `raw / log / norm / imputed` 这套 canonical layer 名；若需要表达 vendor-normalized 线性输入，应写成“`raw` layer with `is_vendor_normalized=true`”，而不是新增默认 layer 名家族。

## 8. Shared Citation Registry Coverage

以下共享高频条目的规范元数据以 `docs/references/citations.json` 为准：

- `diann_docs`
- `demichev2020_natmethods_diann`
- `spectronaut_manual`
- `cox2014_mcp_maxlfq`
- `wang2025_natcom_dia_scp_benchmark`

注：`Spectronaut 19 Manual v4 PDF` 已作为 `spectronaut_manual` 的 URL alias 收敛进 citation registry；本地实现上下文锚点保留在正文 `3.6`，不再作为尾部额外 citation 清单单列。
