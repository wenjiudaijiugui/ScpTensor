# DIA 驱动单细胞蛋白组预处理中对数变换与尺度契约：优先级文献综述（截至 2026-03-12）

## 1. 研究范围

- 研究问题：在 `DIA-NN / Spectronaut` 输入下，ScpTensor 应如何在保持 canonical `raw / log / norm` layer 命名的同时，区分 vendor-normalized provenance 与 logged scale 语义，并据此约束 `log_transform()` 与归一化候选方法的适用边界？
- 目标输出：为以下实现与文档提供证据化约束：
  - `scptensor.transformation.log_transform`
  - `scptensor.autoselect.evaluators.normalization.NormalizationEvaluator`
  - `scptensor.io.load_quant_table` 对 vendor-normalized 列的识别
- 核心边界：项目合同要求最终 deliverable 是完整的 `protein-level quantitative matrix`；除 `aggregation` 外，下游模块默认在 protein 层工作。
- 检索日期：`2026-03-12`

### 1.1 关键词优先级（P1 / P2 / P3）

| 优先级 | 目的 | 关键词 |
|---|---|---|
| P1 | 方法核心 | log transform, scale contract, normalization, pseudo-count, quantile normalization, MaxLFQ |
| P2 | 模态与软件 | DIA-NN, Spectronaut, DIA proteomics, single-cell proteomics |
| P3 | 应用边界 | preprocessing, AutoSelect, low-input proteomics |

## 2. 检索策略与筛选标准

### 2.1 Query 梯度（P1 -> P2 -> P3）

1. 先检索定量值尺度、对数变换与归一化前提的一手文献。
2. 再限定到 DIA-NN / Spectronaut 的官方输出语义。
3. 最后加入单细胞与 benchmark 研究，确认 modern DIA-sc 工作流中的实际执行顺序。

### 2.2 纳入标准

- 一手来源：期刊官网、DOI 页面、PubMed、官方软件手册/文档。
- 直接涉及以下至少一项：
  - vendor 输出量化列的尺度定义
  - `log2` / `log10` 转换时机、零值处理、pseudo-count 处理
  - 量化或归一化方法对输入尺度的隐含假设
  - 可迁移到 ScpTensor layer contract 的工程约束

### 2.3 排除标准

- 只给出可视化示例、没有说明尺度语义的教程。
- 与 DIA 软件输出无关、无法迁移到当前 API 合同的资料。

### 2.4 本轮纳入

- 初筛候选：`20+`
- 深读纳入：`8`
- 其中官方软件实现证据：`2`
- DIA-sc 直接流程证据：`3`
- 可迁移方法学证据：`3`

## 3. 逐篇证据摘要（Per-paper Summaries）

说明：本节统一沿用全仓库资源分型。除特别标注外，单篇文献条目默认记为 `论文证据`；官方软件/手册页记为 `模块规范 / 软件文档`；具体 accession 或 dataset page 记为 `数据入口`；可脚本化分发包记为 `资源包`。
共享高频条目的规范元数据统一以 `docs/references/citations.json` 为准；若本文件历史写法与 registry 在作者简称、发布日期、期刊、DOI 或 canonical URL 上不一致，以 registry 为准，本文件仅保留 scale/provenance 合同解释。

### 3.1 DIA-NN 官方文档（访问于 2026-03-12）

- 资源类型：`模块规范 / 软件文档`
- 链接：https://vdemichev.github.io/DiaNN/
- 文档直接信息：
  - `Quantity` 表示未归一化量，`Normalised` 表示已归一化量。
  - `Precursor.Quantity` 是 non-normalised quantity；`Precursor.Normalised`、`Ms1.Normalised` 是归一化后的线性值。
  - `Normalisation.Factor` 直接定义了 `normalised quantity = normalisation factor × non-normalised quantity`。
  - 官方说明默认报告的是 normalized quantities，并支持 `global`、`RT-dependent`、`no normalization` 等 cross-run normalization 模式。
  - 若 quantity 为 `0`，更合理的解释是低浓度 analyte；若要做 log transform，可先替换为 `NA`。
- 对 ScpTensor 的直接意义：
  - `vendor-normalized` 仍然是线性尺度，不等于“已经 log 过”。
  - 不能把 vendor 输出的 `PG.MaxLFQ` 或 `Precursor.Normalised` 默认为 raw。

### 3.2 Spectronaut 官方手册 / 手册入口（访问于 2026-03-12）

- 资源类型：`模块规范 / 软件文档`
- 链接：https://biognosys.com/resources/spectronaut-manual/
- 官方手册说明的核心点：
  - Spectronaut 提供 `Automatic / Global / Local` 的 cross-run normalization 策略。
  - `PG.Quantity` 一类导出值是否已做跨 run 归一化，依赖导出配置与手册定义。
- 对 ScpTensor 的直接意义：
  - Spectronaut 导出的“Quantity”类字段不能脱离导出参数解释。
  - importer 需要把 `source_software` 与 `source_column` 一并记录，否则后续无法判断应否再次归一化。
- 局限：作为软件手册，它给出的是实现语义而非公开 benchmark。

### 3.3 Cox et al., Molecular & Cellular Proteomics, 2014

- 题目：Accurate Proteome-wide Label-free Quantification by Delayed Normalization and Maximal Peptide Ratio Extraction, Termed MaxLFQ
- 链接：https://doi.org/10.1074/mcp.M113.031591
- 主要发现：
  - `MaxLFQ` 的核心是 delayed normalization 和 peptide ratio extraction，而不是“随便把强度相加再取 log”。
  - 蛋白层 quantity 的语义依赖其上游聚合与归一化算法。
- 对当前问题的意义：
  - `PG.MaxLFQ` 这类列名本身就意味着“已发生特定归一化/聚合逻辑”。
  - 在 ScpTensor 中，对此类输入再次做不加区分的全局归一化，需要文档上明确其前提和风险。

### 3.4 Huffman et al., Nature Methods, 2023

- 题目：Prioritized mass spectrometry increases the depth, sensitivity and data completeness of single-cell proteomics
- 链接：https://doi.org/10.1038/s41592-023-01830-1
- 主要发现：
  - 其公开处理流程明确区分了 `unimputed`、`imputed`、`batch-corrected`、`re-normalized` 等多层矩阵。
  - 蛋白层汇总前后保留不同数据层，避免把不同尺度/状态混成一个默认矩阵。
- 对当前问题的意义：
  - 单细胞蛋白组实践中，“一层矩阵走完全流程”并不是最佳文档策略。
  - ScpTensor 的 layer contract 应显式保留 `source layer -> transformed layer` 的 provenance。

### 3.5 Ctortecka et al., Nature Communications, 2024

- 题目：Automated single-cell proteomics providing sufficient proteome depth to study complex biology beyond cell type classifications
- 链接：https://doi.org/10.1038/s41467-024-49651-w
- 主要发现：
  - 文中 DIA 数据由 DIA-NN 处理，使用 1% FDR 与 retention-time-dependent cross-run normalization。
  - downstream 会在蛋白层进一步做归一化和 `log10` 变换，并以 log 尺度报告 fold change。
- 对当前问题的意义：
  - 真实工作流中常见的是“upstream vendor normalization + downstream task-specific normalization + log transform”的链式处理。
  - 因此 ScpTensor 必须记录“归一化是否来自 upstream”。

### 3.6 Wang et al., Nature Communications, 2025

- 题目：Benchmarking informatics workflows for data-independent acquisition single-cell proteomics
- 链接：https://doi.org/10.1038/s41467-025-65174-4
- 主要发现：
  - benchmark 流程中，normalization 与 `log2(x + 1)` 紧密耦合。
  - 文中明确指出是否归一化取决于生物学假设，例如“总蛋白量是否可视为大致恒定”。
- 对当前问题的意义：
  - `log transform` 不是纯机械步骤，它依赖前面 normalization 的统计假设和零值处理策略。
  - 不存在对所有数据集都最优的单一顺序或单一尺度。

### 3.7 Brombacher et al., Proteomics, 2020

- 题目：Tail-Robust Quantile Normalization
- 链接：https://pubmed.ncbi.nlm.nih.gov/32865322/
- 主要发现：
  - 在低输入、高缺失蛋白组里，TRQN 比普通 quantile normalization 更稳健。
  - 这类 quantile-family 方法隐含比较强的分布对齐假设。
- 对当前问题的意义：
  - `norm_quantile`、`norm_trqn` 这类方法不是“任何线性强度矩阵都可直接比较”的默认选项。
  - 在工程实现上，要求其只对显式 logged layer 自动比较是合理的保守门禁。

### 3.8 Poulos et al., Nature Communications, 2020

- 题目：Strategies to enable large-scale proteomics for reproducible research
- 链接：https://doi.org/10.1038/s41467-020-17641-3
- 主要发现：
  - 论文中的 `ProNorM` pipeline 通过 missing-aware normalization 与 technical replacement 改善大规模 LFQ / DIA 数据的稳定性与 downstream 功能分析准确性。
  - 归一化策略效果受缺失结构和样本组成影响。
- 对当前问题的意义：
  - 不同 normalization family 对输入尺度和缺失结构都有隐含约束。
  - 这支持在 ScpTensor 中把 `scale` 与 `normalization provenance` 写成 layer 元信息，而不是只靠 layer 名字猜测。

### 3.9 二次核查补充（资源分型、稳定入口与场景边界）

- `DIA-NN 官方文档` 与 `Spectronaut 官方手册`（accessed: `2026-03-12`）在本综述中属于 `模块规范 / 软件文档`，负责定义 vendor-normalized linear layer 与 cross-run normalization 语义，不应被视为 benchmark 结论本身。
- `MaxLFQ 2014`、`Huffman 2023`、`Ctortecka 2024`、`Wang 2025`、`Brombacher 2020`、`Poulos 2020` 统一属于 `论文证据`；它们约束的是导入 `raw` 层的线性语义、vendor-normalized provenance 和后续 `log` 层边界。
- 本综述当前不单列 `数据入口` 或 `资源包`，因为其任务是 `scale/provenance contract review`；需要公开数据来源时，应回到公共 benchmark 数据综述。
- 统一边界应保持为：vendor-normalized 是线性输入 provenance，`log` 是变换后 layer 语义；二者不能因为都“不是未经处理 raw”就被混成同一层。

## 4. 横向比较与证据分级

### 4.1 对 ScpTensor 最关键的三层尺度

| 文档默认 layer / 语义类 | 含义 | 典型来源 | 是否适合 `log_transform()` | 是否适合 quantile/TRQN family |
|---|---|---|---|---|
| `raw`（未做 vendor normalization） | 线性、导入后主 quantitative layer；未见 upstream normalization 证据 | `Precursor.Quantity`、`Ms1.Area`、部分 Spectronaut quantity 列 | 是，但需先确认未进入 `log` 层 | 否 |
| `raw` + `is_vendor_normalized=true` | 线性、导入后主 quantitative layer；来源列已做上游 cross-run normalization | `Precursor.Normalised`、`Ms1.Normalised`、`PG.MaxLFQ`、Spectronaut normalized quantity | 是，但仅一次 | 否，除非先生成显式 `log` layer |
| `log` | 明确 `log2` / `log10` / `ln` 后的层 | `log_transform()` 生成层或外部已说明的 log 层 | 否 | 是 |

### 4.2 一致结论（facts）

1. DIA 软件导出的 normalized quantity 通常仍是线性尺度，而不是默认已取对数。
2. vendor 输出列名本身已经携带方法学语义，例如 `MaxLFQ`、`Normalised`、`NormalizedPeakArea`，不能当作“普通 raw intensity”处理。
3. `0` 的处理与 log transform 强耦合；至少在 DIA-NN 官方文档中，`0` 更接近低浓度指示，不应与一般观测值等同。
4. 是否做 normalization、做哪一类 normalization，与样本组成假设密切相关；因此 `raw -> log -> norm` 的处理路径必须可追溯，而 vendor normalization 应作为 provenance 单独记录。
5. quantile-family 方法对输入分布假设更强，作为 AutoSelect 默认候选时应设更严格的尺度门禁。

### 4.3 分歧与解释（inference）

- [推断] Spectronaut 的某些导出列是否已经跨 run 归一化，用户往往只能从导出参数与手册解释判断；因此 importer 若只返回数值矩阵，不附带来源元数据，后续极易发生 double-normalization。
- [推断] 对 ScpTensor 这类强调 stable preprocessing contract 的包，最重要的不是“自动猜出任何来源矩阵的真实尺度”，而是要求调用链显式写明 `source_column / is_normalized / scale / pseudocount`。

### 4.4 证据强度

- 高：DIA-NN 官方文档、MaxLFQ 原始论文、Wang 2025
- 中高：Spectronaut 官方手册、Ctortecka 2024、Brombacher 2020
- 中：Specht 2023、Poulos 2020

## 5. 面向 ScpTensor 的实践建议

### 5.1 明确 `raw`、`log`、`norm` 的最小合同

- `raw`
  - 含义：仓库文档中的 canonical 导入后主 quantitative layer，默认仍是线性尺度。
  - 不承诺“绝对未归一化”，因为 vendor 输入可能本身已做上游 normalization。
  - 必须记录 `source_software`、`source_column`、`data_level`、`is_vendor_normalized`。
- `log`
  - 含义：显式 `log2` / `log10` / `ln` 后的层。
  - 必须记录 `scale` 与 `pseudocount`。
- `norm`
  - 含义：ScpTensor-owned normalization 后的 canonical quantitative layer。
  - 必须记录 `normalized_by=scptensor` 与 `normalization_method`。

### 5.2 `log_transform()` 的当前设计是合理保守的

- 当前实现已经通过 layer 名称和 provenance 历史检测“是否已 log”，默认避免 double-log。
- 这与文献和官方文档一致：
  - vendor-normalized 不等于 `log`
  - `log` layer 不应再次 log
- 建议文档里明确：
  - `detect_logged_by_distribution=False` 应继续作为默认
  - 尺度判断优先依赖 provenance，而不是仅靠数值分布猜测

### 5.3 `NormalizationEvaluator` 的 scale gate 应保留

- 当前实现只在 source layer 有显式 `log` provenance 时才自动纳入 `norm_quantile` 和 `norm_trqn`。
- 这一门禁与 TRQN / quantile-family 的输入假设一致，属于有文献支撑的工程保守策略。
- 建议把这一点写入 AutoSelect 文档，而不是只体现在代码里。

### 5.4 Importer 应把 vendor-normalized 识别提升为显式元信息

- 当前 `scptensor.io` importer 层已经用 `is_vendor_normalized_column()` 识别 `normalized/normalised` 列名。
- 下一步应在文档中明确：
  - `PG.MaxLFQ`、`PG.Normalised`、`Precursor.Normalised`、`F.NormalizedPeakArea` 一类字段导入后不应再被叫做“未经说明的 raw”
  - 若用户需要“真 raw”，应显式选择 `Quantity` / `Area` 类未归一化列

### 5.5 零值与 pseudo-count 要有稳定默认

- 文献和官方文档都说明 `0` 与 `NA` 语义不同。
- 工程上更稳妥的默认是：
  - 若 `0` 代表低浓度占位且需参与 missingness 语义，则先做状态映射
  - 若进入 `log_transform()`，必须显式说明 `offset`
- 对用户文档，建议默认示例写成：
  - `source_layer='raw'`
  - `base=2`
  - `offset=1`
  - 同时说明这只适用于 non-negative linear scale 层

## 6. 风险边界

1. 软件导出列的命名在不同版本中可能有小差异，因此 contract 不应仅依赖列名字符串，还应保留原始来源元数据。
2. vendor-normalized 并不保证适合所有 downstream 分析；Ctortecka 2024 和 Wang 2025 都说明 downstream task-specific normalization 仍可能发生。
3. 若在 batch correction 后得到负值或中心化值，不能再对同一层直接使用普通 `log_transform()`。
4. 对 Spectronaut 输入，如果用户只提供最终 pivot matrix 而不附带导出设置，部分尺度判断只能保守处理。

## 7. 对后续文档/实现的优先建议

1. 在 `io_diann_spectronaut.md` 中增加“vendor-normalized vs raw”对照表。
2. 在 `docs/README.md`、教程和 benchmark 文档中统一使用 `raw / log / norm` 作为默认 layer 名；若需表达 vendor-normalized 输入，统一写成 `raw` + provenance 字段，而不是 `raw_vendor / norm_vendor / log2` 这类第二套默认命名。
3. 在 AutoSelect 报告中把 `source_layer_logged`、`comparison_scale` 和 `input_scale_requirement` 直接展示给用户。

## 8. Shared Citation Registry Coverage

以下共享高频条目的规范元数据以 `docs/references/citations.json` 为准：

- `diann_docs`
- `spectronaut_manual`
- `cox2014_mcp_maxlfq`
- `huffman2023_natmethods_prioritized_ms`
- `ctortecka2024_natcom_automated_scp`
- `wang2025_natcom_dia_scp_benchmark`
- `brombacher2020_proteomics_trqn`
- `poulos2020_natcom_pronorm`
