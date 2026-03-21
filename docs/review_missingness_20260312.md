# DIA 驱动单细胞蛋白组预处理中的缺失值语义与检测状态：优先级文献综述（截至 2026-03-12）

## 1. 研究范围

- 研究问题：在 `DIA-NN / Spectronaut` 驱动的单细胞蛋白组预处理中，`missingness` 应如何与 `detection state`、`MBR/feature matching`、`LOD-like low abundance`、`imputation` 区分？
- 目标输出：为 ScpTensor 当前 `protein-level` 主线中的缺失值语义提供一手证据支撑，重点映射：
  - `scptensor.core.structures.MaskCode`
  - `scptensor.io.load_quant_table` / `load_diann` / `load_spectronaut`
  - `scptensor.impute`
  - `scptensor.qc.filter_features_by_missingness`
- 核心约束：遵循项目合同，除 `scptensor.aggregation` 外，下游 stable 模块默认以 `protein-level matrix` 为主，不把 peptide/run 级语义直接混成 protein 层默认规则。
- 检索日期：`2026-03-12`

### 1.1 关键词优先级（P1 / P2 / P3）

| 优先级 | 目的 | 关键词 |
|---|---|---|
| P1 | 方法核心 | missingness, missing value mechanism, detection limit, left-censoring, match-between-runs, imputation |
| P2 | 模态与软件 | DIA proteomics, DIA-NN, Spectronaut, single-cell proteomics |
| P3 | 应用边界 | preprocessing, QC, data completeness, low-input proteomics |

## 2. 检索策略与筛选标准

### 2.1 Query 梯度（P1 -> P2 -> P3）

1. 先检索缺失值机制与检测语义的一手来源。
2. 再限定到 DIA 蛋白组与 DIA-scProteomics。
3. 最后补充软件文档与 benchmark 论文，确认 `MBR`、`zero quantity`、`data completeness` 在现代 DIA 流程中的实际含义。

### 2.2 纳入标准

- 一手来源：DOI 页面、期刊官网、PubMed / PMC、官方软件文档。
- 直接涉及以下至少一项：
  - 缺失值机制区分（`MCAR / MAR / MNAR` 或 left-censoring）
  - DIA 中“未报告/未检出”与“真实不存在”的语义区分
  - `MBR`、feature matching、data completeness 与 downstream imputation 的关系
  - 可迁移到 ScpTensor `MaskCode` 或 I/O 映射的工程含义

### 2.3 排除标准

- 非一手博客、论坛、二次转载。
- 只给出软件操作说明、但没有方法学或输出语义描述的材料。
- 与 DIA 或低输入/单细胞定量无关、无法支撑状态语义定义的研究。

### 2.4 本轮纳入

- 初筛候选：`20+`
- 深读纳入：`7`
- 其中 DIA / DIA-sc 直接证据：`5`
- 缺失值与 missing-aware 处理基础证据：`2`

## 3. 逐篇证据摘要（Per-paper Summaries）

说明：本节统一沿用全仓库资源分型。除特别标注外，单篇文献条目默认记为 `论文证据`；官方软件/手册页记为 `模块规范 / 软件文档`；具体 accession 或 dataset page 记为 `数据入口`；可脚本化分发包记为 `资源包`。
共享高频条目的规范元数据统一以 `docs/references/citations.json` 为准；若本文件历史写法与 registry 在作者简称、发布日期、期刊、DOI 或 canonical URL 上不一致，以 registry 为准，本文件仅保留 missingness 语义解释。

### 3.1 HarmonizR, Nature Communications, 2022

- 题目：HarmonizR enables data harmonization across independent proteomic datasets with appropriate handling of missing values
- 链接：https://doi.org/10.1038/s41467-022-31007-x
- 主要发现：
  - 跨数据集 harmonization 若忽略 missingness 结构，会显著损伤整合后的可解释性。
  - 缺失值不仅是噪声，也携带平台、批次与数据集间可迁移性的信息。
- 对当前问题的直接意义：
  - missingness 统计本身应被视为状态信息，而不是仅在插补前被“抹平”的空洞。
  - 对 ScpTensor 的 batch-aware / integration-aware 预处理，missingness 语义需要在插补之前就被保留下来。
- 局限：研究对象不是 DIA 单细胞，但对“missing-aware processing”这条原则有直接支撑。

### 3.2 Lazar et al., Journal of Proteome Research, 2016

- 题目：Accounting for the Multiple Natures of Missing Values in Label-Free Quantitative Proteomics Data Sets to Compare Imputation Strategies
- 链接：https://pubmed.ncbi.nlm.nih.gov/26906401/
- 主要发现：
  - 蛋白组缺失值具有多重性质，不能一律按同一机制处理。
  - abundance-dependent missingness 支持把一部分缺失解释为 `MNAR / left-censored`，而不是随机缺失。
- 对 ScpTensor 的意义：
  - `missing_rate` 指标本身不够，需要配合状态语义区分“低丰度未检出”和“后续过滤移除”。
  - 插补默认不应无差别覆盖所有缺失状态。
- 局限：bulk LFQ，不是 DIA 单细胞专用；但仍是缺失机制最常被引用的基础文献之一。

### 3.3 Demichev et al., Nature Methods, 2020

- 题目：DIA-NN: neural networks and interference correction enable deep proteome coverage in high throughput
- 链接：https://doi.org/10.1038/s41592-019-0638-x
- 主要价值：
  - 奠定 DIA-NN 在高通量 DIA 中减少缺失、提高定量鲁棒性的算法基础。
  - 为后续把 `MBR`、cross-run inference 与直接观测分开建模提供现代 DIA 软件背景。
- 对当前问题的意义：
  - DIA 缺失率本身会被采集方式、干扰校正与 cross-run matching 策略显著改变。
  - 因此“是否有值”不能脱离具体软件输出语义解释。
- 局限：不是专门讨论 `missingness semantics` 的论文，更多是算法和性能来源。

### 3.4 DIA-NN 官方文档（访问于 2026-03-12）

- 资源类型：`模块规范 / 软件文档`
- 链接：https://vdemichev.github.io/DiaNN/
- 文档直接信息：
  - 若某 precursor 或 protein 未被 DIA-NN 报告，并不意味着它必然不存在于样本中，更合理的解释通常是“相对低丰度”。
  - DIA-NN 的 `MBR` 会通过第二遍分析提高 identification numbers、data completeness 与 quantification accuracy。
  - DIA-NN 允许 quantity 等于 `0`，官方建议其最好解释为“低浓度 analyte”；若要做 log 变换，可将这些 `0` 换成 `NA`。
- 对 ScpTensor 的直接意义：
  - `unreported`、`MBR-transferred`、`zero quantity`、`imputed` 必须拆开。
  - 不能把“非 NA”一概当成“真实直接检测到的高可信观测值”。
- 局限：软件文档不是独立对照实验；但它是当前 DIA-NN 输出语义的最直接来源。

### 3.5 Wang et al., Nature Communications, 2025

- 题目：Benchmarking informatics workflows for data-independent acquisition single-cell proteomics
- 链接：https://doi.org/10.1038/s41467-025-65174-4
- 主要发现：
  - `sparsity reduction` 是 DIA-SCP 全流程 benchmark 中的重要前置步骤，不宜默认把全部缺失都留给 imputation。
  - 在其同质设计里，`75% data completeness` 是 coverage 与稀疏性负担之间的实用折中。
  - normalization、imputation、batch correction 的最优选择高度依赖任务定义与数据结构。
- 对当前问题的意义：
  - data completeness 既是 QC 指标，也是缺失值语义的任务边界。
  - 某个值“缺失”是否应该被填补，取决于该缺失属于何种上游状态，而不仅仅取决于矩阵空洞比例。
- 局限：结论直接来自其 benchmark 设计；不能机械外推到所有异质数据集。

### 3.6 Yu et al., Molecular & Cellular Proteomics, 2024

- 题目：Quantification Quality Control Emerges as a Crucial Factor to Enhance Single-Cell Proteomics Data Analysis
- 链接：https://pmc.ncbi.nlm.nih.gov/articles/PMC11103571/
- 主要发现：
  - quantification QC 会显著影响单细胞蛋白组的可分析性与 downstream 结果。
  - “是否可信”比“是否非空”更关键。
- 对当前问题的意义：
  - 在 ScpTensor 中，`FILTERED`、`OUTLIER`、`UNCERTAIN` 这类状态不应与 `LOD-like missing` 混在一起统计。
  - 对 feature missingness 的汇总最好支持 `state-aware summary`，而不是只统计 `NaN`。
- 局限：研究重心在 quantification QC，不是专门的缺失机制分型论文。

### 3.7 Derks et al., Nature Biotechnology, 2023

- 题目：Increasing the throughput of sensitive proteomics by plexDIA
- 链接：https://doi.org/10.1038/s41587-022-01389-w
- 主要发现：
  - 该研究强调单细胞 DIA 路线可以获得非常高的数据完整度，并把高 completeness 视为关键优势。
  - 在低输入场景下，缺失程度与灵敏度、采集策略和 multiplex 设计密切相关。
- 对当前问题的意义：
  - 单细胞 DIA 里，missingness 的上游来源常常是“灵敏度/采集/匹配策略不足”，而不是生物学意义上的真零。
  - 这进一步支持将 `left-censored/low-abundance` 与 `structural absence` 区分。
- 局限：重点在 plexDIA 实验策略，不是专门定义状态码的论文。

### 3.8 二次核查补充（资源分型、稳定入口与场景边界）

- `DIA-NN 官方文档`（accessed: `2026-03-12`）在本综述中属于 `模块规范 / 软件文档`，负责约束 `未报告 / MBR / zero quantity` 的软件语义；它不是 benchmark 论文，也不是数据入口：<https://vdemichev.github.io/DiaNN/>
- `HarmonizR 2022`、`Lazar 2016`、`Demichev 2020`、`Wang 2025`、`Yu 2024`、`Derks 2023` 在本综述中统一属于 `论文证据`；它们共同支持 missingness 必须先分语义，再谈 filtering / imputation / integration。
- 本综述不直接提供 `数据入口` 或 `资源包`；它在仓库中的作用是 `state semantics evidence`，应与公共 benchmark 数据、模块规范和资源包类文档分层引用。
- 对后续文档引用，最关键的稳定边界是：`MBR`、vendor `0`、`FILTERED`、`IMPUTED` 都是状态语义，不应被降格成“只是矩阵里的空值或非空值”。

## 4. 横向比较与证据分级

### 4.1 一致结论（facts）

1. 缺失值不是单一语义，至少应区分 `random-like missing`、`abundance-dependent missing`、`filter-induced missing` 与 `post hoc imputed`。
2. 在 DIA 语境下，“未报告”更接近“未通过当前识别/定量流程阈值”或“低丰度不可稳定量化”，不能直接解释为 analyte 真正不存在。
3. `MBR` 或 feature matching 产生的补全值，与在当前 run 中直接观测并定量到的值，不应视为同一观测状态。
4. 引擎输出的 `0` 与 `NA` 不是同一语义；至少在 DIA-NN 文档中，`0` 更接近低浓度指示，而非一般意义上的空缺。
5. 插补策略的适用边界取决于缺失状态；将所有空值无差别插补，容易把 QC 删除、低丰度左删失、匹配转移等不同机制混为一谈。

### 4.2 分歧与解释（inference）

- [推断] 对 Spectronaut 与 DIA-NN 的统一 importer，不宜承诺“每个缺失都能被上游软件唯一判别”；更稳妥的做法是提供保守映射，并把无法明确判断的情况放入 `UNCERTAIN`。
- [推断] 对 protein-level stable 工作流，最有价值的不是增加更多细碎状态码，而是先把当前 `MaskCode` 的边界定义清楚，使统计、过滤、插补都遵守同一套状态语义。

### 4.3 与当前 `MaskCode` 的对照建议

| 当前状态 | 建议语义 | 是否应计入“分析性缺失率”默认统计 |
|---|---|---|
| `VALID` | 当前 run 中可用的真实定量值 | 否 |
| `MBR` | 由 run 间匹配/转移得到的值，非同义于直接观测 | 单独统计，不并入 `VALID` |
| `LOD` | 低丰度/检测极限驱动的 left-censored 缺失，含需要在 log 前从 `0` 转为 `NA` 的情形 | 是 |
| `FILTERED` | 由 FDR/QC/规则过滤造成的失效值 | 默认单独统计 |
| `OUTLIER` | 统计离群后剔除，不代表上游未检出 | 默认单独统计 |
| `IMPUTED` | 后验填补值，不应再视作原始观测 | 否，但需单独汇总 |
| `UNCERTAIN` | 上游导出无法明确判定来源的模糊状态 | 单独统计 |

### 4.4 证据强度

- 高：Lazar 2016、Wang 2025、DIA-NN 官方文档、HarmonizR 2022
- 中高：Yu 2024、Demichev 2020
- 中：Derks 2023

## 5. 面向 ScpTensor 的实践建议

### 5.1 `MaskCode` 文档语义应先收紧，而不是先扩码

- 当前 `MaskCode` 已覆盖 `VALID / MBR / LOD / FILTERED / OUTLIER / IMPUTED / UNCERTAIN`，在 stable 层面已足够表达大部分 DIA-sc 预处理状态。
- 更优先的工作是把每个状态的来源、允许下游操作、统计口径写入文档，而不是立即新增更多枚举值。

### 5.2 I/O 层应以“保守可解释映射”为默认

- `load_diann` / `load_spectronaut` 导入时，不应简单用“非空 = VALID、空 = LOD”。
- 更合理的默认规则：
  - vendor 明确给出可量化值：`VALID`
  - run 间匹配/转移值：`MBR`
  - 明确的低丰度/未稳定量化：`LOD`
  - 明确经 FDR / QC / downstream rule 排除：`FILTERED`
  - 无法从导出表唯一判断来源：`UNCERTAIN`
- 若导入源只有最终矩阵、没有足够 provenance，宁可保守标记，也不要过度宣称“直接检测到”。
- `2026-03-16` 的当前实现快照仍需单独说明：
  - matrix/pivot importer 现在仍以 `finite -> VALID`、`non-finite -> LOD` 作为保守基线
  - 这条规则只表示“工程上先完成导入”，不表示状态语义已经被完整恢复
  - 因此 downstream 文档与报表不应把 matrix-imported `LOD` 自动解释为已经证实的 left-censored low-abundance signal

### 5.3 `filter_features_by_missingness` 需转向 state-aware 统计

- 文档应明确：
  - 默认 missingness 统计至少区分 `LOD-like missing` 与 `FILTERED / OUTLIER / UNCERTAIN`
  - `MBR` 不应无条件并入 `VALID`
- 对 benchmark 报表，建议同时输出：
  - `observed_rate`
  - `mbr_rate`
  - `lod_rate`
  - `filtered_rate`
  - `imputed_rate`

### 5.4 插补默认只针对 `LOD-like missing` 最稳妥

- 基于缺失机制文献，最安全的默认是：
  - `LOD` 可作为默认插补目标
  - `FILTERED`、`OUTLIER`、`UNCERTAIN` 不默认插补
  - `MBR` 不再二次插补，除非基准实验明确需要
- 这样更符合当前项目“先得到完整 protein-level matrix，再进入后续 preprocessing”的工程边界。

### 5.5 `0.0` 不应被 silently 视为普通观测

- 结合 DIA-NN 官方文档，`0.0` 更接近低浓度占位值，而不是“确定的真实零表达”。
- 因此在需要 log 的工作流中，建议把此类值转入 `LOD` 语义或显式保留其来源，再决定是否 `offset` 或转 `NA`。

## 6. 风险边界

1. 直接针对 `DIA single-cell proteomics missingness semantics` 的专门论文仍然较少，很多结论来自缺失机制文献、软件文档与 benchmark 联合推断。
2. `MBR` 在不同软件中的实现并不完全同义，文档中必须注明该状态代表“run 间转移/匹配得到的值”，而不是更窄的某一软件内部算法细节。
3. 若上游只提供最终 protein matrix，ScpTensor 无法完全恢复 precursor/run 级生成机制，此时状态映射必须保守。
4. 若未来要把 vendor-engine `0` 单独设码，应先确认 DIA-NN 与 Spectronaut 导出层都能稳定提供可判别证据，否则会引入伪精确。

## 7. 对后续文档/实现的优先建议

1. 在 `io_diann_spectronaut.md` 中新增“缺失状态映射表”。
2. 在 `scptensor.impute` 文档中明确“默认插补只面向 `LOD`”。
3. 在 `benchmark/imputation` 与 `benchmark/autoselect` 中加入 state-aware completeness 指标，而不是只看总体空值比例。

## 8. Shared Citation Registry Coverage

以下共享高频条目的规范元数据以 `docs/references/citations.json` 为准：

- `schlumbohm2022_natcom_harmonizr`
- `lazar2016_jpr_missing_values`
- `demichev2020_natmethods_diann`
- `diann_docs`
- `wang2025_natcom_dia_scp_benchmark`
- `yu2024_mcp_quant_qc`
- `derks2023_natbiotechnol_plexdia`
