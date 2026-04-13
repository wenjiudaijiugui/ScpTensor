# DIA 驱动单细胞蛋白组预处理中的 QC 与过滤策略：优先级文献综述（截至 2026-03-13）

## 1. 研究范围

- 研究问题：在 DIA 驱动单细胞蛋白组与强相关定量蛋白组场景中，`sample QC` 与 `feature QC/filtering` 应优先参考哪些一手证据，才能既保留真实生物学异质性，又不过度放大技术噪声、批次漂移和背景污染？
- 目标输出：为 `scptensor.qc` 当前稳定接口提供文献支撑，重点覆盖：
  - `calculate_sample_qc_metrics`
  - `filter_low_quality_samples`
  - `filter_doublets_mad`
  - `assess_batch_effects`
  - `calculate_feature_qc_metrics`
  - `filter_features_by_missingness`
  - `filter_features_by_cv`
- 关注维度：
  - 样本层：`n_features`、`total_intensity`、control-aware filtering、样本级 outlier、batch-aware QC
  - 特征层：`detection_rate` / `missing_rate`、feature consistency、`CV`
- 检索日期：`2026-03-13`

### 1.1 关键词优先级（P1 / P2 / P3）

| 优先级 | 目的 | 关键词 |
|---|---|---|
| P1 | 方法核心 | quality control, filtering, sample QC, control-aware filtering, coefficient of variation, outlier detection, data completeness |
| P2 | 模态与数据 | single-cell proteomics, DIA proteomics, quantitative proteomics, mass spectrometry |
| P3 | 应用边界 | preprocessing, benchmark, reproducibility, batch-aware QC |

## 2. 检索策略与筛选标准

### 2.1 Query 梯度（P1 -> P2 -> P3）

1. 先检索 single-cell proteomics 与 quantitative proteomics 中的 QC/filtering 一手来源。
2. 再限定到 DIA / single-cell proteomics 或强可迁移 MS-proteomics QC 框架。
3. 最后补充与 batch-aware QC、controls 和 feature-consistency 直接相关的 benchmark/reporting 证据。

### 2.2 纳入标准

- 一手来源：期刊官网、DOI 页面、PubMed / PMC、官方 benchmark 或方法论文。
- 直接涉及以下至少一项：
  - DIA-SCP 或 SCP 的 sample-level QC/filtering
  - empty/negative/0-cell controls 或 control-aware filtering
  - load/intensity、batch-aware QC、样本级 outlier 识别
  - feature missingness、feature consistency 或 CV 与 protein-level reliability 的关系

### 2.3 排除标准

- 非一手博客、论坛和教程。
- 只讨论鉴定数提升、没有 QC/filtering 解释的文章。
- 没有实证或方法边界说明的单纯软件说明页。

### 2.4 本轮纳入

- 初筛候选：`40+`
- 深读纳入：`11` 篇
- 其中 SCP/DIA 直接或近直接证据：`7`
- 强可迁移 MS-proteomics QC 框架：`2`
- feature-level 质量证据：`2`

## 3. 逐篇证据摘要（Per-paper Summaries）

说明：本节统一沿用全仓库资源分型。除特别标注外，单篇文献条目默认记为 `论文证据`。
共享高频条目的规范元数据统一以 `docs/references/citations.json` 为准；若本文件历史写法与 registry 在作者简称、发布日期、期刊、DOI 或 canonical URL 上不一致，以 registry 为准，本文件仅保留 QC/filtering 的解释与样本边界。

### 3.1 Wang et al., Nature Communications, 2025

- 资源类型：`论文证据`
- 题目：Benchmarking informatics workflows for data-independent acquisition single-cell proteomics
- 链接：https://www.nature.com/articles/s41467-025-65174-4
- 发布日期：`2025-11-21`
- 与 QC 直接相关的结论：
  - `sparsity reduction` 是关键前置步骤，不应把全部缺失都留给 imputation。
  - 在其同质细胞系设计中，`75% data completeness` 是 coverage 与缺失负担之间的较佳折中。
  - 更严格 completeness gate 能降低潜在错误检出或错误转移负担，但也会损失覆盖。
- 对 ScpTensor 的意义：
  - `missingness/completeness` 应作为 stable sample/feature QC 的核心维度。
  - `75% completeness` 只适合作为特定同质设计下的 evidence-backed 起点，不应机械上升为全场景默认。

### 3.2 Yu et al., Molecular & Cellular Proteomics, 2024

- 资源类型：`论文证据`
- 题目：Quantification Quality Control Emerges as a Crucial Factor to Enhance Single-Cell Proteomics Data Analysis
- 链接：https://pmc.ncbi.nlm.nih.gov/articles/PMC11103571/
- 主要发现：
  - 在 SCP 分析链中加入 quantification QC 后，定量蛋白/肽段数、组间分离和差异分析结果都可改善。
  - 文中比较了不同 valid-value filtering 强度，强调阈值必须跟数据集特征绑定，而不是一刀切。
- 对 ScpTensor 的意义：
  - `filter_low_quality_samples` 和 `filter_features_by_missingness` 应继续被表述为 study-design-dependent，而不是 universal fixed cutoffs。
  - “过滤越多越好”不是正确目标，重点是提高 quantification trustworthiness。

### 3.3 Huffman et al., Nature Methods, 2023

- 资源类型：`论文证据`
- 题目：Prioritized mass spectrometry increases the depth, sensitivity and data completeness of single-cell proteomics
- 链接：https://www.nature.com/articles/s41592-023-01830-1
- 主要发现：
  - 在该单细胞蛋白组实验中，negative-control wells 被用于样本级 QC。
  - 文中使用 protein-level `CV` 阈值区分无细胞/对照液滴与真实单细胞，示例阈值为 `0.4`。
- 对 ScpTensor 的意义：
  - 这为 `control-aware filtering > global fixed threshold` 提供了直接 SCP 证据。
  - 但 `CV = 0.4` 明确是平台/实验依赖阈值，不能外推为 stable 全局默认。

### 3.4 Ctortecka et al., Nature Communications, 2024

- 资源类型：`论文证据`
- 题目：Automated single-cell proteomics providing sufficient proteome depth to study complex biology beyond cell type classifications
- 链接：https://www.nature.com/articles/s41467-024-49651-w
- 主要发现：
  - 样本处理与采集存在跨天/跨批次结构时，需要把 day/batch 影响与 biological effect 分开解释。
  - 文章同时强调 cell-size/intensity 等 sample-level covariates 对下游解释的重要性。
- 对 ScpTensor 的意义：
  - `total_intensity`、sample load proxy 和 batch-stratified sample summaries 应成为 stable QC 文档的一部分。
  - 样本级低信号不一定代表“坏样本”，也可能反映细胞大小、批次或处理流程差异。

### 3.5 Sadiku et al., Nature Communications, 2025/2026 article page

- 资源类型：`论文证据`
- 题目：Single cell proteomic analysis defines discrete neutrophil functional states in human glioblastoma
- 链接：https://www.nature.com/articles/s41467-025-67367-3
- 主要发现：
  - 使用显式 `0-cell` QC runs 作为背景/污染参照。
  - 在该研究中，细胞样本需要满足蛋白鉴定数高于 `0-cell` QC 最大值的 `1.75x`，并据此收束到 `>= 400 proteins` 的 study-specific gate。
  - 文中还明确移除一个明显异常的对照样本。
- 对 ScpTensor 的意义：
  - control-aware sample filtering 在最新 SCP 文献中已有直接一手实例。
  - 这进一步支持：存在 empty wells / negative / 0-cell controls 时，应优先用 controls 定阈值，而不是只靠全局 MAD 或固定 `min_features`。

### 3.6 Gatto et al., Nature Methods, 2023

- 资源类型：`论文证据`
- 题目：Initial recommendations for performing, benchmarking and reporting single-cell proteomics experiments
- 链接：https://www.nature.com/articles/s41592-023-01785-3
- 主要发现：
  - single-cell proteomics 的 benchmark 与 reporting 需要清楚记录 controls、metadata、处理路径和可重复性条件。
  - 不应只报告图形结果而不说明 QC/过滤路径。
- 对 ScpTensor 的意义：
  - `scptensor.qc` 文档应继续要求 batch/control metadata 明确进入 QC 报表。
  - 没有 controls 时可以用稳健工程规则，但必须承认那是 heuristic，而不是 community gold standard。

### 3.7 Vanderaa and Gatto, Genome Biology, 2025

- 资源类型：`论文证据`
- 题目：scplainer: using linear models to understand mass spectrometry-based single-cell proteomics data
- 链接：https://genomebiology.biomedcentral.com/articles/10.1186/s13059-025-03713-4
- 主要发现：
  - 单细胞蛋白组中的技术与生物学变异应被联合建模，而不是先验地全部折叠成“低质量样本”。
  - 样本级 intensity/size proxy 和 batch covariates 都可能进入解释框架。
- 对 ScpTensor 的意义：
  - `assess_batch_effects` 不应只是附属函数，而应成为 sample QC 的稳定解释层。
  - 对 sample-level QC，median intensity / load proxy 更适合作为 covariate summary，而不是单独硬判定“好/坏样本”。

### 3.8 Tsantilas et al., Journal of Proteome Research, 2024

- 资源类型：`论文证据`
- 题目：A Framework for Quality Control in Quantitative Proteomics
- 链接：https://pmc.ncbi.nlm.nih.gov/articles/PMC11973981/
- 主要发现：
  - 定量蛋白组 QC 需要分层：system suitability、internal QC、external pooled QC。
  - QC 不仅用于淘汰样本，也用于区分仪器故障、制备偏差和样本本身差异。
- 对 ScpTensor 的意义：
  - 即使在单细胞场景中，也应把 sample QC 与 run/batch/system QC 分层解释。
  - 这支持 `batch / run / plate` 分层汇总，而不是只输出全局样本排名。

### 3.9 Patterson et al., Journal of the American Society for Mass Spectrometry, 2023

- 资源类型：`论文证据`
- 题目：Establishing Quality Control Procedures for Large-Scale Plasma Proteomics Analyses
- 链接：https://pubmed.ncbi.nlm.nih.gov/37163770/
- 主要发现：
  - 大规模定量蛋白组应持续跟踪 pooled QC、IDs、丰度、保留时间、质量误差等指标。
  - 很多异常更像 run-level 或 operator-level 漂移，而不是单一样本低质量。
- 对 ScpTensor 的意义：
  - `calculate_sample_qc_metrics` 输出不应被脱离批次上下文直接解释。
  - `assess_batch_effects` 与 sample QC summary 应在文档层显式联动。

### 3.10 Goeminne et al., Molecular & Cellular Proteomics, 2020

- 资源类型：`论文证据`
- 题目：Selection of Features with Consistent Profiles Improves Relative Protein Quantification in Mass Spectrometry Experiments
- 链接：https://pubmed.ncbi.nlm.nih.gov/32234965/
- 主要发现：
  - 一致性差的 feature 会降低最终 protein quantification 质量。
  - 并非所有 feature 都应平权纳入 downstream quantification。
- 对 ScpTensor 的意义：
  - feature QC 不能只看 missingness，仍需保留 `feature consistency` 视角。
  - `filter_features_by_cv` 的价值更接近“技术稳定性检查”，而不是普适生物学过滤。

### 3.11 Lazar et al., Journal of Proteome Research, 2016

- 资源类型：`论文证据`
- 题目：Accounting for the Multiple Natures of Missing Values in Label-Free Quantitative Proteomics Data Sets to Compare Imputation Strategies
- 链接：https://pubmed.ncbi.nlm.nih.gov/26906401/
- 主要发现：
  - missing values 具有不同机制，不能只用单一 missing rate 解释。
  - abundance-dependent missingness 是 proteomics 中重要基础语义。
- 对 ScpTensor 的意义：
  - `filter_features_by_missingness` 文档应继续与 missingness semantics 一起解释，而不是把任何非空洞都等价看待。
  - feature missingness gate 必须与研究设计和后续 imputation contract 联动。

### 3.12 二次核查补充（资源分型、发布日期与场景边界）

- `Wang 2025` 仍是当前最直接的 DIA-SCP workflow/QC 任务设计证据，但 `75% completeness` 只在特定同质设计下有直接支持，不应升级为全场景默认：<https://www.nature.com/articles/s41467-025-65174-4>
- `Huffman 2023` 与 `Sadiku` 的最新 SCP 文章共同支持 `control-aware filtering`，但它们给出的 `CV` 或 `protein-ID` 阈值都属于 dataset-specific rules，而不是社区统一标准：<https://www.nature.com/articles/s41592-023-01830-1>；<https://www.nature.com/articles/s41467-025-67367-3>
- `Ctortecka 2024`、`scplainer 2025`、`Tsantilas 2024` 和 `Patterson 2023` 共同支持 batch-/run-aware QC：很多异常应先解释为 system/batch drift，而不是立刻解释为单一样本低质量：<https://www.nature.com/articles/s41467-024-49651-w>；<https://genomebiology.biomedcentral.com/articles/10.1186/s13059-025-03713-4>；<https://pmc.ncbi.nlm.nih.gov/articles/PMC11973981/>；<https://pubmed.ncbi.nlm.nih.gov/37163770/>
- `Goeminne 2020` 与 `Lazar 2016` 继续约束 feature-level QC：全局 missingness 或 global CV 不能代替 feature consistency 和缺失机制解释：<https://pubmed.ncbi.nlm.nih.gov/32234965/>；<https://pubmed.ncbi.nlm.nih.gov/26906401/>

## 4. 横向比较与证据分级

### 4.1 一致结论（facts）

1. 当前没有跨 DIA-SCP 研究统一认可的 sample-level universal threshold。
2. 存在 empty wells、negative controls 或 `0-cell` runs 时，`control-aware filtering` 的证据强于纯全局固定阈值。
3. `n_features`、`total_intensity`、median intensity/load proxy 需要结合 `batch / run / plate` 分层解释，不能脱离上下文直接当成坏样本判据。
4. feature QC 不能只看 missingness；feature consistency 与技术稳定性仍然重要。
5. 目前缺少 single-cell proteomics 社区统一认可的 doublet detector；`MAD` 一类方法更接近稳健工程 heuristic。

### 4.2 分歧与解释（inference）

- [推断] `filter_low_quality_samples()` 更合理的 stable 顺序应是：`control-aware threshold > batch-aware robust rule > hard fallback threshold`。
- [推断] `filter_doublets_mad()` 可以保留为无 controls 时的工程后备方案，但不应在文档中被写成文献支持充分的 single-cell proteomics doublet standard。
- [推断] 对高异质组织样本，`global CV filtering` 的误删风险高于同质 cell line、技术重复或 pooled QC 场景。

### 4.3 证据强度

- 高：Wang 2025、Yu 2024、Huffman 2023、Gatto 2023
- 中高：Ctortecka 2024、Sadiku article page、scplainer 2025、Tsantilas 2024、Patterson 2023
- 中：Goeminne 2020、Lazar 2016

## 5. 面向 ScpTensor 的实践建议（映射当前模块）

### 5.1 `calculate_sample_qc_metrics`

- 继续把以下指标作为 stable 输出：
  - `n_features`
  - `total_intensity`
  - `detection/completeness`
  - `batch / run / plate` stratified summaries
- 建议文档补充：
  - `n_features` 与 `total_intensity` 是 sample-level covariates，不是单独 universal pass/fail rules。
  - 若有 controls，最好同时输出 `sample vs control` 对照摘要。

### 5.2 `filter_low_quality_samples`

- 建议明确三类路径：
  1. `control-aware threshold`
  2. `robust distribution-aware threshold`（如 MAD lower-tail）
  3. `hard threshold fallback`
- 建议文档顺序也按这三层写，而不是先给单一固定 `min_features`。

### 5.3 `filter_doublets_mad`

- 建议保留，但文档应明确其定位是：
  - `heuristic outlier / doublet-like filter`
  - `no-controls fallback`
- 不应写成：
  - 社区统一 doublet detector
  - 已在 DIA-SCP benchmark 中定型的方法

### 5.4 `assess_batch_effects`

- 建议提升为 stable QC 文档中的必经解释层。
- 样本级 QC 指标至少应支持按 `batch / run / plate` 分层汇总，否则很容易把系统问题误删成“低质量样本”。

### 5.5 `calculate_feature_qc_metrics` / `filter_features_by_missingness`

- 建议继续把 `detection_rate / missing_rate` 作为 stable 主指标，但必须和缺失语义联动解释。
- `max_missing_rate` 建议被表述为 `study-design-dependent completeness gate`。
- 可保留 evidence-backed 参考档位：
  - `0.10`：严格
  - `0.25`：约 75% completeness，Wang 2025 同质设计的常用起点
  - `0.34`：较宽松
  - `0.50+`：探索性或高异质设计

### 5.6 `filter_features_by_cv`

- 建议明确标注为 `context-sensitive`。
- 更适用场景：
  - 技术重复
  - 同质 cell line
  - pooled QC 或 control channel
- 不应默认作为高异质单细胞 atlas 的 global stable gate。

## 6. 风险边界

1. 直接针对 `DIA single-cell proteomics sample QC` 的一手 benchmark 仍然有限，尤其是 `doublet detection` 与 `global CV gate`。
2. `75% completeness`、`CV 0.4`、`>= 400 proteins` 等数字都有明显的 study-specific 条件。
3. 样本级低信号既可能是低质量，也可能是 cell-size、batch、run drift 或实验设计差异。
4. 若没有 controls，只靠单一固定阈值或单一 MAD 规则做清洗，误删真实生物学样本的风险较高。

## 7. 对文档分层的建议

- `evidence-backed stable defaults`
  - detection / completeness summary
  - sample load / intensity summary
  - batch-stratified sample QC summary
  - control-aware thresholding when controls exist
- `heuristic exploratory filters`
  - MAD doublet-like filtering without controls
  - global CV filtering in heterogeneous cohorts
  - aggressive fixed `min_features` without design context

## 8. Shared Citation Registry Coverage

以下共享高频条目的规范元数据以 `docs/references/citations.json` 为准：

- `wang2025_natcom_dia_scp_benchmark`
- `yu2024_mcp_quant_qc`
- `huffman2023_natmethods_prioritized_ms`
- `ctortecka2024_natcom_automated_scp`
- `sadiku2025_natcom_glioblastoma_scp`
- `gatto2023_natmethods_scp_recommendations`
- `vanderaa2025_genomebio_scplainer`
- `tsantilas2024_jpr_qc_framework`
- `patterson2023_jasms_large_scale_qc`
- `goeminne2020_mcp_feature_consistency`
- `lazar2016_jpr_missing_values`
