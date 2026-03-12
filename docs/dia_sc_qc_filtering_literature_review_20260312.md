# DIA 驱动单细胞蛋白组预处理中的 QC 与过滤策略：优先级文献综述（截至 2026-03-12）

## 1. 研究范围

- 研究问题：在 DIA 驱动单细胞蛋白组与强相关定量蛋白组场景中，`sample QC` 与 `feature QC/filtering` 应优先参考哪些一手证据？
- 目标输出：为 `scptensor.qc` 当前稳定接口提供文献支撑，重点覆盖：
  - `calculate_sample_qc_metrics`
  - `filter_low_quality_samples`
  - `filter_doublets_mad`
  - `assess_batch_effects`
  - `calculate_feature_qc_metrics`
  - `filter_features_by_missingness`
  - `filter_features_by_cv`
- 关注指标：
  - 样本层：`n_features`、`total_intensity`、离群点检测、doublet-like filtering、batch-aware QC
  - 特征层：`detection_rate` / `missing_rate`、feature consistency、`CV`
- 检索日期：`2026-03-12`

### 1.1 关键词优先级（P1 / P2 / P3）

| 优先级 | 目的 | 关键词 |
|---|---|---|
| P1 | 方法核心 | quality control, filtering, missingness, coefficient of variation, outlier detection, data completeness |
| P2 | 模态与数据 | single-cell proteomics, DIA proteomics, quantitative proteomics, mass spectrometry |
| P3 | 应用边界 | preprocessing, benchmark, reproducibility, batch-aware QC |

## 2. 检索策略与筛选标准

### 2.1 Query 梯度（P1 -> P2 -> P3）

1. 先检索 QC / filtering / missingness / CV / outlier detection 的一手来源。
2. 再限定到 `single-cell proteomics`、`DIA proteomics`、`quantitative proteomics`。
3. 最后补充 benchmark、reproducibility、batch-aware QC 语境。

### 2.2 纳入标准

- 一手来源：期刊官网、DOI 页面、PubMed / PMC、官方方法文档。
- 直接涉及以下至少一项：
  - 单细胞蛋白组或 DIA 蛋白组的 QC / filtering
  - 与 protein-level quantification 质量直接相关的 feature selection / missingness 处理
  - 可迁移到 ScpTensor stable QC 的批次感知样本过滤与质量监控框架

### 2.3 排除标准

- 非一手博客、论坛和教程。
- 只讨论鉴定数提升，但没有 QC / filtering 解释的文章。
- 只提供软件操作说明、没有方法学或实证对比价值的材料。

### 2.4 本轮纳入

- 初筛候选：`30+`
- 深读纳入：`7` 篇
- 其中 DIA-SCP 直接证据：`2` 篇
- 强相关可迁移证据：`5` 篇

## 3. 逐篇证据摘要（Per-paper Summaries）

说明：本节统一沿用全仓库资源分型。除特别标注外，单篇文献条目默认记为 `论文证据`；官方软件/手册页记为 `模块规范 / 软件文档`；具体 accession 或 dataset page 记为 `数据入口`；可脚本化分发包记为 `资源包`。

### 3.1 Wang et al., Nature Communications, 2025

- 题目：Benchmarking informatics workflows for data-independent acquisition single-cell proteomics
- 链接：https://www.nature.com/articles/s41467-025-65174-4
- 目标：系统 benchmark DIA 单细胞蛋白组全流程，包括 sparsity reduction、imputation、normalization、batch correction 与 DE。
- 与 QC 直接相关的结论：
  - `sparsity reduction` 是关键前置步骤，不应把全部缺失都留给 imputation。
  - 在其同质细胞系设计中，`75% data completeness` 是 coverage 与缺失负担之间的较佳折中。
  - 更严格 completeness gate 能降低错误检出负担，但可能损失覆盖。
- 局限：结论受样本同质性、任务定义和实验设计影响，不能直接机械外推到高异质细胞群。

### 3.2 Yu et al., Molecular & Cellular Proteomics, 2024

- 题目：Quantification Quality Control Emerges as a Crucial Factor to Enhance Single-Cell Proteomics Data Analysis
- 链接：https://pmc.ncbi.nlm.nih.gov/articles/PMC11103571/
- 目标：强调 single-cell proteomics 中 quantification QC 的必要性。
- 主要发现：
  - 在 SCP 分析链中加入 quantification QC 后，定量蛋白/肽段数、组间分离和差异分析结果都可改善。
  - 重点不是“过滤越多越好”，而是先识别哪些 quantification 可被信任。
- 局限：方法链偏其特定实验与分析框架，不提供可直接泛化的单一阈值。

### 3.3 Specht et al., Nature Methods, 2023

- 题目：Prioritized mass spectrometry increases the depth, sensitivity and data completeness of single-cell proteomics
- 链接：https://www.nature.com/articles/s41592-023-01830-1
- 目标：提升单细胞蛋白组的深度、灵敏度和 completeness。
- 与 QC 相关的要点：
  - 文中明确使用 single-cell quality controls。
  - 用 `CV threshold` 区分无细胞/对照液滴与真实单细胞，说明 sample filtering 更适合锚定 controls 或分布，而不是固定全局阈值。
- 局限：主要是 DDA/TMT 路线，不是 DIA 专项；阈值与平台和实验设计强相关。

### 3.4 Patterson et al., Journal of the American Society for Mass Spectrometry, 2023

- 题目：Establishing Quality Control Procedures for Large-Scale Plasma Proteomics Analyses
- 链接：https://pubmed.ncbi.nlm.nih.gov/37163770/
- 目标：建立大规模定量蛋白组的 QC protocol。
- 主要发现：
  - 应利用长期 pooled QC 跟踪 protein / peptide IDs、平均丰度、质量误差、保留时间等指标。
  - 很多异常本质上是批次/系统问题，不应直接解释为单一样本低质量。
- 局限：bulk plasma，不是单细胞，但对 batch-aware QC 原则很重要。

### 3.5 Goeminne et al., Molecular & Cellular Proteomics, 2020

- 题目：Selection of Features with Consistent Profiles Improves Relative Protein Quantification in Mass Spectrometry Experiments
- 链接：https://pubmed.ncbi.nlm.nih.gov/32234965/
- 目标：提高相对定量时 protein summarization 的可靠性。
- 主要发现：
  - 并非所有 feature 都应平权纳入 protein quantification。
  - 一致性差的 feature 会拉低最终 protein-level 定量质量。
- 局限：不是单细胞专用，更接近 summarization 与 QC 边界，但能支持 `feature consistency` 视角。

### 3.6 Lazar et al., Journal of Proteome Research, 2016

- 题目：Accounting for the Multiple Natures of Missing Values in Label-Free Quantitative Proteomics Data Sets to Compare Imputation Strategies
- 链接：https://pubmed.ncbi.nlm.nih.gov/26906401/
- 目标：系统区分蛋白组缺失值机制。
- 主要发现：
  - 缺失值不是单一类型，应区分 `MCAR / MAR / MNAR`。
  - 低丰度特征更容易表现为 abundance-dependent missingness。
- 局限：bulk LFQ，不是单细胞，也不是 DIA 专属；但对 `missingness filtering` 语义是基础来源。

### 3.7 二次核查补充（资源分型、稳定入口与场景边界）

- 本综述当前纳入的 7 条来源全部属于 `论文证据`，没有单独指定 `数据入口`、`模块规范` 或 `资源包`；因此本文件给出的是 QC/filtering 原则证据，而不是公开数据清单。
- `Wang 2025` 是唯一直接落在 DIA-SCP workflow benchmark 语境中的来源，应继续作为 `QC 任务设计证据` 使用，而不是被扩大解释为全场景固定阈值来源：<https://www.nature.com/articles/s41467-025-65174-4>
- `Yu 2024` 与 `Specht 2023` 提供 SCP 直接或近直接证据，但它们支持的是 `distribution-aware / control-aware` filtering 原则，不支持把某一 completeness 或 CV cutoff 写成仓库稳定默认。
- `Patterson 2023`、`Goeminne 2020`、`Lazar 2016` 更适合作为可迁移框架证据；在文档层应持续与 DIA-SCP 直接证据分开陈述，避免把 bulk QC 框架误写成单细胞专用规范。

## 4. 横向比较与证据分级

### 4.1 一致结论（facts）

1. 在 DIA-SCP 场景中，`data completeness / sparsity reduction` 应作为前置 QC/filtering 重点，而不是默认由 imputation 兜底。
2. 样本层 QC 更可靠的思路是 `distribution-aware` 或 `control-aware` filtering，而不是给所有实验套同一固定阈值。
3. feature QC 不能只看 missingness；feature consistency 与 downstream protein quantification reliability 同样重要。
4. QC 需要 batch-aware 视角。很多异常反映的是系统或批次漂移，而不是单一样本劣质。

### 4.2 分歧与解释（inference）

- [推断] `filter_features_by_cv()` 更适用于技术重复、同质 cell line、spike-in 或 batch 内同组样本；在高异质单细胞 atlas 中直接全局按 CV 过滤容易误删真实生物差异。
- [推断] 对 ScpTensor 这类以 `protein-level matrix` 为稳定主线的包，更合理的 stable QC 应以 `detection/missingness + sample intensity/load + batch-stratified summaries` 为主，而不是默认启用激进的 global CV gate。

### 4.3 证据强度

- 高：Wang 2025（DIA-SCP 直接 benchmark）
- 中高：Yu 2024（SCP quantification QC）、Specht 2023（SCP control-aware QC 实践）
- 中：Patterson 2023、Goeminne 2020、Lazar 2016（可迁移框架证据）

## 5. 面向 ScpTensor 的实践建议（映射当前模块）

### 5.1 `calculate_sample_qc_metrics` / `calculate_feature_qc_metrics`

- 继续把 `n_features`、`detection_rate`、`missing_rate` 作为 stable QC 主指标。
- 检测/缺失计算应基于显式缺失语义，而不是 `X > 0` 或“非 NaN 即检出”。
- 建议在文档中明确：真实 `0.0` 与未检出是不同语义。

### 5.2 `filter_low_quality_samples`

- 不建议在 stable API 写死单一 `min_features`。
- 更合理的表述是：
  - `hard threshold` 适合保底清洗
  - `MAD lower-tail` 适合无 controls 的稳健默认
  - `control-aware threshold` 在存在 empty wells / negative controls 时优先

### 5.3 `filter_features_by_missingness`

- 建议把 `max_missing_rate` 文档化为 `study-design-dependent completeness gate`。
- 可提供证据化参考档位：
  - `0.10`：90% completeness，严格
  - `0.25`：75% completeness，Wang 2025 同质数据的实用起点
  - `0.34`：约 66% completeness，较宽松
  - `0.50+`：异质或探索性
- 对异质细胞群，优先支持 `group-aware` 或 `batch-aware` missingness summary，而不是只做全局过滤。

### 5.4 `filter_features_by_cv`

- 建议明确标注为 `context-sensitive`。
- 若没有技术重复、同质群或 control channel，`global CV filtering` 不宜作为默认 stable gate。
- 更合适的解释是：CV 用于同批次、同组、技术重复或 pooled QC 内部稳定性判断。

### 5.5 `filter_doublets_mad`

- 作为工程化默认可保留，但文档应明确它是 `heuristic outlier removal`，而不是已经在 DIA-SCP 社区统一 benchmark 的标准 doublet detector。
- 若实验包含空孔或 negative controls，应优先用 controls 设阈值；`MAD` 作为无 controls 时的后备方案。

### 5.6 `assess_batch_effects`

- 建议提升为 stable QC 文档中的必经步骤，而不是附属函数。
- 所有样本级 QC 指标都应支持按 `batch / run / plate` 分层汇总，否则容易把系统性批次问题误删成“低质量样本”。

## 6. 风险边界

1. 直接针对 `DIA single-cell proteomics QC filtering` 的一手 benchmark 仍然不多，尤其是 `CV cutoff` 和 `doublet detection`。
2. `75% completeness` 只在特定同质数据中有直接支持，不能机械外推到高异质组织样本。
3. `CV` 在单细胞里同时承载技术噪声和真实生物异质性；若不分组直接过滤，风险很高。
4. `MAD-based doublet/outlier` 更像稳健工程启发式，而不是社区标准答案。

## 7. 对文档分层的建议

- `evidence-backed stable defaults`
  - detection / missingness summary
  - sample load / n_features summary
  - batch-stratified QC summary
- `heuristic exploratory filters`
  - global CV filtering
  - MAD doublet filtering without controls

## 8. 参考文献（点击可访问）

1. Wang et al., 2025, Nature Communications
   https://www.nature.com/articles/s41467-025-65174-4

2. Yu et al., 2024, Molecular & Cellular Proteomics
   https://pmc.ncbi.nlm.nih.gov/articles/PMC11103571/

3. Specht et al., 2023, Nature Methods
   https://www.nature.com/articles/s41592-023-01830-1

4. Patterson et al., 2023, Journal of the American Society for Mass Spectrometry
   https://pubmed.ncbi.nlm.nih.gov/37163770/

5. Goeminne et al., 2020, Molecular & Cellular Proteomics
   https://pubmed.ncbi.nlm.nih.gov/32234965/

6. Lazar et al., 2016, Journal of Proteome Research
   https://pubmed.ncbi.nlm.nih.gov/26906401/
