# DIA 驱动单细胞蛋白组预处理中的 peptide/precursor -> protein aggregation benchmark 设计：优先级文献综述（截至 2026-03-12）

## 1. 研究范围

- 研究问题：ScpTensor 应如何 benchmark `peptide/precursor -> protein` 聚合方法，使评测结果既符合项目合同中的 **protein-level final deliverable**，又能真实反映低输入、缺失和特征异质性对 protein summarization 的影响？
- 目标输出：为以下实现与 benchmark 目录提供证据化设计建议：
  - `scptensor.aggregation.aggregate_to_protein`
  - `scptensor.io.aggregate_to_protein`（项目合同中的对外入口表述）
  - `benchmark/aggregation`
  - `benchmark/autoselect`
- 核心约束：根据项目合同，`scptensor.aggregation` 是唯一专门负责 peptide/precursor -> protein 转换的阶段；除该阶段外，下游 stable 模块默认在 protein 层工作。若用户经 `load_peptide_pivot` 导入 peptide/precursor matrix，本综述结论同样约束其后续 `aggregate_to_protein` 交接合同。
- 检索日期：`2026-03-12`

### 1.1 关键词优先级（P1 / P2 / P3）

| 优先级 | 目的 | 关键词 |
|---|---|---|
| P1 | 方法核心 | protein aggregation, peptide to protein summarization, benchmarking |
| P2 | 模态与数据 | proteomics, DIA, single-cell proteomics |
| P3 | 应用边界 | complete protein matrix, reproducibility, ratio preservation |

## 2. 检索策略与筛选标准

### 2.1 Query 梯度（P1 -> P2 -> P3）

1. 先检索经典 protein summarization / aggregation 方法与官方实现说明。
2. 再补充与 feature consistency、ratio-preservation、protein-level benchmark 直接相关的一手研究。
3. 最后加入 DIA 单细胞与 protein-level benchmark 论文，确认这些聚合设计如何落到 ScpTensor 的最终交付边界。

### 2.2 纳入标准

- 一手来源：期刊官网、PMC / PubMed、官方软件文档。
- 直接涉及以下至少一项：
  - peptide/precursor -> protein summarization 方法语义
  - 聚合质量与 downstream protein quantification 的关系
  - protein-level benchmarking / reproducibility / ratio preservation
  - 可迁移到 ScpTensor `aggregation` benchmark 的评测设计

### 2.3 排除标准

- 只讨论 peptide identification、但不涉及 protein summarization 的资料。
- 无法复核的二手列表或方法汇编。

### 2.4 本轮纳入

- 初筛候选：`15+`
- 深读纳入：`6`
- 其中方法/算法基石：`4`
- protein-level / DIA-sc benchmark 证据：`2`

## 3. 逐篇证据摘要（Per-paper Summaries）

说明：本节统一沿用全仓库资源分型。除特别标注外，单篇文献条目默认记为 `论文证据`；官方软件/手册页记为 `模块规范 / 软件文档`；具体 accession 或 dataset page 记为 `数据入口`；可脚本化分发包记为 `资源包`。

### 3.1 OpenMS ProteinQuantifier 官方文档

- 资源类型：`模块规范 / 软件文档`
- 链接：https://openms.de/documentation/html/TOPP_ProteinQuantifier.html
- 直接价值：
  - 系统列出 `sum / mean / median / weighted_mean` 与 `iBAQ` 等 protein quantification 选项。
  - 说明 aggregation 方法不是单一默认，而是与实验目标和数据结构绑定。
- 对 ScpTensor 的意义：
  - 现有 `sum / mean / median / max / weighted_mean / top_n / ibaq` 方法池有官方生态参照。
  - benchmark 应比较“方法家族”，而不只比较某一篇论文中的单一算法。

### 3.2 Cox et al., Molecular & Cellular Proteomics, 2014

- 题目：Accurate Proteome-wide Label-free Quantification by Delayed Normalization and Maximal Peptide Ratio Extraction, Termed MaxLFQ
- 链接：https://pmc.ncbi.nlm.nih.gov/articles/PMC4159666/
- 主要发现：
  - `MaxLFQ` 通过 pairwise peptide ratio network 和 delayed normalization 提升 protein-level relative quantification 稳定性。
  - 核心目标不是“数值看起来平滑”，而是跨样本 ratio 的保真。
- 对 ScpTensor 的意义：
  - `maxlfq` 评测不能只看总量误差，应显式看 ratio preservation。
  - 对 protein-level benchmark，fold-change / rank consistency 比单点绝对误差更关键。

### 3.3 Ammar et al., Nature Methods, 2023

- 题目：A hybrid method for peptide-centered and protein-centered protein quantification
- 链接：https://www.nature.com/articles/s41592-022-01795-4
- 主要发现：
  - directLFQ 强调在 peptide-centered 与 protein-centered 视角之间折中，提升大规模 proteomics 的 protein quantification 鲁棒性。
  - 方法学焦点仍是 protein-level consistency，而不是仅仅多保留几个 peptide。
- 对 ScpTensor 的意义：
  - aggregation benchmark 不应只看 completeness；还要看“多保留的 peptide 是否真的改善最终 protein matrix”。
  - 这支持把 `protein-direct` 和 `precursor-to-protein` 两条评测赛道分开。

### 3.4 Goeminne et al., Molecular & Cellular Proteomics, 2020

- 题目：Selection of Features with Consistent Profiles Improves Relative Protein Quantification in Mass Spectrometry Experiments
- 链接：https://pubmed.ncbi.nlm.nih.gov/32234965/
- 主要发现：
  - 一致性差的 feature 会降低最终 protein quantification 质量。
  - 并非映射到同一蛋白的所有 peptide/precursor 都应平权进入 summarization。
- 对 ScpTensor 的意义：
  - aggregation benchmark 应显式包含 feature consistency 维度。
  - `top_n`、`weighted_mean`、`maxlfq` 一类方法的价值应在“抗不一致 peptide”场景下比较，而不是只在理想同质数据上比较。

### 3.5 Wang et al., Nature Communications, 2025

- 题目：Benchmarking informatics workflows for data-independent acquisition single-cell proteomics
- 链接：https://www.nature.com/articles/s41467-025-65174-4
- 主要发现：
  - DIA 单细胞工作流中，各步骤最优选择高度依赖数据结构和任务目标。
  - 最终评估应回到完整 protein-level matrix 及其 downstream 任务表现。
- 对 ScpTensor 的意义：
  - aggregation benchmark 不能只做 method micro-benchmark，必须最终回到 protein-level downstream 指标。
  - 在 DIA-sc 里，sparsity 与 low-input 噪声会放大 summarization 差异。

### 3.6 Zheng et al., Nature Communications, 2025

- 题目：Protein-level batch-effect correction enhances robustness in MS-based proteomics
- 链接：https://www.nature.com/articles/s41467-025-64718-y
- 主要发现：
  - protein-level benchmark 才真正对齐单细胞蛋白组下游可解释性和 reproducibility。
  - 只在 precursor / peptide 层报性能，不足以说明最终工作流是否更好。
- 对 ScpTensor 的意义：
  - 聚合 benchmark 主榜单必须以 protein matrix 为终点。
  - precursor/peptide 层可作为压力测试，但不应替代最终胜负标准。

### 3.7 二次核查补充（资源分型、稳定入口与场景边界）

- `OpenMS ProteinQuantifier`（accessed: `2026-03-12`）属于 `模块规范 / 软件文档`，其价值是给出 aggregation family 与参数语义，不应被写成公共 benchmark 数据入口：<https://openms.de/documentation/html/TOPP_ProteinQuantifier.html>
- `MaxLFQ`、`directLFQ`、`Goeminne 2020`、`Wang 2025`、`Zheng 2025` 在本综述里都属于 `论文证据`；它们约束的是 protein-level 终点评价与方法边界，而不是稳定数据入口。
- `Wang 2025` 与 `Zheng 2025` 共同支持 aggregation benchmark 必须回到 `protein-level matrix` 终点，但二者都不是当前仓库应直接绑定的公开 benchmark 数据页：<https://www.nature.com/articles/s41467-025-65174-4>；<https://www.nature.com/articles/s41467-025-64718-y>
- 本综述本身不指定 `数据入口` 或 `资源包`；若需要稳定公开输入，应回到 `docs/review_public_benchmark_data_20260312.md` 中已收束的 public dataset / module 入口。

## 4. 横向比较与证据分级

### 4.1 一致结论（facts）

1. aggregation 方法的优劣不能只看“多保留多少 peptide”，而要看最终 protein quantification 的 ratio preservation、reproducibility 和 downstream usefulness。
2. 不同 peptide/precursor 的一致性差异，是 summarization 质量的关键来源。
3. `MaxLFQ` / directLFQ 一类方法更强调跨样本 relative consistency，而不仅是样本内求和。
4. 对 ScpTensor 这类以 protein matrix 为最终 deliverable 的包，aggregation benchmark 主榜单应落在 protein 层。

### 4.2 建议的 benchmark 设计

| 组件 | 推荐做法 | 理由 |
|---|---|---|
| 评测赛道 | `protein-direct` 与 `precursor-to-protein` 分开 | 区分 summarization 收益与最终 protein 任务收益 |
| 方法池 | `sum / mean / median / max / weighted_mean / top_n / maxlfq / tmp / ibaq` | 与当前实现和主流生态对齐 |
| 数据场景 | 同质、异质、低缺失、高缺失、含不一致 feature | 放大方法间真实差异 |
| 主指标 | protein-level completeness、ratio preservation、within-group CV、DE consistency | 避免只看数值重建 |
| 附指标 | precursor->protein 映射覆盖率、unmapped rate、runtime | 工程可行性同样重要 |

### 4.3 分歧与解释（inference）

- [推断] 对 DIA-sc 稀疏数据，`sum`/`mean` 等简单聚合不一定总差，但其风险在于过度吸收不一致 feature；因此它们应被保留为基线，而不是被预设淘汰。
- [推断] `top_n` 与 `weighted_mean` 的优劣高度依赖 low-input 条件下 feature ranking 是否稳定，因此 benchmark 需要在不同缺失结构和噪声水平下重复比较。

### 4.4 证据强度

- 高：MaxLFQ 2014、Goeminne 2020、Wang 2025、Zheng 2025
- 中高：directLFQ 2023、OpenMS 官方文档

## 5. 面向 ScpTensor 的实践建议

### 5.1 `benchmark/aggregation` 应采用双榜单

1. `protein-direct main board`
   - 若上游软件已提供 protein table，则直接比较最终 protein matrix quality
2. `precursor-to-protein auxiliary board`
   - 从 precursor/peptide 表导入，经 `aggregate_to_protein()` 后回到 protein 层评分

### 5.2 主评分应以 protein-level 结果为中心

- 推荐最小指标集：
  - protein completeness
  - fold-change / ratio preservation
  - within-group CV
  - downstream DE consistency
  - runtime
- 不建议只用 precursor correlation 或 peptide 保留数决定排名。

### 5.3 guardrail 应覆盖当前实现的失败场景

- `No protein mapping column`
- `Missing iBAQ denominator`
- `Unsupported aggregation method`
- `top_n < 0`
- `lfq_min_ratio_count < 1`
- `tmp_log_base <= 1`
- `No protein groups found to aggregate`

这些场景不只是异常测试，也应进入 benchmark 文档的“failure mode”部分。

## 6. 风险边界

1. 直接面向 DIA-sc aggregation benchmark 的专门论文仍不多，很多设计原则来自 proteomics summarization 与 protein-level benchmark 的联合推断。
2. `PG.MaxLFQ` 之类上游 vendor protein outputs 已内含方法学假设，拿它们作为“真值”时必须注明只是参照，不是绝对 ground truth。
3. 对极端稀疏数据，任何 aggregation 方法都可能受 feature mapping 不完整影响，因此 benchmark 需要单独报告 unmapped/ambiguous burden。

## 7. 对后续实现/文档的优先建议

1. 在 `benchmark/README.md` 中单列 aggregation benchmark 轨道和主指标。
2. 在 `benchmark/aggregation` 里新增“ratio preservation + consistency stress”场景。
3. 在 `aggregation_literature.md` 中补一节“benchmark interpretation”，把方法语义与评测标准分开。

## 8. 参考文献（含链接）

1. OpenMS ProteinQuantifier 官方文档
   https://openms.de/documentation/html/TOPP_ProteinQuantifier.html

2. Cox et al., 2014, Molecular & Cellular Proteomics
   https://pmc.ncbi.nlm.nih.gov/articles/PMC4159666/

3. Ammar et al., 2023, Nature Methods
   https://www.nature.com/articles/s41592-022-01795-4

4. Goeminne et al., 2020, Molecular & Cellular Proteomics
   https://pubmed.ncbi.nlm.nih.gov/32234965/

5. Wang et al., 2025, Nature Communications
   https://www.nature.com/articles/s41467-025-65174-4

6. Zheng et al., 2025, Nature Communications
   https://www.nature.com/articles/s41467-025-64718-y
