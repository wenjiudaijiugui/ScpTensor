# DIA 驱动单细胞蛋白组预处理中的 state-aware completeness 与 uncertainty 指标：优先级文献综述（截至 2026-03-12）

## 1. 研究范围

- 研究问题：ScpTensor 应如何把 `VALID / MBR / LOD / FILTERED / OUTLIER / IMPUTED / UNCERTAIN` 这些状态纳入 completeness、QC、AutoSelect 与 benchmark 指标，而不是把所有非空值折叠成单一“完整率”？
- 目标输出：为以下实现与文档提供指标合同：
  - `scptensor.core.structures.MaskCode`
  - `benchmark/README.md`
  - `benchmark/imputation`
  - `benchmark/autoselect`
  - `docs/io_diann_spectronaut.md`
- 核心边界：项目合同要求最终交付是完整 protein-level matrix，但完整性不能仅用“非 NaN 比例”表达。
- 检索日期：`2026-03-12`

### 1.1 关键词优先级（P1 / P2 / P3）

| 优先级 | 目的 | 关键词 |
|---|---|---|
| P1 | 方法核心 | completeness metrics, uncertainty metrics, missingness semantics, quality metrics |
| P2 | 模态与数据 | proteomics, DIA, single-cell proteomics |
| P3 | 应用边界 | MBR, LOD, imputation, QC |

## 2. 检索策略与筛选标准

### 2.1 Query 梯度（P1 -> P2 -> P3）

1. 先检索 proteomics missingness semantics 与质量控制框架。
2. 再限定到 DIA 与 single-cell proteomics。
3. 最后补充 missing-aware harmonization 与 workflow benchmark，确认这些状态如何进入 downstream 评价。

### 2.2 纳入标准

- 一手来源：官方文档、Nature/PMC/PubMed 页面。
- 直接涉及以下至少一项：
  - missingness / detection state 的语义区分
  - completeness、data sparsity、uncertainty、QC 的评价方式
  - missing-aware 或 state-aware 的预处理/整合框架
  - 可迁移到 ScpTensor `MaskCode` 指标合同的结论

### 2.3 排除标准

- 只把缺失值当作单一噪声源、没有状态区分的泛泛讨论。
- 无法映射到 current preprocessing contract 的二手意见。

### 2.4 本轮纳入

- 初筛候选：`20+`
- 深读纳入：`7`
- 其中 DIA / DIA-sc 直接证据：`4`
- missing-aware / semantic 基础来源：`3`

## 3. 逐篇证据摘要（Per-paper Summaries）

说明：本节统一沿用全仓库资源分型。除特别标注外，单篇文献条目默认记为 `论文证据`；官方软件/手册页记为 `模块规范 / 软件文档`；具体 accession 或 dataset page 记为 `数据入口`；可脚本化分发包记为 `资源包`。

### 3.1 DIA-NN 官方文档（访问于 2026-03-12）

- 资源类型：`模块规范 / 软件文档`
- 链接：https://vdemichev.github.io/DiaNN/
- 文档直接信息：
  - `MBR` 是两遍流程产生的 cross-run completion，不等同于同一 run 中直接观测。
  - 未被报告的 precursor / protein 不应被简单解释为“样本里不存在”，更常见的解释是低丰度或未达稳定识别阈值。
  - `Zero quantities` 更接近低浓度 analyte 的占位信息，若要 log-transform，可替换为 `NA`。
- 对 ScpTensor 的意义：
  - `VALID`、`MBR`、`LOD-like zero / missing` 必须分开统计。
  - “非空”不等于“直接观测且高可信”。

### 3.2 Lazar et al., Journal of Proteome Research, 2016

- 题目：Accounting for the Multiple Natures of Missing Values in Label-Free Quantitative Proteomics Data Sets to Compare Imputation Strategies
- 链接：https://pubmed.ncbi.nlm.nih.gov/26906401/
- 主要发现：
  - missingness 有多种机制，不能一律按单一噪声解释。
  - abundance-dependent missingness 为 `MNAR / left-censored` 提供经典 proteomics 基线。
- 对 ScpTensor 的意义：
  - `LOD` 与 `FILTERED`、`OUTLIER`、`IMPUTED` 不能进入同一 completeness 分母而不加区分。

### 3.3 Vanderaa and Gatto, Journal of Proteome Research, 2023

- 题目：Revisiting the thorny issue of missing values in single-cell proteomics
- 链接：https://pubmed.ncbi.nlm.nih.gov/37530557/
- 主要发现：
  - 单细胞蛋白组的 missing values 同时受 low input、随机缺失、检测极限与处理流程影响。
  - 在 single-cell 语境下，missingness 既是技术问题，也是 downstream interpretation 问题。
- 对 ScpTensor 的意义：
  - state-aware completeness 指标在 DIA-sc 中比 bulk proteomics 更重要。
  - 不能把高稀疏性简单折叠成单一低质量标签。

### 3.4 Yu et al., Molecular & Cellular Proteomics, 2024

- 题目：Quantification Quality Control Emerges as a Crucial Factor to Enhance Single-Cell Proteomics Data Analysis
- 链接：https://pmc.ncbi.nlm.nih.gov/articles/PMC11103571/
- 主要发现：
  - quantification QC 是提升 SCP downstream analysis 的关键步骤。
  - “值是否可信”与“是否非空”不是同一问题。
- 对 ScpTensor 的意义：
  - `FILTERED`、`OUTLIER`、`UNCERTAIN` 应单独进入 uncertainty 指标。
  - 质量控制后的值不能 silently 重新并入 `VALID` completeness。

### 3.5 HarmonizR, Nature Communications, 2022

- 题目：HarmonizR enables data harmonization across independent proteomic datasets with appropriate handling of missing values
- 链接：https://www.nature.com/articles/s41467-022-31007-x
- 主要发现：
  - 跨数据集整合若忽略 missingness 结构，会损伤可解释性。
  - missing-aware processing 是整合与 harmonization 的必要条件。
- 对 ScpTensor 的意义：
  - completeness 不是单一“越高越好”，还要看 completeness 是如何获得的。
  - 依赖 `MBR`、`IMPUTED` 或高 `UNCERTAIN` 的“完整率”不能与直接观测完整率等同。

### 3.6 Wang et al., Nature Communications, 2025

- 题目：Benchmarking informatics workflows for data-independent acquisition single-cell proteomics
- 链接：https://www.nature.com/articles/s41467-025-65174-4
- 主要发现：
  - `data completeness` 是 DIA-sc workflow 里的关键前置指标，但它必须和 downstream 任务联合解释。
  - 不同 workflow 在 completeness、聚类、DE 等指标间存在 trade-off。
- 对 ScpTensor 的意义：
  - state-aware 指标不应停留在描述层，还应进入 AutoSelect 和 benchmark 评分体系。

### 3.7 Goeminne et al., Molecular & Cellular Proteomics, 2020

- 题目：Selection of Features with Consistent Profiles Improves Relative Protein Quantification in Mass Spectrometry Experiments
- 链接：https://pubmed.ncbi.nlm.nih.gov/32234965/
- 主要发现：
  - feature consistency 直接影响最终 protein quantification。
  - “有值”不代表“值得信任”。
- 对 ScpTensor 的意义：
  - 除 completeness 外，还应显式保留 uncertainty / consistency burden 指标。

### 3.8 二次核查补充（资源分型、稳定入口与场景边界）

- `DIA-NN 官方文档`（accessed: `2026-03-12`）在本综述中应稳定定位为 `模块规范 / 软件文档`，它约束的是 `MBR / zero / unreported` 等输出语义，而不是 benchmark 数据入口：<https://vdemichev.github.io/DiaNN/>
- 其余 `Lazar 2016`、`Vanderaa and Gatto 2023`、`Yu 2024`、`HarmonizR 2022`、`Wang 2025`、`Goeminne 2020` 均属于 `论文证据`；这些来源共同支持“state-aware completeness 必须分解为状态向量，而非单一完整率”。
- 本综述当前没有单独指定 `数据入口` 或 `资源包`；其职责是给 `state vector contract` 提供语义与统计口径依据，而不是给 benchmark 选数据源。
- 因此文档间引用时，应把本综述定位为 `metric/contract evidence`，并把数据选择继续挂接到公共 benchmark 数据综述。

## 4. 横向比较与证据分级

### 4.1 一致结论（facts）

1. completeness 不是单一数字，而是多个状态共同构成的状态向量。
2. `MBR`、`LOD-like missing`、`FILTERED`、`OUTLIER`、`IMPUTED` 与 `VALID` 在统计意义上不同，不应合并成同一“observed rate”。
3. 在 DIA-sc 中，data completeness 只有和 uncertainty burden、downstream task performance 一起看才有意义。
4. 缺失值与不确定性不仅影响 imputation，也影响 QC、integration 和 AutoSelect。

### 4.2 推荐的 state-aware 指标合同

| 指标 | 定义建议 | 默认用途 |
|---|---|---|
| `valid_rate` | `VALID / total_cells_features` | 直接观测完整性 |
| `mbr_rate` | `MBR / total` | run 间转移依赖程度 |
| `lod_rate` | `LOD / total` | low-abundance / left-censored burden |
| `filtered_rate` | `FILTERED / total` | 规则/阈值导致的删除负担 |
| `outlier_rate` | `OUTLIER / total` | 异常值剔除负担 |
| `imputed_rate` | `IMPUTED / total` | 后验填补依赖程度 |
| `uncertain_rate` | `UNCERTAIN / total` | 语义无法确定或低置信 burden |

补充建议：
- `direct_observation_rate = valid_rate`
- `supported_observation_rate = valid_rate + mbr_rate`
- 不建议默认定义单一 `completeness_score` 替代上述状态向量。

### 4.3 AutoSelect / benchmark 中的用法建议

- `QC`
  - 优先看 `valid_rate` 与 `lod_rate`
  - 不把 `imputed_rate` 当作质量提升
- `imputation`
  - 保留原始状态向量
  - 新增 `imputed_rate`，而不是覆盖 `lod_rate`
- `integration`
  - batch correction 后报告状态向量是否被异常放大
- `AutoSelect`
  - 把 `uncertainty_burden = filtered_rate + outlier_rate + uncertain_rate + imputed_rate` 作为惩罚轴之一

### 4.4 分歧与解释（inference）

- [推断] 对某些真实 workflow，`MBR` 可在 practical analysis 中视作“可用但非直接观测”的次级支持状态，因此既不应算作纯缺失，也不应与 `VALID` 完全等价。
- [推断] 若未来需要单一分数，应至少同时保留原始状态向量，避免总分掩盖来源结构。

### 4.5 证据强度

- 高：DIA-NN 官方文档、Lazar 2016、Vanderaa 2023、Yu 2024、Wang 2025
- 中高：HarmonizR 2022、Goeminne 2020

## 5. 面向 ScpTensor 的实践建议

### 5.1 `benchmark/README.md` 应增加状态向量合同

- 不再只说“缺失率”或“完整率”。
- 明确区分：
  - `valid_rate`
  - `mbr_rate`
  - `lod_rate`
  - `filtered_rate`
  - `outlier_rate`
  - `imputed_rate`
  - `uncertain_rate`

### 5.2 `MaskCode` 不应被后处理 silently 改写

- `IMPUTED` 不应覆盖原始 `LOD`
- `FILTERED` / `OUTLIER` 不应重新记作 `VALID`
- 若生成新 layer，应新增 provenance，而不是抹除原始状态来源

### 5.3 AutoSelect 报告应直接展示状态负担

- 至少展示：
  - 直接观测率
  - 补全依赖率（`mbr_rate + imputed_rate`）
  - uncertainty burden
- 这样用户才能判断某方法的“高完整率”是否只是靠更激进的转移/填补得到。

## 6. 风险边界

1. 目前尚无单一社区标准完全覆盖 `VALID / MBR / LOD / FILTERED / OUTLIER / IMPUTED / UNCERTAIN` 七态；本合同是对 proteomics missingness、QC 和 DIA-sc benchmark 证据的工程化收束。
2. 上游软件若只提供最终 pivot matrix，部分状态只能保守映射为 `LOD` 或 `UNCERTAIN`。
3. 若只看单一 completeness 数字，仍然可能误判 aggressive MBR/imputation 为“质量提升”。

## 7. 对后续实现/文档的优先建议

1. 在 `benchmark/README.md` 中新增 `State-Aware Metric Contract` 一节。
2. 在 `io_diann_spectronaut.md` 中写明哪些状态可以直接从上游推断，哪些必须保守映射。
3. 在 AutoSelect 结果对象里保留 state-aware burden 字段，而不只保留最终总分。

## 8. 参考文献（含链接）

1. DIA-NN 官方文档
   https://vdemichev.github.io/DiaNN/

2. Lazar et al., 2016, Journal of Proteome Research
   https://pubmed.ncbi.nlm.nih.gov/26906401/

3. Vanderaa and Gatto, 2023, Journal of Proteome Research
   https://pubmed.ncbi.nlm.nih.gov/37530557/

4. Yu et al., 2024, Molecular & Cellular Proteomics
   https://pmc.ncbi.nlm.nih.gov/articles/PMC11103571/

5. HarmonizR, 2022, Nature Communications
   https://www.nature.com/articles/s41467-022-31007-x

6. Wang et al., 2025, Nature Communications
   https://www.nature.com/articles/s41467-025-65174-4

7. Goeminne et al., 2020, Molecular & Cellular Proteomics
   https://pubmed.ncbi.nlm.nih.gov/32234965/
