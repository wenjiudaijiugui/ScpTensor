# DIA 驱动单细胞蛋白组预处理中的 z-score standardization 与 downstream scale contract：优先级文献综述（截至 2026-03-13）

## 1. 研究范围

- 研究问题：ScpTensor 当前公开暴露的 `zscore` 接口应如何在文档层被正确定义，才能既不把它误写成核心 quantitative normalization，又能明确它在 `heatmap / PCA / embedding / clustering` 等 downstream exploratory tasks 中的合理边界？
- 目标输出：为以下接口与文档提供合同化依据：
  - `scptensor.standardization.zscore`
  - `scptensor.__init__.zscore`
  - `scptensor.viz` 中依赖标准化表示层的可视化流程
  - `tutorial/` 中如需展示标准化 layer 的示例说明
- 核心边界：
  - ScpTensor 的最终稳定交付仍是 `protein-level quantitative matrix`，而不是 z-scored matrix。
  - `zscore` 应被视为 `representation layer` 或 `downstream helper`，不是上游 vendor normalization 的替代品。
  - `zscore` 与 `log transform / normalization / imputation` 的职责必须分开。
- 检索日期：`2026-03-13`

### 1.1 关键词优先级（P1 / P2 / P3）

| 优先级 | 目的 | 关键词 |
|---|---|---|
| P1 | 方法核心 | z-score standardization, feature scaling, standardization contract, normalization evaluation |
| P2 | 模态与数据 | proteomics, single-cell proteomics, DIA, protein matrix |
| P3 | 应用边界 | preprocessing, clustering, visualization, downstream analysis |

## 2. 检索策略与筛选标准

### 2.1 Query 梯度（P1 -> P2 -> P3）

1. 先检索 omics/proteomics 中与标准化、scale handling 和 normalization evaluation 直接相关的一手来源。
2. 再限定到 single-cell integration / single-cell proteomics benchmark，确认表示空间与任务边界。
3. 最后补充与 z-transform 在 proteomics 中直接相关的专门论文，以及本地 API 约束，收束成可执行合同。

### 2.2 纳入标准

- 一手来源：期刊官网、DOI 页面、PubMed、官方文档。
- 至少满足以下之一：
  - 直接讨论 omics/proteomics normalization 或 scale-handling 的数据依赖性
  - 直接讨论表示空间、output type 或 benchmark metric 的任务依赖性
  - 直接提出 proteomics 中的 z-transform 方法
  - 能对齐当前 `scptensor.standardization.zscore` 的输入输出合同

### 2.3 排除标准

- 只把 z-score 当作图注写法、没有方法学含义的资料。
- 非一手博客、论坛、未核验的二手综述。

### 2.4 本轮纳入

- 初筛候选：`20+`
- 深读纳入：`6` 条外部来源 + `1` 条本地实现上下文
- 其中直接 proteomics scale/normalization 来源：`3`
- single-cell benchmark / reporting 来源：`3`

## 3. 逐篇证据摘要（Per-paper Summaries）

说明：本节沿用全仓库资源分型。单篇论文默认记为 `论文证据`；仓库源码/本地实现说明记为 `本地实现上下文`。
共享高频条目的规范元数据统一以 `docs/references/citations.json` 为准；若本文件历史写法与 registry 在作者简称、发布日期、期刊、DOI 或 canonical URL 上不一致，以 registry 为准，本文件仅保留 z-score representation 的语义与边界。

### 3.1 Chawade et al., Journal of Proteome Research, 2014

- 资源类型：`论文证据`
- 题目：Normalyzer: a tool for rapid evaluation of normalization methods for omics data sets
- 链接：https://pubmed.ncbi.nlm.nih.gov/25234251/
- 主要发现：
  - omics 数据的 normalization/scale 处理是数据依赖的，不能脱离 benchmark 或诊断直接设为默认。
  - 不同 normalization family 对 downstream bias、variance 和 comparability 的影响不同。
- 对 ScpTensor 的意义：
  - `zscore` 不应被包装成“任何 DIA-SCP 数据都更适合的默认 normalization”。
  - 更合理的定位是：它属于特定任务下的 `representation transform`，而不是 stable quantitative contract。

### 3.2 Wang et al., Nature Communications, 2025

- 资源类型：`论文证据`
- 题目：Benchmarking informatics workflows for data-independent acquisition single-cell proteomics
- 链接：https://www.nature.com/articles/s41467-025-65174-4
- 发布日期：`2025-11-21`
- 主要发现：
  - DIA-SCP workflow 中不存在 one-size-fits-all 的 preprocessing choice。
  - normalization、imputation、batch correction 的优劣依赖数据异质性、批次结构和任务定义。
- 对 ScpTensor 的意义：
  - `zscore` 更不应被拔高成稳定主线的默认 quantitative preprocessing。
  - 若提供 `zscore`，文档必须把它限定为 downstream exploratory representation，而不是 protein-level final deliverable。

### 3.3 Gatto et al., Nature Methods, 2023

- 资源类型：`论文证据`
- 题目：Initial recommendations for performing, benchmarking and reporting single-cell proteomics experiments
- 链接：https://www.nature.com/articles/s41592-023-01785-3
- 发布日期：`2023-03-02`
- 主要发现：
  - single-cell proteomics 的 benchmark/reporting 需要清楚记录数据处理路径、metadata 和分析条件。
  - 结果图若脱离处理路径和输入层说明，会降低可复核性。
- 对 ScpTensor 的意义：
  - `zscore` layer 必须保留 provenance，至少写清 `source_layer / axis / ddof`。
  - 教程与报告中若展示 z-score heatmap 或 embedding，必须同时说明它不代表原始 quantitative scale。

### 3.4 Luecken et al., Nature Methods, 2022

- 资源类型：`论文证据`
- 题目：Benchmarking atlas-level data integration in single-cell genomics
- 链接：https://www.nature.com/articles/s41592-021-01336-8
- 发布日期：`2022-01-10`
- 主要发现：
  - 表示空间、预处理与指标选择会改变 integration benchmark 的结论。
  - 文中明确提醒：不同表示空间下的得分不可被机械视为同一含义。
- 对 ScpTensor 的意义：
  - `zscore` 产出的 layer 更接近 `analysis space`，而不是 `quantitative source of truth`。
  - 如果在 z-scored space 上计算 PCA/ASW/聚类指标，文档必须显式写出 `@zscore`，避免与 `@log` 或 `@normalized` 空间混淆。

### 3.5 Liu et al., Nature Methods, 2025

- 资源类型：`论文证据`
- 题目：Multitask benchmarking of single-cell multimodal omics integration methods
- 链接：https://www.nature.com/articles/s41592-025-02856-3
- 发布日期：`2025-10-20`
- 主要发现：
  - 不同 output type 适用的指标与方法不同，不能把所有表示空间压进单一总分或单一默认处理。
  - overall rank 只能作导航，不能替代 task-specific metric battery。
- 对 ScpTensor 的意义：
  - `zscore` 的合同应与任务绑定，例如 `heatmap / clustering / representation learning`。
  - 不应把 z-scored layer 拿去直接替代 protein-level quant layer 做 completeness、batch correction ranking 或 DE-friendly matrix 输出。

### 3.6 Gui et al., Molecular & Cellular Proteomics, 2024

- 资源类型：`论文证据`
- 题目：zMAP: a data normalization strategy enabling comparative analysis of quantitative proteomics data
- 链接：https://doi.org/10.1016/j.mcpro.2024.100791
- 主要发现：
  - proteomics 中若要使用 z-transform 支撑比较分析，通常需要专门的方法学建模，而不是简单把任意矩阵做一遍 generic z-score。
  - z-transform 的统计语义取决于其建模假设和适用数据结构。
- 对 ScpTensor 的意义：
  - 需要明确区分：
    - `generic layer standardization`
    - `method-defined z-transform for quantitative comparison`
  - 当前 `scptensor.standardization.zscore` 不应被写成与 zMAP 同等级的定量 normalization 方法。

### 3.7 本地实现上下文：`scptensor.standardization.zscore`

- 资源类型：`本地实现上下文`
- 代码位置：[scptensor/standardization/zscore.py](/home/shenshang/projects/ScpTensor/scptensor/standardization/zscore.py#L26)
- 当前实现直接语义：
  - 默认 `source_layer="imputed"`
  - 要求输入矩阵无 `NaN`
  - 支持 `axis=0/1`
  - 以新增 layer 方式写回，不覆盖源层
- 对文档合同的意义：
  - 当前 API 已经客观上把 `zscore` 放在 `complete matrix after missingness handling` 之后。
  - 这更支持把它界定为 `downstream helper layer`，而不是上游 vendor/raw quantitative layer 的替代。

### 3.8 二次核查补充（发布日期、表示空间与合同边界）

- `Wang et al., Nat Commun (published: 2025-11-21)` 明确支持 preprocessing choice 必须跟数据结构和任务场景绑定，因此 `zscore` 不应被写成 ScpTensor 的主线默认 normalization：<https://www.nature.com/articles/s41467-025-65174-4>
- `Gatto et al., Nat Methods (published: 2023-03-02)` 支持 transformation provenance 必须在 single-cell proteomics 报告中透明呈现；因此 `zscore` layer 的来源和参数必须文档化：<https://www.nature.com/articles/s41592-023-01785-3>
- `Luecken 2022` 与 `Liu 2025` 共同支持“表示空间依赖任务边界”，因此任何 `@zscore` 空间上的指标都应继续带着表示空间标签报告，而不是与 `@log` 或 `@normalized` 混成同一 quantitative layer：<https://www.nature.com/articles/s41592-021-01336-8>；<https://www.nature.com/articles/s41592-025-02856-3>
- `zMAP 2024` 进一步说明，如果要把 z-transform 用作定量比较方法，应把它视作专门方法，而不是 generic helper；当前 ScpTensor 的 `zscore` 文档必须避免这类语义漂移：<https://doi.org/10.1016/j.mcpro.2024.100791>

## 4. 横向比较与证据分级

### 4.1 一致结论（facts）

1. `zscore` 不是 proteomics quantitative normalization 的通用默认答案。
2. `zscore` 更适合被界定为下游表示空间变换，用于 heatmap、PCA、embedding、clustering 等 exploratory tasks。
3. 表示空间不同，指标解释就不同；`@zscore` 结果不应与 `@log`、`@vendor-normalized`、`@normalized` 空间直接混写。
4. 若要把 z-transform 作为 quantitative-comparison 方法，必须采用明确的方法学定义，而不是把 generic z-score 冒充成专门 normalization。

### 4.2 分歧与解释（inference）

- [推断] `axis=0` 的 feature-wise z-score 更接近 heatmap 或 feature-pattern display 的常见用途；`axis=1` 的 sample-wise z-score 更容易抹掉 sample load、overall intensity 和部分 QC 差异，因此不宜在 stable 文档中被包装成默认路径。
- [推断] 对 MBR-heavy、imputation-heavy 或 high-missingness 数据，直接在后验补全层上做 z-score 会把补全过程引入的结构也一并放大；因此任何 `@zscore` 解释都应同时标注来源层。

### 4.3 证据强度

- 高：Wang 2025、Gatto 2023、Luecken 2022、Liu 2025
- 中高：Chawade 2014、zMAP 2024
- 中：本地实现上下文（用于合同对齐，不替代外部方法学证据）

## 5. 面向 ScpTensor 的实践建议

### 5.1 文档定位

- 建议把 `scptensor.standardization.zscore` 明确写成：
  - `downstream exploratory helper`
  - `representation transform`
- 不应写成：
  - 默认 normalization
  - default batch-correction input
  - final protein-level quantitative deliverable

### 5.2 输入层合同

- 建议 stable 文档继续要求：
  - 输入必须是 `complete matrix`
  - 输入层必须显式命名
- 推荐优先顺序：
  1. `logged + complete`
  2. `normalized + logged + complete`
  3. `imputed + logged + complete`
- 不建议默认在以下层上静默执行：
  - `raw-linear`
  - `vendor-normalized-linear`
  - provenance 不明的 merged layer

### 5.3 适用任务

- 适合：
  - z-score heatmap
  - PCA / embedding 前的 exploratory representation
  - clustering input 的对照实验
  - feature-pattern visualization
- 不适合默认承担：
  - protein-level completeness 统计
  - batch correction 主榜评测
  - DE / fold-change friendly output layer
  - 最终对外 quantitative export

### 5.4 provenance 与报告字段

- 建议在文档或报告中显式保留：
  - `source_layer`
  - `axis`
  - `ddof`
  - `representation_space = zscore`
- 若后续在图表中使用，标签建议写成：
  - `heatmap@zscore(feature-wise)`
  - `pca@zscore`
  - `cluster@zscore`

## 6. 风险边界

1. 直接面向 DIA-SCP 的 z-score benchmark 文献仍然不多，更多证据来自 proteomics normalization 与 single-cell benchmark 的联合约束。
2. sample-wise z-score 容易抹掉与 sample load、global abundance 或 QC 相关的差异，风险高于 feature-wise exploratory use。
3. 对后验补全层做 z-score 时，图中分离度提升不等于 quantitative signal 更真实。
4. 若未来要引入真正用于 quantitative comparison 的 z-transform，应单列为独立方法家族，而不是复用当前 `zscore` helper 的语义。

## 7. Shared Citation Registry Coverage

以下共享高频条目的规范元数据以 `docs/references/citations.json` 为准：

- `chawade2014_jpr_normalyzer`
- `wang2025_natcom_dia_scp_benchmark`
- `gatto2023_natmethods_scp_recommendations`
- `luecken2022_natmethods_scib`
- `liu2025_natmethods_multitask_integration`
- `gui2024_mcpro_zmap`

本文件额外保留的当前非 registry 条目：

1. 本地实现上下文：`scptensor.standardization.zscore`
   [zscore.py](/home/shenshang/projects/ScpTensor/scptensor/standardization/zscore.py)
