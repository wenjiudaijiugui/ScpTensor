# DIA 模式单细胞蛋白组数据处理中的归一化方法：优先级文献综述（截至 2026-03-07）

## 1. 研究范围

- 研究问题：在 **DIA 驱动单细胞蛋白组（DIA-scProteomics）** 预处理中，归一化方法应如何选择？
- 目标输出：为 `DIA-NN / Spectronaut` 输入下、`protein-level` 矩阵分析提供证据化建议。
- 检索日期：`2026-03-07`。

### 1.1 关键词优先级（P1/P2/P3）

| 优先级 | 目的 | 关键词 |
|---|---|---|
| P1 | 方法核心 | normalization, intensity normalization, quantile normalization, median normalization, robust normalization |
| P2 | 模态与软件 | DIA, diaPASEF, SWATH, DIA-NN, Spectronaut |
| P3 | 应用场景 | single-cell proteomics, scProteomics, plexDIA |

## 2. 检索策略与筛选标准

### 2.1 Query 梯度（P1 -> P2 -> P3）

1. 先检索归一化方法与 benchmark/evaluation（P1）。
2. 再加入 DIA/软件约束（P2）。
3. 最后限定单细胞与 multiplex DIA 场景（P3）。

### 2.2 纳入标准

- 一手来源：期刊官网、PubMed、DOI 页面、官方软件文档。
- 直接或强相关于下列至少一项：
  - DIA 单细胞流程中的归一化对比；
  - DIA 常用软件的归一化机制；
  - 缺失值/稀疏矩阵下可迁移的蛋白组归一化方法。

### 2.3 排除标准

- 非一手二次转载、不可核验博客。
- 仅讨论差异分析但未提供归一化策略证据。

### 2.4 本轮筛选结果

- 初筛候选：`>40` 条。
- 深读纳入：`10` 条（DIA-sc 直接证据 3 条，DIA/蛋白组可迁移证据 7 条）。

## 3. 逐篇证据摘要（Per-paper Summaries）

说明：本节统一沿用全仓库资源分型。除特别标注外，单篇文献条目默认记为 `论文证据`；官方软件/手册页记为 `模块规范 / 软件文档`；具体 accession 或 dataset page 记为 `数据入口`；可脚本化分发包记为 `资源包`。
共享高频条目的规范元数据统一以 `docs/references/citations.json` 为准；若本文件历史写法与 registry 在作者简称、发布日期、期刊、DOI 或 canonical URL 上不一致，以 registry 为准，本文件仅保留 normalization 语境下的解释。

### P1. Wang et al., Nature Communications, 2025

- 题目：Benchmarking informatics workflows for data-independent acquisition single-cell proteomics
- 链接：https://www.nature.com/articles/s41467-025-65174-4
- DOI：https://doi.org/10.1038/s41467-025-65174-4
- 目标：在 DIA 单细胞流程中联合评估缺失值处理、归一化、批次校正等步骤。
- 归一化结果（论文直接结论）：
  - 同质样本、同批次：`sum` / `median` 表现较好；
  - 同批次但生物组成差异显著：`no normalization` 更优；
  - 跨批次：`quantile normalization` 表现更优。
- 局限：结论依赖其数据结构与任务定义，不能机械外推到所有数据集。
- 相关性（P1/P2/P3）：`2/2/2`。

### P2. Demichev et al., Nature Methods, 2020（DIA-NN 原始论文）

- 题目：DIA-NN: neural networks and interference correction enable deep proteome coverage in high throughput
- 链接：https://www.nature.com/articles/s41592-019-0638-x
- DOI：https://doi.org/10.1038/s41592-019-0638-x
- 价值：DIA-NN 是当前 DIA 工作流核心软件之一，其量化/批间对齐策略构成归一化实践基础。
- 局限：论文主体不等同于“归一化方法 benchmark”。
- 相关性：`1/2/1`。

### P3. DIA-NN 官方文档（软件实现证据）

- 资源类型：`模块规范 / 软件文档`
- 链接：https://vdemichev.github.io/DiaNN/
- 关键点（文档直接描述）：支持 `RT-dependent`、`global`、`no normalization`；并明确提示单细胞场景在细胞状态差异较大时可考虑禁用归一化。
- 局限：软件文档不是独立对照实验论文。
- 相关性：`2/2/2`。

### P4. Spectronaut 用户手册（软件实现证据）

- 资源类型：`模块规范 / 软件文档`
- 链接：https://biognosys.com/resources/spectronaut-manual/
- 关键点（手册描述）：提供 `global`、`local`、`automatic` 归一化策略；cross-run normalization 在 precursor intensity 层进行。
- 局限：同样属于软件说明，不是公开 benchmark。
- 相关性：`2/2/1`。

### P5. Brombacher et al., Proteomics, 2020（TRQN）

- 题目：Tail-Robust Quantile Normalization
- 链接：https://pubmed.ncbi.nlm.nih.gov/32865322/
- 目标：针对低输入、缺失严重蛋白组数据，提出尾部稳健分位数归一化（TRQN）。
- 主要发现：在缺失值较多场景下，TRQN 相比标准 quantile normalization 有更好稳健性。
- 局限：不是 DIA 单细胞专用，但高度可迁移。
- 相关性：`2/1/1`。

### P6. Poulos et al., Nature Communications, 2020（ProNorM pipeline 证据）

- 题目：Strategies to enable large-scale proteomics for reproducible research
- 链接：https://doi.org/10.1038/s41467-020-17641-3
- 目标：在大规模、长期 DIA-MS 队列中构建兼顾 reproducibility、missing-value handling 与 normalization 的处理框架，其中 `ProNorM` 是论文中的整合 pipeline 名称。
- 主要发现：missing-aware 的归一化可提升下游功能分析稳定性。
- 局限：面向 bulk 蛋白组；对 DIA-sc 需二次验证。
- 相关性：`2/1/1`。

### P7. Ammar et al., Molecular & Cellular Proteomics, 2023（directLFQ）

- 题目：Accurate Label-Free Quantification by directLFQ to Compare Unlimited Numbers of Proteomes
- 链接：https://doi.org/10.1016/j.mcpro.2023.100581
- PubMed：https://pubmed.ncbi.nlm.nih.gov/37225017/
- 目标：提出高效蛋白定量（directLFQ），通过 ratio/强度整合减轻跨样本不一致性。
- 价值：在 DIA-sc 场景中被后续工作用于 normalization + quantification 组合。
- 局限：方法核心是定量框架，不是单独归一化 benchmark。
- 相关性：`1/2/2`。

### P8. Ali et al., Analytical Chemistry, 2024（DIA-ME）

- 题目：DIA-ME: A Cross-Compatible Data Analysis Pipeline for Multiplexed Data-Independent Acquisition
- 链接：https://pubmed.ncbi.nlm.nih.gov/39292979/
- 关键点（论文直接描述）：在复用 DIA-NN feature matching 后，`directLFQ` 量化更稳，且“已做 feature matching 后再做全局归一化可能降低性能”。
- 局限：重点是 multiplex DIA 分析链路，不是单细胞纯 label-free 设计。
- 相关性：`2/2/2`。

### P9. Derks et al., Nature Biotechnology, 2023（plexDIA）

- 题目：Increasing the throughput of sensitive proteomics by plexDIA
- DOI：https://doi.org/10.1038/s41587-022-01389-w
- 价值：确立 multiplex DIA 单细胞路线（channel ratio 思想），为“参考通道/比值型归一化”提供场景基础。
- 局限：论文重心在采集与定量深度，不是归一化方法 benchmark。
- 相关性：`1/2/2`。

### P10. Brunner et al., Molecular Systems Biology, 2023（RefQuant）

- 题目：Exploring single-cell data with deep visualization, quantification, and multiclonal inference with mDIA and RefQuant
- DOI：https://doi.org/10.15252/msb.202211503
- 链接：https://pmc.ncbi.nlm.nih.gov/articles/PMC10486649/
- 关键点（论文直接描述）：通过参考通道（reference channel）将多批次结果映射到统一尺度，减轻 run-to-run 比例偏移。
- 局限：主要适用于 multiplex/参考通道设计，不等同于 label-free DIA。
- 相关性：`2/2/2`。

### P11. Karuppanan et al., Journal of Proteome Research, 2025（normalization + imputation 组合）

- 题目：A Statistical Approach for Identifying the Best Combination of Normalization and Imputation Methods for Label-Free Proteomics Expression Data
- 链接：https://doi.org/10.1021/acs.jproteome.4c00552
- 关键点（论文直接描述）：
  - 最优预处理策略应以 normalization + imputation 组合作为比较单位，而不是把两步完全割裂后再宣称某个单方法是全局默认。
  - 组合优劣随数据集而变化，不存在跨场景稳定的统一赢家。
  - 论文同时给出 `lfproQC` 包与 Shiny 界面，强调组合比较与报告透明性。
- 局限：主体证据仍来自 label-free proteomics，而不是严格的 DIA 单细胞。
- 相关性：`2/2/2`。

### 3.12 二次核查补充（资源分型、稳定入口与场景边界）

- `DIA-NN docs` 与 `Spectronaut manual` 在本综述里都应固定归类为 `模块规范 / 软件文档`；它们提供 upstream normalization 语义与导出层解释，不应与 benchmark paper 混写：<https://vdemichev.github.io/DiaNN/>；<https://biognosys.com/resources/spectronaut-manual/>
- `Wang 2025` 是最直接的 DIA-SCP normalization/task-design `论文证据`，但不应被扩大解释为当前稳定公共 benchmark 输入来源：<https://www.nature.com/articles/s41467-025-65174-4>
- `TRQN`、`ProNorM`、`directLFQ` 提供的是 normalization family / quantification family 的 `论文证据`；它们不是 `模块规范`、`资源包` 或 `数据入口`，其规范元数据应统一回收至 registry。
- `plexDIA` 与 `RefQuant` 更适合作为 multiplex/reference-channel 设计证据，而不是当前合同内 label-free DIA 主线的默认输入或默认 normalization 路线。
- `Karuppanan 2025` 进一步说明 normalization 与 imputation 的最优组合是数据依赖的；因此当前仓库的 stage-specific normalization benchmark 应继续明确其边界，不应把单阶段结果扩大解释成“完整 preprocessing 组合最优”。
- 本综述当前不直接承担 `数据入口` 与 `资源包` 的稳定入口职责；该角色应继续由公共 benchmark 数据综述和后续 manifest 统一管理。

## 4. 横向比较与证据分级

### 4.1 方法对照表

| 方法路线 | 代表来源 | 适用场景 | 风险 |
|---|---|---|---|
| `no normalization` | Wang 2025, DIA-NN docs | 同批次且细胞群体总体强度差异具有生物学意义时 | 可能保留技术漂移 |
| `median/sum` 全局缩放 | Wang 2025 | 同质细胞、技术漂移主要体现在整体强度时 | 可能压缩真实生物差异 |
| `quantile` | Wang 2025 | 跨批次、组成相对可比时 | 分布强行一致，易过校正 |
| `TRQN`/稳健分位数 | Brombacher 2020 | 低输入、高缺失数据 | 实现复杂度高于普通 quantile |
| missing-aware 归一化 | ProNorM 2020 | 大规模、缺失非随机风险高 | 需额外验证参数与可解释性 |
| 比值/参考通道归一化 | RefQuant 2023, plexDIA 2023 | multiplex DIA、存在稳定 reference channel | 对实验设计依赖强 |

### 4.2 一致结论（facts）

1. 归一化不存在全场景单一最优；最优策略受批次结构、细胞异质性、缺失率共同影响。
2. 单细胞中若全局信号差异含真实生物意义，强制全局归一化可能损伤下游生物学可解释性。
3. 高缺失或低输入场景下，稳健分位数/缺失感知方法通常优于简单分位数或裸全局缩放。
4. multiplex DIA 应优先考虑参考通道/比值框架，而不是沿用 label-free 规则。

### 4.3 分歧与解释（inference）

- [推断] `quantile normalization` 在某些 benchmark 中表现好，核心前提是“各批次总体分布可对齐”；当细胞类型组成差异很大时，此前提可能不成立。
- [推断] `feature matching` 已显著降低系统偏移后，再叠加激进全局归一化收益有限，甚至可能引入额外失真。

### 4.4 证据强度

- 高：P1（DIA-sc 直接 benchmark）。
- 中高：P8/P10（multiplex DIA 中归一化策略直接证据）、P5（低输入稳健归一化）。
- 中：P6/P7/P9/P2（可迁移支撑）。

## 5. 面向 ScpTensor 的实践建议（protein-level, DIA-NN/Spectronaut）

### 5.1 默认候选归一化集合（建议）

1. `none`
2. `median`
3. `sum`
4. `quantile`（仅在显式 `log` layer 上比较）
5. `robust_quantile`（TRQN/MBQN 类，仅在显式 `log` layer 上比较，可作为扩展）

注：结合后续 `review_log_scale_20260312.md` 的收束结论，当前 ScpTensor 合同不应在 `raw` 线性层上自动比较 `quantile` / `robust_quantile`；若该 `raw` 层承载 vendor-normalized 输入，也应继续通过 `is_vendor_normalized` provenance 说明，而不是切换到另一套默认 layer 名。若需要比较，应先生成显式 `log` 层并保留 `scale` / `pseudocount` provenance。
另注：这里的 `sum` 属于文献层建议保留的 rescaling baseline；截至 `2026-03-12`，ScpTensor 当前 stable normalization API / benchmark README 已明确落地的方法池仍是 `none / mean / median / quantile / trqn`，因此不应把 `sum` 或泛化的 `robust_quantile` 写成“当前已实现事实”。其中当前实现与 benchmark 直接对应的稳健分位数方法名是 `trqn`。

### 5.2 规则化选择逻辑（建议）

1. 若样本同批且细胞群体异质性高：优先比较 `none` vs `median`，避免直接上 `quantile`。
2. 若跨批次且批次效应明显，且当前比较层为显式 `log` layer：优先比较 `quantile` / `robust_quantile`，并联合批次校正评估。
3. 若为 multiplex DIA 且有 reference channel：优先参考通道比值归一化路线。
4. 若目标是决定完整 preprocessing 组合，而不是只决定单步 normalization：应把 normalization 与 imputation 组合做成单独 sensitivity panel，而不是把单阶段 normalization 榜单直接当作最终全流程答案。

### 5.3 风险边界

- 不能仅用重建误差决定归一化优劣，必须联合下游任务指标（聚类、差异蛋白、通路）。
- 不应把“全体样本分布一致”当作默认前提，尤其在不同细胞类型比例变化时。
- 归一化要与缺失值处理、批次校正联动评估，避免串联步骤互相抵消或放大偏差。
- `quantile` / TRQN 一类方法对输入尺度假设更强；在当前工程合同下，不应对线性 vendor 输出层直接自动比较。

## 6. Shared Citation Registry Coverage

以下共享高频条目的规范元数据以 `docs/references/citations.json` 为准：

- `wang2025_natcom_dia_scp_benchmark`
- `demichev2020_natmethods_diann`
- `diann_docs`
- `spectronaut_manual`
- `brombacher2020_proteomics_trqn`
- `poulos2020_natcom_pronorm`
- `ammar2023_mcpro_directlfq`
- `derks2023_natbiotechnol_plexdia`
- `ali2024_analchem_diame`
- `brunner2023_msb_refquant`
- `callister2006_jpr_normalization`
- `karuppanan2025_jpr_lfproqc_combinations`

注：以上网页链接访问日期均为 `2026-03-07`。
