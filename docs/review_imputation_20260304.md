# 基于 DIA 的单细胞蛋白组数据填充方法测评：公开文献整理（截至 2026-03-04）

## 1. 检索范围与筛选规则

- 检索日期：2026-03-04
- 主题优先级：`填充方法` > `DIA 数据` > `单细胞`
- 主要来源：Nature / Nature Communications / Nature Methods、PubMed、NAR、Genome Biology、Scientific Reports、arXiv
- 纳入标准：
  - 与蛋白组缺失值填充（imputation）方法或测评直接相关；
  - 或直接面向 DIA 单细胞蛋白组工作流并涉及缺失值处理；
  - 优先纳入可公开访问且有明确摘要/方法描述的论文。

说明：本报告中的“引用量”优先采用论文页面展示的可见指标（如 Nature 页面 `Citations`），不同平台统计口径可能不同。

对齐说明（`2026-03-12` 二次核查）：

- 本文是“方法池与证据面”综述，不等同于当前 ScpTensor 的最终 benchmark 合同。
- 当前更严格的 benchmark 口径，应以 `docs/review_masked_imputation_20260312.md` 与 `docs/review_missingness_20260312.md` 为准。
- 表中的“可见引用指标”仅是 `2026-03-04` 检索快照，不应用来直接决定方法优先级或评分权重。

## 2. 重点文献清单（按与你需求的相关性排序）

| 编号 | 论文 | 年份 | 与方向匹配度（填充/DIA/单细胞） | 可见引用指标* | 结论价值 |
|---|---|---:|---|---|---|
| P1 | Benchmarking informatics workflows for data-independent acquisition single-cell proteomics | 2025 | 高/高/高 | 待核实 | 直接给出 DIA 单细胞场景下填充与流程建议 |
| P2 | A simple optimization workflow for missing value imputation of two clinical proteomics data sets by DIA-MS | 2021 | 高/高/低 | 未显示 | 16 种方法 DIA 实证比较，给出 RF/LLS 结论 |
| P3 | NAguideR: performing and prioritizing missing value imputations for consistent bottom-up proteomic analyses | 2020 | 高/中/低 | 未显示 | 23 种方法系统比较，可作方法库与评测框架 |
| P4 | Comparative study of missing value imputation methods in label-free proteomics | 2021 | 高/低/低 | 148（Nature 页面） | 高引用基础测评论文，适合作为通用基线 |
| P5 | Comparative assessment and a novel strategy on methods for imputing proteomics data | 2022 | 高/低/低 | 31（Nature 页面） | 给出新评估视角与校准思路 |
| P6 | Imputation of label-free quantitative mass spectrometry-based proteomics data using self-supervised deep learning | 2024 | 高/中/中 | 待核实 | 深度学习填充在蛋白组数据上的代表性工作 |
| P7 | Single-cell proteomics enabled by next-generation sequencing or mass spectrometry | 2023 | 中/低/高 | 待核实 | 单细胞蛋白组方法学综述，强调缺失值与批次风险 |
| P8 | Enhanced feature matching in single-cell proteomics characterizes IFN-γ response and co-existence of cell states | 2024 | 中/中/高 | 待核实 | 更强调“先减少缺失再填充”的路线 |
| P9 | HarmonizR enables data harmonization across independent proteomic datasets with appropriate handling of missing values | 2022 | 中/中/中 | 待核实 | 以 harmonization 缓解缺失/批次问题 |
| P10 | scplainer: using linear models to understand mass spectrometry-based single-cell proteomics data | 2025 | 中/低/高 | 待核实 | 单细胞建模中显式处理缺失与批次的可解释框架 |
| P11 | Revisiting the Thorny Issue of Missing Values in Single-Cell Proteomics | 2023 | 高/低/高 | 未显示 | 对单细胞缺失值评估误区与方法边界做系统讨论 |
| P12 | Optimizing Missing Value Imputation in Mass Spectrometry-Based Proteomics by Considering Intensity and Missing Rates | 2025 | 高/中/低 | 未显示 | 提出按强度与缺失率分层建模，强调生物学信号保持 |

\* 引用指标均为检索时页面可见值（2026-03-04）。

## 3. 论文摘要（中文整理）

说明：本节统一沿用全仓库资源分型。除特别标注外，单篇文献条目默认记为 `论文证据`；官方软件/手册页记为 `模块规范 / 软件文档`；具体 accession 或 dataset page 记为 `数据入口`；可脚本化分发包记为 `资源包`。当前 P1-P12 条目均属于 `论文证据`。

### P1. Wang et al., Nat Commun, 2025（DIA 单细胞直接证据）

- 核心内容：针对 DIA 单细胞蛋白组数据，系统比较了分析工作流中的关键步骤（包括缺失值填充、归一化、批次校正、数据完整性过滤）在不同缺失程度和样本异质性下的表现。
- 主要结论：
  - 当缺失率低且细胞群体较同质时，严格筛选并避免过度填充可得到更稳健结果；
  - 当异质性高且缺失严重时，矩阵补全类方法可显著改善部分下游分析；但这一观察属于特定 workflow benchmark 结果，不应被解读成跨数据集的普适默认；
  - 归一化与批次校正需要与缺失模式联动选择（文中给出 `Quantile` + `ComBat` 的推荐组合场景）。
- 价值：当前最贴近“DIA + 单细胞 + 填充测评”三重交集的核心论文。

### P2. Dabke et al., J Proteome Res, 2021（DIA 场景填充实测）

- 核心内容：在两个临床 DIA-MS 数据集上比较 16 种缺失值填充方法，并按缺失比例分层评估。
- 主要结论：
  - 随机森林（RF）整体表现最稳健；
  - 局部最小二乘（LLS）在速度与准确性之间平衡较好；
  - 未填充时结果较差，且填充策略选择会显著影响下游统计结论。
- 价值：虽非单细胞，但对 DIA 缺失值方法选型具有直接参考意义。

### P3. Wang et al., NAR, 2020（NAguideR）

- 核心内容：构建 NAguideR 工具，系统评估 23 种 LC-MS 缺失值填充方法，并在不同缺失机制/数据结构下给出优先级建议。
- 主要结论：
  - 不存在单一“全场最优”填充方法；
  - KNN 在多种场景具有稳健性；
  - 在数据同质性高的条件下，模型驱动方法可获得更好效果。
- 价值：可作为方法池和评测框架参考，便于在 ScpTensor 中构建标准对照集。

### P4. Jin et al., Sci Rep, 2021（高引用通用基线）

- 核心内容：在多种模拟与真实数据条件下比较常见填充方法，重点分析缺失机制（随机/非随机）与缺失比例变化对性能的影响。
- 主要结论：
  - 最优方法依赖缺失机制和缺失率；
  - 简单方法在部分条件下可作为可解释基线，但在高缺失下性能不稳定；
  - 建议在方法评估中纳入多指标而非单一误差指标。
- 价值：作为高引用“通用 benchmark 基线文献”。

### P5. Zhang et al., Sci Rep, 2022（评估策略扩展）

- 核心内容：比较多种蛋白组填充方法并提出新的评估/校准策略，以减少仅凭单指标造成的偏差。
- 主要结论：
  - 方法表现高度数据依赖；
  - 引入校准策略可改善部分方法在真实数据上的稳定性；
  - 评估时需同时关注恢复精度与下游生物学结论一致性。
- 价值：对“如何测评填充方法”给出实操层面的补充。

### P6. Webel et al., Nat Commun, 2024（深度学习填充）

- 核心内容：提出自监督深度学习框架，用于 LFQ 蛋白组定量矩阵的缺失值重建。
- 主要结论：
  - 在其评测中较多种传统方法有更好恢复性能；
  - 对高缺失数据具有潜在优势；
  - 结果支持其在下游定量分析中的可用性。
- 价值：为“传统统计方法 vs 深度学习方法”提供强对照。

### P7. Bennett et al., Nat Methods, 2023（单细胞方法学综述）

- 核心内容：综述 NGS 读出与质谱读出两条单细胞蛋白组技术路线，讨论其定量覆盖与可扩展性。
- 主要结论：
  - 单细胞蛋白组分析中，缺失值与批次效应仍是关键挑战；
  - 上游采集与鉴定质量改进对降低缺失有直接帮助；
  - 缺失值处理应与下游任务共同报告和验证。
- 价值：为“测评报告应报告什么”提供社区规范依据。

### P8. Krull et al., Nat Commun, 2024（特征匹配增强）

- 核心内容：提出单细胞蛋白组增强特征匹配策略，以提高蛋白覆盖并支持细胞状态解析。
- 主要结论：
  - 改进特征匹配可减少矩阵稀疏性并提升可定量蛋白数；
  - 能更好解析刺激响应与细胞状态共存结构；
  - 强调“上游减少缺失”与“下游填充”互补。
- 价值：提示工作流应优先减少上游缺失，而非仅依赖下游 imputation。

### P9. Voss et al., Nat Commun, 2022（HarmonizR）

- 核心内容：通过跨数据集 harmonization，并显式考虑缺失值处理，提升独立蛋白组数据集可整合性。
- 主要结论：
  - 可改善跨批次/跨队列数据的一致性；
  - 对缺失值处理策略敏感，需与 harmonization 联动设定；
  - 与后续下游流程结合可进一步提升稳定性。
- 价值：属于“减少缺失影响”的另一条技术路线。

### P10. Vanderaa et al., Genome Biol, 2025（单细胞建模视角）

- 核心内容：提出 scplainer，以线性模型框架解释质谱单细胞蛋白组数据中的关键变异来源。
- 主要结论：
  - 线性可解释框架有助于识别主要生物与技术驱动因素；
  - 结果提示缺失值与批次因素需要在建模阶段一并考虑；
  - 强调“任务导向”的缺失值处理与解释。
- 价值：为单细胞场景提供“任务导向型缺失建模”思路。

### P11. Vanderaa & Gatto, J Proteome Res, 2023（观点/方法边界）

- 核心内容：讨论单细胞蛋白组缺失值的来源、评估误区与常见填充方法的适用边界。
- 主要结论：
  - 单一全局指标可能高估填充收益；
  - 需要在聚类、差异分析、通路解释等任务层面联合评估；
  - 建议报告“填充前后结果一致性”。
- 价值：为测评报告中的“鲁棒性检查”提供理论依据。

### P12. Zhao et al., Comput Struct Biotechnol J, 2025（强度/缺失率分层）

- 核心内容：提出在蛋白强度与缺失率分层框架下优化填充策略，减少整体统一策略带来的偏差。
- 主要结论：
  - 分层建模有助于在缺失修复与生物信号保持间取得更优平衡；
  - 不同强度区间应采用差异化策略；
  - 提示单细胞高稀疏数据可借鉴“分层 + 场景化”流程。
- 价值：为 DIA 单细胞数据中的混合缺失机制提供可迁移思路。

### 3.13 二次核查补充（资源分型、稳定入口与边界）

- 本综述当前纳入的 `P1-P12` 全部属于 `论文证据`；没有单独指定 `数据入口`、`模块规范` 或 `资源包`。因此本文件给出的是 imputation 方法池与证据面，而不是当前仓库的稳定 benchmark 输入清单。
- `Wang 2025` 是最直接的 DIA-SCP workflow `论文证据`，但其关联 accession 当前并不适合作为仓库稳定公共输入，因此本综述不应把它扩大解释为数据入口。
- `Krull 2024`、`HarmonizR 2022`、`scplainer 2025` 一类来源在这里应继续被视作“上游减少缺失或任务导向建模的论文证据”，而不是替代 `masked-value benchmark contract` 的模块规范。
- 对当前 ScpTensor 来说，更严格的 benchmark/contract 入口仍应以后续的 `review_masked_imputation_20260312.md`、`review_missingness_20260312.md` 和公共 benchmark 数据综述为准。

## 4. 文献综合测评报告（面向 DIA 单细胞蛋白组）

### 4.1 证据分级

- 直接证据（最高）：P1（DIA 单细胞 workflow benchmark）
- 强相关证据：P2（DIA 实测）、P8（特征匹配减少缺失）、P7/P9/P10/P11（单细胞或跨队列缺失处理）
- 方法学补充：P3/P4/P5/P6/P12（通用或扩展型填充评测）

### 4.2 一致性结论

- 结论 A：不存在跨场景绝对最优填充方法。缺失机制（MCAR/MNAR）、缺失率、细胞异质性决定最优策略。
- 结论 B：在 DIA 单细胞场景，优先减少上游缺失（特征匹配/批次 harmonization）通常比直接“重填充”更稳健。
- 结论 C：当异质性高且缺失率高时，矩阵补全类方法（如 `softImpute`）在部分直接证据中具有竞争力，但最终排序仍受数据集、任务和 masking 协议影响。
- 结论 D：树模型/邻域类方法（RF/KNN/LLS）在 DIA 或通用 proteomics 中具备稳健性，可作为强基线与资源受限备选。
- 结论 E：必须做任务导向评估（聚类、差异蛋白、通路）与敏感性分析，避免“误差下降但生物学结论偏移”。

### 4.3 面向 ScpTensor 的推荐测评方案（文献驱动）

- 目标：比较候选填充方法在 `protein-level matrix` 上的稳定性与生物学保真度。
- 推荐方法池：
  - 基线：`no_impute`、`min/2`、`row_mean`
  - 传统：`KNN`、`LLS`、`RandomForest`、`SVD/IterativeSVD`
  - 矩阵补全：`softImpute`
  - 深度学习（可选扩展）：`PIMMS` 类模型（前提是固定训练/验证 masking 协议并单独报告复现条件）
- 推荐评估指标：
  - 重建误差：NRMSE / MAE（在可控掩码实验中）
  - 定量稳定性：蛋白 CV、相关性保持
  - 生物学一致性：DE 蛋白恢复、通路富集一致性
  - 结构保真：聚类 ARI / silhouette、批次混合度
- 报告要求：
  - 明确缺失机制假设与缺失率分层；
  - 至少报告 3 类下游任务指标；
  - 给出“填充前 vs 后”关键结论一致性；
  - 对高影响结论做方法敏感性分析（至少 3 种填充策略对照）。

### 4.4 实践型结论（用于方法优先级）

- 若数据较完整、群体较同质：优先 `严格过滤 + 轻度/不填充`。
- 若缺失严重且异质性高：可优先把 `softImpute` 纳入重点对照，但仍应与 `RF/KNN` 等强基线联合验证，而不是直接把它定为默认赢家。
- 若计算资源受限或需要快速迭代：`LLS/KNN` 是性价比较高的工程起点。
- 任何场景都不建议仅使用“单一简单填充 + 单一指标”下结论。

## 5. 参考链接

- P1: https://www.nature.com/articles/s41467-025-65174-4 （PubMed 备选：https://pubmed.ncbi.nlm.nih.gov/41271703/）
- P2: https://pubmed.ncbi.nlm.nih.gov/34735399/
- P3: https://pubmed.ncbi.nlm.nih.gov/32526036/ （DOI: 10.1093/nar/gkaa498）
- P4: https://www.nature.com/articles/s41598-021-81279-4
- P5: https://www.nature.com/articles/s41598-022-04938-0
- P6: https://www.nature.com/articles/s41467-024-48711-5 （PubMed 备选：https://pubmed.ncbi.nlm.nih.gov/38926340/）
- P7: https://www.nature.com/articles/s41592-023-01791-5 （PubMed 备选：https://pubmed.ncbi.nlm.nih.gov/36864196/）
- P8: https://www.nature.com/articles/s41467-024-52605-x （PubMed 备选：https://pubmed.ncbi.nlm.nih.gov/39327420/）
- P9: https://www.nature.com/articles/s41467-022-31007-x （PubMed 备选：https://pubmed.ncbi.nlm.nih.gov/35725563/）
- P10: https://genomebiology.biomedcentral.com/articles/10.1186/s13059-025-03713-4
- P11: https://arxiv.org/abs/2312.06354
- P12: https://pubmed.ncbi.nlm.nih.gov/40155995/
