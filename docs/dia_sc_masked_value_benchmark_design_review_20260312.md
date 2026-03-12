# DIA 驱动单细胞蛋白组预处理中的 masked-value benchmark 设计：优先级文献综述（截至 2026-03-12）

## 1. 研究范围

- 研究问题：ScpTensor 应如何为 `protein-level` 缺失值填充建立可信的 `masked-value recovery` benchmark，而不是只报告总体空值率或单次重建误差？
- 目标输出：为以下模块与 benchmark 目录提供证据化设计建议：
  - `benchmark/imputation`
  - `benchmark/autoselect`
  - `scptensor.impute`
- 核心约束：遵循项目合同，主评测终点应回到 **完整 protein-level quantitative matrix**；若使用 peptide/precursor 数据，也应把 peptide -> protein 聚合作为单独阶段建模。
- 检索日期：`2026-03-12`

### 1.1 关键词优先级（P1 / P2 / P3）

| 优先级 | 目的 | 关键词 |
|---|---|---|
| P1 | 方法核心 | masked-value recovery, imputation benchmark, missing value evaluation, holdout masking |
| P2 | 模态与数据 | proteomics, DIA proteomics, single-cell proteomics |
| P3 | 应用边界 | benchmark design, downstream stability, DE consistency |

## 2. 检索策略与筛选标准

### 2.1 Query 梯度（P1 -> P2 -> P3）

1. 先检索 proteomics 缺失值插补 benchmark 与 masking protocol 的一手研究。
2. 再限定到低输入、DIA 与单细胞相关场景。
3. 最后纳入能直接支持 ScpTensor 任务边界的 downstream-oriented 评估设计。

### 2.2 纳入标准

- 一手来源：期刊官网、Nature/PMC/PubMed/DOI 页面。
- 直接涉及以下至少一项：
  - 显式 masking / holdout 设计
  - 混合缺失机制（`MCAR / MNAR / mixed`）对插补 benchmark 的影响
  - 不只看重建误差，而联合 downstream 指标评估插补效果
  - 能映射到 DIA-sc `protein matrix` 评测的设计原则

### 2.3 排除标准

- 只提出单一插补算法、但没有系统评测协议的论文。
- 无法复核链接或缺少实验设计细节的二手资料。

### 2.4 本轮纳入

- 初筛候选：`20+`
- 深读纳入：`7`
- 其中 benchmark / comparative studies：`6`
- DIA-sc 直接证据：`1`

## 3. 逐篇证据摘要（Per-paper Summaries）

说明：本节统一沿用全仓库资源分型。除特别标注外，单篇文献条目默认记为 `论文证据`；官方软件/手册页记为 `模块规范 / 软件文档`；具体 accession 或 dataset page 记为 `数据入口`；可脚本化分发包记为 `资源包`。

### 3.1 Lazar et al., Journal of Proteome Research, 2016

- 题目：Accounting for the Multiple Natures of Missing Values in Label-Free Quantitative Proteomics Data Sets to Compare Imputation Strategies
- 链接：https://pubmed.ncbi.nlm.nih.gov/26906401/
- 主要发现：
  - 缺失值不是单一分布，而有不同生成机制。
  - 插补方法优劣与缺失机制是否匹配强相关。
- 对 ScpTensor 的意义：
  - masking 设计不能只做纯随机 holdout。
  - 评测协议应显式区分 `MCAR-like`、`MNAR-like` 与混合机制。

### 3.2 Jin et al., Scientific Reports, 2021

- 题目：A comparative study of evaluating missing value imputation methods in label-free proteomics
- 链接：https://www.nature.com/articles/s41598-021-81279-4
- 主要发现：
  - 在高质量基准数据上系统注入不同缺失比例与 `MNAR` 比例，可以稳定区分方法性能。
  - 论文采用 `MV rate × MNAR rate` 网格化设计，并多次重复评测。
- 对 ScpTensor 的意义：
  - 应把 `overall missing rate` 与 `MNAR fraction` 分开建模。
  - 单次 masking 结果不稳，必须多随机种子重复。

### 3.3 Shen et al., Scientific Reports, 2022

- 题目：Comparative assessment and novel strategy on methods for imputing proteomics data
- 链接：https://www.nature.com/articles/s41598-022-04938-0
- 主要发现：
  - 论文区分两类 benchmark 设定，其中一类是在“本就含真实缺失的数据”上额外做 masked set-aside，以更贴近真实分布。
  - 对不同缺失结构，不同方法的最优性并不稳定。
- 对 ScpTensor 的意义：
  - benchmark 不应只在“完全观测真值矩阵”上进行。
  - 更合理的协议是保留真实缺失结构，再额外遮盖一部分可观测值做 recovery。

### 3.4 Harris et al., Journal of Proteome Research, 2023

- 题目：Evaluating Proteomics Imputation Methods with Improved Criteria
- 链接：https://pubmed.ncbi.nlm.nih.gov/37861703/
- 主要发现：
  - 在 `5` 套蛋白组数据上比较 `10` 种方法，并包含 `2` 套 `DIA` 数据。
  - 仅看重建误差不足以判断插补是否真的有利于 downstream proteomics analysis。
  - 文中把差异分析、可定量 analyte 数量与定量下限等指标纳入综合评价。
- 对 ScpTensor 的意义：
  - AutoSelect 或 benchmark 排名不应只靠 `NRMSE/MAE`。
  - 至少要并报 downstream DE / feature retention 一类任务指标。

### 3.5 Webel et al., Nature Communications, 2024

- 题目：Imputation of label-free quantitative mass spectrometry-based proteomics data using self-supervised deep learning
- 链接：https://www.nature.com/articles/s41467-024-48711-5
- 主要发现：
  - 摘要层明确报告对已知信号随机移除 `20%` 再评估，并在 precursor / peptide / protein 多层级比较恢复质量。
  - 自监督框架强调“masking protocol 本身就是训练/评估合同的一部分”。
- 对 ScpTensor 的意义：
  - 若未来要做 learned imputation 或 AutoSelect 训练，必须把 masking protocol 文档化。
  - 主榜单可在 protein 层，压力测试可扩展到 precursor/peptide 层。

### 3.6 NAguideR, Bioinformatics, 2020

- 题目：NAguideR: performing and prioritizing missing value imputations for consistent bottom-up proteomic analyses
- 链接：https://pmc.ncbi.nlm.nih.gov/articles/PMC7641313/
- 主要发现：
  - 系统比较 proteomics 插补方法时，`NRMSE`、rank correlation、方法优先级综合报告都很关键。
  - 缺失模式与数据结构不同，会显著改变方法排序。
- 对 ScpTensor 的意义：
  - benchmark 输出应保留“原始指标表 + 综合分数表”，避免只留最终榜单。
  - 推荐保留 `Spearman` 一类 rank-based 指标，而不只看绝对误差。

### 3.7 Wang et al., Nature Communications, 2025

- 题目：Benchmarking informatics workflows for data-independent acquisition single-cell proteomics
- 链接：https://www.nature.com/articles/s41467-025-65174-4
- 主要发现：
  - 在 DIA 单细胞工作流中，缺失处理必须与 normalization、batch correction、DE 分析联动评估。
  - 论文使用 `ARI`、`pAUC`、`F1` 等 downstream 指标，而不只看矩阵填补误差。
- 对 ScpTensor 的意义：
  - 这是 masked-value benchmark 设计里最直接的 DIA-sc 证据。
  - 单细胞场景中，imputation 排名必须回到 cluster / DE / biology preservation，而不是只看插补值本身。

### 3.8 二次核查补充（资源分型、稳定入口与场景边界）

- 本综述纳入的 `Lazar 2016`、`Jin 2021`、`Shen 2022`、`Harris 2023`、`Webel 2024`、`NAguideR 2020`、`Wang 2025` 全部属于 `论文证据`；它们约束的是 masking protocol 与评价维度，而不是公共数据入口。
- `Wang 2025` 是当前最直接的 DIA-SCP workflow/masked-value 设计证据，但应继续定位为 `task-design evidence`，不应替代稳定公开 benchmark 数据入口：<https://www.nature.com/articles/s41467-025-65174-4>
- `NAguideR` 与 `Webel 2024` 提供的是方法比较与评估协议证据，不是 `模块规范` 或 `资源包`；因此本综述不应把它们写成可直接复用的 benchmark service / package contract。
- 本综述本身不承担 `数据入口` 与 `资源包` 选型；这些角色应回到公共 benchmark 数据综述和后续 benchmark manifest 中统一管理。

## 4. 横向比较与证据分级

### 4.1 一致结论（facts）

1. `masked-value benchmark` 不能只做纯随机 holdout；至少需要区分 `MCAR-like`、`MNAR-like` 或混合机制。
2. 单次 masking 不稳，必须按多个随机种子重复。
3. 在真实 proteomics 数据上，最可信的评测设计通常是：
   - 保留原始缺失结构
   - 只在原本可观测的位置额外 holdout
   - 同时报告 recovery 指标和 downstream 指标
4. 仅看 `NRMSE / MAE` 会高估某些方法的实际收益；DE、一致性、保留 analyte 数量等任务指标必须补充。
5. 在 DIA 单细胞场景中，插补步骤不能脱离 normalization、batch correction 和 DE 任务独立解释。

### 4.2 推荐的评测协议（面向 ScpTensor）

| 组件 | 推荐做法 | 理由 |
|---|---|---|
| 真值池 | 选高完整度 protein features，或用 spike-in / 已知比例样本 | 减少“伪真值”偏差 |
| masking 机制 | 至少含 `mcar`、`mixed_mnar`；有条件时加 intensity-stratified `mnar` | 更接近 proteomics 缺失结构 |
| masking 比例 | `0.1 / 0.2 / 0.3` 为主，必要时扩到 `0.5` 压力测试 | 与当前 benchmark README 和文献区间一致 |
| 重复次数 | `>=10` 个随机种子更稳；资源有限时至少 `>=5` | 降低抽样偶然性 |
| recovery 指标 | `NRMSE`、`MAE`、`Spearman`、holdout coverage | 兼顾绝对误差与排序稳定性 |
| downstream 指标 | `DE pAUC/F1`、cluster stability、retained proteins | 防止“重建好但任务变差” |
| baseline | 必须含 `no-imputation` | 防止默认把插补当成必选步骤 |

### 4.3 分歧与解释（inference）

- [推断] 由于 ScpTensor 的主目标是 protein-level 完整矩阵，主榜单应在 protein 层比较；precursor/peptide 层更适合作为 aggregation path 的压力测试，而不是最终胜负标准。
- [推断] 如果未来引入深度学习插补，必须把 masking protocol、随机种子、失败案例和训练/验证分层一起固化，否则 benchmark 无法复现。
- [推断] 对当前 ScpTensor 文档来说，`0.1 / 0.3 / 0.5, repeats = 1` 更适合“轻量回归默认”，而 `0.1 / 0.2 / 0.3 + pressure 0.5, repeats >= 5` 更接近 literature-style 主协议。

### 4.4 证据强度

- 高：Lazar 2016、Jin 2021、Shen 2022、Wang 2025
- 中高：Harris 2023、Webel 2024、NAguideR 2020

## 5. 面向 ScpTensor 的实践建议

### 5.1 `benchmark/imputation` 应升级为“双层指标”

- 保留恢复误差：
  - `nrmse`
  - `mae`
  - `spearman_r`
- 同时保留任务导向指标：
  - `de_pAUC`
  - `de_f1`
  - cluster preservation
  - retained proteins

### 5.2 masking 设计要从“比例”升级为“结构化协议”

- 当前 README 已有 `mcar + mixed_mnar` 和不同 holdout 比例，这是正确方向。
- 建议下一步再显式加入：
  - intensity 分层 masking
  - missing-rate 分箱 masking
  - 种子重复汇总

### 5.3 明确两条评测赛道

1. `protein-direct`
   - 直接在 protein matrix 上做 masking / recovery
   - 作为主榜单
2. `precursor-to-protein`
   - 在 precursor/peptide 层做 masking
   - 经 `scptensor.aggregation` 后回到 protein 层评分
   - 作为 aggregation robustness 附榜

### 5.4 `no-imputation` 必须保留

- 结合 Wang 2025 与 Harris 2023，某些 downstream 任务下“保守不插补”并不一定差于激进插补。
- 因此 `none` 不是陪跑基线，而是必要比较对象。

## 6. 风险边界

1. 绝大多数插补 benchmark 仍来自 bulk 或低输入 proteomics，而不是严格的 DIA 单细胞；对 DIA-sc 的外推必须明确标注。
2. 如果“真值池”来自高度完整特征，可能会高估方法在极稀疏区域的能力。
3. 若只用 synthetic missingness，而不保留真实缺失结构，会低估方法在真实数据上的偏差。
4. 某些深度学习方法对 masking protocol 高度敏感；协议不固定时，方法排名很容易漂移。

## 7. 对后续实现/文档的优先建议

1. 在 `benchmark/imputation/README.md` 中把 `masked-value recovery protocol` 单独列成一节。
2. 在 `benchmark/autoselect` 中增加“只对 protein-level 主榜单计分、aggregation 路径单独附榜”的说明。
3. 在 imputation benchmark 输出中新增 `seed_summary.csv` 或等价汇总，避免只留单次运行结果。

## 8. 参考文献（含链接）

1. Lazar et al., 2016, Journal of Proteome Research
   https://pubmed.ncbi.nlm.nih.gov/26906401/

2. Jin et al., 2021, Scientific Reports
   https://www.nature.com/articles/s41598-021-81279-4

3. Shen et al., 2022, Scientific Reports
   https://www.nature.com/articles/s41598-022-04938-0

4. Harris et al., 2023, Journal of Proteome Research
   https://pubmed.ncbi.nlm.nih.gov/37861703/

5. Webel et al., 2024, Nature Communications
   https://www.nature.com/articles/s41467-024-48711-5

6. NAguideR, 2020, Bioinformatics
   https://pmc.ncbi.nlm.nih.gov/articles/PMC7641313/

7. Wang et al., 2025, Nature Communications
   https://www.nature.com/articles/s41467-025-65174-4
