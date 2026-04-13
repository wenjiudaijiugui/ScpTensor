# DIA 单细胞蛋白组 AutoSelect 打分框架：优先级文献综述（截至 2026-03-12）

## 1. 研究范围

- 研究问题：ScpTensor `AutoSelect` 的 `scoring / ranking / reporting` framework 应参考哪些一手 benchmark 与 reporting 来源，来支撑多指标评分、权重设置、重复运行、不确定性报告，以及速度与质量折中？
- 目标输出：为以下实现提供方法学依据：
  - `scptensor.autoselect.core`
  - `scptensor.autoselect.strategy`
  - `scptensor.autoselect.evaluators.*`
  - `scptensor.autoselect.metrics.*`
  - `scptensor.autoselect.report`
- 重点问题：
  - 是否应保留单一 `overall_score`
  - `overall_score` 与 `selection_score` 是否应区分
  - 多 stage 是否应共用同一权重逻辑
  - `runtime` 应混入总分还是独立报告
  - `n_repeats` 与 interval 应如何表述
- 检索日期：`2026-03-12`

### 1.1 关键词优先级（P1 / P2 / P3）

| 优先级 | 目的 | 关键词 |
|---|---|---|
| P1 | 框架核心 | benchmark, evaluation metrics, ranking, uncertainty, reproducibility, reporting, runtime trade-off |
| P2 | 数据与模态 | single-cell integration, multi-omics integration, DIA single-cell proteomics, omics computational tools |
| P3 | 应用边界 | AutoML, method selection, reporting standards, software benchmarking |

## 2. 检索策略与筛选标准

### 2.1 Query 梯度（P1 -> P2 -> P3）

1. 先检索 method benchmarking、multi-metric evaluation、reporting、uncertainty 与 reproducibility 标准。
2. 再限定到 single-cell integration、multi-omics integration 与 DIA-SCP workflow benchmark。
3. 最后补充与 AutoML / method selection 相关、但必须是一手 benchmark / framework 来源。

### 2.2 纳入标准

- 一手来源：期刊官网、DOI 页面、官方 benchmark / framework 论文。
- 至少满足以下之一：
  - 提供多指标 benchmark 框架
  - 讨论 ranking / weighting / reporting / reproducibility
  - 与 DIA-SCP 或 single-cell integration 的方法选择直接相关

### 2.3 排除标准

- 泛泛 AutoML 教程或非生命科学 benchmark 博客。
- 只给 leaderboard，不解释 scoring/reporting 原理的材料。

### 2.4 本轮纳入

- 初筛候选：`25+`
- 深读纳入：`7` 篇

## 3. 逐篇证据摘要（Per-paper Summaries）

说明：本节统一沿用全仓库资源分型。除特别标注外，单篇文献条目默认记为 `论文证据`；官方软件/手册页记为 `模块规范 / 软件文档`；具体 accession 或 dataset page 记为 `数据入口`；可脚本化分发包记为 `资源包`。
共享高频条目的规范元数据统一以 `docs/references/citations.json` 为准；若本文件历史写法与 registry 在标题、作者简称、发布日期、DOI 或 canonical URL 上不一致，以 registry 为准，本文件仅保留 AutoSelect 相关解释。

### 3.1 Wang et al., Nature Communications, 2025

- 题目：Benchmarking informatics workflows for data-independent acquisition single-cell proteomics
- 链接：https://www.nature.com/articles/s41467-025-65174-4
- 目标：系统 benchmark DIA-SCP 全流程。
- 对 AutoSelect 的直接启示：
  - 不存在 one-size-fits-all workflow。
  - 不同步骤组合会给出相同或相反的 biological conclusions。
  - 更合理的交付是“一组高表现 workflow 与解释”，而不是绝对唯一赢家。
- 局限：结论依赖数据集、异质性和任务定义。

### 3.2 Luecken et al., Nature Methods, 2022

- 题目：Benchmarking atlas-level data integration in single-cell genomics
- 链接：https://www.nature.com/articles/s41592-021-01336-8
- 目标：建立 single-cell integration benchmark 与统一指标框架。
- 主要贡献：
  - 明确 `batch removal` 与 `bio-conservation` 是并列维度。
  - 给出 `S_overall = 0.6 * S_bio + 0.4 * S_batch` 的聚合示例，强调综合分依赖权重设定。
  - 不鼓励只报告单一综合分。
  - 强调数据集、任务与指标共同决定方法排名。
- 局限：genomics 场景，不是 proteomics。

### 3.3 scib-metrics 官方文档与基准示例（访问于 2026-03-12）

- 资源类型：`模块规范 / 软件文档`
- 链接：https://scib-metrics.readthedocs.io/
- 补充示例：https://scib-metrics.readthedocs.io/en/stable/notebooks/large_scale.html
- 目标：为 scIB 类 benchmark 提供统一指标实现、标准化 API 和可扩展基准接口。
- 主要贡献：
  - benchmark 需要稳定实现、统一 API、可扩展计算与标准化输入/输出。
  - “如何计算指标”与“如何汇总指标”应分层处理。
  - 官方 `Benchmarker` API 明确把 `bio_conservation_metrics` 与 `batch_correction_metrics` 分开，再进行汇总制表。
- 局限：它是实现规范与文档入口，不是独立 peer-reviewed scoring 论文，因此更适合作为 `模块规范` 证据。

### 3.4 Liu et al., Nature Methods, 2025

- 题目：Multitask benchmarking of single-cell multimodal omics integration methods
- 链接：https://www.nature.com/articles/s41592-025-02856-3
- 目标：在多任务、多模态场景下 benchmark integration methods。
- 主要发现：
  - 方法表现依赖 `task + metric + dataset`。
  - overall rank 可以用于导航，但不能替代 task-specific metric battery。
  - 多个方法在特定 output types 上会失败或不适配，不能把跨任务统一总分当作唯一结论。
  - 某些指标不适合某些 output types。
- 局限：不是 proteomics 专项，但对多 stage AutoSelect 最具迁移价值。

### 3.5 Weber et al., Nature Communications, 2019

- 题目：Essential guidelines for computational method benchmarking
- 链接：https://www.nature.com/articles/s41467-019-09406-4
- 目标：总结 computational omics tool benchmarking 的基本原则。
- 主要贡献：
  - benchmark 应独立、透明、标准化。
  - 不应只给赢家；还应给适用场景、失败模式与外部验证。
  - 可扩展性与资源成本应作为独立维度报告。
- 局限：不是 single-cell 或 proteomics 专项。

### 3.6 Nature Methods, 2021

- 题目：Reproducibility standards for machine learning in the life sciences
- 链接：https://www.nature.com/articles/s41592-021-01256-7
- 目标：为生命科学中的 ML 研究建立 reproducibility / reporting 标准。
- 主要贡献：
  - 结果报告应包含代码、数据切分、依赖、随机性控制、自动化重现实验条件。
  - reproducibility 本身是方法评估的一部分。
- 局限：不提供具体 scoring formula。

### 3.7 Nature Methods, 2023

- 题目：Initial recommendations for performing, benchmarking and reporting single-cell proteomics experiments
- 链接：https://www.nature.com/articles/s41592-023-01785-3
- 目标：为 single-cell proteomics 的 benchmark 与 reporting 提供社区建议。
- 主要贡献：
  - 强调 metadata、cross-validation、分析路径透明度与报告完整性。
  - 只给结果图而不报告分析路径是不够的。
- 局限：更偏实验与整体 reporting，而不是 scoring formula。

### 3.8 二次核查补充（发布日期与评分语义）

- `Luecken et al., Nat Methods (published: 2022-01-10)` 给出的综合分公式本质是“按任务目标设权重后的聚合器”，不是方法本体不变量：<https://www.nature.com/articles/s41592-021-01336-8>
- `Liu et al., Nat Methods (published: 2025-10-20)` 明确指出“没有一种方法在所有 output types 上都最优”，因此 AutoSelect 的策略层评分必须与任务语境绑定：<https://www.nature.com/articles/s41592-025-02856-3>
- `scib-metrics` 官方 `Benchmarker` API 在接口层就分开 `bio_conservation_metrics` 与 `batch_correction_metrics`，随后再汇总制表，支持 `overall_score` 与 `selection_score` 分层报告：<https://scib-metrics.readthedocs.io/en/stable/notebooks/large_scale.html>

## 4. 横向比较（Comparative Assessment）

### 4.1 多指标评分 / 权重 / 重复运行 / 不确定性 / 速度折中

| 来源 | 多指标评分 | 权重处理 | 重复运行 / 随机性 | 不确定性报告 | 速度 / 资源 |
|---|---|---|---|---|---|
| Wang 2025 | 强，多步骤组合评分 | 有组合排名，但不主张单指标决定 | workflow 组合比较，强调数据依赖 | 强调不同 workflow 可得不同结论 | 不把 runtime 作为主排序核心 |
| scIB 2022 | 非常强，分 `batch removal` / `bio-conservation` | 支持聚合，但保留分项分数 | 多任务多数据集比单次运行更重要 | 重点是跨任务稳健性 | scalability 是独立考量 |
| scib-metrics docs | 强，统一 metrics 实现 | 不规定单一权重 | 强调标准化实现与可复现 benchmark | 更偏实现可复现 | 强调大规模可计算 |
| Liu 2025 | 非常强，task-specific metric battery | overall rank 仅作导航 | 强调 task / output / dataset 差异 | 强调指标局限必须报告 | 输出类型与成本差异要说明 |
| Weber 2019 | 强，强调 benchmark design | 不鼓励黑箱式单总分 | 强调独立 benchmark 与多验证集 | 强调失败模式与场景边界 | runtime / scalability 应独立报告 |
| Nat Methods 2021 | 间接支持 | 不谈具体权重 | 强调随机种子、环境和自动化 | 强调完整重现实验条件 | 资源与环境是 reproducibility 一部分 |
| SCP recommendations 2023 | 间接支持 | 不谈公式 | 强调复现与 metadata | 强调报告完整性 | 不以速度为主，但要求透明 |

### 4.2 一致结论（facts）

1. `overall_score` 可以存在，但不能取代分项报告。
2. `overall_score` 的聚合权重属于策略参数；在不同任务上应允许不同权重，不应被表述为“普适真理”。
3. 权重不应被包装成“方法学真理”；它们更接近场景化策略。
4. 多任务 benchmark 不支持用单一默认权重横跨所有 stage。
5. runtime / scalability 最好作为独立维度可见，而不是只隐式混入总分。
6. 对随机性较强的方法，应保留 repeated runs 与 interval reporting。

### 4.3 分歧与解释（inference）

- [推断] ScpTensor 当前 `quality / balanced / speed` 预设是合理的工程设计，但应被明确定义为 `selection policy`，而不是“生物信息学优越性”。
- [推断] `normalize / impute / integrate / reduce / cluster` 应维持各自的 metric family，而不是追求一个跨 stage 通用综合分。

### 4.4 证据强度

- 高：Wang 2025、scIB 2022、scib-metrics 官方文档、Liu 2025
- 中高：Weber 2019
- 中：Nat Methods 2021、Nat Methods 2023 SCP recommendations

## 5. 面向 ScpTensor AutoSelect 的实践建议

### 5.1 分离 `overall_score` 与 `selection_score`

- 建议继续保留两层：
  - `overall_score`：纯质量分
  - `selection_score`：质量 + runtime 的策略化排序分
- 文档中应明确说明二者不同，否则用户容易把速度偏好误当作生物学优越性。

### 5.2 权重按 stage family 固定，不追求“全局万能默认”

- 更合理的结构：
  - `normalize / impute`：technical quality + signal preservation 为主
  - `integrate`：batch removal + bio-conservation 双轴并列
  - `reduce / cluster`：structure preservation + clustering quality 为主
- 这更符合 scIB 与 Liu 2025 的证据。

### 5.3 `quality / balanced / speed` 预设保留，但明确属于策略层

- 建议在报告中把三类预设定义为：
  - `quality`：科学比较优先
  - `balanced`：默认工程策略
  - `speed`：快速迭代策略
- 它们改变的是排序偏好，不应改变原始质量指标解释。

### 5.4 `n_repeats` 与 interval 的表述应克制

- 当前做法可保留，但建议文档写成 `empirical repeat interval`。
- 不宜把它包装为严格统计意义上的置信区间。
- 对随机性高的 `UMAP / t-SNE / clustering`，比对 deterministic methods 更需要 repeated runs。

### 5.5 runtime 应独立可见

- 建议长期同时保留：
  - `execution_time`
  - `overall_score`
  - `selection_score`
- 让用户直接看到“慢但质量高”和“快但略差”的真实 trade-off。

### 5.6 失败率与方法合同是一等公民

- 建议 `StageReport` 固定强调：
  - `success_rate`
  - `error summary`
  - `method_contract`
  - `input/output assay/layer/obs key`
- 这比只罗列成功方法更符合 benchmark/reporting 一手来源的要求。

## 6. 风险边界

1. 不应把跨 stage 的 `overall_score` 直接横向比较。`normalize` 与 `integrate` 不在同一任务空间。
2. 不应把 `selection_score` 当作“最优 biological conclusion”的证据；它只是当前策略下的排序函数。
3. graph outputs、embedding outputs、matrix outputs 的指标适用性不同，不应混用。
4. 无标签场景下 `bio-conservation` 指标会退化，报告中应明确“未计算”，而不是默认为 `0`。

## 7. Shared Citation Registry Coverage

以下共享高频条目的规范元数据以 `docs/references/citations.json` 为准：

- `wang2025_natcom_dia_scp_benchmark`
- `luecken2022_natmethods_scib`
- `liu2025_natmethods_multitask_integration`
- `weber2019_natcom_benchmarking_guidelines`
- `heil2021_natmethods_ml_reproducibility`
- `gatto2023_natmethods_scp_recommendations`
- `scib_metrics_docs`

本文件额外保留的实现性补充入口：

1. scib-metrics Benchmarker large-scale example（documentation, accessed 2026-03-12）
   https://scib-metrics.readthedocs.io/en/stable/notebooks/large_scale.html
