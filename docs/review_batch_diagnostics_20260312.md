# DIA 驱动单细胞蛋白组中的批次诊断指标：优先级文献综述（截至 2026-03-12）

## 1. 研究范围

- 研究问题：ScpTensor 的 `batch diagnostics` 应优先参考哪些一手来源与指标，才能在 `protein-level matrix` 与 `embedding-level` 两类输入上，对批次校正结果做可解释、可复现、不过度简化的评估？
- 目标输出：为以下实现提供方法学依据：
  - `scptensor.integration.diagnostics.compute_batch_asw`
  - `scptensor.integration.diagnostics.compute_batch_mixing_metric`
  - `scptensor.integration.diagnostics.compute_lisi_approx`
  - `scptensor.autoselect.metrics.batch.*`
  - `scptensor.autoselect.evaluators.integration.IntegrationEvaluator`
- 核心边界：
  - 区分 `batch removal` 与 `bio-conservation`
  - 区分 `matrix-level` 与 `embedding-level` 指标
  - 识别 `heuristic proxy` 与标准 benchmark metric 的差别
- 检索日期：`2026-03-12`

### 1.1 关键词优先级（P1 / P2 / P3）

| 优先级 | 目的 | 关键词 |
|---|---|---|
| P1 | 指标核心 | batch correction metrics, batch diagnostics, ASW, kBET, LISI, graph connectivity, PCR |
| P2 | 数据与模态 | single-cell omics, single-cell proteomics, mass spectrometry proteomics |
| P3 | 应用场景 | benchmark, integration evaluation, multimodal, DIA single-cell proteomics |

## 2. 检索策略与筛选标准

### 2.1 Query 梯度（P1 -> P2 -> P3）

1. 先检索 `ASW / LISI / kBET / graph connectivity / PCR` 等指标原始来源与 benchmark 框架。
2. 再限定到 single-cell integration / multi-omics integration / proteomics integration。
3. 最后加入 DIA-SCP workflow benchmark 与 protein-level batch correction 实证。

### 2.2 纳入标准

- 一手来源：期刊官网、PubMed、DOI 页面、官方 benchmark / 方法论文。
- 至少满足以下之一：
  - 定义或系统实现 batch diagnostics 指标
  - 在 benchmark 中用多指标评估 batch correction
  - 能直接迁移到 protein-level proteomics 或 DIA-SCP 解释

### 2.3 排除标准

- 非一手博客、论坛与教程。
- 仅报告整合效果图，但不解释指标含义的工作。

### 2.4 本轮纳入

- 初筛候选：`30+`
- 深读纳入：`7` 篇
- 指标定义/框架来源：`5`
- DIA / proteomics 直接相关实证：`2`

## 3. 逐篇证据摘要（Per-paper Summaries）

说明：本节统一沿用全仓库资源分型。除特别标注外，单篇文献条目默认记为 `论文证据`；官方软件/手册页记为 `模块规范 / 软件文档`；具体 accession 或 dataset page 记为 `数据入口`；可脚本化分发包记为 `资源包`。
共享高频条目的规范元数据统一以 `docs/references/citations.json` 为准；若本文件历史写法与 registry 在作者简称、发布日期、期刊、DOI 或 canonical URL 上不一致，以 registry 为准，本文件仅保留 diagnostics metric 的解释与表示空间边界。

### 3.1 Luecken et al., Nature Methods, 2022

- 题目：Benchmarking atlas-level data integration in single-cell genomics
- 链接：https://www.nature.com/articles/s41592-021-01336-8
- 发布日期：`2022-01-10`
- 目标：建立大规模 single-cell integration benchmark 与统一指标框架。
- 主要贡献：
  - 明确把评估拆成 `batch removal` 与 `bio-conservation` 两大维度。
  - 系统使用 ASW、LISI、graph connectivity、isolated label F1/ASW、NMI/ARI 等。
  - 强调单一总分会掩盖 trade-off。
- 局限：原始场景是 genomics，不是 proteomics，但指标框架高度可迁移。

### 3.2 scib-metrics 官方文档（访问于 2026-03-12）

- 资源类型：`模块规范 / 软件文档`
- 链接：https://scib-metrics.readthedocs.io/
- 目标：把 scIB / integration benchmark 常用指标整理成统一、可扩展、可复现的工程实现与 API 文档。
- 主要贡献：
  - 为 iLISI、kBET、ASW、graph connectivity 等提供统一实现语义与输入要求。
  - 直接暴露 benchmarker / metrics API，有助于把“指标定义”和“工程实现”分开校对。
- 局限：它是 package/documentation 层的一手实现语义来源，不是 DIA-SCP 专项 benchmark 论文。

### 3.3 Buttner et al., Nature Methods, 2019

- 题目：A test metric for assessing single-cell RNA-seq batch correction
- 链接：https://doi.org/10.1038/s41592-018-0254-1
- 发布日期：`2019-01-07`
- 目标：提出 `kBET`。
- 主要贡献：
  - 检验局部邻域的 batch 组成是否显著偏离全局期望。
  - 相比只计算“邻域里有几种 batch”，更接近统计检验视角。
- 局限：对邻居数、样本量、小 batch 和类别不平衡敏感；主要定义在 embedding / neighborhood graph 上。

### 3.4 Korsunsky et al., Nature Methods, 2019

- 题目：Fast, sensitive and accurate integration of single-cell data with Harmony
- 链接：https://www.nature.com/articles/s41592-019-0619-0
- 发布日期：`2019-11-18`
- 目标：提出 Harmony，并以 `LISI` 系列指标评估整合效果。
- 主要贡献：
  - `iLISI` 用于 batch mixing。
  - `cLISI` 用于 biological label conservation。
  - LISI 本质上衡量局部多样性，比简单统计邻域中 batch 种类更有信息量。
- 局限：原始定义依赖局部概率权重，不等同于“固定 kNN + inverse Simpson”的简化实现。

### 3.5 Liu et al., Nature Methods, 2025

- 题目：Multitask benchmarking of single-cell multimodal omics integration methods
- 链接：https://www.nature.com/articles/s41592-025-02856-3
- 发布日期：`2025-10-20`
- 目标：在多任务、多模态 single-cell integration 中比较方法。
- 主要发现：
  - 方法表现依赖 `task + metric + dataset`。
  - 不鼓励用单一总分掩盖不同任务目标。
  - 不同输出类型适用的指标并不一致。
- 局限：不是 MS-proteomics 专项 benchmark。

### 3.6 Wang et al., Nature Communications, 2025

- 题目：Benchmarking informatics workflows for data-independent acquisition single-cell proteomics
- 链接：https://www.nature.com/articles/s41467-025-65174-4
- 发布日期：`2025-11-21`
- 目标：benchmark DIA-SCP workflow。
- 与 diagnostics 相关的直接证据：
  - 不能只看 batch mixing；还要看 ARI、pAUC、F1 等任务性指标。
  - 在 DIA-SCP 中，不同预处理步骤会改变 downstream biological conclusions。
- 局限：更偏 workflow-level benchmark，不是专门的 diagnostics 指标论文。

### 3.7 Zheng et al., Nature Communications, 2025

- 题目：Protein-level batch-effect correction enhances robustness in MS-based proteomics
- 链接：https://www.nature.com/articles/s41467-025-64718-y
- 发布日期：`2025-11-04`
- 目标：比较 precursor / peptide / protein 三个层级上的 batch correction 时机。
- 主要贡献：
  - 说明在 MS 蛋白组里，batch correction 评估要结合数据层级、balanced/confounded 设计与多指标联合判断。
  - 支持“protein-level stable diagnostics 不能只靠一个 neighborhood proxy”。
- 局限：非 single-cell 专项，但对 ScpTensor `protein-level` 主线很重要。

### 3.8 二次核查补充（发布日期、输入空间与指标边界）

- `Luecken et al., Nat Methods (published: 2022-01-10)` 明确把 `batch removal` 与 `bio-conservation` 作为并列目标，而不是单一胜负分数；ScpTensor 文档应继续把两轴分开报告：<https://www.nature.com/articles/s41592-021-01336-8>
- `scib-metrics` 官方文档与 `scIB` 官方文档（accessed: `2026-03-12`）都把指标与输入表示空间绑定，至少要区分 `feature/matrix`、`embedding`、`graph` 三类输入，不能把同一指标跨空间直接等价解释：<https://scib-metrics.readthedocs.io/>；<https://scib.readthedocs.io/en/latest/user_guide.html>
- `Buttner et al., Nat Methods (published: 2019-01-07)` 的 `kBET` 是局部邻域 batch 组成是否偏离全局期望的检验，不是泛化的“mixing score”；因此任何内部 proxy 都不应被表述成 kBET 替代：<https://doi.org/10.1038/s41592-018-0254-1>
- `Korsunsky et al., Nat Methods (published: 2019-11-18)` 的原始 `LISI` 建立在 integrated representation 的局部多样性定义上，不等同于“固定 kNN + inverse Simpson”的简化近似；因此 `compute_lisi_approx` 必须继续明确写为 approximate proxy：<https://www.nature.com/articles/s41592-019-0619-0>
- `Rautenstrauch and Ohler, Nat Biotechnol, 2025` 进一步指出 silhouette 家族指标在 integration benchmarking 中存在系统性偏置风险，因此 `Batch ASW` 只能作为一个维度，而不能升级为单一主分数：<https://www.nature.com/articles/s41587-025-02743-4>
- `Wang 2025` 与 `Zheng 2025` 共同支持 `protein-level endpoint + task-linked interpretation`：在 DIA-SCP 或 MS 蛋白组里，不能只看局部混合度，还要结合下游标签保留、balanced/confounded 设计和最终 protein-level 交付物解释：<https://www.nature.com/articles/s41467-025-65174-4>；<https://www.nature.com/articles/s41467-025-64718-y>

## 4. 指标对比与证据分级

### 4.1 指标定义与适用边界对比表

| 指标 | 核心含义 | 最适用输入 | 优点 | 主要风险 / 局限 |
|---|---|---|---|---|
| Batch ASW | 用 batch label 计算 silhouette；常转成 `1-ASW` 作得分 | embedding / PCA / integrated representation | 快、易解释、实现简单 | 强 representation-sensitive；对高维 raw matrix、batch imbalance、稀有 batch 敏感 |
| iLISI | 局部邻域的 batch 多样性，越高越混合 | embedding / integrated kNN graph | 比“只数 batch 种类”更有信息量 | 原始定义不是固定 kNN 上的简化 Simpson；参数敏感 |
| cLISI | 局部邻域的 biological label purity / conservation | embedding / kNN graph | 可与 iLISI 配对看 trade-off | 需要可靠 biological labels |
| kBET | 检验局部 batch 组成是否偏离全局期望 | embedding / kNN graph | 是检验型指标，不只是描述型指标 | 对样本量、邻居数和类别不平衡敏感；不是 generic mixing proxy |
| Graph connectivity | 同类 biological labels 在图上是否仍连通 | neighbor graph / integrated graph | 补足只看局部几何的缺口 | 依赖 graph 构建方式，不适合 raw matrix 直接算 |
| PCR / variance contribution | 批次解释的全局方差比例 | matrix 或 PCA（global view） | 适合监控批次残留的全局效应 | 不能单独代表 biological conservation |
| ARI / NMI / bio-ASW | biological label preservation | embedding / clustering result | 直接连接 downstream task | 依赖标签质量；无标签场景受限 |

### 4.2 一致结论（facts）

1. 单一指标不够。`batch removal` 与 `bio-conservation` 必须成对报告。
2. 邻域类指标更适合在 PCA / integrated embedding 上计算，不宜在高维 raw protein matrix 上直接解释。
3. `iLISI / kBET / ASW / graph connectivity / PCR` 各自覆盖不同失真维度，互不可替代。
4. DIA-SCP 直接证据支持把 task outcome 一并纳入解释，而不是只报告 mixing 指标。

### 4.3 分歧与解释（inference）

- [推断] 对 ScpTensor 的 stable diagnostics，`batch ASW + iLISI-like + biological conservation metric + optional PCR` 是更稳妥的最小面板。
- [推断] 现有 `compute_batch_mixing_metric()` 更像 `heuristic local mixing proxy`，不应等同于 kBET 或原始 iLISI。

### 4.4 证据强度

- 高：scIB 2022、scib-metrics 官方文档、Wang 2025、Liu 2025
- 中高：Buttner 2019（kBET）、Korsunsky 2019（LISI）
- 中：Zheng 2025（protein-level proteomics timing benchmark）

## 5. 面向 ScpTensor 的实践建议（映射当前实现）

### 5.1 `compute_batch_asw`

- 建议保留。
- 但文档应明确：推荐输入 `PCA-like embedding`，不建议默认在 `protein/raw` 上解释结果。
- 结果标签应固定附带输入空间，例如 `batch_asw@pca`、`batch_asw@embedding`，避免跨表示空间直接比较。
- 语义需要统一：
  - diagnostics 版目前是原始 ASW（低更好）
  - autoselect metrics 版目前是 `1-ASW` 风格分数（高更好）
- 建议文档与 API 命名上显式区分：
  - `batch_asw`
  - `batch_asw_score`

### 5.2 `compute_batch_mixing_metric`

- 当前实现不应表述成标准 benchmark metric。
- 更准确的定位是 `heuristic local mixing proxy`。
- 建议：
  - 在 stable 文档中降级为 exploratory 指标
  - 若继续公开，考虑改名为 `compute_batch_mixing_proxy`
  - 不要与 kBET、iLISI 在文档中并列成“同义替代”

### 5.3 `compute_lisi_approx`

- 方向正确，但必须明确这是 `approximate iLISI proxy`，不是原始 LISI。
- 建议：
  - 文档中写清“基于固定 kNN + inverse Simpson 的近似实现”
  - 若要进入 stable benchmark，优先补成更接近 scIB / scib-metrics 语义的实现
  - 增加 batch 数和 batch 比例不平衡下的解释说明

### 5.4 对 `IntegrationEvaluator` 的最小推荐指标集

- stable integration diagnostics 推荐最小面板：
  - `batch ASW` 或 `batch ASW score`
  - `iLISI` 或高质量 `iLISI-like`
  - 一个 biological conservation 指标：`bio-ASW`、`ARI/NMI` 或有标签时的 task metric
  - 可选全局补充：`PCR / variance explained by batch`
- 若没有 embedding，应先做 `PCA`，再计算这些指标。

## 6. 风险边界

1. 不应把单一 mixing 指标作为 batch correction 优劣的唯一依据。
2. `protein-level raw matrix` 上直接做邻域指标，解释风险高。
3. 在 balanced 与 confounded 设计下，同一指标的含义会变化。
4. 在 DIA-SCP 高缺失场景里，缺失处理会改变局部几何结构，因此 diagnostics 需与 normalization / imputation 联同解释。
5. 近似指标若继续保留，命名与文档必须克制，避免把启发式 proxy 写成标准文献指标。

## 7. Shared Citation Registry Coverage

以下共享高频条目的规范元数据以 `docs/references/citations.json` 为准：

- `luecken2022_natmethods_scib`
- `scib_metrics_docs`
- `buettner2019_natmethods_kbet`
- `korsunsky2019_natmethods_harmony_lisi`
- `liu2025_natmethods_multitask_integration`
- `wang2025_natcom_dia_scp_benchmark`
- `zheng2025_natcom_protein_batch`
- `scib_user_guide`
- `rautenstrauch2025_natbiotechnol_silhouette`
