# 基于单细胞蛋白组（SCP）数据处理的去批次方法评估：优先级文献综述（2026-03-05）

## 1. 研究范围（Scope）

**核心问题**

在单细胞蛋白组（尤其是 DIA 驱动的 SCP）预处理中，去批次（batch correction）方法如何被评估？常用方法、评估指标、公开数据与可复现资源有哪些？

**关键词优先级（Priority Keywords）**

| Priority | 关键词 |
| --- | --- |
| P1（方法核心） | batch effect correction, ComBat, limma, Scanorama, Harmony, MNN, benchmark, evaluation metrics |
| P2（数据/模态） | single-cell proteomics, DIA, plexDIA, mass spectrometry proteomics |
| P3（应用域） | preprocessing workflow, data integration, confounded design, reproducibility |

**检索日期**：2026-03-05
**检索平台**：Nature / Genome Biology / PubMed / Oxford Academic / ProteomeXchange / 方法原始论文页面

---

## 2. 检索策略与纳入标准

### 2.1 Query 梯度（按 P1→P2→P3）

1. 先检索“去批次方法 + benchmark/evaluation”
2. 再限定到“single-cell proteomics / DIA / mass spectrometry proteomics”
3. 最后叠加“workflow / confounded / preprocessing / reproducibility”

### 2.2 纳入标准

- 一手来源（期刊论文页面、PubMed、DOI 页面、官方数据仓库）。
- 明确涉及以下至少一项：
  - SCP 或 MS 蛋白组去批次方法对比；
  - 去批次评估指标体系；
  - 可复用数据/代码资源。

### 2.3 排除标准

- 纯教程、论坛讨论、二手博客（仅用于发现线索，不作为证据）。
- 与 SCP/DIA 场景关联弱、且无可迁移评估价值的研究。

### 2.4 筛选结果（本轮）

- 初筛候选：约 30+ 条
- 深读纳入：10 篇（含 7 篇直接相关 + 3 篇指标/方法框架支撑）

---

## 3. 逐篇证据摘要（Per-paper Summaries）

说明：本节统一沿用全仓库资源分型。除特别标注外，单篇文献条目默认记为 `论文证据`；官方软件/手册页记为 `模块规范 / 软件文档`；具体 accession 或 dataset page 记为 `数据入口`；可脚本化分发包记为 `资源包`。
共享高频条目的规范元数据统一以 `docs/references/citations.json` 为准；若本文件历史写法与 registry 在作者简称、发布日期、期刊、DOI 或 canonical URL 上不一致，以 registry 为准，本文件仅保留 batch-correction 任务解释、资源分型和场景边界。

## 3.1 Wang et al., Nat Commun 2025（DIA SCP workflow benchmark）

- 题目：Benchmarking informatics workflows for data-independent acquisition single-cell proteomics
- 链接：https://doi.org/10.1038/s41467-025-65174-4
- 目标：系统评估 DIA-SCP 全流程方法组合（含缺失处理、归一化、去批次、统计检验）。
- 去批次方法：NoBC、limma、ComBat-P、ComBat-NP、Scanorama。
- 关键评估：ARI、pAUC、F1 及组合排名；并进行方法交互贡献分析（SHAP）。
- 关键发现：
  - 不做去批次时 ARI 明显偏低；加入去批次可显著改善聚类一致性。
  - 在其任务中，ComBat/limma 常出现在高表现组合内；Scanorama 贡献相对不稳定。
- 数据/代码：论文提供 Source data 与代码仓库；其关联 accession `PXD056832` 可解析到 ProteomeXchange 页面，但该页面在 `2026-03-12` 仍显示 `reserved / not yet released`，因此当前不应被表述为“稳定可直接复用”的公开数据入口。
- 局限：评估任务与数据构成对结论有依赖，方法优劣存在数据集特异性。
- P1/P2/P3 相关性：2 / 2 / 2（高）

## 3.2 Ctortecka et al., Nat Commun 2024（SCP 实战中 limma 去批次）

- 题目：Automated single-cell proteomics providing sufficient proteome depth to study complex biology beyond cell type classifications
- 链接：https://doi.org/10.1038/s41467-024-49651-w
- 目标：展示自动化 SCP 工作流在复杂生物学问题中的可用性。
- 去批次实践：SCnorm + log10 后使用 limma::removeBatchEffect，并将处理变量（drug effect）显式保留在模型中。
- 评估：通过 PCA 回归估计技术批次贡献，并比较批次校正前后。
- 关键发现：可在保留处理效应前提下降低技术方差。
- 数据入口：MassIVE `MSV000093867`。
- 局限：不是“多方法 benchmark”，更偏 pipeline 实战验证。
- P1/P2/P3：1 / 2 / 2（中高）

## 3.3 Vanderaa et al., Genome Biology 2025（scplainer 批次校正 benchmark）

- 题目：scplainer: using linear models to understand mass spectrometry-based single-cell proteomics data
- 链接：https://doi.org/10.1186/s13059-025-03713-4
- 目标：提出可解释线性模型框架，并对比 SCP 场景下批次校正策略。
- 对比方法：None、HarmonizR-ComBat、limma、scplainer、以及 Leduc 工作流结果。
- 评估指标：ARI、NMI、PS、ASW；区分 biological separation 与 technical mixing 两类评价。
- 关键发现：
  - 各方法相对无校正均有提升；
  - HarmonizR-ComBat 整体增益较小，但在部分批次混合方面有优势；
  - limma 与 scplainer 表现接近（线性模型家族）。
- 资源包/代码：`scpdata`（Bioconductor）+ `scp` 包 + GitHub/Zenodo/Docker。
- 局限：主 benchmark 重点仍在其选定数据与任务设置。
- P1/P2/P3：2 / 2 / 2（高）

## 3.4 Schlumbohm et al., Nat Commun 2022（HarmonizR）

- 题目：HarmonizR enables data harmonization across independent proteomic datasets with appropriate handling of missing values
- 链接：https://doi.org/10.1038/s41467-022-31007-x
- 目标：解决蛋白组高缺失下 ComBat/limma 难直接应用的问题。
- 方法要点：按缺失模式分解子矩阵后再调用 ComBat/limma，尽量避免“先全局补值再校正”的误差传递。
- 关键发现：在多批次高缺失蛋白组中，相比先补值后校正策略，HarmonizR 路径更稳健，信息损失更低。
- 局限：以蛋白组整合为主，非 SCP 专属；但对 SCP 高缺失特性有直接借鉴价值。
- P1/P2/P3：2 / 1 / 2（中高）

## 3.5 Zheng et al., Nat Commun 2025（蛋白组层级去批次时机 benchmark）

- 题目：Protein-level batch-effect correction enhances robustness in MS-based proteomics
- 链接：https://doi.org/10.1038/s41467-025-64718-y
- 目标：比较 precursor/peptide/protein 三个层级进行去批次的效果。
- 设计亮点：
  - balanced 与 confounded 两种场景；
  - 多 quantification method（MaxLFQ/TopPep3/iBAQ）与多 BECA 联合评估。
- 关键发现：在其设计下，晚阶段（尤其 protein-level）策略总体更稳健。
- 数据：Figshare 提供统计数据与处理后数据。
- 与 SCP 关系：非单细胞专属，但“confounded 设计 + 多指标评估框架”对 SCP benchmark 设计高度可迁移。
- P1/P2/P3：2 / 1 / 2（中高）

## 3.6 Koca & Sevilgen, Proteomics 2024（SCPRO-HI）

- 题目：Integration of single-cell proteomic datasets through distinctive proteins in cell clusters
- 链接：https://pubmed.ncbi.nlm.nih.gov/38135888/
- 目标：提出面向低维单细胞蛋白组整合的算法（SCPRO-HI）。
- 评估：在模拟与真实数据上，与对照方法比较，报告轮廓系数与特征保留表现。
- 价值：补充“非线性/深度模型在 SCP 整合中的路线”。
- 局限：任务与数据设置较特定，且与 DIA 蛋白矩阵预处理场景并不完全一致。
- P1/P2/P3：1 / 2 / 1（中）

## 3.7 Liu et al., Nat Methods 2025（多模态单细胞整合 benchmark）

- 题目：Multitask benchmarking of single-cell multimodal omics integration methods
- 链接：https://doi.org/10.1038/s41592-025-02856-3
- 目标：建立多任务、多模态单细胞整合 benchmark 框架。
- 价值：提供“任务分解 + 指标面板 + 评分逻辑”的通用评测范式，可迁移到 SCP。
- 局限：非 MS-SCP 专项 benchmark。
- P1/P2/P3：1 / 1 / 2（中）

## 3.8 Luecken et al., Nat Methods 2022（scIB）

- 题目：Benchmarking atlas-level data integration in single-cell genomics
- 链接：https://doi.org/10.1038/s41592-021-01336-8
- 目标：系统比较单细胞整合方法并建立指标体系。
- 贡献：明确 batch removal 与 bio-conservation 的权衡，给出 ASW/LISI 等指标定义与聚合思路。
- 与 SCP 关系：跨模态指标框架来源，常被用于 SCP benchmark 指标借鉴。
- 局限：原始任务是单细胞基因组学，不是蛋白组。
- P1/P2/P3：1 / 0 / 2（中）

## 3.9 ComBat 原始论文（方法学来源）

- 题目：Adjusting batch effects in microarray expression data using empirical Bayes methods
- 链接：https://pubmed.ncbi.nlm.nih.gov/16632515/
- 价值：ComBat 参数化/非参数化经验贝叶斯框架来源。

## 3.10 limma 框架论文（方法学来源）

- 题目：limma powers differential expression analyses for RNA-sequencing and microarray studies
- 链接：https://pubmed.ncbi.nlm.nih.gov/25605792/
- 价值：线性模型 + 经验贝叶斯框架来源（removeBatchEffect 在生态中广泛使用）。

## 3.11 二次核查补充（资源分型、稳定入口与场景边界）

- `Wang 2025`、`Ctortecka 2024`、`scplainer 2025`、`HarmonizR 2022`、`Zheng 2025`、`SCPRO-HI 2024`、`Liu 2025`、`scIB 2022`、`ComBat`、`limma` 在本综述里统一属于 `论文证据`；它们约束的是 batch-correction 方法、指标和场景边界。
- `MSV000093867` 应单独归类为 `数据入口`，不能继续用论文页替代数据页：<https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?accession=MSV000093867>
- `scpdata` 在本综述里应归类为 `资源包`，而不是数据集或论文页；其稳定入口应写成 Bioconductor 页面：<https://bioconductor.org/packages/release/data/experiment/html/scpdata.html>
- `ProteoBench DIA single-cell module` 应归类为 `模块规范`，它约束任务与评估接口，不是单个 dataset：<https://proteobench.readthedocs.io/en/stable/available-modules/active-modules/9-quant-lfq-ion-dia-singlecell/>
- `Wang 2025` 关联的 `PXD056832` 当前仍不应写成稳定公开数据入口，因此本综述中它继续只承担 `task-design evidence` 角色。

---

## 4. 横向比较（Comparative Assessment）

## 4.1 一致结论（高一致性）

1. **去批次是 SCP 预处理中不可忽略环节**：NoBC 通常显著劣于至少一种校正方法。
2. **线性模型家族（limma / ComBat）仍是主力基线**：在多个 SCP/MS 蛋白组研究中持续出现。
3. **指标必须成组使用**：仅看“batch mixing”或仅看“生物分离”会导致偏差，需联合评估。

## 4.2 争议与分歧（并存证据）

1. **ComBat vs limma 并无绝对赢家**：
   - 在 DIA-SCP workflow benchmark（Nat Commun 2025）中，ComBat 与 limma 都可进入高表现组合；
   - 在 scplainer benchmark（Genome Biology 2025）中，HarmonizR-ComBat 的总体增益小于 limma/scplainer。
   **推断**：优劣依赖于缺失结构、协变量建模与实验设计（尤其标签/批次是否混杂）。

2. **非线性方法（Scanorama/Harmony/MNN 类）是否优于线性方法**：
   - 在部分研究中可改善混合度；
   - 但在蛋白矩阵预处理任务中，稳定性和可解释性未必持续领先。
   **推断**：若目标是“生成可追溯蛋白定量矩阵并保留协变量解释”，线性模型更可控；若批次形态明显非线性，可再引入非线性方法对照。

## 4.3 证据强度评级

- **高**：Nat Commun 2025（DIA-SCP workflow benchmark）、Genome Biology 2025（scplainer benchmark）。
- **中高**：Nat Commun 2022（HarmonizR）、Nat Commun 2025（protein-level timing benchmark）。
- **中**：Proteomics 2024（SCPRO-HI）、多模态框架类 benchmark（跨模态迁移）。

---

## 5. 对 DIA-SCP 去批次 benchmark 的实践建议

### 5.1 推荐最小方法集（首轮）

- `none`（基线）
- `limma`
- `ComBat-parametric`
- `ComBat-nonparametric`
- `MNN` 或 `Scanorama`（至少一个非线性对照）

### 5.2 推荐最小指标集（蛋白矩阵层）

- 批次去除：between-batch variance ratio / batch-ASW / iLISI（或近似）
- 生物保真：cell type ARI/NMI/ASW（有标签时）
- 工程维度：runtime、失败率、对缺失值与协变量的鲁棒性

### 5.3 场景设计建议

- 最小要同时包含 **balanced** 与 **confounded** 场景（文献反复强调）。
- 在当前 ScpTensor benchmark 设计中，建议进一步把 `confounded` 拆成 `partially confounded` 与 `fully confounded` 两类，而不是继续放成单一大类。
- `fully confounded` 更适合作为 `guardrail / failure benchmark`，不应与 `balanced` 主榜单混排。
- 明确协变量策略：
  - 有协变量：进入设计矩阵；
  - 协变量与批次共线：应返回可解释错误（rank-deficient guardrail）。

### 5.4 风险边界

1. 在高缺失场景，直接套 ComBat/limma 可能因矩阵缺口导致不稳。
2. 先补值再校正可能引入偏差（特别是 MNAR）；需要谨慎。
3. 仅追求 batch mixing 可能过校正，损失真实生物差异。

---

## 6. 数据与资源清单（需区分稳定公开与待复核）

1. **论文证据**：Wang 2025
   https://www.nature.com/articles/s41467-025-65174-4
   说明：截至 `2026-03-12`，其关联 accession `PXD056832` 仍不宜表述为稳定公开主数据入口，因此这里继续只把 Wang 2025 作为 task-design 论文证据。

2. **数据入口**：MassIVE `MSV000093867`
   https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?accession=MSV000093867

3. **论文证据**：Ctortecka 2024
   https://www.nature.com/articles/s41467-024-49651-w

4. **资源包**：`scpdata`（Bioconductor）
   https://bioconductor.org/packages/release/data/experiment/html/scpdata.html

5. **论文证据**：scplainer 2025
   https://doi.org/10.1186/s13059-025-03713-4

6. **论文证据 / task-design evidence**：Zheng 2025
   https://www.nature.com/articles/s41467-025-64718-y

7. **模块规范**：ProteoBench DIA Single Cell 模块（社区基准）
   https://proteobench.readthedocs.io/en/stable/available-modules/active-modules/9-quant-lfq-ion-dia-singlecell/

---

## 7. 未解决问题（Gaps）

1. 目前“专门针对 DIA-SCP 蛋白矩阵去批次”的公开 benchmark 仍较少，许多结论来自 workflow 级评估或跨模态迁移。
2. 单一总分（overall score）容易掩盖“batch removal 与 bio-conservation”的结构性权衡，建议长期保留分项分数。
3. 对“缺失机制（MCAR/MNAR）× 去批次方法”的交互评估仍不充分，值得单独设计。

---

## 8. Shared Citation Registry Coverage

以下共享高频条目的规范元数据以 `docs/references/citations.json` 为准：

- `wang2025_natcom_dia_scp_benchmark`
- `ctortecka2024_natcom_automated_scp`
- `vanderaa2025_genomebio_scplainer`
- `schlumbohm2022_natcom_harmonizr`
- `zheng2025_natcom_protein_batch`
- `liu2025_natmethods_multitask_integration`
- `luecken2022_natmethods_scib`
- `johnson2007_biostatistics_combat`
- `ritchie2015_nar_limma`
- `haghverdi2018_natbiotechnol_mnn`
- `korsunsky2019_natmethods_harmony_lisi`
- `hie2019_natbiotechnol_scanorama`
- `massive_msv000093867`
- `scpdata_bioconductor`
- `proteobench_dia_singlecell_module`
- `koca2024_proteomics_scpro_hi`

---

## 9. 访问日期说明

上述链接均于 **2026-03-05** 访问与核对。
