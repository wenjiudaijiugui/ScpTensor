# DIA 驱动单细胞蛋白组预处理中的 batch confounding benchmark 设计：优先级文献综述（截至 2026-03-12）

## 1. 研究范围

- 研究问题：ScpTensor 应如何设计 `balanced`、`partially confounded`、`fully confounded` 三类 integration/batch-correction benchmark，使其既能评估 batch removal，又不把真实生物信号抹掉？
- 目标输出：为以下目录与模块提供设计依据：
  - `benchmark/integration`
  - `benchmark/autoselect`
  - `scptensor.integration`
- 核心约束：项目合同要求评测最终回到 **protein-level matrix**，且默认 upstream 限定为 `DIA-NN` 与 `Spectronaut`。
- 检索日期：`2026-03-12`

### 1.1 关键词优先级（P1 / P2 / P3）

| 优先级 | 目的 | 关键词 |
|---|---|---|
| P1 | 方法核心 | batch confounding, batch correction evaluation, integration benchmark |
| P2 | 模态与数据 | DIA proteomics, single-cell proteomics, single-cell integration |
| P3 | 应用边界 | balanced design, confounded design, biological conservation |

## 2. 检索策略与筛选标准

### 2.1 Query 梯度（P1 -> P2 -> P3）

1. 先检索批次效应 benchmark 与混杂设计的一手方法学来源。
2. 再补充 single-cell integration benchmark 框架。
3. 最后纳入 DIA-sc 蛋白组的直接 benchmark 论文，确认这些原则在当前场景的落点。

### 2.2 纳入标准

- 一手来源：期刊官网、DOI 页面、Nature/Genome Biology 页面、PubMed。
- 直接涉及以下至少一项：
  - 批次校正方法在平衡/混杂设计下的可识别性
  - batch removal 与 biological conservation 的双目标评价
  - fully confounded 设计下“不应强行得出干净结果”的风险
  - 可迁移到 DIA-sc protein-level benchmark 的设计原则

### 2.3 排除标准

- 只报告单一方法性能、没有实验设计边界的论文。
- 只给出嵌入图，没有量化指标或没有设计可识别性说明的资料。

### 2.4 本轮纳入

- 初筛候选：`20+`
- 深读纳入：`7`
- 其中基础批次校正文献：`3`
- single-cell benchmark 框架：`3`
- DIA-sc 直接证据：`1`

## 3. 逐篇证据摘要（Per-paper Summaries）

说明：本节统一沿用全仓库资源分型。除特别标注外，单篇文献条目默认记为 `论文证据`；官方软件/手册页记为 `模块规范 / 软件文档`；具体 accession 或 dataset page 记为 `数据入口`；可脚本化分发包记为 `资源包`。
共享高频条目的规范元数据统一以 `docs/references/citations.json` 为准；若本文件历史写法与 registry 在作者简称、发布日期、期刊、DOI 或 canonical URL 上不一致，以 registry 为准，本文件仅保留 confounding design 的解释与边界。

### 3.1 Johnson et al., Biostatistics, 2007

- 题目：Adjusting batch effects in microarray expression data using empirical Bayes methods
- 链接：https://pubmed.ncbi.nlm.nih.gov/16632515/
- 主要发现：
  - ComBat 奠定了经验贝叶斯批次校正的经典框架。
  - 但其使用前提是 batch 与 biological signal 至少部分可分离。
- 对 ScpTensor 的意义：
  - `combat_*` 方法的 benchmark 不应脱离设计可识别性讨论。
  - 不能把所有 batch correction 失败都归咎于方法实现，很多时候是设计本身不可解。

### 3.2 Nygaard et al., Biostatistics, 2016

- 题目：Methods that remove batch effects while retaining group differences may lead to exaggerated confidence in downstream analyses
- 链接：https://pubmed.ncbi.nlm.nih.gov/26272994/
- 主要发现：
  - 在不平衡或混杂设计下，先校正再检验可能夸大 downstream 显著性。
  - “看起来批次消失了”并不意味着结果更可信。
- 对 ScpTensor 的意义：
  - fully confounded benchmark 不应被定义为“谁去得最干净谁赢”。
  - DE-ground-truth 或保真指标必须并报。

### 3.3 Song et al., Nature Communications, 2020

- 题目：Flexible experimental designs for valid single-cell RNA-sequencing experiments allowing batch effects correction
- 链接：https://doi.org/10.1038/s41467-020-16905-2
- 主要发现：
  - 提出 `completely randomized`、`reference panel`、`chain-type` 等可识别设计。
  - 完全混杂设计在统计上不可辨识，不应寄望于后验算法完全修复。
- 对 ScpTensor 的意义：
  - `partially confounded` 场景应依赖共享样本/桥接样本设计。
  - fully confounded 数据更适合作为 guardrail / failure benchmark。

### 3.4 Tran et al., Genome Biology, 2020

- 题目：A benchmark of batch-effect correction methods for single-cell RNA sequencing data
- 链接：https://doi.org/10.1186/s13059-019-1850-9
- 主要发现：
  - 不同 integration 方法在 `kBET / LISI / ASW / ARI` 等指标上表现并不一致。
  - 批次去除与生物保真之间存在稳定 trade-off。
- 对 ScpTensor 的意义：
  - protein-level integration benchmark 也应双轴报告，而不是单一总分。
  - 嵌入图不能替代量化指标。

### 3.5 Luecken et al., Nature Methods, 2022

- 题目：Benchmarking atlas-level data integration in single-cell genomics
- 链接：https://www.nature.com/articles/s41592-021-01336-8
- 主要发现：
  - atlas-level integration benchmark 的核心是同时度量 batch removal 与 biological conservation。
  - 数据场景不同，confounding degree 不同，方法排名也会显著变化。
- 对 ScpTensor 的意义：
  - AutoSelect 的 integration scoring 应显式保留这两个轴，而不是提前合并成一个黑箱分数。
  - balanced 与 partially confounded 不应混在同一排行榜。

### 3.6 Chazarra-Gil et al., Nature Communications, 2023

- 题目：Benchmarking integration of single-cell differential expression
- 链接：https://doi.org/10.1038/s41467-023-37126-3
- 主要发现：
  - integration benchmark 若要与 differential expression 兼容，必须采用能识别真实组差异的平衡设计。
  - 只优化混合度指标会牺牲 downstream DE interpretability。
- 对 ScpTensor 的意义：
  - integration benchmark 不应只看 batch mixing。
  - 若下游目标包含 DE，则需要把 FC 方向、真阳性/假阳性一并纳入。

### 3.7 Wang et al., Nature Communications, 2025

- 题目：Benchmarking informatics workflows for data-independent acquisition single-cell proteomics
- 链接：https://www.nature.com/articles/s41467-025-65174-4
- 主要发现：
  - 在 DIA 单细胞蛋白组里，batch correction 必须与 normalization、imputation、DE 等步骤联动评估。
  - 不同条件与批次结构下，最优工作流不同。
- 对 ScpTensor 的意义：
  - 这是当前最直接的 DIA-sc integration benchmark 语境。
  - 评测不能只在单一 balanced 场景上得出“最优去批次方法”。

### 3.8 Gong et al., Analytical Chemistry, 2025

- 题目：Benchmark of Data Integration in Single-Cell Proteomics
- 链接：https://doi.org/10.1021/acs.analchem.4c04933
- 主要发现：
  - 直接在 single-cell proteomics 数据上比较多种 integration / batch-correction 方法。
  - 评估维度不只包含 batch correction quality，也同时包含 biology preservation 与 marker-level consistency。
  - 不同数据集与评价轴下的方法优势并不一致，说明不存在脱离场景的统一赢家。
- 对 ScpTensor 的意义：
  - `benchmark/integration` 的主合同不应只保留 `batch removal + biological conservation` 双轴；长期还应补充 marker / feature consistency 一类第三报告轴。
  - `balanced` 主榜与 `fully confounded` guardrail 仍应分开解释，不能为了单一总分牺牲边界清晰度。

### 3.9 二次核查补充（发布日期与场景分拆证据）

- `Song et al., Nat Commun (published: 2020-08-03)` 直接把可识别设计拆为 `CR / RP / CHAIN`，并指出 `CC`（complete confounding）在统计上不可辨识；这为 `fully confounded -> guardrail` 提供了最直接的一手依据：<https://www.nature.com/articles/s41467-020-16905-2>
- `Nygaard et al., Biostatistics (published: 2016-10-01)` 指出在 batch 与 group 不平衡时，校正后差异分析可能出现被夸大的统计信心；这支持 `fully confounded` 不应当成主榜选优场景：<https://pubmed.ncbi.nlm.nih.gov/26272994/>
- `Chazarra-Gil et al., Nat Commun (published: 2023-03-20)` 在 integration-DE benchmark 中显式将 `balanced` 与 `unbalanced` 设计分开评测，说明不拆场景会造成解释混叠：<https://www.nature.com/articles/s41467-023-37126-3>
- `Gong et al., Anal Chem (published: 2025-06-17)` 则把 SCP integration 直接拆成 `batch correction / biology preservation / marker consistency` 三类评价，说明 protein-level integration benchmark 继续只保留单一总分会过度丢失解释面：<https://doi.org/10.1021/acs.analchem.4c04933>
- `Quartet, Genome Biol (published: 2023-09-13)` 直接对比 balanced 与 confounded 采样设计，并给出 reference/bridge 设计可显著提升跨批次可比性的证据：<https://genomebiology.biomedcentral.com/articles/10.1186/s13059-023-03047-z>

## 4. 横向比较与证据分级

### 4.1 三类场景的推荐定义

| 场景 | 定义 | 应如何评分 | 预期风险 |
|---|---|---|---|
| `balanced` | 每个 batch 都有各 biological groups，比例尽量接近 | 同时报 batch removal + biological conservation | 最适合主评测 |
| `partially confounded` | 只有部分 groups 在批次间重叠，或依赖 bridge/reference | 同时报双轴指标，并加不确定性说明 | 方法排序更不稳定 |
| `fully confounded` | condition 与 batch 共线，没有共享锚点 | 不应按“越混越好”评分；允许 failure/guardrail 通过 | 统计上不可辨识 |

### 4.2 一致结论（facts）

1. 批次校正是否“成功”首先取决于实验设计是否可识别，而不仅仅是算法选择。
2. `balanced`、`partially confounded`、`fully confounded` 必须分开评测，否则排名没有解释力。
3. batch removal 与 biological conservation 是稳定 trade-off，不能只报告其中一侧。
4. fully confounded 场景下，给出“漂亮但不可解释”的结果并不比明确报警更好。
5. 在 DIA-sc 场景中，batch correction 还要与 normalization、imputation 和 DE 联动评估。

### 4.3 分歧与解释（inference）

- [推断] 对 ScpTensor 来说，`fully confounded` 最合理的通过标准之一是“明确拒绝过度解释”或“输出 guardrail 报错/警告”，而不是必须产出校正矩阵。
- [推断] `partially confounded` 应加入 shared-group/reference-channel/bridge sample 覆盖度报告，否则很难解释方法失败是算法问题还是设计信息不足。

### 4.4 证据强度

- 高：Nygaard 2016、Song 2020、Tran 2020、Luecken 2022、Wang 2025、Gong 2025
- 中高：Johnson 2007、Chazarra-Gil 2023

## 5. 面向 ScpTensor 的实践建议

### 5.1 `benchmark/integration` 应固定三场景

1. `balanced`
   - 作为主排行榜
   - 同时报 `batch removal` 与 `biology conservation`
2. `partially_confounded`
   - 作为鲁棒性评测
   - 必须说明 bridge/reference 覆盖度
3. `fully_confounded`
   - 作为 guardrail benchmark
   - 允许“失败即正确”

### 5.2 指标应按两大轴展示

- `batch removal`
  - batch separability
  - mixing metrics
  - batch ASW / LISI 类指标
- `biological conservation`
  - condition separability
  - ARI / NMI / kNN purity
  - DE 真阳性/假阳性、FC direction consistency

### 5.3 `benchmark/autoselect` 的 integration scoring 建议

- 不要把 fully confounded 场景直接并入主总分。
- 主总分建议只由：
  - balanced
  - partially confounded
  组成。
- fully confounded 应作为 `guardrail pass/fail` 单独展示。

### 5.4 与当前 README 的对齐

- 当前 `benchmark/integration/README.md` 已经有 `balanced_amount_by_sample` 与 `confounded_amount_as_batch` 两条赛道，方向正确。
- 下一步最值得补的是：
  - 单独加入 `partially confounded` 场景
  - 在输出中显式展示“failure is correct”判据

## 6. 风险边界

1. 许多可迁移原则来自 single-cell transcriptomics benchmark，而非 DIA-sc proteomics 原文；在文档中应继续标注这是跨模态迁移。
2. fully confounded 设计下，任何声称“完全恢复真实 biology”的结果都应高度怀疑。
3. 只用 embedding 可视化会低估过度校正风险。
4. 不同 upstream software 输出的 protein matrix 结构不同，benchmark 还应比较 `DIA-NN` 与 `Spectronaut` 输入的一致性。

## 7. 对后续实现/文档的优先建议

1. 在 `benchmark/integration` 中新增 `partially_confounded` 数据生成与报告。
2. 在 integration 报告里加入 `design_identifiability` 或等价说明字段。
3. 在 AutoSelect 文档中明确：`fully confounded` 不是为了选赢家，而是为了验证 guardrail。

## 8. Shared Citation Registry Coverage

以下共享高频条目的规范元数据以 `docs/references/citations.json` 为准：

- `johnson2007_biostatistics_combat`
- `luecken2022_natmethods_scib`
- `wang2025_natcom_dia_scp_benchmark`
- `nygaard2016_biostatistics_exaggerated_confidence`
- `song2020_natcom_designs_batch`
- `tran2020_genomebio_batch_benchmark`
- `chazarra_gil2023_natcom_integration_de`
- `gong2025_analchem_scp_integration_benchmark`
