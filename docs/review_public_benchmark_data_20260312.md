# DIA 驱动单细胞蛋白组预处理中的公共 benchmark 数据集与任务设计：优先级文献综述（截至 2026-03-12）

## 1. 研究范围

- 研究问题：ScpTensor 若要系统 benchmark `QC / missingness / normalization / imputation / integration / AutoSelect`，应优先使用哪些公共数据资源与任务设计原则？
- 目标输出：为以下目录与后续评测方案提供文献依据：
  - `benchmark/normalization`
  - `benchmark/imputation`
  - `benchmark/integration`
  - `benchmark/autoselect`
- 核心约束：项目合同要求 benchmark 仍围绕 `DIA-NN / Spectronaut` 输入与最终 `protein-level matrix` 展开，不扩展到非 DIA 软件兼容层。
- 检索日期：`2026-03-12`

### 1.1 关键词优先级（P1 / P2 / P3）

| 优先级 | 目的 | 关键词 |
|---|---|---|
| P1 | 方法核心 | benchmark dataset, benchmark design, evaluation metric, reproducibility |
| P2 | 模态与软件 | DIA single-cell proteomics, label-free single-cell proteomics, DIA-NN, Spectronaut |
| P3 | 应用边界 | QC, imputation, normalization, batch correction, AutoSelect |

## 2. 检索策略与筛选标准

### 2.1 Query 梯度（P1 -> P2 -> P3）

1. 先检索直接面向 DIA-scProteomics benchmark 的一手论文和公开模块。
2. 再补充高通量单细胞蛋白组公开数据资源。
3. 最后纳入单细胞 omics benchmark 设计原则文献，约束任务拆分和指标选择。

### 2.2 纳入标准

- 一手来源：期刊官网、官方 benchmark 文档、公共数据仓库页面、Bioconductor 资源页。
- 至少满足以下一项：
  - 公开可复用的 DIA-sc / SCP 数据集
  - 直接提供 benchmark 任务与评价指标
  - 可迁移到 ScpTensor preprocessing benchmark 的任务设计原则

### 2.3 排除标准

- 仅有方法演示、没有公开数据或没有稳定访问路径的材料。
- 只覆盖下游分类/生物发现、不支撑 preprocessing benchmark 的论文。

### 2.4 本轮纳入

- 初筛候选：`20+`
- 深读纳入：`7`
- 其中公共数据/数据入口：`4`
- 任务设计原则来源：`3`

## 3. 逐篇证据摘要（Per-paper Summaries）

说明：本节统一沿用全仓库资源分型。除特别标注外，单篇文献条目默认记为 `论文证据`；官方软件/手册页记为 `模块规范 / 软件文档`；具体 accession 或 dataset page 记为 `数据入口`；可脚本化分发包记为 `资源包`。
共享高频条目的规范元数据统一以 `docs/references/citations.json` 为准；若本文件历史写法与 registry 在作者简称、发布日期、期刊、DOI 或 canonical URL 上不一致，以 registry 为准，本文件仅保留 public benchmark panel 的角色定义和场景边界。

### 3.1 Wang et al., Nature Communications, 2025

- 题目：Benchmarking informatics workflows for data-independent acquisition single-cell proteomics
- 链接：https://doi.org/10.1038/s41467-025-65174-4
- 发布日期：`2025-11-21`
- 价值：
  - 这是目前最直接面向 `DIA single-cell proteomics preprocessing` 的一手 benchmark 论文之一。
  - 同时覆盖 `sparsity reduction`、imputation、normalization、batch correction 与 differential analysis。
- 对 ScpTensor 的意义：
  - 提供“不要只 benchmark 单一步骤，要看串联流程”的直接依据。
  - 说明同质与异质设计、同批次与跨批次设计，必须拆开评测。
- 最新访问核查：
  - 文中关联数据集 accession `PXD056832` 当前可解析到 ProteomeXchange 页面，但该页面在 `2026-03-12` 仍标记为 `reserved`，语义上对应“已存储但尚未公开发布/公告”，因此不宜直接作为 CI 级稳定公共输入。

### 3.2 Ctortecka et al., Nature Communications, 2024

- 题目：Automated single-cell proteomics providing sufficient proteome depth to study complex biology beyond cell type classifications
- 论文链接：https://doi.org/10.1038/s41467-024-49651-w
- 发布日期：`2024-07-23`
- 数据入口：https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?accession=MSV000093867
- 主要价值：
  - 给出高通量、公开可访问的单细胞蛋白组数据资源与实验描述。
  - MassIVE 页面可公开访问，适合被纳入可重复 benchmark 面板。
- 适合支持的任务：
  - 样本层 QC
  - missingness / completeness 描述
  - normalization 与 imputation 的真实数据评测
  - 部分 batch-aware 评测
- 场景边界：
  - 这类公开 SCP 数据更适合作为 `public SCP reference layer`，不能在未补充 provenance 与软件出口说明前，直接等同于 ScpTensor 合同内 `DIA-NN / Spectronaut` 主线 benchmark 输入。

### 3.3 ProteoBench DIA single-cell 模块（官方文档）

- 链接：https://proteobench.readthedocs.io/en/stable/available-modules/active-modules/9-quant-lfq-ion-dia-singlecell/
- 访问日期：`2026-03-12`
- 主要价值：
  - 提供社区化、任务化的 benchmark 入口，而不只是单篇论文。
  - 文档直接规定了挑战模块、输出物和评价逻辑，更适合 ScpTensor 后续与外部基准对齐。
- 适合支持的任务：
  - end-to-end DIA LFQ single-cell benchmarking
  - completeness / accuracy / consistency 类指标对齐
  - AutoSelect 报告指标对齐
- 场景边界：
  - 这是 `module/specification` 入口，不是单一原始数据集入口；适合约束任务定义与输出格式，不应被写成“公开数据集本身”。

### 3.4 `scpdata`（Bioconductor 公共数据资源）

- 链接：https://bioconductor.org/packages/release/data/experiment/html/scpdata.html
- 访问日期：`2026-03-12`
- 主要价值：
  - 提供公开、可脚本化获取的单细胞蛋白组数据资源入口，适合可复现实验与教程。
  - 对 ScpTensor 来说，它更像“数据资源层”，不是单一 benchmark 论文。
- 适合支持的任务：
  - 快速复现 benchmark 原型
  - 教程/文档中的小规模公共示例
  - 合成掩码实验与 benchmarking scaffold
- 场景边界：
  - `scpdata` 适合作为 `resource package / example layer`，不应单独承担完整 DIA-SCP benchmark panel 的角色。

### 3.5 Zheng et al., Nature Communications, 2025

- 题目：Protein-level batch-effect correction enhances robustness in MS-based proteomics
- 链接：https://doi.org/10.1038/s41467-025-64718-y
- 发布日期：`2025-11-04`
- 主要价值：
  - 把 `protein-level endpoint`、balanced/confounded 设计和 correction timing 放到同一评价框架中，比只看 peptide/feature 级指标更贴近 ScpTensor 合同边界。
  - 适合作为“最终 deliverable 是否是完整 protein matrix”这一标准的外部证据。
- 适合支持的任务：
  - protein-level completeness
  - protein-level reproducibility
  - balanced / confounded 场景下的批次解释边界
- 场景边界：
  - 这篇论文是 `protein-level task-design evidence`，不是 DIA-SCP 公共数据集论文，也不是稳定公开数据入口本身。

### 3.6 Weber et al., Nature Communications, 2019

- 题目：Essential guidelines for computational method benchmarking
- 链接：https://doi.org/10.1038/s41467-019-09406-4
- 发布日期：`2019-04-02`
- 主要结论：
  - benchmark 必须把数据生成假设、评价指标、调参范围、失败情形和可重复性清楚分离。
  - 不能只报告“单一分数赢家”，必须解释任务边界与失败模式。
- 对 ScpTensor 的意义：
  - 这为 AutoSelect 的“多维评分 + 风险提示”提供通用 benchmark 设计原则。
  - 也支持把 `quality / balanced / speed` 策略显式拆开，而不是追求单一总分。

### 3.7 公共数据与任务模块的联合结论

- 论文与数据入口共同表明：
  - 没有单一数据集能同时覆盖 `QC`、missingness、normalization、integration、AutoSelect 的全部需求。
  - 因此 ScpTensor 需要“最小 benchmark panel”而不是“唯一 benchmark dataset”。

### 3.8 二次核查补充（发布日期、稳定入口与场景边界）

- `Wang et al., Nat Commun (published: 2025-11-21)` 是当前最直接的 DIA-SCP workflow/task-design 证据，但其关联 accession `PXD056832` 在 `2026-03-12` 的官方核查中仍为 ProteomeXchange `reserved` 状态，因此只能作为设计证据，不能写成“当前稳定公开主数据入口”：<https://doi.org/10.1038/s41467-025-65174-4>
- `Ctortecka et al., Nat Commun (published: 2024-07-23)` 提供稳定可访问的论文页，而 `MSV000093867` 提供稳定可访问的数据页；二者应分开书写为“论文证据”和“数据入口”，避免把 paper URL 与 dataset URL 混成同一层级：<https://doi.org/10.1038/s41467-024-49651-w>；<https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?accession=MSV000093867>
- `ProteoBench DIA single-cell module`（accessed: `2026-03-12`）是社区 benchmark 模块规范，不是单个 public dataset；适合作为任务定义、输出格式和评分接口的稳定入口：<https://proteobench.readthedocs.io/en/stable/available-modules/active-modules/9-quant-lfq-ion-dia-singlecell/>
- `scpdata`（accessed: `2026-03-12`）是 Bioconductor 资源包入口，适合作为教程、示例与可脚本化 reference layer，而不是完整的 DIA-SCP benchmark panel：<https://bioconductor.org/packages/release/data/experiment/html/scpdata.html>
- `Zheng et al., Nat Commun (published: 2025-11-04)` 提供 protein-level endpoint 与 balanced/confounded 解释框架，但它不是 DIA-SCP public dataset 论文；更合适的定位是 `task-design / interpretation evidence`：<https://doi.org/10.1038/s41467-025-64718-y>
- `Weber et al., Nat Commun (published: 2019-04-02)` 继续约束文档层的 benchmark 设计原则，即必须把数据、指标、失败情形与可重复性分开陈述：<https://doi.org/10.1038/s41467-019-09406-4>

## 4. 横向比较与证据分级

### 4.1 公共资源对照表

| 资源 | 资源类型 | 稳定入口（2026-03-12） | 合同内角色 | CI 适用性 | 当前风险 |
|---|---|---|---|---|---|
| Wang 2025 paper | 论文 / 设计证据 | Nature article | DIA-SCP workflow 与任务拆分证据 | 否，论文本身不是输入数据 | 必须与真实数据入口分开 |
| `PXD056832` | 数据入口 | ProteomeXchange | Wang 2025 关联 accession | 否，当前为 `reserved` | 尚未稳定公开 |
| Ctortecka 2024 paper | 论文 / 实验描述 | Nature article | public SCP reference evidence | 否，论文本身不是输入数据 | 不是合同内 DIA 主线输入 |
| `MSV000093867` | 数据入口 | MassIVE dataset | public SCP reference layer | 条件性可用，仅适合 reference / prototype | 需额外标注 provenance、软件出口与数据层级 |
| ProteoBench DIA-SC module | 模块规范 | ReadTheDocs stable page | 合同内任务定义与外部对齐 | 是，但作为规范入口而非原始数据入口 | 外部模块版本可能演进 |
| `scpdata` | 资源包 / example layer | Bioconductor release page | 教程、示例、小规模回归 scaffold | 是，但不应承担主 benchmark 面板 | 覆盖面有限 |
| Zheng 2025 paper | 论文 / task-design evidence | Nature article | protein-level endpoint 与场景解释证据 | 否，论文本身不是输入数据 | 非 DIA-SCP primary dataset |

### 4.2 一致结论（facts）

1. benchmark 不能只评单一步骤，必须把 preprocessing 流程拆成 `QC -> missingness/sparsity handling -> normalization -> imputation -> batch correction -> downstream readout`。
2. 同质设计与异质设计必须分开；同批次与跨批次也必须分开，否则方法排名会不稳定。
3. 公共资源必须先分型为 `论文证据 / 数据入口 / 模块规范 / 资源包`，再决定它在 benchmark 里的角色；不能把这四类资源混写成“数据集”。
4. 公开可重复 benchmark 更依赖“稳定访问的数据入口 + 明确任务定义”，而不只是高影响因子论文。
5. 对 ScpTensor 这种以 `protein-level complete matrix` 为交付物的包，benchmark 应优先在 protein 层报告 completeness、reproducibility 和批次混杂改善，而不是停留在 precursor 指标。

### 4.3 分歧与解释（inference）

- [推断] 由于当前最直接的 DIA-sc benchmark 论文所关联 accession 仍存在公开状态不稳定问题，ScpTensor 的主线回归 benchmark 不应绑定单一 reserved accession，而应把 `合同内主线` 与 `public SCP reference layer` 分开维护。
- [推断] `benchmark/` 目录更适合采用“三层设计”：
  - 合同内主线 benchmark：`DIA-NN / Spectronaut` 可复用输入或官方模块规范
  - 稳定公共 reference layer：公开 SCP 数据入口与资源包
  - 合成 / 半合成 stress benchmark：用于 controlled failure mode 检查

### 4.4 证据强度

- 高：Wang 2025、Ctortecka 2024、ProteoBench 模块、Weber 2019
- 中高：Zheng 2025、`scpdata`
- 中：与具体任务对齐时的二次推断

## 5. 面向 ScpTensor 的实践建议

### 5.1 推荐的最小 benchmark panel

1. `合同内主线任务规范`
   - 首选：ProteoBench DIA single-cell 模块 + Wang 2025 的 workflow/task design
   - 用途：约束 benchmark 任务拆分、输出接口与评分语义
2. `稳定公共 SCP reference layer`
   - 首选：`MSV000093867` + `scpdata`
   - 用途：教程、原型回放、公开 reference 数据对照
   - 边界：不能直接替代合同内 DIA 主线回归面板
3. `protein-level interpretation evidence`
   - 首选：Zheng 2025
   - 用途：约束最终交付物仍是 `protein-level matrix`，并把 balanced/confounded 解释写清楚
4. `合成/半合成 stress 数据`
   - 继续使用仓库现有 `benchmark/autoselect/run_synthetic_normalization_stress.py`
   - 用途：控制 confounding、scale shift、missingness 结构

### 5.2 各任务应优先看的指标

- `QC`
  - 样本级 completeness
  - `n_features`
  - total intensity
  - batch-stratified outlier summaries
- `missingness / imputation`
  - state-aware completeness
  - masked-value recovery
  - downstream DE / clustering stability
- `normalization`
  - batch mixing 改善
  - biological conservation
  - distribution distortion 风险
- `integration`
  - batch removal 与 biological conservation 的双指标
  - confounded vs balanced 设计下的稳健性
- `AutoSelect`
  - 方法排名稳定性
  - 报告可解释性
  - 运行时间与内存成本

### 5.3 与现有 `benchmark/` 目录的映射建议

- `benchmark/normalization`
  - 若引入 `MSV000093867`，应标记为 `public_scp_reference`，不替代合同内 DIA 主线面板
- `benchmark/imputation`
  - 可在 public reference layer 上增加 masked-value 实验，但结论需与合同内主线任务分开报告
- `benchmark/integration`
  - 把 balanced / confounded 任务显式拆开，并把 `task-design evidence` 与 `dataset evidence` 分开挂接
- `benchmark/autoselect`
  - 保留当前 synthetic stress 脚本，同时把 `contract mainline` 与 `reference layer` 评分分层汇总

### 5.4 不建议的 benchmark 设计

- 不建议只用单一数据集得出“全场景最优方法”。
- 不建议只看单一总分，不拆分 batch removal、biological conservation、runtime。
- 不建议把暂未稳定公开的数据 accession 作为 CI 的唯一输入源。

## 6. 风险边界

1. 直接适配 DIA-sc 的公开 benchmark 数据仍然不多，且部分论文关联 accession 的公开状态会变化，必须在引入 CI 前二次核查。
2. 高通量 label-free 数据与 multiplex/参考通道数据的 benchmark 目标不同，不应混在同一排行榜里。
3. 若只用合成数据，很容易高估方法的稳定性；若只用真实数据，又难以获得 ground truth，因此必须组合两类设计。
4. 文献里的最佳方法排名往往受任务定义影响，ScpTensor 更应强调“报告任务条件”而非“宣布唯一赢家”。

## 7. 对后续调研/实现的优先建议

1. 已完成“masked-value benchmark 设计”与“batch confounding benchmark 设计”两篇拆分综述；下一步应保持它们与 `benchmark/README.md`、`benchmark/imputation/README.md`、`benchmark/integration/README.md` 的交叉引用同步。
2. 把当前公共 benchmark panel 固化为 machine-readable manifest，至少记录 `accession / source_url / public_status / data_level / recommended_tasks / CI_suitability`。
3. 对 accession 可用性建立定期复核与降级策略，避免将 `reserved`、临时失效或镜像不稳定的数据源放成 CI 主路径唯一依赖。

## 8. Shared Citation Registry Coverage

以下共享高频条目的规范元数据以 `docs/references/citations.json` 为准：

- `wang2025_natcom_dia_scp_benchmark`
- `ctortecka2024_natcom_automated_scp`
- `massive_msv000093867`
- `proteobench_dia_singlecell_module`
- `scpdata_bioconductor`
- `zheng2025_natcom_protein_batch`
- `weber2019_natcom_benchmarking_guidelines`
