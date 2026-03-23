# ScpTensor Benchmarks

`benchmark/` stores executable benchmark scripts, stage-specific benchmark notes,
and generated benchmark assets that are not source-of-truth implementation
contracts.

## Current Directories

- [`aggregation/README.md`](aggregation/README.md): 肽段->蛋白聚合 benchmark（当前主线是 `precursor-to-protein auxiliary board`）
- [`normalization/README.md`](normalization/README.md): 蛋白层归一化 benchmark
- [`imputation/README.md`](imputation/README.md): 蛋白层缺失值填充 benchmark
- [`integration/README.md`](integration/README.md): 蛋白层去批次 / batch confounding benchmark
- [`autoselect/README.md`](autoselect/README.md): AutoSelect 评测脚本与报告资产

## Boundary

- `benchmark/` 负责可执行 benchmark 协议、脚本与报告资产，不替代 `docs/` 中的冻结实现合同。
- benchmark 的数据入口、任务分型与证据边界，优先回到 `docs/review_*.md` 与 `docs/review_manifest_20260312.json`。
- 若 benchmark README 与 contract 文档表述不同，应以仓库 `AGENTS.md` 和 `docs/*_contract.md` 为准。
- benchmark 输出里出现的实现局部 layer / assay / key 名，只用于脚本比较与产物追踪；它们不重定义仓库 canonical `raw / log / norm / imputed / zscore` 命名。
- `reduce / cluster` 之类 experimental downstream helper 即使被某些 benchmark 当作评估终点或敏感性面板，也不构成 stable preprocessing release 的验收边界。
- README 中任何带日期的“运行快照 / 示例结果”都只应视为 archival audit trail，不得盖过当前合同、当前默认输入和当前待补项。

## Evidence Taxonomy

Benchmark docs and review docs now use the same source typing contract:

- `论文证据`: benchmark/design/method papers
- `数据入口`: stable public dataset landing pages
- `模块规范 / 软件文档`: benchmark module pages, official manuals, API docs
- `资源包`: reusable package/resource distribution pages

Read boundary:

- paper pages justify method/task design
- dataset pages justify replayable public inputs
- module/spec pages justify task interface and scoring semantics
- package pages justify example-layer reuse, not benchmark ownership

Machine-readable manifest:

- [`review_manifest_20260312.json`](../docs/review_manifest_20260312.json)
- [`citations.json`](../docs/references/citations.json)
- [`citation_usage.json`](../docs/references/citation_usage.json)

## Benchmark Architecture

ScpTensor 的 benchmark 不再只是一组脚本，而是三层结构：

1. `真实公共数据`
   - 目标：验证方法在可复核 DIA / SCP 数据上的实际表现
   - 适用：`aggregation`、`normalization`、`imputation`、`integration`
2. `合成 / 半合成 stress`
   - 目标：显式控制 scale shift、missingness、batch confounding、feature inconsistency
   - 适用：`autoselect`、`integration`、`imputation`
3. `文献约束评分`
   - 目标：把主流 benchmark 维度转成统一报告合同，避免只留单一总分
   - 适用：`autoselect`

对应稳定入口与资源分型，应优先查阅：

- [`review_public_benchmark_data_20260312.md`](../docs/review_public_benchmark_data_20260312.md)
- [`review_manifest_20260312.json`](../docs/review_manifest_20260312.json)
- [`docs/README.md`](../docs/README.md)

对应文献整理：

- `docs/review_public_benchmark_data_20260312.md`
- `docs/review_masked_imputation_20260312.md`
- `docs/review_batch_confounding_20260312.md`
- `docs/review_aggregation_benchmark_20260312.md`
- `docs/review_state_metrics_20260312.md`

补充核查（用于解释 benchmark 合同为什么这样写）：

- `MSstats` 工作流把预处理明确拆成 `log transformation -> normalization -> feature selection -> missing value imputation -> summarization`：<https://www.bioconductor.org/packages/release/bioc/vignettes/MSstats/inst/doc/MSstatsWorkflow.html>
- `QFeatures` 教程也把 `logTransform()`、`normalize()`、`aggregateFeatures()` 明确拆层，并且聚合发生在已标准化的 peptide assay 上：<https://bioconductor.org/packages/release/bioc/vignettes/QFeatures/inst/doc/Processing.html>
- `scIB` 把 integration 评估拆成 `batch removal` 与 `bio-conservation` 两轴，另行报告 usability / scalability：<https://www.nature.com/articles/s41592-021-01336-8>
- `Gong et al.`（Analytical Chemistry 2025）直接在 single-cell proteomics integration benchmark 中把评价拆成 `batch correction`、`biology preservation` 与 `marker consistency` 三类，说明 SCP integration 不应被单一总分压平：<https://doi.org/10.1021/acs.analchem.4c04933>
- `NAguideR` 与 `PIMMS` 都支持“恢复误差 + 额外 proteomic/downstream 证据”而不是单一重建误差：<https://pmc.ncbi.nlm.nih.gov/articles/PMC7641313/>，<https://www.nature.com/articles/s41467-024-48711-5>
- `Karuppanan et al.`（JPR 2025）表明 normalization 与 imputation 的最优组合具有明显数据依赖性，因此 stage-specific benchmark 必须明确边界，组合结论应单独报告：<https://doi.org/10.1021/acs.jproteome.4c00552>

## Stage Contracts

### Normalization

- 当前 `benchmark/normalization` 在脚本层固定执行 `raw -> log_transform(base=2, offset=1) -> normalization`，因此 `quantile` / `trqn` 的比较发生在显式 `log` layer 上。
- 这一点与 `docs/review_log_scale_20260312.md` 一致，不应把 `quantile` / `trqn` 解释为对线性 vendor 输出层的直接默认比较。
- 当前实现方法池是：
  - `none`
  - `mean`
  - `median`
  - `quantile`
  - `trqn`
- 文献综述中提到的 `sum` 属于后续可补候选，但当前 stable normalization API / benchmark 尚未覆盖。
- 当前已落地主指标：
  - distribution alignment
  - RLE stability
  - within-group SD
  - LFQbench ratio accuracy
- 待补：
  - state-aware completeness / uncertainty burden
  - 与 batch/confounding 设计联动的 normalization 主榜
  - normalization × imputation 组合敏感性 panel；当前单阶段 normalization 榜单不应被扩大解释成完整 preprocessing 组合最优

### Aggregation

- `scptensor.aggregation` 是唯一稳定的 peptide/precursor -> protein 转换阶段。
- 文献推荐 benchmark 采用双赛道：
  - `protein-direct main board`
  - `precursor-to-protein auxiliary board`
- 当前 `benchmark/aggregation` 已落地的是 `precursor-to-protein auxiliary board`，使用 Spectronaut peptide-long 输入经 `aggregate_to_protein()` 回到 protein 层评分。
- `protein-direct main board` 仍属于待补项，README 不应把它写成当前目录已实现事实。
- 主榜单的最终解释必须回到 protein 层，而不是停留在 precursor / peptide 层。
- 方法池对齐当前实现：
  - `sum`
  - `mean`
  - `median`
  - `max`
  - `weighted_mean`
  - `top_n`
  - `maxlfq`
  - `tmp`
  - `ibaq`

推荐主指标：

- protein completeness
- ratio / fold-change preservation
- within-group CV
- downstream DE consistency
- runtime

推荐附指标：

- precursor->protein 映射覆盖率
- unmapped feature rate
- ambiguous mapping burden
- 说明：当前脚本已输出 LFQ-style accuracy / precision / coverage 指标，以及 `DE consistency` proxy、`ambiguous mapping burden` 与部分 state-aware burden（当前为 `valid / non_valid / lod / uncertain` 子集）；但完整状态向量、`protein-direct main board` 与更强 external-ground-truth 终点仍未全面落地。

### Imputation

- `benchmark/imputation` 的主协议是 `masked-value recovery + downstream stability`，而不是只看 reconstruction error。
- `no-imputation` 必须保留为基线。
- 当前实现是 protein-level main board；`precursor-to-protein auxiliary board` 仍未在该目录单独落地。
- 当前脚本的 holdout 由观测到的有限值构造，并未按原始 `MaskCode` 分层，因此还不能声称已经实现 state-aware masking benchmark。
- 当前实现已落地：
  - `NRMSE`
  - `MAE`
  - `Spearman`
  - `DE log2fc_pearson / topk_jaccard / topk_sign_agreement`
  - `DE topk_f1 / pAUC(0.01/0.05/0.10)`
  - `retained_proteins_ratio / fully_observed_proteins_ratio`
  - `runtime / success_rate`
- review 目标但当前待补：
  - state-aware holdout burden
  - `precursor-to-protein auxiliary board`
  - 外部 ground-truth 驱动而非 pseudo-truth 驱动的 `DE pAUC / F1`

### Integration

- 文献推荐 integration benchmark 最终拆成：
  - `balanced`
  - `partially confounded`
  - `fully confounded`
- 当前 `benchmark/integration` 已实现：
  - `balanced_amount_by_sample`
  - `partially_confounded_bridge_sample`
  - `confounded_amount_as_batch`（语义上等价于 `fully confounded`）
- `fully confounded` 用于 guardrail，不应与主榜单混排，也不应跨场景合并出单一全局赢家。
- 指标必须同时覆盖：
  - `batch removal`
  - `biological conservation`
- 当前脚本现已把 `marker / feature consistency` 作为第三报告轴输出，但保持与当前主评分权重分离。
- 当前脚本已输出三场景结果、第三报告轴和 `guardrail_checks.csv`，但尚未输出 state-aware uncertainty burden。
- 当前 `overall_scores.png` 仍属于 legacy 聚合可视化；解释时必须回到按场景分开的 `metrics_scores.csv`，不能把它当作官方主榜。

### AutoSelect

- `benchmark/autoselect` 保存的是策略对比和选择合同验证资产，不是单一“社区总榜”。
- 当前目录的主资产仍聚焦 stable preprocessing 阶段的选择合同；`reduce / cluster` 若后续进入 benchmark，也只能按 exploratory downstream 解释。
- 输出中应持续区分：
  - `overall_score`：方法质量本身
  - `selection_score`：叠加 `quality / balanced / speed` 策略后的选择分
- `quality / balanced / speed` 属于 selection policy，而不是生物学优越性声明。
- 当前目录已落地：
  - normalization literature score
  - synthetic normalization stress
  - integration strategy comparison（含 `balanced / partially_confounded / fully_confounded` 三场景）
  - script-local `state_penalized_selection_score` 辅助列
- 待补：
  - 真实公共数据 panel 上的统一 AutoSelect 评分
  - 真正进入 AutoSelect 主评分合同的 state-aware uncertainty 轴
  - normalization × imputation 组合敏感性结果在统一报告中的显式位置

## State-Aware Metric Contract

文献综述推荐 ScpTensor benchmark 最终不要把所有非空值折叠成单一“完整率”，而应统一报告以下状态向量：

- `valid_rate`
- `mbr_rate`
- `lod_rate`
- `filtered_rate`
- `outlier_rate`
- `imputed_rate`
- `uncertain_rate`

补充解释：

- `direct_observation_rate = valid_rate`
- `supported_observation_rate = valid_rate + mbr_rate`
- `imputed_rate` 反映后验填补依赖程度，不应被当作原始观测完整性

这套状态向量适用于：

- `aggregation`：区分高完整性来自真实观测还是 transfer / fill
- `imputation`：保留原始缺失来源，而不是被填补后静默抹平
- `integration`：检查 batch correction 是否放大 uncertainty burden
- `autoselect`：把 uncertainty burden 作为惩罚轴之一

当前实现状态：

- `aggregation` 已输出 `valid / non_valid / lod / uncertain` 子集，`autoselect` 已输出 script-local `state_non_valid_fraction / state_imputed_fraction` 负担列。
- `normalization / imputation / integration` 仍未统一输出完整状态向量，`autoselect` 也尚未把状态轴接入主评分合同。
- 因此这套状态向量目前仍只能视为“正在分阶段落地的目标合同”，不是“已在所有 benchmark 中完整统一”的既成事实。

## Failure Modes

benchmark 与文档都应显式记录失败情形，而不是只记录成功方法：

- importer:
  - unsupported extension
  - unsupported `table_format`
  - invalid `fdr_threshold`
  - software auto-detect failure
  - feature column auto-detect failure
  - no rows remain after FDR filtering
- aggregation:
  - missing protein mapping column
  - missing iBAQ denominator
  - unsupported aggregation method
  - invalid `top_n` / `lfq_min_ratio_count` / `tmp_log_base`
- integration:
  - fully confounded design should warn or fail instead of pretending to be solved

## Notes

- 主榜单始终以 protein-level matrix 为终点，符合项目合同。
- precursor / peptide 层 benchmark 只作为压力测试或 aggregation 辅榜。
- `benchmark/**/data/` 与 `benchmark/**/outputs/` 默认不跟踪。
