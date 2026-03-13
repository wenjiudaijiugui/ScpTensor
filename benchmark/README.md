# ScpTensor Benchmarks

- `aggregation/`: 肽段->蛋白聚合方法基准测试（LFQbench HYE124 数据）
- `normalization/`: 蛋白层归一化方法基准测试（LFQbench + DIA-NN 数据）
- `imputation/`: 蛋白层缺失值填充方法基准测试（DIA-NN 单细胞数据 + 可选 LFQbench）
- `integration/`: 蛋白层去批次方法基准测试（DIA-NN 单细胞真实数据 + confounding 场景）
- `autoselect/`: AutoSelect 相关评测脚本与报告资产
  - 当前主入口脚本：`run_normalization_literature_score.py`、`run_synthetic_normalization_stress.py`、`run_strategy_comparison.py`
  - 结果目录为自动生成资产，默认不跟踪

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
- `NAguideR` 与 `PIMMS` 都支持“恢复误差 + 额外 proteomic/downstream 证据”而不是单一重建误差：<https://pmc.ncbi.nlm.nih.gov/articles/PMC7641313/>，<https://www.nature.com/articles/s41467-024-48711-5>

## Stage Contracts

### Normalization

- 当前 `benchmark/normalization` 在脚本层固定执行 `raw -> log_transform(base=2, offset=1) -> normalization`，因此 `quantile` / `trqn` 的比较发生在显式 `logged` layer 上。
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
- 说明：当前脚本已输出 LFQ-style accuracy / precision / coverage 指标；`DE consistency`、`ambiguous mapping burden` 与 state-aware completeness 尚未全面落地。

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
  - `runtime / success_rate`
- review 目标但当前待补：
  - `DE pAUC / F1`
  - retained proteins
  - state-aware holdout burden

### Integration

- 文献推荐 integration benchmark 最终拆成：
  - `balanced`
  - `partially confounded`
  - `fully confounded`
- 当前 `benchmark/integration` 已实现：
  - `balanced_amount_by_sample`
  - `confounded_amount_as_batch`（语义上等价于 `fully confounded`）
- `partially confounded` 是 review 已确认的待补场景，不应在 README 中被暗示为已落地。
- `fully confounded` 用于 guardrail，不应与主榜单混排，也不应跨场景合并出单一全局赢家。
- 指标必须同时覆盖：
  - `batch removal`
  - `biological conservation`
- 当前脚本已输出双轴数值指标和 `guardrail_checks.csv`，但尚未输出 state-aware uncertainty burden。
- 当前 `overall_scores.png` 仍属于 legacy 聚合可视化；解释时必须回到按场景分开的 `metrics_scores.csv`，不能把它当作官方主榜。

### AutoSelect

- `benchmark/autoselect` 保存的是策略对比和选择合同验证资产，不是单一“社区总榜”。
- 输出中应持续区分：
  - `overall_score`：方法质量本身
  - `selection_score`：叠加 `quality / balanced / speed` 策略后的选择分
- `quality / balanced / speed` 属于 selection policy，而不是生物学优越性声明。
- 当前目录已落地：
  - normalization literature score
  - synthetic normalization stress
  - integration strategy comparison
- 待补：
  - state-aware completeness / uncertainty penalty
  - 真实公共数据 panel 上的统一 AutoSelect 评分

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

- `benchmark` 目录里的脚本还没有在 `aggregation / normalization / imputation / integration / autoselect` 五条赛道上全面输出这套状态向量。
- 当前最多只能把它视为“目标合同”，而不是“已在所有 benchmark 中落地”的既成事实。

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
