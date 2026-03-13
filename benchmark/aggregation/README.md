# 肽段->蛋白聚合方法基准测试

本目录用于比较 `scptensor.aggregation.aggregate_to_protein` 中多种聚合方法在公开已知混样数据上的表现。

当前实现边界：

- 已落地的是 `precursor-to-protein auxiliary board`：
  - `Spectronaut peptide-long -> aggregate_to_protein -> protein-level scoring`
- 文献综述推荐的 `protein-direct main board` 尚未在本目录单独实现，因此本 README 不把它写成当前事实。
- 当前脚本尚未输出 state-aware completeness、`DE consistency` 与 `ambiguous mapping burden`。
- 当前文档合同下，主排名只能依据最终 `protein-level` endpoint；`peptide/protein mapping`、`.n` 或覆盖率类指标仅用于解释方法行为，不单独决定胜负。

经二次核查的来源性细节：

- `OpenMS ProteinQuantifier` 官方文档明确支持 `top` / `iBAQ`，并允许 `median / mean / weighted_mean / sum` 作为 peptide abundance 到 protein abundance 的聚合方式，这与当前方法池的主体部分直接对齐：<https://openms.de/documentation/html/TOPP_ProteinQuantifier.html>
- `QFeatures` 教程把 aggregation 放在 `peptides_norm -> proteins` 这一步，并显式保留每个 protein 由多少条 peptide 聚合而成的 `.n` 变量；这支持把 peptide 数量与映射负担当作解释性指标，而不仅仅是最终分数：<https://bioconductor.org/packages/release/bioc/vignettes/QFeatures/inst/doc/Processing.html>
- `DIA-ME` 的定量分析显示，在 directLFQ 归一化下，protein CV、重复相关性和 ratio accuracy 可以同时维持稳定；这支持把 `precision + ratio preservation` 视为 aggregation-adjacent 的关键评价轴，而不是只看覆盖率：<https://www.nature.com/articles/s41467-024-52605-x>

## Resource Roles

- `论文证据`
  - MaxLFQ / directLFQ / Goeminne 2020 / Wang 2025 / Zheng 2025
- `数据入口`
  - LFQbench HYE124 Spectronaut TSV
- `模块规范 / 软件文档`
  - OpenMS ProteinQuantifier
- `资源包`
  - 当前 README 不把 package/tutorial 页面当作主数据入口

## 1. 当前基准数据（已实现）

当前首个 benchmark 使用 LFQbench 的 HYE124 Spectronaut 示例数据（小体量，适合快速复现）：

- 数据文件：`Spectronaut_TTOF6600_64w_example.tsv`
- 来源仓库：<https://github.com/IFIproteomics/LFQbench>
- 直链：<https://raw.githubusercontent.com/IFIproteomics/LFQbench/master/ext/data/vignette_examples/hye124/Spectronaut_TTOF6600_64w_example.tsv>

对应 LFQbench 文档中的混样真值（A/B 对比）：

- HUMAN: `log2FC = 0`
- YEAST: `log2FC = +1`（A:B = 2:1）
- ECOLI: `log2FC = -2`（A:B = 1:4）

参考：<https://github.com/IFIproteomics/LFQbench/blob/master/vignettes/LFQbench.Rmd>

## 2. 方法范围

默认评测以下聚合方法：

- `sum`
- `mean`
- `median`
- `max`
- `weighted_mean`
- `top_n`
- `maxlfq`
- `tmp`
- `ibaq`

## 3. 评分指标

当前脚本已落地的指标，参考 LFQbench 的 accuracy / precision / species separation 思路，并增加覆盖率统计：

- 准确性：`MAE`, `RMSE`, `|Bias|`, `median_abs_error`
- 精确性：`cv_median_all`, `technical_variance_bg_cv_median`
- 物种分离能力：`species_overlap_auc_mean`, `changed_vs_background_auc`
- 覆盖率：`coverage_ratio`, `n_quantified`

按 `docs/review_aggregation_benchmark_20260312.md`，后续仍应补充：

- `protein-direct main board`
- `downstream DE consistency`
- precursor/protein 映射不确定性与 `ambiguous mapping burden`
- state-aware completeness / uncertainty 指标

## 4. 可视化输出

脚本会生成多类图用于综合判断：

- `summary_metrics.png`：核心指标总览柱图
- `log2fc_distribution.png`：各方法按物种的 log2FC 分布
- `observed_vs_expected.png`：观测 vs 期望
- `cv_distribution.png`：CV 分布
- `species_coverage.png`：各物种覆盖率
- `metric_heatmap.png`：多指标归一化热图

## 5. 运行方式（uv）

在仓库根目录执行：

```bash
UV_CACHE_DIR=.uv-cache uv run python benchmark/aggregation/run_benchmark.py
```

常用参数：

```bash
UV_CACHE_DIR=.uv-cache uv run python benchmark/aggregation/run_benchmark.py \
  --methods sum mean median max weighted_mean top_n maxlfq tmp ibaq \
  --fdr-threshold 0.01 \
  --output-dir benchmark/aggregation/outputs
```

## 6. 输出文件

运行后主要产物：

- `benchmark/aggregation/outputs/metrics_summary.csv`
- `benchmark/aggregation/outputs/protein_level_results.csv`
- `benchmark/aggregation/outputs/species_coverage_summary.csv`
- `benchmark/aggregation/outputs/pairwise_auc.csv`
- `benchmark/aggregation/outputs/run_metadata.json`
- `benchmark/aggregation/outputs/failed_methods.csv`（若有失败）
- 以及上述 PNG 图

## 7. Git 大文件控制

benchmark 原始数据与输出图表目录默认加入忽略：

- `benchmark/**/data/`
- `benchmark/**/outputs/`

因此不会把下载数据或大图加入版本库。
