# 蛋白层面归一化方法基准测试

本目录用于评测 `scptensor.normalization` 中蛋白层归一化方法在公开 DIA 数据上的表现。

## Current Entry

- `benchmark/normalization/run_benchmark.py`

## Boundary

默认方法：

- `none`
- `mean`
- `median`
- `quantile`
- `trqn`

当前实现边界：

- benchmark 脚本会先对 `raw` 层执行 `log_transform(base=2, offset=1)`，再在显式 `log2` 层比较归一化方法。
- 这里的 `log2` 是当前 benchmark 脚本的实现局部层名，不等于仓库文档主线的 canonical `log` 命名。
- 因此 `quantile` / `trqn` 并不是对线性 vendor 输出层直接比较，这与 `docs/review_log_scale_20260312.md` 的尺度门禁一致。
- 当前 stable normalization API 的方法池是 `none / mean / median / quantile / trqn`。
- 文献综述里提到的 `sum` 是可补候选，但当前实现与本 benchmark 尚未覆盖。
- 当前 benchmark 还没有输出 state-aware completeness / uncertainty burden。
- 输出中若出现 `median_centered`、`quantile_norm`、`trqn_norm` 等实现兼容名，它们只表示具体方法产物；benchmark 排名不重定义仓库主线 canonical `norm` 层命名。
- 当前默认脚本、当前默认数据集与当前参考下载资产只表示“今天仓库默认怎么跑”；后续即使更换默认输入或增删辅助数据，也不应反向改写本目录的长期合同边界。

## Review Links

- `docs/review_normalization_20260307.md`
- `docs/review_log_scale_20260312.md`
- `docs/review_public_benchmark_data_20260312.md`
- `docs/normalization_contract.md`

## Resource Roles

- `论文证据`
  - Wang 2025 / Nature Communications 2022 / PRONE / TRQN / ProNorM / directLFQ
- `数据入口`
  - LFQbench HYE124 Spectronaut TSV
  - PRIDE `PXD054343`
- `模块规范 / 软件文档`
  - MSstats workflow / QFeatures processing / PRONE preprocessing tutorial
- `资源包`
  - 当前 README 不把 package/tutorial 页面写成数据入口，而只把它们用作 logged-layer workflow contract evidence

经二次核查的来源性细节：

- `MSstats` 的 `dataProcess()` 明确把 `log transformation` 放在 `normalization` 之前，并把 quantile normalization 作为该流程中的一个归一化选项；这与“先显式 log，再比较 quantile-family 方法”的合同一致：<https://www.bioconductor.org/packages/release/bioc/vignettes/MSstats/inst/doc/MSstatsWorkflow.html>
- `QFeatures` 教程先构造 `peptides_log`，再对该 assay 调用 `normalize()`，并在图中对比 log2 强度归一化前后分布；这再次支持把 normalization 解释为 logged assay 上的操作：<https://bioconductor.org/packages/release/bioc/vignettes/QFeatures/inst/doc/Processing.html>
- `PRONE` 预处理教程在 marker 可视化中直接使用 `log2` protein intensities，并围绕 log2 assay 进行后续过滤与比较；因此把 `quantile / trqn` 限定到显式 logged layer 不是 ScpTensor 特例，而是与现代 proteomics workflow 表达方式一致：<https://daisybio.github.io/PRONE/articles/Preprocessing.html>
- `Karuppanan et al.`（JPR 2025）进一步说明 normalization 与 imputation 的最优组合具有显著数据依赖性；因此当前目录继续只声明“stage-specific normalization benchmark”，而不把它扩大解释成完整 preprocessing 组合的统一赢家：<https://doi.org/10.1021/acs.jproteome.4c00552>

## 1. 长期证据与数据边界

### 1.1 已确认存在“蛋白层面归一化 benchmark”的公开论文

1. **Nature Communications 2025（DIA 单细胞工作流 benchmark）**
   链接：<https://www.nature.com/articles/s41467-025-65174-4>
   文中明确比较了不同 normalization 方案，并给出 Data availability（PXD054343 / PXD061065 / MSV000093301）和 Source Data。

2. **Nature Communications 2022（DIA 分析策略 benchmark）**
   链接：<https://www.nature.com/articles/s41467-022-28993-0>
   文中将 normalization 作为 workflow 关键组件进行系统评估，并公开了 raw 数据与 source data。

3. **Briefings in Bioinformatics 2025（PRONE：proteomics normalization benchmark 框架）**
   链接：<https://pubmed.ncbi.nlm.nih.gov/40336172/>
   `2026-03-12` 二次核查显示，PRONE 的正确 DOI 是 `10.1093/bib/bbaf201`；当前以 PubMed 入口作为稳定可核查链接。文中 Data availability 列出多套公开 benchmark 数据来源（ProteomeXchange / PRIDE）。

### 1.2 当前默认数据与参考资产（实现事实，不构成合同）

以下条目只说明当前默认脚本会优先使用或留档哪些输入/参考文件，不等价于 normalization benchmark 的完整长期数据边界：

1. **LFQbench HYE124 Spectronaut 示例（用于真值比值指标）**
   来源：<https://github.com/IFIproteomics/LFQbench>
   直链：<https://raw.githubusercontent.com/IFIproteomics/LFQbench/master/ext/data/vignette_examples/hye124/Spectronaut_TTOF6600_64w_example.tsv>
   体量：约 10 MB（适合快速复现）
   用法：先肽段->蛋白聚合（固定 `sum`），再进行蛋白层归一化。

2. **PXD054343 DIA-NN 2x 报告（仓库内已提供）**
   本地文件：`data/dia/diann/PXD054343/2_3_SC_SILAC_2x_report.tsv`
   原始数据链接：<https://www.ebi.ac.uk/pride/archive/projects/PXD054343>

3. **Nature Communications 2025 Source Data（下载留档）**
   直链：<https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-025-65174-4/MediaObjects/41467_2025_65174_MOESM12_ESM.xlsx>
   体量：约 24 MB（已纳入 `benchmark/normalization/data/references/`）

### 1.3 当前未纳入默认运行的数据

- Nature Communications 2022 Source Data（Zenodo）
  <https://doi.org/10.5281/zenodo.6379087>
  `Source Data.zip` 约 3.02 GB，不适合默认快速 benchmark，因此本轮不自动下载。

解释：

- 是否被纳入“默认运行”主要受复现成本、体量和脚本稳定性约束。
- 这类默认运行决策是工程层实现事实，不等同于长期合同；长期合同仍由上面的证据边界、方法边界和评分边界定义。

## 2. 指标体系

本 benchmark 默认输出多维指标（越全面越接近文献常见评测思路）：

- 覆盖与可用性：`coverage_ratio`, `feature_quantified_ratio`
- 分布对齐：`sample_median_mad`, `sample_iqr_cv`, `pairwise_wasserstein_median`
- 稳定性（RLE）：`rle_mad_median`
- 组内精度：`within_group_sd_median`
- 组间信号保留：`group_eta2_median`
- 真值比值准确性（仅对 LFQbench）：`ratio_mae`, `ratio_rmse`, `ratio_pairwise_auc_mean`, `ratio_changed_vs_bg_auc`

并提供 0-1 归一化综合得分（按“高好/低好”方向自动处理）。

说明：

- 这是 stage-specific normalization benchmark，不等同于 AutoSelect 的 `selection_score`。
- 当前脚本尚未把 `VALID / MBR / LOD / FILTERED / IMPUTED` 等状态向量纳入该阶段评分。
- 当前 README 也不把本目录的单阶段 normalization 排名扩大解释成 normalization+imputation 组合最优；若要做完整组合结论，应另行建立 paired sensitivity panel。
- 如果后续加入 `sum` 一类候选，需要先明确它是“normalization”还是“rescaling baseline”；当前文献与软件生态对 `mean/median/quantile/TRQN` 的归一化语义更稳定。

## 3. 运行方式（uv）

在仓库根目录执行：

```bash
UV_CACHE_DIR=.uv-cache uv run python benchmark/normalization/run_benchmark.py
```

常用参数：

```bash
UV_CACHE_DIR=.uv-cache uv run python benchmark/normalization/run_benchmark.py \
  --datasets lfqbench_hye124_spectronaut pxd054343_diann_2x \
  --methods none mean median quantile trqn \
  --output-dir benchmark/normalization/outputs
```

如需跳过文献 source data 下载：

```bash
UV_CACHE_DIR=.uv-cache uv run python benchmark/normalization/run_benchmark.py \
  --no-reference-download
```

## 4. 输出文件

运行后主要产物：

- `benchmark/normalization/outputs/metrics_summary.csv`
- `benchmark/normalization/outputs/metrics_scores.csv`
- `benchmark/normalization/outputs/ratio_protein_table.csv`（若包含真值数据集）
- `benchmark/normalization/outputs/run_metadata.json`
- `benchmark/normalization/outputs/summary_metrics.png`
- `benchmark/normalization/outputs/score_heatmap.png`
- `benchmark/normalization/outputs/overall_scores.png`
- `benchmark/normalization/outputs/ratio_distribution.png`（若包含真值数据集）

## 5. Git 大文件控制

`benchmark/**/data/` 与 `benchmark/**/outputs/` 已在仓库 `.gitignore` 中忽略，避免提交大文件。
