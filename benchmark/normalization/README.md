# 蛋白层面归一化方法基准测试

本目录用于评测 `scptensor.normalization` 中蛋白层归一化方法在公开 DIA 数据上的表现。

默认方法：

- `none`
- `mean`
- `median`
- `quantile`
- `trqn`

## 1. 文献与数据可得性核查

### 1.1 已确认存在“蛋白层面归一化 benchmark”的公开论文

1. **Nature Communications 2025（DIA 单细胞工作流 benchmark）**
   链接：<https://www.nature.com/articles/s41467-025-65174-4>
   文中明确比较了不同 normalization 方案，并给出 Data availability（PXD054343 / PXD061065 / MSV000093301）和 Source Data。

2. **Nature Communications 2022（DIA 分析策略 benchmark）**
   链接：<https://www.nature.com/articles/s41467-022-28993-0>
   文中将 normalization 作为 workflow 关键组件进行系统评估，并公开了 raw 数据与 source data。

3. **Bioinformatics 2025（PRONE：proteomics normalization benchmark 框架）**
   链接：<https://doi.org/10.1093/bioinformatics/btaf397>
   Data availability 中列出多套公开 benchmark 数据来源（ProteomeXchange / PRIDE）。

### 1.2 本 benchmark 实际使用/下载的数据

1. **LFQbench HYE124 Spectronaut 示例（用于真值比值指标）**
   来源：<https://github.com/IFIproteomics/LFQbench>
   直链：<https://raw.githubusercontent.com/IFIproteomics/LFQbench/master/ext/data/vignette_examples/hye124/Spectronaut_TTOF6600_64w_example.tsv>
   体量：约 10 MB（适合快速复现）
   用法：先肽段->蛋白聚合（固定 `sum`），再进行蛋白层归一化。

2. **PXD054343 DIA-NN 2x 报告（仓库内已提供）**
   本地文件：`data/dia/diann/PXD054343/2_3_SC_SILAC_2x_report.tsv`
   原始数据链接：<https://proteomecentral.proteomexchange.org/cgi/GetDataset?ID=PXD054343>

3. **Nature Communications 2025 Source Data（下载留档）**
   直链：<https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-025-65174-4/MediaObjects/41467_2025_65174_MOESM12_ESM.xlsx>
   体量：约 24 MB（已纳入 `benchmark/normalization/data/references/`）

### 1.3 体量过大未直接纳入本轮运行的数据

- Nature Communications 2022 Source Data（Zenodo）
  <https://zenodo.org/records/6379087>
  `Source Data.zip` 约 3.02 GB，不适合默认快速 benchmark，因此本轮不自动下载。

## 2. 指标体系

本 benchmark 默认输出多维指标（越全面越接近文献常见评测思路）：

- 覆盖与可用性：`coverage_ratio`, `feature_quantified_ratio`
- 分布对齐：`sample_median_mad`, `sample_iqr_cv`, `pairwise_wasserstein_median`
- 稳定性（RLE）：`rle_mad_median`
- 组内精度：`within_group_sd_median`
- 组间信号保留：`group_eta2_median`
- 真值比值准确性（仅对 LFQbench）：`ratio_mae`, `ratio_rmse`, `ratio_pairwise_auc_mean`, `ratio_changed_vs_bg_auc`

并提供 0-1 归一化综合得分（按“高好/低好”方向自动处理）。

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
