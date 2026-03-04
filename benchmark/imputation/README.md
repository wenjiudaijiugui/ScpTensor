# 蛋白层缺失值填充（Imputation）基准测试

本目录用于评测 `scptensor.impute` 在 **DIA 单细胞蛋白组预处理场景** 下的表现。

## 1. 文献与公开实现调研结论

### 1.1 与 DIA 单细胞直接相关的 workflow benchmark

1. **Nature Communications 2025（DIA 单细胞工作流）**
   链接：<https://www.nature.com/articles/s41467-025-65174-4>
   关键信息：
   - 在 workflow 中系统比较了缺失填充方法（`none`, `randomforest`, `QRILC`, `MinDet`, `KNN`）
   - 评价维度包含缺失率、CV，以及差异分析相关指标（如 pAUC / F1）
   - Data availability 提供了 PXD056832、PXD054343、PXD061065 等公开数据来源

### 1.2 与 DIA 缺失值 benchmark 直接相关的方法学

2. **MultiPro（Nature Communications 2025）**
   链接：<https://pubmed.ncbi.nlm.nih.gov/40947414/>
   关键信息：
   - 在 DIA 数据中提供了专门的缺失值填充 benchmark 设计
   - 使用缺失率 **10%~50%**、并设置 **MNAR:MCAR=3:1** 场景
   - 使用 **NRMSE** 作为核心填充误差指标

3. **NAguideR（Bioinformatics 2020）**
   链接：<https://pmc.ncbi.nlm.nih.gov/articles/PMC7641313/>
   关键信息：
   - 给出蛋白组缺失填充系统 benchmark 框架
   - 汇总经典评估指标：NRMSE、SOR、ACC/OI、PSS
   - 公开数据中包含 DIA/SWATH（PXD014777）与 DIA(PASEF)（PXD017476）

4. **PIMMS（Nature Methods 2024）**
   链接：<https://pmc.ncbi.nlm.nih.gov/articles/PMC10949645/>
   关键信息：
   - 单细胞蛋白组场景下进行缺失恢复评估
   - 采用 **MAE** 与 **Pearson r** 作为核心恢复指标

## 2. 本 benchmark 的数据与指标设计

### 2.1 实际执行数据

默认数据集：

- `pxd054343_diann_2x`（本地）
  `data/dia/diann/PXD054343/2_3_SC_SILAC_2x_report.tsv`

可选数据集：

- `lfqbench_hye124_spectronaut`（在线下载）
  <https://raw.githubusercontent.com/IFIproteomics/LFQbench/master/ext/data/vignette_examples/hye124/Spectronaut_TTOF6600_64w_example.tsv>

### 2.2 评估策略

- 先在蛋白层执行：`log_transform -> normalize(mean)`（固定前处理，避免混入归一化变量）
- 在“原始已观测值”上额外造缺失并回填，只在 holdout 位置评估误差
- 缺失场景：
  - `mcar`
  - `mixed_mnar`（低丰度优先缺失，默认 MNAR:MCAR=3:1）
- 缺失比例默认：`0.1, 0.3, 0.5`

### 2.3 指标

核心恢复指标：

- `NRMSE`（MultiPro/NAguideR 常用）
- `MAE`、`RMSE`
- `Pearson r`（PIMMS 常用）
- `Spearman r`
- `holdout_coverage`（方法是否把 holdout 全部填上）

工程与稳定性指标：

- `runtime_sec`
- `post_missing_rate`
- `within_group_cv_median`（对齐 workflow benchmark 中常见 CV 视角）

此外在 LFQbench 数据上额外输出已实现的 ratio 指标（若分组和真值可用）。

## 3. 运行方式（uv）

在仓库根目录执行：

```bash
UV_CACHE_DIR=/tmp/.uv-cache uv run --no-project python benchmark/imputation/run_benchmark.py
```

常用参数示例：

```bash
UV_CACHE_DIR=/tmp/.uv-cache uv run --no-project python benchmark/imputation/run_benchmark.py \
  --datasets pxd054343_diann_2x lfqbench_hye124_spectronaut \
  --methods none zero row_mean row_median half_row_min knn lls bpca iterative_svd missforest qrilc minprob \
  --holdout-rates 0.1 0.3 0.5 \
  --mechanisms mcar mixed_mnar \
  --repeats 1
```

## 4. 输出文件

- `benchmark/imputation/outputs/metrics_raw.csv`
- `benchmark/imputation/outputs/metrics_summary.csv`
- `benchmark/imputation/outputs/metrics_scores.csv`
- `benchmark/imputation/outputs/run_metadata.json`
- `benchmark/imputation/outputs/overall_scores.png`
- `benchmark/imputation/outputs/score_heatmap.png`
- `benchmark/imputation/outputs/nrmse_curves.png`
- `benchmark/imputation/outputs/runtime_vs_accuracy.png`

## 5. Git 大文件控制

`benchmark/**/data/` 与 `benchmark/**/outputs/` 已在 `.gitignore` 忽略。
