# 蛋白层缺失值填充（Imputation）基准测试

本目录用于评测 `scptensor.impute` 在 **DIA 单细胞蛋白组预处理场景** 下的表现。

该 benchmark 已根据 `docs/dia_sc_imputation_literature_review_20260304.md` 的建议调整为推荐路径：
- 推荐方法池（基线 + 传统 + 矩阵补全）
- 多缺失机制（MCAR + mixed MNAR）
- 多缺失比例（10% / 30% / 50%）
- 任务导向指标（恢复误差 + 聚类保真 + DE 信号一致性）

## 1. 文献依据（摘要）

1. **Nature Communications 2025（DIA 单细胞工作流）**  
   <https://www.nature.com/articles/s41467-025-65174-4>  
   提供了 DIA 单细胞 workflow 中缺失处理比较与下游任务评估语境。

2. **MultiPro（Nature Communications 2025）**  
   <https://pubmed.ncbi.nlm.nih.gov/40947414/>  
   强调缺失率分层（10%~50%）及 MNAR:MCAR 混合机制评估。

3. **NAguideR（Bioinformatics 2020）**  
   <https://pmc.ncbi.nlm.nih.gov/articles/PMC7641313/>  
   支持多方法并行比较与 NRMSE 等重建指标。

4. **PIMMS（Nature Methods 2024）**  
   <https://pmc.ncbi.nlm.nih.gov/articles/PMC10949645/>  
   支持使用相关性等恢复指标并关注单细胞场景。

## 2. 推荐路径配置

## 2.1 数据

- `pxd054343_diann_2x`（本地 DIA-NN）：  
  `data/dia/diann/PXD054343/2_3_SC_SILAC_2x_report.tsv`
- `lfqbench_hye124_spectronaut`（在线 LFQbench 示例）：  
  <https://raw.githubusercontent.com/IFIproteomics/LFQbench/master/ext/data/vignette_examples/hye124/Spectronaut_TTOF6600_64w_example.tsv>

## 2.2 方法池

- 基线：`none`, `half_row_min`(min/2), `row_mean`
- 传统：`knn`, `lls`, `missforest`, `iterative_svd`
- 矩阵补全：`softimpute`（依赖 `fancyimpute`）

## 2.3 评估设定

- 预处理固定：`log_transform -> normalize(mean)`
- 缺失机制：`mcar`, `mixed_mnar`
- 缺失比例：`0.1`, `0.3`, `0.5`
- 只在 holdout 位置计算恢复误差（masked-value recovery）

## 2.4 指标

- 重建层：
  - `nrmse`, `mae`, `rmse`, `pearson_r`, `spearman_r`, `holdout_coverage`
- 稳定性：
  - `within_group_cv_median`
- 聚类保真：
  - `cluster_ari`, `cluster_nmi`, `cluster_asw`, `cluster_knn_purity`
- DE 信号一致性（任务导向）：
  - `de_log2fc_pearson`, `de_topk_jaccard`, `de_topk_sign_agreement`
- LFQbench 额外真值指标：
  - `ratio_pairwise_auc_mean`, `ratio_changed_vs_bg_auc`, `ratio_mae`, `ratio_rmse`
- 工程维度：
  - `runtime_sec`, `post_missing_rate`, `success_rate`

## 3. 运行方式（uv）

在仓库根目录执行（推荐路径）：

```bash
UV_CACHE_DIR=/tmp/.uv-cache uv run --no-project python benchmark/imputation/run_benchmark.py \
  --datasets pxd054343_diann_2x lfqbench_hye124_spectronaut \
  --methods none half_row_min row_mean knn lls iterative_svd softimpute missforest \
  --holdout-rates 0.1 0.3 0.5 \
  --mechanisms mcar mixed_mnar \
  --repeats 1 \
  --normalization mean \
  --max-features 500 \
  --output-dir benchmark/imputation/outputs
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

## 5. 本轮执行结论（2026-03-05）

- 推荐路径已执行完成（96 runs）。
- 所有方法运行成功（`n_failed_runs = 0`）。
- 评分包含 `success_rate` 惩罚项（`score_success_rate`），可防止失败方法误导排序。
- 在当前数据与配置下，`knn` / `lls` 在两套数据中表现最稳定；`softimpute` 在 LFQbench 上进入前列，但在 PXD054343 上不占优；`missforest` 精度接近但耗时显著更高。

## 6. Git 大文件控制

`benchmark/**/data/` 与 `benchmark/**/outputs/` 已在 `.gitignore` 忽略。
