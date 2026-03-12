# 蛋白层缺失值填充（Imputation）基准测试

本目录用于评测 `scptensor.impute` 在 **DIA 单细胞蛋白组预处理场景** 下的表现。

该 benchmark 当前主要对齐以下两篇综述：

- `docs/dia_sc_imputation_literature_review_20260304.md`
- `docs/dia_sc_masked_value_benchmark_design_review_20260312.md`

当前实现边界：

- 已落地的是 protein-level `masked-value recovery` 主榜。
- `precursor-to-protein auxiliary board` 尚未在本目录单独实现。
- holdout 是从当前矩阵中已观测的有限值构造，并未按原始 `MaskCode` 分层，因此还不能称为 state-aware masking benchmark。

## Resource Roles

- `论文证据`
  - Wang 2025 / Harris 2023 / NAguideR / PIMMS
- `数据入口`
  - `pxd054343_diann_2x` 本地 DIA-NN 输入
  - LFQbench HYE124 Spectronaut TSV
- `模块规范 / 软件文档`
  - 当前 README 主要依赖 review 合同，不单列外部 module spec 作为主入口
- `资源包`
  - 当前 README 不把 package 页面写成主 benchmark 输入

经二次核查的来源性细节：

- `PIMMS` 明确展示了“移除 20% 强度后再比较恢复质量”的 masked-value 评估思路，并把显著蛋白恢复能力作为下游效应的一部分：<https://www.nature.com/articles/s41467-024-48711-5>
- `NAguideR` 在三套蛋白组数据上把随机产生的 missingness 从 5% 拉到 70%（每 5% 一档），并在每个比例点重复插补与评估；这支持把“多缺失比例、多轮评估”视为更完整的 literature-style 协议：<https://pmc.ncbi.nlm.nih.gov/articles/PMC7641313/>
- `PIMMS` 的摘要层明确报告“对已知信号随机移除 20% 再评估”，但模型开发内部还可能使用额外 masking 细节；ScpTensor 当前借鉴的是其“外部 masked-value evaluation”思想，而不是整套深度学习训练协议：<https://www.nature.com/articles/s41467-024-48711-5>
- `Harris et al. 2023` 在 5 套蛋白组数据上比较 10 种方法，并把 differential abundance、可定量 analyte 数与 lower limit of quantification 纳入评价；这直接支持当前文档把 downstream 指标与 retained-feature 类指标纳入主协议：<https://pubmed.ncbi.nlm.nih.gov/37861703/>

该 benchmark 已根据上述综述调整为当前推荐路径：
- 推荐方法池（基线 + 传统 + 矩阵补全）
- 多缺失机制（MCAR + mixed MNAR）
- 多缺失比例（10% / 30% / 50%）
- 任务导向指标（恢复误差 + 聚类保真 + DE 信号一致性）

## 1. 文献依据（摘要）

1. **Nature Communications 2025（DIA 单细胞工作流）**
   <https://www.nature.com/articles/s41467-025-65174-4>
   提供了 DIA 单细胞 workflow 中缺失处理比较与下游任务评估语境。

2. **Harris et al.（Journal of Proteome Research 2023）**
   <https://pubmed.ncbi.nlm.nih.gov/37861703/>
   明确提出插补 benchmark 不应只看 reconstruction error，而要联合 differential abundance、可定量 analyte 数与 lower limit of quantification。

3. **NAguideR（Bioinformatics 2020）**
   <https://pmc.ncbi.nlm.nih.gov/articles/PMC7641313/>
   支持多方法并行比较与 NRMSE 等重建指标。

4. **PIMMS（Nature Communications 2024）**
   <https://www.nature.com/articles/s41467-024-48711-5>
   支持在高缺失 LFQ 蛋白组上使用外部 masked-value recovery，并联合显著蛋白恢复等 downstream 证据；它不是单细胞专文，但对高稀疏 DIA-sc benchmark 具有可迁移价值。

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

- 预处理默认：`log_transform -> normalize(mean)`
- 可通过 `--normalization` 改为 `none / mean / median / quantile / trqn`
- 缺失机制：`mcar`, `mixed_mnar`
- 缺失比例：`0.1`, `0.3`, `0.5`
- 只在 holdout 位置计算恢复误差（masked-value recovery）

当前默认 vs review 目标：

- 当前脚本默认：
  - `holdout_rates = 0.1 / 0.3 / 0.5`
  - `repeats = 1`
- review 更偏好的主协议：
  - 主区间优先 `0.1 / 0.2 / 0.3`
  - `0.5` 作为压力测试
  - 资源允许时 `repeats >= 5`，更稳妥时参考 `NAguideR` 风格的多档比例 + 多轮重复

解释：

- 当前 `0.1 / 0.3 / 0.5, repeats = 1` 更接近“轻量可复现默认”，便于快速跑通。
- 若目标是更贴近 literature-style 稳健比较，应补上更密的 holdout 网格与重复 masking，而不是把一次性结果当作最终排名。
- 因此当前默认更适合快速回归或脚本冒烟，不应被表述为“最终文献级主协议”。

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

说明：

- 当前实现已经有 `DE consistency` 指标，但尚未输出 review 目标中的 `DE pAUC / F1` 与 `retained proteins`。
- `Nature Communications 2024` 的 DEA workflow benchmark 明确把 `pAUC(0.01/0.05/0.1) + nMCC + G-mean` 组合用于 workflow ranking；因此如果后续扩展 imputation benchmark 的 DE 终点，优先补 `pAUC / confusion-matrix` 一类指标比继续堆更多相关系数更有文献依据：<https://www.nature.com/articles/s41467-024-47899-w>

待补项（按 review 合同）：

- 按 `LOD / MBR / FILTERED / UNCERTAIN` 分层的 holdout 协议
- state-aware completeness / uncertainty burden
- `precursor-to-protein` 辅榜
- `DE pAUC / F1` 与 `retained proteins`

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

## 5. 示例执行快照（2026-03-05，可过时）

- 推荐路径已执行完成（96 runs）。
- 所有方法运行成功（`n_failed_runs = 0`）。
- 评分包含 `success_rate` 惩罚项（`score_success_rate`），可防止失败方法误导排序。
- 在当前数据与配置下，`knn` / `lls` 在两套数据中表现最稳定；`softimpute` 在 LFQbench 上进入前列，但在 PXD054343 上不占优；`missforest` 精度接近但耗时显著更高。
- 该段仅作为一次历史运行示例，实际排序应以你当前环境下重新运行 benchmark 结果为准。

## 6. Git 大文件控制

`benchmark/**/data/` 与 `benchmark/**/outputs/` 已在 `.gitignore` 忽略。
