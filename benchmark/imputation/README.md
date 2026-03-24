# 蛋白层缺失值填充（Imputation）基准测试

本目录用于评测 `scptensor.impute` 在 **DIA 单细胞蛋白组预处理场景** 下的表现。

## Current Entry

- `benchmark/imputation/run_benchmark.py`

## Boundary

该 benchmark 当前主要对齐以下两篇综述：

- `docs/review_imputation_20260304.md`
- `docs/review_masked_imputation_20260312.md`

当前实现边界：

- 已落地的是 protein-level `masked-value recovery` 主榜。
- 已落地 `precursor-to-protein auxiliary board`：
  `Spectronaut peptide-long -> log/normalize -> precursor holdout/imputation -> aggregate_to_protein -> protein-level scoring`
- holdout 已升级为基于 source-layer `MaskCode` 的 state-aware masking benchmark：
  当前只会在“当前有限值且可恢复”的位置上做 holdout，但会按 `all_observed / valid / mbr / lod / uncertain`
  等 strata 分层；无可恢复条目的 strata 会在 `run_metadata.json` 里记录为 skipped，而不是伪造结果。
- 脚本现已内置 `smoke / default / literature` 三档 tier profile，并允许 `--board main / auxiliary / both`。
- 输出里若出现 `raw_norm_median_knn` 一类 layer 名，它们是比较过程的 artifact naming；若后续要把选中结果纳入 stable mainline，仍需显式 promote 到 canonical `imputed` 层并保留 provenance。
- 当前默认数据集、默认 holdout 比例、默认 `normalization` 设定和默认方法池，只表示今天仓库默认怎么跑，不应反向改写长期 benchmark 合同。

## Review Links

- `docs/review_imputation_20260304.md`
- `docs/review_masked_imputation_20260312.md`
- `docs/review_state_metrics_20260312.md`
- `docs/imputation_contract.md`

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

## 1. 长期证据边界

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

## 2. 当前默认推荐路径（实现事实，不构成合同）

当前默认脚本目前按下述推荐路径组织；这只是当前实现事实，不等价于长期合同：

- 推荐方法池（基线 + 传统 + 矩阵补全）
- 多缺失机制（MCAR + mixed MNAR）
- 多缺失比例（10% / 30% / 50%）
- 任务导向指标（恢复误差 + 聚类保真 + DE 信号一致性）
- 双榜结构：
  - `main`: protein-direct masked recovery
  - `auxiliary`: precursor holdout -> protein endpoint

### 2.1 数据

- `pxd054343_diann_2x`（本地 DIA-NN）：
  `data/dia/diann/PXD054343/2_3_SC_SILAC_2x_report.tsv`
- `lfqbench_hye124_spectronaut`（在线 LFQbench 示例）：
  <https://raw.githubusercontent.com/IFIproteomics/LFQbench/master/ext/data/vignette_examples/hye124/Spectronaut_TTOF6600_64w_example.tsv>

### 2.2 方法池

- 基线：`none`, `half_row_min`(min/2), `row_mean`
- 传统：`knn`, `lls`, `missforest`, `iterative_svd`
- 矩阵补全：`softimpute`（依赖 `fancyimpute`）

文献级首版运行策略：

- `tier = literature` 的默认方法池现在收紧为
  `none / half_row_min / row_mean / knn / lls / iterative_svd`
- `missforest` 仍保留为可显式选择的方法，但不再放在默认 literature 首跑里，
  原因是其 wall-clock 成本会把双榜实跑拉成数小时级
- `softimpute` 仍保留为可显式选择的方法，但需要先安装 `fancyimpute`；
  若显式请求而环境缺依赖，runner 会在启动前 fail closed，而不是把整批 run
  记成运行期失败

### 2.3 评估设定

- 预处理默认：`log_transform -> normalize(mean)`
- 可通过 `--normalization` 改为 `none / mean / median / quantile / trqn`
- 缺失机制：`mcar`, `mixed_mnar`
- 缺失状态分层：默认 `all_observed`, `valid`, `mbr`, `lod`, `uncertain`
- 缺失比例：`0.1`, `0.3`, `0.5`
- 只在 holdout 位置计算恢复误差（masked-value recovery）
- `metrics_raw.csv` 保留逐 seed 结果；`metrics_summary.csv` 按
  `dataset / method / mechanism / holdout_rate / holdout_state` 做 seed summary

当前默认 vs review 目标：

- 当前脚本默认：
  - `tier = default`
  - `board = main`
  - `holdout_rates = 0.1 / 0.3 / 0.5`
  - `holdout_states = all_observed / valid / mbr / lod / uncertain`
  - `repeats = 1`
- review 更偏好的主协议：
  - `tier = literature`
  - `board = both`
  - 主区间优先 `0.1 / 0.2 / 0.3`
  - `0.5` 作为压力测试
  - 资源允许时 `repeats >= 5`，更稳妥时参考 `NAguideR` 风格的多档比例 + 多轮重复

解释：

- 当前 `0.1 / 0.3 / 0.5, repeats = 1` 更接近“轻量可复现默认”，便于快速跑通。
- 若目标是更贴近 literature-style 稳健比较，应补上更密的 holdout 网格与重复 masking，而不是把一次性结果当作最终排名。
- 因此当前默认更适合快速回归或脚本冒烟，不应被表述为“最终文献级主协议”。
- 当前 literature 默认方法池也采用“先保证可复跑”的策略：
  可选依赖方法和超重方法默认不进入首版双榜，只有显式 `--methods` 才会加入。

### 2.4 指标

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
- state-aware holdout 维度：
  - `holdout_state`, `holdout_state_fraction`, `holdout_rate_within_state`
  - `state_valid_fraction / state_mbr_fraction / state_lod_fraction / state_uncertain_fraction`
  - `state_direct_observation_rate / state_supported_observation_rate / state_uncertainty_burden`

说明：

- 当前实现除 `DE consistency` 外，已额外输出：
  - `de_topk_f1`
  - `de_pauc_01 / 05 / 10`
  - `retained_proteins_ratio`
  - `fully_observed_proteins_ratio`
- 这些 `DE pAUC / F1` 当前仍是 **pre-holdout truth vs imputed contrast** 的 task proxy，不应被过度解释成外部金标准 DEA benchmark。
- `cluster_*` 与 `DE_*` 指标在这里是下游评估终点，用于判断插补是否破坏后续分析；它们不把 `cluster` 本身提升为 stable preprocessing contract。
- `Nature Communications 2024` 的 DEA workflow benchmark 明确把 `pAUC(0.01/0.05/0.1) + nMCC + G-mean` 组合用于 workflow ranking；因此如果后续扩展 imputation benchmark 的 DE 终点，优先补 `pAUC / confusion-matrix` 一类指标比继续堆更多相关系数更有文献依据：<https://www.nature.com/articles/s41467-024-47899-w>

仍待补项（按 review 合同）：

- 更强的外部 ground-truth 驱动 `DE pAUC / F1`
- auxiliary board 的更多公共 peptide/precursor 数据面板

## 3. 运行方式（uv）

在仓库根目录执行（推荐路径）：

```bash
UV_CACHE_DIR=/tmp/.uv-cache uv run python benchmark/imputation/run_benchmark.py \
  --tier default \
  --board main \
  --datasets pxd054343_diann_2x lfqbench_hye124_spectronaut \
  --normalization mean \
  --max-features 500 \
  --output-dir benchmark/imputation/outputs
```

文献级双榜运行：

```bash
UV_CACHE_DIR=/tmp/.uv-cache uv run python benchmark/imputation/run_benchmark.py \
  --tier literature \
  --board both \
  --normalization mean \
  --aux-aggregation-method sum \
  --output-dir benchmark/imputation/outputs
```

若要显式加入扩展方法：

```bash
UV_CACHE_DIR=/tmp/.uv-cache uv run python benchmark/imputation/run_benchmark.py \
  --tier literature \
  --board both \
  --methods none half_row_min row_mean knn lls iterative_svd missforest \
  --normalization mean \
  --output-dir benchmark/imputation/outputs
```

若要加入 `softimpute`，需先安装 `fancyimpute`；否则脚本会在启动前报错并停止。

## 4. 输出文件

- `--board main` 或 `--board auxiliary`：
  - `benchmark/imputation/outputs/metrics_raw.csv`
  - `benchmark/imputation/outputs/metrics_summary.csv`
  - `benchmark/imputation/outputs/metrics_scores.csv`
  - `benchmark/imputation/outputs/run_metadata.json`
  - PNG 图表
- `--board both`：
  - `benchmark/imputation/outputs/main/*`
  - `benchmark/imputation/outputs/auxiliary/*`
  - 根目录 `benchmark/imputation/outputs/run_metadata.json` 用于指向双榜子目录
  - 根级 metadata 会在每个 board 完成后增量更新，因此即使双榜长跑在第二块中断，已完成的 board 仍可被审计

长跑 checkpoint 语义：

- runner 现在会周期性刷新当前 board 的 `metrics_raw.csv / metrics_summary.csv / metrics_scores.csv / run_metadata.json`
- 若长跑被中断，当前 board 的 `run_metadata.json` 会记录 `run_status = interrupted`
- `--board both` 时，根级 `run_metadata.json` 也会记录各 board 的 `pending / running / completed / interrupted` 状态

## 5. 附录：历史运行快照（2026-03-05，非合同）

这一段只保留一次历史运行的最小审计信息，不参与当前合同解释，也不代表当前环境下的默认赢家：

- 当时的推荐路径共完成 `96` 次 run，`n_failed_runs = 0`，并已纳入 `success_rate` 惩罚项以抑制失败方法误导排序。
- 该次运行里，`knn / lls` 在两套数据上相对稳定，`softimpute` 在 LFQbench 上更强但在 `PXD054343` 上不占优，`missforest` 精度接近但耗时偏高。
- 若需要当前可复核比较，应重新运行脚本，并以当次产出的 `metrics_scores.csv` 与 `run_metadata.json` 为准，而不是引用本节文字摘要。

## 6. Git 大文件控制

`benchmark/**/data/` 与 `benchmark/**/outputs/` 已在 `.gitignore` 忽略。
