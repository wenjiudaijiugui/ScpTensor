# 蛋白层去批次（Integration）基准测试

本目录用于评测 `scptensor.integration` 在 **DIA 单细胞蛋白组** 场景下的去批次表现。

## 1. 重构后的 benchmark 思路

基于文献建议（DIA-SCP workflow benchmark + scplainer + protein-level batch benchmark），本 benchmark 从“单一场景”升级为：

1. **双场景设计**
   - `balanced_amount_by_sample`：
     - `batch = sample_group(S1/S2/S3)`
     - `condition = amount_group(300ng/600ng)`
     - 用于评估“去批次 + 生物信号保留”
   - `confounded_amount_as_batch`：
     - `batch = condition = amount_group`
     - 用于评估“混杂设计鲁棒性 + 报错机制”

2. **双维度指标**
   - Batch removal：
     - `between_batch_ratio`（低好）
     - `batch_asw`（低好）
     - `batch_mixing`（高好）
     - `lisi_approx`（高好）
   - Biological conservation（balanced 场景重点）：
     - `between_condition_ratio`（高好）
     - `condition_asw`（高好）
     - `condition_ari`（高好）
     - `condition_nmi`（高好）
     - `condition_knn_purity`（高好）

3. **场景化评分**
   - `balanced`：同时纳入 batch + condition 指标
   - `confounded`：仅纳入 batch removal + runtime（避免把“保留 confounded 标签”误当生物保真）

4. **guardrail 检查**
   - 对 `combat` / `limma` 在带协变量建模下执行显式检查：
     - balanced 场景预期成功
     - confounded 场景预期 rank-deficient 报错

## 2. 运行方式（uv）

在仓库根目录执行：

```bash
UV_CACHE_DIR=/tmp/.uv-cache uv run --no-project python benchmark/integration/run_benchmark.py \
  --output-dir benchmark/integration/outputs
```

可选参数：

```bash
UV_CACHE_DIR=/tmp/.uv-cache uv run --no-project python benchmark/integration/run_benchmark.py \
  --scenarios balanced_amount_by_sample confounded_amount_as_batch \
  --methods combat_parametric combat_nonparametric limma mnn \
  --fdr-threshold 0.01
```

## 3. 统一输出文件

- `benchmark/integration/outputs/metrics_raw.csv`
- `benchmark/integration/outputs/metrics_summary.csv`
- `benchmark/integration/outputs/metrics_scores.csv`
- `benchmark/integration/outputs/guardrail_checks.csv`
- `benchmark/integration/outputs/run_metadata.json`
- `benchmark/integration/outputs/summary_metrics.png`
- `benchmark/integration/outputs/score_heatmap.png`
- `benchmark/integration/outputs/overall_scores.png`
- `benchmark/integration/outputs/real_dia_batch_confounding_report.json`（legacy 兼容）

## 4. 本次实测结果（2026-03-05）

数据：
- `data/dia/diann/PXD054343/1_SC_LF_report.tsv`
- `24` samples, `2719` proteins

### 4.1 Balanced 场景（主评估）

按 `overall_score` 排名：

1. `combat_parametric`：`0.9487`
2. `limma`：`0.8967`
3. `combat_nonparametric`：`0.8512`
4. `mnn`：`0.1034`
5. `none`：`0.0909`

关键观察：
- `combat_parametric` 与 `limma` 在 batch 去除上都很强，且 condition 相关指标保持较好（`condition_ari=1.0`, `condition_nmi=1.0`）。
- `mnn` 在该数据上 batch 去除与生物保真均落后于线性方法。

### 4.2 Confounded 场景（鲁棒性评估）

按 `overall_score` 排名（仅 batch 维度参与评分）：

1. `mnn`：`0.7804`
2. `limma`：`0.6261`
3. `combat_parametric`：`0.6156`
4. `combat_nonparametric`：`0.6079`
5. `none`：`0.4630`

### 4.3 Guardrail 结果

- balanced 场景：
  - `combat_parametric_with_covariate`：成功（符合预期）
  - `limma_with_covariate`：成功（符合预期）
- confounded 场景：
  - `combat_parametric_with_covariate`：`rank deficient` 报错（符合预期）
  - `limma_with_covariate`：`rank deficient` 报错（符合预期）

总通过率：`guardrail_pass_rate = 1.0`

## 5. 参考依据（设计层面）

- DIA-SCP workflow benchmark: <https://www.nature.com/articles/s41467-025-65174-4>
- scplainer benchmark: <https://genomebiology.biomedcentral.com/articles/10.1186/s13059-025-03713-4>
- Protein-level batch benchmark: <https://www.nature.com/articles/s41467-025-64718-y>
- scIB metrics framework: <https://doi.org/10.1038/s41592-021-01336-8>
- ComBat original: <https://pubmed.ncbi.nlm.nih.gov/16632515/>
- limma framework: <https://pubmed.ncbi.nlm.nih.gov/25605792/>
- MNN original: <https://pubmed.ncbi.nlm.nih.gov/29608177/>
