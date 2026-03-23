# 蛋白层去批次（Integration）基准测试

本目录用于评测 `scptensor.integration` 在 **DIA 单细胞蛋白组** 场景下的去批次表现。

## Current Entry

- `benchmark/integration/run_benchmark.py`
- 该入口当前直接委托 `benchmark/integration/run_real_dia_batch_confounding.py`

## Boundary

- 当前目录已落地 `balanced`、`partially confounded` 与 `fully confounded` 三类场景。
- 主解释必须基于按场景拆分的评分结果，而不是跨场景单一总榜。
- `fully confounded` 的正确角色是 guardrail / failure-style 场景，不是常规赢家榜单。
- 当前目录的主榜只覆盖 protein-level、matrix-level integration 方法；embedding-level `reduce / cluster` 只能作为外部评估终点或 exploratory panel，不能混入 integration 主榜边界。
- 若输出中出现 `limma`、`combat`、`mnn_corrected`、`scanorama` 等 layer 名，它们是 method-specific compatibility / artifact naming，不等同于仓库 canonical `raw / log / norm / imputed / zscore` taxonomy。
- 当前默认场景集合、当前 guardrail 输出资产和当前默认方法列表，只表示今天仓库默认怎么跑，不应反向改写长期场景合同。

## Review Links

- `docs/review_batch_correction_20260305.md`
- `docs/review_batch_diagnostics_20260312.md`
- `docs/review_batch_confounding_20260312.md`
- `docs/integration_contract.md`

## Resource Roles

- `论文证据`
  - Wang 2025 / scplainer / Zheng 2025 / scIB / Nygaard / Song / Chazarra-Gil / Quartet
- `数据入口`
  - 当前脚本主输入是本地 `PXD054343` DIA-NN 报告；稳定公开数据入口的分型以公共 benchmark 数据综述为准
- `模块规范 / 软件文档`
  - 当前 README 不把任何单一 module page 误写成真实数据入口
- `资源包`
  - `scpdata` 属于 reference/resource layer，不等价于 integration 主输入数据页

经二次核查的来源性细节：

- `Nygaard et al.`（Biostatistics）明确指出：当 batch 与 group 存在不平衡/混杂时，校正后做 downstream 检验可能产生“被夸大的信心”；因此 `fully confounded` 不应被当作普通主榜场景：<https://pubmed.ncbi.nlm.nih.gov/26272994/>
- `Song et al.`（Nature Communications 2020）把实验设计显式拆成 `completely randomized / reference panel / chain-type`，并直接指出在 complete confounding 下 biological effect 不可辨识；这支持把 `fully confounded` 定义为 guardrail/failure 场景而非赢家场景：<https://www.nature.com/articles/s41467-020-16905-2>
- `Chazarra-Gil et al.`（Nature Communications 2023）在 integration-DE benchmark 中把 `balanced` 与 `unbalanced` 设计分开仿真和评估，说明场景不分拆会混淆解释：<https://www.nature.com/articles/s41467-023-37126-3>
- `scIB` 把 integration accuracy 明确拆成 `batch removal` 与 `bio-conservation` 两类指标，同时把 usability / scalability 单独评估；这支持把当前文档中的双轴指标与策略层解释分开：<https://www.nature.com/articles/s41592-021-01336-8>
- `scib-metrics` 的 `Benchmarker` 接口同样要求显式传入 `bio_conservation_metrics` 与 `batch_correction_metrics`，说明这两组指标在现代 benchmark 工具层也是并列对象：<https://scib-metrics.readthedocs.io/en/stable/notebooks/large_scale.html>
- `Quartet` 参考样本比值法表明，在强混杂场景下，如果实验设计阶段就有共同 reference material / bridge sample，可以显著提升跨批次可比性；当前脚本新增的 `partially_confounded_bridge_sample` 正是受这一设计启发，但它仍只是最小 bridge-style 场景，而不是完整参考面板复刻：<https://genomebiology.biomedcentral.com/articles/10.1186/s13059-023-03047-z>
- `Gong et al.`（Analytical Chemistry 2025）直接在 single-cell proteomics integration benchmark 中把评价拆成 `batch correction`、`biology preservation` 与 `marker consistency` 三类；当前脚本现已把 `marker consistency` 作为第三报告轴输出到表格与总览图中，但仍保持“报告轴”和“当前主评分权重”分离：<https://doi.org/10.1021/acs.analchem.4c04933>

## 1. 长期场景与评分边界

长期合同要求 integration benchmark 最终覆盖三类场景；当前默认脚本现已全部落地：

1. **当前实现：三场景**
   - `balanced_amount_by_sample`：
     - `batch = sample_group(S1/S2/S3)`
     - `condition = amount_group(300ng/600ng)`
     - 用于评估“去批次 + 生物信号保留”
   - `partially_confounded_bridge_sample`：
     - 子集化设计：`S1 -> 300ng only`, `S2 -> 300ng/600ng bridge`, `S3 -> 600ng only`
     - `batch = sample_group`
     - `condition = amount_group`
     - 语义上属于 `partially confounded with bridge/reference`
     - 用于评估“可识别但不完全平衡设计下的鲁棒性”
   - `confounded_amount_as_batch`：
     - `batch = condition = amount_group`
     - 语义上属于 `fully confounded`
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
   - Marker consistency（read-only 第三报告轴）：
     - `marker_log2fc_pearson`（高好）
     - `marker_topk_jaccard`（高好）
     - `marker_topk_sign_agreement`（高好）
     - 当前用于报告，不直接并入 `overall_score`

3. **场景化评分**
   - `balanced`：同时纳入 batch + condition 指标
   - `partially_confounded`：仍纳入 batch + condition 指标，但解释时必须同时带上 `design_identifiability`
   - `confounded`：仅纳入 batch removal + runtime（避免把“保留 confounded 标签”误当生物保真）
   - 当前输出是“按场景分别评分”；不应跨 `balanced` 与 `fully confounded` 合并出单一总榜

4. **guardrail 检查**
   - 对 `combat` / `limma` 在带协变量建模下执行显式检查：
     - balanced 场景预期成功
     - partially confounded 场景预期成功
     - confounded 场景预期 rank-deficient 报错

5. **当前仍未落地**
   - state-aware completeness / uncertainty burden
   - 真实公共 panel 上的多数据集统一比较
   - 去除对 `overall_scores.png` 这类跨场景聚合图的主榜依赖
   - 真正 external marker panel 驱动的 feature consistency（当前第三轴仍是 baseline-vs-integrated 的 marker proxy）

## 2. 运行方式（uv）

在仓库根目录执行：

```bash
UV_CACHE_DIR=/tmp/.uv-cache uv run python benchmark/integration/run_benchmark.py \
  --output-dir benchmark/integration/outputs
```

可选参数：

```bash
UV_CACHE_DIR=/tmp/.uv-cache uv run python benchmark/integration/run_benchmark.py \
  --scenarios balanced_amount_by_sample partially_confounded_bridge_sample confounded_amount_as_batch \
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
- `benchmark/integration/outputs/overall_scores.png`（legacy 聚合图；解释时优先看按场景分开的 `metrics_scores.csv`）
- `benchmark/integration/outputs/real_dia_batch_confounding_report.json`（legacy 兼容）

当前输出表还会显式携带：

- `design_identifiability`
- `n_samples_scenario`

用于区分 balanced / bridge-partial / fully-confounded 三类场景的解释边界。

## 4. 附录：历史运行快照（2026-03-05，非合同）

这一段只保留一次历史运行的最小审计信息，用于解释旧输出资产，不参与当前合同解释：

- 数据：`data/dia/diann/PXD054343/1_SC_LF_report.tsv`，`24` samples，`2719` proteins
- `balanced` 场景下，当时是 `combat_parametric > limma > combat_nonparametric`，`mnn` 与 `none` 明显落后；这说明在线性可辨识设计里，线性校正方法当时更适配该数据。
- `fully confounded` 场景下，批次维度打分一度把 `mnn` 排到前面；但该场景本来就是 guardrail/failure-style 设计，不能把这类分数解释成“integration 主榜赢家”。
- guardrail 检查当时全部符合预期：balanced 场景下带协变量的 `combat_parametric` / `limma` 成功；fully confounded 场景下二者都因 `rank deficient` 失败。
- 若需要当前可复核比较，应重新运行脚本，并以当次 `metrics_scores.csv`、`guardrail_checks.csv` 和 `run_metadata.json` 为准。

## 5. 参考依据（设计层面）

- DIA-SCP workflow benchmark: <https://www.nature.com/articles/s41467-025-65174-4>
- scplainer benchmark: <https://genomebiology.biomedcentral.com/articles/10.1186/s13059-025-03713-4>
- Protein-level batch benchmark: <https://www.nature.com/articles/s41467-025-64718-y>
- scIB metrics framework: <https://doi.org/10.1038/s41592-021-01336-8>
- ComBat original: <https://pubmed.ncbi.nlm.nih.gov/16632515/>
- limma framework: <https://pubmed.ncbi.nlm.nih.gov/25605792/>
- MNN original: <https://pubmed.ncbi.nlm.nih.gov/29608177/>

对应综述：

- `docs/review_batch_correction_20260305.md`
- `docs/review_batch_diagnostics_20260312.md`
- `docs/review_batch_confounding_20260312.md`
