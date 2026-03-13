# 蛋白层去批次（Integration）基准测试

本目录用于评测 `scptensor.integration` 在 **DIA 单细胞蛋白组** 场景下的去批次表现。

当前入口：

- `benchmark/integration/run_benchmark.py`
- 该入口当前直接委托 `benchmark/integration/run_real_dia_batch_confounding.py`

## 1. 当前实现与目标合同

基于文献建议（DIA-SCP workflow benchmark + scplainer + protein-level batch benchmark），integration benchmark 的目标合同是三场景；但当前代码已落地的是其中两类：

1. **当前实现：两场景**
   - `balanced_amount_by_sample`：
     - `batch = sample_group(S1/S2/S3)`
     - `condition = amount_group(300ng/600ng)`
     - 用于评估“去批次 + 生物信号保留”
   - `confounded_amount_as_batch`：
     - `batch = condition = amount_group`
     - 语义上属于 `fully confounded`
     - 用于评估“混杂设计鲁棒性 + 报错机制”

2. **文献建议但当前待补：`partially confounded`**
   - review 已要求把 `balanced / partially confounded / fully confounded` 三类拆开
   - 当前脚本尚未生成独立 `partially confounded` 场景

3. **双维度指标**
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

4. **场景化评分**
   - `balanced`：同时纳入 batch + condition 指标
   - `confounded`：仅纳入 batch removal + runtime（避免把“保留 confounded 标签”误当生物保真）
   - 当前输出是“按场景分别评分”；不应跨 `balanced` 与 `fully confounded` 合并出单一总榜

5. **guardrail 检查**
   - 对 `combat` / `limma` 在带协变量建模下执行显式检查：
     - balanced 场景预期成功
     - confounded 场景预期 rank-deficient 报错

6. **当前未落地**
   - `partially confounded` 独立赛道
   - state-aware completeness / uncertainty burden
   - 真实公共 panel 上的多数据集统一比较
   - 去除对 `overall_scores.png` 这类跨场景聚合图的主榜依赖

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
- `Quartet` 参考样本比值法表明，在强混杂场景下，如果实验设计阶段就有共同 reference material / bridge sample，可以显著提升跨批次可比性；这为后续单独补 `partially confounded + bridge/reference` 场景提供了设计依据，但不等于当前脚本已经覆盖：<https://genomebiology.biomedcentral.com/articles/10.1186/s13059-023-03047-z>

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
- `benchmark/integration/outputs/overall_scores.png`（legacy 聚合图；解释时优先看按场景分开的 `metrics_scores.csv`）
- `benchmark/integration/outputs/real_dia_batch_confounding_report.json`（legacy 兼容）

## 4. 示例实测快照（2026-03-05，可过时）

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

### 4.2 Fully Confounded 场景（guardrail 评估）

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

以上分数仅代表一次历史运行快照，后续评测请以当前代码与当前数据重新执行得到的结果为准。

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
