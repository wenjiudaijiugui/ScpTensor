# ScpTensor 运行时基线说明（PR-0，2026-03-17）

## 1. 文档定位

本文档定义 `PR-0` 的运行时基线工具，用于后续代码优化前后的 **runtime / memory / densify / copy-path** 对比。

它的角色是：

- 为后续优化提供工程回归基线
- 补足当前 `benchmark/` 目录不覆盖的 runtime 观测面
- 服务于 `docs/optimization_checklist.md` 中的 `PR-0`

它不是：

- 方法学 benchmark 排名文档
- 稳定实现合同
- 文献综述

若本文档与合同文档冲突，仍以 `AGENTS.md` 与对应 `docs/*_contract.md` 为准。

## 2. Authority Docs

执行 `PR-0` 运行时基线时，优先依据：

1. `AGENTS.md`
2. `docs/optimization_checklist.md`
3. 对应模块合同：
   - `docs/core_data_contract.md`
   - `docs/core_compute_contract.md`
   - `docs/io_diann_spectronaut.md`
   - `docs/aggregation_contract.md`
   - `docs/transformation_contract.md`
   - `docs/normalization_contract.md`
   - `docs/imputation_contract.md`
   - `docs/integration_contract.md`
   - `docs/qc_contract.md`

## 3. 与 `benchmark/` 的边界

`benchmark/` 当前负责：

- 方法质量
- 公开数据 replay
- literature-aligned scoring

`scripts/perf/run_runtime_baseline.py` 负责：

- wall-time
- RSS/peak memory
- sparse -> dense 边界观察
- 返回对象 / 共享引用 / copy path 观察

因此：

- 不要把 runtime baseline 结果写成“方法学优劣”
- 不要把科学 benchmark 排名当作运行时优化依据

## 4. 当前脚本

入口脚本：

- `scripts/perf/run_runtime_baseline.py`

默认输出目录：

- `outputs/runtime_baseline/`

输出资产：

- `stage_runs.csv`
- `scenario_summary.json`
- `environment.json`
- `errors.json`
- `gate_results.json`（仅在传入 `--gate-policy` 时生成）

这些目录默认被 `.gitignore` 忽略，不作为仓库提交内容。

`environment.json` 还应记录会影响 sparse log 分支选择的环境量，例如：

- `SCPTENSOR_JIT_THRESHOLD`
- `gate_policy`
- `gate_failures`

## 5. 当前场景

### 5.1 `import_diann_protein_long`

目的：

- 基线化 `load_diann()` 的 long-table 导入路径

当前观测：

- wall-time
- RSS peak
- 导入后 `proteins/raw` 的存储类型和字节规模

### 5.2 `aggregate_peptide_to_protein`

目的：

- 基线化 stable 主线中的 peptide/precursor -> protein 转换

当前方法：

- `aggregate_to_protein(..., method="sum")`

说明：

- 这里只测 stable 主线聚合成本
- 不把聚合质量结论混进 runtime baseline

### 5.3 `stable_chain_dense`

目的：

- 基线化稳定 dense 主链：
  - `log_transform`
  - `normalize(method="median")`
  - `impute(method="row_median")`
  - `integrate_limma`
  - `calculate_sample_qc_metrics`
  - `calculate_feature_qc_metrics`

### 5.4 `stable_chain_quantile`

目的：

- 单独补齐较重的 logged quantile normalization 路径

说明：

- 不把它当成默认主线
- 只把它当作性能回归的特定重路径

### 5.5 `stable_chain_trqn`

目的：

- 单独补齐较重的 logged TRQN normalization 路径

说明：

- 与 `stable_chain_quantile` 一样，它不是默认主线
- 它用于观察 rank-invariant 选择与 balanced quantile 步骤带来的运行时回归
- 当前沿用 quantile 链路同级别的 synthetic 输入规模，便于横向比较

### 5.6 `normalize_quantile_only`

目的：

- 只观测 logged `quantile` normalization 本体

说明：

- 该场景会先构造 dense protein 容器并预先生成 `log` 层
- baseline 只记录 `normalize_quantile` 这一个 stage
- 适合对比 quantile 内部排序 / rank 映射实现的前后变化

### 5.7 `normalize_trqn_only`

目的：

- 只观测 logged `TRQN` normalization 本体

说明：

- 该场景同样使用预先生成的 `log` 层
- baseline 只记录 `normalize_trqn` 这一个 stage
- 适合对比 rank-invariant 选择、balanced subset、内部 quantile 子步骤的前后变化

### 5.8 `sparse_transform_normalize`

目的：

- 观测当前 sparse 输入在 `log_transform -> normalize(median)` 下的存储边界

说明：

- 当前场景故意停在 normalization
- 不继续接 imputation，因为当前 stable imputation 语义仍以 `np.isnan(X)` 为缺失判定主入口
- 该场景主要用于捕获 sparse->dense 漂移
- `log_transform_sparse` 这一步同时承担 sparse log fast path 的工程基线观察；当前 `use_jit=True` 只表示允许自动判断，不代表默认一定进入 numba 分支

### 5.9 `sparse_log_only`

目的：

- 把 sparse `log_transform` 单独拆成 micro baseline

说明：

- 该场景只记录 `log_transform_sparse`
- 用于单独对比 sparse log 的 JIT / NumPy 分支、offset/base 透传与输出存储类型
- 不把 normalization 的 densify 成本混进这条 baseline

### 5.10 `autoselect_integrate_only`

目的：

- 单独基线化 stable `autoselect` 的 integration 选择阶段

说明：

- 输入使用预先生成的 stable baseline：`raw -> log -> norm -> imputed`
- 场景只记录 `AutoSelector.run_stage(stage="integrate")`
- 该场景观测的是选择层的调度 / 评估 / 结果保留开销，不把前处理链成本重复计入

### 5.11 `viz_qc_overview`

目的：

- 为 `viz` 提供 read-only runtime baseline

说明：

- 当前记录：
  - `plot_data_overview`
  - `plot_qc_completeness`
  - `plot_qc_matrix_spy`
- 这条 baseline 只服务于绘图读取层，不把 figure aesthetics 当成 benchmark 排名依据

## 6. 场景选择规则

### 6.1 Full-chain gate

以下场景用于确认一次优化没有把局部实现收益换成链路级回归：

- `stable_chain_dense`
  - 用于 stable dense 主链的通用改动
- `stable_chain_quantile`
  - 用于 logged `quantile` normalization 所在完整链路
- `stable_chain_trqn`
  - 用于 logged `TRQN` normalization 所在完整链路
- `sparse_transform_normalize`
  - 用于 sparse `log_transform`、sparse write path、densify 边界、JIT 选择相关改动

### 6.2 Micro gate

以下场景只测 normalization 本体，适合缩小热点定位范围：

- `normalize_quantile_only`
  - 只测 `normalize_quantile`
  - 适合比较排序、rank 映射、dense 临时分配变化
- `normalize_trqn_only`
  - 只测 `normalize_trqn`
  - 适合比较 rank-invariant 选择、balanced subset、内部 quantile 子步骤变化
- `sparse_log_only`
  - 只测 sparse `log_transform`
  - 适合比较 JIT 阈值、branch 选择和 sparse 输出保持情况
- `autoselect_integrate_only`
  - 只测 stable integration AutoSelect 的选择层
  - 适合比较 evaluator 调度、评分和 artifact 保留开销
- `viz_qc_overview`
  - 只测 read-only plotting 路径
  - 适合比较 `viz` 层的 figure 生成成本，而不混入 preprocessing 写路径

### 6.3 使用边界

- 若改动只发生在 normalization 算法内部，优先运行对应 micro gate
- 若改动触及 normalization 前后的 layer 写入、overwrite、history、provenance 或 densify 边界，必须再运行对应 full-chain gate
- micro gate 只回答“算法本体是否更快、峰值分配是否更低”，不替代链路级回归确认
- full-chain gate 只回答“整体主线是否回归”，不替代算法内部热点分析

## 7. 当前输出字段重点

`stage_runs.csv` 当前重点字段包括：

- `elapsed_s`
- `rss_before_mb`
- `rss_after_mb`
- `rss_peak_mb`
- `rss_peak_delta_mb`
- `returned_same_object`
- `source_assay_same_object`
- `source_layer_same_object`
- `source_x_same_object`
- `source_m_same_object`
- `source_x_unchanged`
- `source_m_unchanged`
- `input_x_storage`
- `output_x_storage`
- `input_x_bytes`
- `output_x_bytes`
- `output_shares_x_with_source`
- `output_shares_m_with_source`
- `densified_output`
- `obs_cols_added`
- `var_cols_added`

这些字段的目标不是把所有 copy 行为量化到字节级精确来源，而是把“语义级别的 copy / alias / densify 边界”稳定记录下来。

## 8. 运行方式

列出可用场景：

```bash
uv run python scripts/perf/run_runtime_baseline.py --list-scenarios
```

运行默认 baseline：

```bash
uv run python scripts/perf/run_runtime_baseline.py
```

运行快速 smoke baseline：

```bash
uv run python scripts/perf/run_runtime_baseline.py --profile quick
```

只运行指定场景：

```bash
uv run python scripts/perf/run_runtime_baseline.py \
  --profile quick \
  --scenario stable_chain_dense \
  --scenario stable_chain_trqn \
  --scenario normalize_trqn_only \
  --scenario sparse_transform_normalize
```

指定输出目录：

```bash
uv run python scripts/perf/run_runtime_baseline.py \
  --output-dir outputs/runtime_baseline_local
```

运行带门禁的 quick baseline：

```bash
uv run python scripts/perf/run_runtime_baseline.py \
  --profile quick \
  --gate-policy scripts/perf/runtime_gate_policy.json \
  --fail-on-gate
```

说明：

- `scripts/perf/runtime_gate_policy.json` 当前是仓库内置的 **CI-oriented quick gate**
- 它把 `quick` profile 下各 scenario 的 `max_elapsed_s / max_peak_delta_mb / allowed_densify_stages` 固化为 machine-readable budget
- `--fail-on-gate` 打开后，只要任一 scenario 超预算，脚本就返回非零退出码

### 8.1 当前内置 quick gate

| scenario | `max_elapsed_s` | `max_peak_delta_mb` | `allowed_densify_stages` |
| --- | ---: | ---: | --- |
| `import_diann_protein_long` | `1.50` | `128.0` | `[]` |
| `aggregate_peptide_to_protein` | `0.25` | `32.0` | `[]` |
| `stable_chain_dense` | `0.50` | `64.0` | `[]` |
| `stable_chain_quantile` | `0.50` | `64.0` | `[]` |
| `stable_chain_trqn` | `0.75` | `64.0` | `[]` |
| `normalize_quantile_only` | `0.25` | `32.0` | `[]` |
| `normalize_trqn_only` | `0.50` | `32.0` | `[]` |
| `sparse_log_only` | `0.10` | `16.0` | `[]` |
| `sparse_transform_normalize` | `0.25` | `32.0` | `["normalize_median_after_sparse_log"]` |
| `autoselect_integrate_only` | `1.00` | `128.0` | `[]` |
| `viz_qc_overview` | `2.50` | `256.0` | `[]` |

解释边界：

- 这些阈值是 **工程回归 gate**，不是性能宣传值
- 当前仓库只把 `quick` profile 固化为自动门禁；`default` profile 仍保留为本地较大规模对比基线
- 若后续更换 baseline 规模、绘图实现或 AutoSelect stage 组合，应先同步更新 gate policy，再调整测试/工作流

## 9. 使用规则

后续任何“优化 PR”若涉及：

- `core`
- `io`
- `aggregation`
- `transformation`
- `normalization`
- `imputation`
- `integration`
- `qc`

都应在修改前后至少运行与自己相关的 runtime baseline 场景。

建议最少要求：

1. 修改前记录一次基线结果。
2. 修改后复跑相同场景。
3. 若 PR 触及已建 gate 的 scenario，必须同时跑带 `--gate-policy ... --fail-on-gate` 的基线并保留 `gate_results.json`。
4. 比较 wall-time、peak RSS、densify 和 copy-path 漂移。
5. 若 stable 行为变化，先回到 contract 判断是否允许。

## 10. 当前限制

当前 baseline 仍有以下限制：

1. 主要基于 synthetic / generated inputs，不是公开数据 replay。
2. RSS 采用单进程采样，不是 OS 级最严格 profiler。
3. copy-path 目前是“对象共享/变更观察”，不是底层 allocator 级 tracing。
4. 当前 `autoselect` 与 `viz` 已有最小独立 runtime baseline，但覆盖面仍只到 stable `integrate` 选择层与 QC/workflow 读取层，不代表所有 stage 或所有 plotting recipe 都已单独建模。
5. sparse log JIT / NumPy 分支已拆出 `sparse_log_only` 独立 scenario；但具体进入哪条分支，仍取决于环境与阈值设置，因此结果解释必须结合 `environment.json` 中的 `SCPTENSOR_JIT_THRESHOLD`。
6. 当前内置 gate policy 只固化 `quick` profile；`default` profile 仍以本地人工对比为主，待积累更多机器与数据点后再决定是否升格为统一自动门禁。

这些限制在 `PR-0` 是可接受的，因为目标是先建立稳定、快速、可复用的工程基线。

## 11. 一句话结论

`PR-0` 的 runtime baseline 是后续优化的工程门禁，不是科学 benchmark；后续重构必须在不突破合同边界的前提下，用它验证“是否真的更快、是否引入了新的 densify/copy 漂移”。
