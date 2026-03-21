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

这些目录默认被 `.gitignore` 忽略，不作为仓库提交内容。

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

### 5.5 `sparse_transform_normalize`

目的：

- 观测当前 sparse 输入在 `log_transform -> normalize(median)` 下的存储边界

说明：

- 当前场景故意停在 normalization
- 不继续接 imputation，因为当前 stable imputation 语义仍以 `np.isnan(X)` 为缺失判定主入口
- 该场景主要用于捕获 sparse->dense 漂移

## 6. 当前输出字段重点

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

## 7. 运行方式

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
  --scenario sparse_transform_normalize
```

指定输出目录：

```bash
uv run python scripts/perf/run_runtime_baseline.py \
  --output-dir outputs/runtime_baseline_local
```

## 8. 使用规则

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
3. 比较 wall-time、peak RSS、densify 和 copy-path 漂移。
4. 若 stable 行为变化，先回到 contract 判断是否允许。

## 9. 当前限制

当前 baseline 仍有以下限制：

1. 主要基于 synthetic / generated inputs，不是公开数据 replay。
2. RSS 采用单进程采样，不是 OS 级最严格 profiler。
3. copy-path 目前是“对象共享/变更观察”，不是底层 allocator 级 tracing。
4. 尚未覆盖 `autoselect` 与 `viz` 的单独 runtime baseline。

这些限制在 `PR-0` 是可接受的，因为目标是先建立稳定、快速、可复用的工程基线。

## 10. 一句话结论

`PR-0` 的 runtime baseline 是后续优化的工程门禁，不是科学 benchmark；后续重构必须在不突破合同边界的前提下，用它验证“是否真的更快、是否引入了新的 densify/copy 漂移”。
