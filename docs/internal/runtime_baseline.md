# Runtime Baseline

本文档定义 `scripts/perf/run_runtime_baseline.py` 的最小使用边界。

## Role

运行时基线用于：

- 比较优化前后的 wall-time / memory / densify / copy-path
- 补 benchmark 不覆盖的工程回归面
- 服务 `docs/internal/optimization_checklist.md` 中的 runtime gate

它不是：

- 方法学 benchmark 排名
- 稳定实现合同
- 文献综述

若与 `AGENTS.md` 或 `docs/*_contract.md` 冲突，以合同为准。

## Authority Docs

优先级：

1. `AGENTS.md`
2. `docs/internal/optimization_checklist.md`
3. 对应模块合同

## Boundary vs Benchmark

`benchmark/` 负责方法质量、公开数据 replay、literature-aligned scoring。

`scripts/perf/run_runtime_baseline.py` 负责：

- wall-time
- RSS / peak memory
- sparse -> dense 边界
- copy / shared-reference 观察

不要把 runtime baseline 写成“方法更优”结论。

## Current Script

- entry: `scripts/perf/run_runtime_baseline.py`
- default outputs: `outputs/runtime_baseline/`
- current output assets:
  - `stage_runs.csv`
  - `scenario_summary.json`
  - `environment.json`
  - `errors.json`
  - `gate_results.json` when `--gate-policy` is provided

## Scenario Set

当前保留的场景名：

- `import_diann_protein_long`
- `aggregate_peptide_to_protein`
- `stable_chain_dense`
- `stable_chain_quantile`
- `stable_chain_trqn`
- `normalize_quantile_only`
- `normalize_trqn_only`
- `sparse_transform_normalize`
- `sparse_log_only`
- `autoselect_integrate_only`
- `viz_qc_overview`

理解方式：

- `stable_chain_*`: full-chain gate
- `normalize_*_only`, `sparse_log_only`, `autoselect_integrate_only`, `viz_qc_overview`: micro gate
- `sparse_*`: sparse / JIT / densify 边界观察

## Gate Selection

通用规则：

- dense 主链改动：至少跑 `stable_chain_dense`
- `quantile` 全链路改动：跑 `stable_chain_quantile`
- `TRQN` 全链路改动：跑 `stable_chain_trqn`
- sparse / JIT / densify 改动：跑 `sparse_transform_normalize`
- 只改 `quantile` 内部：先跑 `normalize_quantile_only`，再用 full-chain 复核
- 只改 `TRQN` 内部：先跑 `normalize_trqn_only`，再用 full-chain 复核
- 只改 sparse log：跑 `sparse_log_only`
- 只改 integration AutoSelect 调度：跑 `autoselect_integrate_only`
- 只改可视化读取层：跑 `viz_qc_overview`

如果改动触及 layer 写入、history、provenance、overwrite 或 densify 边界，micro gate 不能替代 full-chain gate。

## Run

基础运行：

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/perf/run_runtime_baseline.py
```

带 gate policy：

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/perf/run_runtime_baseline.py \
  --gate-policy scripts/perf/runtime_gate_policy.json \
  --fail-on-gate
```

## Notes

- `benchmark/**` 的结果不要拿来替代 runtime baseline。
- `outputs/runtime_baseline/` 默认是本地产物，不是长期文档资产。
- 如果合理改动了 runtime budget，要同步解释原因，再更新 gate policy、文档和测试。
