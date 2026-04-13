# 蛋白层 Integration Benchmark

本目录用于评测 `scptensor.integration` 在 DIA 单细胞蛋白组场景下的去批次
表现。

## Current Entry

- `benchmark/integration/run_benchmark.py`

## Boundary

- 当前目录按场景解释结果，不做跨场景单一总榜。
- 已实现三类场景：
  - `balanced_amount_by_sample`
  - `partially_confounded_bridge_sample`
  - `confounded_amount_as_batch`
- `fully confounded` 的角色是 guardrail / failure-style 场景，不是常规赢家榜。
- 当前主榜只覆盖 protein-level, matrix-level integration 方法；embedding-level downstream helper 只能作为外部评估终点。
- 输出里的 method-specific layer 名不重定义 canonical `raw / log / norm / imputed / zscore` 命名。

## Review Links

- `docs/integration_contract.md`
- `docs/internal/review_batch_correction_20260305.md`

## Resource Roles

- `论文证据`
  - batch correction, confounding, and integration scoring evidence
- `数据入口`
  - replayable DIA protein-level inputs and scenario definitions
- `模块规范 / 软件文档`
  - metric and method docs used only to justify interface semantics
- `资源包`
  - reusable assets only; not stable benchmark data entrypoints

## Run

```bash
UV_CACHE_DIR=/tmp/.uv-cache uv run python benchmark/integration/run_benchmark.py \
  --output-dir benchmark/integration/outputs
```

Explicit scenario run:

```bash
UV_CACHE_DIR=/tmp/.uv-cache uv run python benchmark/integration/run_benchmark.py \
  --scenarios balanced_amount_by_sample partially_confounded_bridge_sample confounded_amount_as_batch \
  --output-dir benchmark/integration/outputs
```

## Main Outputs

- `benchmark/integration/outputs/metrics_raw.csv`
- `benchmark/integration/outputs/metrics_summary.csv`
- `benchmark/integration/outputs/metrics_scores.csv`
- `benchmark/integration/outputs/guardrail_checks.csv`
- `benchmark/integration/outputs/run_metadata.json`

## Notes

- Interpret `metrics_scores.csv` by scenario first. Legacy aggregate plots are secondary.
- Scenario splitting and diagnostics panel semantics are now consolidated in
  `docs/internal/review_batch_correction_20260305.md`.
- State-aware uncertainty burden is still a follow-up item, not a fully unified current output.
