# 蛋白层归一化 Benchmark

本目录用于评测 `scptensor.normalization` 的 protein-level normalization
方法。

## Current Entry

- `benchmark/normalization/run_benchmark.py`

## Boundary

- 当前脚本比较的是显式 logged layer 上的归一化，而不是对线性 vendor 输出直接比较。
- benchmark 本地可能出现 `log2`、`median_centered`、`quantile_norm`、`trqn_norm` 等 artifact naming；它们不重定义仓库 canonical `log / norm` 命名。
- 当前方法池：`none / mean / median / quantile / trqn`
- `sum` 仍只是候选，不应写成当前已实现事实。
- 这是 stage-specific normalization benchmark，不应扩展解释成完整 preprocessing 组合最优。

## Review Links

- `docs/normalization_contract.md`
- `docs/internal/review_normalization_20260307.md`
- `docs/internal/review_log_scale_20260312.md`
- `docs/internal/review_public_benchmark_data_20260312.md`

## Resource Roles

- `论文证据`
  - normalization benchmark and workflow evidence
- `数据入口`
  - replayable public DIA protein-level inputs
- `模块规范 / 软件文档`
  - workflow docs used to justify logged-layer comparison semantics
- `资源包`
  - reusable assets only; not a substitute for data entrypoints

## Run

```bash
UV_CACHE_DIR=.uv-cache uv run python benchmark/normalization/run_benchmark.py
```

Common explicit run:

```bash
UV_CACHE_DIR=.uv-cache uv run python benchmark/normalization/run_benchmark.py \
  --methods none mean median quantile trqn \
  --output-dir benchmark/normalization/outputs
```

## Main Outputs

- `benchmark/normalization/outputs/metrics_summary.csv`
- `benchmark/normalization/outputs/metrics_scores.csv`
- `benchmark/normalization/outputs/run_metadata.json`
- benchmark plots under `benchmark/normalization/outputs/`

## Notes

- Longer evidence discussion stays in the linked review docs.
- `benchmark/**/data/` and `benchmark/**/outputs/` are ignored by default.
