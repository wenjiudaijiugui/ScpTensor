# 肽段/前体 -> 蛋白聚合 Benchmark

本目录用于评测 `scptensor.aggregation.aggregate_to_protein` 的 protein-level
endpoint 表现。

## Current Entry

- `benchmark/aggregation/run_benchmark.py`

## Boundary

- 当前已实现的是 `precursor-to-protein auxiliary board`：
  `Spectronaut peptide-long -> aggregate_to_protein -> protein-level scoring`
- `protein-direct main board` 仍未在本目录落地，不应写成当前事实。
- 主解释必须回到 protein-level endpoint，不能把 precursor / peptide 层指标当作最终主榜。
- 当前方法池对齐实现：`sum / mean / median / max / weighted_mean / top_n / maxlfq / tmp / ibaq`
- 输出里的 mapping burden、coverage 或 `.n` 类指标只用于解释方法行为，不单独决定胜负。

## Review Links

- `docs/aggregation_contract.md`
- `docs/internal/review_aggregation_benchmark_20260312.md`
- `docs/internal/review_public_benchmark_data_20260312.md`

## Resource Roles

- `论文证据`
  - aggregation / LFQ-style method and scoring evidence
- `数据入口`
  - replayable public mixed-sample peptide/precursor inputs
- `模块规范 / 软件文档`
  - aggregation tool or workflow docs used to justify metric semantics
- `资源包`
  - reusable assets only; not a replacement for stable public data entrypoints

## Run

```bash
UV_CACHE_DIR=.uv-cache uv run python benchmark/aggregation/run_benchmark.py
```

Common explicit run:

```bash
UV_CACHE_DIR=.uv-cache uv run python benchmark/aggregation/run_benchmark.py \
  --methods sum mean median max weighted_mean top_n maxlfq tmp ibaq \
  --output-dir benchmark/aggregation/outputs
```

## Main Outputs

- `benchmark/aggregation/outputs/metrics_summary.csv`
- `benchmark/aggregation/outputs/protein_level_results.csv`
- `benchmark/aggregation/outputs/run_metadata.json`
- benchmark plots under `benchmark/aggregation/outputs/`

## Notes

- `benchmark/**/data/` and `benchmark/**/outputs/` are ignored by default.
- Review docs own the longer evidence trail; this README stays as the execution entry and boundary note.
