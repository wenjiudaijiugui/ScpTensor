# 蛋白层 Imputation Benchmark

本目录用于评测 `scptensor.impute` 在 DIA 单细胞蛋白组预处理场景下的
表现。

## Current Entry

- `benchmark/imputation/run_benchmark.py`

## Boundary

- 主协议是 `masked-value recovery + downstream stability`，不是只看 reconstruction error。
- `no-imputation` 必须保留为基线。
- 当前已实现 `main` 与 `auxiliary` 两块：
  - `main`: protein-direct masked recovery
  - `auxiliary`: precursor holdout -> protein endpoint
- holdout 当前按 source-layer 状态分层，属于 state-aware masking benchmark。
- benchmark 产物里出现的 `raw_norm_median_knn` 等名字只是 artifact naming；若进入 stable mainline，仍需显式 promote 到 canonical `imputed` 层。

## Review Links

- `docs/imputation_contract.md`
- `docs/internal/review_masked_imputation_20260312.md`
- `docs/internal/review_state_metrics_20260312.md`

## Resource Roles

- `论文证据`
  - masked-value recovery and downstream-stability evidence
- `数据入口`
  - replayable DIA protein or precursor inputs used by benchmark scripts
- `模块规范 / 软件文档`
  - benchmark semantics are governed primarily by review and contract docs
- `资源包`
  - reusable assets only; not benchmark ownership

## Run

Default run:

```bash
UV_CACHE_DIR=/tmp/.uv-cache uv run python benchmark/imputation/run_benchmark.py \
  --tier default \
  --board main \
  --output-dir benchmark/imputation/outputs
```

Literature-style dual-board run:

```bash
UV_CACHE_DIR=/tmp/.uv-cache uv run python benchmark/imputation/run_benchmark.py \
  --tier literature \
  --board both \
  --output-dir benchmark/imputation/outputs
```

## Main Outputs

- `benchmark/imputation/outputs/metrics_raw.csv`
- `benchmark/imputation/outputs/metrics_summary.csv`
- `benchmark/imputation/outputs/metrics_scores.csv`
- `benchmark/imputation/outputs/run_metadata.json`
- dual-board runs also create `benchmark/imputation/outputs/main/` and `.../auxiliary/`

## Notes

- Optional heavy methods may require extra dependencies or explicit method selection.
- Review docs own the longer benchmark rationale; this README stays minimal on purpose.
