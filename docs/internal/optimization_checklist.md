# Optimization Checklist

本文档是维护者执行优化 / 重构 / 性能改进时的短版规则。

## Source Of Truth Order

后续实现按这个顺序读：

1. `AGENTS.md`
2. 对应 `docs/*_contract.md`
3. 对应 `docs/internal/review_*.md`
4. `benchmark/README.md` 与子 README
5. tutorial notebooks
6. 当前源码实现

若 contract 与 review 冲突，contract 优先。
若 benchmark / tutorial 与 contract 冲突，修 benchmark / tutorial，不反向定义稳定实现。

## Stop Conditions

发现以下情况，先停优化，先做对齐判断：

1. 当前代码行为与 contract 不一致
2. history action / params 与 contract 不一致
3. benchmark README 漂出 contract 边界
4. tutorial 把 experimental helper 说成 stable mainline
5. 改动会碰 stable layer naming、missingness semantics、copy/mutation、overwrite、provenance

## PR Start Checklist

开工前至少明确：

- 触及模块
- authority docs
- 本 PR 不得破坏的 frozen invariants
- stable 还是 experimental
- 是否需要同步文档
- 是否需要 targeted regression tests
- 是否需要 runtime baseline

## PR Template

每个优化 PR 最少应写：

1. `Authority Docs`
2. `Frozen Invariants`
3. `Planned Scope`
4. `Verification Gates`

`Verification Gates` 至少包含：

- targeted regression tests
- docs sync check
- changed-file link / navigation check
- 必要时 runtime baseline
- 必要时 densify / memory 风险检查

## Runtime Gate Rules

runtime baseline 入口见 `docs/internal/runtime_baseline.md`。

快速映射：

- dense 主链改动：`stable_chain_dense`
- `quantile` 全链路：`stable_chain_quantile`
- `TRQN` 全链路：`stable_chain_trqn`
- sparse / JIT / densify：`sparse_transform_normalize`
- 只改 `quantile` 内部：`normalize_quantile_only`
- 只改 `TRQN` 内部：`normalize_trqn_only`
- 只改 sparse log：`sparse_log_only`
- 只改 AutoSelect integration 选择层：`autoselect_integrate_only`
- 只改 `viz` 读取层：`viz_qc_overview`

若已进入 `scripts/perf/runtime_gate_policy.json` 的场景，PR 完成前至少跑一次带 `--gate-policy ... --fail-on-gate` 的 quick baseline。

## Module -> Authority Docs

快速映射：

| module | contract | review / support |
| --- | --- | --- |
| `core-data` | `docs/core_data_contract.md` | `docs/README.md` |
| `core-compute` | `docs/core_compute_contract.md` | `docs/internal/runtime_baseline.md` |
| `io` | `docs/io_diann_spectronaut.md` | `docs/internal/review_io_state_mapping_20260312.md` |
| `aggregation` | `docs/aggregation_contract.md` | `docs/aggregation_literature.md`, `docs/internal/review_aggregation_benchmark_20260312.md` |
| `transformation` | `docs/transformation_contract.md` | `docs/internal/review_log_scale_20260312.md` |
| `normalization` | `docs/normalization_contract.md` | `docs/internal/review_normalization_20260307.md`, `docs/internal/review_log_scale_20260312.md` |
| `standardization` | `docs/standardization_contract.md` | `docs/README.md` |
| `impute` | `docs/imputation_contract.md` | `docs/internal/review_masked_imputation_20260312.md`, `docs/internal/review_state_metrics_20260312.md` |
| `integration` | `docs/integration_contract.md` | `docs/internal/review_batch_correction_20260305.md` |
| `qc` | `docs/qc_contract.md` | `docs/internal/review_qc_filtering_20260312.md`, `docs/internal/review_batch_correction_20260305.md` |
| `qc_psm` | `docs/qc_psm_contract.md` | `docs/experimental_downstream_contract.md` |
| `autoselect` | `docs/autoselect_contract.md` | `docs/internal/review_autoselect_scoring_20260312.md` |
| `utils` | `docs/utils_contract.md` | `tests/utils/*` |
| `viz` | `docs/viz_contract.md` | `docs/README.md` |
| `experimental` | `docs/experimental_downstream_contract.md` | `tests/core/test_experimental_api.py` |

## Do Not Bypass

不要直接越过这些边界：

- 不把非 DIA-NN / Spectronaut 输入写进 stable I/O
- 不把 peptide/precursor 处理说成默认 protein normalization
- 不把 experimental downstream helper 升格成 stable release contract
- 不把 benchmark artifact naming 写成 canonical layer naming
- 不把代码现状反向包装成 contract

## One-Line Rule

先看合同，再动代码；先保 frozen invariants，再谈优化收益。
