# AutoSelect Benchmark Assets

本目录保存 AutoSelect 相关评测脚本与生成资产，不属于用户主线文档。

## Current Entry

- `benchmark/autoselect/run_normalization_literature_score.py`
- `benchmark/autoselect/run_synthetic_normalization_stress.py`
- `benchmark/autoselect/run_strategy_comparison.py`

## Boundary

- 该目录用于验证选择合同与策略输出，不是社区总榜。
- 当前主资产聚焦 stable preprocessing 相关的 `normalize / impute / integrate` 选择合同。
- `reduce / cluster` 仍属于 experimental downstream helper，不应在这里被写成 stable 主榜能力。
- `overall_score` 表示方法质量本身，`selection_score` 表示叠加策略后的选择分；两者不能混成一个概念。
- AutoSelect 结果里出现的 `raw_norm_median` 等名字只是 comparison artifact naming，不等同于 canonical output layer。

## Review Links

- `docs/autoselect_contract.md`
- `docs/internal/review_autoselect_scoring_20260312.md`
- `docs/internal/review_batch_correction_20260305.md`
- `docs/internal/review_state_metrics_20260312.md`

## Resource Roles

- `论文证据`
  - scoring, reporting, and benchmark framework evidence
- `数据入口`
  - 当前目录不把任何单一公开数据页写成 AutoSelect 总榜稳定入口
- `模块规范 / 软件文档`
  - metric or benchmark APIs used to justify report-field semantics
- `资源包`
  - reusable assets only; not a replacement for benchmark ownership

## Run

Strategy comparison:

```bash
uv run python benchmark/autoselect/run_strategy_comparison.py
```

Synthetic normalization stress:

```bash
uv run python benchmark/autoselect/run_synthetic_normalization_stress.py \
  --scenario-mode literature_matched \
  --max-scenarios 48
```

## Main Outputs

- `benchmark/autoselect/strategy_compare/`
- `benchmark/autoselect/synthetic_normalization_stress/`
- `benchmark/autoselect/normalization_literature/`

## Notes

- Real public-data AutoSelect panels and unified state-aware scoring are still follow-up items.
- This README stays as an execution/index surface; longer rationale belongs in the linked review docs.
