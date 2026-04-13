# Repository File Tiering

本文档只保留仓库清理时真正要看的四层分级。

## T0: 发布核心

禁止随意删除。包括：

- `AGENTS.md`
- `README.md`
- `pyproject.toml`
- `uv.lock`
- `.github/workflows/*.yml`
- stable package code:
  `scptensor/core io aggregation transformation normalization impute integration qc standardization utils viz`
- stable tests:
  `tests/core fileio aggregation transformation normalization impute integration qc docs`

规则：

- 除非有明确替代方案和回归测试，否则不删。
- 优先移动或收口实现，不直接消能力。

## T1: 重要支撑

默认保留。包括：

- `scptensor/experimental/`, `scptensor/dim_reduction/`, `scptensor/cluster/`
- `tests/dim_reduction/`, `tests/cluster/`, `tests/autoselect/`, `tests/benchmark/`
- tutorial notebooks 与 toy outputs
- `scripts/docs/validate_review_manifest.py`
- `scripts/perf/run_runtime_baseline.py`
- `benchmark/**/*.py`
- `benchmark/**/README.md`

规则：

- 若要删除，先确认没有 README、docs、CI、tests 依赖。

## T2: 文档与治理资产

允许收敛，但不能盲删。包括：

- `docs/*_contract.md`
- `docs/internal/runtime_baseline.md`
- `docs/internal/optimization_checklist.md`
- `docs/internal/review_*.md`
- `docs/internal/review_manifest_20260312.json`
- `docs/references/*.json`
- `docs/README.md`
- `benchmark/README.md`
- `CHANGELOG.md`
- `.github/workflows/README.md`
- tutorial notebooks

规则：

- 可以合并、归档、压短。
- 先处理引用关系与索引，再删文件。

## T3: 本地生成物

默认应清理，不应视为仓库长期资产：

- `__pycache__/`
- `.mypy_cache/`
- `.pytest_cache/`
- `.ruff_cache/`
- `.coverage`
- `htmlcov/`
- `dist/`
- `build/`
- `*.egg-info/`
- `tmp/`
- `outputs/`
- `benchmark/**/outputs/`
- `tutorial/_tutorial_outputs/autoselect/`

通常不自动删除：

- `.venv/`
- `data/`
- `.uv-cache/`
- `.uv-tmp/`

## Recommended Cleanup

```bash
rm -rf .mypy_cache .pytest_cache .ruff_cache htmlcov
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
rm -f .coverage
rm -rf dist build *.egg-info tmp outputs
rm -rf tutorial/_tutorial_outputs/autoselect
rm -rf benchmark/*/outputs benchmark/*/__pycache__
```

一句话：

- 先清 T3
- 再收敛 T2
- 不要误删 T0 / T1
