# ScpTensor 仓库文件分层与清理规则

本文档用于回答两个问题：

1. 仓库里的文件哪些是发布核心，哪些只是支撑材料；
2. 哪些内容应长期保留，哪些内容应视为本地生成物并定期清理。

该文档是仓库整理与瘦身的维护规则，不替代 `AGENTS.md` 中的项目边界合同。

## 分层定义

### T0: 发布核心，禁止随意删除

这些文件直接定义项目的发布边界、安装方式、公开 API、核心实现和主线验收。

- 根目录治理文件：
  - `AGENTS.md`
  - `README.md`
  - `pyproject.toml`
  - `uv.lock`
  - `.github/workflows/*.yml`
- 核心实现：
  - `scptensor/core/`
  - `scptensor/io/`
  - `scptensor/aggregation/`
  - `scptensor/transformation/`
  - `scptensor/normalization/`
  - `scptensor/impute/`
  - `scptensor/integration/`
  - `scptensor/qc/`
  - `scptensor/standardization/`
  - `scptensor/utils/`
  - `scptensor/viz/`
- 主线测试：
  - `tests/acceptance/`
  - `tests/core/`
  - `tests/fileio/`
  - `tests/aggregation/`
  - `tests/transformation/`
  - `tests/normalization/`
  - `tests/impute/`
  - `tests/integration/`
  - `tests/qc/`
  - `tests/docs/`

处理原则：

- 除非有明确替代方案和配套测试，否则不删除。
- 若需要重构，优先移动实现而不是消除能力。

### T1: 重要支撑，默认保留

这些文件不是发布主线本身，但为开发、验证、教程和 CI 提供关键支撑。

- 实验与边界命名空间：
  - `scptensor/experimental/`
  - `scptensor/dim_reduction/`
  - `scptensor/cluster/`
- 对应测试：
  - `tests/dim_reduction/`
  - `tests/cluster/`
  - `tests/autoselect/`
  - `tests/benchmark/`
- 教程与可复现实例：
  - `tutorial/tutorial.ipynb`
  - `tutorial/autoselect_tutorial.ipynb`
  - `tutorial/_tutorial_outputs/toy_spectronaut_peptide_matrix.tsv`
  - `tutorial/_tutorial_outputs/toy_spectronaut_protein_long.tsv`
- 工具脚本：
  - `scripts/docs/validate_review_manifest.py`
  - `scripts/perf/run_runtime_baseline.py`
- benchmark 代码：
  - `benchmark/**/*.py`
  - `benchmark/**/README.md`

处理原则：

- 默认保留，因为它们仍服务于开发闭环、性能核查或教程运行。
- 若要删除，必须先确认没有 README、docs、CI 或测试引用。

### T2: 参考资料与治理文档，可收敛但不应盲删

这些文件主要承担证据、规范、背景或维护说明角色，不属于运行时核心，但仍是仓库知识资产。

- 合同与规范文档：
  - `docs/*_contract.md`
  - `docs/runtime_baseline.md`
  - `docs/optimization_checklist.md`
  - `docs/experimental_downstream_alignment_plan.md`
- 评审与综述：
  - `docs/review_*.md`
  - `docs/review_manifest_20260312.json`
  - `docs/references/*.json`
- 项目说明：
  - `CHANGELOG.md`
  - `.github/workflows/README.md`
  - `docs/README.md`
  - `tutorial/README.md`
  - `benchmark/README.md`

处理原则：

- 允许后续合并、归档或精简，但应先处理引用关系与文档索引。
- 若只是不常读，不构成删除理由。

### T3: 本地生成物，默认应清理且不应跟踪

这些内容不应作为仓库长期资产保留。它们只对本地运行瞬时有价值。

- Python / test 缓存：
  - `__pycache__/`
  - `.mypy_cache/`
  - `.pytest_cache/`
  - `.ruff_cache/`
  - `.coverage`
  - `htmlcov/`
- 打包与安装残留：
  - `dist/`
  - `build/`
  - `*.egg-info/`
- 运行期临时输出：
  - `tmp/`
  - `outputs/`
  - `benchmark/**/outputs/`
  - `tutorial/_tutorial_outputs/autoselect/`
- 本地下载数据与用户环境：
  - `data/`
  - `.venv/`
  - `.uv-cache/`
  - `.uv-tmp/`

处理原则：

- 默认定期清理。
- 不把它们作为“仓库内容的一部分”参与发布判断。
- 不要把用户本地环境或下载数据误当成项目资产。

## 当前仓库结论

在本轮历史瘦身之后，当前 Git 跟踪内容已经相对收敛：

- 当前跟踪文件主要由 `scptensor/`、`tests/`、`docs/`、`benchmark/`、`tutorial/` 构成。
- 当前跟踪的非代码大文件很少，主要是两个 tutorial notebook 和锁文件。
- 当前没有明显应直接从 Git 中删除的“低价值二进制产物”或大体积测试输出。

因此本轮“清理非重要内容”的重点应放在：

- 清掉本地生成物；
- 维持 `.gitignore`；
- 保持 T0/T1/T2 文件不被误删。

## 推荐清理动作

适合定期执行的保守清理：

```bash
rm -rf .mypy_cache .pytest_cache .ruff_cache htmlcov
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
rm -f .coverage
rm -rf dist scptensor.egg-info tmp outputs
rm -rf tutorial/_tutorial_outputs/autoselect
rm -rf benchmark/*/outputs benchmark/*/__pycache__
```

不建议自动删除的本地内容：

- `.venv/`
- `data/`
- `.uv-cache/`
- `.claude/`
- `.serena/`

这些目录可能包含用户环境、下载数据或个人工具状态，应由使用者按需处理。
