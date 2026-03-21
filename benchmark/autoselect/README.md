# AutoSelect Benchmark Assets

本目录保存 AutoSelect 相关的评测脚本和评测结果，不属于用户文档。

## Current Entry

- `benchmark/autoselect/run_normalization_literature_score.py`
- `benchmark/autoselect/run_synthetic_normalization_stress.py`
- `benchmark/autoselect/run_strategy_comparison.py`

## Boundary

- 该目录用于集中存放 AutoSelect 评测资产，避免与用户教程文档混放。
- `docs/` 仅保留可阅读教程，评测脚本与产物统一归档在 `benchmark/`。
- 历史 `norm_test` 文件资产已清理；若目录残留为空壳，可忽略。当前以 `run_*` 脚本 + 自动生成结果目录为准。
- 当前目录保存的是“策略与合同验证资产”，不是单一社区 benchmark 总榜。
- 当前主资产聚焦 stable preprocessing 相关的 `normalize / impute / integrate` 选择合同；`reduce / cluster` 仍属于 experimental downstream helper，不是本目录的稳定主榜边界。
- 当前默认脚本资产只说明“仓库今天默认提供哪些 replay / stress / strategy 工具”；长期合同仍是评分语义、stable/experimental stage 边界，以及 comparison artifact naming 与 canonical layer naming 的区分。

## Review Links

- `docs/review_autoselect_scoring_20260312.md`
- `docs/review_batch_confounding_20260312.md`
- `docs/review_state_metrics_20260312.md`
- `docs/autoselect_contract.md`

## Resource Roles

- `论文证据`
  - scIB / Liu 2025 / Weber 2019 / Wang 2025 等评分与报告框架来源
- `数据入口`
  - 当前 README 不把任何单一公开数据页写成 AutoSelect 总榜稳定入口
- `模块规范 / 软件文档`
  - `scib-metrics` bench API 文档可作为指标/报告接口参照
- `资源包`
  - 当前目录不依赖单独资源包页来定义评分合同

## 1. 评分字段合同（长期合同）

- `overall_score`
  - 表示方法质量本身
- `selection_score`
  - 表示在 `quality / balanced / speed` 策略下，综合质量与运行代价后的选择分
- `quality / balanced / speed`
  - 是 selection policy 预设，不应被表述为“生物学优越性”
- `layer_or_key` / `output_layer`
  - 若出现 `raw_norm_median`、`raw_norm_median_iterative_svd` 之类名字，它们是 AutoSelect 比较产物的 artifact naming，不等同于仓库 canonical `norm / imputed`

当前脚本输出情况：

- `run_strategy_comparison.py`
  - 明确同时输出 `selection_score` 与 `overall_score`
- `run_synthetic_normalization_stress.py`
  - 场景行与汇总中同时输出 `overall_score` / `selection_score`
- `run_normalization_literature_score.py`
  - 当前主要保留 `overall_score` 驱动的 literature-style 汇总

经二次核查的来源性细节：

- `scIB` 原始 benchmark 明确把 integration method 的评估拆成 `accuracy / usability / scalability` 三层，并给出 `S_overall = 0.6 * S_bio + 0.4 * S_batch` 的组合分示例；这说明“综合分”依赖权重设定，权重本质上是场景化策略而非方法学真理：<https://www.nature.com/articles/s41592-021-01336-8>
- `scib-metrics` 的 `Benchmarker` API 也是分开接收 `bio_conservation_metrics` 与 `batch_correction_metrics`，随后再统一制表；这支持在 AutoSelect 报告里长期保留 `overall_score` 与 `selection_score` 两层：<https://scib-metrics.readthedocs.io/en/stable/notebooks/large_scale.html>
- `Liu et al.`（Nature Methods 2025）在多任务多模态 integration benchmark 中明确指出不存在“统一最优方法”，且不同输出类型适配的指标与方法不同；这支持把 `selection_score` 约束在具体策略/任务内解释，而不是把它当作跨任务绝对排名：<https://www.nature.com/articles/s41592-025-02856-3>
- `Karuppanan et al.`（JPR 2025）进一步说明 normalization 与 imputation 的最优组合具有明显数据依赖性；这支持把 AutoSelect 报告继续解释为“当前数据与策略下的选择结果”，而不是脱离场景的永久最佳方法声明：<https://doi.org/10.1021/acs.jproteome.4c00552>
- 因此，`selection_score` 更适合作为策略层排序分，而不是替代 `overall_score` 成为“方法本体质量”的唯一展示值。

## 2. 当前默认脚本资产（实现事实，不构成合同）

- `run_normalization_literature_score.py`：基于文献指标的归一化综合评分多数据集评测脚本
- `normalization_literature/*.json|*.md`：文献指标评分评测输出（自动生成）
- `run_synthetic_normalization_stress.py`：大规模合成数据归一化压力评测脚本（50-500 行、500-10000 列）
- `synthetic_normalization_stress/*.json|*.csv|*.md`：大规模合成数据评测输出（自动生成）
- `run_strategy_comparison.py`：integration 阶段 `quality / balanced / speed` 策略并排对比脚本
- `strategy_compare/strategy_comparison.{json,csv,md}`：策略对比输出（自动生成）

解释：

- 这里列的是当前仓库默认保留和维护的 AutoSelect benchmark 资产。
- 后续即使增删 `run_*` 脚本、调整输出目录或替换默认 stress 场景，也不应反向改写 AutoSelect 的长期评分合同、stable/experimental 边界和 artifact naming 解释。

## 3. 当前默认策略脚本（实现事实）

运行示例：

```bash
uv run python benchmark/autoselect/run_strategy_comparison.py
```

该脚本会：

1. 构造带有 batch confounding 的合成 DIA 蛋白矩阵；
2. 固定前处理基线为 `log_transform -> median_norm -> row_median impute`；
3. 分别在 `quality`、`balanced`、`speed` 三种策略下运行 integration 自动选择；
4. 输出并排对比结果到 `benchmark/autoselect/strategy_compare/`。

说明：

- 这是策略层对比，不等同于真实公共数据 panel 上的 integration benchmark。
- 当前数据是单个合成 confounded 容器；`partially confounded` 与 state-aware penalty 尚未在该脚本中落地。
- 即使后续为 `reduce / cluster` 增加策略对比，也只能作为 exploratory downstream 资产解释，不能反向定义 stable preprocessing release。

## 4. 当前默认合成归一化评测脚本（实现事实）

运行示例（文献条件匹配模式）：

```bash
uv run python benchmark/autoselect/run_synthetic_normalization_stress.py \
  --scenario-mode literature_matched \
  --max-scenarios 48
```

该脚本支持两种模式：

1. `literature_matched`：按条件分层（同批次低差异 / 同批次高差异 / 多批次混杂）评测；
2. `stress`：广覆盖压力测试模式。

当前未统一落地的 review 合同：

- state-aware completeness / uncertainty penalty
- 基于真实公共数据 panel 的统一 AutoSelect 报告
- integration 三场景（`balanced / partially confounded / fully confounded`）的单目录汇总

解释：

- 上面两类脚本只是当前默认实现资产：一个强调 strategy replay，一个强调 synthetic normalization stress。
- 未来即使补入真实公共数据 panel、替换默认 stress 目录或新增 stage-specific script，也应把它们视为资产层扩展，而不是对评分合同本体的自动改写。
