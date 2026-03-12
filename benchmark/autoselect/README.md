# AutoSelect Benchmark Assets

本目录保存 AutoSelect 相关的评测脚本和评测结果，不属于用户文档。

## 目录

- `run_normalization_literature_score.py`：基于文献指标的归一化综合评分多数据集评测脚本
- `normalization_literature/*.json|*.md`：文献指标评分评测输出（自动生成）
- `run_synthetic_normalization_stress.py`：大规模合成数据归一化压力评测脚本（50-500 行、500-10000 列）
- `synthetic_normalization_stress/*.json|*.csv|*.md`：大规模合成数据评测输出（自动生成）
- `run_strategy_comparison.py`：integration 阶段 `quality / balanced / speed` 策略并排对比脚本
- `strategy_compare/strategy_comparison.{json,csv,md}`：策略对比输出（自动生成）

## 说明

- 该目录用于集中存放 AutoSelect 评测资产，避免与用户教程文档混放。
- `docs/` 仅保留可阅读教程，评测脚本与产物统一归档在 `benchmark/`。
- 历史 `norm_test` 文件资产已清理；若目录残留为空壳，可忽略。当前以 `run_*` 脚本 + 自动生成结果目录为准。
- 当前目录保存的是“策略与合同验证资产”，不是单一社区 benchmark 总榜。

## Resource Roles

- `论文证据`
  - scIB / Liu 2025 / Weber 2019 / Wang 2025 等评分与报告框架来源
- `数据入口`
  - 当前 README 不把任何单一公开数据页写成 AutoSelect 总榜稳定入口
- `模块规范 / 软件文档`
  - `scib-metrics` bench API 文档可作为指标/报告接口参照
- `资源包`
  - 当前目录不依赖单独资源包页来定义评分合同

## 评分字段合同

- `overall_score`
  - 表示方法质量本身
- `selection_score`
  - 表示在 `quality / balanced / speed` 策略下，综合质量与运行代价后的选择分
- `quality / balanced / speed`
  - 是 selection policy 预设，不应被表述为“生物学优越性”

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
- 因此，`selection_score` 更适合作为策略层排序分，而不是替代 `overall_score` 成为“方法本体质量”的唯一展示值。

## 策略对比脚本

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

## 合成归一化评测脚本

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
