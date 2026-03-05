# AutoSelect Benchmark Assets

本目录保存 AutoSelect 相关的评测脚本和评测结果，不属于用户文档。

## 目录

- `norm_test/test_normalization_autoselect.py`：归一化自动选择综合评测脚本
- `norm_test/*.png|*.html|*.json`：评测报告与图表输出
- `run_strategy_comparison.py`：integration 阶段 `quality / balanced / speed` 策略并排对比脚本
- `strategy_compare/strategy_comparison.{json,csv,md}`：策略对比输出（自动生成）

## 说明

- 该目录用于集中存放 AutoSelect 评测资产，避免与用户教程文档混放。
- `docs/` 仅保留可阅读教程，评测脚本与产物统一归档在 `benchmark/`。

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
