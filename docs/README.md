# Docs Index

`docs/` 目录只放“用户文档内容”，不放评测脚本和评测产物。

## 当前结构

- `tutorial.ipynb`：主教程
- `autoselect_tutorial.ipynb`：AutoSelect 教程
- `io_input_spec_diann_spectronaut.md`：DIA-NN / Spectronaut 输入规范
- `aggregation_methods_from_literature.md`：肽段->蛋白聚合方法文献整理
- `_tutorial_outputs/`：教程运行输出（临时产物）

## AutoSelect 位置说明

- **正确位置（文档）**：`docs/autoselect_tutorial.ipynb`
- **不应放在 docs 的内容（评测）**：`autoselect` 评测脚本与图表
  - 已迁移到 `benchmark/autoselect/norm_test/`

这保证了 `docs/` 聚焦“可阅读文档”，`benchmark/` 聚焦“可执行评测”。
