# Imputation 补齐清单（临时开发指导）

> 用途：仅用于编码阶段指导，不属于正式文档。

## P0（优先）

- [x] 新增 `impute_none`（显式不填充策略，可参与 AutoSelect/benchmark）
- [x] 新增 `impute_zero`
- [x] 新增 `impute_row_mean`
- [x] 新增 `impute_row_median`
- [x] 新增 `impute_half_row_min`（或 `impute_min_det_fraction`, 默认 0.5）

## P0（统一接口）

- [x] 统一参数：`random_state`, `assay_name`, `source_layer`, `new_layer_name`
- [x] 在 `scptensor.impute.base` 完成统一注册
- [x] 在 `scptensor.impute.__init__` 完成统一导出
- [x] 明确“观测值不改动”硬约束
- [x] 明确全缺失行/列 fallback 行为

## P1（方法扩展）

- [x] 新增 `iterative_svd`
- [x] 新增 `softimpute`（可选依赖，缺依赖时给清晰报错）

## P1（机制感知）

- [x] `impute(...)` 增加 `missing_mechanism={"auto","mcar","mar","mnar"}`
- [x] `auto` 输出判定依据（透明可解释）

## P1（教程/示例）

- [ ] 教程加入 `none` 与基线法对比
- [ ] 增加方法选择表（MCAR/MAR/MNAR）

## P2（Benchmark）

- [ ] 新建 `benchmark/imputation/`
- [ ] 固定 DIA 数据 + 合成缺失机制
- [ ] 输出排名图与汇总指标
- [ ] 增加 smoke benchmark 入口
