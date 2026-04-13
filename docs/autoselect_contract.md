# ScpTensor AutoSelect 冻结实现合同（冻结到 2026-03-16）

## 1. 文档目标

本文档冻结 `scptensor.autoselect` 的当前实现合同，供后续核心代码优化、API 收敛、测试补强与 benchmark 对齐使用。

它回答的不是“未来理想上该怎么做”，而是“当前实现已经对外暴露了哪些语义，重构时不能被悄悄改坏”。

本文档覆盖：

- `scptensor.autoselect.core`
- `scptensor.autoselect.strategy`
- `scptensor.autoselect.evaluators.*`
- `scptensor.autoselect.report`
- `scptensor.autoselect.__init__` 中的快捷入口

本文档不覆盖：

- 具体归一化 / 填补 / 去批次算法内部公式
- 文献综述本身
- README、教程和 benchmark 叙事层组织
- `dim_reduction / cluster` 的实验性下游算法细节

项目总边界仍以仓库 `AGENTS.md` 为准：ScpTensor 的稳定主线是 DIA 单细胞蛋白组预处理，目标是得到可审计的 protein-level quantitative matrix；`reduce / cluster` 仅是 experimental downstream helpers。

## 2. 模块边界与公共入口

### 2.1 当前公开对象

当前 `autoselect` 对外公开的核心对象是：

- `EvaluationResult`
- `StageReport`
- `AutoSelectReport`
- `AutoSelector`
- `StrategyPreset`
- `get_strategy_preset`
- `list_strategy_presets`

快捷入口包括：

- `auto_normalize`
- `auto_impute`
- `auto_integrate`
- `auto_reduce`
- `auto_cluster`

其中：

- `normalize / impute / integrate` 属于稳定预处理阶段。
- `reduce / cluster` 在代码中可用，但在仓库边界上属于实验性下游辅助阶段，不应被表述为稳定 preprocessing release contract 的一部分。

### 2.2 支持的 stage 键

当前 `AutoSelector.SUPPORTED_STAGES` 固定为：

- `normalize`
- `impute`
- `integrate`
- `reduce`
- `cluster`

实现约束：

- `AutoSelector(stages=None)` 当前会默认展开为这五个 stage 的完整列表。
- 这只是当前代码行为，不等价于“五个 stage 都同属稳定发布边界”。
- 面向稳定预处理流水线的调用方，若不希望实验性 stage 混入，应显式传入 `["normalize", "impute", "integrate"]` 之类的 stage 列表。

### 2.3 当前未实现但已暴露的控制项

`AutoSelector` 构造函数包含：

- `parallel`
- `n_jobs`

但当前执行路径仍是串行 `for stage in self.stages`。因此：

- 这两个参数目前是预留位，不构成并行执行保证。
- 后续优化不能仅因为参数存在，就把行为文档写成“当前已并行”。

## 3. 数据结构合同

## 3.1 `EvaluationResult`

`EvaluationResult` 表示“单个方法在单个 stage 上的一条评估结果”。其字段语义应固定如下：

| 字段 | 类型 | 当前语义 |
|---|---|---|
| `method_name` | `str` | 被评估的方法名，也是 report 中的主键之一 |
| `scores` | `dict[str, float]` | 分项指标分数。语义由各 evaluator 决定，通常期望在 `0..1` |
| `overall_score` | `float` | 纯质量聚合分，不直接混入 runtime |
| `execution_time` | `float` | 该方法评估耗时，单位秒 |
| `layer_name` | `str` | 该方法产物在容器中的标识名；对 layer 型 stage，通常是 `{source_layer}_{method_name}` |
| `error` | `str \| None` | 方法失败时的异常摘要；成功时为 `None` |
| `method_contract` | `dict[str, Any] \| None` | 每方法的合同元数据；当前不是所有 stage 都会填充 |
| `selection_score` | `float \| None` | 面向最终排序的策略化分数，质量分 + runtime 偏好后的结果 |
| `n_repeats` | `int` | 请求的重复次数，不是“成功重复次数” |
| `overall_score_std` | `float \| None` | 成功重复上的总体分标准差，`ddof=0` |
| `overall_score_ci_lower` | `float \| None` | 成功重复总体分的经验分位区间下界 |
| `overall_score_ci_upper` | `float \| None` | 成功重复总体分的经验分位区间上界 |
| `repeat_overall_scores` | `list[float]` | 每次成功重复得到的 `overall_score` 列表 |

实现性约束：

- `to_dict()` 必须保留上述全部字段。
- 当方法失败时，`overall_score` 当前固定为 `0.0`。
- `selection_score` 可能为 `None`，直到 stage 级排序逻辑运行后才被写入。
- `method_contract` 允许为空，消费方不能假定每个 stage 都有。

## 3.2 `StageReport`

`StageReport` 表示“一个 stage 的完整比较报告”。其字段语义应固定如下：

| 字段 | 类型 | 当前语义 |
|---|---|---|
| `stage_name` | `str` | stage 名；经 `AutoSelector` 统一后为 canonical stage 键 |
| `stage_key` | `str \| None` | 机器稳定键；当前应与 `stage_name` 同步为 canonical stage 键 |
| `results` | `list[EvaluationResult]` | 所有被比较方法的结果，成功与失败都保留 |
| `best_method` | `str` | 当前被选中的方法名；若全失败则为空字符串 |
| `best_result` | `EvaluationResult \| None` | 当前被选中的结果对象；若全失败则为 `None` |
| `recommendation_reason` | `str` | 面向人类的推荐原因字符串，不应作为机器稳定协议解析 |
| `method_contracts` | `dict[str, dict[str, Any]]` | 以方法名为键的合同元数据表；当前是可选增强信息 |
| `metric_weights` | `dict[str, float]` | 本次实际用于质量聚合的权重快照 |
| `input_assay` | `str \| None` | 本 stage 输入 assay |
| `input_layer` | `str \| None` | 本 stage 输入 layer |
| `output_assay` | `str \| None` | 选中结果对应的输出 assay |
| `output_layer` | `str \| None` | 选中结果对应的输出 layer；对 cluster stage 可为空 |
| `output_obs_key` | `str \| None` | 仅 cluster stage 使用；表示写入 `obs[...]` 的键 |
| `selection_strategy` | `str` | 最终排序使用的策略名 |
| `n_repeats` | `int` | 本 stage 请求的每方法重复次数 |
| `confidence_level` | `float` | 重复区间使用的经验分位置信水平 |

派生字段：

- `success_rate` = `error is None` 的结果数 / `results` 总数

实现性约束：

- `success_rate` 是 report 级 property，不是显式存储字段。
- `to_dict()` 必须展开为带 `results` 的完整嵌套结构。
- `recommendation_reason` 当前会被 stage-specific gate 文案拼接，因此只保证“可读解释”，不保证稳定模板。

## 3.3 `AutoSelectReport`

`AutoSelectReport` 表示多 stage 运行的总报告：

| 字段 | 类型 | 当前语义 |
|---|---|---|
| `stages` | `dict[str, StageReport]` | 以 canonical stage key 为键的 stage 报告表 |
| `total_time` | `float` | 多 stage 总耗时，单位秒 |
| `warnings` | `list[str]` | stage 级异常警告集合 |

实现性约束：

- `run()` 会按执行顺序将 stage 报告写入 `report.stages[stage]`。
- `summary()` 是人类可读摘要，不是稳定机器协议。
- `save(format=...)` 当前仅支持 `markdown / json / csv`。

## 4. 评分语义合同

### 4.1 `overall_score` 的定义

`overall_score` 是质量分，不直接包含 runtime。

当前实现：

- 每个 evaluator 提供 `metric_weights`。
- `BaseEvaluator.compute_overall_score()` 按权重做加权平均。
- 公式等价于：

```text
overall_score = sum(scores[k] * weight[k]) / sum(weight.values())
```

补充约束：

- 若总权重为 `0`，则返回 `0.0`。
- 若某个 `scores` 键缺失，则按 `0.0` 参与加权。
- override 权重时，未知键、负数权重、非有限数会触发 `ValueError`。

语义边界：

- `overall_score` 只在同一 stage 内有可比性。
- 不能把 `normalize` 的 `overall_score` 与 `integrate` 的 `overall_score` 直接横向比较。

### 4.2 `selection_score` 的定义

`selection_score` 是“最终选择时实际使用的排序分”。它与 `overall_score` 明确不同：

- `overall_score`：质量
- `selection_score`：质量 + 速度偏好

当前实现：

```text
runtime_score = 1 - (time - min_time) / (max_time - min_time)
selection_score = quality_weight * overall_score + runtime_weight * runtime_score
```

其中：

- runtime 只在当前 stage 的“成功方法集合”内部做 min-max 归一化。
- 若所有成功方法耗时相同，则所有成功方法的 `runtime_score = 1.0`。
- 失败方法的 `selection_score` 被强制写成 `0.0`。
- 成功方法的 `selection_score` 最终被裁剪到 `0..1`。

因此必须明确：

- `selection_score` 不是纯科学质量分。
- `selection_score` 依赖候选集合、策略 preset 和相对 runtime，不能跨 stage、跨候选集、跨运行配置直接比较。

### 4.3 选优 tie-breaker

当前 `BaseEvaluator._select_best_result()` 的 tie-breaker 顺序固定为：

1. 更高的 `selection_score`
2. 更高的 `overall_score`
3. 更短的 `execution_time`
4. Python 字符串顺序下更“靠后”的 `method_name`

这最后一条只是为了确定性，不代表生物学或统计学优先级。

后续优化可以改善 tie-breaker 设计，但不能在未同步更新合同与测试的情况下悄悄改变现有排序语义。

## 5. Strategy Preset 合同

当前策略 preset 是稳定 API，名称和值如下：

| 名称 | `quality_weight` | `runtime_weight` | 当前语义 |
|---|---|---|---|
| `quality` | `1.00` | `0.00` | 只按质量分排序 |
| `balanced` | `0.85` | `0.15` | 默认工程策略，轻度考虑 runtime |
| `speed` | `0.65` | `0.35` | 更偏向快速迭代 |

实现性约束：

- `get_strategy_preset(name)` 会做 `strip().lower()` 规范化。
- 非法策略名会触发 `ValueError`。
- `list_strategy_presets()` 的当前 canonical display order 是 `["speed", "balanced", "quality"]`。
- `AutoSelector` 构造时会先把策略名规范化再存入 `self.selection_strategy`。

文档解释边界：

- preset 是 `selection policy`，不是方法学真理。
- 改变 preset 只能改变最终排序偏好，不能改变各 `scores` 与 `overall_score` 的原始解释。

## 6. Repeat 语义合同

### 6.1 重复运行是如何执行的

`BaseEvaluator.evaluate_method_repeated()` 当前按“同一方法重复评估多次，再聚合”的方式工作：

1. 调 `evaluate_method()` 执行 `n_repeats` 次。
2. 若 `kwargs["random_state"]` 是整数，则每次重复自动改为 `base_random_state + repeat_idx`。
3. 仅用成功重复来计算聚合分、标准差和区间。
4. 返回一个聚合后的 `EvaluationResult`，以及一个用于产物落盘的单个最佳成功容器。

前置约束：

- `n_repeats >= 1`
- `0 < confidence_level < 1`

否则会触发 `ValueError`。

### 6.2 聚合统计如何定义

若至少有一次成功重复：

- `scores` 是“成功重复上的均值”
- `overall_score` 是“成功重复上的均值”
- `execution_time` 是“成功重复上的均值”
- `overall_score_std` 是成功重复总体分的标准差，使用 `ddof=0`
- `overall_score_ci_lower / upper` 是基于经验分位数的区间，不是参数统计学意义下的严格置信区间
- `repeat_overall_scores` 只保存成功重复的总体分

重要实现事实：

- `n_repeats` 记录的是请求重复次数，不是成功次数。
- 只要存在成功重复，聚合结果的 `error` 就会是 `None`。
- 当前结果对象不显式记录“有多少次重复失败”；也就是说，部分失败信息在聚合后不会完整保留。

### 6.3 所返回的实际产物不是“均值产物”

这是后续重构最容易误改的一点：

- 报告里的 `overall_score / std / interval` 来自“成功重复上的聚合统计”。
- 但实际附加到结果容器中的 layer / assay / obs 产物，并不是平均后的产物。
- 当前实现会选取“该方法所有成功重复中 `overall_score` 最高的一次”的产物作为实际保留产物。

因此：

- report 统计与最终附加到 container 的矩阵，不是一一同构的“同一个对象”。
- `keep_all=True` 时，每个方法保留的也是该方法“最佳成功重复”的产物，不是 repeat ensemble。

后续若要改成 ensemble / mean artifact，必须视为合同级变更。

### 6.4 Imputation 的重复特殊性

`ImputationEvaluator` 当前使用 holdout-based scoring：

- 先从 observed entries 中抽取一批 deterministic holdout
- 将这些 holdout 临时遮蔽为缺失
- 用 ground truth 与预测值比较 `rmse / correlation`
- 最后在输出层中把 holdout 位置恢复回原始 observed 值

当前 holdout mask 使用固定随机种子 `42` 构造，因此：

- 在默认实现下，repeat 之间不会因为 holdout 再抽样而改变评估切分。
- repeat 区间主要反映方法自身随机性或传入 `random_state` 的变化，不反映“不同 holdout 抽样”的不确定性。

## 7. Stage 边界合同

### 7.1 统一 stage I/O 描述

`AutoSelector._attach_stage_io()` 会在 evaluator 返回后统一补齐：

- `input_assay`
- `input_layer`
- `output_assay`
- `output_layer`
- `output_obs_key`

当前输出 kind 规则固定为：

- `normalize / impute / integrate` -> `layer`
- `reduce` -> `assay`
- `cluster` -> `obs`

因此：

- 稳定预处理 stage 的输出仍在同一 assay 下，只是新增 layer。
- `reduce` 产物被视为新 assay，后续上下文层名固定为 `"X"`。
- `cluster` 产物写入 `obs[...]`，`output_obs_key` 才是权威输出位置。

### 7.2 `normalize`

当前 `NormalizationEvaluator` 额外冻结以下边界：

- 只在 source layer 具有显式 log provenance 时，才自动纳入 `norm_quantile` 与 `norm_trqn`
- 若 source layer 是 `raw` 或 unknown scale，则候选集限制为 `norm_none / norm_mean / norm_median`
- logged 判定来自共享内部 detector，而不是对 `transformation` 私有 helper 的跨阶段直接导入
- `method_contracts` 当前会为每个方法填充：
  - `input_scale_requirement`
  - `source_layer_logged`
  - `comparison_scale`
  - `candidate_scope`
- `recommendation_reason` 会前置拼接 scale gate 解释

### 7.3 `impute`

当前 `ImputationEvaluator` 额外冻结以下边界：

- 方法名是精简名，如 `zero / row_mean / knn / qrilc`
- 结果 layer 名仍按 `{source_layer}_{method_name}` 写入
- 若 holdout 条件不足，会回退到基类单次评估路径
- 当前没有像 `normalize / integrate` 那样统一填充 `method_contracts`
- 输出层必须存在且 shape 必须与源矩阵一致，否则该方法按失败处理

### 7.4 `integrate`

当前 `IntegrationEvaluator` 额外冻结以下边界：

- stage 前置要求 `obs[batch_key]` 存在，否则 `run_stage()` 直接报错
- 默认只比较 matrix-level 且 `recommended_for_de=True` 的 stable 候选
- 只有 `include_embedding_methods=True` 时，才纳入 `mnn / harmony / scanorama` 这类 exploratory embedding-level 方法
- `method_contracts` 当前会为每个方法填充：
  - `integration_level`
  - `recommended_for_de`
  - `candidate_scope`
- 其余 integration 合同元数据当前还会显式写明：
  - `selection_batch_metric = "batch_mixing"`
  - `selection_batch_metric_kind = "heuristic_proxy"`
  - `standardized_batch_metrics = ["batch_kbet", "batch_ilisi"]`
- 也就是说：
  - `overall_score` / `selection_score` 当前仍使用旧的 proxy `batch_mixing`
  - `batch_kbet` / `batch_ilisi` 已进入结果分项，用于 standardized diagnostics / reporting
  - 在显式版本化前，不得把排序逻辑静默切换到 standardized 指标
- `recommendation_reason` 会明确声明本次是 stable-only 还是 exploratory-inclusive 候选集

### 7.5 `reduce / cluster`

`reduce / cluster` 当前仍在 `AutoSelector` 的实现范围内，但文档边界必须保持：

- 代码层可运行
- 仓库发布边界上属于 experimental downstream helpers
- 它们不应反向改写稳定 preprocessing stage 的设计合同

## 8. 输出合同

### 8.1 `run_stage()`

`AutoSelector.run_stage()` 返回：

```python
(result_container, stage_report)
```

其中：

- `result_container` 是一个复制后再附加结果的容器，不应与输入容器共享“隐式原地修改”语义
- `stage_report` 是本 stage 的完整评估记录

当前前置校验包括：

- assay 必须存在
- source layer 必须存在
- `integrate` 时 `batch_key` 必须在 `obs` 中

### 8.2 `run()`

`AutoSelector.run()` 会按 stage 顺序串行执行，并把上一 stage 的选中产物作为下一 stage 上下文：

- 对 layer 型 stage，下一 stage 的 `source_layer = best_result.layer_name`
- 对 assay 型 stage，下一 stage 的上下文切到新 assay，layer 固定 `"X"`
- 对 cluster stage，不更新 layer 上下文

因此：

- `run()` 是“按最佳结果串联”的流水线，而不是“保留多分支并行候选”的 DAG。
- `report.stages` 记录的是每个 stage 自己的完整对比，但 stage 之间的衔接只沿 `best_result` 前进。

### 8.3 `keep_all`

`keep_all` 只影响“结果容器里保留哪些成功产物”，不影响 report 完整性：

- `keep_all=False`：只把选中方法的产物附加到结果容器
- `keep_all=True`：把所有成功方法的产物都附加到结果容器，并始终包含最佳方法产物
- 无论 `keep_all` 取值如何，`StageReport.results` 都必须保留全部已评估方法

### 8.4 AutoSelect 结果命名与仓库 canonical layer 命名是两套语义

当前 AutoSelect 评估产物名通常是：

```text
{source_layer}_{method_name}
```

例如：

- `raw_norm_mean`
- `log_knn`
- `imputed_limma`

这类名字的语义是“比较阶段中的方法产物标识”，不是仓库面向终态数据层的 canonical layer taxonomy。

因此必须区分：

- AutoSelect artifact naming：为了比较和追踪方法来源
- canonical layer naming：仓库层面稳定使用的 `raw / log / norm / imputed / zscore`

后续若要把 AutoSelect 选中结果 promote 成 canonical stage layer，应作为显式转换步骤处理，不能假定 `{source_layer}_{method_name}` 本身就是 canonical final layer。

### 8.5 导出格式的保真度

`AutoSelectReport.save()` 当前支持：

- `markdown`
- `json`
- `csv`

保真度约束：

- `json` 是最接近完整结构的机器输出，保留 stage 嵌套与 result 全字段
- `csv` 是扁平化导出，适合表格分析，但不是完整结构镜像
- `markdown` 是人类可读汇总，不适合作为严格机器协议

当前 `csv` 不保留完整 `method_contracts` 结构，因此若需要完整合同元数据，应优先消费 `json`。

## 9. Failure 合同

### 9.1 构造期失败

以下情况会在构造或准备阶段直接抛错，而不是写入某个方法失败结果：

- 非法 `stage`
- 非法 `selection_strategy`
- `n_repeats < 1`
- `confidence_level` 不在 `(0, 1)` 内
- stage 输入 assay / layer 缺失
- `integrate` 所需 `batch_key` 缺失

这些都属于“stage 不可运行”的硬错误。

### 9.2 单方法失败

单方法失败时：

- evaluator 捕获异常
- `error` 记录为 `"{ExceptionType}: {message}"`
- `scores` 退化为对应指标键的全 `0.0`
- `overall_score = 0.0`
- `selection_score` 最终也会被写为 `0.0`

这类失败不会让整个 stage 立即抛异常，除非外层还有独立前置校验未通过。

### 9.3 全方法失败

若一个 stage 内所有方法都失败：

- `best_method = ""`
- `best_result = None`
- `recommendation_reason = "All methods failed"`
- 返回 `container.copy()`，不附加新产物

这属于“stage 已执行完，但没有可选赢家”，不是输入校验型错误。

### 9.4 重复运行中的失败合并

若某方法的全部重复都失败：

- 返回单条失败 `EvaluationResult`
- `error` 为各失败消息去重后再用 `"; "` 拼接
- `execution_time` 为所有失败重复耗时均值
- `repeat_overall_scores` 为空列表
- `overall_score_std = 0.0`
- `overall_score_ci_lower = 0.0`
- `overall_score_ci_upper = 0.0`

若某方法只有部分重复失败、部分成功：

- 当前聚合结果会被视为成功结果
- 失败重复不会在最终 `EvaluationResult` 中被完整保留

这是一项当前实现限制，后续若补充 repeat-level failure accounting，属于兼容增强。

### 9.5 多 stage 运行中的异常传播

`AutoSelector.run()` 当前会先把 stage 异常写入 warning，再立即重新抛出异常。

因此当前语义不是“跳过坏 stage 继续跑后续 stage”，而是：

- 记录 warning
- 终止本次多 stage 运行
- 将异常继续交给调用方

## 10. 优化安全不变量

后续对 `autoselect` 做性能优化、结构重构或文档收敛时，以下约束应默认视为不可破坏：

1. `overall_score` 与 `selection_score` 必须保持语义分离。
2. runtime 只能影响 `selection_score`，不能偷偷混入 `overall_score`。
3. `StageReport.results` 必须保留成功与失败方法，不能只保留 winner。
4. `keep_all` 只能影响结果容器中的保留产物，不能影响 report 内容。
5. `StageReport.metric_weights` 必须记录“本次实际生效的权重快照”，不能只保留默认值名称。
6. `method_contracts` 和 `EvaluationResult.method_contract` 必须继续允许为空；消费端不能被迫假设“所有 stage 都填充”。
7. 重复评估的统计汇总与最终保留产物必须继续被视为两层对象，不能默认视为同一个矩阵。
8. `selection_score` 的 runtime 归一化范围必须局限在当前 stage、当前成功候选集内部。
9. `recommendation_reason` 应继续保持人类可读解释属性，不应被提升为必须稳定解析的机器协议。
10. `reduce / cluster` 不得因为代码可用而被文档表述成稳定 preprocessing core。
11. AutoSelect artifact naming 不得与仓库 canonical layer naming 混为一谈。
12. `parallel / n_jobs` 在未真正实现前，不应被文档或调用方推断为已生效的执行保证。

## 11. 当前建议的测试锚点

若后续要补充或重构 `autoselect` 测试，最应优先冻结的行为包括：

1. `selection_strategy` 非法值报错。
2. `n_repeats / confidence_level` 非法值报错。
3. `run_stage()` 的 assay / layer / batch_key 前置校验。
4. `overall_score` 与 `selection_score` 的分离。
5. `keep_all` 只影响容器产物，不影响 `StageReport.results`。
6. 单方法失败被捕获而不是导致整 stage 崩溃。
7. 全方法失败时 `best_method == ""` 且 `best_result is None`。
8. 重复运行时统计值来自成功重复，而实际保留产物来自最佳成功重复。
9. `normalize` 的 log-scale gate 和 `integrate` 的 stable-vs-exploratory 候选边界。
10. `json` 导出保留完整结构，而 `csv` 为扁平化子集。

## 12. 本合同的依据

本合同基于以下当前仓库材料整理：

- `scptensor/autoselect/core.py`
- `scptensor/autoselect/strategy.py`
- `scptensor/autoselect/evaluators/base.py`
- `scptensor/autoselect/evaluators/normalization.py`
- `scptensor/autoselect/evaluators/imputation.py`
- `scptensor/autoselect/evaluators/integration.py`
- `scptensor/autoselect/report.py`
- `docs/internal/review_autoselect_scoring_20260312.md`

若未来实现与本文档冲突，应把冲突先暴露出来，再决定是修代码、修文档，还是显式升级合同版本，而不是静默漂移。
