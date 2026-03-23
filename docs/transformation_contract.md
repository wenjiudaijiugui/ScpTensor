# ScpTensor 变换冻结实现合同（`scptensor.transformation`，2026-03-16）

## 1. 文档目标

本文档冻结 `scptensor.transformation` 当前实现合同，供后续核心代码优化、API 收敛、测试补强与 workflow 文档统一使用。

当前该模块的稳定公开能力只有一个：

- `log_transform`

本文档回答的是：

- 当前 log transform 到底覆盖什么，不覆盖什么。
- 输入层、已 logged 检测、负值处理、稀疏路径、mask/provenance 合同是什么。
- 哪些当前行为虽然不完美，但不能在重构时被悄悄改掉。

本文档基于以下仓库内事实：

- `scptensor/transformation/__init__.py`
- `scptensor/transformation/base.py`
- `scptensor/transformation/log_transform.py`
- `scptensor/__init__.py`
- `scptensor/autoselect/evaluators/normalization.py`
- `tests/core/test_transformation_api.py`
- `tests/transformation/test_log_transform.py`
- `tests/autoselect/test_normalization_evaluator.py`
- `docs/core_data_contract.md`
- `docs/review_log_scale_20260312.md`
- `docs/review_io_state_mapping_20260312.md`

## 2. 范围与非范围

### 2.1 范围

本文档只覆盖：

- `log_transform`
- `validate_assay_and_layer`
- `create_result_layer`

### 2.2 非范围

以下内容不属于本文档：

- normalization
- imputation
- batch correction / integration
- peptide -> protein aggregation
- benchmark 排名结论

必须明确：

- `log_transform` 属于 `scptensor.transformation`，不是 `scptensor.normalization` 的一部分。
- tests 已固定：`scptensor.normalization.__all__` 中不应导出 `log_transform`。

## 3. 稳定主线边界

按项目合同，stable preprocessing 主线最终回到 protein-level quantitative matrix。因此：

- `log_transform` 的稳定默认场景是 `proteins/raw -> log`
- `peptides` 技术上可用，但不应作为 stable preprocessing contract 的主驱动场景
- `transformation` 只改变数值尺度，不改变 assay feature universe

## 4. 当前公开 API

`scptensor.transformation.__all__` 当前只导出：

- `log_transform`

`scptensor.__all__` 当前也只从 transformation 顶层重导出：

- `log_transform`

其默认参数为：

- `assay_name="protein"`
- `source_layer="raw"`
- `new_layer_name="log"`
- `base=2.0`
- `offset=1.0`
- `use_jit=True`
- `detect_logged=True`
- `skip_if_logged=True`
- `detect_logged_by_distribution=False`

这意味着：

1. 文档级 canonical 输出层是 `log`。
2. 当前 API 默认开启显式 logged 检测，但默认不启用 distribution heuristic。
3. `use_jit=True` 表示允许底层 sparse helper 按阈值判断是否尝试 JIT，不保证一定进入 numba 分支。
4. `log_transform` 是 transformation 唯一的顶层 stable API；helper 不升格到顶层 `scptensor`。

## 5. assay / layer 输入合同

### 5.1 assay alias

`validate_assay_and_layer()` 当前通过 `resolve_assay_name()` 支持：

- `protein <-> proteins`
- `peptide <-> peptides`

因此：

- 调用层可以使用 `protein` / `proteins`
- 仓库文档仍优先使用 `proteins`

### 5.2 输入层验证

当前 `log_transform()` 会强制要求：

- assay 存在，否则 `AssayNotFoundError`
- `source_layer` 存在，否则 `LayerNotFoundError`

除此之外，它不会额外强制检查：

- 是否是 protein-level assay
- 是否已经 normalization
- 是否已经 imputation

这些仍属于 caller responsibility。

## 6. 已 logged 检测合同

### 6.1 默认检测路径

当前 `detect_logged=True` 时，`log_transform()` 会按以下顺序检测 source 是否看起来已经是 log scale：

1. layer name 是否匹配 `log / log2 / log10 / ln`
2. provenance history 中是否存在 `log_transform` 或 `log_transform_skipped`，且其 `new_layer_name == source_layer`
3. 仅当 `detect_logged_by_distribution=True` 时，再对数值分布做启发式推断

默认情况下，第 3 条是关闭的。

### 6.2 distribution heuristic 的当前定位

当前 `_data_suggests_logged()` 是启发式推断，不是严格统计判别器。

它会基于：

- 分位数
- 动态范围
- 负值比例
- 小值/大值比例

来判断数据是否“像 log scale”。

稳定解释应为：

- 这是 optional heuristic
- 默认关闭，避免把低范围线性强度误判成已 logged
- 它只能作为保护性 gate，不能升级成 stable scientific classifier

### 6.3 `skip_if_logged` 的当前行为

若检测到 source layer 已 logged：

- 总是发出 `UserWarning`
- 若 `skip_if_logged=True`
  - 不再重复做 log
  - 若 `new_layer_name != source_layer`，创建 passthrough layer
  - 写入 history action `log_transform_skipped`
  - 返回原 container
- 若 `skip_if_logged=False`
  - 仍发 warning
  - 继续强制做第二次 log
  - 写入 history action `log_transform`

tests 已固定以上两条路径。

### 6.4 transformation 与 normalization 的 logged 边界

`scptensor.autoselect.evaluators.normalization.NormalizationEvaluator` 当前不会自己发明一套新的 logged 判别规则，而是复用：

- `scptensor.transformation.log_transform._detect_already_logged(...)`

但调用时明确固定：

- `detect_logged_by_distribution=False`

因此，对 normalization scale gate 来说，当前“显式 log provenance”只来自两类信号：

1. source layer 名本身看起来就是 log scale
   - 例如 `log` / `log2` / `log10` / `ln`
2. history 中存在：
   - `log_transform`
   - `log_transform_skipped`
   且其 `new_layer_name == source_layer`

这意味着当前稳定边界是：

- raw/unknown-scale layer 不会仅因数值分布“像 log”就自动重新开放 `norm_quantile` / `norm_trqn`
- 但如果用户显式运行了 `log_transform()`，即使最终走的是 `log_transform_skipped` passthrough 路径，只要生成了新的目标层并留下 matching history，normalization scale gate 就会把该层视为 logged
- `normalization` 中针对 vendor-normalized `raw` 的 warning 不会把 raw 层升级为 logged layer provenance

## 7. 数值变换合同

### 7.1 主变换公式

当前 dense path 的实际变换是：

```text
log(x + offset) / log(base)
```

默认即 `log2(x + 1)`。

### 7.2 负值处理

当前实现不会拒绝负值输入，而是：

- 发出 `UserWarning`
- 把所有负值裁剪到 `0`
- 然后再做 log transform

这对 dense 与 sparse 都成立。

因此当前 stable contract 不是“负值即报错”，而是“warning + clip-to-zero”。

### 7.3 稀疏路径

若输入是 sparse matrix：

- 仅当 `offset == 1.0` 时，使用 `sparse_safe_log1p_with_scale(...)`
- 且再通过 `ensure_sparse_format(..., "csr")` 强制转成 `csr`
- 若 `offset != 1.0`，为了满足
  - `log(x + offset) / log(base)`
  - 以及 structural zeros 也必须变成 `log(offset) / log(base)`
  当前实现会 densify 后走 dense NumPy 公式路径

若输入是 dense：

- 直接用 NumPy 公式计算

### 7.4 `base` 与 `offset` 验证

当前显式参数检查与定义域检查是：

- `base` 必须有限、`> 0` 且 `!= 1`
- `offset` 必须有限且 `>= 0`
- 若 `offset == 0`，则输入值必须严格 `> 0`
  - 否则会因 `log(0)` 进入非法域，当前实现会直接报错，而不是产出 `-inf`

这意味着当前代码事实是：

- `base <= 0`、`base == 1`、非有限 `base` 会抛 `ScpValueError`
- `offset < 0`、非有限 `offset` 会抛 `ScpValueError`
- `offset == 0` 且输入中存在 `0/负值` 会抛 `ValidationError`

## 8. 输出层与对象语义合同

### 8.1 transform 不改变轴定义

`log_transform()` 必须保持：

- 相同 assay
- 相同 `obs` 样本轴
- 相同 `var` 特征轴
- 相同矩阵 shape

不得在变换阶段混入：

- feature filtering
- sample filtering
- aggregation
- normalization

### 8.2 返回模式

当前 `log_transform()` 是：

1. 验证 assay / layer
2. 检测是否已 logged
3. 必要时做剪裁和数值变换
4. 在原 assay 上新增或覆盖 layer
5. 写 `container.history`
6. 返回同一个 `container`

因此它不是纯函数式 copy-return API。

### 8.3 同名目标层行为

通过 `assay.add_layer()` 写层时：

- 若 `new_layer_name` 已存在，则静默覆盖
- transformation 模块当前不加 warning

但 `skip_if_logged=True` 的 skipped 路径有一个特例：

- 若 `new_layer_name == source_layer`
  - 不会创建新层
  - 不会替换 source layer registry entry
  - 只会 append `log_transform_skipped` history

### 8.4 mask 与 metadata 的当前事实

`create_result_layer()` 当前定义为：

- `X = transformed values`
- `M = source_layer.M`

这里必须明确：

1. 当前不是 `M.copy()`，而是直接复用同一个 mask 引用。
2. `metadata` 不会复制，新 layer 的 metadata 当前是默认 `None`。

因此，当前 transformation 与 normalization / integration 的 mask 行为并不相同。

这属于重要实现事实；若后续要改成 `M.copy()` 或复制 metadata，都属于显式合同变更。

### 8.5 skipped passthrough 层合同

当检测到已 logged 且 `skip_if_logged=True`、`new_layer_name != source_layer` 时：

- 会创建 passthrough layer
- `X` 是 source 的安全拷贝
- `M` 仍通过 `create_result_layer()` 复用源层 mask 引用

因此 passthrough 层的数值不共享，但 mask 当前共享。

## 9. provenance 与 history 合同

当前 `log_transform()` 只会写两类 history action：

- `log_transform`
- `log_transform_skipped`

### 9.1 `log_transform` 最小 params

当前至少包含：

- `assay`
- `source_layer`
- `new_layer_name`
- `base`
- `offset`
- `sparse_input`
- `use_jit`
- `detect_logged`
- `detect_logged_by_distribution`
- `skip_if_logged`
- `already_logged_detected`
- `logged_detection_reason`

### 9.2 `log_transform_skipped` 最小 params

当前至少包含：

- `assay`
- `source_layer`
- `new_layer_name`
- `base`
- `offset`
- `detect_logged`
- `detect_logged_by_distribution`
- `skip_if_logged`
- `reason`

### 9.3 assay 名写入 history 的规范化

当前 history 中写入的是 resolved assay name，而不是调用时原样字符串。

tests 已固定：

- 即使调用时传 `proteins`
- history params 里的 `assay` 仍会写成解析后的 canonical 当前值

### 9.4 与 normalization scale gate 的 provenance 协作

当前 normalization 的 scale-sensitive candidate gate 直接消费 transformation 的 logged provenance 结果：

- `log_transform` 写出的目标层
- `log_transform_skipped` 写出的 passthrough 目标层

都可作为 normalization 侧“显式 logged provenance”的依据。

因此后续若修改以下任一项：

- `log_transform` / `log_transform_skipped` action 名
- history params 中的 `new_layer_name`
- `_detect_already_logged()` 的 history 判定条件

都必须同步评估 normalization / AutoSelect 的兼容性影响，不能只改 transformation 单侧。

## 10. failure / warning contract

### 10.1 显式异常

- `base <= 0`：`ScpValueError`
- `offset < 0`：`ScpValueError`
- 缺失 assay：`AssayNotFoundError`
- 缺失 layer：`LayerNotFoundError`

### 10.2 显式 warning

- source 看起来已 logged：`UserWarning`
- 输入包含负值：`UserWarning`

### 10.3 当前不会抛出的情况

当前实现不会因为以下条件直接报错：

- 负值输入
- 已 logged 输入
- distribution heuristic 判定已 logged

这些都走 warning 路径，而不是硬失败。

## 11. 优化安全不变量

后续若对 `scptensor.transformation` 做性能优化或 API 收敛，以下行为不能被悄悄改掉：

1. `log_transform` 必须继续与 `normalization` 边界分离。
2. 输出层 shape、样本顺序、特征顺序必须保持不变。
3. 默认主线语义仍是 `raw -> log`，而不是隐式 `raw -> norm/log` 混合变换。
4. 已 logged 检测默认启用，但 distribution heuristic 默认关闭，这一保守 gate 不能无声改变。
5. `skip_if_logged=True` 时必须保留 passthrough + history 的可审计语义，而不是简单返回不留痕。
6. 负值当前是 warning + clip-to-zero；若未来改成硬失败，需要显式更新合同与测试。
7. 当前 `M` 共享引用、metadata 不复制的事实若要改变，必须视为显式兼容性决策。
