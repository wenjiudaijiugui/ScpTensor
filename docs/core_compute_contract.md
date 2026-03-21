# ScpTensor Core 低层计算冻结实现合同（`scptensor.core` compute surface，2026-03-17）

## 1. 文档目标

本文档冻结 `scptensor.core` 中低层计算面的当前实现合同，服务于后续：

- `core` 代码优化
- 稀疏/稠密路径重构
- JIT 加速替换
- 测试补强与回归保护

它回答的不是“未来怎样实现更快”，而是“当前哪些计算语义不能在优化时被悄悄改掉”。

本文档基于以下仓库内事实：

- `scptensor/core/__init__.py`
- `scptensor/core/structures.py`
- `scptensor/core/matrix_ops.py`
- `scptensor/core/jit_ops.py`
- `scptensor/core/sparse_utils.py`
- `tests/core/test_mask_codes.py`
- `tests/core/test_matrix_ops_regressions.py`
- `tests/core/test_matrix_ops_sparse.py`
- `tests/core/test_sparse_utils.py`
- `tests/core/test_jit_sparse_ops.py`
- `tests/core/test_sparse_row_operation.py`

## 2. 范围与非范围

### 2.1 范围

本文档覆盖的低层计算面包括：

- `MatrixOps`
- 稀疏格式检测、转换、乘法、row/col reduction、log helper
- `NUMBA_AVAILABLE` 与对外导出的 JIT helper
- 稀疏 mask 的隐式 `VALID` 语义
- 稠密 / 稀疏切换时的 shape、dtype、返回类型、不变量

### 2.2 非范围

本文档不覆盖：

- `ScpContainer / Assay / ScpMatrix` 的整体数据模型合同
- aggregation / transformation / normalization / imputation / integration / qc 的阶段逻辑
- 下游 embedding / clustering 算法
- benchmark 结果解释

这些内容分别受其他合同文档约束。

## 3. 当前低层公开面

### 3.1 `scptensor.core` 顶层已导出计算面

`scptensor.core.__all__` 当前对外暴露的低层计算入口包括：

- `MatrixOps`
- `NUMBA_AVAILABLE`
- `count_mask_codes`
- `find_missing_indices`
- `apply_mask_threshold`
- `fill_missing_with_value`
- `is_sparse_matrix`
- `get_sparsity_ratio`
- `to_sparse_if_beneficial`
- `ensure_sparse_format`
- `sparse_copy`
- `cleanup_layers`
- `get_memory_usage`
- `optimal_format_for_operation`
- `auto_convert_for_operation`
- `sparse_row_operation`
- `sparse_col_operation`
- `sparse_multiply_rowwise`
- `sparse_multiply_colwise`
- `sparse_center_rows`
- `sparse_safe_log1p`
- `get_format_recommendation`

### 3.2 仓库内已形成的“合同相关但非顶层导出” helper

虽然以下函数没有从 `scptensor.core.__init__` 统一导出，但它们已被仓库内稳定模块或测试依赖，因此属于“实现重构不可随意破坏”的内部计算面：

- `scptensor.core.sparse_utils.sparse_safe_log1p_with_scale`
- `scptensor.core.jit_ops._sparse_row_sum_jit`
- `scptensor.core.jit_ops._sparse_row_mean_jit`

这里的判断标准不是“是否 public marketing API”，而是“是否已经形成稳定可观测行为”。

## 4. 基础不变量

### 4.1 shape 不变量

所有低层计算都必须维持：

- `X.shape == (n_samples, n_features)`
- `M.shape == X.shape`

若 `ScpMatrix.M is None`，`ScpMatrix.get_m()` 当前会返回同 shape 的全零 mask：

- 稠密 `X` -> 稠密 `int8` 零矩阵
- 稀疏 `X` -> `csr_matrix` 稀疏零矩阵

因此低层计算代码不能把 “`M is None`” 和 “没有 mask 语义” 混为一谈；当前真实语义是“缺省时所有位置都被解释为 `VALID`”。

### 4.2 dtype 不变量

`ScpMatrix.__post_init__()` 当前保证：

- `X` 会被规范到浮点类型
- `M` 会被规范到 `int8`

低层工具在优化时可以改变中间实现，但不能把稳定输出改成：

- 非浮点 `X`
- 非整数 mask
- shape 不一致的 `X/M`

### 4.3 非原地语义

`MatrixOps` 当前所有对外操作都先 `matrix.copy()`，因此输入 `ScpMatrix` 不应被原地改写。

这条语义对后续 provenance、缓存和上游模块非常关键。即使未来内部使用 view、临时 buffer 或 JIT kernel，也必须保持“返回新对象、输入对象不被静默修改”这一稳定合同。

### 4.4 mask code 枚举不变量

当前稳定 mask code 数值为：

- `VALID = 0`
- `MBR = 1`
- `LOD = 2`
- `FILTERED = 3`
- `OUTLIER = 4`
- `IMPUTED = 5`
- `UNCERTAIN = 6`

这些整数值已被：

- `MaskCode`
- `jit_ops`
- 回归测试
- 多个上游/下游模块

共同使用，因此不得在优化中重排或复用。

## 5. 稀疏 mask 语义

### 5.1 规范语义

对稀疏 `M`，当前稳定解释是：

- 未显式存储的位置 = `VALID`
- 非零存储位置 = 显式非 `VALID` 状态，或显式存储的状态码

`tests/core/test_matrix_ops_regressions.py` 已明确锁定：

- 稀疏统计必须按完整 shape 计数，而不是只按 `nnz`
- 当 `VALID` 不在保留集合中时，隐式 `VALID` 项也必须被过滤
- `apply_mask_to_values()` 只能处理真正无效的位置，不能误清空所有结构零

因此，后续任何稀疏优化都不能把“未存储位置”解释成“未知”或“缺失”。

### 5.2 结构零不等于缺失

对稀疏 `X`：

- `X` 的结构零只是数值零 / 未存储零
- 是否有效由 `M` 决定

也就是说，低层计算不能用 “`X == 0`” 代替 mask 语义判断。

## 6. `MatrixOps` 合同

### 6.1 查询类 helper

以下函数当前都返回与矩阵同 shape 的布尔结果：

- `get_valid_mask()`: `M == VALID`
- `get_missing_mask()`: `M != VALID`
- `get_missing_type_mask()`: `M == 指定 mask code`

这里的“missing”在当前实现里是广义的“任何非 `VALID` 状态”，并不只等于 `MBR + LOD`。

### 6.2 标记类 helper

以下函数都通过 `mark_values()` 实现：

- `mark_imputed()`
- `mark_outliers()`
- `mark_lod()`
- `mark_uncertain()`

当前稳定行为：

- 返回新的 `ScpMatrix`
- 只修改 `M`
- 不修改 `X`
- 稀疏 `M` 路径使用 `LIL -> CSR` 转换以完成写入

因此后续若更换实现，至少要保留：

- 输入对象不变
- 输出 `M` 上指定位置被写成对应 mask code
- 稀疏结果仍可继续被下游 `MatrixOps` / `sparse_utils` 消费

### 6.3 `combine_masks()`

`combine_masks(masks, operation)` 当前支持：

- `union`
- `intersection`
- `majority`

稳定边界：

- 空列表抛 `ValueError("No masks provided")`
- 单个 mask 时返回其 copy
- 未知操作抛 `ValueError("Unknown operation: ...")`

### 6.4 `get_mask_statistics()`

这是当前最重要的回归保护点之一。

稳定语义：

- 返回每个 `MaskCode` 的 `count` 和 `percentage`
- 百分比以完整矩阵元素数 `n_rows * n_cols` 为分母
- 对稀疏 `M`，隐式未存储位置必须计为 `VALID`

也就是说：

- 稀疏 `M` 的 `VALID` 数不能等于 `np.count_nonzero(M.data == 0)`
- 必须按“总元素数 - 显式非零项数”回推

### 6.5 `filter_by_mask()`

`filter_by_mask(matrix, keep_codes)` 的当前稳定行为是：

- 保留 `keep_codes` 内的位置
- 其余位置的 `X` 设为 `NaN`
- 其余位置的 `M` 设为 `FILTERED`

需要特别冻结的细节：

1. 若 `M` 为稀疏且 `VALID` 在保留集合内：
   - 隐式 `VALID` 位置保持可用
   - 只对显式非保留位置做过滤
2. 若 `M` 为稀疏且 `VALID` 不在保留集合内：
   - 隐式 `VALID` 也必须一起过滤
   - 当前实现会 materialize 全量 dense mask 以保证正确性
   - 此路径可能导致 `X` 从 sparse 变成 dense

因此，后续优化不能为了“避免 densify”而牺牲这条正确性语义。

### 6.6 `apply_mask_to_values()`

`apply_mask_to_values(matrix, operation)` 当前仅支持：

- `zero`
- `nan`
- `keep`

稳定行为：

- 未知 `operation` 抛 `ValueError`
- `keep` 是 no-op
- 对无效位置：
  - `zero` 写 `0.0`
  - `nan` 写 `np.nan`

稀疏路径还存在两个已被测试锁定的细节：

1. 只处理显式无效位置，不能误伤隐式 `VALID`
2. `zero` 在 sparse `X` 上会 `eliminate_zeros()`，因此结果仍偏向 sparse-preserving

## 7. 稀疏工具合同

### 7.1 稀疏识别与转换

当前稳定入口：

- `is_sparse_matrix()`
- `get_sparsity_ratio()`
- `to_sparse_if_beneficial()`
- `ensure_sparse_format()`
- `auto_convert_for_operation()`
- `optimal_format_for_operation()`
- `get_format_recommendation()`

其中：

- `get_sparsity_ratio()` 对稀疏矩阵按 `1 - nnz / total` 计算
- `to_sparse_if_beneficial()` 仅在 `sparsity > threshold` 时将 dense 转为 sparse
- `ensure_sparse_format()` 当前支持 `csr / csc / coo / lil`
- 非支持格式会抛 `ValueError`

### 7.2 格式推荐是启发式，不是数值语义

`optimal_format_for_operation()` 和 `get_format_recommendation()` 当前只是工程启发式推荐器，不承诺最优性能证明。

但它们的输出方向已形成稳定语义：

- row-wise -> 倾向 `csr`
- col-wise -> 倾向 `csc`
- modification / construction -> 倾向 `lil`
- 稀疏度低于阈值时可推荐 `dense`

后续可以调整阈值或策略，但不能让调用方对这些返回值的基本含义失去判断依据。

### 7.3 `sparse_copy()` 与 `get_memory_usage()`

当前合同：

- `sparse_copy()` 对稠密 / 稀疏都返回独立副本
- `get_memory_usage()` 返回：
  - `nbytes`
  - `is_sparse`
  - `shape`
  - `dtype`

对稀疏矩阵，`nbytes` 当前按 `data + indices + indptr` 统计。

### 7.4 `sparse_row_operation()` / `sparse_col_operation()`

这两个函数的当前稳定语义非常重要：

- 输入会先转换成最适合遍历的格式：
  - row -> CSR
  - col -> CSC
- `func` 只作用于显式存储的非零数据向量
- 空行 / 空列当前返回 `0.0`

因此：

- `sparse_row_operation(X, np.mean)` 当前计算的是“显式存储值的均值”
- 它不是 dense 语义下把结构零也纳入分母的 mean

这一点已被 `tests/core/test_jit_sparse_ops.py` 与 `tests/core/test_sparse_row_operation.py` 锁定，后续优化不得偷偷改成 dense-style mean。

### 7.5 JIT 快路径与 fallback 行为

`sparse_row_operation()` 当前存在两层语义：

1. 若 numba 可用且函数是 `np.sum` / `np.mean`，走 `_sparse_row_sum_jit` / `_sparse_row_mean_jit`
2. 否则逐行取 `row_data` 走 Python/NumPy fallback

这意味着稳定合同不是“必须用 numba”，而是：

- 有无 numba 都必须给出同一数值语义
- JIT 只是性能实现细节

### 7.6 行列乘法

`sparse_multiply_rowwise()` / `sparse_multiply_colwise()` 当前稳定行为：

- 输入必须是 sparse-like，可被转成 `csr` / `csc`
- `factors` 长度必须与对应轴相等
- 长度不符抛 `ValueError`
- 若输入数据不是浮点，会先转浮点
- 返回 sparse 结果，不 densify

### 7.7 `sparse_center_rows()` 的真实边界

函数名容易误导，但当前真实行为是：

- 计算每行显式非零项的均值
- 返回 `(原始 CSR 矩阵, row_means)`
- 并不会真正把矩阵数值改成“已中心化结果”

因此：

- 下游若依赖它，必须把返回的第二项当作中心化参数使用
- 后续重构不能把它悄悄改成“返回 dense centered matrix”而不改合同

### 7.8 `sparse_safe_log1p()` / `sparse_safe_log1p_with_scale()`

当前稳定语义：

- 稀疏输入只变换显式存储的 `data`
- 结构零仍保持为零，不会 materialize
- 稠密输入走标准 NumPy 表达式
- 大矩阵时可按 `_JIT_THRESHOLD` 选择 numba kernel

因此：

- `log1p` helper 的核心合同是“数值语义一致 + 尽量保 sparse”
- 不是“必须强制开启 JIT”

## 8. `jit_ops` 合同

### 8.1 `NUMBA_AVAILABLE`

`NUMBA_AVAILABLE` 当前只是能力标记，不是行为开关合同。

稳定解释应为：

- `True`：相关 helper 可走 JIT 快路径
- `False`：必须存在功能等价的 pure NumPy / Python fallback

上层代码不应把 `NUMBA_AVAILABLE=False` 当作“不支持此功能”。

### 8.2 对外导出的 dense helper

当前 `scptensor.core` 顶层导出的 JIT helper 包括：

- `count_mask_codes(M)`
- `find_missing_indices(M, mask_codes)`
- `apply_mask_threshold(X, threshold, comparison)`
- `fill_missing_with_value(X, M, fill_value, fill_mask_code)`

稳定行为：

- 输入面向 dense `np.ndarray`
- `count_mask_codes()` 返回长度 7 的计数数组
- `find_missing_indices()` 返回 `(rows, cols)`
- `apply_mask_threshold()` 返回布尔矩阵
- `fill_missing_with_value()` 原地修改 `X`，不返回新数组

### 8.3 fallback 不是次级语义

`jit_ops.py` 当前对 JIT / fallback 两套实现都做了同名函数提供。

因此后续优化必须保持：

- JIT 与 fallback 的参数、返回结构一致
- 不能让 “有 numba” 与 “无 numba” 产生不同结果方向
- 不能让 fallback 缺失导致顶层导入失败

### 8.4 内部稀疏 JIT kernel

`_sparse_row_sum_jit()` / `_sparse_row_mean_jit()` 虽是内部 helper，但已被测试显式引用，因此其可观测语义当前也被冻结：

- 输入 `(indptr, data, n_rows)`
- 输出 shape 为 `(n_rows,)`
- `sum` 仅统计显式存储值
- `mean` 对空行返回 `0.0`

## 9. 当前已知实现不对称点

以下问题在当前仓库中真实存在，文档必须如实记录，避免后续重构时误以为这些地方已经一致化：

### 9.1 `jit_ops.__all__` 与定义体不完全对齐

`jit_ops.__all__` 当前包含：

- `vectorized_mannwhitney_row`

但源码里该函数只在 `NUMBA_AVAILABLE=False` 的 fallback 分支定义；当 `NUMBA_AVAILABLE=True` 时，它不会在同名位置出现。

这意味着当前 `jit_ops` 存在“声明为可用，但实现只在部分导入路径存在”的不对称点。后续修复可以做，但在修复前不能把它当成稳定可依赖的对称 API。

### 9.2 `sparse_center_rows()` 的名称与真实返回不一致

函数文档与命名都指向“中心化”，但当前真实返回是：

- 原始 `CSR` 矩阵
- 行均值数组

它并不返回数值上已减去均值的新矩阵。

### 9.3 `MatrixOps` mask query 的类型注解偏窄

`get_valid_mask()`、`get_missing_mask()`、`get_missing_type_mask()` 当前类型注解写成 `np.ndarray`，但在稀疏 `M` 路径下可返回稀疏布尔矩阵。

因此当前稳定合同应以“形状与布尔语义正确”为准，而不能依赖其必定是 dense ndarray。

### 9.4 JIT 与 fallback 在极端 NaN 场景仍有差异风险

例如：

- JIT 版 `mean_no_nan()` / `var_no_nan()` 在全 NaN 输入下偏向返回 `0.0`
- fallback 版分别调用 `np.nanmean()` / `np.nanvar()`，全 NaN 时会产生 `NaN`

也就是说，仓库当前已具备“主路径语义大体一致，但极端边界仍未完全收口”的事实。后续若修复，应通过新增 golden tests 显式收敛，而不是靠优化顺手改变。

## 10. 优化时不得破坏的高优先级约束

后续如果要重写 `core` 低层计算实现，必须保留以下高优先级不变量：

1. 稀疏 `M` 的隐式 `VALID` 解释
2. `MatrixOps` 的非原地返回语义
3. `filter_by_mask()` 在 “不保留 `VALID`” 时的正确过滤行为
4. `apply_mask_to_values()` 对稀疏结构零的保护
5. `sparse_row_operation(np.mean)` 的“显式非零均值”语义
6. `fill_missing_with_value()` 的原地修改合同
7. JIT 与 fallback 的数值等价方向

## 11. 对后续重构的直接指导

基于当前实现，后续 `core` 优化应优先朝以下方向推进，而不是改动对外语义：

- 用更低开销的 sparse write path 替换 `LIL -> CSR`，但保持返回语义不变
- 为 `filter_by_mask()` 的 densify 路径加更明确的测试和注释，而不是删除该路径
- 把“显式非零 reduction”和“dense 语义 reduction”区分为两个明确函数族，而不是偷偷改变现有函数
- 将 JIT kernel 与 fallback 的测试对齐到统一 golden cases
- 明确 `sparse_center_rows()` 命名或后续拆分，但在改名之前不要改变它的返回结构

这意味着：`core` 下一阶段最需要的是“语义收口与测试固化”，不是随意替换数值定义。
