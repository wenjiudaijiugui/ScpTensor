# ScpTensor Utilities 冻结实现合同（`scptensor.utils`，2026-03-17）

## 1. 文档目标

本文档冻结 `scptensor.utils` 当前实现边界，服务于后续：

- helper 包公共面收口
- 测试与文档对齐
- 避免把 utility helper 误写成主线 preprocessing API
- 为后续拆分或重构保留可验证边界

它回答的是：

- `scptensor.utils` 当前哪些内容是 public package surface；
- 哪些 helper 只是通用工具，哪些已经是用户可直接依赖的稳定入口；
- batch / stats / transform / synthetic generator 各自遵守什么合同；
- 当前实现有哪些明显不对称点，后续不能悄悄改。

本文档基于以下仓库内事实：

- `scptensor/utils/__init__.py`
- `scptensor/utils/batch.py`
- `scptensor/utils/stats.py`
- `scptensor/utils/transform.py`
- `scptensor/utils/data_generator.py`
- `scptensor/__init__.py`
- `tests/utils/test_utils_batch.py`
- `tests/utils/test_utils_stats.py`
- `tests/utils/test_utils_transform.py`
- `tests/utils/test_utils_data_generator.py`
- `tests/core/test_utils_api.py`

## 2. 范围与非范围

### 2.1 范围

本文档覆盖：

- `scptensor.utils.__all__`
- `batch` utilities
- `stats` utilities
- `transform` utilities
- `ScpDataGenerator`

### 2.2 非范围

本文档不覆盖：

- normalization / transformation / qc / integration 等主线模块的算法合同
- `viz.recipes.statistics` 的 figure-level统计图合同
- benchmark 数据设计

## 3. 模块定位与是否需要单独冻结合同

结论先行：`utils` 需要单独冻结合同。

原因不是它“很核心”，而是它已经同时具备以下条件：

- 有完整的包级 `__all__`
- 有独立测试目录，覆盖面很大
- 已经暴露出多个用户可直接导入的 helper

因此 `utils` 已不再只是“仓库内部杂项函数桶”，而是一个需要边界说明的公共 helper 包。

同时，`utils` 的正确定位必须冻结为：

- stable helper bundle
- 为测试、实验、辅助统计、数组级操作服务
- 不是核心 preprocessing 主线模块

也就是说，`utils` 可以稳定存在，但它不应替代：

- `scptensor.normalization`
- `scptensor.transformation`
- `scptensor.qc`
- `scptensor.integration`

这些有单独阶段语义的模块。

## 4. 当前公开面

### 4.1 `scptensor.utils` 包级公开面

`scptensor.utils.__all__` 当前导出：

- `ScpDataGenerator`
- `correlation_matrix`
- `partial_correlation`
- `spearman_correlation`
- `cosine_similarity`
- `quantile_normalize`
- `robust_scale`
- `batch_iterator`
- `apply_by_batch`
- `batch_apply_along_axis`
- `BatchProcessor`

### 4.2 顶层 `scptensor` 的重导出边界

`scptensor.__all__` 当前不再从 `utils` 顶层重导出：

- `ScpDataGenerator`

不会重导出：

- `quantile_normalize`
- `robust_scale`
- `correlation_matrix`
- `BatchProcessor`
- 其他 utils helper

因此当前稳定边界是：

- `ScpDataGenerator` 是 `scptensor.utils` 包级 public utility
- 其他 utility helper 是 `scptensor.utils` 包级 public surface，但不是顶层 `scptensor` 主入口

## 5. Batch Utilities 合同

### 5.1 `batch_iterator`

`batch_iterator` 当前支持输入：

- `np.ndarray`
- `scipy.sparse` matrix
- 具备 `__len__` / `__getitem__` 的 sequence

稳定行为：

- `batch_size <= 0` -> `ValueError`
- 可沿 `axis=0/1` 分 batch
- 对 generic sequence 仅支持 `axis=0`
- `shuffle=True` 时可由 `random_seed` 保证可复现顺序
- 空数组输入会产生 `0` 个 batch

### 5.2 `apply_by_batch`

当前稳定行为：

- 逐 batch 执行 `func(batch, **kwargs)`
- `concat=False` 时返回原始结果列表
- `concat=True` 且结果为空时返回 `None`

拼接策略当前按首个结果类型决定：

- `np.ndarray` -> `np.concatenate(results, axis=axis)`
- sparse -> `sp.vstack(results)`
- `list` -> 扁平化成单个 list
- 其他类型 -> 保留结果列表

因此 `apply_by_batch` 当前是“结果类型驱动的轻量拼接器”，而不是严格的 axis-aware typed batch engine。

### 5.3 `batch_apply_along_axis`

当前稳定语义：

- 面向数组级 helper，而不是 `ScpContainer`
- sparse 输入会先 densify
- 输出总是 `np.ndarray`
- 支持 `dtype=` 强制转换

它更接近：

- batched `numpy.apply_along_axis`

而不是：

- 稀疏保真 / provenance-aware 计算接口

### 5.4 `BatchProcessor`

`BatchProcessor` 当前在 `apply_by_batch` 之外又提供了一套有状态接口。

稳定行为：

- `process()` 会累计：
  - `total_batches`
  - `total_samples`
  - `_history`
- `reset_stats()` 会清空统计与历史
- `get_stats()` 返回批次数与样本数
- 空结果时返回 `None`

需要特别冻结的细节：

- 对 `np.ndarray` 结果做 `np.concatenate`
- 对 sparse 结果做 `sp.vstack`
- 对 list 结果不扁平化，而是保留“batch 列表”

这与 `apply_by_batch()` 的 list flatten 语义不同，属于已形成的可观测不对称点。

## 6. Statistical Utilities 合同

### 6.1 `_ensure_dense`

这是内部 helper，但测试已覆盖其行为：

- dense 输入原样返回
- sparse 输入 `.toarray()`

### 6.2 `correlation_matrix`

当前稳定语义：

- 输入 shape 解释为 `(n_samples, n_features)`
- 输出是 feature-feature correlation matrix
- 支持 `method="pearson" | "spearman"`
- 稀疏输入先 densify
- 输入含 non-finite 值 -> `ValueError`
- 输出 `float64`
- 对角线强制设为 `1.0`
- 数值结果会 `clip` 到 `[-1, 1]`

当前它是数组级统计 helper，不承担 `ScpContainer` 或分组语义。

### 6.3 `partial_correlation`

当前稳定行为：

- `i` / `j` 越界 -> `ValueError`
- conditioning set 非法 -> `ValueError`
- 输入含 non-finite 值 -> `ValueError`
- 样本数不足 2 -> `ValueError`
- 若协方差矩阵奇异，则加 `_REGULARIZATION_EPS` 后再求逆
- 若 `conditioning_set` 为空或 `None`，退化为简单 Pearson correlation

### 6.4 `spearman_correlation`

这是当前 `utils.stats` 里最需要明说返回类型边界的函数。

当前真实返回是：

- `Y is None` 时：
  - 一般返回 correlation matrix
  - 但对某些低维输入，底层 `scipy.stats.spearmanr` 可能返回标量
- `Y is not None` 时：
  - 返回单个相关系数 `float`
- 任一输入含 non-finite 值 -> `ValueError`

因此它的稳定合同应是：

- `NDArray[np.float64] | float`

而不是永远二维矩阵。

### 6.5 `cosine_similarity`

当前稳定语义：

- `Y is None` -> 行与行之间的 pairwise cosine similarity
- `Y is not None` -> `X` 与 `Y` 行之间的 similarity
- 稀疏/稠密都支持
- 会先把 `X` / `Y` 解释成 2D row-wise 矩阵
- 任一输入含 non-finite 值 -> `ValueError`
- `Y is not None` 且特征维度不一致 -> `ValueError`
- 零范数通过后处理归一化并 `nan_to_num` 收口到有限值

当前它返回的是数组级 pairwise similarity matrix，而不是单个 sklearn-like estimator 对象。

## 7. Transform Utilities 合同

### 7.1 helper 定位

`quantile_normalize` 与 `robust_scale` 当前是：

- array/matrix helper
- 不是 `ScpContainer` preprocessing stage API

因此文档必须避免把它们误写成：

- `scptensor.normalization` 的等价替代
- protein-level pipeline 默认入口
- 会自动定义 `raw / log / norm / imputed / zscore` 这套 canonical layer 名或 provenance 的容器级 stage

### 7.2 `quantile_normalize`

当前稳定行为：

- `axis` 只能是 `0` 或 `1`
- sparse 输入先 densify
- 返回 dense `np.ndarray`
- 输出 shape 与输入一致
- 当前实现委托给共享内部 rank-normalization kernel，而不是跨模块直接引用
  `scptensor.normalization` 的 stage-private helper

语义上：

- `axis=0` -> 列分布对齐
- `axis=1` -> 行分布对齐

### 7.3 `robust_scale`

当前稳定行为：

- 支持 `with_centering`
- 支持 `with_scaling`
- `axis=0/1`
- `scale == 0` 时替换为 `1.0`

返回类型当前存在一个必须冻结记录的分支：

- 若输入 sparse 且 `copy=True` -> 返回 `csr_matrix`
- 若输入 sparse 且 `copy=False` -> 返回 dense `np.ndarray`

因此它不是“无条件 sparse-preserving” helper。

## 8. `ScpDataGenerator` 合同

### 8.1 模块定位

`ScpDataGenerator` 是 `scptensor.utils` 包级 public utility。

因此它已经不是纯内部测试 helper，而是稳定用户辅助入口，但不再通过顶层 `scptensor` 平铺导出。

### 8.2 生成结果结构

`generate()` 当前返回：

- 一个 `ScpContainer`
- 含一个 assay
- 含一个 layer
- 带 `obs` / `var` / `history`

默认命名是：

- `assay_name="proteins"`
- `layer_name="raw"`

但调用方可自定义 assay / layer / ID 列名。

### 8.3 元数据合同

默认生成的元数据包括：

- `obs`
  - `sample_id`
  - `batch`
  - `group`
  - `efficiency`
- `var`
  - `protein_id`
  - `gene_name`
  - `mean_abundance`

若自定义 `sample_id_col` / `feature_id_col` 不等于默认列名，当前会额外再复制一列对应 ID。

### 8.4 missingness 语义

这是 `ScpDataGenerator` 最重要的合同点之一。

当前生成器的真实行为是：

- `X` 仍保留完整有限数值矩阵
- missingness 主要编码在 `M`
- 不会把缺失位置直接写成 `NaN`

因此：

- generator 产出的 “missing” 是 mask-driven synthetic missingness
- 下游若只看 `X` 的 `NaN`，可能看不到这些缺失状态

### 8.5 mask_kind 与稀疏选项

当前支持：

- `mask_kind="none"`
- `mask_kind="bool"`
- `mask_kind="int8"`
- `use_sparse_X`
- `use_sparse_M`

但需要注意：

- dense bool mask 进入 `ScpMatrix` 后，通常仍会被规范成 `int8`
- sparse `M` 会以 `csr_matrix` 写出

因此 `mask_kind="bool"` 更接近“布尔语义输入请求”，不等于最终对象中一定保持原生 `bool` dtype。

### 8.6 synthetic code 语义

当前生成器把缺失状态大致编码为：

- `0` -> valid
- `1` -> random missing
- `2` -> probabilistic dropout missing

但在全仓库通用 `MaskCode` 语义里：

- `1` 对应 `MBR`
- `2` 对应 `LOD`

因此生成器当前是在复用全局 mask code 数值来近似表达 synthetic MCAR / MNAR 机制，而不是精确复刻 vendor provenance。

### 8.7 provenance 合同

`generate()` 当前会追加：

- `action="generate_synthetic_data"`

并记录参数，包括：

- `n_samples`
- `n_features`
- `missing_rate`
- `lod_ratio`
- `n_batches`
- `n_groups`
- `assay_name`
- `layer_name`
- `mask_kind`
- `use_sparse_X`
- `use_sparse_M`

## 9. 当前已知实现不对称点

### 9.1 顶层导出与包级导出不对称

当前所有 utils helper 都需从 `scptensor.utils` 访问。

这意味着：

- `utils` 是 public package
- 但不是所有 helper 都是 top-level public API

### 9.2 `apply_by_batch()` 与 `BatchProcessor` 的结果拼接策略不一致

当前真实差异：

- `apply_by_batch()` 对 list 结果做 flatten
- `BatchProcessor.process()` 对 list 结果保留 batch list

两者不能被文档写成同一行为。

### 9.3 sparse 结果拼接按 `vstack` 固定处理

无论 `axis` 是多少，只要 batch 结果是 sparse，当前都走：

- `sp.vstack(results)`

因此对 `axis=1` 的 sparse batch 拼接，并没有被显式收口成严格 axis-aware 语义。当前稳定且有测试支撑的主路径应理解为：

- dense 数组更稳定
- sparse 返回主要针对 `axis=0`

### 9.4 generator 的参数范围文档与实际校验不完全一致

`data_generator.py` 中定义了多组 `_MIN_* / _MAX_*` 常量，但 `_validate_params()` 当前只真正校验：

- `missing_rate`
- `lod_ratio`

因此：

- `n_samples`
- `n_features`

的很多范围更像文档意图，而不是当前强约束实现。

### 9.5 helper 名称与主线模块语义有重叠

例如：

- `quantile_normalize`
- `robust_scale`
- `correlation_matrix`

这些名称与主线 normalization / statistics / viz 语义有重叠，但当前 `utils` 版本只是数组级 helper，不应被误当成主线容器级 pipeline API。

## 10. 优化时不得破坏的高优先级约束

后续若要重构 `utils`，必须优先保留：

1. `ScpDataGenerator` 作为 `scptensor.utils` 包级 public utility 的地位
2. `scptensor.utils.__all__` 当前 helper 集合的大体稳定性
3. batch utilities 的基本返回语义与 stats tracking
4. stats helpers 的数组级而非容器级定位
5. transform helpers 与主线 normalization 模块的边界
6. synthetic missingness 主要存于 `M` 而不是 `X=NaN` 的事实
7. `generate_synthetic_data` provenance 行为

## 11. 对后续重构的直接指导

基于当前仓库状态，`utils` 下一阶段最合理的完善方向是：

- 明确哪些 helper 继续作为 package-level public API，哪些应降为 internal helper
- 为 `apply_by_batch` / `BatchProcessor` 收敛一套更一致的 list 与 sparse 拼接策略
- 把 `ScpDataGenerator` 的 synthetic mask semantics 写得更明确，避免和真实 vendor provenance 混淆
- 明确 `quantile_normalize` / `robust_scale` 是否需要继续保留为独立 helper，还是转为更明确的 internal utility

这意味着：`utils` 不仅需要单独冻结合同，而且该合同必须强调它是“稳定 helper bundle”，不是“主线 preprocessing stage”的兜底替代层。
