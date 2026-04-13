# ScpTensor 标准化冻结实现合同（`scptensor.standardization`，2026-03-17）

## 1. 文档目标

本文档冻结 `scptensor.standardization` 当前实现合同，服务于后续：

- `zscore` API 收口
- 文档与教程边界统一
- 下游表示层与核心 quantitative layer 的职责分离
- 后续代码优化与测试补强

它回答的是：

- 当前 `scptensor.standardization` 真实公开面是什么；
- `zscore` 在项目中的正确定位是什么；
- 哪些输入、输出、报错、写回行为已经形成稳定合同；
- 哪些当前实现不对称点必须被文档显式记录，不能被误写成“已统一”。

本文档基于以下仓库内事实：

- `scptensor/standardization/__init__.py`
- `scptensor/standardization/zscore.py`
- `scptensor/__init__.py`
- `tests/standardization/test_zscore.py`
- `tests/core/test_standardization_api.py`
- `docs/core_data_contract.md`

## 2. 范围与非范围

### 2.1 范围

本文档覆盖：

- `scptensor.standardization.zscore`
- `zscore` 的包级导出边界
- `zscore` 的输入层、输出层、写回方式、异常类型与 provenance 语义

### 2.2 非范围

本文档不覆盖：

- log transform / normalization / imputation / integration / qc
- z-score 在外部文献中的方法学综述本体
- heatmap / embedding / clustering 的可视化实现细节

这些内容分别由其他合同或综述文档负责。

## 3. 当前公开 API

### 3.1 模块级公开面

`scptensor.standardization.__all__` 当前只导出一个函数：

- `zscore`

### 3.2 顶层包重导出

`scptensor.__all__` 当前不再重导出 `zscore`。

这意味着 `zscore` 的稳定导入边界是 `scptensor.standardization`，而不是顶层 convenience import。

## 4. 模块定位与稳定边界

结合项目主合同、当前 benchmark/reporting 约束与现有 API 行为，`zscore`
的当前正确定位应冻结为：

- stable package area 中的表示层 helper
- downstream exploratory representation transform
- 不是核心 quantitative normalization 主线
- 不是最终 protein-level quantitative deliverable

因此后续文档、教程和代码示例必须避免把 `zscore` 写成：

- vendor normalization 的替代品
- 默认 preprocessing 主线
- protein-level final matrix 的标准输出

更准确的写法应是：

- `zscore` 生成下游表示层
- 适合 heatmap、表示空间对照实验、部分 clustering / embedding 辅助场景
- 不应默认接管稳定 quantitative pipeline

因此与 `zscore` 绑定的报告语义也应冻结为：

- 图表或指标若在 `zscore` 空间上计算，应显式写成 `@zscore`
- `zscore` 不默认承担 completeness 统计、integration 主榜评分或对外 quantitative export
- 若需要 quantitative-comparison 专用 z-transform，应单列为新的方法家族，而不是复用当前 helper 语义

## 5. 当前输入合同

### 5.1 当前默认参数

`zscore()` 当前默认参数为：

- `assay_name="proteins"`
- `source_layer="imputed"`
- `new_layer_name="zscore"`
- `axis=0`
- `ddof=1`

### 5.2 assay alias 当前已对齐仓库 canonical naming

当前实现已通过 `resolve_assay_name()` 支持：

- `protein <-> proteins`
- `peptide <-> peptides`

因此当前真实行为是：

- 默认 `assay_name="proteins"` 已与仓库主线文档命名对齐
- 若容器实际 assay 名是 `protein`，默认参数与显式 `assay_name="proteins"` 仍可通过 alias 正确解析
- `source_layer` 也必须与当前对象中的 layer 名完全一致

这意味着 `zscore` 的 assay 入口已不再是单独一套历史命名分支。

### 5.3 `axis` 与 `ddof` 验证

当前实现显式验证：

- `axis` 必须是 `0` 或 `1`
- `ddof` 必须非负
- `ddof < axis_len`
  - 即对应标准化轴上至少要有 `ddof + 1` 个值

因此 `ddof` 的合同现在已经不是“仅做非负检查”，而是包含基本统计定义域校验。

### 5.4 当前没有 logged/unlogged gate

当前实现没有任何 logged/unlogged 场景检测，也没有 source layer scale gate。

也就是说，当前代码不会像 `log_transform()` 那样做：

- layer-name heuristic
- history-based logged detection
- distribution heuristic
- raw/log/norm/imputed 场景分流

因此当前真实行为是：

- 只要 `assay_name` / `source_layer` 精确命中
- 且 `X` 中没有 `NaN/Inf`

`zscore` 就会继续执行，不会因为 source layer 名是：

- `raw`
- `log`
- `norm`
- `imputed`

而自动改变行为或拒绝运行。

这意味着 logged/unlogged 的选择责任当前仍在调用方，而不在 `zscore` 内部。

### 5.5 “complete matrix” 的当前真实定义

当前实现用下面的条件拒绝输入：

- `np.isnan(input_layer.X).any()`

也就是说，当前代码检查的是：

- `X` 中是否存在 `NaN`

而不是：

- `M` 中是否仍有非 `VALID` mask code

因此当前真实行为是：

- 如果缺失只编码在 `M` 中，但 `X` 本身仍是有限数值，`zscore` 仍会继续运行
- 当前“complete matrix”更接近 “`X` 中无 `NaN/Inf`”，而不是 “mask 上无缺失状态”

这是一条必须冻结记录的实现事实。

### 5.6 稀疏输入的当前边界

当前 `zscore` 直接对 `input_layer.X` 使用：

- `np.isnan`
- `np.mean`
- `np.std`

但仓库中没有针对 sparse `X` 的专门测试。

因此当前稳定合同应写成：

- 已测试且可依赖的主路径是 dense complete matrix
- sparse 输入暂不应被当作已经冻结的稳定行为

## 6. 当前变换与写回合同

### 6.1 计算语义

当前 `zscore` 直接计算：

- `mean = np.mean(X, axis=axis, keepdims=True)`
- `std = np.std(X, axis=axis, keepdims=True, ddof=ddof)`
- `std[std == 0] = 1.0`
- `x_z = (X - mean) / std`

因此稳定数值语义是：

- 零标准差位置不会报错，而是按 `std = 1.0` 处理
- 输出 shape 与输入 `X` 相同

### 6.2 mask 写回

输出 layer 当前写成：

- `ScpMatrix(X=x_z, M=input_layer.M.copy() if input_layer.M is not None else None)`

因此稳定行为是：

- `zscore` 不重算 mask
- 只复制源层 `M`
- source layer 与 new layer 的 mask 语义保持一致

### 6.3 写回位置与返回值

当前实现通过 `assay.add_layer(layer_name, ScpMatrix(...))` 把新 layer 直接写回原对象，并返回同一个 `container`。

因此当前稳定合同是：

- `zscore` 是原对象写回式 API
- 返回值与输入 `container` 是同一逻辑对象，而不是 copy-on-write 新容器
- 默认使用路径会写入新的 `zscore` layer
- 但若 `new_layer_name` 与现有 layer 同名，包含 `new_layer_name == source_layer`，当前仍遵循 `Assay.add_layer()` 的静默覆盖语义

### 6.4 provenance 合同

当前 `zscore` 会记录：

- `action="standardization_zscore"`
- `assay`
- `source_layer`
- `new_layer_name`
- `axis`
- `ddof`

因此后续优化不能删除这条 provenance 记录，也不能悄悄改成不带参数的模糊日志。

## 7. 当前失败合同

### 7.1 已显式抛错路径

当前代码显式抛出：

- `ScpValueError`
  - `axis` 非 `0/1`
  - `ddof < 0`
- `AssayNotFoundError`
  - `assay_name` 不存在
  - 附带 available assay hint
- `LayerNotFoundError`
  - `source_layer` 不存在
  - 附带 available layers hint
- `ValidationError`
  - `X` 中存在 `NaN`
  - `X` 中存在 `Inf`
  - `ddof` 对当前轴长度非法

### 7.2 未显式收口但可能暴露的边界

以下边界当前仍没有单独合同化处理：

- sparse `X` 路径上的更细粒度性能/格式语义
- 输出 layer 名冲突

因此这些情形当前不能被写成“已显式校验”。

## 8. 当前已知实现不对称点

### 8.1 默认 assay 名已收口到仓库 canonical naming

当前默认已收敛为：

- `assay_name="proteins"`

同时保留 singular/plural alias 解析，因此：

- canonical 文档命名与默认参数一致
- 历史容器若仍使用 `protein`，也不会因为默认值变化而失配

### 8.2 “complete matrix” 只按 `X` 检查，不按 `M` 检查

这是当前最重要的不对称点之一：

- `NaN` 会被拒绝
- 非 `VALID` mask code 本身不会阻止 z-score

因此若源层把缺失状态编码在 `M` 而不是 `X=NaN`，当前 `zscore` 可能仍然执行。

### 8.3 稀疏路径未冻结

虽然 `ScpMatrix` 支持 sparse `X`，但 `zscore` 没有 sparse 专项测试，因此不能把它写成“已稳定支持 sparse standardization”。

### 8.4 现有 layer 名冲突时会静默覆盖

`Assay.add_layer()` 当前本质上是：

- `self.layers[name] = matrix`

没有“layer 已存在”的保护性检查。

因此若 `new_layer_name` 与现有 layer 同名，当前行为是静默覆盖。

## 9. 优化时不得破坏的高优先级约束

后续若要重构 `standardization`，必须优先保留：

1. `zscore` 作为唯一 stable public standardization API 的地位
2. 表示层 helper、而非主线 normalization 的模块定位
3. 默认路径仍是写入新的 `zscore` layer；但若 `new_layer_name` 与现有 layer 同名，当前仍遵循静默覆盖语义
4. mask 复制而不是重算的当前语义
5. provenance `standardization_zscore` 记录
6. `axis` / `ddof` / assay / layer 的显式报错路径
7. 当前没有 logged/unlogged 自动检测或 raw-layer gate 的事实，不能无声改变

## 10. 对后续重构的直接指导

基于当前实现，下一阶段最合理的完善方向是：

- 把 “complete matrix” 从仅检查 `X` 的 `NaN` 扩展为文档和代码一致的状态检查
- 决定 sparse `X` 是否要作为正式支持路径，并用测试锁定
- 给 `new_layer_name` 冲突增加显式策略，而不是继续依赖静默覆盖

这意味着：`standardization` 已经需要单独冻结合同，而且该合同必须强调它的表示层边界，不能再被混写到 normalization 主线里。
