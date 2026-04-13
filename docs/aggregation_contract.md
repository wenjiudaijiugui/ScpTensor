# ScpTensor 聚合冻结实现合同（`scptensor.aggregation`，2026-03-17）

## 1. 文档目标

本文档冻结 `scptensor.aggregation` 当前实现合同，服务于后续代码优化、API 收敛、测试补强与 benchmark 对齐。

它回答的是：

- 当前 peptide / precursor -> protein 聚合到底已经实现了哪些方法；
- 输入 assay / layer、protein mapping、unmapped 处理、mask/link/provenance 目前遵守什么语义；
- `scptensor.aggregation.aggregate_to_protein` 与 `scptensor.io.aggregate_to_protein` 之间的对外边界是什么；
- 哪些当前行为即使不完美，也不能在重构时被悄悄改掉。

本文档基于以下仓库内事实：

- `scptensor/aggregation/__init__.py`
- `scptensor/aggregation/protein.py`
- `scptensor/__init__.py`
- `scptensor/io/__init__.py`
- `scptensor/io/__init__.py`
- `scptensor/io/api.py`
- `tests/core/test_aggregation_api.py`
- `tests/aggregation/test_protein_aggregation.py`
- `docs/core_data_contract.md`
- `docs/aggregation_literature.md`
- `docs/internal/review_aggregation_benchmark_20260312.md`
- `benchmark/aggregation/README.md`

## 2. 范围与非范围

### 2.1 范围

本文档覆盖：

- `scptensor.aggregation.aggregate_to_protein`
- `scptensor.aggregation.resolve_protein_mapping_column`
- `scptensor.io.aggregate_to_protein` 的稳定重导出入口
- `AggregationLink` / `ProvenanceLog` 在聚合阶段的当前写法

### 2.2 非范围

本文档不覆盖：

- DIA-NN / Spectronaut 导入解析
- normalization / log transform / imputation / integration / qc
- literature ranking 与 benchmark 胜负结论
- downstream DE / clustering

## 3. 稳定主线边界

按项目总合同，`scptensor.aggregation` 是唯一专门负责 **peptide/precursor -> protein** 转换的稳定阶段。

因此必须明确：

- 稳定输入主线是 `peptides` assay 上的定量 layer
- 稳定输出主线是 `proteins` assay 上的 protein-level quantitative matrix
- stable downstream preprocessing 默认在 protein 层继续进行

也就是说：

- `aggregation` 是 stable preprocessing 主线里唯一合法的 feature-universe 变更阶段
- 其余模块不应再承担 `peptides -> proteins` 转换职责

## 4. 当前公开 API

### 4.1 模块级 API

`scptensor.aggregation.__all__` 当前导出：

- `AggMethod`
- `BasicAggMethod`
- `aggregate_to_protein`
- `resolve_protein_mapping_column`

### 4.2 I/O 层稳定重导出入口

`scptensor.io.__all__` 也导出：

- `aggregate_to_protein`

`scptensor.__all__` 当前只从 aggregation 顶层重导出：

- `aggregate_to_protein`

这是对 `scptensor.aggregation.aggregate_to_protein` 的稳定重导出入口。

### 4.3 两个入口的当前关系

模块级真实实现签名包含：

- `unmapped_label`

而 `scptensor.io.aggregate_to_protein` 当前与 aggregation 入口保持同一函数签名。

因此稳定解释应为：

1. `scptensor.aggregation.aggregate_to_protein` 与 `scptensor.io.aggregate_to_protein` 当前是同一个函数对象。
2. 对外 canonical I/O 入口仍可以写 `scptensor.io.aggregate_to_protein`
3. 顶层 `scptensor` 当前不重导出：
   - `AggMethod`
   - `BasicAggMethod`
   - `resolve_protein_mapping_column`

## 5. 当前已实现的方法池

### 5.1 支持的方法全集

当前 `aggregate_to_protein()` 只接受以下方法名：

- `sum`
- `mean`
- `median`
- `max`
- `weighted_mean`
- `top_n`
- `maxlfq`
- `tmp`
- `ibaq`

若传入其他方法名，会抛 `ValidationError("Unsupported aggregation method: ...")`。

### 5.2 方法家族与当前语义

| 方法 | 当前语义 | 结果尺度 | 当前额外参数 |
|---|---|---|---|
| `sum` | 按 protein 内 peptide 强度求和 | linear-like input scale | 无 |
| `mean` | 按 protein 内 peptide 强度求均值 | linear-like input scale | 无 |
| `median` | 按 protein 内 peptide 强度求中位数 | linear-like input scale | 无 |
| `max` | 取 protein 内最大 peptide 强度 | linear-like input scale | 无 |
| `weighted_mean` | 用 peptide 跨样本中位强度作权重的行均值 | linear-like input scale | 无 |
| `top_n` | 先按 peptide 全局丰度排名选前 N 条，再做 basic aggregation | linear-like input scale | `top_n`, `top_n_aggregate` |
| `maxlfq` | pairwise median log-ratio + least squares 的工程化近似 | 输出回到 linear scale | `lfq_min_ratio_count` |
| `tmp` | Tukey median polish 风格 summarization | 输出回到 linear scale | `tmp_log_base` |
| `ibaq` | peptide 强度和 / denominator | linear-like input scale | `ibaq_denominator` |

这里要强调：

- `maxlfq` 是工程近似实现，不是 vendor-level 全量 MaxLFQ 复刻
- `tmp` 会在内部进入 log 空间，但输出会反变换回 linear scale
- `weighted_mean` 当前权重不是外部传入，而是该 protein 内每条 peptide 的跨样本 `nanmedian`

## 6. 输入 assay / layer 合同

### 6.1 source assay 当前不做 alias 解析

这是当前聚合模块最容易被误解的一点：

- `aggregate_to_protein()` 直接用 `if source_assay not in container.assays`
- **不会** 调 `resolve_assay_name()`

因此当前真实行为是：

- 如果容器里 assay 名是 `peptides`，传 `source_assay="peptide"` 会失败
- 如果容器里 assay 名是 `proteins`，传 `target_assay="protein"` 只是写入一个新名字，不会自动解析成现有 assay

稳定合同必须按这个真实行为写，而不是按别的模块的 alias 语义类推。

### 6.2 默认输入与输出 assay

当前默认参数是：

- `source_assay="peptides"`
- `source_layer="raw"`
- `target_assay="proteins"`

这也是当前文档级 canonical 主线。

### 6.3 source layer 验证

当前 source layer 缺失时，聚合模块抛的是：

- `ValidationError`

不是 `LayerNotFoundError`。

### 6.4 target assay 当前不会做保护性检查

当前实现没有显式拦截：

- `target_assay` 已存在
- `source_assay == target_assay`

真实行为是：

1. 构造新的 protein assay
2. `new_assays[target_assay] = protein_assay`

因此：

- 若 `target_assay` 已存在，会被静默覆盖
- 若 `source_assay == target_assay`，聚合函数本身不会提前拦截，但返回新 `ScpContainer` 时会因 `AggregationLink` 校验失败而抛 `ValueError`

这属于当前实现事实，不应被文档掩盖。

## 7. protein mapping 合同

### 7.1 `protein_column="auto"` 的解析规则

当前 `resolve_protein_mapping_column()` 会按如下顺序寻找列：

- `PG.ProteinGroups`
- `PG.ProteinAccessions`
- `Protein.Group`
- `Protein.Ids`
- `EG.ProteinId`
- `FG.ProteinGroups`

若上面都没有：

1. 收集所有列名中包含 `"protein"` 的列
2. 若恰好一个，直接使用
3. 若多个，抛 `ValidationError`，要求显式传入 `protein_column`
4. 若没有，抛 `ValidationError("No protein mapping column found ...")`

### 7.2 null protein mapping 的处理

当前 protein mapping 会先 `cast(pl.Utf8, strict=False)`。

然后分两条路径：

- `keep_unmapped=True`
  - `null` 会被替换成 `unmapped_label`
  - 默认 label 是 `"NA"`
  - unmapped peptide 会被聚合成一个真实的 target protein group
- `keep_unmapped=False`
  - `null` mapping 对应的 peptide 会被直接移除
  - 若移除后一个都不剩，抛 `ValidationError`

### 7.3 unmapped label 的当前风险

当前实现不会检查：

- `unmapped_label` 是否与真实 protein ID 冲突

因此若真实 protein ID 也叫 `"NA"`，当前会与 unmapped group 合并。这是当前实现缺口，文档必须如实反映。

## 8. 数值聚合合同

### 8.1 输入矩阵的当前预处理

聚合前当前总是：

- `X` 转为 dense `float64`
- `M` 转为 dense `int8`

因此当前 aggregation 不是 sparse-preserving pipeline。

对应地，输出也当前总是：

- `X`: dense `float64`
- `M`: dense `int8`

### 8.2 basic methods 的 `NaN` 语义

- `sum`
  - 用 `np.nansum`
  - 但若整行全缺失，则显式保留为 `NaN`
- `mean / median / max`
  - 都是 row-wise `nan`-aware reducer
  - 全缺失行保留为 `NaN`

### 8.3 `weighted_mean`

当前实现里：

- 先对该 protein 的每条 peptide 算跨样本 `nanmedian`
- 只保留 `finite 且 >0` 的权重
- 每个 sample 上只对同时满足“该 sample 值有限且权重有效”的 peptide 计算加权均值

这意味着：

- 权重是 protein 内局部权重，不是全局学习到的权重
- 权重不随 sample 变化

### 8.4 `top_n`

当前 `top_n` 的 peptide 选择不是 per-sample，而是：

- 先对每条 peptide 计算跨样本 `nanmedian`
- 再按这个全局 abundance score 排序

并且：

- `top_n <= 0` 时，当前实现实际上等于“使用全部 peptide”
- `top_n >= n_peptides` 也等于“使用全部 peptide”

虽然 API 只验证了 `top_n >= 0`，但 `top_n=0` 当前并不会报错，而是退化为 all-peptide aggregation。

### 8.5 `maxlfq`

当前 `maxlfq` 的真实行为是：

- 只对 `values > 0` 的观测进入 log-ratio 网络
- 基于 sample-pair 的 peptide log-ratio 中位数建边
- 边数不足时退回到 `exp(row median(log_vals))`
- 最终输出回到 linear scale

`lfq_min_ratio_count` 当前必须 `>= 1`。

### 8.6 `tmp`

当前 `tmp` 的真实行为是：

- 只对 `values > 0` 的观测进入 `log(values) / log(tmp_log_base)`
- 做 median polish 风格迭代
- 最终通过 `np.power(log_base, ...)` 回到 linear scale

因此 `tmp` 当前不是“输出 log protein intensity”，而是“内部在 log 空间拟合，最终回到 linear protein intensity”。

`tmp_log_base` 当前必须 `> 1`。

### 8.7 `ibaq`

`ibaq` 当前的分母解析规则是：

- 若 `ibaq_denominator is None`
  - 分母默认取该 protein 当前映射 peptide 数
- 若提供字典
  - 每个 protein 必须存在条目
  - 且 denominator 必须 `> 0`

若缺失某个 protein 的 denominator，会抛 `ValidationError("Missing iBAQ denominator ...")`。

## 9. 输出 assay / var / layer 合同

### 9.1 目标 assay 的当前结构

聚合后会创建一个新的 `Assay`：

- `feature_id_col="_index"`
- `var` 当前只包含：
  - `_index`
  - 若 `protein_col != "_index"`，再额外复制一列同名 protein mapping 列
- `layers` 当前只包含：
  - `{source_layer: ScpMatrix(...)}`

也就是说：

- 目标 assay 不会自动生成 `raw/log/norm/...` 多层
- 目标 `var` 不会继承 peptide assay 的其它 feature metadata

### 9.2 protein 顺序合同

当前 protein 顺序来自：

- `np.unique(protein_ids)`

因此它是 **排序后的唯一值顺序**，不是“首次出现顺序”。

这是当前行为，不应假设 target proteins 保持 source peptide 的首次映射顺序。

### 9.3 输出 layer 名

输出 layer 名当前固定复用 `source_layer`。

默认就是：

- `peptides/raw -> proteins/raw`

因此 aggregation 当前不会顺手把 layer 重命名成别的默认值。

## 10. mask、metadata 与 link 合同

### 10.1 mask 聚合规则

当前 protein mask 的计算是：

```text
m_protein[:, j] = np.max(masks[:, used], axis=1)
```

这意味着：

- 采用当前 `MaskCode` 整数值的 `max` 作为 protein-level mask
- 不是 majority vote
- 不是 VALID-only
- 也不是 state-aware 优先级表

并且需要明确：

- 即使 source layer 的 `M is None`
- 当前也会先把它转成全 `0` 的 dense `int8` mask
- 因此输出 protein layer 的 `M` 当前 **永远不是 `None`**

对于不同方法：

- `top_n`：只对被选中的 peptide 做 mask 聚合
- 其他方法：对该 protein 当前全部 peptide 做 mask 聚合

### 10.2 metadata

当前输出 `ScpMatrix` 只写：

- `X`
- `M`

不会复制或生成 `metadata`。

并且当前输出 protein layer 的 `M` 是新的 dense `int8` 聚合结果，不共享 source layer 的 `M` 引用。

### 10.3 `AggregationLink`

当前一定会 append 一个新的 `AggregationLink`：

- `source_assay`
- `target_assay`
- `linkage`，列为 `source_id` / `target_id`

但需要明确一个非常具体的实现事实：

- `linkage` 记录的是 **该 protein 当前全部映射 peptide**
- 即使方法是 `top_n`，link 也不会只记录被选中的 top-N peptide

换句话说：

- link 表达的是 assay 间映射关系
- 不是本次聚合计算时真正参与数值汇总的 used-feature 子集

### 10.4 缺失 source feature ID

当前 source feature ID 若为 null，会被替换成：

- `"__MISSING_SOURCE_ID__"`

然后写入 linkage。

## 11. 返回对象与复制语义合同

### 11.1 返回的是新 `ScpContainer`

当前 `aggregate_to_protein()` 不是原位修改，而是构造并返回一个新的 `ScpContainer`。

### 11.2 但这不是深拷贝全容器

当前返回容器的构造方式是：

- `obs=container.obs.clone()`
- `new_assays = dict(container.assays)`
- `new_assays[target_assay] = protein_assay`

这意味着：

- `obs` 是 clone
- 新 target assay 是新对象
- 但原有 assays 只是字典浅拷贝后复用引用

因此当前合同不能写成“aggregation 返回完全独立深拷贝容器”。

## 12. provenance 合同

当前 aggregation 不通过 `container.log_operation()` 记录历史，而是手工 append 一个 `ProvenanceLog`。

这条 history 的当前稳定字段包括：

- `action="aggregate_to_protein"`
- `source_assay`
- `source_layer`
- `target_assay`
- `method`
- `protein_column`
- `keep_unmapped`
- `top_n`
- `top_n_aggregate`
- `lfq_min_ratio_count`
- `tmp_log_base`
- `has_ibaq_denominator`

当前没有记录：

- `unmapped_label`
- 实际参与计算的 selected peptide 集
- 每个 protein 的 denominator 明细

这些都不应被误写成“当前已有 provenance”。

## 13. failure contract

### 13.1 assay / layer

- 缺失 `source_assay`：`AssayNotFoundError`
- 缺失 `source_layer`：`ValidationError`

### 13.2 protein mapping

- 显式 `protein_column` 不存在：`ValidationError`
- 自动解析找不到 mapping 列：`ValidationError`
- 自动解析发现多个 protein-like 候选列：`ValidationError`
- `keep_unmapped=False` 且过滤后没有 mapped peptide：`ValidationError`
- 最终没有任何 protein group：`ValidationError`

### 13.3 参数错误

- 非法 `method`：`ValidationError`
- `top_n < 0`：`ValidationError`
- `lfq_min_ratio_count < 1`：`ValidationError`
- `tmp_log_base <= 1`：`ValidationError`
- `ibaq` 分母缺失或非正：`ValidationError`

## 14. 优化安全不变量

后续若对 `scptensor.aggregation` 做性能优化、算法替换或 API 收敛，以下行为不能被悄悄改掉：

1. `aggregation` 仍必须是 stable 主线中唯一的 peptide/precursor -> protein 转换阶段。
2. 当前公开方法池及其方法名不能无声漂移。
3. `source_assay` 当前不做 alias 解析，这一行为若要改变，必须显式更新合同与调用文档。
4. 输出 layer 默认复用 `source_layer` 名称，这一行为不能被悄悄重命名。
5. `AggregationLink` 当前记录的是全部映射关系，不是 used-feature 子集；若未来想记录 used subset，必须作为新合同扩展。
6. 返回值是新 `ScpContainer`，但不是深拷贝全部 assays；若未来改成全量深拷贝，需要显式说明兼容性影响。
7. target assay 当前会静默覆盖已有同名 assay；若未来加保护机制，也必须作为显式接口变更处理。
