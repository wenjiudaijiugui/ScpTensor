# ScpTensor QC 实现合同（2026-03-16）

## 1. 文档目标

本文档定义 `scptensor.qc` 的冻结实现合同，服务于后续性能优化、接口收敛、测试补强与文档统一。它回答的不是“QC 算法还可以怎么做”，而是“当前稳定 QC 语义里，哪些行为不能被无声改掉”。

本文档以三类事实为基础：

- 当前稳定代码：`scptensor/qc/*.py`
- 当前测试锚点：`tests/qc/test_qc.py`
- 已完成的综述约束：
  - `docs/internal/review_qc_filtering_20260312.md`
  - `docs/internal/review_batch_correction_20260305.md`
  - `docs/core_data_contract.md`

本文档只覆盖 `QC`。不覆盖 normalization、imputation、integration、AutoSelect 的实现合同。

## 2. 范围与边界

### 2.1 稳定范围

`scptensor.qc` 的稳定预处理合同只面向 **protein-level** 主线终点。

- 若输入来自 peptide / precursor 层，稳定路径必须先经 `scptensor.aggregation` 聚合到 protein level，再进入本文档定义的 stable QC。
- 样本轴语义沿用 `ScpContainer`：`X.shape == (n_samples, n_features)`。
- feature 轴语义沿用 `Assay.var`。

### 2.2 文档命名与兼容别名

仓库文档的 canonical assay 命名是：

- `proteins`
- `peptides`

但当前 `qc` 代码通过 `resolve_assay_name()` 兼容以下别名组：

- `protein <-> proteins`
- `peptide <-> peptides`

因此本合同冻结如下规则：

1. 仓库级实现说明与接口说明优先写 `proteins`。
2. 当前稳定 QC 接口必须继续接受 `protein` / `proteins` 两种写法，直到 alias 合同被显式更新。
3. stable downstream preprocessing 的默认目标仍然是 `proteins` assay，而不是 peptide / PSM 级对象。

### 2.3 明确不纳入 stable QC 合同的内容

以下内容不属于本文档的 stable contract：

- `scptensor.qc.qc_psm`
- PSM / peptide / precursor 级过滤规则
- embedding-level batch diagnostics
- batch correction 本身
- doublet detection 的社区金标准声明

`qc_psm` 当前保留在源码中，但 `scptensor.qc.__init__` 已明确其不属于 stable preprocessing contract；它应被视为 experimental / pre-aggregation helper，而不是 protein-level stable QC 主线的一部分。其独立边界见 `docs/qc_psm_contract.md`。

## 3. 稳定入口与输入层选择合同

### 3.1 稳定入口

`scptensor.qc.__all__` 当前冻结的 package-level stable surface 是：

- `qc_sample`
- `qc_feature`
- `calculate_sample_qc_metrics`
- `filter_low_quality_samples`
- `filter_doublets_mad`
- `assess_batch_effects`
- `calculate_feature_qc_metrics`
- `filter_features_by_missingness`
- `filter_features_by_cv`

这些名称构成当前 canonical package-level 入口。

对应的稳定功能入口如下：

- `calculate_sample_qc_metrics`
- `filter_low_quality_samples`
- `filter_doublets_mad`
- `assess_batch_effects`
- `calculate_feature_qc_metrics`
- `filter_features_by_missingness`
- `filter_features_by_cv`

其底层统计工具包括：

- `compute_mad`
- `is_outlier_mad`
- `compute_cv`

但需要明确区分：

1. 这些统计工具当前属于 `scptensor.qc.metrics` 模块级 helper。
2. 它们不是 `scptensor.qc.__all__` 当前冻结的 package-level public surface。
3. `qc_psm` 同样不属于 `scptensor.qc.__all__` 的 stable package surface。

补充导出层级边界：

- `scptensor.__all__` 当前不再把 stable QC surface 重导出到根包
- `scptensor.qc.metrics` helper 与 `qc_psm` 也不会上浮到根包 `scptensor`

因此当前推荐导入层级是：

- stable QC package boundary：`scptensor.qc`
- 非 stable helper：仍需显式走模块路径，不应被误写成 root/package-level stable API

### 3.2 输入 assay / layer 选择语义

当前 stable 代码已经把 layer 选择统一为显式 `raw` 合同。该规则已被实现与测试固化，后续重构不能无声改写：

| 入口 | assay 解析 | layer 选择 |
|---|---|---|
| `calculate_sample_qc_metrics` | 先做 alias 解析 | 默认显式要求 `layer_name="raw"`；若 `raw` 不存在，直接报错，除非调用者显式传其他 layer |
| `filter_low_quality_samples` | 先做 alias 解析 | 默认显式要求 `layer_name="raw"`；若 `raw` 不存在，直接报错，除非调用者显式传其他 layer |
| `filter_doublets_mad` | 先做 alias 解析 | 默认显式要求 `layer_name="raw"`；若 `raw` 不存在，直接报错，除非调用者显式传其他 layer |
| `assess_batch_effects` | 先做 alias 解析 | 默认显式要求 `layer_name="raw"`；若 `raw` 不存在，直接报错，除非调用者显式传其他 layer |
| `calculate_feature_qc_metrics` | 先做 alias 解析 | 默认显式要求 `layer_name="raw"`；若 `raw` 不存在，直接报错，除非调用者显式传其他 layer |
| `filter_features_by_missingness` | 先做 alias 解析 | 默认显式要求 `layer_name="raw"`；若 `raw` 不存在，直接报错，除非调用者显式传其他 layer |
| `filter_features_by_cv` | 先做 alias 解析 | 默认显式要求 `layer_name="raw"`；若 `raw` 不存在，直接报错，除非调用者显式传其他 layer |

冻结含义：

1. 这是当前 stable 行为，不是临时实现细节。
2. QC 稳定入口不会根据 `assay.layers` 的插入顺序自动猜测 source layer。
3. 若未来要放宽为别的默认 layer 规则，应视为接口变更，需同步更新文档和测试。
4. QC 稳定入口不做隐式 aggregation、log transform、normalization 或 imputation。

### 3.3 空 assay 行为不是 QC stable contract

当前 QC 入口默认假设“被选中的 assay 至少包含一个 quantitative layer”。对空 `layers` assay 的行为并未定义为 stable contract，不应把对空 assay 的支持作为当前优化目标。

## 4. 检测语义与状态码合同

### 4.1 检测定义是 mask-first，而不是 value-first

当前 `qc` 的 detection semantics 由 `scptensor.qc._utils.get_detection_mask()` 冻结：

1. 若 `M` 存在，则 `M` 的状态码优先于 `X` 的数值。
2. 默认只有 `MaskCode.VALID` 计为“detected”。
3. 若 `M` 不存在：
   - sparse `X`：结构性非零计为 detected
   - dense `X`：有限值计为 detected

### 4.2 `MaskCode` 在 QC 检测语义中的当前冻结含义

对当前 QC 检测统计而言，以下状态默认 **不计入 detected**：

- `MBR`
- `LOD`
- `FILTERED`
- `OUTLIER`
- `IMPUTED`
- `UNCERTAIN`

只有：

- `VALID`

计入 detected。

这意味着：

1. “有效的数值零”只要 `M == VALID`，仍应被计为 detected。
2. “数值非空但状态不是 VALID”的条目，不应被 `n_features`、`detection_rate` 或 `missing_rate` 计入 detected。
3. 后续若要把 `IMPUTED` 或 `MBR` 计入某类 completeness，应在别的模块或新的显式参数下完成，不能无声改变当前 stable QC 语义。

### 4.3 当前数值统计与检测统计并不完全同语义

当前 stable 实现中，以下指标使用 detection semantics：

- `n_features`
- `missing_rate`
- `detection_rate`
- `mean_expression` 的有效分母
- `filter_features_by_missingness`

而以下指标使用 `X` 的数值矩阵本身，不按 `M` 重新加权：

- `total_intensity`
- `log1p_total_intensity`
- `cv`
- `filter_features_by_cv`

这不是文档笔误，而是当前代码事实。后续优化不能偷偷把 `cv` 改成 detected-only 或 mask-aware 版本；若要改，必须显式升级合同。

## 5. Sample QC 合同

### 5.1 `calculate_sample_qc_metrics`

职责：

- 计算 sample-level QC 摘要，不做过滤。

稳定输出：

- 写入 `container.obs`
- 追加以下列名，列名带 resolved assay name 后缀：
  - `n_features_{assay}`
  - `total_intensity_{assay}`
  - `log1p_total_intensity_{assay}`

当前冻结语义：

- `n_features`：按 `VALID` detection semantics 计数。
- `total_intensity`：按行对 `X` 求和。
  - dense：`np.nansum`
  - sparse：矩阵求和
- `log1p_total_intensity`：对 `total_intensity` 应用 `np.log1p`。

结构合同：

1. 返回新的 `ScpContainer`。
2. 输入容器不得被原地修改。
3. 样本顺序必须保持不变。
4. 不创建新 layer，不改变 assay 注册表。

当前冻结的不对称点：

- 该函数当前 **不写 history**。
- 后续若要加 provenance log，必须作为显式合同变更处理，不能把它混在“性能优化”里无声引入。

### 5.2 `filter_low_quality_samples`

职责：

- 基于 sample-level detected feature count 进行样本过滤。

当前冻结判定逻辑：

1. 基础门槛：`n_features >= min_features`
2. 若 `use_mad=True`，再附加 lower-tail MAD rule：
   - 对 `n_features.astype(float)` 调用 `is_outlier_mad(..., direction="lower")`
   - lower outlier 被移除
3. 最终 keep 规则是两者取交集

结构合同：

1. 过滤的是 **sample 轴**，因此所有 assay 都必须沿样本轴同步子集化。
2. feature 轴不得变化。
3. 保留样本的相对顺序不得变化。
4. 返回新容器；输入容器不得原地修改。

语义边界：

- 这是 sample-quality filter，不是 control-aware benchmark-grade gate。
- 它当前实现的是 `hard threshold + optional robust lower-tail rule`。
- 它当前 **没有** 内建 control-aware threshold，也没有 batch-stratified MAD。

### 5.3 `filter_doublets_mad`

职责：

- 基于高总强度样本的 upper-tail outlier 过滤潜在 doublet-like samples。

当前冻结判定逻辑：

1. 先计算每个 sample 的 `total_intensity`
2. 再做 `log1p(total_intensity)`
3. 调用 `is_outlier_mad(..., direction="upper")`
4. upper-tail outlier 视为潜在 doublet-like sample 并移除

结构合同：

1. 过滤的是 sample 轴，因此所有 assay 都要同步子集化。
2. 返回新容器；输入容器不得原地修改。
3. 不新增 obs / var 字段。

语义边界：

- 这是 **heuristic outlier filter**，不是社区定型的 single-cell proteomics doublet detector。
- 后续实现可以优化速度或稳健性，但不能把它在文档里升级成“文献标准 doublet 方法”。

## 6. Feature QC 合同

### 6.1 `calculate_feature_qc_metrics`

职责：

- 计算 feature-level QC 摘要，不做 feature 过滤。

稳定输出：

- 写入目标 assay 的 `var`
- 追加以下列：
  - `missing_rate`
  - `detection_rate`
  - `mean_expression`
  - `cv`

当前冻结语义：

- `missing_rate = 1 - detection_rate`
- `detection_rate`：按 `VALID` detection semantics 计算
- `mean_expression`：仅在 detected entries 上求均值
- `cv`：
  - 基于 `X` 的数值矩阵计算
  - dense 使用 `np.nanmean / np.nanstd`
  - sparse 使用均值与平方均值推导方差
  - 对接近零均值的 feature 置为 `NaN`

结构合同：

1. 只改目标 assay 的 `var`。
2. 不过滤 feature。
3. 不改变其他 assay。
4. 返回新容器；输入容器不得原地修改。

当前冻结的不对称点：

- 该函数当前 **会写 history**，`action="calculate_feature_qc_metrics"`。
- 不要在优化中把这条 history 静默去掉。

### 6.2 `filter_features_by_missingness`

职责：

- 基于 feature missingness 过滤目标 assay 的 feature。

当前冻结判定逻辑：

1. 用 detection semantics 计算每个 feature 的 `missing_rate`
2. keep 规则：`missing_rate <= max_missing_rate`

结构合同：

1. 只过滤目标 assay 的 feature 轴。
2. `obs` 不变。
3. 其他 assay 不变。
4. 目标 assay 所有 layer 必须同步裁剪相同 feature index。
5. 保留 feature 的相对顺序不得变化。

### 6.3 `filter_features_by_cv`

职责：

- 基于 feature CV 过滤目标 assay 的 feature。

当前冻结判定逻辑：

1. 调用 `compute_cv(X, axis=0, min_mean=min_mean)`
2. keep 规则：
   - `cv <= max_cv`
   - `cv` 非 `NaN`

结构合同：

1. 只过滤目标 assay 的 feature 轴。
2. `obs` 不变。
3. 其他 assay 不变。
4. 所有 layer 同步裁剪相同 feature index。

语义边界：

- 这是 context-sensitive 技术稳定性过滤，不应在 stable 文档中被包装成对高异质单细胞队列的 universal gate。

## 7. Batch Assessment 合同

### 7.1 `assess_batch_effects`

职责：

- 生成 QC 层面的 batch summary。
- 它是 **描述性 QC 汇总**，不是 integration benchmark metric，也不是 batch correction 方法评分器。

稳定输入：

- `batch_col` 必须存在于 `container.obs`

稳定输出：

- 返回 `polars.DataFrame`
- 第一列名保持为调用方提供的 `batch_col`
- 其余稳定列：
  - `n_cells`
  - `median_features`
  - `std_features`
  - `median_intensity`

当前冻结语义：

- `n_features` 仍按 `VALID` detection semantics 计算
- `total_intensity` 仍按 `X` 的数值求和
- 聚合方式：
  - count -> `n_cells`
  - median -> `median_features`
  - std -> `std_features`
  - median -> `median_intensity`
- 结果按 `batch_col` 升序排序

结构合同：

1. 不返回 `ScpContainer`，返回独立 `pl.DataFrame`。
2. 不过滤任何样本或 feature。
3. 不写 history。

### 7.2 与 integration diagnostics 的边界

`assess_batch_effects` 不能无声扩展成以下内容：

- ASW
- LISI
- kBET
- graph connectivity
- PCR

这些属于 integration / batch-diagnostics 语义，应留在 `integration` 或 `autoselect` 侧。

## 8. 阈值类别合同

本文档冻结的是“阈值类别”，不是“全仓统一硬数字”。

### 8.1 Sample QC 阈值类别

按当前综述与 stable 边界，sample QC 的优先级应是：

1. `control-aware threshold`
2. `batch-aware robust threshold`
3. `hard fallback threshold`

当前代码实际已实现的是：

- `hard fallback threshold`：`min_features`
- `robust threshold`：lower-tail MAD on `n_features`

当前代码尚未内建的是：

- `control-aware threshold`
- 按 `batch / run / plate` 分层的 robust threshold

冻结要求：

1. 不要把当前 `min_features` 或 MAD 规则写成“优于 controls 的默认准则”。
2. 若后续补 control-aware sample QC，应作为新增 stable 能力，而不是改写现有函数语义。

### 8.2 Doublet-like 阈值类别

`filter_doublets_mad` 的 `nmads` 仅定义启发式强弱档位：

- `2.0`：aggressive
- `3.0`：standard
- `4.0`：conservative

冻结要求：

1. 这是 outlier stringency，不是单细胞蛋白组 doublet benchmark 标准。
2. 不应在 stable 文档中赋予其 community-standard 语义。

### 8.3 Feature missingness 阈值类别

按现有综述，可保留 evidence-aware threshold classes，但不得把其中任何一档升级成 universal default：

- `0.10`：严格 completeness gate
- `0.25`：约 75% completeness 的常见起点
- `0.34`：较宽松
- `0.50+`：探索性 / 高异质场景

冻结要求：

1. 这些是阈值类别，不是全仓统一硬默认。
2. `filter_features_by_missingness` 的核心合同是比较运算本身，而不是某个固定数字。

### 8.4 Feature CV 阈值类别

按当前 stable 边界，CV gate 只能被视为 context-sensitive threshold class：

- 技术重复 / pooled QC / 同质细胞系：更适合使用
- 高异质单细胞样本：误删风险更高

当前 docstring 中的经验档位可保留为实现参考：

- `0.3-0.5`：严格
- `1.0`：宽松

但冻结要求是：

- 不把 `max_cv` 升格为单细胞蛋白组 global stable default

### 8.5 Batch assessment 不定义全仓 pass/fail 阈值

`assess_batch_effects` 只输出摘要；当前 stable QC 合同不定义 repo-wide batch fail threshold。

## 9. Provenance 与报告字段合同

### 9.1 obs / var 报告字段

稳定报告字段如下：

| 入口 | 写入位置 | 稳定字段 |
|---|---|---|
| `calculate_sample_qc_metrics` | `obs` | `n_features_{assay}`, `total_intensity_{assay}`, `log1p_total_intensity_{assay}` |
| `calculate_feature_qc_metrics` | 目标 assay `var` | `missing_rate`, `detection_rate`, `mean_expression`, `cv` |
| `assess_batch_effects` | 独立 `pl.DataFrame` | `batch_col`, `n_cells`, `median_features`, `std_features`, `median_intensity` |

冻结要求：

1. 不要无声改列名。
2. 若新增列，应保持现有字段继续可用。
3. 若未来引入 richer QC report object，也必须至少能恢复当前字段集。

### 9.2 history 行为矩阵

当前 stable history 行为如下：

| 入口 | 当前 history 行为 |
|---|---|
| `calculate_sample_qc_metrics` | 不写 history |
| `filter_low_quality_samples` | 先由 `container.filter_samples()` 写一条 `filter_samples`，再写一条 `filter_low_quality_samples` |
| `filter_doublets_mad` | 先写 `filter_samples`，再写 `filter_doublets_mad` |
| `assess_batch_effects` | 不写 history |
| `calculate_feature_qc_metrics` | 写一条 `calculate_feature_qc_metrics` |
| `filter_features_by_missingness` | 先写 `filter_features`，再写 `filter_features_by_missingness` |
| `filter_features_by_cv` | 先写 `filter_features`，再写 `filter_features_by_cv` |

冻结要求：

1. 当前 action names 视为稳定可见字符串，不应在重构中静默重命名。
2. 若后续要压缩成单条 log 或改为结构化 provenance schema，应提供兼容层，保证现有 action names 和核心参数仍能恢复。

### 9.3 generic filtering provenance

当前 core filtering 会生成通用 history entry：

- `filter_samples`
  - `n_samples_kept`
  - `n_samples_original`
  - `kept_sample_ids`
- `filter_features`
  - `assay_name`
  - `n_features_kept`
  - `n_features_original`
  - `kept_feature_ids`

对 QC 优化的冻结要求：

1. sample / feature 过滤必须继续保留“结构过滤发生过”的 provenance 事实。
2. 若未来为性能原因不再写完整 `kept_*_ids`，必须在合同层显式更新，而不是静默删掉。

### 9.4 QC-specific provenance

当前 QC-specific action 及其最小参数集如下：

- `filter_low_quality_samples`
  - `assay`
  - `min_features`
  - `use_mad`
  - `nmads`
- `filter_doublets_mad`
  - `assay`
  - `nmads`
  - `method`
- `calculate_feature_qc_metrics`
  - `assay`
  - `layer`
  - `n_features`
  - `n_samples`
- `filter_features_by_missingness`
  - `assay`
  - `n_removed`
  - `n_total`
  - `max_missing_rate`
- `filter_features_by_cv`
  - `assay`
  - `n_removed`
  - `n_total`
  - `max_cv`
  - `min_mean`

冻结要求：

1. 这些参数键名当前视为 stable provenance surface。
2. 可以新增键，但不要静默删除或改名。

## 10. Failure Contract

### 10.1 稳定异常类型

对 user-facing stable QC 入口，当前冻结异常面如下：

- assay 不存在：`AssayNotFoundError`
- layer 不存在：`LayerNotFoundError`
- 阈值不合法：`ScpValueError`
- `batch_col` 不存在：`ScpValueError`

### 10.2 各入口的失败边界

- `calculate_sample_qc_metrics`
  - 非法 assay -> `AssayNotFoundError`
  - 非法 layer -> `LayerNotFoundError`
- `filter_low_quality_samples`
  - 非法 assay -> `AssayNotFoundError`
- `filter_doublets_mad`
  - 非法 assay -> `AssayNotFoundError`
- `assess_batch_effects`
  - 非法 assay -> `AssayNotFoundError`
  - 缺少 `batch_col` -> `ScpValueError`
- `calculate_feature_qc_metrics`
  - 非法 assay -> `AssayNotFoundError`
  - 非法 layer -> `LayerNotFoundError`
- `filter_features_by_missingness`
  - `max_missing_rate` 不在 `[0, 1]` -> `ScpValueError`
  - 非法 assay -> `AssayNotFoundError`
  - 非法 layer -> `LayerNotFoundError`
- `filter_features_by_cv`
  - `max_cv <= 0` -> `ScpValueError`
  - 非法 assay -> `AssayNotFoundError`
  - 非法 layer -> `LayerNotFoundError`

### 10.3 统计工具的空输入行为

当前底层统计工具的空输入行为也已被实现固定：

- `compute_mad([])` -> `np.nan`
- `is_outlier_mad([])` -> 空 `bool` 数组
- `compute_cv(...)` 对接近零均值项 -> `np.nan`

冻结要求：

- 不要把这些情况改成抛异常，除非同步修改更高层调用合同。

## 11. 可安全优化但不可破坏的不变量

后续对 `scptensor.qc` 的优化，只有在不破坏以下不变量时才是“安全优化”。

### 11.1 容器与轴不变量

1. QC 入口默认返回新对象或独立 DataFrame，不得原地改输入容器。
2. sample filtering 必须作用于所有 assay 的样本轴。
3. feature filtering 只作用于目标 assay 的 feature 轴。
4. 保留条目的相对顺序不得变化。
5. 不得借 QC 操作偷偷改 assay / layer 名称。

### 11.2 detection semantics 不变量

1. `VALID`-only detection 语义不能无声改变。
2. sample-level `n_features` 与 feature-level `detection_rate/missing_rate` 必须继续共享同一 detection semantics。
3. `VALID` zero counts as detected 这一点不能被性能优化破坏。

### 11.3 报告字段不变量

1. `obs` / `var` / batch summary 的稳定列名不能无声改变。
2. 当前 history action names 与最小参数集不能无声改变。
3. 过滤操作必须保留 provenance，可多写但不能少写到“无法判断结构过滤是否发生”。

### 11.4 场景边界不变量

1. stable QC 主线是 protein-level，而不是 peptide / PSM-level。
2. `assess_batch_effects` 仍然是 QC summary，不是 integration metric。
3. `filter_doublets_mad` 仍然是 heuristic outlier filter，不是 benchmark-certified doublet detector。
4. `filter_features_by_cv` 仍然是 context-sensitive technical filter，不是全场景 universal gate。

## 12. 后续实现工作的建议约束

本文档不是 roadmap，但它为后续实现给出三条明确约束：

1. 若要新增 control-aware sample QC，应新增接口或新增显式参数，不要改写 `filter_low_quality_samples` 现有语义。
2. 若要统一 sample/feature QC 的 provenance 行为，必须把“当前不对称点”作为显式 contract change 处理。
3. 若要引入 richer batch QC report，应保持 `assess_batch_effects` 的当前摘要字段仍能稳定导出。

## 13. 一句话结论

`scptensor.qc` 的 stable contract 目前是：在 **protein-level quantitative matrix** 上，以 **VALID-only detection semantics** 生成 sample / feature QC 摘要与结构过滤结果；它保留 batch-aware 的解释入口，但尚未把 control-aware threshold、integration diagnostics 或 peptide/PSM QC 纳入 stable preprocessing 主线。
