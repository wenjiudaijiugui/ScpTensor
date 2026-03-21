# ScpTensor 可视化冻结实现合同（`scptensor.viz`，2026-03-17）

## 1. 文档目标

本文档冻结 `scptensor.viz` 当前实现合同，服务于后续：

- 文档示例统一
- plot API 收口
- 返回类型与错误语义稳定化
- 可视化辅助代码重构

它回答的是：

- `scptensor.viz` 当前有哪些稳定公开入口；
- 不同 recipe 家族默认依赖哪些 assay / layer / obs / var 约定；
- 返回的是 `Axes`、`Figure` 还是 `np.ndarray[Axes]`；
- 哪些行为虽然还不完美，但已经被代码与测试锁住，后续不能悄悄改掉。

本文档基于以下仓库内事实：

- `scptensor/viz/__init__.py`
- `scptensor/viz/base/validation.py`
- `scptensor/viz/recipes/__init__.py`
- `scptensor/viz/recipes/embedding.py`
- `scptensor/viz/recipes/feature.py`
- `scptensor/viz/recipes/matrix.py`
- `scptensor/viz/recipes/qc.py`
- `scptensor/viz/recipes/impute.py`
- `scptensor/viz/recipes/statistics.py`
- `scptensor/viz/recipes/workflow.py`
- `scptensor/viz/recipes/report.py`
- `tests/viz/*.py`

## 2. 范围与非范围

### 2.1 范围

本文档覆盖：

- `scptensor.viz` 顶层导出的 canonical `plot_*` API
- backward-compatible alias
- `base` primitives 的输入/输出合同
- recipe 层对 assay / layer / obs / var / history 的当前依赖
- 单面板、多面板、报告级函数的返回结构
- 主要报错路径与校验边界

### 2.2 非范围

本文档不覆盖：

- Matplotlib 风格微调细节
- 调色板优劣评价
- 下游降维 / 聚类算法本身是否合理
- notebook / tutorial 编排

## 3. 模块定位

`scptensor.viz` 的当前稳定定位是：

- 作为预处理与分析结果的可视化读取层
- 为 QC、归一化、填补、聚合、integration、统计汇总提供图形摘要
- 为实验性 downstream helper 提供只读展示接口

它不是：

- 核心 preprocessing 语义定义层
- reduction / clustering 算法实现合同
- 持久化 artifact 管理层

因此，`viz` 可以消费实验性结果，但不应反过来定义核心预处理合同。

## 4. 当前公开 API 分型

### 4.1 base primitives

`scptensor.viz` 顶层当前暴露：

- `base_scatter`
- `heatmap`
- `violin`

这些函数的稳定定位是：

- 面向通用绘图原语
- 通常直接返回 `matplotlib.axes.Axes`

### 4.2 canonical `plot_*` API

当前 `scptensor.viz` 顶层 canonical API 包括：

- embedding:
  - `plot_embedding_scatter`
  - `plot_embedding_umap`
  - `plot_embedding_pca`
  - `plot_embedding_tsne`
  - `plot_embedding`
- feature / matrix:
  - `plot_feature_dotplot`
  - `plot_matrixplot`
  - `plot_matrix_heatmap`
  - `plot_tracksplot`
- QC:
  - `plot_qc_completeness`
  - `plot_qc_matrix_spy`
  - `plot_qc_pca_overview`
  - `plot_qc_missing_value_patterns`
- statistics:
  - `plot_correlation_matrix`
  - `plot_dendrogram`
- imputation:
  - `plot_imputation_comparison`
  - `plot_imputation_scatter`
  - `plot_imputation_metrics`
  - `plot_missing_pattern`
- workflow:
  - `plot_aggregation_summary`
  - `plot_data_overview`
  - `plot_qc_filtering_summary`
  - `plot_preprocessing_summary`
  - `plot_normalization_summary`
  - `plot_missingness_reduction`
  - `plot_integration_batch_summary`
  - `plot_reduction_summary`
  - `plot_embedding_panels`
  - `plot_saved_artifact_sizes`
  - `plot_recent_operations`
- report:
  - `generate_analysis_report`
  - `ReportTheme`

### 4.3 backward-compatible alias

当前仍保留 alias：

- `scatter`
- `umap`
- `pca`
- `tsne`
- `embedding`
- `qc_completeness`
- `qc_matrix_spy`
- 以及 recipe 子模块中的 `dotplot` / `matrixplot` / `tracksplot` / `correlation_matrix` / `dendrogram`

仓库级文档与教程应优先使用 canonical `plot_*` 名称；alias 主要用于兼容旧代码。

### 4.4 根包 `scptensor` 的重导出边界

需要和 `scptensor.viz` 顶层区分开：

- `scptensor.viz` 是 canonical plotting namespace
- 根包 `scptensor` 当前只重导出一个 **子集**

当前根包 `scptensor.__all__` 保留的可视化导出主要是：

- base / alias：
  - `scatter`
  - `heatmap`
  - `violin`
  - `embedding`
  - `qc_completeness`
  - `qc_matrix_spy`
- workflow summary helpers：
  - `plot_data_overview`
  - `plot_qc_filtering_summary`
  - `plot_preprocessing_summary`
  - `plot_missingness_reduction`
  - `plot_reduction_summary`
  - `plot_embedding_panels`
  - `plot_saved_artifact_sizes`
  - `plot_recent_operations`

当前**不会**从根包 `scptensor` 重导出大多数 canonical `plot_*` API，例如：

- `plot_embedding_scatter`
- `plot_qc_completeness`
- `plot_imputation_comparison`
- `plot_matrix_heatmap`
- `generate_analysis_report`
- `ReportTheme`

因此冻结解释应为：

- 若写稳定 plotting 文档与教程，优先使用 `scptensor.viz`
- 根包 `scptensor` 的可视化导出更多是 convenience/compatibility 子集，而不是完整 canonical plotting surface

## 5. 当前数据边界与默认假设

### 5.1 assay 主线默认是 `proteins`

从实现与测试看，`viz` 当前绝大多数 recipe 默认都围绕：

- `assay_name="proteins"`
- `layer="raw"` 或其直接后续层

尤其是 `embedding.scatter()` 当前直接硬编码先找 `container.assays["proteins"]`。

因此稳定解释必须是：

- `viz` 主线是 protein-level visualization
- peptide 级别展示可以存在，但不是当前默认路径

这与项目总合同一致。

### 5.2 layer 假设是“调用方显式给出”

大部分 recipe 都要求调用方明确给出 layer，例如：

- `raw`
- `normalized`
- `norm`
- `imputed`
- `integrated`
- `scores`

这里要明确两点：

1. `viz` 不统一强制仓库级 canonical layer 命名
2. 它读取的是“当前对象里真实存在的 layer 名”

因此：

- 文档主线仍应优先写 `raw / log / norm / imputed`
- 但实现合同上必须接受测试里已经形成的 `normalized`、`log2`、`scores`、`X` 等兼容或局部命名

### 5.3 `obs` 是大多数分组与坐标的 source of truth

当前可视化广泛依赖 `container.obs`：

- `groupby` / `group_by`
- batch / condition / cluster 列
- embedding 坐标列

稳定合同：

- 分类分组信息默认来自 `obs`
- embedding 坐标在很多 recipe 中也默认来自 `obs`

### 5.4 `var` 提供 feature 解析与附加统计

当前 feature 解析普遍按如下顺序尝试列名：

- assay 自己的 `feature_id_col`
- `protein`
- `gene`
- `feature`
- `name`
- `_index`

因此：

- `viz` 不是严格绑定单一列名
- 但 protein-level assay 至少要有一个能解析出 feature 名称的列

## 6. 返回类型合同

### 6.1 单面板函数

当前以下家族通常返回 `matplotlib.axes.Axes`：

- base primitives
- `plot_embedding_*`
- `plot_feature_dotplot`
- `plot_matrixplot`
- `plot_matrix_heatmap`
- `plot_tracksplot`
- `plot_qc_completeness`
- `plot_qc_matrix_spy`
- `plot_missing_pattern`
- `plot_imputation_comparison`
- `plot_imputation_scatter`
- `plot_imputation_metrics`
- `plot_missingness_reduction`
- `plot_saved_artifact_sizes`
- `plot_recent_operations`

### 6.2 多面板摘要函数

当前 workflow 里的多面板摘要函数大量返回 `np.ndarray` of `Axes`，例如：

- `plot_aggregation_summary()` -> 3 panels
- `plot_data_overview()` -> 3 panels
- `plot_qc_filtering_summary()` -> 3 panels
- `plot_preprocessing_summary()` -> 3 panels
- `plot_normalization_summary()` -> 3 panels
- `plot_integration_batch_summary()` -> 3 panels
- `plot_reduction_summary()` -> 2 panels
- `plot_embedding_panels()` -> 1..N panels

也就是说，当前返回合同按“图像结构”而不是“统一单一 figure 类型”组织。

### 6.3 figure 级函数

当前以下函数返回 `matplotlib.figure.Figure`：

- `plot_correlation_matrix()`
- `plot_dendrogram()`
- `plot_qc_pca_overview()`
- `plot_qc_missing_value_patterns()`
- `generate_analysis_report()`

因此，后续 API 重构若想统一返回类型，必须显式设计迁移方案，不能直接偷偷把 `Axes` 改成 `Figure` 或相反。

## 7. 校验与报错合同

### 7.1 validation helper 的当前职责

`scptensor.viz.base.validation` 当前明确区分：

- `validate_container()`:
  - `None` 或错误类型 -> `VisualizationError`
- `validate_layer()`:
  - assay 缺失 -> `VisualizationError`
  - layer 缺失 -> `LayerNotFoundError`
- `validate_features()`:
  - assay 缺失、feature 缺失、空 `var` -> `VisualizationError`
- `validate_groupby()`:
  - obs 列缺失 -> `VisualizationError`
- `validate_plot_data()`:
  - 数据不足 -> `VisualizationError`

这套错误分工已经被 `tests/viz/test_validation_base.py` 锁定。

### 7.2 recipe 层并不完全统一使用同一异常族

需要明确一个当前现实：

- 一部分 recipe 使用 `validate_*()`，因此更偏 `VisualizationError / LayerNotFoundError`
- 另一部分 recipe 直接用 `ValueError`
- imputation 部分还会抛 `AssayNotFoundError`

因此，当前 `viz` 的稳定合同是“报错信息尽量明确”，而不是“所有函数都统一到同一异常类”。

后续若要统一异常体系，需要显式做 API 收敛，不应在小修里静默改变。

## 8. Embedding 家族合同

### 8.1 坐标来源

`plot_embedding*` / `embedding()` 当前默认从 `obs` 读取：

- `{basis}_1`
- `{basis}_2`

例如：

- `umap_1`, `umap_2`
- `pca_1`, `pca_2`
- `reduce_umap_1`, `reduce_umap_2`

若列不存在，当前抛 `ValueError`，并把可用列名写进错误信息。

### 8.2 颜色来源

当前 `color` 可以来自：

- `obs` 元数据列
- assay `var` 中能解析到的 feature 名称

若是 feature 着色：

- 从指定 assay/layer 取一列表达值
- 同时取对应 mask

### 8.3 缺失值可视化语义

embedding 家族当前支持 `show_missing_values=True`。

当前稳定语义不是“直接把缺失点删除”，而是：

- 检测 mask
- 对有效点与缺失点做分层绘制

`tests/viz/test_embedding_mask_codes.py` 与 `tests/viz/test_viz.py` 已锁定：

- 含 imputed/missing code 时会产生额外 overlay
- `mask_style="subtle"` / `mask_style="explicit"` 两种风格都必须工作

### 8.4 默认 assay 的当前限制

`embedding.scatter()` 当前内部固定先使用 assay `"proteins"`。

这意味着它还不是一个完全 assay-agnostic 的通用接口。后续若要放宽此限制，需要显式加参数并补测试，而不是默默改默认行为。

## 9. Feature / Matrix / Statistics 家族合同

### 9.1 Feature dotplot

`plot_feature_dotplot()` / `dotplot()` 当前稳定行为：

- `var_names` 不能为空
- `dendrogram=True` 当前显式不支持，并抛 `VisualizationError`
- `standard_scale` 仅接受 `var` / `obs` / `None`
- 对 sparse `X` 会先 densify 成 NumPy array 再做绘图

### 9.2 Matrix plot / heatmap / tracksplot

这组函数当前都依赖：

- assay + layer 明确存在
- 指定 `var_names` 能在 assay.var 中解析
- `groupby` 在 `obs` 中存在

因此它们是“以 protein matrix 为中心的汇总图”，而不是任意对象的通用热图封装。

### 9.3 Statistics 图

`plot_correlation_matrix()` / `plot_dendrogram()` 当前属于 figure 级分析摘要：

- `groupby=None` 时可按样本做统计
- `groupby` 存在时可按组做统计
- group 数不足等异常当前走 `ValueError`

## 10. QC 家族合同

### 10.1 `plot_qc_completeness()`

当前逻辑：

- 优先使用 `matrix.M`
- 若 `M is None`，则构造全零 mask
- completeness 按 `M == 0` 计数

这意味着当前 completeness 的核心定义是 mask-driven，而不是直接按 `np.isfinite(X)`。

### 10.2 `plot_qc_matrix_spy()`

当前逻辑：

- `spy_data = (M > 0).astype(np.uint8)`
- 标题包含 `assay_name/layer`
- 结果是 measured vs missing 的二元热图

测试已锁定标题中含 `proteins/raw` 之类路径信息。

### 10.3 `plot_qc_pca_overview()`

当前要求：

- 需要 PCA assay，默认 `pca`
- PCA layer 通常为 `scores`
- `pca` assay 的 `var` 里要有 `explained_variance_ratio` 或兼容列
- 原 assay 的 `var` 中若存在 `pca_PC*_loading`，会读作 loading

缺失 explained variance 列时，当前抛 `ValueError`。

### 10.4 `plot_qc_missing_value_patterns()`

当前依赖 `matrix.M` 做 missing pattern 分析。

若 `M is None`：

- 不直接失败
- 会在图里显示 “No missing values found” 之类提示

这条“优雅降级而不是硬失败”的行为已被测试固定。

## 11. Imputation 家族合同

### 11.1 `plot_imputation_comparison()`

当前稳定行为：

- 在给定 assay 中比较多个 imputed layer
- 若 `methods is None`，自动搜常见前缀：
  - `knn_`
  - `qrilc_`
  - `bpca_`
  - `nmf_`
  - `lls_`
  - `svd_`
  - `mf_`
  - `ppca_`
  - `minprob_`
  - `mindet_`
- 若找不到任何 layer，抛 `ValueError`
- 结果 `Axes` 上会附带 `ax.imputation_results`

因此它不仅是绘图函数，还是轻量结果承载器。

这里还必须和 `docs/imputation_contract.md` 的命名边界对齐：

- 这些 auto-detect 前缀是当前 `viz.recipes.impute` 的 **legacy visualization-local heuristic**
- 它们既不等于仓库级 canonical `imputed` 主结果层命名
- 也不等于 `scptensor.impute` 当前 wrapper 默认层名集合，例如 `imputed_knn`

因此稳定解释应为：

- `plot_imputation_comparison(methods=None)` 只是在消费现有对象中的局部命名惯例
- 它不是 `imputation` 主线命名规范的 source of truth
- 若调用方使用 canonical `imputed` 或 method-wrapper 默认层名，当前应显式传入 `methods=[...]`，而不是依赖该 auto-detect heuristic

### 11.2 `plot_imputation_scatter()` / `plot_imputation_metrics()` / `plot_missing_pattern()`

这几个函数当前共同假设：

- 输入容器中存在指定 assay/layer
- `raw` 与 imputed layer 关系由调用方保证
- sparse 输入会在绘图前 densify

因此可视化层不会替调用方决定“哪个 layer 才是真实金标准”。

## 12. Workflow / Report 家族合同

### 12.1 workflow helpers 是 stage-level summary

`plot_aggregation_summary()`、`plot_preprocessing_summary()`、`plot_integration_batch_summary()` 等函数当前定位明确：

- 为流程阶段提供摘要图
- 不负责执行阶段逻辑本身

例如：

- `plot_aggregation_summary()` 读取 `container.links` 里的最近 link
- 若无匹配 link，抛明确 `ValueError`
- 它假设调用方已先跑过 `aggregate_to_protein()`

### 12.2 workflow helpers 当前允许混合多种 layer/assay 命名

测试已覆盖这些实现事实：

- `before_layer="raw"` / `after_layer="normalized"`
- `raw_layer="raw"` + `transformed_layers=("log2", "norm", "imputed")`
- `plot_embedding_panels(assay_names=("pca",), layer="X", color_by="kmeans_k3")`

这里要额外明确：

- `("log2", "norm", "imputed")` 是当前 `workflow.py` 与测试基线里的实现局部默认
- 它不等于仓库级 canonical layer naming
- 仓库主线示例、教程与跨模块文档仍应优先写 `raw / log / norm / imputed`

因此 workflow 层当前消费的是“现有对象状态”，不是严格的 canonical naming checker。

### 12.3 `generate_analysis_report()`

`generate_analysis_report()` 当前返回整张 `Figure`，并由 `ReportTheme` 控制视觉参数。

当前只有它内建了文件写出路径：

- `output_path is not None` -> `fig.savefig(...)`

其余绝大多数 `viz` helper 当前只负责返回 `Axes` / `Figure` / `np.ndarray[Axes]`，不承担统一 artifact 写出责任。

`ReportTheme` 当前稳定支持：

- 默认主题
- `dark()`
- `colorblind()`

测试已锁定一组默认值与 preset 值，因此即使未来要换主题系统，也不能无迁移地改变这些 preset 的基本语义。

## 13. 当前已知实现不对称点

以下实现差异在当前源码里真实存在，文档必须把它们记成“现状”，而不是默认它们已经统一：

### 13.1 `qc_advanced` 导出层级未上浮到顶层 `scptensor.viz`

`scptensor.viz.recipes.__all__` 当前包含：

- `plot_cumulative_sensitivity`
- `plot_cv_by_feature`
- `plot_cv_comparison`
- `plot_cv_distribution`
- `plot_jaccard_heatmap`
- `plot_missing_summary`
- `plot_missing_type_heatmap`
- `plot_sensitivity_summary`

但它们没有同步上浮到 `scptensor.viz.__all__`。

因此当前真实边界是：

- `scptensor.viz.recipes` 比 `scptensor.viz` 顶层暴露更多高级 QC 图
- 顶层 `viz` 还不是完整的 recipe mirror

### 13.2 缺失定义存在双轨

当前缺失判断并未统一：

- QC 与 embedding 的一部分逻辑更依赖 `M`
- `workflow.py` 中 `_detected_mask()` 使用 `(X > 0) & np.isfinite(X)`
- `workflow.py` 中 `_missing_rate()` 使用 `~np.isfinite(X)`

因此 workflow 里“检测率/缺失率”与 QC 家族中的 mask-driven missing 并不是同一套定义。

### 13.3 默认 layer 并不统一

当前仓库内已经形成三类默认取值：

- embedding / many examples 常用 `normalized`
- QC / workflow 多数默认 `raw`
- PCA overview 常用 `scores`

因此 `viz` 当前是“消费现有 layer 状态”的读取层，而不是强制 canonical layer 命名的约束器。

### 13.4 `generate_analysis_report(panels=...)` 参数当前未被消费

`generate_analysis_report()` 签名里有 `panels: list[str] | None = None`，但当前实现固定渲染 8 个 panel，没有根据该参数裁剪布局。

因此调用方此刻不能把 `panels` 当成真正生效的稳定功能。

### 13.5 report 中 `group_col` 与 `batch_col` 作用域不同

当前 report 实现里：

- `_render_qc_panel()` 使用的是 `batch_col`
- `_render_embedding_panel()` 使用的是 `group_col`
- `_render_batch_panel()` 也使用的是 `batch_col`

因此 `group_col` 并不是“所有 panel 的统一分组列”，而更接近 embedding 着色列；`batch_col` 则承担 QC/batch panel 的主线分组职责。

## 14. 重构时不得破坏的高优先级约束

后续若要重构 `scptensor.viz`，必须优先保留：

1. canonical `plot_*` API 名称
2. alias 的 backward compatibility
3. 单图 / 多图 / figure 三类返回结构
4. `embedding` 对 obs 坐标列 `{basis}_1/{basis}_2` 的读取约定
5. protein-level 主线默认假设
6. validation helper 的错误分工
7. `plot_qc_missing_value_patterns()` 在 `M is None` 时的优雅降级
8. workflow helper 只读摘要、不执行 preprocessing 本体的边界

## 15. 对后续优化的直接指导

基于当前仓库状态，`viz` 下一阶段最合理的完善方向是：

- 统一 recipe 层异常类型，但以兼容层方式推进
- 把“默认 `proteins`”与“可传任意 assay”明确拆分为不同 API 或新增显式参数
- 为 figure/axes 返回类型建立更明确的文档分层，而不是强行全部统一
- 将 workflow helper 对 layer 名的兼容消费保留在实现层，同时在文档层继续坚持 canonical naming
- 逐步补齐 `plot_*` API 的参数表和稳定示例，而不是先改绘图风格

这意味着：`viz` 当前已经足以作为核心代码优化时的“读取层合同”，但还不适合被当作强约束的统一 plotting framework，需要在后续重构中继续收口异常和默认参数边界。
