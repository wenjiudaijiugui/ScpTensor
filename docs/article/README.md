# 文献记录索引（不可采纳来源）

`docs/article/` 用于存放“与研究主题相关，但不满足可复现基准采纳条件”的文献记录。
这些记录可以用于启发工程改进，但不能作为 ScpTensor 的定量对照基线。

每份记录必须包含：

- 文献标题、来源链接、记录日期
- 明确状态（如 `inadmissible_for_reference`）
- 不能采纳为 benchmark 参考的具体原因
- 可迁移的启示与可执行改进项

## 当前文件

- `benchmarking_informatics_workflows_for_data-independent_acquisition_single-cell_proteomics_unusable_20260329.md`：
  记录该文献为何不可作为基准参考，以及对 ScpTensor 的改进启示。
- `dia_single_cell_processed_matrix_download_list_20260329.md`：
  汇总适合 ScpTensor 的 DIA 单细胞文献、可直接使用的定量矩阵/长表及下载清单。
