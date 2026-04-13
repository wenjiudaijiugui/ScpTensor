# 文献记录索引（不可采纳来源）

`docs/article/` 用于存放“与研究主题相关，但不满足可复现基准采纳条件”的文献记录。
这些记录可以用于启发工程改进，但不能作为 ScpTensor 的定量对照基线。

每份记录必须包含：

- 文献标题、来源链接、记录日期
- 明确状态（如 `inadmissible_for_reference`）
- 不能采纳为 benchmark 参考的具体原因
- 可迁移的启示与可执行改进项

## 当前文件

当前只保留一条 consolidated inadmissible record：

- 文献：Benchmarking informatics workflows for data-independent acquisition
  single-cell proteomics
- DOI：<https://doi.org/10.1038/s41467-025-65174-4>
- 记录日期：`2026-03-29`
- 状态：`inadmissible_for_reference`
- 不能采纳的原因：
  - 公开可执行代码链路未完成核验
  - 公共数据复现包未完成核验
  - 关联数据入口仍不满足当前仓库的稳定可复现要求
- 可保留的工程启示：
  - benchmark 结论前先做 admissibility preflight
  - `log -> normalize -> impute -> integration -> z-score -> dim_reduction -> cluster`
    各阶段都要留结构化审计资产
  - marker 稳定性要做多策略并行审计
  - 参考对标输出要固定图板和指标口径
- 仓库内使用结论：
  - 不可作为 benchmark 参考
  - 可作为 backlog 启示来源
  - 不可用于支持“方法优劣定论”
