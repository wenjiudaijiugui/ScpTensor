# 文献不可采纳记录：Benchmarking informatics workflows for data-independent acquisition single-cell proteomics

- 文献标题：Benchmarking informatics workflows for data-independent acquisition single-cell proteomics
- DOI：<https://doi.org/10.1038/s41467-025-65174-4>
- 期刊链接：<https://www.nature.com/articles/s41467-025-65174-4>
- 记录日期：2026-03-29
- 记录状态：`inadmissible_for_reference`
- 关联性：主题与 DIA 单细胞蛋白组流程评测高度相关，但当前不满足本仓库 benchmark 参考采纳条件。

## 一、为何标记为不可采纳

1. 公共可执行代码链路未完成核验。
   - 未确认可直接复跑并重现主要图表的公开代码仓库。
2. 公共数据复现包未完成核验。
   - 未确认“原始输入矩阵 + 样本元数据 + 参数配置 + 结果映射”四件套可以完整获取并一键重跑。
3. 数据入口公开状态不足以支持当前仓库的复现实验要求。
   - 结合仓库既有评审记录（`docs/review_public_benchmark_data_20260312.md`），该文献关联的公开数据入口在复核时仍存在公开性约束，不能直接作为稳定主数据入口。
4. 因此该文献当前不能作为 ScpTensor 的定量 benchmark 对照基线。

## 二、可保留的工程启示（仅作启发，不作证据）

1. 先做可采纳性预检，再跑方法评测。
   - 若文献正文、代码、数据任一缺失，应直接中止“对标结论”流程并打上不可采纳标签。
2. 全流程必须分阶段留痕。
   - 对 `log -> normalize -> impute -> integration -> z-score -> dim_reduction -> cluster` 每一步输出结构化审计摘要，避免把差异误归因到单一模块。
3. marker 评估应做多策略并行审计。
   - 除当前 heuristic 外，保留如 max fold-change（可借助 Scanpy 兼容流程）的并行对照，检查 marker 稳定性而非只看单一排名。
4. 可视化对照要标准化。
   - 参考结果与 ScpTensor 结果应使用同一图形规范与同一指标口径，输出成对图板，减少主观解释空间。

## 三、该文献暴露出的 ScpTensor 薄弱环节与改进方向

| 环节 | 暴露问题 | 建议改进 |
| --- | --- | --- |
| 参考来源管理 | 文献可复现性检查前置不足，容易先跑再判 | 增加 `benchmark admissibility preflight`，缺正文/代码/数据即 hard fail |
| 流程可解释性 | 各阶段输出缺少统一审计资产，差异定位慢 | 增加每阶段 `json + 图` 的审计产物与统一 schema |
| marker 稳定性 | 更换 marker 规则后差异不明显时，缺少解释框架 | 增加 effect size 分布、top-k 一致性、符号一致性三类报告 |
| 结果对标严谨性 | 对比输出容易停留在叙述层面 | 固化“输入版本 + 参数 + 指标 + 图板 + 结论边界”的对比模板 |

## 四、在仓库中的使用结论

- 可否作为 benchmark 参考：否
- 可否作为 backlog 启示来源：是
- 可否用于支持“方法优劣定论”：否
