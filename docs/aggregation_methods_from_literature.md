# 肽段到蛋白聚合方法（文献/官方资料整理）

本文档汇总了常用 peptide/precursor -> protein 聚合策略，并对应到 ScpTensor 当前实现。

## 1. 方法清单与来源

| 方法 | 核心思想 | 典型来源 | ScpTensor 实现状态 |
|---|---|---|---|
| Sum | 同一蛋白下肽段强度求和 | OpenMS ProteinQuantFromPeptideAbundances | 已实现 (`method="sum"`) |
| Mean | 同一蛋白下肽段强度求平均 | OpenMS ProteinQuantFromPeptideAbundances | 已实现 (`method="mean"`) |
| Median | 同一蛋白下肽段强度取中位数 | OpenMS ProteinQuantFromPeptideAbundances | 已实现 (`method="median"`) |
| Max | 同一蛋白下取最大肽段强度 | 常见汇总策略 | 已实现 (`method="max"`) |
| Weighted mean | 按肽段丰度权重做加权均值 | OpenMS ProteinQuantFromPeptideAbundances | 已实现 (`method="weighted_mean"`) |
| Top-N | 选蛋白最丰富的 N 条肽段再汇总 | OpenMS Top N + 文献中的 TOPn/Top3 讨论 | 已实现 (`method="top_n"`) |
| MaxLFQ | 基于样本间肽段比值网络求蛋白强度 | MaxLFQ 原始论文 + DIA-NN 输出列 `PG.MaxLFQ` | 已实现（近似版，`method="maxlfq"`） |
| TMP (Tukey Median Polish) | 对 log 强度做鲁棒分解汇总 | MSstats `dataProcess` 文档中的 TMP 汇总 | 已实现 (`method="tmp"`) |
| iBAQ | 蛋白肽段强度和 / 可观测肽段数 | OpenMS iBAQ 说明 + iBAQ 文献背景 | 已实现 (`method="ibaq"`) |

## 2. 主要参考链接

- OpenMS: `ProteinQuantFromPeptideAbundances`
  - https://openms.de/documentation/html/TOPP_ProteinQuantFromPeptideAbundances.html
- MaxLFQ 原始论文（Cox et al., 2014, MCP）
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC4159666/
- MSstats `dataProcess`（TMP summarization）
  - https://www.bioconductor.org/packages/release/bioc/manuals/MSstats/man/MSstats.pdf
  - https://rdrr.io/bioc/MSstats/man/dataProcess.html
- TOPn/Top3 背景综述（包含 TOPn 思路与引用）
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC3916661/
- iBAQ 相关背景综述（riBAQ/rTop3 比较）
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC10145028/
- DIA-NN 官方 README（`PG.MaxLFQ` / `PG.TopN` 输出字段）
  - https://raw.githubusercontent.com/vdemichev/DiaNN/master/README.md

## 3. 在 ScpTensor 中的接口

```python
from scptensor.aggregation import aggregate_to_protein

container = aggregate_to_protein(
    peptide_container,
    source_assay="peptides",
    source_layer="raw",
    method="top_n",                # sum/mean/median/max/weighted_mean/top_n/maxlfq/tmp/ibaq
    top_n=3,
    top_n_aggregate="median",
)
```

备注：
- `maxlfq` 为工程化近似实现（基于 pairwise median log-ratio + least squares）。
- `ibaq` 若未提供 `ibaq_denominator`，默认使用“映射到该蛋白的肽段数”作为分母近似。
