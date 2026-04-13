# Aggregation Background Note

本文档只保留 `peptide/precursor -> protein` 聚合的背景信息。
冻结实现边界以 `docs/aggregation_contract.md` 为准；benchmark 胜负解释以
`docs/internal/review_aggregation_benchmark_20260312.md` 为准。

## Method Families

| 方法 | 背景语义 | 当前实现 |
| --- | --- | --- |
| `sum` / `mean` / `median` / `max` | basic protein summarization | 已实现 |
| `weighted_mean` | abundance-weighted summarization | 已实现 |
| `top_n` | select top peptides then aggregate | 已实现 |
| `maxlfq` | pairwise ratio network summarization | 已实现，工程近似版 |
| `tmp` | Tukey Median Polish summarization | 已实现 |
| `ibaq` | summed intensity divided by observable-peptide denominator | 已实现 |

## External Anchors

- OpenMS `ProteinQuantifier`
  - http://www.openms.de/doxygen/release/3.4.1/html/TOPP_ProteinQuantifier.html
- MaxLFQ paper
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC4159666/
- MSstats `dataProcess` / TMP
  - https://rdrr.io/bioc/MSstats/man/dataProcess.html
- DIA-NN README fields such as `PG.MaxLFQ`
  - https://raw.githubusercontent.com/vdemichev/DiaNN/master/README.md

## Benchmark Interpretation

- 方法语义不等于 benchmark 胜负。一个方法“来自经典文献”不代表它会在当前
  DIA-sc 稀疏度、mapping 质量或 scoring panel 下自动获胜。
- `maxlfq`、`tmp` 这类方法带有更强的方法学假设；benchmark 里应把它们当
  family comparison，而不是把 vendor output 当绝对真值。
- `sum` / `mean` / `median` / `max` 更像 aggregation baseline family；适合做
  可解释对照，但不应单靠单一指标得出“普遍更优”。
- benchmark 至少应把方法语义与评测目标分开：
  - 方法语义：如何从 peptide/precursor 汇总到 protein
  - 评测目标：ratio preservation、feature consistency、unmapped burden、runtime

## Canonical Surface

```python
from scptensor.aggregation import aggregate_to_protein
```

当前主线仍要求：

- source assay 在 peptide/precursor 层
- target assay 在 protein 层
- 下游 preprocessing 从 protein-level matrix 继续
