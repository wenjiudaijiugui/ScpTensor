# DIA 单细胞蛋白组可直接使用的定量矩阵与长列表文献清单

记录日期：2026-03-29

## 1. 目的与筛选规则

这份清单只保留满足以下条件的文献：

- 主题契合 ScpTensor 主线，即 `DIA-based single-cell proteomics`
- 文中或数据页明确提供了可直接使用的定量结果
- 可下载内容不是只有 `raw` 文件，而是至少包含以下一种：
  - `source data`
  - `supplementary data`
  - `processed quantification table`
  - `Spectronaut quantification outputs`
  - `protein / peptide long table`
  - `Zenodo / figshare / MassIVE / ProteomeXchange / iProX` 上的处理后表格

不纳入条件：

- 只有 raw 文件，没有可直接使用的矩阵或长表
- 仅有方法演示、没有明确处理后定量表
- 与项目主线冲突的 isobaric / TMT-only 入口

## 2. 文献清单

| 优先级 | 文献 | 为什么契合 ScpTensor | 可直接用的数据形态 | 下载入口 | 适配结论 |
|---|---|---|---|---|---|
| A | [Exploration of cell state heterogeneity using single-cell proteomics through sensitivity-tailored data-independent acquisition](https://doi.org/10.1038/s41467-023-41602-1) | 直接是 DIA + single-cell；正文明确给出处理后表和补充数据 | `Supplementary Data 1 (CSV)`, `Supplementary Data 2 (CSV)`, `Source Data (XLSX)`, Zenodo 处理后表 | Nature supplementary 下载页；Zenodo `10.5281/zenodo.7433298`、`10.5281/zenodo.8146605`；WISH-DIA 仓库 | 最推荐先用作导入与流程兼容性验证 |
| A | [Automated single-cell proteomics providing sufficient proteome depth to study complex biology beyond cell type classifications](https://doi.org/10.1038/s41467-024-49651-w) | 单细胞蛋白组公开资源较完整，适合做公开 reference layer | `Supplementary Data 1/4 (XLSX)`, `Supplementary Data 2/3 (TXT)`, `Source Data (XLSX)` | Nature supplementary 下载页；MassIVE `MSV000093867` | 适合做公开参考面板，但要标清 provenance |
| A | [Streamlined single-cell proteomics by an integrated microfluidic chip and data-independent acquisition mass spectrometry](https://doi.org/10.1038/s41467-021-27778-4) | 明确给出 `Spectronaut quantification outputs`，并且是 DIA single-cell | `reference spectral libraries`, `Spectronaut quantification outputs` | JPOST `JPST000971`；ProteomeXchange `PXD023325` | 适合验证“长表/结果表 -> protein matrix”导入链路 |
| A | [Pick-up single-cell proteomic analysis for quantifying up to 3000 proteins in a Mammalian cell](https://doi.org/10.1038/s41467-024-45659-4) | 文中明确包含 DIA 模式分析与 `Source data` | `Supplementary Data 3 (XLSX)`, `Source Data (XLSX)` | ProteomeXchange `PXD041966`；iProX `IPX0006351000` | 适合验证高覆盖 single-cell 表格的读取与下游处理 |
| B | [Protein-level batch-effect correction enhances robustness in MS-based proteomics](https://doi.org/10.1038/s41467-025-64718-y) | 虽然不是 single-cell 专属，但对 protein-level matrix 与 batch/integration 评测非常有价值 | `processed Quartet and simulated proteomics data`, `processed ChiHOPE proteomics data`, `Source Data (XLSX)` | Figshare `10.6084/m9.figshare.29567366.v2`、`10.6084/m9.figshare.29567333.v2`、`10.6084/m9.figshare.30028336` | 适合作为 protein-level integration 参考，不作为 single-cell 主数据入口 |

## 3. 数据下载清单

以下清单按“先下载可直接用的表，再补文献附件”的顺序组织。

### 3.1 主线优先下载

1. `Exploration of cell state heterogeneity using single-cell proteomics through sensitivity-tailored data-independent acquisition`
   - 下载 `Supplementary Data 1 (CSV)`
   - 下载 `Supplementary Data 2 (CSV)`
   - 下载 `Source Data (XLSX)`
   - 下载 Zenodo 处理后表 `10.5281/zenodo.7433298`
   - 下载 Zenodo 处理后表 `10.5281/zenodo.8146605`
   - 如需复现处理链，再拉取 `WISH-DIA` 代码仓库
2. `Automated single-cell proteomics providing sufficient proteome depth to study complex biology beyond cell type classifications`
   - 下载 `Supplementary Data 1 (XLSX)`
   - 下载 `Supplementary Data 2 (TXT)`
   - 下载 `Supplementary Data 3 (TXT)`
   - 下载 `Supplementary Data 4 (XLSX)`
   - 下载 `Source Data (XLSX)`
   - 如需公开参考层，再拉取 `MassIVE MSV000093867`
3. `Streamlined single-cell proteomics by an integrated microfluidic chip and data-independent acquisition mass spectrometry`
   - 下载 `Spectronaut quantification outputs`
   - 下载 `reference spectral libraries`
   - 如需关联原始数据，再看 `JPST000971` / `PXD023325`
4. `Pick-up single-cell proteomic analysis for quantifying up to 3000 proteins in a Mammalian cell`
   - 下载 `Supplementary Data 3 (XLSX)`
   - 下载 `Source Data (XLSX)`
   - 如需完整 proteomics 数据，再看 `PXD041966` / `IPX0006351000`

### 3.2 扩展参考下载

5. `Protein-level batch-effect correction enhances robustness in MS-based proteomics`
   - 下载 Figshare `10.6084/m9.figshare.29567366.v2`
   - 下载 Figshare `10.6084/m9.figshare.29567333.v2`
   - 下载 Figshare `10.6084/m9.figshare.30028336`
   - 下载 `Source Data (XLSX)`

## 4. 面向 ScpTensor 的使用建议

1. 优先把 `Exploration...` 和 `Streamlined...` 作为导入与矩阵组织的主测试集。
2. 把 `Automated...` 作为公开 reference layer，检查不同公开资源页的结构一致性。
3. 把 `Pick-up...` 作为高覆盖单细胞定量矩阵的补充测试。
4. 把 `Protein-level batch-effect correction...` 作为 integration / batch correction 的外部参考，不要当成 single-cell 主线数据源。
5. 所有下载项中，`raw` 只放在附加信息，不作为主输入。

## 5. 结论

- 这份清单里的文献都满足“至少存在可直接用的定量矩阵或长列表”这一条件。
- 对 ScpTensor 来说，最有价值的是它们的 `source data / supplementary data / processed tables`，不是 raw 文件本身。
- 如果后续要做自动化导入，优先围绕这些表格入口写下载脚本和格式检测，不要把 raw-only 资源混进主线。
