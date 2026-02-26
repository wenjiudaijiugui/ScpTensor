# 数据集自动审查报告

**生成时间**: 2026-02-26
**评分标准**: 数据完整性(40%) > 仪器类型(30%) > 发表质量(20%) > 样本规模(10%)
**下载阈值**: >=60分

---

## 摘要

| 项目 | 数量 |
|------|------|
| 搜索候选 | 68 |
| 已审查 | 10 |
| 符合条件 | 2 |
| 已下载 | 2 |

---

## 已下载数据集

### 1. PXD064564 (评分: 95/100) OK
- **标题**: Challenging the Astral mass analyzer - up to 5300 proteins per single-cell
- **仪器**: Orbitrap Astral OK (30分)
- **数据**: Spectronaut pivot 报告, 137.85 MB OK (40分)
- **发表**: Nature Methods 2025, DOI: 10.1038/s41592-024-02559-1 OK (20分)
- **样本**: 165 个单细胞 (10分)
- **下载日期**: 2026-02-25

### 2. PXD049412 (评分: 90/100) OK
- **标题**: Challenging the Astral mass analyzer（同论文）
- **仪器**: Orbitrap Astral OK (30分)
- **数据**: Spectronaut pivot 报告, 36.20 MB OK (40分)
- **发表**: Nature Methods 2025, DOI: 10.1038/s41592-024-02559-1 OK (20分)
- **样本**: 15 个单细胞 (5分)
- **下载日期**: 2026-02-26

---

## 跳过的数据集

### PXD058457 (评分: 50/100) X
- **标题**: High throughput single-cell proteomics of in-vivo cells
- **跳过原因**: 仪器不匹配 (timsTOF SCP, 0分)
- **数据**: 6个 DIA-NN pg_matrix 文件 OK
- **发表**: DOI: 10.1016/J.MCPRO.2025.101018

### PXD069842 (评分: 30/100) X
- **标题**: Colonic spatial single-cell proteomics...
- **跳过原因**: 仪器不匹配 (timsTOF HT) + 无发表信息
- **数据**: 1个 DIANN_report 文件

### PXD067019 (评分: 40/100) X
- **标题**: Fine-Tuning of Label-Free Single-Cell Proteomics Workflows
- **跳过原因**: 无 Astral 文件，仅有 1 个 Search_report

### PXD046357 (评分: 20/100) X
- **标题**: Ultra-fast label-free quantification...
- **跳过原因**: 无报告文件，无 Astral

### PXD071075 (评分: 20/100) X
- **标题**: Single-cell proteomic landscape of the developing human brain
- **跳过原因**: 无报告文件，无 Astral

### PXD063590 (评分: 20/100) X
- **标题**: Single cell proteomic analysis of human prefrontal cortex in Alzheimer's Disease
- **跳过原因**: 无报告文件，无 Astral

---

## 待人工审核

以下数据集接近阈值，可能需要人工判断：

| 数据集 | 评分 | 备注 |
|--------|------|------|
| PXD061065 | 30 | 有 Astral 但无报告文件 |

---

## 结论

1. **成功下载**: 2个高质量 Orbitrap Astral 单细胞蛋白质组学数据集
2. **共同特点**: 均来自 Nature Methods 2025 论文，使用 Spectronaut 分析
3. **主要问题**: 大多数候选数据集使用 timsTOF 系列仪器，非 Orbitrap Astral
4. **建议**: 可考虑扩展仪器类型要求，纳入 timsTOF SCP 数据

---

**报告生成**: ScpTensor 自动审查系统
