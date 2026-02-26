# 数据集完整审查报告

**生成时间**: 2026-02-26
**评分标准**: 数据完整性(40%) > 仪器类型(30%) > 发表质量(20%) > 样本规模(10%)
**下载阈值**: ≥60分

---

## 摘要

| 项目 | 数量 |
|------|------|
| 搜索候选 | 68 |
| 已审查 | 68 |
| Orbitrap Astral | 13 |
| 符合条件 | 3 |
| 已下载 | 3 |

---

## 已下载数据集 (3个)

### 1. PXD064564 (评分: 95/100) ✓
- **标题**: Challenging the Astral mass analyzer – up to 5300 proteins per single-cell
- **仪器**: Orbitrap Astral ✓
- **数据**: Spectronaut pivot, 137.85 MB, 165 samples ✓
- **发表**: Nature Methods 2025, DOI: 10.1038/s41592-024-02559-1 ✓
- **下载**: 2026-02-25

### 2. PXD049412 (评分: 90/100) ✓
- **标题**: Challenging the Astral mass analyzer（同论文）
- **仪器**: Orbitrap Astral ✓
- **数据**: Spectronaut pivot, 36.20 MB, 15 samples ✓
- **发表**: Nature Methods 2025, DOI: 10.1038/s41592-024-02559-1 ✓
- **下载**: 2026-02-26

### 3. PXD049211 (评分: 70/100) ✓
- **标题**: Deep single-cell proteomics quantifies low-abundant embryonic transcription factors
- **仪器**: Orbitrap Astral ✓
- **数据**: DIA-NN BGS Factory, 724 MB ✓
- **发表**: 无 DOI (0分)
- **下载**: 2026-02-26

---

## Orbitrap Astral 数据集完整列表 (13个)

| 数据集 | 状态 | 文件大小 | 跳过原因 |
|--------|------|----------|----------|
| PXD064564 | ✓ 已下载 | 137 MB | - |
| PXD049412 | ✓ 已下载 | 36 MB | - |
| PXD049211 | ✓ 已下载 | 724 MB | - |
| PXD054445 | ✗ 跳过 | 8.8 GB | 文件过大 |
| PXD054083 | ✗ 跳过 | 4.0 GB | 文件过大 |
| PXD054066 | ✗ 跳过 | 4.4 GB | 文件过大 |
| PXD051942 | ✗ 跳过 | 18 GB | 文件过大 |
| PXD062231 | ✗ 跳过 | - | 无 TSV 文件 |
| PXD061065 | ✗ 跳过 | - | 无 TSV 文件 |
| PXD058753 | ✗ 跳过 | - | 无 TSV 文件 |
| PXD056327 | ✗ 跳过 | - | 无 TSV 文件 |
| PXD049181 | ✗ 跳过 | - | 无 TSV 文件 |
| PXD046357 | ✗ 跳过 | - | 无 TSV 文件 |

---

## 其他值得注意的数据集

### timsTOF 系列 (有 DIA-NN 输出，但非 Astral)

| 数据集 | 仪器 | 评分 | 决定 |
|--------|------|------|------|
| PXD058457 | timsTOF SCP | 50 | 仪器不匹配 |
| PXD069842 | timsTOF HT | 30 | 仪器不匹配 |
| PXD059079 | Orbitrap Ascend/timsTOF HT | 40 | 混合仪器 |
| PXD062696 | timsTOF Pro 2 | 50 | 仪器不匹配 |
| PXD061287 | timsTOF | 40 | 仪器不匹配 |

---

## 数据分布分析

### 仪器分布 (68个候选)

| 仪器类型 | 数量 | 占比 |
|----------|------|------|
| Orbitrap Astral | 13 | 19% |
| timsTOF 系列 | 35 | 51% |
| Orbitrap Exploris | 8 | 12% |
| 其他 | 12 | 18% |

### 下载率

- Astral 数据集下载率: 3/13 (23%)
  - 主要原因: 6个无TSV文件，4个文件过大

---

## 结论

1. **成功下载**: 3个高质量 Orbitrap Astral 单细胞蛋白质组学数据集
2. **总数据量**: 898 MB
3. **覆盖范围**: 从 Nature Methods 论文到胚胎发育研究
4. **主要障碍**:
   - 46% 的 Astral 数据集无可下载的 TSV 文件
   - 31% 的 Astral 数据集文件过大 (>1GB)
5. **建议**:
   - 可考虑联系作者获取未上传的文件
   - 可考虑扩展到 timsTOF SCP 数据集（同等质量）
   - 可设置文件大小阈值来筛选数据集

---

**报告生成**: ScpTensor 自动审查系统
**审查日期**: 2026-02-26
