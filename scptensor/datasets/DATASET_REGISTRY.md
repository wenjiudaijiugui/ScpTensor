# ScpTensor 数据集注册表

**Version:** 1.0
**Last Updated:** 2025-01-12
**Source:** TencentAILabHealthcare/scPROTEIN GitHub Repository

---

## 数据集概览

| 类别 | 数据集 | 规模 | 文件大小 | 格式 |
|------|--------|------|----------|------|
| SCoPE2 | Peptides-raw.csv | 9355 行 | 80 MB | CSV |
| plexDIA | plexDIA_data.csv | ~165 行 | 2.2 MB | CSV |
| pSCoPE | pSCoPE_Leduc_data.csv | ~1544 行 | 28 MB | CSV |
| pSCoPE | pSCoPE_Huffman_data.csv | ~207 行 | 2.4 MB | CSV |
| nanoPOTS | nanoPOTS_data.csv | 62 行 | 396 KB | CSV |
| N2 | N2_data.csv | 109 行 | 1.3 MB | CSV |
| 空间蛋白质组学 | BaselTMA_SP41_83_X14Y2.csv | 2043 行 | 1.5 MB | CSV |
| 细胞周期 | T-SCP_data.csv | 226 行 | 5.5 MB | CSV |
| 临床蛋白质组学 | ECCITE_seq_processed_data.csv | 5402 行 | 2.3 MB | CSV |

**总计数据量:** ~150 MB

---

## 数据集详情

### 1. SCoPE2 (Single-Cell ProtEomics v2)

**目录:** `sccope/`

| 文件 | 描述 |
|------|------|
| `Peptides-raw.csv` | 肽段 x 单细胞矩阵，1% FDR |
| `Cells.csv` | 细胞元数据 (Annotation x 单细胞) |

**数据特征:**
- 肽段级别数据
- 经过 MaxQuant 和 DART-ID 处理
- 适合归一化、插补、批次校正算法验证

**文献:** Specht et al., Genome Biology 2021

---

### 2. plexDIA (Multiplexed DIA)

**目录:** `plexdia/`

| 文件 | 描述 |
|------|------|
| `plexDIA_data.csv` | 蛋白质定量数据 |
| `plexDIA_cells.csv` | 细胞元数据 |

**数据特征:**
- DIA 方法获取
- 约 1000 蛋白质/细胞
- 98% 数据完整性

**文献:** Derks et al., Nature Methods 2022

---

### 3. pSCoPE (parallel SCoPE)

**目录:** `pscope/`

| 文件 | 描述 |
|------|------|
| `pSCoPE_Leduc_data.csv` | Leduc 实验数据 (28 MB) |
| `pSCoPE_Leduc_cells.csv` | Leduc 细胞元数据 |
| `pSCoPE_Huffman_data.csv` | Huffman 实验数据 (2.4 MB) |
| `pSCoPE_Huffman_cells.csv` | Huffman 细胞元数据 |

**数据特征:**
- 两个独立实验数据
- 适合批次校正算法验证

---

### 4. Integration 数据集

**目录:** `integration/`

| 文件 | 描述 |
|------|------|
| `nanoPOTS_data.csv` | nanoPOTS 平台数据 |
| `nanoPOTS_cells.csv` | nanoPOTS 细胞元数据 |
| `N2_data.csv` | N2 数据集 |
| `N2_cells.csv` | N2 细胞元数据 |
| `SCoPE2_Leduc_data.csv` | SCoPE2 Leduc 数据 |
| `SCoPE2_Leduc_cells.csv` | SCoPE2 Leduc 细胞元数据 |

**用途:** 跨平台数据整合和批次校正

---

### 5. 空间蛋白质组学

**目录:** `spatial/`

| 文件 | 描述 |
|------|------|
| `BaselTMA_SP41_83_X14Y2.csv` | 组织微阵列空间蛋白质组数据 |

**数据特征:**
- 空间位置信息
- 83 个蛋白质标记
- 适合空间分析算法

---

### 6. 细胞周期分析

**目录:** `cell_cycle/`

| 文件 | 描述 |
|------|------|
| `T-SCP_data.csv` | 细胞周期蛋白质组数据 (5.5 MB) |
| `T-SCP_cells.csv` | 细胞周期元数据 |

**用途:** 细胞周期相关蛋白质分析

---

### 7. 临床蛋白质组学

**目录:** `clinical/`

| 文件 | 描述 |
|------|------|
| `ECCITE_seq_processed_data.csv` | ECCITE-seq 蛋白质组数据 (2.3 MB) |
| `ECCITE_seq_processed_cells.csv` | ECCITE-seq 细胞元数据 |

**用途:** 临床应用、CRISPR 筛选验证

---

## 数据格式说明

### 数据矩阵格式

大多数数据集遵循以下格式:

```csv
Protein,Peptide,Cell_0,Cell_1,Cell_2,...
P08865,LLVVTDPR_2,0.215,1.825,0.171,...
P26447,RTDEAAFQK_3,1.873,1.425,2.354,...
...
```

- `Protein`: 蛋白质标识符 (UniProt ID)
- `Peptide`: 肽段序列
- `Cell_N`: 第 N 个细胞的定量值
- `NA`: 缺失值

### 细胞元数据格式

```csv
,Cell_0,Cell_1,Cell_2,...
cell_type,Monocyte,Macrophage,T_Cell,...
batch,B1,B1,B2,...
...
```

---

## 使用建议

### 按分析类型选择数据集

| 分析类型 | 推荐数据集 |
|----------|-----------|
| 归一化 | SCoPE2, plexDIA |
| 插补 | SCoPE2 (高缺失率) |
| 批次校正 | pSCoPE (Leduc + Huffman), integration/ |
| 聚类 | plexDIA, cell_cycle/ |
| 差异分析 | clinical/ |
| 空间分析 | spatial/ |

### 性能测试

| 测试类型 | 推荐数据集 |
|----------|-----------|
| 小规模快速测试 | nanoPOTS, N2 |
| 中等规模 | plexDIA |
| 大规模测试 | pSCoPE_Leduc (28 MB), SCoPE2 (80 MB) |

---

## 数据来源

**主来源:** [TencentAILabHealthcare/scPROTEIN](https://github.com/TencentAILabHealthcare/scPROTEIN)

**原始文献:**
- Li W, Yang F, Wang F, et al. scPROTEIN: a versatile deep graph contrastive learning framework for single-cell proteomics embedding. Nature Methods, 2024.

---

## 引用

如果使用这些数据集，请引用:

1. scPROTEIN: Li et al., Nature Methods 2024
2. SCoPE2: Specht et al., Genome Biology 2021
3. plexDIA: Derks et al., Nature Methods 2022
4. 原始数据文献 (见各数据集详情)

---

## 更新日志

| 日期 | 更新内容 |
|------|----------|
| 2025-01-12 | 初始版本，添加 scPROTEIN 数据集 |

---

**维护者:** ScpTensor Team
