# DIA 单细胞蛋白质组学数据集

**Version:** 1.0
**Last Updated:** 2025-01-12
**Data Source:** PXD054343 - PRIDE ProteomeXchange

---

## 数据集概览

| 文件 | 描述 | 大小 | 行数 | 格式 |
|------|------|------|------|------|
| `1_SC_LF_report.tsv` | 单细胞无标记定量数据 | 212 MB | 156,805 | DIA-NN report |
| `2_1_SC_SILAC_20x_report.tsv` | 单细胞 SILAC 20倍稀释数据 | 7.3 MB | 4,199 | DIA-NN report |
| `2_2_SC_SILAC_5x_report.tsv` | 单细胞 SILAC 5倍稀释数据 | 7.3 MB | 4,494 | DIA-NN report |
| `2_3_SC_SILAC_2x_report.tsv` | 单细胞 SILAC 2倍稀释数据 | 7.0 MB | 4,529 | DIA-NN report |

**总数据量:** ~234 MB

---

## 数据集详情

### PXD054343: DIA-SiS (DIA with Spike-in SILAC)

**标题:** DIA Spike-in SILAC Revisions; additional submission including single-cell like amounts

**文献:** Welter AS, Gerwien M, Kerridge R, et al. Combining Data Independent Acquisition With Spike-In SILAC (DIA-SiS) Improves Proteome Coverage and Quantification. Mol Cell Proteomics, 2024

**数据特点:**
- DIA 方法获取
- 包含单细胞量级数据 (300ng)
- 使用 SILAC 标记进行定量
- 包含多种稀释比例 (2x, 5x, 20x)

**仪器:**
- timsTOF Pro 2
- Orbitrap Exploris 480

---

## DIA-NN report.tsv 格式

### 标准列 (DIA-NN 输出格式)

| 列名 | 描述 |
|------|------|
| `File.Name` | 原始文件路径 |
| `Run` | 运行标识符 |
| `Protein.Group` | 蛋白质组标识 |
| `Protein.Ids` | 蛋白质 ID (UniProt) |
| `Protein.Names` | 蛋白质名称 |
| `Genes` | 基因名称 |
| `PG.Quantity` | 蛋白质组定量值 |
| `PG.Normalised` | 归一化蛋白质组定量 |
| `Precursor.Quantity` | 前体离子定量值 |
| `Precursor.Normalised` | 归一化前体定量 |
| `Modified.Sequence` | 修饰肽段序列 |
| `Stripped.Sequence` | 裸肽段序列 |
| `Precursor.Charge` | 前体电荷 |
| `Q.Value` | Q值 (错误发现率) |
| `RT` | 保留时间 |
| `iRT` | 预测保留时间 |
| `Lib.Q.Value` | 库Q值 |
| `MS2.Scan` | MS2扫描号 |
| `Precursor.Mz` | 前体质荷比 |
| `IM` | 离子淌度 |
| `Fragment.Info` | 碎片信息 |

### 示例行

```tsv
File.Name    Run    Protein.Group    Protein.Ids    PG.Quantity    Precursor.Quantity    RT    Q.Value
...           ...    A6NIH7;...       A6NIH7;...      485.02         485.02                9.19  0.0026
```

---

## 数据获取方式

### FTP 下载

```bash
# 基础 URL
ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2024/10/PXD054343/

# 下载单个文件
curl -O ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2024/10/PXD054343/1_SC_LF_report.tsv
```

### 使用 Python 读取

```python
import polars as pl

# 读取 DIA-NN report
report = pl.read_csv("1_SC_LF_report.tsv", separator="\t")

# 提取蛋白质定量数据
protein_cols = ["Protein.Group", "PG.Quantity", "Genes", "Run"]
protein_data = report.select(protein_cols)

# 过滤高可信度数据
filtered = report.filter(pl.col("Q.Value") < 0.01)
```

---

## 数据使用建议

### 按分析类型选择数据

| 分析类型 | 推荐数据集 | 说明 |
|----------|-----------|------|
| 蛋白质定量 | `1_SC_LF_report.tsv` | 大规模无标记数据 |
| 稀释系列分析 | `2_1_SC_SILAC_20x_report.tsv` | 高稀释倍数 |
| SILAC 定量 | `2_*_SC_SILAC_*.tsv` | SILAC 标记数据 |
| 方法比较 | 全部 | 比较不同定量方法 |

### 数据过滤建议

```python
# 推荐的过滤条件
filtered = report.filter(
    (pl.col("Q.Value") < 0.01) &           # FDR < 1%
    (pl.col("Protein.Q.Value") < 0.01) &   # 蛋白质 FDR < 1%
    (pl.col("PG.Quantity") > 0)            # 有效定量值
)
```

---

## 引用

如果使用此数据集，请引用:

> Welter AS, Gerwien M, Kerridge R, Alp KM, Mertins P, Selbach M. Combining Data Independent Acquisition With Spike-In SILAC (DIA-SiS) Improves Proteome Coverage and Quantification. Mol Cell Proteomics. 2024;23(10):100839. doi:10.1016/j.mcpro.2024.100839

---

## 相关资源

- **PRIDE 数据集:** https://www.ebi.ac.uk/pride/archive/projects/PXD054343
- **FTP 地址:** ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2024/10/PXD054343/
- **DIA-NN 软件:** https://github.com/vdemichev/DiaNN
- **文献 DOI:** 10.1016/j.mcpro.2024.100839

---

## 更新日志

| 日期 | 更新内容 |
|------|----------|
| 2025-01-12 | 初始版本，添加 PXD054343 DIA-NN report 文件 |

---

**维护者:** ScpTensor Team
**数据许可:** 请参考 PRIDE 数据集使用条款
