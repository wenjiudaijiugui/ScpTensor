# ScpTensor 验证数据集获取规划

**Version:** 1.0
**Last Updated:** 2025-01-12
**Purpose:** 为 ScpTensor 测试和验证提供多样化的单细胞蛋白质组学数据集

---

## 目录

1. [数据集优先级矩阵](#数据集优先级矩阵)
2. [ProteomeXchange/PRIDE 数据集](#proteomeexchangepride-数据集)
3. [MassIVE 数据集](#massive-数据集)
4. [文献补充数据](#文献补充数据)
5. [数据获取方式](#数据获取方式)
6. [数据组织结构](#数据组织结构)
7. [使用建议](#使用建议)

---

## 数据集优先级矩阵

### 按用途分类

| 优先级 | 数据集 | 规模 | 用途 | 状态 |
|--------|--------|------|------|------|
| 🔴 P0 | PXD010710 | ~50 细胞 | 首批大规模 SCP，领域基准 | 待获取 |
| 🔴 P0 | PXD014894 | ~100 细胞 | 多组织数据 | 待获取 |
| 🟡 P1 | PXD021010 | ~200 细胞 | 不同平台验证 | 待获取 |
| 🟡 P1 | PXD019764 | ~500 细胞 | 大规模数据 | 待获取 |
| 🟡 P1 | PXD040141 | ~1000 细胞 | 超大规模数据 | 待获取 |
| 🟢 P2 | PXD032230 | 肿瘤微环境 | 特定生物学场景 | 待获取 |
| 🟢 P2 | PXD044546 | 免疫细胞 | 免疫学应用 | 待获取 |
| 🟢 P2 | MSV000086535 | 配套数据 | 与 PRIDE 互补 | 待获取 |

### 按数据特征分类

| 特征 | 数据集 | 说明 |
|------|--------|------|
| **多组织** | PXD014894, PXD019764 | 涵盖多种组织类型 |
| **多细胞类型** | PXD010710, PXD044546 | 包含多种细胞亚群 |
| **大规模** | PXD040141, PXD019764 | >500 单细胞 |
| **标准基准** | PXD010710 | Brunner et al. 2020 数据 |

---

## ProteomeXchange/PRIDE 数据集

### P0 - 核心基准数据集

#### 1. PXD010710 ⭐ 领域基准
- **标题:** Single-Cell Proteomics Data Analysis Workflow for Mass Cytometry
- **文献:** Brunner et al., Nature Biotechnology 2020
- **规模:** ~50 单细胞
- **特点:**
  - 首批大规模单细胞蛋白质组学数据
  - 包含 T 细胞亚群
  - 详细的实验设计
  - 广泛被后续研究引用
- **获取方式:**
  ```bash
  # PRIDE FTP
  ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2020/06/PXD010710/
  ```
- **文件:** RAW 文件, 搜索结果, 元数据
- **用途:** 归一化、批次校正、聚类算法验证

#### 2. PXD014894 ⭐ 多组织数据
- **标题:** Single-cell proteomics reveals functionally distinct cell states in the human pancreas
- **文献:** Lombardi et al., Nature Communications 2021
- **规模:** ~100 单细胞
- **特点:**
  - 胰腺组织
  - 内分泌细胞亚群
  - 包含健康和疾病状态
- **获取方式:**
  ```bash
  ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2020/12/PXD014894/
  ```
- **用途:** 差异分析、细胞类型鉴定

### P1 - 扩展验证数据集

#### 3. PXD021010
- **标题:** Ultra-high sensitivity single-cell proteomics using multiplexed data-independent acquisition
- **文献:** Ludwig et al., MCP 2021
- **规模:** ~200 单细胞
- **特点:** 不同平台/方法的数据
- **获取:**
  ```bash
  ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2021/03/PXD021010/
  ```

#### 4. PXD019764 📊 大规模数据
- **标题:** High-throughput single-cell proteomics using isobaric labeling
- **文献:** Sridharan et al., 2021
- **规模:** ~500 单细胞
- **特点:** 大规模数据，适合性能测试
- **获取:**
  ```bash
  ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2021/01/PXD019764/
  ```

#### 5. PXD040141 📊 超大规模数据
- **标题:** Deep profiling of single cells using ultra-high-field mass spectrometry
- **规模:** ~1000+ 单细胞
- **特点:** 当前最大规模 SCP 数据集之一
- **获取:**
  ```bash
  ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2022/10/PXD040141/
  ```

### P2 - 特定应用场景

#### 6. PXD032230 - 肿瘤微环境
- **标题:** Single-cell proteomics of tumor microenvironment
- **规模:** ~100 细胞
- **特点:** 肿瘤相关成纤维细胞、免疫细胞
- **获取:**
  ```bash
  ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2022/06/PXD032230/
  ```

#### 7. PXD044546 - 免疫细胞
- **标题:** Single-cell proteomics reveals immune cell heterogeneity
- **规模:** ~150 细胞
- **特点:** 免疫细胞亚群精细分类
- **获取:**
  ```bash
  ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2022/12/PXD044546/
  ```

---

## MassIVE 数据集

MassIVE (Mass spectrometry Interactive Virtual Environment) 是另一个重要数据仓库：

### MSV000086535
- **关联:** 与 PXD010710 配套
- **内容:** 额外的分析文件和结果
- **获取:**
  ```bash
  # MassIVE 下载
  https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?accession=MSV000086535
  ```

### MSV000087844
- **标题:** Single-cell proteomics benchmarking
- **规模:** ~200 细胞
- **特点:** 包含多种分析方法的对比
- **获取:**
  ```bash
  https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?accession=MSV000087844
  ```

---

## 文献补充数据

许多论文在 GitHub 或期刊补充材料中提供处理后的数据：

### 1. SCOPE (Single-Cell Omics Data Explorer)
- **URL:** https://scope.slab.hku.hk/
- **内容:** 整合的单细胞多组学数据集
- **格式:** 预处理的矩阵格式

### 2. scPDB (Single Cell Proteomics Database)
- **URL:** 待发布 (关注相关期刊)
- **内容:** 专门的 SCP 数据库

### 3. GitHub 仓库
- **dpq/pSCoPE:** https://github.com/dpq/pSCoPE
- **Mann Labs:** 各论文的补充数据仓库

---

## 数据获取方式

### 方法一：PRIDE API (推荐用于自动化)

```python
import requests

def get_pride_dataset(accession: str, output_dir: str):
    """
    从 PRIDE 下载数据集

    Parameters
    ----------
    accession : str
        数据集 ID (如 "PXD010710")
    output_dir : str
        输出目录
    """
    base_url = f"https://www.ebi.ac.uk/pride/ws/archive/v2/files/{accession}"
    response = requests.get(base_url)
    files = response.json()

    for file_info in files:
        download_url = file_info['downloadLink']
        filename = file_info['fileName']
        # 下载文件...
```

### 方法二：FTP 批量下载

```bash
# 使用 wget 批量下载
wget -r -np -nH --cut-dirs=3 ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2020/06/PXD010710/

# 使用 lftp (支持断点续传)
lftp -c "mirror -c --parallel=3 ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2020/06/PXD010710/ ./data/PXD010710/"
```

### 方法三：Aspera Connect (高速下载)

PRIDE 支持 Aspera 高速下载协议：

```bash
# 安装 Aspera Connect 后
ascp -QT -l 100M -P 33001 \
  era-fasp@fasp.ebi.ac.uk:pride/data/archive/2020/06/PXD010710/ \
  ./data/PXD010710/
```

---

## 数据组织结构

建议目录结构：

```
scptensor/datasets/
├── raw/                      # 原始数据 (RAW 文件)
│   ├── PXD010710/
│   ├── PXD014894/
│   └── ...
├── processed/                # 预处理数据 (矩阵格式)
│   ├── PXD010710/
│   │   ├── expression.tsv    # 表达矩阵
│   │   ├── obs.tsv           # 样本元数据
│   │   ├── var.tsv           # 蛋白质元数据
│   │   └── metadata.json     # 数据集描述
│   └── ...
├── scptensor/                # ScpTensor 格式
│   ├── PXD010710.npz
│   ├── PXD014894.npz
│   └── ...
└── registry.json             # 数据集注册表
```

### registry.json 格式

```json
{
  "datasets": [
    {
      "accession": "PXD010710",
      "name": "Brunner_2020_TCells",
      "size": 50,
      "n_proteins": 2000,
      "tissue": "blood",
      "cell_types": ["T_CD4", "T_CD8", "T_Reg"],
      "platform": "TMT-10plex",
      "citation": "Brunner et al., Nat Biotech 2020",
      "doi": "10.1038/s41587-020-0602-7",
      "local_path": "scptensor/PXD010710.npz",
      "status": "processed"
    }
  ]
}
```

---

## 使用建议

### 1. 分阶段获取

**第一阶段 (P0 - 基准数据):**
- PXD010710 (Brunner 2020) - 核心基准
- PXD014894 (Lombardi 2021) - 多组织
- PXD061065 (已有) - 现有数据

**第二阶段 (P1 - 扩展验证):**
- PXD021010 - 平台对比
- PXD019764 - 大规模
- PXD040141 - 超大规模

**第三阶段 (P2 - 特定场景):**
- PXD032230 - 肿瘤微环境
- PXD044546 - 免疫细胞

### 2. 数据预处理流程

原始数据 → ScpTensor 格式转换：

```
RAW 文件
    ↓ (MSFragger/DIA-NN/etc)
蛋白质定量结果
    ↓ (处理脚本)
表达矩阵 + 元数据
    ↓ (scptensor.core.io)
ScpContainer (.npz)
```

### 3. 测试策略

| 测试类型 | 使用数据集 | 验证内容 |
|----------|-----------|----------|
| **单元测试** | toy_example | 基本功能 |
| **集成测试** | PXD010710 | 端到端流程 |
| **性能测试** | PXD040141 | 大规模数据处理 |
| **算法对比** | PXD014894, PXD021010 | 与文献结果对比 |
| **特定场景** | PXD032230, PXD044546 | 生物学应用 |

---

## 数据质量检查清单

获取数据后，验证：

- [ ] 数据完整性（文件大小、校验和）
- [ ] 元数据完整性（样本信息、实验设计）
- [ ] 缺失值比例和分布
- [ ] 数据格式一致性
- [ ] 与原始论文结果一致性

---

## 参考资源

### 数据仓库
- **PRIDE:** https://www.ebi.ac.uk/pride/archive/
- **MassIVE:** https://massive.ucsd.edu/
- **ProteomeXchange:** http://proteomecentral.proteomexchange.org/

### 工具
- **PRIDE API:** https://www.ebi.ac.uk/pride/ws/archive/v2/
- **Aspera Connect:** https://www.ibm.com/aspera/connect/

### 文献
- Brunner et al., Nature Biotechnology 2020 (PXD010710)
- Lombardi et al., Nature Communications 2021 (PXD014894)
- Ludwig et al., MCP 2021 (PXD021010)

---

**维护说明:** 当有新的重要 SCP 数据集发布时，请更新本文档。

**贡献:** 如果您发现其他有用的数据集，请添加到本文档并提交 PR。
