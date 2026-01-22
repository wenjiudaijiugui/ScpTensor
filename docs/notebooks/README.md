# ScpTensor Tutorial Notebooks

本目录包含交互式Jupyter notebook，演示ScpTensor单细胞蛋白质组学分析框架。

## 快速开始

### 1. 安装ScpTensor

```bash
# 克隆仓库
git clone https://github.com/your-org/ScpTensor.git
cd ScpTensor

# 使用uv安装（推荐）
uv pip install -e .

# 或使用pip
pip install -e .
```

### 2. 安装Jupyter

```bash
# JupyterLab（推荐）
pip install jupyterlab

# 或经典Jupyter Notebook
pip install notebook
```

### 3. 启动Jupyter

```bash
# 从项目根目录
jupyter lab

# 然后导航到 docs/notebooks/ 并打开notebook
```

## 可用教程

| Notebook | 主题 | 预计时间 | 状态 |
|----------|------|----------|------|
| **01_basic_workflow.ipynb** | QC与归一化完整工作流程 | 30分钟 | ✅ 最新 |

## Notebook详情

### 01_basic_workflow.ipynb

**完整的质控与归一化工作流程**，涵盖：

#### 第一部分：质控(QC)
1. **PSM层级质控**
   - 过滤污染物（角蛋白、胰蛋白酶等）
   - PIF过滤（母离子纯度）

2. **样本层级质控**
   - 计算QC指标（特征数、总强度）
   - 过滤低质量样本（基于MAD）
   - 过滤双细胞（检测右侧离群值）

3. **特征层级质控**
   - 计算QC指标（缺失率、CV）
   - 过滤高缺失率蛋白
   - 过滤高变异系数蛋白

#### 第二部分：归一化
1. **对数归一化** - 必须步骤，稳定方差
2. **中位数归一化** - 推荐方法，对异常值鲁棒
3. **均值归一化** - 可选方法，对异常值敏感
4. **分位数归一化** - 高级方法，强制相同分布

**适用对象**: 所有ScpTensor用户

**学习成果**:
- ✅ 理解SCP数据的QC流程
- ✅ 掌握不同的归一化方法
- ✅ 学会选择合适的参数
- ✅ 了解最佳实践

## Notebook特点

1. **自包含**: 所有notebook使用模拟数据，无需外部文件
2. **可执行**: 逐cell运行，每个输出都有说明
3. **可视化**: 包含多个出版质量图表（SciencePlots, 300 DPI）
4. **中文注释**: 详细的中文代码注释和说明
5. **最佳实践**: 展示推荐的参数和工作流程

## 系统要求

- **Python**: 3.11或更高
- **内存**: 至少4GB RAM
- **依赖**: 自动安装（随ScpTensor）

## 依赖包

```
numpy
polars
scipy
scikit-learn
matplotlib
scienceplots
```

## 常见问题

### 导入错误

如果看到 `ModuleNotFoundError: No module named 'scptensor'`:
```bash
# 重新安装ScpTensor（开发模式）
pip install -e .
```

### 缺少SciencePlots

如果看到样式错误：
```bash
pip install scienceplots
```

### Kernel问题

如果Jupyter找不到Python kernel：
```bash
# 安装ipykernel
pip install ipykernel

# 注册环境
python -m ipykernel install --user --name=scptensor
```

## 其他资源

- **API文档**: [API Reference](../api/)
- **设计文档**: [Design Docs](../design/)
- **GitHub仓库**: [ScpTensor GitHub](https://github.com/your-org/ScpTensor)
- **问题追踪**: [ISSUES_AND_LIMITATIONS.md](../ISSUES_AND_LIMITATIONS.md)

## 引用

如果您在研究中使用了ScpTensor，请引用：
```
ScpTensor: A Framework for Single-Cell Proteomics Analysis
[Authors et al., Year]
```

## 支持

遇到问题？
- 在GitHub上提issue
- 查看[设计文档](../design/)了解架构细节
- 参考[ISSUES_AND_LIMITATIONS.md](../ISSUES_AND_LIMITATIONS.md)了解已知问题

---

**最后更新**: 2026-01-21
**版本**: v0.1.0-beta
