# 自动方法选择系统设计文档

**日期**: 2026-03-01
**版本**: v1.0
**状态**: 已批准

---

## 1. 概述

### 1.1 创新点

ScpTensor 的核心创新点：**为不同数据集自动选择最优分析方法**。

传统单细胞蛋白质组学分析工具需要用户手动选择每个环节的方法，但：
- 不同数据集的最优方法可能不同
- 用户难以判断哪个方法最适合自己的数据
- 方法选择需要专业知识和经验

本系统通过自动运行所有可用方法并评估结果，为每个数据集选择最优方案。

### 1.2 目标

1. **提高分析质量** - 自动选择最适合数据特性的方法
2. **降低使用门槛** - 用户无需深入了解每个方法的原理
3. **提供透明度** - 生成详细报告说明选择依据
4. **保持灵活性** - 支持用户自定义权重和偏好

---

## 2. 架构设计

### 2.1 模块结构

```
scptensor/autoselect/
├── __init__.py              # 导出 AutoSelector, evaluate_all
├── core.py                  # AutoSelector 主类
├── evaluators/
│   ├── __init__.py
│   ├── base.py              # BaseEvaluator 抽象类
│   ├── normalization.py     # 归一化评估器
│   ├── imputation.py        # 填充评估器
│   ├── integration.py       # 批次校正评估器
│   ├── dim_reduction.py     # 降维评估器
│   └── clustering.py        # 聚类评估器
├── metrics/
│   ├── __init__.py
│   ├── quality.py           # 通用质量指标 (CV, MAD, sparsity)
│   ├── batch.py             # 批次效应指标 (ASW, LISI)
│   └── clustering.py        # 聚类指标 (silhouette, ARI)
└── report.py                # 评估报告生成
```

### 2.2 核心类设计

```python
@dataclass
class EvaluationResult:
    """单个方法的评估结果"""
    method_name: str
    scores: dict[str, float]      # 各指标得分
    overall_score: float          # 加权综合得分
    execution_time: float         # 运行时间
    layer_name: str               # 结果存储的层名

@dataclass
class StageReport:
    """单个环节的评估报告"""
    stage_name: str
    results: list[EvaluationResult]
    best_method: str
    best_result: EvaluationResult
    recommendation_reason: str

class AutoSelector:
    """统一自动方法选择器"""

    def __init__(
        self,
        stages: list[str] | None = None,  # 要优化的环节
        strategy: str = "comprehensive",   # 评估策略
        keep_all: bool = False,            # 是否保留所有结果
        weights: dict[str, float] | None = None,  # 指标权重
        parallel: bool = True,             # 并行执行
        n_jobs: int = -1,                  # 并行进程数
    ): ...

    def run(self, container: ScpContainer) -> tuple[ScpContainer, AutoSelectReport]:
        """执行全流程自动选择"""
        ...

    def run_stage(
        self,
        container: ScpContainer,
        stage: str
    ) -> tuple[ScpContainer, StageReport]:
        """执行单个环节的自动选择"""
        ...
```

---

## 3. 各环节评估指标

### 3.1 归一化 (Normalization)

| 指标 | 说明 | 权重 |
|------|------|------|
| CV稳定性 | 变异系数分布的稳定性 | 0.25 |
| 偏度改善 | 数据分布偏度接近0的程度 | 0.20 |
| 峰度改善 | 数据分布峰度接近正态 | 0.15 |
| 动态范围 | 保留的动态范围合理性 | 0.20 |
| 异常值比例 | 极端值的减少程度 | 0.20 |

### 3.2 缺失值填充 (Imputation)

| 指标 | 说明 | 权重 |
|------|------|------|
| 恢复精度 | 模拟缺失值的恢复准确度 (RMSE) | 0.30 |
| 分布一致性 | 填充后分布与观测分布的一致性 | 0.25 |
| 生物学相关性 | 填充后特征间相关性的合理性 | 0.25 |
| 噪声引入 | 是否引入过多人工噪声 | 0.20 |

### 3.3 批次校正 (Integration)

| 指标 | 说明 | 权重 |
|------|------|------|
| ASW (批次) | 批次轮廓系数，越低越好 | 0.25 |
| ASW (生物学) | 生物学分组轮廓系数，越高越好 | 0.25 |
| LISI (批次) | 批次混合度，越高越好 | 0.25 |
| 变异保留 | 生物学变异保留程度 | 0.25 |

### 3.4 降维 (Dimensionality Reduction)

| 指标 | 说明 | 权重 |
|------|------|------|
| 方差解释率 | 前N个成分解释的方差比例 | 0.30 |
| 重构误差 | 降维后重构的误差 | 0.25 |
| 局部结构保留 | k近邻保留率 | 0.25 |
| 聚类潜力 | 降维后聚类的轮廓系数 | 0.20 |

### 3.5 聚类 (Clustering)

| 指标 | 说明 | 权重 |
|------|------|------|
| 轮廓系数 | 聚类紧密度和分离度 | 0.30 |
| Calinski-Harabasz | 类间/类内方差比 | 0.25 |
| Davies-Bouldin | 类内散度/类间距离 | 0.20 |
| 稳定性 | 子采样聚类一致性 | 0.25 |

---

## 4. API 设计

### 4.1 基础用法

```python
from scptensor import load_diann
from scptensor.autoselect import AutoSelector

# 加载数据
container = load_diann("data.tsv")

# 创建自动选择器
selector = AutoSelector(
    stages=["normalize", "impute", "integrate", "reduce", "cluster"],
    keep_all=False
)

# 执行全流程自动选择
container, report = selector.run(container)

# 查看报告
print(report.summary())
```

### 4.2 单环节使用

```python
from scptensor.autoselect import AutoSelector

selector = AutoSelector(stages=["impute"])
container, report = selector.run_stage(container, "impute")
```

### 4.3 自定义权重

```python
selector = AutoSelector(
    stages=["impute"],
    weights={
        "recovery_accuracy": 0.4,
        "distribution_consistency": 0.3,
        "biological_correlation": 0.2,
        "noise_introduction": 0.1
    }
)
```

### 4.4 报告导出

```python
report.save("autoselect_report.md")
report.save_json("autoselect_report.json")
report.save_csv("detailed_scores.csv")
```

---

## 5. 实现细节

### 5.1 评估器基类

```python
from abc import ABC, abstractmethod

class BaseEvaluator(ABC):
    """评估器基类"""

    @property
    @abstractmethod
    def stage_name(self) -> str:
        """环节名称"""
        pass

    @property
    @abstractmethod
    def methods(self) -> list[Callable]:
        """该环节所有可用方法"""
        pass

    @property
    @abstractmethod
    def metrics(self) -> list[Metric]:
        """评估指标列表"""
        pass

    @abstractmethod
    def evaluate(
        self,
        container: ScpContainer,
        method: Callable,
        **kwargs
    ) -> EvaluationResult:
        """评估单个方法"""
        pass
```

### 5.2 错误处理

- 某方法失败 -> 记录错误，继续评估其他方法
- 所有方法失败 -> 抛出 AutoSelectError
- 部分指标无法计算 -> 使用默认值 + 警告

### 5.3 性能优化

- 并行执行多个方法 (joblib)
- 进度条显示 (tqdm)
- 中间结果缓存

---

## 6. 与现有模块集成

### 6.1 统一方法注册

需要为各模块添加统一的方法注册机制：

```python
# scptensor/normalization/__init__.py
NORMALIZE_METHODS = {
    "log_transform": log_transform,
    "norm_mean": norm_mean,
    "norm_median": norm_median,
    "norm_quantile": norm_quantile,
}

def list_normalize_methods() -> list[str]:
    return list(NORMALIZE_METHODS.keys())
```

### 6.2 快捷函数

```python
# scptensor/autoselect/__init__.py
def auto_normalize(container, **kwargs):
    """自动选择归一化方法"""
    selector = AutoSelector(stages=["normalize"])
    return selector.run_stage(container, "normalize")

def auto_impute(container, **kwargs):
    """自动选择填充方法"""
    selector = AutoSelector(stages=["impute"])
    return selector.run_stage(container, "impute")

# ... 其他环节类似
```

---

## 7. 测试计划

1. 单元测试：每个评估器的指标计算
2. 集成测试：完整流程自动选择
3. 边界测试：空数据、单样本、全缺失等
4. 性能测试：大规模数据集

---

## 8. 里程碑

- [ ] Phase 1: 核心框架 (AutoSelector, BaseEvaluator)
- [ ] Phase 2: 指标计算模块 (metrics/)
- [ ] Phase 3: 各环节评估器 (evaluators/)
- [ ] Phase 4: 报告生成 (report.py)
- [ ] Phase 5: 测试和文档
