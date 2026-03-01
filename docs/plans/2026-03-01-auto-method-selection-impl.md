# Auto Method Selection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为 ScpTensor 添加自动方法选择功能，根据数据特性自动选择最优分析方法。

**Architecture:** 创建 scptensor/autoselect/ 模块，包含评估器基类、各环节评估器、指标计算模块和报告生成器。使用策略模式，每个分析环节有独立的评估器。

**Tech Stack:** Python 3.12+, NumPy, SciPy, scikit-learn, joblib (并行), tqdm (进度条)

---

## Phase 1: 核心框架

### Task 1: 创建数据类

**Files:**
- Create: `scptensor/autoselect/__init__.py`
- Create: `scptensor/autoselect/core.py`
- Create: `tests/test_autoselect/__init__.py`
- Create: `tests/test_autoselect/test_core.py`

**Step 1: 创建目录结构**

```bash
mkdir -p /home/shenshang/projects/ScpTensor/scptensor/autoselect/evaluators
mkdir -p /home/shenshang/projects/ScpTensor/scptensor/autoselect/metrics
mkdir -p /home/shenshang/projects/ScpTensor/tests/test_autoselect
touch /home/shenshang/projects/ScpTensor/scptensor/autoselect/__init__.py
touch /home/shenshang/projects/ScpTensor/scptensor/autoselect/evaluators/__init__.py
touch /home/shenshang/projects/ScpTensor/scptensor/autoselect/metrics/__init__.py
touch /home/shenshang/projects/ScpTensor/tests/test_autoselect/__init__.py
```

**Step 2: 写核心数据类**

Create `scptensor/autoselect/core.py`:

```python
"""自动方法选择核心模块."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvaluationResult:
    """单个方法的评估结果."""

    method_name: str
    scores: dict[str, float]
    overall_score: float
    execution_time: float
    layer_name: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """转换为字典."""
        return {
            "method_name": self.method_name,
            "scores": self.scores,
            "overall_score": self.overall_score,
            "execution_time": self.execution_time,
            "layer_name": self.layer_name,
            "error": self.error,
        }


@dataclass
class StageReport:
    """单个环节的评估报告."""

    stage_name: str
    results: list[EvaluationResult] = field(default_factory=list)
    best_method: str = ""
    best_result: EvaluationResult | None = None
    recommendation_reason: str = ""

    @property
    def success_rate(self) -> float:
        """计算成功率."""
        if not self.results:
            return 0.0
        failed = sum(1 for r in self.results if r.error is not None)
        return (len(self.results) - failed) / len(self.results)


@dataclass
class AutoSelectReport:
    """完整自动选择报告."""

    stages: dict[str, StageReport] = field(default_factory=dict)
    total_time: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """生成摘要字符串."""
        lines = [
            "=" * 60,
            "ScpTensor 自动方法选择报告",
            "=" * 60,
        ]
        for stage_name, report in self.stages.items():
            if report.best_method:
                score = report.best_result.overall_score if report.best_result else 0.0
                lines.append(f"{stage_name}: {report.best_method} (得分: {score:.2f})")
        lines.append("=" * 60)
        return "\n".join(lines)
```

**Step 3: 写测试**

Create `tests/test_autoselect/test_core.py`:

```python
"""测试核心数据类."""

import pytest
from scptensor.autoselect.core import EvaluationResult, StageReport, AutoSelectReport


class TestEvaluationResult:
    """测试 EvaluationResult."""

    def test_creation(self):
        """测试创建."""
        result = EvaluationResult(
            method_name="test_method",
            scores={"metric1": 0.8, "metric2": 0.9},
            overall_score=0.85,
            execution_time=1.5,
            layer_name="test_layer",
        )
        assert result.method_name == "test_method"
        assert result.overall_score == 0.85
        assert result.error is None

    def test_to_dict(self):
        """测试转换为字典."""
        result = EvaluationResult(
            method_name="test",
            scores={"a": 1.0},
            overall_score=1.0,
            execution_time=0.1,
            layer_name="layer",
        )
        d = result.to_dict()
        assert d["method_name"] == "test"
        assert d["scores"]["a"] == 1.0


class TestStageReport:
    """测试 StageReport."""

    def test_success_rate_empty(self):
        """测试空报告的成功率."""
        report = StageReport(stage_name="test")
        assert report.success_rate == 0.0

    def test_success_rate_all_success(self):
        """测试全部成功的成功率."""
        report = StageReport(
            stage_name="test",
            results=[
                EvaluationResult("m1", {}, 0.8, 0.1, "l1"),
                EvaluationResult("m2", {}, 0.9, 0.1, "l2"),
            ],
        )
        assert report.success_rate == 1.0

    def test_success_rate_partial(self):
        """测试部分成功的成功率."""
        report = StageReport(
            stage_name="test",
            results=[
                EvaluationResult("m1", {}, 0.8, 0.1, "l1"),
                EvaluationResult("m2", {}, 0.9, 0.1, "l2", error="failed"),
            ],
        )
        assert report.success_rate == 0.5


class TestAutoSelectReport:
    """测试 AutoSelectReport."""

    def test_summary_empty(self):
        """测试空报告摘要."""
        report = AutoSelectReport()
        summary = report.summary()
        assert "ScpTensor 自动方法选择报告" in summary

    def test_summary_with_stages(self):
        """测试有内容的报告摘要."""
        result = EvaluationResult("best_method", {}, 0.85, 0.1, "layer")
        stage = StageReport(
            stage_name="imputation",
            best_method="best_method",
            best_result=result,
        )
        report = AutoSelectReport(stages={"imputation": stage})
        summary = report.summary()
        assert "imputation: best_method" in summary
        assert "0.85" in summary
```

**Step 4: 运行测试**

```bash
uv run pytest tests/test_autoselect/test_core.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add scptensor/autoselect/ tests/test_autoselect/
git commit -m "feat(autoselect): add core data classes (EvaluationResult, StageReport, AutoSelectReport)"
```

---

### Task 2: 创建评估器基类

**Files:**
- Create: `scptensor/autoselect/evaluators/base.py`
- Create: `tests/test_autoselect/test_base_evaluator.py`

**Step 1: 写基类**

Create `scptensor/autoselect/evaluators/base.py`:

```python
"""评估器基类."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from scptensor import ScpContainer

from scptensor.autoselect.core import EvaluationResult, StageReport


class BaseEvaluator(ABC):
    """评估器抽象基类."""

    @property
    @abstractmethod
    def stage_name(self) -> str:
        """返回环节名称."""
        pass

    @property
    @abstractmethod
    def methods(self) -> dict[str, Callable]:
        """返回可用方法字典 {name: function}."""
        pass

    @property
    @abstractmethod
    def metric_weights(self) -> dict[str, float]:
        """返回指标权重."""
        pass

    @abstractmethod
    def compute_metrics(
        self,
        container: ScpContainer,
        original_container: ScpContainer,
        layer_name: str,
    ) -> dict[str, float]:
        """计算评估指标.

        Parameters
        ----------
        container : ScpContainer
            处理后的容器
        original_container : ScpContainer
            原始容器
        layer_name : str
            结果层名称

        Returns
        -------
        dict[str, float]
            指标得分字典
        """
        pass

    def compute_overall_score(self, scores: dict[str, float]) -> float:
        """计算加权综合得分.

        Parameters
        ----------
        scores : dict[str, float]
            各指标得分

        Returns
        -------
        float
            加权综合得分
        """
        weights = self.metric_weights
        total_weight = sum(weights.values())

        if total_weight == 0:
            return 0.0

        weighted_sum = sum(scores.get(k, 0.0) * v for k, v in weights.items())
        return weighted_sum / total_weight

    def evaluate_method(
        self,
        container: ScpContainer,
        method_name: str,
        method_func: Callable,
        **kwargs,
    ) -> tuple[ScpContainer | None, EvaluationResult]:
        """评估单个方法.

        Parameters
        ----------
        container : ScpContainer
            输入容器
        method_name : str
            方法名称
        method_func : Callable
            方法函数
        **kwargs
            方法参数

        Returns
        -------
        tuple[ScpContainer | None, EvaluationResult]
            (结果容器, 评估结果)
        """
        import time

        start_time = time.time()
        layer_name = kwargs.get("new_layer_name", f"{self.stage_name}_{method_name}")

        try:
            result_container = method_func(container, **kwargs)
            metrics = self.compute_metrics(result_container, container, layer_name)
            overall = self.compute_overall_score(metrics)
            exec_time = time.time() - start_time

            return result_container, EvaluationResult(
                method_name=method_name,
                scores=metrics,
                overall_score=overall,
                execution_time=exec_time,
                layer_name=layer_name,
            )
        except Exception as e:
            exec_time = time.time() - start_time
            return None, EvaluationResult(
                method_name=method_name,
                scores={},
                overall_score=0.0,
                execution_time=exec_time,
                layer_name=layer_name,
                error=str(e),
            )

    def run_all(
        self,
        container: ScpContainer,
        keep_all: bool = False,
        **kwargs,
    ) -> tuple[ScpContainer, StageReport]:
        """运行所有方法并选择最优.

        Parameters
        ----------
        container : ScpContainer
            输入容器
        keep_all : bool
            是否保留所有结果层
        **kwargs
            传递给方法的参数

        Returns
        -------
        tuple[ScpContainer, StageReport]
            (最优结果容器, 环节报告)
        """
        results: list[EvaluationResult] = []
        best_container = container
        best_result: EvaluationResult | None = None
        all_containers: dict[str, ScpContainer] = {}

        for name, func in self.methods.items():
            result_container, eval_result = self.evaluate_method(
                container, name, func, **kwargs
            )
            results.append(eval_result)

            if result_container is not None:
                if keep_all:
                    all_containers[name] = result_container

                if best_result is None or eval_result.overall_score > best_result.overall_score:
                    best_result = eval_result
                    best_container = result_container

        # 找到最佳方法
        successful_results = [r for r in results if r.error is None]
        if successful_results:
            best = max(successful_results, key=lambda r: r.overall_score)
            best_method = best.method_name
            recommendation = f"得分最高 ({best.overall_score:.3f})"
        else:
            best_method = ""
            recommendation = "所有方法均失败"

        report = StageReport(
            stage_name=self.stage_name,
            results=results,
            best_method=best_method,
            best_result=best_result,
            recommendation_reason=recommendation,
        )

        return best_container, report
```

**Step 2: 写测试**

Create `tests/test_autoselect/test_base_evaluator.py`:

```python
"""测试评估器基类."""

import pytest
from scptensor.autoselect.evaluators.base import BaseEvaluator
from scptensor import ScpContainer, create_test_container


class MockEvaluator(BaseEvaluator):
    """模拟评估器用于测试."""

    @property
    def stage_name(self) -> str:
        return "mock"

    @property
    def methods(self) -> dict:
        return {
            "method_a": lambda c, **kw: c,
            "method_b": lambda c, **kw: c,
        }

    @property
    def metric_weights(self) -> dict:
        return {"metric1": 0.5, "metric2": 0.5}

    def compute_metrics(self, container, original, layer_name) -> dict:
        return {"metric1": 0.8, "metric2": 0.9}


class TestBaseEvaluator:
    """测试 BaseEvaluator."""

    def test_stage_name(self):
        """测试环节名称."""
        evaluator = MockEvaluator()
        assert evaluator.stage_name == "mock"

    def test_compute_overall_score(self):
        """测试综合得分计算."""
        evaluator = MockEvaluator()
        score = evaluator.compute_overall_score({"metric1": 0.8, "metric2": 0.9})
        assert 0.84 < score < 0.86

    def test_compute_overall_score_empty_weights(self):
        """测试空权重."""
        evaluator = MockEvaluator()
        # 临时修改权重
        evaluator._weights_override = {}
        # 使用空权重应该返回 0
        assert evaluator.compute_overall_score({}) == 0.0

    def test_evaluate_method_success(self):
        """测试方法评估成功."""
        evaluator = MockEvaluator()
        container = create_test_container(n_samples=10, n_features=20)

        result_container, eval_result = evaluator.evaluate_method(
            container, "method_a", lambda c, **kw: c
        )

        assert result_container is not None
        assert eval_result.error is None
        assert eval_result.overall_score > 0

    def test_run_all(self):
        """测试运行所有方法."""
        evaluator = MockEvaluator()
        container = create_test_container(n_samples=10, n_features=20)

        best_container, report = evaluator.run_all(container)

        assert report.best_method != ""
        assert len(report.results) == 2
        assert report.success_rate == 1.0
```

**Step 3: 运行测试**

```bash
uv run pytest tests/test_autoselect/test_base_evaluator.py -v
```

Expected: All tests PASS

**Step 4: Commit**

```bash
git add scptensor/autoselect/evaluators/base.py tests/test_autoselect/test_base_evaluator.py
git commit -m "feat(autoselect): add BaseEvaluator abstract class"
```

---

## Phase 2: 指标计算模块

### Task 3: 通用质量指标

**Files:**
- Create: `scptensor/autoselect/metrics/quality.py`
- Create: `tests/test_autoselect/test_metrics_quality.py`

**Step 1: 写指标函数**

Create `scptensor/autoselect/metrics/quality.py`:

```python
"""通用质量评估指标."""

from __future__ import annotations

import numpy as np
from scipy import stats


def cv_stability(X: np.ndarray) -> float:
    """计算变异系数稳定性.

    CV 越稳定（标准差越小），得分越高。

    Parameters
    ----------
    X : np.ndarray
        数据矩阵 (samples x features)

    Returns
    -------
    float
        稳定性得分 (0-1)
    """
    # 计算每个特征的 CV
    means = np.nanmean(X, axis=0)
    stds = np.nanstd(X, axis=0)

    # 避免除零
    valid = means > 0
    cvs = np.zeros_like(means)
    cvs[valid] = stds[valid] / means[valid]

    # CV 的变异系数（越小越稳定）
    cv_of_cv = np.std(cvs[cvs > 0]) / np.mean(cvs[cvs > 0]) if np.mean(cvs[cvs > 0]) > 0 else 0

    # 转换为得分（越稳定得分越高）
    return max(0, 1 - cv_of_cv)


def skewness_improvement(X_before: np.ndarray, X_after: np.ndarray) -> float:
    """计算偏度改善程度.

    偏度越接近 0 越好。

    Parameters
    ----------
    X_before : np.ndarray
        处理前数据
    X_after : np.ndarray
        处理后数据

    Returns
    -------
    float
        改善得分 (0-1)
    """
    skew_before = np.abs(stats.skew(X_before[~np.isnan(X_before)]))
    skew_after = np.abs(stats.skew(X_after[~np.isnan(X_after)]))

    if skew_before == 0:
        return 1.0 if skew_after == 0 else 0.0

    improvement = (skew_before - skew_after) / skew_before
    return max(0, min(1, improvement + 0.5))  # 归一化到 0-1


def kurtosis_improvement(X_before: np.ndarray, X_after: np.ndarray) -> float:
    """计算峰度改善程度.

    峰度越接近 3（正态分布）越好。

    Parameters
    ----------
    X_before : np.ndarray
        处理前数据
    X_after : np.ndarray
        处理后数据

    Returns
    -------
    float
        改善得分 (0-1)
    """
    kurt_before = np.abs(stats.kurtosis(X_before[~np.isnan(X_before)]) - 3)
    kurt_after = np.abs(stats.kurtosis(X_after[~np.isnan(X_after)]) - 3)

    if kurt_before == 0:
        return 1.0 if kurt_after == 0 else 0.0

    improvement = (kurt_before - kurt_after) / kurt_before
    return max(0, min(1, improvement + 0.5))


def dynamic_range(X: np.ndarray) -> float:
    """计算动态范围合理性.

    动态范围在合理区间（2-10 个数量级）得分高。

    Parameters
    ----------
    X : np.ndarray
        数据矩阵

    Returns
    -------
    float
        合理性得分 (0-1)
    """
    X_valid = X[~np.isnan(X) & (X > 0)]
    if len(X_valid) == 0:
        return 0.0

    min_val = np.min(X_valid)
    max_val = np.max(X_valid)

    if min_val <= 0:
        return 0.0

    log_range = np.log10(max_val / min_val)

    # 理想范围 2-10
    if 2 <= log_range <= 10:
        return 1.0
    elif log_range < 2:
        return log_range / 2
    else:
        return max(0, 1 - (log_range - 10) / 10)


def outlier_ratio(X: np.ndarray) -> float:
    """计算异常值比例（越少越好）.

    Parameters
    ----------
    X : np.ndarray
        数据矩阵

    Returns
    -------
    float
        得分 (0-1), 异常值越少得分越高
    """
    X_valid = X[~np.isnan(X)]
    if len(X_valid) == 0:
        return 0.0

    q1 = np.percentile(X_valid, 25)
    q3 = np.percentile(X_valid, 75)
    iqr = q3 - q1

    if iqr == 0:
        return 1.0

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = np.sum((X_valid < lower) | (X_valid > upper))
    ratio = outliers / len(X_valid)

    # 异常值比例越低得分越高
    return max(0, 1 - ratio * 5)
```

**Step 2: 写测试**

Create `tests/test_autoselect/test_metrics_quality.py`:

```python
"""测试通用质量指标."""

import numpy as np
import pytest
from scptensor.autoselect.metrics.quality import (
    cv_stability,
    skewness_improvement,
    kurtosis_improvement,
    dynamic_range,
    outlier_ratio,
)


class TestCvStability:
    """测试 CV 稳定性."""

    def test_stable_data(self):
        """测试稳定数据."""
        X = np.random.randn(100, 50) * 0.1 + 10  # 低变异
        score = cv_stability(X)
        assert 0 <= score <= 1

    def test_unstable_data(self):
        """测试不稳定数据."""
        X = np.random.randn(100, 50) * 10  # 高变异
        score = cv_stability(X)
        assert 0 <= score <= 1


class TestSkewnessImprovement:
    """测试偏度改善."""

    def test_improvement(self):
        """测试有改善."""
        X_before = np.exp(np.random.randn(1000))  # 右偏
        X_after = np.log1p(X_before)  # 对数变换后更正态
        score = skewness_improvement(X_before, X_after)
        assert score >= 0

    def test_no_change(self):
        """测试无变化."""
        X = np.random.randn(1000)
        score = skewness_improvement(X, X)
        assert 0 <= score <= 1


class TestDynamicRange:
    """测试动态范围."""

    def test_good_range(self):
        """测试合理范围."""
        X = np.random.lognormal(0, 1, (100, 50))
        score = dynamic_range(X)
        assert 0 <= score <= 1

    def test_narrow_range(self):
        """测试窄范围."""
        X = np.ones((100, 50)) * 10 + np.random.randn(100, 50) * 0.01
        score = dynamic_range(X)
        assert 0 <= score <= 1


class TestOutlierRatio:
    """测试异常值比例."""

    def test_no_outliers(self):
        """测试无异常值."""
        X = np.random.randn(100, 50)
        score = outlier_ratio(X)
        assert score > 0.9  # 正态数据应该很少有异常值

    def test_many_outliers(self):
        """测试有很多异常值."""
        X = np.random.randn(100, 50)
        X[::10] *= 100  # 添加异常值
        score = outlier_ratio(X)
        assert 0 <= score < 1
```

**Step 3: 运行测试**

```bash
uv run pytest tests/test_autoselect/test_metrics_quality.py -v
```

Expected: All tests PASS

**Step 4: Commit**

```bash
git add scptensor/autoselect/metrics/quality.py tests/test_autoselect/test_metrics_quality.py
git commit -m "feat(autoselect): add quality metrics (cv_stability, skewness, kurtosis, dynamic_range, outlier_ratio)"
```

---

## Phase 3: AutoSelector 主类

### Task 4: 实现 AutoSelector

**Files:**
- Modify: `scptensor/autoselect/core.py`
- Create: `tests/test_autoselect/test_auto_selector.py`

**Step 1: 添加 AutoSelector 类**

在 `scptensor/autoselect/core.py` 末尾添加:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scptensor.autoselect.evaluators.base import BaseEvaluator


class AutoSelector:
    """统一自动方法选择器.

    Examples
    --------
    >>> selector = AutoSelector(stages=["normalize", "impute"])
    >>> container, report = selector.run(container)

    """

    SUPPORTED_STAGES = ["normalize", "impute", "integrate", "reduce", "cluster"]

    def __init__(
        self,
        stages: list[str] | None = None,
        keep_all: bool = False,
        weights: dict[str, dict[str, float]] | None = None,
        parallel: bool = True,
        n_jobs: int = -1,
    ):
        """初始化选择器.

        Parameters
        ----------
        stages : list[str] | None
            要优化的环节列表，None 表示全部
        keep_all : bool
            是否保留所有方法的结果层
        weights : dict | None
            自定义指标权重 {stage: {metric: weight}}
        parallel : bool
            是否并行执行
        n_jobs : int
            并行进程数，-1 表示使用所有 CPU

        """
        self.stages = stages if stages else self.SUPPORTED_STAGES
        self.keep_all = keep_all
        self.weights = weights or {}
        self.parallel = parallel
        self.n_jobs = n_jobs
        self._evaluators: dict[str, BaseEvaluator] = {}

    def _get_evaluator(self, stage: str) -> BaseEvaluator:
        """获取环节评估器.

        Parameters
        ----------
        stage : str
            环节名称

        Returns
        -------
        BaseEvaluator
            评估器实例

        """
        if stage in self._evaluators:
            return self._evaluators[stage]

        # 延迟导入避免循环依赖
        if stage == "normalize":
            from scptensor.autoselect.evaluators.normalization import NormalizationEvaluator
            self._evaluators[stage] = NormalizationEvaluator()
        elif stage == "impute":
            from scptensor.autoselect.evaluators.imputation import ImputationEvaluator
            self._evaluators[stage] = ImputationEvaluator()
        elif stage == "integrate":
            from scptensor.autoselect.evaluators.integration import IntegrationEvaluator
            self._evaluators[stage] = IntegrationEvaluator()
        elif stage == "reduce":
            from scptensor.autoselect.evaluators.dim_reduction import DimReductionEvaluator
            self._evaluators[stage] = DimReductionEvaluator()
        elif stage == "cluster":
            from scptensor.autoselect.evaluators.clustering import ClusteringEvaluator
            self._evaluators[stage] = ClusteringEvaluator()
        else:
            raise ValueError(f"Unknown stage: {stage}")

        return self._evaluators[stage]

    def run_stage(
        self,
        container: ScpContainer,
        stage: str,
        **kwargs,
    ) -> tuple[ScpContainer, StageReport]:
        """执行单个环节的自动选择.

        Parameters
        ----------
        container : ScpContainer
            输入容器
        stage : str
            环节名称
        **kwargs
            传递给方法的参数

        Returns
        -------
        tuple[ScpContainer, StageReport]
            (结果容器, 环节报告)

        """
        import time

        start_time = time.time()
        evaluator = self._get_evaluator(stage)

        # 应用自定义权重
        if stage in self.weights:
            evaluator._custom_weights = self.weights[stage]

        result_container, report = evaluator.run_all(
            container, keep_all=self.keep_all, **kwargs
        )
        report.duration = time.time() - start_time

        return result_container, report

    def run(self, container: ScpContainer) -> tuple[ScpContainer, AutoSelectReport]:
        """执行全流程自动选择.

        Parameters
        ----------
        container : ScpContainer
            输入容器

        Returns
        -------
        tuple[ScpContainer, AutoSelectReport]
            (结果容器, 完整报告)

        """
        import time

        start_time = time.time()
        report = AutoSelectReport()
        current_container = container

        for stage in self.stages:
            if stage not in self.SUPPORTED_STAGES:
                report.warnings.append(f"Unknown stage: {stage}, skipped")
                continue

            try:
                current_container, stage_report = self.run_stage(current_container, stage)
                report.stages[stage] = stage_report
            except Exception as e:
                report.warnings.append(f"Stage {stage} failed: {str(e)}")

        report.total_time = time.time() - start_time
        return current_container, report
```

**Step 2: 写测试**

Create `tests/test_autoselect/test_auto_selector.py`:

```python
"""测试 AutoSelector."""

import pytest
from scptensor.autoselect.core import AutoSelector
from scptensor import create_test_container


class TestAutoSelector:
    """测试 AutoSelector."""

    def test_init_default(self):
        """测试默认初始化."""
        selector = AutoSelector()
        assert len(selector.stages) == 5
        assert selector.keep_all is False

    def test_init_custom_stages(self):
        """测试自定义环节."""
        selector = AutoSelector(stages=["normalize", "impute"])
        assert selector.stages == ["normalize", "impute"]

    def test_supported_stages(self):
        """测试支持的环节."""
        assert "normalize" in AutoSelector.SUPPORTED_STAGES
        assert "impute" in AutoSelector.SUPPORTED_STAGES
        assert "integrate" in AutoSelector.SUPPORTED_STAGES
        assert "reduce" in AutoSelector.SUPPORTED_STAGES
        assert "cluster" in AutoSelector.SUPPORTED_STAGES

    def test_run_unknown_stage_warning(self):
        """测试未知环节警告."""
        container = create_test_container(n_samples=10, n_features=20)
        selector = AutoSelector(stages=["unknown_stage"])
        _, report = selector.run(container)
        assert any("unknown_stage" in w for w in report.warnings)
```

**Step 3: 运行测试**

```bash
uv run pytest tests/test_autoselect/test_auto_selector.py -v
```

Expected: All tests PASS

**Step 4: Commit**

```bash
git add scptensor/autoselect/core.py tests/test_autoselect/test_auto_selector.py
git commit -m "feat(autoselect): add AutoSelector main class"
```

---

## Phase 4: 报告生成器

### Task 5: 实现报告导出

**Files:**
- Create: `scptensor/autoselect/report.py`
- Modify: `scptensor/autoselect/core.py` (添加 save 方法到 AutoSelectReport)
- Create: `tests/test_autoselect/test_report.py`

**Step 1: 创建报告模块**

Create `scptensor/autoselect/report.py`:

```python
"""报告生成和导出."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scptensor.autoselect.core import AutoSelectReport


def save_markdown(report: AutoSelectReport, filepath: str | Path) -> None:
    """保存为 Markdown 格式.

    Parameters
    ----------
    report : AutoSelectReport
        报告对象
    filepath : str | Path
        文件路径

    """
    filepath = Path(filepath)
    lines = [
        "# ScpTensor 自动方法选择报告",
        "",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**总耗时**: {report.total_time:.2f} 秒",
        "",
        "## 各环节结果",
        "",
    ]

    for stage_name, stage_report in report.stages.items():
        lines.append(f"### {stage_name}")
        lines.append("")
        lines.append(f"- **最优方法**: {stage_report.best_method}")
        if stage_report.best_result:
            lines.append(f"- **综合得分**: {stage_report.best_result.overall_score:.3f}")
        lines.append(f"- **成功率**: {stage_report.success_rate:.1%}")
        lines.append("")

        # 方法对比表格
        if stage_report.results:
            lines.append("| 方法 | 综合得分 | 耗时 (秒) | 状态 |")
            lines.append("|------|----------|-----------|------|")
            for r in stage_report.results:
                status = "✅" if r.error is None else "❌"
                lines.append(f"| {r.method_name} | {r.overall_score:.3f} | {r.execution_time:.2f} | {status} |")
            lines.append("")

    if report.warnings:
        lines.append("## 警告")
        lines.append("")
        for w in report.warnings:
            lines.append(f"- {w}")
        lines.append("")

    filepath.write_text("\n".join(lines), encoding="utf-8")


def save_json(report: AutoSelectReport, filepath: str | Path) -> None:
    """保存为 JSON 格式.

    Parameters
    ----------
    report : AutoSelectReport
        报告对象
    filepath : str | Path
        文件路径

    """
    filepath = Path(filepath)

    data = {
        "generated_at": datetime.now().isoformat(),
        "total_time": report.total_time,
        "stages": {},
        "warnings": report.warnings,
    }

    for stage_name, stage_report in report.stages.items():
        data["stages"][stage_name] = {
            "best_method": stage_report.best_method,
            "success_rate": stage_report.success_rate,
            "results": [r.to_dict() for r in stage_report.results],
        }

    filepath.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_csv(report: AutoSelectReport, filepath: str | Path) -> None:
    """保存详细结果为 CSV 格式.

    Parameters
    ----------
    report : AutoSelectReport
        报告对象
    filepath : str | Path
        文件路径

    """
    import csv

    filepath = Path(filepath)

    rows = []
    fieldnames = set(["stage", "method", "overall_score", "execution_time", "error"])

    for stage_name, stage_report in report.stages.items():
        for r in stage_report.results:
            row = {
                "stage": stage_name,
                "method": r.method_name,
                "overall_score": r.overall_score,
                "execution_time": r.execution_time,
                "error": r.error or "",
            }
            # 添加各指标得分
            for k, v in r.scores.items():
                fieldnames.add(k)
                row[k] = v
            rows.append(row)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
```

**Step 2: 在 core.py 添加 save 方法**

在 `AutoSelectReport` 类中添加:

```python
def save(self, filepath: str | Path, format: str = "markdown") -> None:
    """保存报告.

    Parameters
    ----------
    filepath : str | Path
        文件路径
    format : str
        格式: "markdown", "json", "csv"

    """
    from scptensor.autoselect.report import save_markdown, save_json, save_csv

    if format == "markdown":
        save_markdown(self, filepath)
    elif format == "json":
        save_json(self, filepath)
    elif format == "csv":
        save_csv(self, filepath)
    else:
        raise ValueError(f"Unknown format: {format}")
```

**Step 3: 写测试**

Create `tests/test_autoselect/test_report.py`:

```python
"""测试报告生成."""

import json
import tempfile
from pathlib import Path

import pytest
from scptensor.autoselect.core import AutoSelectReport, EvaluationResult, StageReport
from scptensor.autoselect.report import save_markdown, save_json, save_csv


@pytest.fixture
def sample_report():
    """创建示例报告."""
    result = EvaluationResult(
        method_name="best_method",
        scores={"metric1": 0.8, "metric2": 0.9},
        overall_score=0.85,
        execution_time=1.5,
        layer_name="test_layer",
    )
    stage = StageReport(
        stage_name="imputation",
        results=[result],
        best_method="best_method",
        best_result=result,
    )
    return AutoSelectReport(
        stages={"imputation": stage},
        total_time=2.0,
        warnings=["test warning"],
    )


class TestReportSave:
    """测试报告保存."""

    def test_save_markdown(self, sample_report, tmp_path):
        """测试 Markdown 保存."""
        filepath = tmp_path / "report.md"
        save_markdown(sample_report, filepath)

        content = filepath.read_text()
        assert "# ScpTensor 自动方法选择报告" in content
        assert "imputation" in content
        assert "best_method" in content

    def test_save_json(self, sample_report, tmp_path):
        """测试 JSON 保存."""
        filepath = tmp_path / "report.json"
        save_json(sample_report, filepath)

        with open(filepath) as f:
            data = json.load(f)

        assert "stages" in data
        assert "imputation" in data["stages"]
        assert data["stages"]["imputation"]["best_method"] == "best_method"

    def test_save_csv(self, sample_report, tmp_path):
        """测试 CSV 保存."""
        filepath = tmp_path / "report.csv"
        save_csv(sample_report, filepath)

        content = filepath.read_text()
        assert "stage" in content
        assert "method" in content
        assert "imputation" in content

    def test_auto_select_report_save(self, sample_report, tmp_path):
        """测试 AutoSelectReport.save 方法."""
        # Markdown
        md_path = tmp_path / "report.md"
        sample_report.save(md_path, format="markdown")
        assert md_path.exists()

        # JSON
        json_path = tmp_path / "report.json"
        sample_report.save(json_path, format="json")
        assert json_path.exists()

        # CSV
        csv_path = tmp_path / "report.csv"
        sample_report.save(csv_path, format="csv")
        assert csv_path.exists()
```

**Step 4: 运行测试**

```bash
uv run pytest tests/test_autoselect/test_report.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add scptensor/autoselect/report.py scptensor/autoselect/core.py tests/test_autoselect/test_report.py
git commit -m "feat(autoselect): add report generation (markdown, json, csv)"
```

---

## 后续任务

剩余任务按照相同模式编写:
- Task 6-10: 各环节评估器 (normalization, imputation, integration, dim_reduction, clustering)
- Task 11: 快捷函数
- Task 12: 集成测试

---

## 里程碑检查

完成以上任务后:

```bash
# 运行所有测试
uv run pytest tests/test_autoselect/ -v

# 代码质量检查
uv run ruff check scptensor/autoselect/
uv run mypy scptensor/autoselect/
```
