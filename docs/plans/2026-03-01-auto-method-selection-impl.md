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
- Test: `tests/test_autoselect/test_core.py`

**Step 1: 创建目录结构**

```bash
mkdir -p scptensor/autoselect/evaluators
mkdir -p scptensor/autoselect/metrics
mkdir -p tests/test_autoselect
```

**Step 2: 创建 `scptensor/autoselect/__init__.py`**

```python
"""Automatic method selection for ScpTensor."""

from scptensor.autoselect.core import (
    AutoSelector,
    EvaluationResult,
    StageReport,
    AutoSelectReport,
)

__all__ = [
    "AutoSelector",
    "EvaluationResult", 
    "StageReport",
    "AutoSelectReport",
]
```

**Step 3: 创建 `scptensor/autoselect/core.py`**

```python
"""Core classes for automatic method selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvaluationResult:
    """Evaluation result for a single method.
    
    Parameters
    ----------
    method_name : str
        Name of the evaluated method.
    scores : dict[str, float]
        Individual metric scores.
    overall_score : float
        Weighted overall score (0-1).
    execution_time : float
        Execution time in seconds.
    layer_name : str
        Name of the result layer.
    error : str | None
        Error message if method failed.
    """
    method_name: str
    scores: dict[str, float]
    overall_score: float
    execution_time: float
    layer_name: str
    error: str | None = None


@dataclass
class StageReport:
    """Evaluation report for a single analysis stage.
    
    Parameters
    ----------
    stage_name : str
        Name of the analysis stage.
    results : list[EvaluationResult]
        Results for all evaluated methods.
    best_method : str
        Name of the best performing method.
    best_result : EvaluationResult
        Result of the best method.
    recommendation_reason : str
        Explanation for the recommendation.
    """
    stage_name: str
    results: list[EvaluationResult]
    best_method: str
    best_result: EvaluationResult
    recommendation_reason: str = ""


@dataclass
class AutoSelectReport:
    """Complete report for automatic method selection.
    
    Parameters
    ----------
    stages : dict[str, StageReport]
        Reports for each stage.
    total_time : float
        Total execution time in seconds.
    warnings : list[str]
        List of warnings encountered.
    """
    stages: dict[str, StageReport] = field(default_factory=dict)
    total_time: float = 0.0
    warnings: list[str] = field(default_factory=list)
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "=" * 60,
            "ScpTensor 自动方法选择报告",
            "=" * 60,
        ]
        for stage_name, report in self.stages.items():
            lines.append(
                f"{stage_name}: {report.best_method} (得分: {report.best_result.overall_score:.2f})"
            )
        lines.append("=" * 60)
        return "\n".join(lines)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate across all methods."""
        total = sum(len(s.results) for s in self.stages.values())
        if total == 0:
            return 0.0
        failed = sum(
            1 for s in self.stages.values() 
            for r in s.results if r.error is not None
        )
        return (total - failed) / total


class AutoSelector:
    """Automatic method selector for ScpTensor analysis pipeline.
    
    Parameters
    ----------
    stages : list[str] | None
        List of stages to optimize. None for all stages.
    strategy : str
        Evaluation strategy ("comprehensive" or "fast").
    keep_all : bool
        Whether to keep all method results in container.
    weights : dict[str, float] | None
        Custom metric weights.
    parallel : bool
        Whether to run methods in parallel.
    n_jobs : int
        Number of parallel jobs (-1 for all cores).
    """
    
    SUPPORTED_STAGES = [
        "normalize", "impute", "integrate", 
        "reduce", "cluster"
    ]
    
    def __init__(
        self,
        stages: list[str] | None = None,
        strategy: str = "comprehensive",
        keep_all: bool = False,
        weights: dict[str, float] | None = None,
        parallel: bool = True,
        n_jobs: int = -1,
    ) -> None:
        self.stages = stages or self.SUPPORTED_STAGES
        self.strategy = strategy
        self.keep_all = keep_all
        self.weights = weights or {}
        self.parallel = parallel
        self.n_jobs = n_jobs
        
        # Validate stages
        invalid = set(self.stages) - set(self.SUPPORTED_STAGES)
        if invalid:
            raise ValueError(f"Unknown stages: {invalid}")
    
    def run(self, container: Any) -> tuple[Any, AutoSelectReport]:
        """Run automatic selection for all configured stages.
        
        Parameters
        ----------
        container : ScpContainer
            Input data container.
            
        Returns
        -------
        tuple[ScpContainer, AutoSelectReport]
            Container with best results and selection report.
        """
        import time
        
        report = AutoSelectReport()
        start_time = time.time()
        
        for stage in self.stages:
            container, stage_report = self.run_stage(container, stage)
            report.stages[stage] = stage_report
        
        report.total_time = time.time() - start_time
        return container, report
    
    def run_stage(
        self, 
        container: Any, 
        stage: str
    ) -> tuple[Any, StageReport]:
        """Run automatic selection for a single stage.
        
        Parameters
        ----------
        container : ScpContainer
            Input data container.
        stage : str
            Stage name to optimize.
            
        Returns
        -------
        tuple[ScpContainer, StageReport]
            Container with best result and stage report.
        """
        # Placeholder - will be implemented with evaluators
        raise NotImplementedError(f"Stage {stage} evaluator not implemented")
```

**Step 4: 创建测试文件 `tests/test_autoselect/test_core.py`**

```python
"""Tests for autoselect core classes."""

import pytest
from scptensor.autoselect import (
    AutoSelector,
    EvaluationResult,
    StageReport,
    AutoSelectReport,
)


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""
    
    def test_create_result(self):
        """Test creating an evaluation result."""
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
    
    def test_result_with_error(self):
        """Test creating a result with error."""
        result = EvaluationResult(
            method_name="failed_method",
            scores={},
            overall_score=0.0,
            execution_time=0.1,
            layer_name="",
            error="Method failed",
        )
        assert result.error == "Method failed"


class TestStageReport:
    """Tests for StageReport dataclass."""
    
    def test_create_stage_report(self):
        """Test creating a stage report."""
        best_result = EvaluationResult(
            method_name="best",
            scores={"m": 0.9},
            overall_score=0.9,
            execution_time=1.0,
            layer_name="best_layer",
        )
        report = StageReport(
            stage_name="test_stage",
            results=[best_result],
            best_method="best",
            best_result=best_result,
        )
        assert report.stage_name == "test_stage"
        assert report.best_method == "best"


class TestAutoSelectReport:
    """Tests for AutoSelectReport class."""
    
    def test_empty_report(self):
        """Test empty report."""
        report = AutoSelectReport()
        assert len(report.stages) == 0
        assert report.success_rate == 0.0
    
    def test_summary(self):
        """Test summary generation."""
        best = EvaluationResult(
            method_name="knn",
            scores={},
            overall_score=0.85,
            execution_time=1.0,
            layer_name="knn_layer",
        )
        stage = StageReport(
            stage_name="impute",
            results=[best],
            best_method="knn",
            best_result=best,
        )
        report = AutoSelectReport(stages={"impute": stage})
        
        summary = report.summary()
        assert "impute" in summary
        assert "knn" in summary


class TestAutoSelector:
    """Tests for AutoSelector class."""
    
    def test_create_selector(self):
        """Test creating a selector."""
        selector = AutoSelector(stages=["impute"])
        assert selector.stages == ["impute"]
    
    def test_invalid_stage(self):
        """Test invalid stage raises error."""
        with pytest.raises(ValueError):
            AutoSelector(stages=["invalid_stage"])
    
    def test_default_stages(self):
        """Test default stages includes all."""
        selector = AutoSelector()
        assert len(selector.stages) == 5
```

**Step 5: 运行测试**

```bash
uv run pytest tests/test_autoselect/test_core.py -v
```

Expected: All tests PASS

**Step 6: Commit**

```bash
git add scptensor/autoselect/ tests/test_autoselect/
git commit -m "feat(autoselect): add core data classes and AutoSelector"
```

---

## Phase 2: 指标计算模块

### Task 2: 创建通用质量指标

**Files:**
- Create: `scptensor/autoselect/metrics/__init__.py`
- Create: `scptensor/autoselect/metrics/quality.py`
- Test: `tests/test_autoselect/test_metrics_quality.py`

**Step 1: 创建 `scptensor/autoselect/metrics/__init__.py`**

```python
"""Metrics for automatic method evaluation."""

from scptensor.autoselect.metrics.quality import (
    cv_stability,
    skewness_improvement,
    kurtosis_improvement,
    dynamic_range,
    outlier_ratio,
)

__all__ = [
    "cv_stability",
    "skewness_improvement",
    "kurtosis_improvement",
    "dynamic_range",
    "outlier_ratio",
]
```

**Step 2: 创建 `scptensor/autoselect/metrics/quality.py`**

```python
"""Quality metrics for data evaluation."""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.sparse import issparse


def _to_dense(X: np.ndarray) -> np.ndarray:
    """Convert to dense array if sparse."""
    if issparse(X):
        return X.toarray()
    return np.asarray(X)


def cv_stability(X: np.ndarray) -> float:
    """Calculate CV (coefficient of variation) stability.
    
    Higher score means more stable CV distribution.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix (samples x features).
        
    Returns
    -------
    float
        CV stability score (0-1).
    """
    X = _to_dense(X)
    
    # Calculate CV per feature
    means = np.nanmean(X, axis=0)
    stds = np.nanstd(X, axis=0)
    
    # Avoid division by zero
    valid = means > 0
    cvs = np.zeros_like(means)
    cvs[valid] = stds[valid] / means[valid]
    
    # CV stability: lower variance in CV is better
    cv_var = np.nanvar(cvs[valid])
    
    # Normalize to 0-1 (assuming typical CV variance range)
    score = max(0, 1 - cv_var)
    return float(score)


def skewness_improvement(X_before: np.ndarray, X_after: np.ndarray) -> float:
    """Calculate improvement in skewness toward normal distribution.
    
    Parameters
    ----------
    X_before : np.ndarray
        Data before transformation.
    X_after : np.ndarray
        Data after transformation.
        
    Returns
    -------
    float
        Skewness improvement score (0-1).
    """
    X_before = _to_dense(X_before)
    X_after = _to_dense(X_after)
    
    # Calculate skewness
    skew_before = abs(np.nanmean(stats.skew(X_before[~np.isnan(X_before)])))
    skew_after = abs(np.nanmean(stats.skew(X_after[~np.isnan(X_after)])))
    
    # Improvement = reduction in skewness
    if skew_before == 0:
        return 1.0
    
    improvement = (skew_before - skew_after) / skew_before
    return float(max(0, min(1, improvement)))


def kurtosis_improvement(X_before: np.ndarray, X_after: np.ndarray) -> float:
    """Calculate improvement in kurtosis toward normal distribution.
    
    Parameters
    ----------
    X_before : np.ndarray
        Data before transformation.
    X_after : np.ndarray
        Data after transformation.
        
    Returns
    -------
    float
        Kurtosis improvement score (0-1).
    """
    X_before = _to_dense(X_before)
    X_after = _to_dense(X_after)
    
    # Normal kurtosis = 0 (excess kurtosis)
    kurt_before = abs(np.nanmean(stats.kurtosis(X_before[~np.isnan(X_before)])))
    kurt_after = abs(np.nanmean(stats.kurtosis(X_after[~np.isnan(X_after)])))
    
    if kurt_before == 0:
        return 1.0
    
    improvement = (kurt_before - kurt_after) / kurt_before
    return float(max(0, min(1, improvement)))


def dynamic_range(X: np.ndarray) -> float:
    """Calculate dynamic range score.
    
    Moderate dynamic range is preferred (not too narrow, not too wide).
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix.
        
    Returns
    -------
    float
        Dynamic range score (0-1).
    """
    X = _to_dense(X)
    
    X_valid = X[~np.isnan(X)]
    if len(X_valid) == 0:
        return 0.0
    
    min_val = np.min(X_valid)
    max_val = np.max(X_valid)
    
    if min_val <= 0:
        return 0.0
    
    log_range = np.log10(max_val / min_val)
    
    # Optimal range is 2-4 orders of magnitude
    if 2 <= log_range <= 4:
        return 1.0
    elif log_range < 2:
        return float(log_range / 2)
    else:
        return float(max(0, 1 - (log_range - 4) / 4))


def outlier_ratio(X: np.ndarray, threshold: float = 3.0) -> float:
    """Calculate outlier ratio (lower is better).
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix.
    threshold : float
        Z-score threshold for outliers.
        
    Returns
    -------
    float
        Outlier score (0-1, higher is better = fewer outliers).
    """
    X = _to_dense(X)
    
    X_valid = X[~np.isnan(X)]
    if len(X_valid) == 0:
        return 0.0
    
    mean = np.mean(X_valid)
    std = np.std(X_valid)
    
    if std == 0:
        return 1.0
    
    z_scores = np.abs((X_valid - mean) / std)
    outlier_count = np.sum(z_scores > threshold)
    outlier_rate = outlier_count / len(X_valid)
    
    # Convert to score (lower outlier rate = higher score)
    return float(max(0, 1 - outlier_rate * 10))
```

**Step 3: 创建测试文件**

```python
"""Tests for quality metrics."""

import numpy as np
import pytest
from scptensor.autoselect.metrics.quality import (
    cv_stability,
    skewness_improvement,
    dynamic_range,
    outlier_ratio,
)


class TestCVStability:
    """Tests for cv_stability function."""
    
    def test_stable_data(self):
        """Test with stable CV."""
        np.random.seed(42)
        X = np.random.randn(100, 50) * 0.1 + 10  # Low variance
        score = cv_stability(X)
        assert 0 <= score <= 1
    
    def test_unstable_data(self):
        """Test with unstable CV."""
        X = np.random.randn(100, 50) * 10 + 1  # High variance
        score = cv_stability(X)
        assert 0 <= score <= 1


class TestSkewnessImprovement:
    """Tests for skewness_improvement function."""
    
    def test_improvement(self):
        """Test skewness improvement."""
        X_before = np.exp(np.random.randn(100, 50))  # Right-skewed
        X_after = np.random.randn(100, 50)  # Normal
        score = skewness_improvement(X_before, X_after)
        assert 0 <= score <= 1


class TestDynamicRange:
    """Tests for dynamic_range function."""
    
    def test_optimal_range(self):
        """Test with optimal dynamic range."""
        X = np.random.lognormal(0, 0.5, (100, 50))
        score = dynamic_range(X)
        assert 0 <= score <= 1


class TestOutlierRatio:
    """Tests for outlier_ratio function."""
    
    def test_no_outliers(self):
        """Test with no outliers."""
        X = np.random.randn(100, 50)
        score = outlier_ratio(X)
        assert score > 0.9
    
    def test_with_outliers(self):
        """Test with outliers."""
        X = np.random.randn(100, 50)
        X[0, 0] = 100  # Add outlier
        score = outlier_ratio(X)
        assert 0 <= score <= 1
```

**Step 4: 运行测试**

```bash
uv run pytest tests/test_autoselect/test_metrics_quality.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add scptensor/autoselect/metrics/ tests/test_autoselect/test_metrics_quality.py
git commit -m "feat(autoselect): add quality metrics module"
```

---

## Phase 3-5: 后续任务

(由于篇幅限制，完整计划包含 15 个任务，涵盖评估器、主类实现和集成测试)

详细任务列表:
- Task 3: 批次效应指标 (batch.py)
- Task 4: 聚类指标 (clustering.py)
- Task 5: 评估器基类 (base.py)
- Task 6-10: 各环节评估器实现
- Task 11: AutoSelector 完整实现
- Task 12: 报告生成器
- Task 13: 快捷函数
- Task 14: 集成测试
- Task 15: 文档更新

---

## 执行选项

计划已保存。两种执行方式:

**1. Subagent-Driven (本会话)** - 每个任务派发新子代理，任务间审查

**2. Parallel Session (新会话)** - 在新会话中使用 executing-plans skill

选择哪种方式?
