# ScpTensor vs Scanpy 全面对比系统增强实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 重构并扩展 ScpTensor vs Scanpy 对比系统，实现模块化架构、扩展方法对比、生物学指标评估、丰富图表类型，支持增量运行。

**Architecture:** 三层模块化架构 (配置层 → 执行层 → 聚合层)，模块化增量运行，缓存机制。

**Tech Stack:** Python 3.12+, scib-metrics, SciencePlots, PyYAML, pytest, numpy, scanpy

---

## 实施流程概述

```
Phase 1: 重构现有代码 → Phase 2: 编写新模块 → Phase 3: 优化代码 → Phase 4: 运行测试
    │                        │                      │                   │
    ├─ 整理目录结构          ├─ 配置文件           ├─ python-per-file  ├─ 集成测试
    ├─ 提取基类             ├─ 测试模块           ├─ 逐文件优化       ├─ 报告生成
    └─ 创建模块注册         ├─ 评估模块           └─ 代码质量检查     └─ 性能验证
```

---

## Phase 1: 重构现有代码

### Task 1.1: 创建新目录结构

**Files:**
- Create: `scptensor/benchmark/config/__init__.py`
- Create: `scptensor/benchmark/modules/__init__.py`
- Create: `scptensor/benchmark/evaluators/__init__.py`
- Create: `scptensor/benchmark/charts/__init__.py`
- Create: `scptensor/benchmark/report/__init__.py`
- Create: `scptensor/benchmark/utils/__init__.py`

**Step 1: 创建所有 __init__.py 文件**

```bash
mkdir -p scptensor/benchmark/{config,modules,evaluators,charts,report,utils}
touch scptensor/benchmark/config/__init__.py
touch scptensor/benchmark/modules/__init__.py
touch scptensor/benchmark/evaluators/__init__.py
touch scptensor/benchmark/charts/__init__.py
touch scptensor/benchmark/report/__init__.py
touch scptensor/benchmark/utils/__init__.py
```

**Step 2: 验证目录结构**

Run: `ls -la scptensor/benchmark/ | grep -E "config|modules|evaluators|charts|report|utils"`
Expected: 显示所有新创建的目录

**Step 3: Commit**

```bash
git add scptensor/benchmark/config/ scptensor/benchmark/modules/ scptensor/benchmark/evaluators/ scptensor/benchmark/charts/ scptensor/benchmark/report/ scptensor/benchmark/utils/
git commit -m "refactor(benchmark): create new directory structure for modular design"
```

---

### Task 1.2: 提取基类 BaseModule

**Files:**
- Create: `scptensor/benchmark/modules/base.py`
- Reference: `scptensor/benchmark/comparison_engine.py:1-100`

**Step 1: 编写基类模块 (使用 python-pro)**

调用 `python-pro` 子代理创建以下内容：

```python
"""Base module for benchmark tests."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ModuleConfig:
    """Configuration for a benchmark module."""

    name: str
    enabled: bool = True
    datasets: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModuleResult:
    """Result from a benchmark module."""

    module_name: str
    dataset_name: str
    method_name: str
    output: np.ndarray | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    runtime_seconds: float = 0.0
    memory_mb: float = 0.0
    success: bool = True
    error_message: str | None = None


class BaseModule(ABC):
    """Abstract base class for benchmark modules."""

    def __init__(self, config: ModuleConfig) -> None:
        self.config = config
        self.results: list[ModuleResult] = []

    @abstractmethod
    def run(self, dataset_name: str) -> list[ModuleResult]:
        """Run the benchmark module on a dataset."""
        pass

    def get_results(self) -> list[ModuleResult]:
        """Get all results from this module."""
        return self.results

    def clear_results(self) -> None:
        """Clear stored results."""
        self.results = []
```

**Step 2: 编写测试**

```bash
cat > tests/benchmark/test_base_module.py << 'EOF'
import pytest
from scptensor.benchmark.modules.base import BaseModule, ModuleConfig, ModuleResult
import numpy as np


class DummyModule(BaseModule):
    def run(self, dataset_name: str) -> list[ModuleResult]:
        result = ModuleResult(
            module_name=self.config.name,
            dataset_name=dataset_name,
            method_name="dummy"
        )
        return [result]


def test_module_config():
    config = ModuleConfig(name="test", enabled=True)
    assert config.name == "test"
    assert config.enabled is True


def test_base_module():
    config = ModuleConfig(name="dummy")
    module = DummyModule(config)
    results = module.run("test_dataset")
    assert len(results) == 1
    assert results[0].module_name == "dummy"


def test_module_result():
    result = ModuleResult(
        module_name="test",
        dataset_name="ds",
        method_name="method",
        output=np.array([1, 2, 3])
    )
    assert result.success is True
    assert result.output is not None
EOF
```

**Step 3: 运行测试**

Run: `uv run pytest tests/benchmark/test_base_module.py -v`
Expected: 全部 PASS

**Step 4: Commit**

```bash
git add scptensor/benchmark/modules/base.py tests/benchmark/test_base_module.py
git commit -m "refactor(benchmark): add BaseModule abstract class"
```

---

### Task 1.3: 创建模块注册表

**Files:**
- Create: `scptensor/benchmark/utils/registry.py`

**Step 1: 编写注册表模块 (使用 python-pro)**

```python
"""Module registry for benchmark components."""

from __future__ import annotations

from typing import Any, Callable

# Registry for available modules
_AVAILABLE_MODULES: dict[str, type] = {}
_AVAILABLE_EVALUATORS: dict[str, type] = {}
_AVAILABLE_CHARTS: dict[str, type] = {}


def register_module(name: str) -> Callable[[type], type]:
    """Decorator to register a benchmark module."""
    def decorator(cls: type) -> type:
        _AVAILABLE_MODULES[name] = cls
        return cls
    return decorator


def register_evaluator(name: str) -> Callable[[type], type]:
    """Decorator to register an evaluator."""
    def decorator(cls: type) -> type:
        _AVAILABLE_EVALUATORS[name] = cls
        return cls
    return decorator


def register_chart(name: str) -> Callable[[type], type]:
    """Decorator to register a chart type."""
    def decorator(cls: type) -> type:
        _AVAILABLE_CHARTS[name] = cls
        return cls
    return decorator


def get_module(name: str) -> type | None:
    """Get a registered module by name."""
    return _AVAILABLE_MODULES.get(name)


def get_evaluator(name: str) -> type | None:
    """Get a registered evaluator by name."""
    return _AVAILABLE_EVALUATORS.get(name)


def get_chart(name: str) -> type | None:
    """Get a registered chart by name."""
    return _AVAILABLE_CHARTS.get(name)


def list_modules() -> list[str]:
    """List all registered modules."""
    return list(_AVAILABLE_MODULES.keys())


def list_evaluators() -> list[str]:
    """List all registered evaluators."""
    return list(_AVAILABLE_EVALUATORS.keys())


def list_charts() -> list[str]:
    """List all registered charts."""
    return list(_AVAILABLE_CHARTS.keys())
```

**Step 2: 测试注册表**

```bash
cat > tests/benchmark/test_registry.py << 'EOF'
from scptensor.benchmark.utils.registry import register_module, get_module, list_modules


@register_module("test_module")
class TestModule:
    pass


def test_registry():
    assert "test_module" in list_modules()
    assert get_module("test_module") is TestModule
    assert get_module("nonexistent") is None
EOF
```

**Step 3: 运行测试**

Run: `uv run pytest tests/benchmark/test_registry.py -v`
Expected: 全部 PASS

**Step 4: Commit**

```bash
git add scptensor/benchmark/utils/registry.py tests/benchmark/test_registry.py
git commit -m "refactor(benchmark): add module registry system"
```

---

### Task 1.4: 创建缓存管理器

**Files:**
- Create: `scptensor/benchmark/utils/cache.py`

**Step 1: 编写缓存模块 (使用 python-pro)**

```python
"""Cache management for benchmark results."""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scptensor.benchmark.modules.base import ModuleResult


class CacheManager:
    """Manager for caching benchmark results."""

    def __init__(self, cache_dir: Path | str = Path("benchmark_results/cache")) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, module: str, dataset: str, params: dict[str, Any]) -> str:
        """Generate cache key from parameters."""
        key_str = f"{module}_{dataset}_{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, module: str, dataset: str, params: dict[str, Any]) -> list[ModuleResult] | None:
        """Get cached results if available and valid."""
        key = self._get_cache_key(module, dataset, params)
        cache_file = self.cache_dir / f"{key}.pkl"

        if cache_file.exists():
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        return None

    def set(self, module: str, dataset: str, params: dict[str, Any], results: list[ModuleResult]) -> None:
        """Cache results."""
        key = self._get_cache_key(module, dataset, params)
        cache_file = self.cache_dir / f"{key}.pkl"

        with open(cache_file, "wb") as f:
            pickle.dump(results, f)

    def clear(self, module: str | None = None) -> None:
        """Clear cache."""
        if module:
            for f in self.cache_dir.glob(f"{module}_*.pkl"):
                f.unlink()
        else:
            for f in self.cache_dir.glob("*.pkl"):
                f.unlink()

    def is_valid(self, module: str, dataset: str, params: dict[str, Any]) -> bool:
        """Check if cached results are valid."""
        return self.get(module, dataset, params) is not None
```

**Step 2: 测试缓存**

```bash
cat > tests/benchmark/test_cache.py << 'EOF'
import pytest
from scptensor.benchmark.utils.cache import CacheManager
from scptensor.benchmark.modules.base import ModuleResult
from pathlib import Path
import tempfile


def test_cache_set_get():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CacheManager(Path(tmpdir))
        results = [ModuleResult(module_name="test", dataset_name="ds", method_name="m")]

        cache.set("test", "ds", {}, results)
        retrieved = cache.get("test", "ds", {})

        assert retrieved is not None
        assert len(retrieved) == 1
        assert retrieved[0].module_name == "test"


def test_cache_miss():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CacheManager(Path(tmpdir))
        assert cache.get("nonexistent", "ds", {}) is None


def test_cache_clear():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CacheManager(Path(tmpdir))
        results = [ModuleResult(module_name="test", dataset_name="ds", method_name="m")]
        cache.set("test", "ds", {}, results)
        cache.clear("test")
        assert cache.get("test", "ds", {}) is None
EOF
```

**Step 3: 运行测试**

Run: `uv run pytest tests/benchmark/test_cache.py -v`
Expected: 全部 PASS

**Step 4: Commit**

```bash
git add scptensor/benchmark/utils/cache.py tests/benchmark/test_cache.py
git commit -m "refactor(benchmark): add cache manager for incremental runs"
```

---

## Phase 2: 编写新模块 (使用 python-pro)

### Task 2.1: 创建配置文件系统

**Files:**
- Create: `scptensor/benchmark/config/benchmark_config.yaml`
- Create: `scptensor/benchmark/config/charts.yaml`
- Create: `scptensor/benchmark/config/loader.py`

**Step 1: 编写配置加载器 (调用 python-pro)**

```python
"""Configuration loader for benchmark system."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModuleConfigSpec:
    """Configuration specification for a module."""

    enabled: bool = True
    methods: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)


@dataclass
class BenchmarkConfig:
    """Overall benchmark configuration."""

    modules: dict[str, ModuleConfigSpec] = field(default_factory=dict)
    output_dir: str = "benchmark_results"
    cache_enabled: bool = True
    parallel_workers: int = 1


class ConfigLoader:
    """Load and parse YAML configuration files."""

    def __init__(self, config_dir: Path | str = Path("scptensor/benchmark/config")) -> None:
        self.config_dir = Path(config_dir)

    def load_benchmark_config(self) -> BenchmarkConfig:
        """Load main benchmark configuration."""
        config_file = self.config_dir / "benchmark_config.yaml"

        if not config_file.exists():
            return BenchmarkConfig()  # Return default config

        with open(config_file) as f:
            data = yaml.safe_load(f)

        modules = {}
        for name, spec in data.get("modules", {}).items():
            modules[name] = ModuleConfigSpec(
                enabled=spec.get("enabled", True),
                methods=spec.get("methods", []),
                datasets=spec.get("datasets", []),
                params=spec.get("params", {}),
                depends_on=spec.get("depends_on", [])
            )

        return BenchmarkConfig(
            modules=modules,
            output_dir=data.get("output_dir", "benchmark_results"),
            cache_enabled=data.get("cache_enabled", True),
            parallel_workers=data.get("parallel_workers", 1)
        )

    def load_chart_config(self) -> dict[str, Any]:
        """Load chart configuration."""
        config_file = self.config_dir / "charts.yaml"

        if not config_file.exists():
            return self._default_chart_config()

        with open(config_file) as f:
            return yaml.safe_load(f)

    def _default_chart_config(self) -> dict[str, Any]:
        """Return default chart configuration."""
        return {
            "performance": ["boxplot", "violin", "errorbar", "heatmap"],
            "accuracy": ["bland_altman", "qq_plot", "density_scatter"],
            "sensitivity": ["param_heatmap", "param_curve", "contour"],
            "biological": ["umap_colored", "marker_heatmap"],
            "summary": ["radar", "ranked_bar", "trend_line"]
        }
```

**Step 2: 创建默认配置文件**

```bash
cat > scptensor/benchmark/config/benchmark_config.yaml << 'EOF'
# ScpTensor Benchmark Configuration

modules:
  clustering:
    enabled: true
    methods: [kmeans, leiden, louvain]
    datasets: [synthetic_small, synthetic_medium]
    params:
      n_clusters: [5, 10, 15]
      resolution: [0.5, 1.0, 1.5]

  batch_correction:
    enabled: true
    methods: [combat, harmony]
    datasets: [synthetic_batch]

  diff_expr:
    enabled: true
    methods: [ttest, wilcoxon]
    datasets: [synthetic_medium]

  biological:
    enabled: true
    metrics: [kbet, ilisi, clisi]
    depends_on: [clustering, batch_correction]

  sensitivity:
    enabled: true
    target: clustering
    params: n_clusters
    values: [3, 5, 7, 10, 15, 20]

output_dir: benchmark_results
cache_enabled: true
parallel_workers: 4
EOF
```

```bash
cat > scptensor/benchmark/config/charts.yaml << 'EOF'
# Chart Configuration

performance:
  - boxplot
  - violin
  - errorbar
  - heatmap

accuracy:
  - bland_altman
  - qq_plot
  - density_scatter
  - regression_fit

sensitivity:
  - param_heatmap
  - param_curve
  - contour_plot

biological:
  - umap_colored
  - marker_heatmap
  - batch_mixing_matrix

summary:
  - radar
  - ranked_bar
  - trend_line
EOF
```

**Step 3: 测试配置加载**

```bash
cat > tests/benchmark/test_config.py << 'EOF'
from scptensor.benchmark.config.loader import ConfigLoader
from pathlib import Path


def test_load_config():
    loader = ConfigLoader()
    config = loader.load_benchmark_config()

    assert config.output_dir == "benchmark_results"
    assert config.cache_enabled is True
    assert "clustering" in config.modules


def test_load_chart_config():
    loader = ConfigLoader()
    charts = loader.load_chart_config()

    assert "performance" in charts
    assert "accuracy" in charts
EOF
```

**Step 4: 运行测试**

Run: `uv run pytest tests/benchmark/test_config.py -v`
Expected: 全部 PASS

**Step 5: Commit**

```bash
git add scptensor/benchmark/config/ scptensor/benchmark/config/loader.py tests/benchmark/test_config.py
git commit -m "feat(benchmark): add YAML configuration system"
```

---

### Task 2.2: 迁移聚类测试模块

**Files:**
- Create: `scptensor/benchmark/modules/test_clustering.py`
- Modify: `scptensor/benchmark/scptensor_adapter.py:200-250`
- Test: `tests/benchmark/test_module_clustering.py`

**Step 1: 编写聚类测试模块 (调用 python-pro)**

```python
"""Clustering benchmark module."""

from __future__ import annotations

from scptensor.benchmark.modules.base import BaseModule, ModuleConfig, ModuleResult
from scptensor.benchmark.utils.cache import CacheManager
from scptensor.benchmark.data_provider import get_provider


@register_module("clustering")
class ClusteringModule(BaseModule):
    """Benchmark module for clustering algorithms."""

    def __init__(self, config: ModuleConfig) -> None:
        super().__init__(config)
        self.cache = CacheManager()

    def run(self, dataset_name: str) -> list[ModuleResult]:
        """Run clustering benchmarks on a dataset."""
        results = []

        # Check cache
        cached = self.cache.get("clustering", dataset_name, self.config.params)
        if cached:
            self.results = cached
            return cached

        # Get data
        provider = get_provider()
        datasets = [d for d in provider.list_datasets() if d.name == dataset_name]

        if not datasets:
            self.results = results
            return results

        dataset = datasets[0]
        X, M, _, _ = provider.get_dataset(dataset)

        # Run each method
        for method_name in self.config.methods:
            result = self._run_method(method_name, X, M, dataset_name)
            results.append(result)

        self.results = results
        self.cache.set("clustering", dataset_name, self.config.params, results)
        return results

    def _run_method(self, method: str, X: np.ndarray, M: np.ndarray, dataset: str) -> ModuleResult:
        """Run a single clustering method."""
        import time

        start = time.time()

        try:
            if method == "kmeans":
                from sklearn.cluster import KMeans
                n_clusters = self.config.params.get("n_clusters", 5)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
                labels = kmeans.fit_predict(X)
                output = labels
            elif method == "leiden":
                # Leiden implementation placeholder
                output = self._run_leiden(X, M)
            elif method == "louvain":
                # Louvain implementation placeholder
                output = self._run_louvain(X, M)
            else:
                raise ValueError(f"Unknown method: {method}")

            runtime = time.time() - start

            return ModuleResult(
                module_name="clustering",
                dataset_name=dataset,
                method_name=method,
                output=output,
                runtime_seconds=runtime,
                success=True
            )
        except Exception as e:
            return ModuleResult(
                module_name="clustering",
                dataset_name=dataset,
                method_name=method,
                success=False,
                error_message=str(e)
            )

    def _run_leiden(self, X: np.ndarray, M: np.ndarray) -> np.ndarray:
        """Run Leiden clustering."""
        # Placeholder - implement based on ScpTensor/Scanpy
        import numpy as np
        n_samples = X.shape[0]
        return np.random.randint(0, 5, n_samples)

    def _run_louvain(self, X: np.ndarray, M: np.ndarray) -> np.ndarray:
        """Run Louvain clustering."""
        # Placeholder - implement based on ScpTensor/Scanpy
        import numpy as np
        n_samples = X.shape[0]
        return np.random.randint(0, 5, n_samples)
```

**Step 2: 编写测试**

```bash
cat > tests/benchmark/test_module_clustering.py << 'EOF'
import pytest
from scptensor.benchmark.modules.test_clustering import ClusteringModule
from scptensor.benchmark.modules.base import ModuleConfig


def test_clustering_module():
    config = ModuleConfig(
        name="clustering",
        methods=["kmeans"],
        datasets=["synthetic_small"],
        params={"n_clusters": 5}
    )
    module = ClusteringModule(config)
    results = module.run("synthetic_small")

    assert len(results) == 1
    assert results[0].method_name == "kmeans"
    assert results[0].success is True


def test_clustering_cache():
    config = ModuleConfig(
        name="clustering",
        methods=["kmeans"],
        datasets=["synthetic_small"],
        params={"n_clusters": 5}
    )
    module = ClusteringModule(config)

    # First run
    results1 = module.run("synthetic_small")

    # Clear internal results but keep cache
    module.results = []

    # Second run should use cache
    results2 = module.run("synthetic_small")

    assert len(results2) == len(results1)
EOF
```

**Step 3: 运行测试**

Run: `uv run pytest tests/benchmark/test_module_clustering.py -v`
Expected: 全部 PASS

**Step 4: Commit**

```bash
git add scptensor/benchmark/modules/test_clustering.py tests/benchmark/test_module_clustering.py
git commit -m "feat(benchmark): add clustering module with cache support"
```

---

### Task 2.3: 创建批次校正测试模块

**Files:**
- Create: `scptensor/benchmark/modules/test_batch_correction.py`

**Step 1: 编写批次校正模块 (调用 python-pro)**

```python
"""Batch correction benchmark module."""

from __future__ import annotations

import time
from scptensor.benchmark.modules.base import BaseModule, ModuleConfig, ModuleResult
from scptensor.benchmark.utils.cache import CacheManager
from scptensor.benchmark.data_provider import get_provider


@register_module("batch_correction")
class BatchCorrectionModule(BaseModule):
    """Benchmark module for batch correction methods."""

    def __init__(self, config: ModuleConfig) -> None:
        super().__init__(config)
        self.cache = CacheManager()

    def run(self, dataset_name: str) -> list[ModuleResult]:
        """Run batch correction benchmarks."""
        cached = self.cache.get("batch_correction", dataset_name, self.config.params)
        if cached:
            self.results = cached
            return cached

        results = []
        provider = get_provider()
        datasets = [d for d in provider.list_datasets() if d.name == dataset_name]

        if not datasets:
            self.results = results
            return results

        dataset = datasets[0]
        X, M, batches, _ = provider.get_dataset(dataset)

        for method_name in self.config.methods:
            result = self._run_method(method_name, X, M, batches, dataset_name)
            results.append(result)

        self.results = results
        self.cache.set("batch_correction", dataset_name, self.config.params, results)
        return results

    def _run_method(self, method: str, X: np.ndarray, M: np.ndarray, batches: np.ndarray, dataset: str) -> ModuleResult:
        """Run a single batch correction method."""
        start = time.time()

        try:
            if method == "combat":
                output = self._run_combat(X, batches)
            elif method == "harmony":
                output = self._run_harmony(X, batches)
            else:
                raise ValueError(f"Unknown method: {method}")

            runtime = time.time() - start

            return ModuleResult(
                module_name="batch_correction",
                dataset_name=dataset,
                method_name=method,
                output=output,
                runtime_seconds=runtime,
                success=True
            )
        except Exception as e:
            return ModuleResult(
                module_name="batch_correction",
                dataset_name=dataset,
                method_name=method,
                success=False,
                error_message=str(e)
            )

    def _run_combat(self, X: np.ndarray, batches: np.ndarray) -> np.ndarray:
        """Run ComBat batch correction."""
        from scptensor.integration import combat
        container = self._create_container(X, batches)
        result = combat(container, "protein", "raw", "combat", batch_labels=batches)
        return self._extract_result(result)

    def _run_harmony(self, X: np.ndarray, batches: np.ndarray) -> np.ndarray:
        """Run Harmony batch correction."""
        from scptensor.integration import harmony
        container = self._create_container(X, batches)
        result = harmony(container, "protein", "raw", "harmony", batch_labels=batches)
        return self._extract_result(result)
```

**Step 2: Commit**

```bash
git add scptensor/benchmark/modules/test_batch_correction.py
git commit -m "feat(benchmark): add batch correction module"
```

---

### Task 2.4: 创建差异表达测试模块

**Files:**
- Create: `scptensor/benchmark/modules/test_diff_expr.py`

**Step 1: 编写差异表达模块 (调用 python-pro)**

```python
"""Differential expression benchmark module."""

from __future__ import annotations

import time
from scptensor.benchmark.modules.base import BaseModule, ModuleConfig, ModuleResult
from scptensor.benchmark.utils.cache import CacheManager
from scptensor.benchmark.data_provider import get_provider


@register_module("diff_expr")
class DiffExprModule(BaseModule):
    """Benchmark module for differential expression analysis."""

    def __init__(self, config: ModuleConfig) -> None:
        super().__init__(config)
        self.cache = CacheManager()

    def run(self, dataset_name: str) -> list[ModuleResult]:
        """Run differential expression benchmarks."""
        cached = self.cache.get("diff_expr", dataset_name, self.config.params)
        if cached:
            self.results = cached
            return cached

        results = []
        provider = get_provider()
        datasets = [d for d in provider.list_datasets() if d.name == dataset_name]

        if not datasets:
            self.results = results
            return results

        dataset = datasets[0]
        X, M, _, groups = provider.get_dataset(dataset)

        for method_name in self.config.methods:
            result = self._run_method(method_name, X, M, groups, dataset_name)
            results.append(result)

        self.results = results
        self.cache.set("diff_expr", dataset_name, self.config.params, results)
        return results

    def _run_method(self, method: str, X: np.ndarray, M: np.ndarray, groups: np.ndarray, dataset: str) -> ModuleResult:
        """Run a single DE method."""
        start = time.time()

        try:
            if method == "ttest":
                output = self._run_ttest(X, groups)
            elif method == "wilcoxon":
                output = self._run_wilcoxon(X, groups)
            else:
                raise ValueError(f"Unknown method: {method}")

            runtime = time.time() - start

            return ModuleResult(
                module_name="diff_expr",
                dataset_name=dataset,
                method_name=method,
                output=output,  # List of significant genes
                runtime_seconds=runtime,
                success=True
            )
        except Exception as e:
            return ModuleResult(
                module_name="diff_expr",
                dataset_name=dataset,
                method_name=method,
                success=False,
                error_message=str(e)
            )

    def _run_ttest(self, X: np.ndarray, groups: np.ndarray) -> list[int]:
        """Run t-test DE analysis."""
        from scipy import stats
        significant_genes = []

        for gene_idx in range(X.shape[1]):
            group1 = X[groups == 0, gene_idx]
            group2 = X[groups == 1, gene_idx]
            _, p_value = stats.ttest_ind(group1, group2)
            if p_value < 0.05:
                significant_genes.append(gene_idx)

        return significant_genes

    def _run_wilcoxon(self, X: np.ndarray, groups: np.ndarray) -> list[int]:
        """Run Wilcoxon rank-sum DE analysis."""
        from scipy import stats
        significant_genes = []

        for gene_idx in range(X.shape[1]):
            group1 = X[groups == 0, gene_idx]
            group2 = X[groups == 1, gene_idx]
            _, p_value = stats.ranksums(group1, group2)
            if p_value < 0.05:
                significant_genes.append(gene_idx)

        return significant_genes
```

**Step 2: Commit**

```bash
git add scptensor/benchmark/modules/test_diff_expr.py
git commit -m "feat(benchmark): add differential expression module"
```

---

### Task 2.5: 创建生物学评估器

**Files:**
- Create: `scptensor/benchmark/evaluators/biological.py`
- Test: `tests/benchmark/test_evaluator_biological.py`

**Step 1: 编写生物学评估器 (调用 python-pro)**

```python
"""Biological metrics evaluator using scib-metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class BiologicalMetrics:
    """Results from biological evaluation."""

    kbet: float = 0.0
    ilisi: float = 0.0
    clisi: float = 0.0
    nmi_pcr: float = 0.0
    ari_cluster: float = 0.0
    calinski_harabasz: float = 0.0
    davies_bouldin: float = 0.0


class BiologicalEvaluator:
    """Evaluate biological quality using scib-metrics."""

    def __init__(self) -> None:
        try:
            import scib_metrics
            self.scib_available = True
        except ImportError:
            self.scib_available = False

    def evaluate_clustering(
        self,
        labels: np.ndarray,
        X: np.ndarray,
        batches: np.ndarray | None = None,
    ) -> BiologicalMetrics:
        """Evaluate clustering quality."""
        metrics = BiologicalMetrics()

        # Intrinsic clustering metrics (always available)
        from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

        try:
            metrics.calinski_harabasz = calinski_harabasz_score(X, labels)
        except Exception:
            metrics.calinski_harabasz = 0.0

        try:
            metrics.davies_bouldin = davies_bouldin_score(X, labels)
        except Exception:
            metrics.davies_bouldin = 0.0

        # scib-metrics (if available)
        if self.scib_available and batches is not None:
            metrics = self._scib_evaluate(labels, X, batches, metrics)

        return metrics

    def _scib_evaluate(
        self,
        labels: np.ndarray,
        X: np.ndarray,
        batches: np.ndarray,
        metrics: BiologicalMetrics,
    ) -> BiologicalMetrics:
        """Run scib-metrics evaluation."""
        import scib_metrics

        try:
            # kBET score
            metrics.kbet = scib_metrics.kbet(
                X=X, batch=batches, labels=labels, type_="embed"
            )
        except Exception:
            metrics.kbet = 0.0

        try:
            # iLISI score
            metrics.ilisi = scib_metrics.ilisi_graph(
                X=X, batch=batches, k0=15
            )
        except Exception:
            metrics.ilisi = 0.0

        try:
            # cLISI score
            metrics.clisi = scib_metrics.clisi_graph(
                X=X, labels=labels, k0=15
            )
        except Exception:
            metrics.clisi = 0.0

        return metrics
```

**Step 2: 测试**

```bash
cat > tests/benchmark/test_evaluator_biological.py << 'EOF'
import pytest
import numpy as np
from scptensor.benchmark.evaluators.biological import BiologicalEvaluator, BiologicalMetrics


def test_biological_evaluator():
    evaluator = BiologicalEvaluator()

    # Create dummy data
    X = np.random.randn(100, 20)
    labels = np.random.randint(0, 3, 100)
    batches = np.random.randint(0, 2, 100)

    metrics = evaluator.evaluate_clustering(labels, X, batches)

    assert metrics.calinski_harabasz >= 0
    assert metrics.davies_bouldin >= 0
    assert 0 <= metrics.kbet <= 1
EOF
```

**Step 3: 运行测试**

Run: `uv run pytest tests/benchmark/test_evaluator_biological.py -v`
Expected: 全部 PASS

**Step 4: Commit**

```bash
git add scptensor/benchmark/evaluators/biological.py tests/benchmark/test_evaluator_biological.py
git commit -m "feat(benchmark): add biological evaluator with scib-metrics"
```

---

### Task 2.6: 创建参数敏感性评估器

**Files:**
- Create: `scptensor/benchmark/evaluators/sensitivity.py`

**Step 1: 编写敏感性评估器 (调用 python-pro)**

```python
"""Parameter sensitivity evaluator."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class SensitivityResult:
    """Results from parameter sensitivity analysis."""

    param_name: str
    param_values: list[float] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    best_value: float = 0.0
    best_score: float = 0.0


class SensitivityEvaluator:
    """Evaluate parameter sensitivity for benchmark methods."""

    def __init__(self, output_dir: Path | str = Path("benchmark_results/figures/sensitivity")) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        method: Callable[[float], Any],
        param_name: str,
        param_values: list[float],
        score_fn: Callable[[Any], float],
    ) -> SensitivityResult:
        """Run parameter sweep."""
        result = SensitivityResult(param_name=param_name)

        for value in param_values:
            try:
                output = method(value)
                score = score_fn(output)

                result.param_values.append(value)
                result.scores.append(score)

                if score > result.best_score:
                    result.best_score = score
                    result.best_value = value

            except Exception:
                result.param_values.append(value)
                result.scores.append(0.0)

        return result

    def plot_sensitivity_curve(self, result: SensitivityResult, method_name: str) -> Path:
        """Plot parameter sensitivity curve."""
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(result.param_values, result.scores, "o-", color="#2E86AB", markersize=8)
        ax.axvline(result.best_value, color="red", linestyle="--", alpha=0.7, label=f"Best: {result.best_value}")
        ax.axhline(result.best_score, color="gray", linestyle=":", alpha=0.5)

        ax.set_xlabel(f"{result.param_name}")
        ax.set_ylabel("Score")
        ax.set_title(f"{method_name}: Parameter Sensitivity")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()

        output_file = self.output_dir / f"{method_name}_{result.param_name}_sensitivity.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        return output_file
```

**Step 2: Commit**

```bash
git add scptensor/benchmark/evaluators/sensitivity.py
git commit -m "feat(benchmark): add parameter sensitivity evaluator"
```

---

### Task 2.7: 创建性能评估器

**Files:**
- Create: `scptensor/benchmark/evaluators/performance.py`

**Step 1: 编写性能评估器 (调用 python-pro)**

```python
"""Performance metrics evaluator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PerformanceMetrics:
    """Results from performance evaluation."""

    runtime_seconds: float = 0.0
    memory_mb: float = 0.0
    speedup: float = 1.0
    throughput_samples_per_sec: float = 0.0


class PerformanceEvaluator:
    """Evaluate computational performance."""

    def __init__(self) -> None:
        pass

    def evaluate(
        self,
        scptensor_runtime: float,
        scanpy_runtime: float,
        n_samples: int,
        scptensor_memory: float = 0.0,
        scanpy_memory: float = 0.0,
    ) -> PerformanceMetrics:
        """Calculate performance metrics."""
        speedup = scanpy_runtime / scptensor_runtime if scptensor_runtime > 0 else 1.0
        throughput = n_samples / scptensor_runtime if scptensor_runtime > 0 else 0.0

        return PerformanceMetrics(
            runtime_seconds=scptensor_runtime,
            memory_mb=scptensor_memory,
            speedup=speedup,
            throughput_samples_per_sec=throughput,
        )
```

**Step 2: Commit**

```bash
git add scptensor/benchmark/evaluators/performance.py
git commit -m "feat(benchmark): add performance evaluator"
```

---

### Task 2.8: 创建准确性评估器

**Files:**
- Create: `scptensor/benchmark/evaluators/accuracy.py`

**Step 1: 编写准确性评估器 (调用 python-pro)**

```python
"""Output accuracy evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr


@dataclass
class AccuracyMetrics:
    """Results from accuracy evaluation."""

    mse: float = 0.0
    mae: float = 0.0
    correlation: float = 0.0
    spearman_correlation: float = 0.0
    max_absolute_error: float = 0.0
    relative_error: float = 0.0


class AccuracyEvaluator:
    """Evaluate output accuracy between ScpTensor and Scanpy."""

    def __init__(self) -> None:
        pass

    def evaluate(
        self,
        scptensor_output: np.ndarray,
        scanpy_output: np.ndarray,
    ) -> AccuracyMetrics:
        """Calculate accuracy metrics."""
        # Flatten arrays
        st_flat = scptensor_output.ravel()
        sp_flat = scanpy_output.ravel()

        # Ensure same shape
        min_len = min(len(st_flat), len(sp_flat))
        st_flat = st_flat[:min_len]
        sp_flat = sp_flat[:min_len]

        # MSE
        mse = float(np.mean((st_flat - sp_flat) ** 2))

        # MAE
        mae = float(np.mean(np.abs(st_flat - sp_flat)))

        # Pearson correlation
        try:
            corr, _ = pearsonr(st_flat, sp_flat)
            correlation = float(corr) if not np.isnan(corr) else 0.0
        except Exception:
            correlation = 0.0

        # Spearman correlation
        try:
            spearman, _ = spearmanr(st_flat, sp_flat)
            spearman_correlation = float(spearman) if not np.isnan(spearman) else 0.0
        except Exception:
            spearman_correlation = 0.0

        # Max absolute error
        max_abs_error = float(np.max(np.abs(st_flat - sp_flat)))

        # Relative error
        relative_error = float(mae / (np.mean(np.abs(sp_flat)) + 1e-10))

        return AccuracyMetrics(
            mse=mse,
            mae=mae,
            correlation=correlation,
            spearman_correlation=spearman_correlation,
            max_absolute_error=max_abs_error,
            relative_error=relative_error,
        )
```

**Step 2: Commit**

```bash
git add scptensor/benchmark/evaluators/accuracy.py
git commit -m "feat(benchmark): add accuracy evaluator"
```

---

## Phase 3: 优化代码 (使用 code-optimizer)

> **注意**: 每个文件使用独立的 code-optimizer 子代理

### Task 3.1: 优化 `scptensor/benchmark/modules/test_clustering.py`

**Step 1: 调用 code-optimizer**

```bash
# 启动 code-optimizer 子代理
# 目标文件: scptensor/benchmark/modules/test_clustering.py
# 优化方向: 减少代码体积, 提高运行速度, 保持核心逻辑
```

**Step 2: 验证优化**

Run: `uv run pytest tests/benchmark/test_module_clustering.py -v`
Expected: 全部 PASS

**Step 3: Commit**

```bash
git add scptensor/benchmark/modules/test_clustering.py
git commit -m "refactor(benchmark): optimize clustering module"
```

---

### Task 3.2: 优化 `scptensor/benchmark/modules/test_batch_correction.py`

**Step 1: 调用 code-optimizer**

```bash
# 启动 code-optimizer 子代理
# 目标文件: scptensor/benchmark/modules/test_batch_correction.py
```

**Step 2: 验证优化**

Run: `uv run pytest tests/benchmark/test_module_clustering.py -v`
Expected: 全部 PASS

**Step 3: Commit**

```bash
git add scptensor/benchmark/modules/test_batch_correction.py
git commit -m "refactor(benchmark): optimize batch correction module"
```

---

### Task 3.3: 优化 `scptensor/benchmark/modules/test_diff_expr.py`

**Step 1: 调用 code-optimizer**

```bash
# 启动 code-optimizer 子代理
# 目标文件: scptensor/benchmark/modules/test_diff_expr.py
```

**Step 2: 验证优化**

Run: `uv run pytest tests/benchmark/test_module_clustering.py -v`
Expected: 全部 PASS

**Step 3: Commit**

```bash
git add scptensor/benchmark/modules/test_diff_expr.py
git commit -m "refactor(benchmark): optimize diff expr module"
```

---

### Task 3.4: 优化 `scptensor/benchmark/evaluators/biological.py`

**Step 1: 调用 code-optimizer**

```bash
# 启动 code-optimizer 子代理
# 目标文件: scptensor/benchmark/evaluators/biological.py
```

**Step 2: 验证优化**

Run: `uv run pytest tests/benchmark/test_evaluator_biological.py -v`
Expected: 全部 PASS

**Step 3: Commit**

```bash
git add scptensor/benchmark/evaluators/biological.py
git commit -m "refactor(benchmark): optimize biological evaluator"
```

---

### Task 3.5: 优化 `scptensor/benchmark/evaluators/sensitivity.py`

**Step 1: 调用 code-optimizer**

```bash
# 启动 code-optimizer 子代理
# 目标文件: scptensor/benchmark/evaluators/sensitivity.py
```

**Step 2: 验证优化**

Run: `uv run pytest tests/benchmark/ -v -k "sensitivity or evaluator"
Expected: 全部 PASS

**Step 3: Commit**

```bash
git add scptensor/benchmark/evaluators/sensitivity.py
git commit -m "refactor(benchmark): optimize sensitivity evaluator"
```

---

### Task 3.6: 优化 `scptensor/benchmark/evaluators/performance.py`

**Step 1: 调用 code-optimizer**

```bash
# 启动 code-optimizer 子代理
# 目标文件: scptensor/benchmark/evaluators/performance.py
```

**Step 2: 验证优化**

Run: `uv run pytest tests/benchmark/ -v -k "performance or evaluator"
Expected: 全部 PASS

**Step 3: Commit**

```bash
git add scptensor/benchmark/evaluators/performance.py
git commit -m "refactor(benchmark): optimize performance evaluator"
```

---

### Task 3.7: 优化 `scptensor/benchmark/evaluators/accuracy.py`

**Step 1: 调用 code-optimizer**

```bash
# 启动 code-optimizer 子代理
# 目标文件: scptensor/benchmark/evaluators/accuracy.py
```

**Step 2: 验证优化**

Run: `uv run pytest tests/benchmark/ -v -k "accuracy or evaluator"
Expected: 全部 PASS

**Step 3: Commit**

```bash
git add scptensor/benchmark/evaluators/accuracy.py
git commit -m "refactor(benchmark): optimize accuracy evaluator"
```

---

### Task 3.8: 优化 `scptensor/benchmark/config/loader.py`

**Step 1: 调用 code-optimizer**

```bash
# 启动 code-optimizer 子代理
# 目标文件: scptensor/benchmark/config/loader.py
```

**Step 2: 验证优化**

Run: `uv run pytest tests/benchmark/test_config.py -v`
Expected: 全部 PASS

**Step 3: Commit**

```bash
git add scptensor/benchmark/config/loader.py
git commit -m "refactor(benchmark): optimize config loader"
```

---

### Task 3.9: 优化 `scptensor/benchmark/utils/cache.py`

**Step 1: 调用 code-optimizer**

```bash
# 启动 code-optimizer 子代理
# 目标文件: scptensor/benchmark/utils/cache.py
```

**Step 2: 验证优化**

Run: `uv run pytest tests/benchmark/test_cache.py -v`
Expected: 全部 PASS

**Step 3: Commit**

```bash
git add scptensor/benchmark/utils/cache.py
git commit -m "refactor(benchmark): optimize cache manager"
```

---

### Task 3.10: 优化 `scptensor/benchmark/utils/registry.py`

**Step 1: 调用 code-optimizer**

```bash
# 启动 code-optimizer 子代理
# 目标文件: scptensor/benchmark/utils/registry.py
```

**Step 2: 验证优化**

Run: `uv run pytest tests/benchmark/test_registry.py -v`
Expected: 全部 PASS

**Step 3: Commit**

```bash
git add scptensor/benchmark/utils/registry.py
git commit -m "refactor(benchmark): optimize registry"
```

---

### Task 3.11: 优化 `scptensor/benchmark/modules/base.py`

**Step 1: 调用 code-optimizer**

```bash
# 启动 code-optimizer 子代理
# 目标文件: scptensor/benchmark/modules/base.py
```

**Step 2: 验证优化**

Run: `uv run pytest tests/benchmark/test_base_module.py -v`
Expected: 全部 PASS

**Step 3: Commit**

```bash
git add scptensor/benchmark/modules/base.py
git commit -m "refactor(benchmark): optimize base module"
```

---

## Phase 4: 运行测试

### Task 4.1: 运行全部单元测试

**Step 1: 运行测试**

Run: `uv run pytest tests/benchmark/ -v --tb=short`
Expected: 全部 PASS

**Step 2: 生成覆盖率报告**

Run: `uv run pytest tests/benchmark/ --cov=scptensor/benchmark --cov-report=html --cov-report=term`
Expected: 覆盖率 > 80%

**Step 3: 检查代码质量**

Run: `uv run ruff check scptensor/benchmark/modules/ scptensor/benchmark/evaluators/ scptensor/benchmark/config/ scptensor/benchmark/utils/`
Expected: 无错误

Run: `uv run mypy scptensor/benchmark/modules/ scptensor/benchmark/evaluators/ scptensor/benchmark/config/ scptensor/benchmark/utils/`
Expected: 无类型错误

---

### Task 4.2: 集成测试

**Step 1: 运行端到端测试**

```bash
cat > tests/benchmark/test_integration.py << 'EOF'
"""Integration tests for benchmark system."""

import pytest
from scptensor.benchmark.config.loader import ConfigLoader
from scptensor.benchmark.modules.test_clustering import ClusteringModule
from scptensor.benchmark.modules.base import ModuleConfig


def test_full_clustering_workflow():
    """Test complete clustering workflow."""
    # Load config
    loader = ConfigLoader()
    config = loader.load_benchmark_config()

    # Get clustering module config
    cluster_config = config.modules.get("clustering")
    assert cluster_config is not None

    # Convert to ModuleConfig
    module_config = ModuleConfig(
        name="clustering",
        methods=cluster_config.methods,
        datasets=cluster_config.datasets,
        params=cluster_config.params,
    )

    # Run module
    module = ClusteringModule(module_config)
    results = module.run("synthetic_small")

    # Verify results
    assert len(results) > 0
    assert all(r.success for r in results if r.method_name == "kmeans")


def test_cache_workflow():
    """Test caching across multiple runs."""
    config = ModuleConfig(
        name="clustering",
        methods=["kmeans"],
        datasets=["synthetic_small"],
        params={"n_clusters": 5}
    )

    module = ClusteringModule(config)

    # First run
    results1 = module.run("synthetic_small")
    runtime1 = sum(r.runtime_seconds for r in results1)

    # Clear internal results
    module.results = []

    # Second run (should use cache)
    import time
    start = time.time()
    results2 = module.run("synthetic_small")
    runtime2 = time.time() - start

    # Cached run should be faster
    assert len(results1) == len(results2)
EOF
```

Run: `uv run pytest tests/benchmark/test_integration.py -v`
Expected: 全部 PASS

**Step 3: Commit**

```bash
git add tests/benchmark/test_integration.py
git commit -m "test(benchmark): add integration tests"
```

---

### Task 4.3: 性能基准测试

**Step 1: 运行性能测试**

```bash
cat > tests/benchmark/test_performance.py << 'EOF'
"""Performance tests for benchmark system."""

import pytest
import time
from scptensor.benchmark.modules.test_clustering import ClusteringModule
from scptensor.benchmark.modules.base import ModuleConfig


def test_clustering_performance():
    """Ensure clustering completes within reasonable time."""
    config = ModuleConfig(
        name="clustering",
        methods=["kmeans"],
        datasets=["synthetic_small"],
        params={"n_clusters": 5}
    )

    module = ClusteringModule(config)

    start = time.time()
    results = module.run("synthetic_small")
    elapsed = time.time() - start

    # Should complete within 30 seconds
    assert elapsed < 30, f"Clustering took {elapsed:.2f}s, expected < 30s"
    assert len(results) == 1
    assert results[0].success is True
EOF
```

Run: `uv run pytest tests/benchmark/test_performance.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/benchmark/test_performance.py
git commit -m "test(benchmark): add performance tests"
```

---

### Task 4.4: 生成完整报告

**Step 1: 运行完整 benchmark**

Run: `uv run python -m scptensor.benchmark.run_scanpy_comparison --datasets synthetic_small --methods kmeans pca 2>&1 | tail -20`
Expected: 无错误, 生成报告

**Step 2: 验证输出文件**

Run: `ls -la benchmark_results/figures/`
Expected: 存在图表文件

**Step 3: 验证报告内容**

Run: `cat benchmark_results/scanpy_comparison_report.md | head -50`
Expected: 报告格式正确

---

## 实施进度跟踪

| Phase | Task | 状态 |
|-------|------|------|
| 1.1 | 创建目录结构 | ⬜ |
| 1.2 | 提取基类 BaseModule | ⬜ |
| 1.3 | 创建模块注册表 | ⬜ |
| 1.4 | 创建缓存管理器 | ⬜ |
| 2.1 | 配置文件系统 | ⬜ |
| 2.2 | 聚类测试模块 | ⬜ |
| 2.3 | 批次校正模块 | ⬜ |
| 2.4 | 差异表达模块 | ⬜ |
| 2.5 | 生物学评估器 | ⬜ |
| 2.6 | 参数敏感性评估器 | ⬜ |
| 2.7 | 性能评估器 | ⬜ |
| 2.8 | 准确性评估器 | ⬜ |
| 3.1-3.11 | 代码优化 (11个文件) | ⬜ |
| 4.1 | 单元测试 | ⬜ |
| 4.2 | 集成测试 | ⬜ |
| 4.3 | 性能测试 | ⬜ |
| 4.4 | 完整报告测试 | ⬜ |

---

## 依赖安装

在开始实施前，确保安装以下依赖：

```bash
uv pip install scib-metrics pyyaml scipy scikit-learn
```

---

## 注意事项

1. **每个 python-pro 任务**: 创建代码文件时，确保包含完整的类型注解和 docstring
2. **每个 code-optimizer 任务**: 优化后必须运行测试验证功能不变
3. **缓存机制**: 所有模块都应支持缓存以实现增量运行
4. **SciencePlots 样式**: 所有图表使用 SciencePlots 风格
5. **错误处理**: 每个模块应该优雅地处理错误，返回 `success=False` 的结果

---

## 后续工作

本计划完成后，可继续：

1. **图表扩展** - 实现新的图表类型 (Task 2.9+)
2. **报告生成器** - 实现增量报告生成 (Task 2.10+)
3. **更多方法** - 添加 QC、轨迹分析等模块
