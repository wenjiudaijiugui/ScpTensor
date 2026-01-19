"""Tests for benchmark cache manager."""

import pickle
import pytest
import tempfile
from pathlib import Path

from scptensor.benchmark.modules import ModuleResult
from scptensor.benchmark.utils.cache import CacheManager
import numpy as np


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_cache_manager_initialization(temp_cache_dir):
    """Test CacheManager initialization."""
    cache = CacheManager(temp_cache_dir)
    assert cache.cache_dir == temp_cache_dir
    assert temp_cache_dir.exists()


def test_cache_manager_default_dir():
    """Test CacheManager with default directory."""
    cache = CacheManager()
    assert cache.cache_dir == Path("benchmark_results/cache")


def test_cache_key_generation(temp_cache_dir):
    """Test cache key generation."""
    cache = CacheManager(temp_cache_dir)

    key1 = cache.get_cache_key("test_module", "test_dataset", {"n": 5})
    key2 = cache.get_cache_key("test_module", "test_dataset", {"n": 5})
    key3 = cache.get_cache_key("test_module", "test_dataset", {"n": 10})

    # Same params should generate same key
    assert key1 == key2
    # Different params should generate different key
    assert key1 != key3
    # Keys should be MD5 hex strings (32 chars)
    assert len(key1) == 32
    assert all(c in "0123456789abcdef" for c in key1)


def test_cache_set_and_get(temp_cache_dir):
    """Test setting and getting cache."""
    cache = CacheManager(temp_cache_dir)
    results = [
        ModuleResult(
            module_name="test",
            dataset_name="ds",
            method_name="m1",
            output=np.array([1, 2, 3]),
        ),
        ModuleResult(
            module_name="test",
            dataset_name="ds",
            method_name="m2",
            output=np.array([4, 5, 6]),
        ),
    ]

    cache.set("test_module", "test_dataset", {}, results)
    retrieved = cache.get("test_module", "test_dataset", {})

    assert retrieved is not None
    assert len(retrieved) == 2
    assert retrieved[0].module_name == "test"
    assert retrieved[0].method_name == "m1"
    assert np.array_equal(retrieved[0].output, np.array([1, 2, 3]))


def test_cache_miss(temp_cache_dir):
    """Test getting nonexistent cache."""
    cache = CacheManager(temp_cache_dir)
    assert cache.get("nonexistent", "ds", {}) is None


def test_cache_is_valid(temp_cache_dir):
    """Test cache validity check."""
    cache = CacheManager(temp_cache_dir)
    results = [ModuleResult(module_name="test", dataset_name="ds", method_name="m")]

    assert cache.is_valid("test", "ds", {}) is False

    cache.set("test", "ds", {}, results)
    assert cache.is_valid("test", "ds", {}) is True


def test_cache_clear_module(temp_cache_dir):
    """Test clearing cache for a specific module."""
    cache = CacheManager(temp_cache_dir)

    # Create results with different module names
    result1 = [ModuleResult(module_name="module1", dataset_name="ds", method_name="m")]
    result2 = [ModuleResult(module_name="module2", dataset_name="ds", method_name="m")]

    # Set cache with different module names
    cache.set("test1", "ds", {}, result1)
    cache.set("test2", "ds", {}, result2)

    # Check both are valid
    assert cache.is_valid("test1", "ds", {}) is True
    assert cache.is_valid("test2", "ds", {}) is True

    # Clear module1
    cache.clear("module1")

    # module1 cache should be cleared, module2 should remain
    assert cache.is_valid("test1", "ds", {}) is False
    assert cache.is_valid("test2", "ds", {}) is True


def test_cache_clear_all(temp_cache_dir):
    """Test clearing all cache."""
    cache = CacheManager(temp_cache_dir)
    results = [ModuleResult(module_name="test", dataset_name="ds", method_name="m")]

    cache.set("module1", "ds", {}, results)
    cache.set("module2", "ds", {}, results)

    cache.clear()  # Clear all

    assert cache.is_valid("module1", "ds", {}) is False
    assert cache.is_valid("module2", "ds", {}) is False


def test_cache_params_ordering(temp_cache_dir):
    """Test that params with different orders generate same key."""
    cache = CacheManager(temp_cache_dir)

    # Different order should generate same key
    key1 = cache.get_cache_key("mod", "ds", {"a": 1, "b": 2})
    key2 = cache.get_cache_key("mod", "ds", {"b": 2, "a": 1})

    assert key1 == key2


def test_cache_with_numpy_output(temp_cache_dir):
    """Test caching results with numpy array outputs."""
    cache = CacheManager(temp_cache_dir)
    results = [
        ModuleResult(
            module_name="test",
            dataset_name="ds",
            method_name="m",
            output=np.random.randn(100, 50),
            runtime_seconds=1.5,
        )
    ]

    cache.set("test", "ds", {"n": 50}, results)
    retrieved = cache.get("test", "ds", {"n": 50})

    assert retrieved is not None
    assert len(retrieved) == 1
    assert retrieved[0].output.shape == (100, 50)
    assert retrieved[0].runtime_seconds == 1.5


def test_cache_info(temp_cache_dir):
    """Test getting cache information."""
    cache = CacheManager(temp_cache_dir)

    # Empty cache
    info = cache.get_cache_info()
    assert info["total_files"] == 0
    assert info["total_size_mb"] == 0.0

    # Add some cache
    results = [ModuleResult(module_name="test", dataset_name="ds", method_name="m")]
    cache.set("module1", "ds", {}, results)
    cache.set("module2", "ds", {}, results)

    info = cache.get_cache_info()
    assert info["total_files"] == 2
    assert info["total_size_mb"] > 0
    assert "module_counts" in info


def test_cache_prune_by_size(temp_cache_dir):
    """Test pruning cache by size."""
    cache = CacheManager(temp_cache_dir)
    results = [ModuleResult(module_name="test", dataset_name="ds", method_name="m")]

    # Add multiple cache entries
    for i in range(5):
        cache.set(f"module{i}", "ds", {}, results)

    # Prune to max 3 files
    removed = cache.prune_by_size(max_size_mb=0.001)  # Very small limit to force pruning

    # Check that cache was pruned (removed some files)
    assert removed > 0
