"""Tests for benchmark base module."""

import pytest
import numpy as np
from scptensor.benchmark.modules import BaseModule, ModuleConfig, ModuleResult


class DummyModule(BaseModule):
    """Dummy module for testing."""

    def run(self, dataset_name: str) -> list[ModuleResult]:
        """Run dummy module."""
        result = ModuleResult(
            module_name=self.config.name,
            dataset_name=dataset_name,
            method_name="dummy",
            output=np.array([1, 2, 3]),
        )
        self._add_result(result)
        return [result]


def test_module_config():
    """Test ModuleConfig creation."""
    config = ModuleConfig(name="test", enabled=True)
    assert config.name == "test"
    assert config.enabled is True
    assert config.datasets == []
    assert config.methods == []
    assert config.params == {}


def test_module_config_with_params():
    """Test ModuleConfig with parameters."""
    config = ModuleConfig(
        name="test",
        datasets=["ds1", "ds2"],
        methods=["method1"],
        params={"n_clusters": 5},
    )
    assert len(config.datasets) == 2
    assert len(config.methods) == 1
    assert config.params["n_clusters"] == 5


def test_module_result():
    """Test ModuleResult creation."""
    result = ModuleResult(
        module_name="test",
        dataset_name="ds",
        method_name="method",
        output=np.array([1, 2, 3]),
        runtime_seconds=1.5,
        memory_mb=100.0,
    )
    assert result.module_name == "test"
    assert result.dataset_name == "ds"
    assert result.method_name == "method"
    assert result.success is True
    assert result.runtime_seconds == 1.5
    assert result.memory_mb == 100.0
    assert np.array_equal(result.output, np.array([1, 2, 3]))


def test_module_result_to_dict():
    """Test ModuleResult serialization."""
    result = ModuleResult(
        module_name="test",
        dataset_name="ds",
        method_name="method",
        output=np.array([1, 2, 3]),
    )
    result_dict = result.to_dict()
    assert result_dict["module_name"] == "test"
    assert result_dict["success"] is True
    assert "output" in result_dict


def test_base_module():
    """Test BaseModule with dummy implementation."""
    config = ModuleConfig(name="dummy")
    module = DummyModule(config)
    results = module.run("test_dataset")

    assert len(results) == 1
    assert results[0].module_name == "dummy"
    assert results[0].dataset_name == "test_dataset"
    assert results[0].method_name == "dummy"


def test_base_module_get_results():
    """Test get_results method."""
    config = ModuleConfig(name="dummy")
    module = DummyModule(config)

    module.run("test_dataset")
    results = module.get_results()

    assert len(results) == 1


def test_base_module_clear_results():
    """Test clear_results method."""
    config = ModuleConfig(name="dummy")
    module = DummyModule(config)

    module.run("test_dataset")
    assert len(module.get_results()) == 1

    module.clear_results()
    assert len(module.get_results()) == 0


def test_base_module_is_enabled():
    """Test is_enabled method."""
    config_enabled = ModuleConfig(name="test", enabled=True)
    config_disabled = ModuleConfig(name="test", enabled=False)

    module_enabled = DummyModule(config_enabled)
    module_disabled = DummyModule(config_disabled)

    assert module_enabled.is_enabled() is True
    assert module_disabled.is_enabled() is False


def test_base_module_should_process():
    """Test dataset/method filtering methods."""
    config = ModuleConfig(
        name="test",
        datasets=["ds1", "ds2"],
        methods=["m1", "m2"],
    )
    module = DummyModule(config)

    assert module.should_process_dataset("ds1") is True
    assert module.should_process_dataset("ds3") is False
    assert module.should_process_method("m1") is True
    assert module.should_process_method("m3") is False


def test_module_result_error():
    """Test ModuleResult with error."""
    result = ModuleResult(
        module_name="test",
        dataset_name="ds",
        method_name="method",
        success=False,
        error_message="Something went wrong",
    )
    assert result.success is False
    assert result.error_message == "Something went wrong"
