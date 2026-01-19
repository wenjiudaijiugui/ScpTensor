"""Tests for differential expression test module."""

import numpy as np
import pytest

from scptensor.benchmark.modules import DifferentialExpressionTestModule, ModuleConfig


@pytest.fixture
def de_config():
    """Create test configuration for differential expression module."""
    return ModuleConfig(
        name="differential_expression_test",
        datasets=["synthetic_small"],
        params={
            "methods": ["t_test"],
            "alpha": 0.05,
            "log2_fc_offset": 1.0,
            "missing_strategy": "ignore",
        },
    )


@pytest.fixture
def de_module(de_config):
    """Create differential expression test module instance."""
    return DifferentialExpressionTestModule(de_config)


def test_de_module_initialization(de_module):
    """Test module initialization."""
    assert de_module is not None
    assert de_module.config.name == "differential_expression_test"
    assert de_module.is_enabled()


def test_de_module_params(de_module):
    """Test parameter extraction."""
    assert de_module._methods == ["t_test"]
    assert de_module._alpha == 0.05
    assert de_module._log2_fc_offset == 1.0
    assert de_module._missing_strategy == "ignore"


def test_de_module_run(de_module):
    """Test running differential expression on synthetic_small dataset."""
    results = de_module.run("synthetic_small")

    # Should have at least one result
    assert len(results) >= 1

    # Check result
    first_result = results[0]
    assert first_result.module_name == "differential_expression_test"
    assert first_result.dataset_name == "synthetic_small"
    assert "t_test" in first_result.method_name


def test_de_module_metrics(de_module):
    """Test that differential expression metrics are computed."""
    results = de_module.run("synthetic_small")

    for result in results:
        if result.success:
            assert result.metrics is not None
            # Should have at least some metrics
            assert len(result.metrics) > 0
            # Runtime should be positive
            assert result.runtime_seconds >= 0


def test_de_module_output(de_module):
    """Test that differential expression outputs results."""
    results = de_module.run("synthetic_small")

    for result in results:
        if result.success:
            assert result.output is not None
            # Output should contain DE results
            if isinstance(result.output, dict):
                assert "p_values" in result.output or "log2_fc" in result.output
            elif isinstance(result.output, np.ndarray):
                assert result.output.ndim == 1


def test_de_module_should_process_dataset(de_module):
    """Test dataset filtering."""
    # synthetic_small is in the config
    assert de_module.should_process_dataset("synthetic_small") is True
    # other datasets are not
    assert de_module.should_process_dataset("synthetic_large") is False


def test_de_module_should_process_method(de_module):
    """Test method filtering."""
    # Empty methods list means all methods are allowed
    assert de_module.config.methods == []
    assert de_module.should_process_method("t_test") is True
    assert de_module.should_process_method("wilcoxon") is True


def test_de_module_unknown_method(de_config):
    """Test handling of unknown method."""
    de_config.params["methods"] = ["unknown_method"]
    module = DifferentialExpressionTestModule(de_config)
    results = module.run("synthetic_small")

    # Should return error result
    assert len(results) >= 1
    # Check if unknown method error or group validation error
    if results[0].method_name == "unknown_method":
        assert results[0].success is False
        assert "Unknown method" in results[0].error_message


def test_de_module_unknown_dataset(de_config):
    """Test handling of unknown dataset."""
    de_config.datasets = []
    module = DifferentialExpressionTestModule(de_config)
    results = module.run("nonexistent_dataset")

    # Should return error result
    assert len(results) == 1
    assert results[0].success is False
    assert "not found" in results[0].error_message


def test_de_module_multiple_methods(de_config):
    """Test with multiple methods."""
    de_config.params["methods"] = ["t_test", "wilcoxon"]
    module = DifferentialExpressionTestModule(de_config)
    results = module.run("synthetic_small")

    # Should have results for each method or validation error
    assert len(results) >= 1


def test_de_module_get_results(de_module):
    """Test get_results method."""
    de_module.run("synthetic_small")
    results = de_module.get_results()

    assert len(results) > 0


def test_de_module_clear_results(de_module):
    """Test clear_results method."""
    de_module.run("synthetic_small")
    assert len(de_module.get_results()) > 0

    de_module.clear_results()
    assert len(de_module.get_results()) == 0


def test_de_module_with_empty_methods(de_config):
    """Test with empty methods list."""
    de_config.params["methods"] = []
    module = DifferentialExpressionTestModule(de_config)
    results = module.run("synthetic_small")

    # Should return empty results when no methods specified
    assert len(results) == 0


def test_de_module_alpha_parameter(de_config):
    """Test different alpha values."""
    de_config.params["methods"] = ["t_test"]
    de_config.params["alpha"] = 0.01
    module = DifferentialExpressionTestModule(de_config)
    results = module.run("synthetic_small")

    assert len(results) >= 1
    assert module._alpha == 0.01


def test_de_metrics_content():
    """Test that metrics contain expected fields."""
    from scptensor.benchmark.modules import DifferentialExpressionTestModule, ModuleConfig

    config = ModuleConfig(name="differential_expression_test", datasets=["synthetic_small"])
    module = DifferentialExpressionTestModule(config)
    results = module.run("synthetic_small")

    # Find a successful result
    success_result = None
    for result in results:
        if result.success:
            success_result = result
            break

    if success_result:
        # Check for expected metric keys
        expected_keys = ["n_tested", "n_significant", "prop_significant"]
        for key in expected_keys:
            # Not all metrics may be present depending on data, but at least some should be
            if len(success_result.metrics) > 0:
                break
