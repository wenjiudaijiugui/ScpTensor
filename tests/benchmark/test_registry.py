"""Tests for benchmark registry."""

import pytest
from scptensor.benchmark.utils.registry import (
    register_module,
    get_module,
    list_modules,
    has_module,
    clear_modules,
    register_evaluator,
    get_evaluator,
    list_evaluators,
    has_evaluator,
    clear_evaluators,
    register_chart,
    get_chart,
    list_charts,
    has_chart,
    clear_charts,
    clear_all,
    get_registry_info,
)


class TestModuleRegistry:
    """Test module registration."""

    def setup_method(self):
        """Clear registry before each test."""
        clear_modules()

    def test_register_and_get_module(self):
        """Test registering and retrieving a module."""

        @register_module("test_module")
        class TestModule:
            pass

        assert has_module("test_module") is True
        retrieved = get_module("test_module")
        assert retrieved is TestModule

    def test_get_nonexistent_module(self):
        """Test getting a module that doesn't exist."""
        assert get_module("nonexistent") is None

    def test_list_modules(self):
        """Test listing all registered modules."""

        @register_module("module1")
        class Module1:
            pass

        @register_module("module2")
        class Module2:
            pass

        modules = list_modules()
        assert "module1" in modules
        assert "module2" in modules
        assert len(modules) >= 2

    def test_has_module(self):
        """Test has_module function."""

        @register_module("existing")
        class ExistingModule:
            pass

        assert has_module("existing") is True
        assert has_module("nonexistent") is False

    def test_clear_modules(self):
        """Test clearing all modules."""

        @register_module("temp")
        class TempModule:
            pass

        assert has_module("temp") is True
        clear_modules()
        assert has_module("temp") is False


class TestEvaluatorRegistry:
    """Test evaluator registration."""

    def setup_method(self):
        """Clear registry before each test."""
        clear_evaluators()

    def test_register_and_get_evaluator(self):
        """Test registering and retrieving an evaluator."""

        @register_evaluator("test_evaluator")
        class TestEvaluator:
            pass

        assert has_evaluator("test_evaluator") is True
        retrieved = get_evaluator("test_evaluator")
        assert retrieved is TestEvaluator

    def test_list_evaluators(self):
        """Test listing all registered evaluators."""

        @register_evaluator("eval1")
        class Eval1:
            pass

        @register_evaluator("eval2")
        class Eval2:
            pass

        evaluators = list_evaluators()
        assert "eval1" in evaluators
        assert "eval2" in evaluators

    def test_clear_evaluators(self):
        """Test clearing all evaluators."""

        @register_evaluator("temp")
        class TempEval:
            pass

        assert has_evaluator("temp") is True
        clear_evaluators()
        assert has_evaluator("temp") is False


class TestChartRegistry:
    """Test chart registration."""

    def setup_method(self):
        """Clear registry before each test."""
        clear_charts()

    def test_register_and_get_chart(self):
        """Test registering and retrieving a chart."""

        @register_chart("test_chart")
        class TestChart:
            pass

        assert has_chart("test_chart") is True
        retrieved = get_chart("test_chart")
        assert retrieved is TestChart

    def test_list_charts(self):
        """Test listing all registered charts."""

        @register_chart("chart1")
        class Chart1:
            pass

        charts = list_charts()
        assert "chart1" in charts

    def test_clear_charts(self):
        """Test clearing all charts."""

        @register_chart("temp")
        class TempChart:
            pass

        assert has_chart("temp") is True
        clear_charts()
        assert has_chart("temp") is False


class TestRegistryUtilities:
    """Test registry utility functions."""

    def setup_method(self):
        """Clear all registries before each test."""
        clear_all()

    def test_clear_all(self):
        """Test clearing all registries."""

        @register_module("m")
        class M:
            pass

        @register_evaluator("e")
        class E:
            pass

        @register_chart("c")
        class C:
            pass

        assert has_module("m") is True
        assert has_evaluator("e") is True
        assert has_chart("c") is True

        clear_all()

        assert has_module("m") is False
        assert has_evaluator("e") is False
        assert has_chart("c") is False

    def test_get_registry_info(self):
        """Test getting registry information."""

        @register_module("test_mod")
        class TestMod:
            pass

        info = get_registry_info()
        assert "modules" in info
        # info["modules"] contains metadata with count and names
        assert info["modules"]["count"] >= 1
        assert "test_mod" in info["modules"]["names"]
