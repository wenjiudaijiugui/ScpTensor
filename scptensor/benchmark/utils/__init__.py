"""Utility functions and helpers for the benchmark module."""

from .cache import CacheManager
from .registry import (
    # Registry storage
    _AVAILABLE_CHARTS,
    _AVAILABLE_EVALUATORS,
    _AVAILABLE_MODULES,
    # Decorators
    register_chart,
    register_evaluator,
    register_module,
    # Query functions - charts
    clear_charts,
    get_chart,
    has_chart,
    list_charts,
    # Query functions - evaluators
    clear_evaluators,
    get_evaluator,
    has_evaluator,
    list_evaluators,
    # Query functions - modules
    clear_modules,
    get_module,
    has_module,
    list_modules,
    # Utility functions
    clear_all,
    get_registry_info,
)

__all__ = [
    # Cache manager
    "CacheManager",
    # Registry storage
    "_AVAILABLE_MODULES",
    "_AVAILABLE_EVALUATORS",
    "_AVAILABLE_CHARTS",
    # Decorators
    "register_module",
    "register_evaluator",
    "register_chart",
    # Query functions - modules
    "get_module",
    "list_modules",
    "has_module",
    "clear_modules",
    # Query functions - evaluators
    "get_evaluator",
    "list_evaluators",
    "has_evaluator",
    "clear_evaluators",
    # Query functions - charts
    "get_chart",
    "list_charts",
    "has_chart",
    "clear_charts",
    # Utility functions
    "get_registry_info",
    "clear_all",
]
