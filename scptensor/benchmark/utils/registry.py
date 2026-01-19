"""Component registry system for benchmark modules.

This module provides a centralized registry for managing benchmark components
including modules, evaluators, and charts. It uses a decorator-based pattern
for easy registration and retrieval of components.
"""

from collections import defaultdict
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

__all__ = [
    # Registry storage access
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

# =============================================================================
# Type Variables
# =============================================================================

T = TypeVar("T")
P = ParamSpec("P")

# =============================================================================
# Registry Storage
# =============================================================================

_AVAILABLE_MODULES: dict[str, type] = {}
_AVAILABLE_EVALUATORS: dict[str, type] = {}
_AVAILABLE_CHARTS: dict[str, type] = {}

# Module metadata storage
_MODULE_METADATA: dict[str, dict[str, Any]] = defaultdict(dict)


# =============================================================================
# Decorator Functions
# =============================================================================


def register_module(name: str) -> Callable[[type[T]], type[T]]:
    """Decorator to register a benchmark module.

    Parameters
    ----------
    name : str
        Unique identifier for the module. If a module with this name
        already exists, it will be overwritten.

    Returns
    -------
    Callable[[type[T]], type[T]]
        Decorator function that registers the class and returns it unchanged.

    Raises
    ------
    TypeError
        If the decorated object is not a class.

    Examples
    --------
    >>> from scptensor.benchmark.utils.registry import register_module
    >>> from scptensor.benchmark.modules.base import BaseModule
    >>>
    >>> @register_module("normalization_benchmark")
    ... class NormalizationBenchmark(BaseModule):
    ...     def run(self, dataset_name: str):
    ...         return []
    >>>
    >>> from scptensor.benchmark.utils.registry import get_module
    >>> cls = get_module("normalization_benchmark")
    >>> assert cls is NormalizationBenchmark
    """
    def decorator(cls: type[T]) -> type[T]:
        if not isinstance(cls, type):
            raise TypeError(
                f"@register_module can only be used on classes, "
                f"got {type(cls).__name__}"
            )
        _AVAILABLE_MODULES[name] = cls
        _MODULE_METADATA[name]["registered_name"] = name
        return cls

    return decorator


def register_evaluator(name: str) -> Callable[[type[T]], type[T]]:
    """Decorator to register a benchmark evaluator.

    Evaluators compute metrics on benchmark results, such as accuracy,
    precision, recall, or custom domain-specific scores.

    Parameters
    ----------
    name : str
        Unique identifier for the evaluator. If an evaluator with this
        name already exists, it will be overwritten.

    Returns
    -------
    Callable[[type[T]], type[T]]
        Decorator function that registers the class and returns it unchanged.

    Raises
    ------
    TypeError
        If the decorated object is not a class.

    Examples
    --------
    >>> from scptensor.benchmark.utils.registry import register_evaluator
    >>>
    >>> @register_evaluator("accuracy")
    ... class AccuracyEvaluator:
    ...     def compute(self, y_true, y_pred):
    ...         return (y_true == y_pred).mean()
    >>>
    >>> from scptensor.benchmark.utils.registry import get_evaluator
    >>> cls = get_evaluator("accuracy")
    >>> assert cls is AccuracyEvaluator
    """
    def decorator(cls: type[T]) -> type[T]:
        if not isinstance(cls, type):
            raise TypeError(
                f"@register_evaluator can only be used on classes, "
                f"got {type(cls).__name__}"
            )
        _AVAILABLE_EVALUATORS[name] = cls
        return cls

    return decorator


def register_chart(name: str) -> Callable[[type[T]], type[T]]:
    """Decorator to register a benchmark chart type.

    Charts define visualization methods for benchmark results, such as
    bar charts, line plots, heatmaps, or custom visualizations.

    Parameters
    ----------
    name : str
        Unique identifier for the chart type. If a chart with this
        name already exists, it will be overwritten.

    Returns
    -------
    Callable[[type[T]], type[T]]
        Decorator function that registers the class and returns it unchanged.

    Raises
    ------
    TypeError
        If the decorated object is not a class.

    Examples
    --------
    >>> from scptensor.benchmark.utils.registry import register_chart
    >>>
    >>> @register_chart("performance_bar")
    ... class PerformanceBarChart:
    ...     def plot(self, data):
    ...         import matplotlib.pyplot as plt
    ...         fig, ax = plt.subplots()
    ...         ax.bar(data.keys(), data.values())
    ...         return fig
    >>>
    >>> from scptensor.benchmark.utils.registry import get_chart
    >>> cls = get_chart("performance_bar")
    >>> assert cls is PerformanceBarChart
    """
    def decorator(cls: type[T]) -> type[T]:
        if not isinstance(cls, type):
            raise TypeError(
                f"@register_chart can only be used on classes, "
                f"got {type(cls).__name__}"
            )
        _AVAILABLE_CHARTS[name] = cls
        return cls

    return decorator


# =============================================================================
# Query Functions - Modules
# =============================================================================


def get_module(name: str) -> type | None:
    """Get a registered module class by name.

    Parameters
    ----------
    name : str
        Name of the module to retrieve.

    Returns
    -------
    type | None
        The registered module class, or None if not found.

    Examples
    --------
    >>> from scptensor.benchmark.utils.registry import get_module
    >>> cls = get_module("normalization_benchmark")
    >>> if cls is not None:
    ...     print(f"Found: {cls.__name__}")
    """
    return _AVAILABLE_MODULES.get(name)


def list_modules() -> list[str]:
    """List all registered module names.

    Returns
    -------
    list[str]
        Sorted list of registered module names.

    Examples
    --------
    >>> from scptensor.benchmark.utils.registry import list_modules
    >>> names = list_modules()
    >>> print(f"Available modules: {names}")
    """
    return sorted(_AVAILABLE_MODULES.keys())


def has_module(name: str) -> bool:
    """Check if a module is registered.

    Parameters
    ----------
    name : str
        Name of the module to check.

    Returns
    -------
    bool
        True if the module is registered, False otherwise.

    Examples
    --------
    >>> from scptensor.benchmark.utils.registry import has_module
    >>> if has_module("normalization_benchmark"):
    ...     print("Module is available")
    """
    return name in _AVAILABLE_MODULES


def clear_modules() -> None:
    """Clear all registered modules.

    This removes all modules from the registry. Use with caution,
    as this cannot be undone.

    Examples
    --------
    >>> from scptensor.benchmark.utils.registry import clear_modules
    >>> clear_modules()
    >>> assert len(list_modules()) == 0
    """
    _AVAILABLE_MODULES.clear()
    _MODULE_METADATA.clear()


# =============================================================================
# Query Functions - Evaluators
# =============================================================================


def get_evaluator(name: str) -> type | None:
    """Get a registered evaluator class by name.

    Parameters
    ----------
    name : str
        Name of the evaluator to retrieve.

    Returns
    -------
    type | None
        The registered evaluator class, or None if not found.

    Examples
    --------
    >>> from scptensor.benchmark.utils.registry import get_evaluator
    >>> cls = get_evaluator("accuracy")
    >>> if cls is not None:
    ...     print(f"Found: {cls.__name__}")
    """
    return _AVAILABLE_EVALUATORS.get(name)


def list_evaluators() -> list[str]:
    """List all registered evaluator names.

    Returns
    -------
    list[str]
        Sorted list of registered evaluator names.

    Examples
    --------
    >>> from scptensor.benchmark.utils.registry import list_evaluators
    >>> names = list_evaluators()
    >>> print(f"Available evaluators: {names}")
    """
    return sorted(_AVAILABLE_EVALUATORS.keys())


def has_evaluator(name: str) -> bool:
    """Check if an evaluator is registered.

    Parameters
    ----------
    name : str
        Name of the evaluator to check.

    Returns
    -------
    bool
        True if the evaluator is registered, False otherwise.

    Examples
    --------
    >>> from scptensor.benchmark.utils.registry import has_evaluator
    >>> if has_evaluator("accuracy"):
    ...     print("Evaluator is available")
    """
    return name in _AVAILABLE_EVALUATORS


def clear_evaluators() -> None:
    """Clear all registered evaluators.

    This removes all evaluators from the registry. Use with caution,
    as this cannot be undone.

    Examples
    --------
    >>> from scptensor.benchmark.utils.registry import clear_evaluators
    >>> clear_evaluators()
    >>> assert len(list_evaluators()) == 0
    """
    _AVAILABLE_EVALUATORS.clear()


# =============================================================================
# Query Functions - Charts
# =============================================================================


def get_chart(name: str) -> type | None:
    """Get a registered chart class by name.

    Parameters
    ----------
    name : str
        Name of the chart to retrieve.

    Returns
    -------
    type | None
        The registered chart class, or None if not found.

    Examples
    --------
    >>> from scptensor.benchmark.utils.registry import get_chart
    >>> cls = get_chart("performance_bar")
    >>> if cls is not None:
    ...     print(f"Found: {cls.__name__}")
    """
    return _AVAILABLE_CHARTS.get(name)


def list_charts() -> list[str]:
    """List all registered chart names.

    Returns
    -------
    list[str]
        Sorted list of registered chart names.

    Examples
    --------
    >>> from scptensor.benchmark.utils.registry import list_charts
    >>> names = list_charts()
    >>> print(f"Available charts: {names}")
    """
    return sorted(_AVAILABLE_CHARTS.keys())


def has_chart(name: str) -> bool:
    """Check if a chart is registered.

    Parameters
    ----------
    name : str
        Name of the chart to check.

    Returns
    -------
    bool
        True if the chart is registered, False otherwise.

    Examples
    --------
    >>> from scptensor.benchmark.utils.registry import has_chart
    >>> if has_chart("performance_bar"):
    ...     print("Chart is available")
    """
    return name in _AVAILABLE_CHARTS


def clear_charts() -> None:
    """Clear all registered charts.

    This removes all charts from the registry. Use with caution,
    as this cannot be undone.

    Examples
    --------
    >>> from scptensor.benchmark.utils.registry import clear_charts
    >>> clear_charts()
    >>> assert len(list_charts()) == 0
    """
    _AVAILABLE_CHARTS.clear()


# =============================================================================
# Utility Functions
# =============================================================================


def get_registry_info() -> dict[str, dict[str, Any]]:
    """Get information about all registered components.

    Returns a dictionary containing counts and names of all registered
    modules, evaluators, and charts.

    Returns
    -------
    dict[str, dict[str, Any]]
        Dictionary with keys 'modules', 'evaluators', 'charts', each
        containing 'count' and 'names' entries.

    Examples
    --------
    >>> from scptensor.benchmark.utils.registry import get_registry_info
    >>> info = get_registry_info()
    >>> print(f"Modules: {info['modules']['count']}")
    >>> print(f"Evaluators: {info['evaluators']['count']}")
    >>> print(f"Charts: {info['charts']['count']}")
    """
    return {
        "modules": {
            "count": len(_AVAILABLE_MODULES),
            "names": list_modules(),
        },
        "evaluators": {
            "count": len(_AVAILABLE_EVALUATORS),
            "names": list_evaluators(),
        },
        "charts": {
            "count": len(_AVAILABLE_CHARTS),
            "names": list_charts(),
        },
    }


def clear_all() -> None:
    """Clear all registered components.

    This removes all modules, evaluators, and charts from the registry.
    Use with caution, as this cannot be undone.

    Examples
    --------
    >>> from scptensor.benchmark.utils.registry import clear_all
    >>> clear_all()
    >>> info = get_registry_info()
    >>> assert info['modules']['count'] == 0
    >>> assert info['evaluators']['count'] == 0
    >>> assert info['charts']['count'] == 0
    """
    clear_modules()
    clear_evaluators()
    clear_charts()
