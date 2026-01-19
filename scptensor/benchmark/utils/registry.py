"""Component registry for benchmark modules."""

from collections import defaultdict
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

T = TypeVar("T")
P = ParamSpec("P")

_MODULES: dict[str, type] = {}
_EVALUATORS: dict[str, type] = {}
_CHARTS: dict[str, type] = {}
_METADATA: dict[str, dict[str, Any]] = defaultdict(dict)

# Backward compatibility aliases
_AVAILABLE_MODULES = _MODULES
_AVAILABLE_EVALUATORS = _EVALUATORS
_AVAILABLE_CHARTS = _CHARTS

__all__ = [
    "_AVAILABLE_MODULES", "_AVAILABLE_EVALUATORS", "_AVAILABLE_CHARTS",
    "register_module", "register_evaluator", "register_chart",
    "get_module", "list_modules", "has_module", "clear_modules",
    "get_evaluator", "list_evaluators", "has_evaluator", "clear_evaluators",
    "get_chart", "list_charts", "has_chart", "clear_charts",
    "get_registry_info", "clear_all",
]


def _make_register(registry: dict, metadata: dict | None = None) -> Callable[[str], Callable[[type[T]], type[T]]]:
    def decorator(name: str) -> Callable[[type[T]], type[T]]:
        def wrapper(cls: type[T]) -> type[T]:
            if not isinstance(cls, type):
                raise TypeError(f"Decorator can only be used on classes, got {type(cls).__name__}")
            registry[name] = cls
            if metadata is not None:
                metadata[name]["registered_name"] = name
            return cls
        return wrapper
    return decorator


register_module = _make_register(_MODULES, _METADATA)
register_evaluator = _make_register(_EVALUATORS)
register_chart = _make_register(_CHARTS)


def _make_get(registry: dict) -> Callable[[str], type | None]:
    return lambda name: registry.get(name)


def _make_list(registry: dict) -> Callable[[], list[str]]:
    return lambda: sorted(registry.keys())


def _make_has(registry: dict) -> Callable[[str], bool]:
    return lambda name: name in registry


def _make_clear(registry: dict, metadata: dict | None = None) -> Callable[[], None]:
    def fn() -> None:
        registry.clear()
        if metadata is not None:
            metadata.clear()
    return fn


get_module = _make_get(_MODULES)
list_modules = _make_list(_MODULES)
has_module = _make_has(_MODULES)
clear_modules = _make_clear(_MODULES, _METADATA)

get_evaluator = _make_get(_EVALUATORS)
list_evaluators = _make_list(_EVALUATORS)
has_evaluator = _make_has(_EVALUATORS)
clear_evaluators = _make_clear(_EVALUATORS)

get_chart = _make_get(_CHARTS)
list_charts = _make_list(_CHARTS)
has_chart = _make_has(_CHARTS)
clear_charts = _make_clear(_CHARTS)


def get_registry_info() -> dict[str, dict[str, Any]]:
    return {
        "modules": {"count": len(_MODULES), "names": list_modules()},
        "evaluators": {"count": len(_EVALUATORS), "names": list_evaluators()},
        "charts": {"count": len(_CHARTS), "names": list_charts()},
    }


def clear_all() -> None:
    clear_modules()
    clear_evaluators()
    clear_charts()
