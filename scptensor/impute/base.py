"""Base utilities for imputation modules.

Provides unified interface for missing value imputation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from scptensor.core.exceptions import ScpValueError

if TYPE_CHECKING:
    from scptensor.core.structures import ScpContainer

# Registry for imputation methods
_IMPUTE_METHODS: dict[str, ImputeMethod] = {}


@dataclass
class ImputeMethod:
    """Registration entry for an imputation method.

    Attributes
    ----------
    name : str
        Method name.
    supports_sparse : bool
        Whether method supports sparse matrices.
    validate : Callable
        Validation function.
    apply : Callable
        Application function.
    """

    name: str
    supports_sparse: bool = True
    validate: Callable[..., Any] | None = None
    apply: Callable[..., ScpContainer] | None = None


def register_impute_method(method: ImputeMethod) -> ImputeMethod:
    """Register an imputation method.

    Parameters
    ----------
    method : ImputeMethod
        Method to register.

    Returns
    -------
    ImputeMethod
        The registered method.
    """
    _IMPUTE_METHODS[method.name] = method
    return method


def get_impute_method(name: str) -> ImputeMethod:
    """Get a registered imputation method by name.

    Parameters
    ----------
    name : str
        Method name.

    Returns
    -------
    ImputeMethod
        The registered method.

    Raises
    ------
    ScpValueError
        If method not found.
    """
    if name not in _IMPUTE_METHODS:
        available = list(_IMPUTE_METHODS.keys())
        raise ScpValueError(
            f"Imputation method '{name}' not found. Available methods: {available}",
            parameter="method",
            value=name,
        )
    return _IMPUTE_METHODS[name]


def list_impute_methods() -> list[str]:
    """List all registered imputation methods.

    Returns
    -------
    list[str]
        List of method names.
    """
    return list(_IMPUTE_METHODS.keys())


def impute(
    container: ScpContainer,
    method: str = "knn",
    **kwargs,
) -> ScpContainer:
    """Unified interface for missing value imputation.

    Parameters
    ----------
    container : ScpContainer
        Input container.
    method : str, default="knn"
        Imputation method name.
    **kwargs
        Additional arguments passed to the method.

    Returns
    -------
    ScpContainer
        Container with imputed data.

    Examples
    --------
    >>> container = impute(container, method='knn', n_neighbors=5)
    >>> container = impute(container, method='bpca', n_components=10)
    """
    entry = get_impute_method(method)
    if entry.apply is None:
        raise ScpValueError(
            f"Method '{method}' has no apply function.",
            parameter="method",
            value=method,
        )
    return entry.apply(container, **kwargs)


__all__ = [
    "ImputeMethod",
    "register_impute_method",
    "get_impute_method",
    "list_impute_methods",
    "impute",
]
