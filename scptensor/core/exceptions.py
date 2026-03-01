"""Core exceptions for ScpTensor.

All exceptions inherit from ScpTensorError for consistent error handling.
"""

from collections.abc import Collection
from typing import Any


class ScpTensorError(Exception):
    """Base exception for all ScpTensor errors."""

    def __init__(self, message: str = "") -> None:
        super().__init__(message)


class StructureError(ScpTensorError):
    """Error in data structure validation (e.g., shape mismatch)."""


class ValidationError(ScpTensorError):
    """Error in input validation (e.g., invalid parameters)."""

    def __init__(self, message: str, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field


class DimensionError(ScpTensorError):
    """Error in dimensionality (e.g., incompatible shapes)."""

    def __init__(
        self,
        message: str,
        expected_shape: tuple[int, ...] | None = None,
        actual_shape: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__(message)
        self.expected_shape = expected_shape
        self.actual_shape = actual_shape


class MaskCodeError(ScpTensorError):
    """Error in mask code handling (e.g., invalid code)."""

    def __init__(self, message: str, invalid_code: int | None = None) -> None:
        super().__init__(message)
        self.invalid_code = invalid_code


class ScpValueError(ScpTensorError):
    """Error in value operations (e.g., division by zero, invalid range)."""

    def __init__(
        self,
        message: str,
        parameter: str | None = None,
        value: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.parameter = parameter
        self.value = value


class AssayNotFoundError(ScpTensorError):
    """Error when requested assay does not exist in container."""

    def __init__(
        self,
        assay_name: str,
        hint: str | None = None,
        available_assays: Collection[str] | None = None,
    ) -> None:
        message = f"Assay '{assay_name}' not found in container"

        # Add fuzzy match suggestion if available assays provided
        suggestion = None
        if available_assays is not None:
            from scptensor.core.utils import _find_closest_match

            suggestion = _find_closest_match(assay_name, available_assays)

        if suggestion:
            message += f". Did you mean '{suggestion}'?"
        elif hint:
            message += f". {hint}"
        elif available_assays:
            available_list = ", ".join(f"'{a}'" for a in sorted(available_assays))
            message += f". Available assays: {available_list}"

        super().__init__(message)
        self.assay_name = assay_name
        self.hint = hint
        self.suggestion = suggestion


class LayerNotFoundError(ScpTensorError):
    """Error when requested layer does not exist in assay."""

    def __init__(
        self,
        layer_name: str,
        assay_name: str | None = None,
        hint: str | None = None,
        available_layers: Collection[str] | None = None,
    ) -> None:
        message = f"Layer '{layer_name}' not found"
        if assay_name:
            message += f" in assay '{assay_name}'"

        # Add fuzzy match suggestion if available layers provided
        suggestion = None
        if available_layers is not None:
            from scptensor.core.utils import _find_closest_match

            suggestion = _find_closest_match(layer_name, available_layers)

        if suggestion:
            message += f". Did you mean '{suggestion}'?"
        elif hint:
            message += f". {hint}"
        elif available_layers:
            available_list = ", ".join(f"'{layer}'" for layer in sorted(available_layers))
            message += f". Available layers: {available_list}"

        super().__init__(message)
        self.layer_name = layer_name
        self.assay_name = assay_name
        self.hint = hint
        self.suggestion = suggestion


class MissingDependencyError(ScpTensorError):
    """Error when required optional dependency is not installed."""

    def __init__(self, dependency_name: str) -> None:
        message = (
            f"Required dependency '{dependency_name}' is not installed. "
            f"Please install it using: pip install {dependency_name}"
        )
        super().__init__(message)
        self.dependency_name = dependency_name


class VisualizationError(ScpTensorError):
    """Error in visualization operations (e.g., invalid data, missing layers)."""

    def __init__(
        self,
        message: str,
        parameter: str | None = None,
    ) -> None:
        super().__init__(message)
        self.parameter = parameter


__all__ = [
    "ScpTensorError",
    "StructureError",
    "ValidationError",
    "LayerNotFoundError",
    "AssayNotFoundError",
    "MissingDependencyError",
    "DimensionError",
    "ScpValueError",
    "MaskCodeError",
    "VisualizationError",
]
