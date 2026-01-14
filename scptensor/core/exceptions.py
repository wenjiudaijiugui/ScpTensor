"""Core exceptions for ScpTensor.

All exceptions inherit from ScpTensorError for consistent error handling.
"""

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

    def __init__(self, assay_name: str) -> None:
        super().__init__(f"Assay '{assay_name}' not found in container")
        self.assay_name = assay_name


class LayerNotFoundError(ScpTensorError):
    """Error when requested layer does not exist in assay."""

    def __init__(self, layer_name: str, assay_name: str | None = None) -> None:
        message = f"Layer '{layer_name}' not found"
        if assay_name:
            message += f" in assay '{assay_name}'"
        super().__init__(message)
        self.layer_name = layer_name
        self.assay_name = assay_name


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


if __name__ == "__main__":
    print("Testing ScpTensor exception hierarchy...")

    # Test base exception
    try:
        raise ScpTensorError("Base error test")
    except ScpTensorError as e:
        assert str(e) == "Base error test"
        print("ScpTensorError: OK")

    # Test: ValidationError with field
    try:
        raise ValidationError("Invalid input", field="sample_id")
    except ValidationError as e:
        assert e.field == "sample_id"
        print("ValidationError: OK")

    # Test: AssayNotFoundError
    try:
        raise AssayNotFoundError("metabolites")
    except AssayNotFoundError as e:
        assert e.assay_name == "metabolites"
        assert "not found" in str(e)
        print("AssayNotFoundError: OK")

    # Test: LayerNotFoundError
    try:
        raise LayerNotFoundError("normalized", assay_name="proteins")
    except LayerNotFoundError as e:
        assert e.layer_name == "normalized"
        assert e.assay_name == "proteins"
        print("LayerNotFoundError: OK")

    # Test: MissingDependencyError
    try:
        raise MissingDependencyError("scanpy")
    except MissingDependencyError as e:
        assert e.dependency_name == "scanpy"
        assert "pip install" in str(e)
        print("MissingDependencyError: OK")

    # Test: DimensionError
    try:
        raise DimensionError("Shape mismatch", expected_shape=(100, 50), actual_shape=(100, 45))
    except DimensionError as e:
        assert e.expected_shape == (100, 50)
        assert e.actual_shape == (100, 45)
        print("DimensionError: OK")

    # Test: ScpValueError
    try:
        raise ScpValueError("Negative value", parameter="scale", value=-1.0)
    except ScpValueError as e:
        assert e.parameter == "scale"
        assert e.value == -1.0
        print("ScpValueError: OK")

    # Test: MaskCodeError
    try:
        raise MaskCodeError("Invalid code", invalid_code=9)
    except MaskCodeError as e:
        assert e.invalid_code == 9
        print("MaskCodeError: OK")

    # Test hierarchy
    assert issubclass(ValidationError, ScpTensorError)
    assert issubclass(DimensionError, ScpTensorError)
    assert issubclass(AssayNotFoundError, ScpTensorError)
    print("Exception hierarchy: OK")

    print("\nAll exception tests passed!")
