"""I/O module exception hierarchy."""

from scptensor.core.exceptions import ScpTensorError


class IOPasswordError(ScpTensorError):
    """HDF5 file password protection errors."""

    pass


class IOFormatError(ScpTensorError):
    """File format corruption or version incompatibility."""

    pass


class IOWriteError(ScpTensorError):
    """Write failures (disk space, permissions, etc.)."""

    def __init__(self, message: str, path: str | None = None) -> None:
        self.path = path
        if path:
            message = f"{message}: {path}"
        super().__init__(message)
