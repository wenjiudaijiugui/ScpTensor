class ScpTensorError(Exception):
    """Base class for exceptions in ScpTensor."""
    pass

class MissingDependencyError(ScpTensorError):
    """Raised when a required dependency is missing."""
    pass
