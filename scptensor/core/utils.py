import importlib
from functools import wraps
from .exceptions import MissingDependencyError

def requires_dependency(package_name: str, install_hint: str):
    """
    Decorator to ensure a dependency is installed before executing a function.
    
    Args:
        package_name: The name of the package to import.
        install_hint: Command or instruction to install the package.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                importlib.import_module(package_name)
            except ImportError:
                raise MissingDependencyError(
                    f"Method '{func.__name__}' requires '{package_name}'. "
                    f"Please install it via: {install_hint}"
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator
