"""Benchmark modules for extensible benchmark operations."""

from .base import BaseModule, ModuleConfig, ModuleResult

# Import clustering test module to trigger registration
from .clustering_test import ClusteringTestModule  # noqa: F401

__all__ = [
    "BaseModule",
    "ModuleConfig",
    "ModuleResult",
    "ClusteringTestModule",
]
