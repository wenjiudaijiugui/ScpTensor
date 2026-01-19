"""Benchmark modules for extensible benchmark operations."""

from .base import BaseModule, ModuleConfig, ModuleResult

# Import modules to trigger registration
from .batch_correction_test import BatchCorrectionTestModule  # noqa: F401
from .clustering_test import ClusteringTestModule  # noqa: F401
from .differential_expression_test import (  # noqa: F401
    DifferentialExpressionTestModule,
)

__all__ = [
    "BaseModule",
    "ModuleConfig",
    "ModuleResult",
    "BatchCorrectionTestModule",
    "ClusteringTestModule",
    "DifferentialExpressionTestModule",
]
