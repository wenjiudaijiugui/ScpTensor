"""Compatibility facade for ScpTensor core data structures.

The stable public import path remains ``scptensor.core.structures`` while the
concrete implementations live in smaller private modules organized by object
layer:

- ``ScpMatrix`` and related payload-level types
- ``Assay`` for per-feature-space organization
- ``ScpContainer`` for top-level multi-assay coordination
"""

from __future__ import annotations

from ._structure_assay import Assay
from ._structure_container import ScpContainer
from ._structure_matrix import (
    AggregationLink,
    MaskCode,
    MatrixMetadata,
    ProvenanceLog,
    ScpMatrix,
)

for _public_cls in (
    AggregationLink,
    Assay,
    MaskCode,
    MatrixMetadata,
    ProvenanceLog,
    ScpContainer,
    ScpMatrix,
):
    _public_cls.__module__ = __name__

__all__ = [
    "AggregationLink",
    "Assay",
    "MaskCode",
    "MatrixMetadata",
    "ProvenanceLog",
    "ScpContainer",
    "ScpMatrix",
]
