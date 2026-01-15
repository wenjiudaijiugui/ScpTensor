"""Tests for ScpTensor core structures.

This package contains tests for the fundamental data structures:
- ScpContainer: Top-level container for multi-assay data
- Assay: Feature-space with multiple layers
- ScpMatrix: Physical storage with mask-based provenance
- ProvenanceLog: Operation audit trail
- AggregationLink: Inter-assay relationship mapping
- MaskCode: Provenance tracking enum
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # This is a test package - no public exports needed
    pass
