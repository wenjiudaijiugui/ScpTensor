"""Matrix-level core data structures for ScpTensor."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import polars as pl
import scipy.sparse as sp

from scptensor.core.types import LayerMetadataDict, ProvenanceParams


class MaskCode(IntEnum):
    """Data status codes for ScpMatrix mask matrix."""

    VALID = 0
    MBR = 1
    LOD = 2
    FILTERED = 3
    OUTLIER = 4
    IMPUTED = 5
    UNCERTAIN = 6


@dataclass
class ProvenanceLog:
    """Record of operations performed on a container."""

    timestamp: str
    action: str
    params: ProvenanceParams
    software_version: str | None = None
    description: str | None = None


@dataclass
class MatrixMetadata:
    """Enhanced metadata for ScpMatrix containing quality information."""

    confidence_scores: np.ndarray | sp.spmatrix | None = None
    detection_limits: np.ndarray | sp.spmatrix | None = None
    imputation_quality: np.ndarray | sp.spmatrix | None = None
    outlier_scores: np.ndarray | sp.spmatrix | None = None
    creation_info: LayerMetadataDict | None = None


_VALID_MASK_CODES = {code.value for code in MaskCode}


def _validate_mask_matrix(m: np.ndarray | sp.spmatrix, x_shape: tuple[int, int]) -> None:
    """Validate mask matrix shape and values."""
    if m.shape != x_shape:
        raise ValueError(f"Shape mismatch: X {x_shape} != M {m.shape}")

    if isinstance(m, np.ndarray):
        invalid_values = np.setdiff1d(np.unique(m), list(_VALID_MASK_CODES))
        if invalid_values.size > 0:
            raise ValueError(
                f"Invalid mask codes: {invalid_values}. Valid: {sorted(_VALID_MASK_CODES)}",
            )
        return

    if sp.issparse(m):
        invalid_values = np.setdiff1d(np.unique(m.data), list(_VALID_MASK_CODES))
        if invalid_values.size > 0:
            raise ValueError(
                f"Invalid mask codes: {invalid_values}. Valid: {sorted(_VALID_MASK_CODES)}",
            )


@dataclass(slots=True)
class ScpMatrix:
    """Minimal data unit: Physical storage layer for values and status."""

    X: np.ndarray | sp.spmatrix
    M: np.ndarray | sp.spmatrix | None = None
    metadata: MatrixMetadata | None = None

    def __post_init__(self) -> None:
        if not np.issubdtype(self.X.dtype, np.floating):
            object.__setattr__(self, "X", self.X.astype(np.float64))

        if self.M is not None:
            _validate_mask_matrix(self.M, self.X.shape)
            if not np.issubdtype(self.M.dtype, np.integer) or self.M.dtype != np.int8:
                object.__setattr__(self, "M", self.M.astype(np.int8))

    def get_m(self) -> np.ndarray | sp.spmatrix:
        """Return mask matrix, creating zero matrix if M is None."""
        if self.M is not None:
            return self.M

        if sp.issparse(self.X):
            return sp.csr_matrix(self.X.shape, dtype=np.int8)
        return np.zeros(self.X.shape, dtype=np.int8)

    def copy(self) -> ScpMatrix:
        """Create a deep copy of the matrix."""
        new_x = self.X.copy()
        new_m = self.M.copy() if self.M is not None else None
        new_metadata = copy.deepcopy(self.metadata) if self.metadata is not None else None
        return ScpMatrix(X=new_x, M=new_m, metadata=new_metadata)


@dataclass
class AggregationLink:
    """Describes feature aggregation relationship between assays."""

    source_assay: str
    target_assay: str
    linkage: pl.DataFrame

    def __post_init__(self) -> None:
        required_cols = {"source_id", "target_id"}
        if not required_cols.issubset(set(self.linkage.columns)):
            raise ValueError(f"Linkage DataFrame must contain columns: {required_cols}")


__all__ = [
    "AggregationLink",
    "MaskCode",
    "MatrixMetadata",
    "ProvenanceLog",
    "ScpMatrix",
]
