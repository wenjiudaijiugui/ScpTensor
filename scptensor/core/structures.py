"""Core data structures for ScpTensor.

This module defines the hierarchical data structure used throughout ScpTensor:
- ScpContainer: Top-level container managing global sample metadata and assays
- Assay: Feature-space object managing data layers and feature metadata
- ScpMatrix: Physical storage layer for values and mask codes
- MaskCode: Enumeration for data status codes
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import scipy.sparse as sp

if TYPE_CHECKING:
    from collections.abc import Sequence

from scptensor.core.exceptions import (
    AssayNotFoundError,
    DimensionError,
    ValidationError,
)


class MaskCode(IntEnum):
    """Data status codes for ScpMatrix mask matrix.

    Values:
        VALID (0): Valid, detected values
        MBR (1): Match Between Runs missing
        LOD (2): Below Limit of Detection
        FILTERED (3): Filtered out by quality control
        OUTLIER (4): Statistical outlier
        IMPUTED (5): Imputed/filled value
        UNCERTAIN (6): Uncertain data quality
    """

    VALID = 0
    MBR = 1
    LOD = 2
    FILTERED = 3
    OUTLIER = 4
    IMPUTED = 5
    UNCERTAIN = 6


@dataclass
class ProvenanceLog:
    """Record of operations performed on a container.

    Attributes:
        timestamp: ISO format timestamp of operation
        action: Name of the operation performed
        params: Parameters passed to the operation
        software_version: Version of software used
        description: Human-readable description
    """

    timestamp: str
    action: str
    params: dict[str, Any]
    software_version: str | None = None
    description: str | None = None


@dataclass
class MatrixMetadata:
    """Enhanced metadata for ScpMatrix containing quality information.

    Attributes:
        confidence_scores: Data confidence scores [0, 1]
        detection_limits: Detection limit values
        imputation_quality: Imputation quality scores [0, 1]
        outlier_scores: Outlier detection scores
        creation_info: Creation tracking information
    """

    confidence_scores: np.ndarray | sp.spmatrix | None = None
    detection_limits: np.ndarray | sp.spmatrix | None = None
    imputation_quality: np.ndarray | sp.spmatrix | None = None
    outlier_scores: np.ndarray | sp.spmatrix | None = None
    creation_info: dict[str, Any] | None = None


# Valid mask codes for validation
_VALID_MASK_CODES = {code.value for code in MaskCode}


def _validate_mask_matrix(M: np.ndarray | sp.spmatrix, X_shape: tuple[int, int]) -> None:
    """Validate mask matrix shape and values.

    Args:
        M: Mask matrix to validate
        X_shape: Expected shape from data matrix

    Raises:
        ValueError: If shape mismatch or invalid mask codes found
    """
    if M.shape != X_shape:
        raise ValueError(f"Shape mismatch: X {X_shape} != M {M.shape}")

    if isinstance(M, np.ndarray):
        invalid_values = np.setdiff1d(np.unique(M), list(_VALID_MASK_CODES))
        if invalid_values.size > 0:
            raise ValueError(
                f"Invalid mask codes: {invalid_values}. Valid: {sorted(_VALID_MASK_CODES)}"
            )


@dataclass
class ScpMatrix:
    """Minimal data unit: Physical storage layer for values and status.

    Attributes:
        X: Quantitative value matrix (f64/f32). Supports sparse (CSR/CSC).
           Shape: (n_samples, n_features)
        M: Status mask matrix (int8). 0=Valid, 1=MBR, 2=LOD, 3=Filtered.
           Shape: (n_samples, n_features). If None, all data assumed valid.
        metadata: Optional quality scores and tracking information

    The mask matrix tracks the provenance of each value through the analysis
    pipeline, enabling reproducible and auditable data processing.
    """

    X: np.ndarray | sp.spmatrix
    M: np.ndarray | sp.spmatrix | None = None
    metadata: MatrixMetadata | None = None

    def __post_init__(self) -> None:
        # Ensure X is float type
        if not np.issubdtype(self.X.dtype, np.floating):
            object.__setattr__(self, "X", self.X.astype(np.float64))

        # Validate M if provided
        if self.M is not None:
            _validate_mask_matrix(self.M, self.X.shape)

            # Ensure M is int8
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
        new_X = self.X.copy()
        new_M = self.M.copy() if self.M is not None else None
        return ScpMatrix(X=new_X, M=new_M, metadata=self.metadata)


@dataclass
class AggregationLink:
    """Describes feature aggregation relationship between assays (e.g., Peptide -> Protein).

    Attributes:
        source_assay: Name of the source assay
        target_assay: Name of the target assay
        linkage: DataFrame with 'source_id' and 'target_id' columns mapping features
    """

    source_assay: str
    target_assay: str
    linkage: pl.DataFrame

    def __post_init__(self) -> None:
        required_cols = {"source_id", "target_id"}
        if not required_cols.issubset(set(self.linkage.columns)):
            raise ValueError(f"Linkage DataFrame must contain columns: {required_cols}")


class Assay:
    """Feature-space object managing data under a specific feature type.

    The Assay coordinates feature metadata (var) with multiple data layers,
    each containing the same features but potentially transformed values.

    Attributes:
        var: Feature metadata DataFrame. Must contain unique feature IDs.
        layers: Dictionary of named data layers (e.g., 'raw', 'log', 'imputed')
        feature_id_col: Column name in var serving as unique feature identifier

    Example:
        >>> var = pl.DataFrame({"_index": ["P1", "P2"], "mean": [1.0, 2.0]})
        >>> X = np.array([[1, 2], [3, 4]])
        >>> assay = Assay(var, layers={"X": ScpMatrix(X=X)})
    """

    def __init__(
        self,
        var: pl.DataFrame,
        layers: dict[str, ScpMatrix] | None = None,
        feature_id_col: str = "_index",
    ):
        self.feature_id_col = feature_id_col

        if feature_id_col not in var.columns:
            raise ValueError(f"Feature ID column '{feature_id_col}' not found in var.")

        if var[feature_id_col].n_unique() != var.height:
            raise ValueError(f"Feature ID column '{feature_id_col}' is not unique.")

        self.var: pl.DataFrame = var
        self.layers: dict[str, ScpMatrix] = layers if layers is not None else {}

        self._validate()

    def _validate(self) -> None:
        """Validate all layers have matching feature dimensions."""
        for name, matrix in self.layers.items():
            if matrix.X.shape[1] != self.n_features:
                raise ValueError(
                    f"Layer '{name}': Features {matrix.X.shape[1]} != Assay {self.n_features}"
                )

    @property
    def n_features(self) -> int:
        """Number of features in this assay."""
        return self.var.height

    @property
    def feature_ids(self) -> pl.Series:
        """Unique feature identifiers."""
        return self.var[self.feature_id_col]

    @property
    def X(self) -> np.ndarray | sp.spmatrix | None:
        """Shortcut to access the 'X' layer matrix if it exists."""
        layer = self.layers.get("X")
        return layer.X if layer else None

    def add_layer(self, name: str, matrix: ScpMatrix) -> None:
        """Add a new data layer to this assay.

        Args:
            name: Layer name (e.g., 'raw', 'log', 'imputed')
            matrix: Matrix object with matching feature dimension

        Raises:
            ValueError: If feature dimensions don't match
        """
        if matrix.X.shape[1] != self.n_features:
            raise ValueError(f"Layer has {matrix.X.shape[1]} features, Assay has {self.n_features}")
        self.layers[name] = matrix

    def __repr__(self) -> str:
        return f"<Assay n_features={self.n_features}, layers={list(self.layers.keys())}>"

    def subset(self, feature_indices: list[int] | np.ndarray, copy_data: bool = True) -> Assay:
        """Return a new Assay with a subset of features.

        Args:
            feature_indices: Indices of features to keep
            copy_data: Whether to copy underlying data (default: True)

        Returns:
            New Assay with subsetted features and layers
        """
        new_var = self.var[feature_indices, :]
        new_layers: dict[str, ScpMatrix] = {}

        for name, matrix in self.layers.items():
            new_X = matrix.X[:, feature_indices]
            new_M = matrix.M[:, feature_indices] if matrix.M is not None else None

            if copy_data:
                if isinstance(new_X, np.ndarray) or sp.issparse(new_X):
                    new_X = new_X.copy()
                if new_M is not None and (isinstance(new_M, np.ndarray) or sp.issparse(new_M)):
                    new_M = new_M.copy()

            new_layers[name] = ScpMatrix(X=new_X, M=new_M)

        return Assay(var=new_var, layers=new_layers, feature_id_col=self.feature_id_col)


class ScpContainer:
    """Top-level container managing global sample metadata and assay registry.

    The ScpContainer provides a unified interface for multi-omics single-cell
    proteomics data, coordinating sample-level metadata across multiple feature
    spaces (assays).

    Attributes:
        obs: Global sample metadata DataFrame. Must contain unique sample IDs.
        assays: Dictionary of named assays (e.g., 'proteins', 'peptides')
        links: List of aggregation relationships between assays
        history: Provenance log of operations performed
        sample_id_col: Column name in obs serving as unique sample identifier

    Example:
        >>> obs = pl.DataFrame({"_index": ["S1", "S2"], "batch": ["A", "B"]})
        >>> container = ScpContainer(obs)
        >>> container.log_operation("test", {"n": 2}, "Test operation")
    """

    def __init__(
        self,
        obs: pl.DataFrame,
        assays: dict[str, Assay] | None = None,
        links: list[AggregationLink] | None = None,
        history: list[ProvenanceLog] | None = None,
        sample_id_col: str = "_index",
    ):
        self.sample_id_col = sample_id_col

        if sample_id_col not in obs.columns:
            raise ValueError(f"Sample ID column '{sample_id_col}' not found in obs.")

        if obs[sample_id_col].n_unique() != obs.height:
            raise ValueError(f"Sample ID column '{sample_id_col}' is not unique.")

        self.obs: pl.DataFrame = obs
        self.assays: dict[str, Assay] = assays if assays is not None else {}
        self.links: list[AggregationLink] = links if links is not None else []
        self.history: list[ProvenanceLog] = history if history is not None else []

        self._validate()
        if self.links:
            self.validate_links()

    @property
    def n_samples(self) -> int:
        """Number of samples in the container."""
        return self.obs.height

    @property
    def n_features(self) -> int:
        """Number of features in the first assay.

        For multi-assay containers, access features via:
        container.assays[assay_name].n_features
        """
        if not self.assays:
            return 0
        return next(iter(self.assays.values())).n_features

    @property
    def sample_ids(self) -> pl.Series:
        """Unique sample identifiers."""
        return self.obs[self.sample_id_col]

    @property
    def shape(self) -> tuple[int, int]:
        """Shape as (n_samples, n_features).

        Compatible with Scanpy AnnData.shape interface.
        For multi-assay containers, returns first assay shape.
        """
        return (self.n_samples, self.n_features)

    def _validate(self) -> None:
        """Validate all assays have matching sample dimensions."""
        for assay_name, assay in self.assays.items():
            for layer_name, matrix in assay.layers.items():
                if matrix.X.shape[0] != self.n_samples:
                    raise ValueError(
                        f"Assay '{assay_name}', Layer '{layer_name}': "
                        f"Samples {matrix.X.shape[0]} != {self.n_samples}"
                    )

    def validate_links(self) -> None:
        """Validate that all links connect to existing assays and features."""
        for link in self.links:
            if link.source_assay not in self.assays:
                raise ValueError(f"Link source assay '{link.source_assay}' not found.")
            if link.target_assay not in self.assays:
                raise ValueError(f"Link target assay '{link.target_assay}' not found.")

    def add_assay(self, name: str, assay: Assay) -> ScpContainer:
        """Register a new assay to the container.

        Args:
            name: Assay name (e.g., 'proteins', 'peptides')
            assay: Assay object with matching sample dimension

        Returns:
            Self for method chaining

        Raises:
            ValueError: If assay already exists or dimensions don't match
        """
        if name in self.assays:
            raise ValueError(f"Assay '{name}' already exists.")

        for layer_name, matrix in assay.layers.items():
            if matrix.X.shape[0] != self.n_samples:
                raise ValueError(
                    f"New Assay '{name}', Layer '{layer_name}': "
                    f"Samples {matrix.X.shape[0]} != {self.n_samples}"
                )
        self.assays[name] = assay
        return self

    def log_operation(
        self,
        action: str,
        params: dict[str, Any],
        description: str | None = None,
        software_version: str | None = None,
    ) -> None:
        """Record an operation to the provenance history.

        Args:
            action: Name of the operation
            params: Parameters passed to the operation
            description: Human-readable description
            software_version: Version of software used
        """
        log = ProvenanceLog(
            timestamp=datetime.now().isoformat(),
            action=action,
            params=params,
            software_version=software_version,
            description=description,
        )
        self.history.append(log)

    def __repr__(self) -> str:
        assays_desc = ", ".join([f"{k}({v.n_features})" for k, v in self.assays.items()])
        return f"<ScpContainer n_samples={self.n_samples}, assays=[{assays_desc}]>"

    def copy(self, deep: bool = True) -> ScpContainer:
        """Copy the container.

        Args:
            deep: If True, create deep copy; otherwise shallow copy

        Returns:
            Copied container
        """
        return self.deepcopy() if deep else self.shallow_copy()

    def shallow_copy(self) -> ScpContainer:
        """Create a shallow copy of the container."""
        return ScpContainer(
            obs=self.obs,
            assays=self.assays.copy(),
            links=list(self.links),
            history=list(self.history),
            sample_id_col=self.sample_id_col,
        )

    def deepcopy(self) -> ScpContainer:
        """Create a deep copy of the container."""
        new_obs = self.obs.clone()

        new_assays: dict[str, Assay] = {}
        for name, assay in self.assays.items():
            new_assays[name] = assay.subset(np.arange(assay.n_features), copy_data=True)

        new_links = [
            AggregationLink(
                source_assay=link.source_assay,
                target_assay=link.target_assay,
                linkage=link.linkage.clone(),
            )
            for link in self.links
        ]

        new_history = [copy.deepcopy(log) for log in self.history]

        return ScpContainer(
            obs=new_obs,
            assays=new_assays,
            links=new_links,
            history=new_history,
            sample_id_col=self.sample_id_col,
        )

    def filter_samples(
        self,
        sample_ids: (
            Sequence[str] | np.ndarray | pl.Expr | pl.Series | Sequence[int] | None
        ) = None,
        *,
        sample_indices: Sequence[int] | np.ndarray | None = None,
        boolean_mask: np.ndarray | pl.Series | None = None,
        polars_expression: pl.Expr | None = None,
        copy: bool = True,
    ) -> ScpContainer:
        """Filter samples from the container.

        Provides multiple ways to filter samples:
        - By sample IDs (list of IDs to keep)
        - By sample indices (positional indices)
        - By boolean mask (True = keep)
        - By Polars expression on the obs DataFrame

        Args:
            sample_ids: Sample identifiers to keep
            sample_indices: Positional indices of samples to keep
            boolean_mask: Boolean mask where True indicates samples to keep
            polars_expression: Polars expression evaluated on obs
            copy: Whether to copy underlying data (default: True)

        Returns:
            New container with filtered samples

        Raises:
            ValidationError: If filtering criteria are invalid
            DimensionError: If mask has incorrect dimensions

        Examples:
            >>> container.filter_samples(["sample1", "sample2"])
            >>> container.filter_samples(sample_indices=[0, 1, 2])
            >>> container.filter_samples(pl.col("n_detected") > 100)
        """
        indices: np.ndarray = self._resolve_sample_indices(
            sample_ids, sample_indices, boolean_mask, polars_expression
        )

        new_obs = self.obs[indices, :]
        new_assays = self._filter_assays_samples(indices, copy)
        new_history = self._updated_history(
            "filter_samples",
            {
                "n_samples_kept": len(indices),
                "n_samples_original": self.n_samples,
                "kept_sample_ids": self.sample_ids[indices].to_list(),
            },
            f"Filtered to {len(indices)}/{self.n_samples} samples",
        )

        return ScpContainer(
            obs=new_obs,
            assays=new_assays,
            links=list(self.links),
            history=new_history,
            sample_id_col=self.sample_id_col,
        )

    def filter_features(
        self,
        assay_name: str,
        feature_ids: (
            Sequence[str] | np.ndarray | pl.Expr | pl.Series | Sequence[int] | None
        ) = None,
        *,
        feature_indices: Sequence[int] | np.ndarray | None = None,
        boolean_mask: np.ndarray | pl.Series | None = None,
        polars_expression: pl.Expr | None = None,
        copy: bool = True,
    ) -> ScpContainer:
        """Filter features for a specific assay.

        Args:
            assay_name: Name of the assay to filter
            feature_ids: Feature identifiers to keep
            feature_indices: Positional indices of features to keep
            boolean_mask: Boolean mask where True indicates features to keep
            polars_expression: Polars expression evaluated on var
            copy: Whether to copy underlying data (default: True)

        Returns:
            New container with filtered features in specified assay

        Raises:
            AssayNotFoundError: If assay doesn't exist
            ValidationError: If filtering criteria are invalid
            DimensionError: If mask has incorrect dimensions

        Examples:
            >>> container.filter_features("proteins", ["P123", "P456"])
            >>> container.filter_features("proteins", feature_indices=[0, 1, 2])
            >>> container.filter_features("proteins", pl.col("n_detected") > 10)
        """
        if assay_name not in self.assays:
            raise AssayNotFoundError(assay_name)

        assay = self.assays[assay_name]
        indices = self._resolve_feature_indices(
            assay, feature_ids, feature_indices, boolean_mask, polars_expression
        )

        new_assays = self._filter_assay_features(assay_name, assay, indices, copy)
        new_history = self._updated_history(
            "filter_features",
            {
                "assay_name": assay_name,
                "n_features_kept": len(indices),
                "n_features_original": assay.n_features,
                "kept_feature_ids": assay.feature_ids[indices].to_list(),
            },
            f"Filtered assay '{assay_name}' to {len(indices)}/{assay.n_features} features",
        )

        return ScpContainer(
            obs=self.obs,
            assays=new_assays,
            links=list(self.links),
            history=new_history,
            sample_id_col=self.sample_id_col,
        )

    # ==========================================================================
    # Internal helper methods
    # ==========================================================================

    def _resolve_sample_indices(
        self,
        sample_ids,
        sample_indices,
        boolean_mask,
        polars_expression,
    ) -> np.ndarray:
        """Resolve various input formats to sample indices array."""
        indices: np.ndarray

        # Polars expression
        expr: pl.Expr | None = None
        if isinstance(sample_ids, pl.Expr):
            expr = sample_ids
        elif polars_expression is not None:
            expr = polars_expression

        if expr is not None:
            mask_result = self.obs.select(expr).to_series()
            if mask_result.dtype != pl.Boolean:
                raise ValueError(f"Expression must produce boolean, got {mask_result.dtype}")
            return np.where(mask_result.to_numpy())[0]

        # Boolean mask
        if boolean_mask is not None:
            mask_arr = (
                boolean_mask.to_numpy()
                if isinstance(boolean_mask, pl.Series)
                else np.asarray(boolean_mask)
            )
            if mask_arr.shape[0] != self.n_samples:
                raise DimensionError(f"Mask length {mask_arr.shape[0]} != samples {self.n_samples}")
            if mask_arr.dtype != bool:
                raise ValueError(f"Mask must be boolean, got {mask_arr.dtype}")
            return np.where(mask_arr)[0]

        # Sample indices
        if sample_indices is not None:
            return np.asarray(sample_indices)

        # Sample IDs
        if sample_ids is not None:
            if isinstance(sample_ids, np.ndarray):
                id_list = sample_ids.tolist()
            elif isinstance(sample_ids, pl.Series):
                id_list = sample_ids.to_list()
            else:
                id_list = list(sample_ids)

            all_ids = self.sample_ids.to_list()
            id_to_idx = {sid: i for i, sid in enumerate(all_ids)}
            try:
                return np.array([id_to_idx[sid] for sid in id_list])
            except KeyError as e:
                missing = set(id_list) - set(all_ids)
                raise ValueError(f"Sample IDs not found: {missing}") from e

        raise ValidationError(
            "Must specify: sample_ids, sample_indices, boolean_mask, or polars_expression"
        )

    def _resolve_feature_indices(
        self,
        assay: Assay,
        feature_ids,
        feature_indices,
        boolean_mask,
        polars_expression,
    ) -> np.ndarray:
        """Resolve various input formats to feature indices array."""
        indices: np.ndarray

        # Polars expression
        expr: pl.Expr | None = None
        if isinstance(feature_ids, pl.Expr):
            expr = feature_ids
        elif polars_expression is not None:
            expr = polars_expression

        if expr is not None:
            mask_result = assay.var.select(expr).to_series()
            if mask_result.dtype != pl.Boolean:
                raise ValueError(f"Expression must produce boolean, got {mask_result.dtype}")
            return np.where(mask_result.to_numpy())[0]

        # Boolean mask
        if boolean_mask is not None:
            mask_arr = (
                boolean_mask.to_numpy()
                if isinstance(boolean_mask, pl.Series)
                else np.asarray(boolean_mask)
            )
            if mask_arr.shape[0] != assay.n_features:
                raise DimensionError(
                    f"Mask length {mask_arr.shape[0]} != features {assay.n_features}"
                )
            if mask_arr.dtype != bool:
                raise ValueError(f"Mask must be boolean, got {mask_arr.dtype}")
            return np.where(mask_arr)[0]

        # Feature indices
        if feature_indices is not None:
            return np.asarray(feature_indices)

        # Feature IDs
        if feature_ids is not None:
            if isinstance(feature_ids, np.ndarray):
                id_list = feature_ids.tolist()
            elif isinstance(feature_ids, pl.Series):
                id_list = feature_ids.to_list()
            else:
                id_list = list(feature_ids)

            all_ids = assay.feature_ids.to_list()
            id_to_idx = {fid: i for i, fid in enumerate(all_ids)}
            try:
                return np.array([id_to_idx[fid] for fid in id_list])
            except KeyError as e:
                missing = set(id_list) - set(all_ids)
                raise ValueError(f"Feature IDs not found in assay: {missing}") from e

        raise ValidationError(
            "Must specify: feature_ids, feature_indices, boolean_mask, or polars_expression"
        )

    def _filter_assays_samples(self, indices: np.ndarray, copy: bool) -> dict[str, Assay]:
        """Filter all assays to keep only specified samples."""
        new_assays: dict[str, Assay] = {}

        for assay_name, assay in self.assays.items():
            new_layers: dict[str, ScpMatrix] = {}

            for layer_name, matrix in assay.layers.items():
                new_X = matrix.X[indices, :]
                new_M = matrix.M[indices, :] if matrix.M is not None else None

                if copy:
                    if isinstance(new_X, np.ndarray) or sp.issparse(new_X):
                        new_X = new_X.copy()
                    if new_M is not None and (isinstance(new_M, np.ndarray) or sp.issparse(new_M)):
                        new_M = new_M.copy()

                new_layers[layer_name] = ScpMatrix(X=new_X, M=new_M)

            new_assays[assay_name] = Assay(
                var=assay.var.clone() if copy else assay.var,
                layers=new_layers,
                feature_id_col=assay.feature_id_col,
            )

        return new_assays

    def _filter_assay_features(
        self, assay_name: str, assay: Assay, indices: np.ndarray, copy: bool
    ) -> dict[str, Assay]:
        """Filter specified assay to keep only specified features."""
        new_assays: dict[str, Assay] = {}

        for name, current_assay in self.assays.items():
            if name == assay_name:
                new_layers: dict[str, ScpMatrix] = {}

                for layer_name, matrix in current_assay.layers.items():
                    new_X = matrix.X[:, indices]
                    new_M = matrix.M[:, indices] if matrix.M is not None else None

                    if copy:
                        if isinstance(new_X, np.ndarray) or sp.issparse(new_X):
                            new_X = new_X.copy()
                        if new_M is not None and (isinstance(new_M, np.ndarray) or sp.issparse(new_M)):
                            new_M = new_M.copy()

                    new_layers[layer_name] = ScpMatrix(X=new_X, M=new_M)

                new_var = assay.var[indices, :].clone() if copy else assay.var[indices, :]
                new_assays[name] = Assay(
                    var=new_var,
                    layers=new_layers,
                    feature_id_col=assay.feature_id_col,
                )
            else:
                new_assays[name] = current_assay

        return new_assays

    def _updated_history(
        self, action: str, params: dict[str, Any], description: str
    ) -> list[ProvenanceLog]:
        """Create new history list with added log entry."""
        new_history = list(self.history)
        new_history.append(
            ProvenanceLog(
                timestamp=datetime.now().isoformat(),
                action=action,
                params=params,
                description=description,
            )
        )
        return new_history
