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
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import scipy.sparse as sp

if TYPE_CHECKING:
    pass

from scptensor.core.exceptions import (
    AssayNotFoundError,
)
from scptensor.core.filtering import FilterCriteria, resolve_filter_criteria
from scptensor.core.types import LayerMetadataDict, ProvenanceParams


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
    params: ProvenanceParams
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
    creation_info: LayerMetadataDict | None = None


# Valid mask codes for validation
_VALID_MASK_CODES = {code.value for code in MaskCode}


def _validate_mask_matrix(M: np.ndarray | sp.spmatrix, X_shape: tuple[int, int]) -> None:
    """
    Validate mask matrix shape and values.

    Parameters
    ----------
    M : np.ndarray | sp.spmatrix
        Mask matrix to validate
    X_shape : tuple[int, int]
        Expected shape from data matrix

    Raises
    ------
    ValueError
        If shape mismatch or invalid mask codes found
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
        """
        Return mask matrix, creating zero matrix if M is None.

        Returns
        -------
        np.ndarray | sp.spmatrix
            Mask matrix, or zero matrix if M is None
        """
        if self.M is not None:
            return self.M

        if sp.issparse(self.X):
            return sp.csr_matrix(self.X.shape, dtype=np.int8)
        return np.zeros(self.X.shape, dtype=np.int8)

    def copy(self) -> ScpMatrix:
        """
        Create a deep copy of the matrix.

        Returns
        -------
        ScpMatrix
            Deep copy of the matrix
        """
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
        validate_on_init: bool = True,
    ) -> None:
        """Initialize an Assay with feature metadata and data layers.

        Parameters
        ----------
        var : pl.DataFrame
            Feature metadata DataFrame. Must contain a unique identifier column.
        layers : dict[str, ScpMatrix] | None, default None
            Dictionary of named data layers. If None, initializes empty dict.
        feature_id_col : str, default "_index"
            Column name in var serving as unique feature identifier.
        validate_on_init : bool, default True
            Whether to validate assay integrity on initialization.
            Set to False to speed up loading of large datasets, then call
            .validate() manually when ready.

        Raises
        ------
        ValueError
            If feature_id_col is not found in var columns or is not unique.
        """
        self.feature_id_col = feature_id_col

        if feature_id_col not in var.columns:
            raise ValueError(f"Feature ID column '{feature_id_col}' not found in var.")

        if var[feature_id_col].n_unique() != var.height:
            raise ValueError(f"Feature ID column '{feature_id_col}' is not unique.")

        self.var: pl.DataFrame = var
        self.layers: dict[str, ScpMatrix] = layers if layers is not None else {}

        if validate_on_init:
            self._validate()

    def _validate(self) -> None:
        """Validate all layers have matching feature dimensions.

        Raises
        ------
        ValueError
            If any layer has feature count different from assay n_features.
        """
        for name, matrix in self.layers.items():
            if matrix.X.shape[1] != self.n_features:
                raise ValueError(
                    f"Layer '{name}': Features {matrix.X.shape[1]} != Assay {self.n_features}"
                )

    def validate(self) -> None:
        """Manually validate assay integrity.

        This method should be called if the Assay was created with
        validate_on_init=False. It performs the same validation checks
        that would have been run during initialization.

        Raises
        ------
        ValueError
            If any layer has feature count different from assay n_features.
        """
        self._validate()

    @property
    def n_features(self) -> int:
        """Number of features in this assay.

        Returns
        -------
        int
            Count of features in the assay.
        """
        return self.var.height

    @property
    def feature_ids(self) -> pl.Series:
        """Unique feature identifiers.

        Returns
        -------
        pl.Series
            Series containing feature IDs from the feature_id_col column.
        """
        return self.var[self.feature_id_col]

    @property
    def X(self) -> np.ndarray | sp.spmatrix | None:
        """Shortcut to access the 'X' layer matrix if it exists.

        Returns
        -------
        np.ndarray | sp.spmatrix | None
            Data matrix from the 'X' layer, or None if layer doesn't exist.
        """
        layer = self.layers.get("X")
        return layer.X if layer else None

    def add_layer(self, name: str, matrix: ScpMatrix) -> None:
        """Add a new data layer to this assay.

        Parameters
        ----------
        name : str
            Layer name (e.g., 'raw', 'log', 'imputed').
        matrix : ScpMatrix
            Matrix object with matching feature dimension.

        Raises
        ------
        ValueError
            If matrix feature count doesn't match assay n_features.
        """
        if matrix.X.shape[1] != self.n_features:
            raise ValueError(f"Layer has {matrix.X.shape[1]} features, Assay has {self.n_features}")
        self.layers[name] = matrix

    def __repr__(self) -> str:
        """Return string representation of the assay.

        Returns
        -------
        str
            String showing feature count and layer names.
        """
        return f"<Assay n_features={self.n_features}, layers={list(self.layers.keys())}>"

    def subset(self, feature_indices: list[int] | np.ndarray, copy_data: bool = True) -> Assay:
        """Return a new Assay with a subset of features.

        Parameters
        ----------
        feature_indices : list[int] | np.ndarray
            Indices of features to keep.
        copy_data : bool, default True
            Whether to copy underlying data.

        Returns
        -------
        Assay
            New Assay with subsetted features and layers.
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
        validate_on_init: bool = True,
    ) -> None:
        """Initialize a ScpContainer with sample metadata and assays.

        Parameters
        ----------
        obs : pl.DataFrame
            Sample metadata DataFrame. Must contain unique sample IDs.
        assays : dict[str, Assay] | None, default None
            Dictionary of named assays. If None, initializes empty dict.
        links : list[AggregationLink] | None, default None
            List of aggregation relationships between assays.
        history : list[ProvenanceLog] | None, default None
            Provenance log of operations. If None, initializes empty list.
        sample_id_col : str, default "_index"
            Column name in obs serving as unique sample identifier.
        validate_on_init : bool, default True
            Whether to validate container integrity on initialization.
            Set to False to speed up loading of large datasets, then call
            .validate() manually when ready.

        Raises
        ------
        ValueError
            If sample_id_col is not found in obs columns or is not unique.
        """
        self.sample_id_col = sample_id_col

        if sample_id_col not in obs.columns:
            raise ValueError(f"Sample ID column '{sample_id_col}' not found in obs.")

        if obs[sample_id_col].n_unique() != obs.height:
            raise ValueError(f"Sample ID column '{sample_id_col}' is not unique.")

        self.obs: pl.DataFrame = obs
        self.assays: dict[str, Assay] = assays if assays is not None else {}
        self.links: list[AggregationLink] = links if links is not None else []
        self.history: list[ProvenanceLog] = history if history is not None else []

        if validate_on_init:
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
        """
        Validate all assays have matching sample dimensions.

        Raises
        ------
        ValueError
            If any assay layer has sample count different from n_samples
        """
        for assay_name, assay in self.assays.items():
            for layer_name, matrix in assay.layers.items():
                if matrix.X.shape[0] != self.n_samples:
                    raise ValueError(
                        f"Assay '{assay_name}', Layer '{layer_name}': "
                        f"Samples {matrix.X.shape[0]} != {self.n_samples}"
                    )

    def validate(self) -> None:
        """Manually validate container integrity.

        This method should be called if the ScpContainer was created with
        validate_on_init=False. It performs the same validation checks
        that would have been run during initialization, including link
        validation if links are present.

        Raises
        ------
        ValueError
            If any assay layer has sample count different from container n_samples,
            or if links reference non-existent assays or features.
        """
        self._validate()
        if self.links:
            self.validate_links()

    def validate_links(self) -> None:
        """
        Validate that all links connect to existing assays and features.

        Raises
        ------
        ValueError
            If any link references non-existent assay
        """
        for link in self.links:
            if link.source_assay not in self.assays:
                raise ValueError(f"Link source assay '{link.source_assay}' not found.")
            if link.target_assay not in self.assays:
                raise ValueError(f"Link target assay '{link.target_assay}' not found.")

    def add_assay(self, name: str, assay: Assay) -> ScpContainer:
        """
        Register a new assay to the container.

        Parameters
        ----------
        name : str
            Assay name (e.g., 'proteins', 'peptides')
        assay : Assay
            Assay object with matching sample dimension

        Returns
        -------
        ScpContainer
            Self for method chaining

        Raises
        ------
        ValueError
            If assay already exists or dimensions don't match
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
        params: ProvenanceParams,
        description: str | None = None,
        software_version: str | None = None,
    ) -> None:
        """
        Record an operation to the provenance history.

        Parameters
        ----------
        action : str
            Name of the operation
        params : ProvenanceParams
            Parameters passed to the operation
        description : str | None, optional
            Human-readable description
        software_version : str | None, optional
            Version of software used
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
        """
        Copy the container.

        Parameters
        ----------
        deep : bool, default True
            If True, create deep copy; otherwise shallow copy

        Returns
        -------
        ScpContainer
            Copied container
        """
        return self.deepcopy() if deep else self.shallow_copy()

    def shallow_copy(self) -> ScpContainer:
        """
        Create a shallow copy of the container.

        Returns
        -------
        ScpContainer
            Shallow copy of the container
        """
        return ScpContainer(
            obs=self.obs,
            assays=self.assays.copy(),
            links=list(self.links),
            history=list(self.history),
            sample_id_col=self.sample_id_col,
        )

    def deepcopy(self) -> ScpContainer:
        """
        Create a deep copy of the container.

        Returns
        -------
        ScpContainer
            Deep copy of the container
        """
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
        criteria: FilterCriteria,
        *,
        copy: bool = True,
    ) -> ScpContainer:
        """Filter samples from the container.

        Parameters
        ----------
        criteria : FilterCriteria
            Filtering criteria object for type-safe filtering. Create using:
            - FilterCriteria.by_ids(["sample1", "sample2"]) - Filter by sample IDs
            - FilterCriteria.by_indices([0, 1, 2]) - Filter by positional indices
            - FilterCriteria.by_mask(mask_array) - Filter by boolean mask
            - FilterCriteria.by_expression(pl.col("n_detected") > 100) - Filter by expression
        copy : bool, default=True
            Whether to copy underlying data

        Returns
        -------
        ScpContainer
            New container with filtered samples

        Raises
        ------
        ValidationError
            If no filtering criteria specified
        DimensionError
            If mask has incorrect dimensions

        Examples
        --------
        >>> from scptensor.core.filtering import FilterCriteria
        >>> criteria = FilterCriteria.by_ids(["sample1", "sample2"])
        >>> filtered = container.filter_samples(criteria)
        >>> criteria = FilterCriteria.by_expression(pl.col("n_detected") > 100)
        >>> filtered = container.filter_samples(criteria)
        """
        # Use unified function to resolve indices
        indices: np.ndarray = resolve_filter_criteria(criteria, self, is_sample=True)

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
        criteria: FilterCriteria,
        *,
        copy: bool = True,
    ) -> ScpContainer:
        """Filter features for a specific assay.

        Parameters
        ----------
        assay_name : str
            Name of the assay to filter
        criteria : FilterCriteria
            Filtering criteria object for type-safe filtering. Create using:
            - FilterCriteria.by_ids(["P123", "P456"]) - Filter by feature IDs
            - FilterCriteria.by_indices([0, 1, 2]) - Filter by positional indices
            - FilterCriteria.by_mask(mask_array) - Filter by boolean mask
            - FilterCriteria.by_expression(pl.col("n_detected") > 10) - Filter by expression
        copy : bool, default=True
            Whether to copy underlying data

        Returns
        -------
        ScpContainer
            New container with filtered features in specified assay

        Raises
        ------
        AssayNotFoundError
            If assay doesn't exist
        ValidationError
            If no filtering criteria specified
        DimensionError
            If mask has incorrect dimensions

        Examples
        --------
        >>> from scptensor.core.filtering import FilterCriteria
        >>> criteria = FilterCriteria.by_ids(["P123", "P456"])
        >>> filtered = container.filter_features("proteins", criteria)
        >>> criteria = FilterCriteria.by_expression(pl.col("n_detected") > 10)
        >>> filtered = container.filter_features("proteins", criteria)
        """
        if assay_name not in self.assays:
            raise AssayNotFoundError(assay_name)

        assay = self.assays[assay_name]

        # Use unified function to resolve indices
        indices = resolve_filter_criteria(criteria, assay, is_sample=False)

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

    def _filter_assays_samples(self, indices: np.ndarray, copy: bool) -> dict[str, Assay]:
        """
        Filter all assays to keep only specified samples.

        Parameters
        ----------
        indices : np.ndarray
            Sample indices to keep
        copy : bool
            Whether to copy underlying data

        Returns
        -------
        dict[str, Assay]
            Dictionary of filtered assays
        """
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
        """
        Filter specified assay to keep only specified features.

        Parameters
        ----------
        assay_name : str
            Name of assay to filter
        assay : Assay
            Assay object to filter
        indices : np.ndarray
            Feature indices to keep
        copy : bool
            Whether to copy underlying data

        Returns
        -------
        dict[str, Assay]
            Dictionary of assays with specified assay filtered
        """
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
                        if new_M is not None and (
                            isinstance(new_M, np.ndarray) or sp.issparse(new_M)
                        ):
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
        self, action: str, params: ProvenanceParams, description: str
    ) -> list[ProvenanceLog]:
        """
        Create new history list with added log entry.

        Parameters
        ----------
        action : str
            Action name to log
        params : ProvenanceParams
            Parameters passed to the operation
        description : str
            Human-readable description

        Returns
        -------
        list[ProvenanceLog]
            New history list with added log entry
        """
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

    def save(
        self,
        path: str | Path,
        *,
        compression: str | None = "gzip",
        compression_level: int = 4,
        overwrite: bool = False,
    ) -> None:
        """Save container to file, auto-detecting format from extension.

        Parameters
        ----------
        path : str | Path
            Output file path. Extension determines format (.h5, .parquet)
        compression : str | None, default "gzip"
            Compression algorithm for HDF5
        compression_level : int, default 4
            Compression level (0-9)
        overwrite : bool, default False
            Whether to overwrite existing file

        Raises
        ------
        ValueError
            If file extension is not recognized
        """
        from scptensor.io import save_hdf5

        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == ".h5" or suffix == ".hdf5":
            save_hdf5(
                self,
                path,
                compression=compression,
                compression_level=compression_level,
                overwrite=overwrite,
            )
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Use .h5")

    @classmethod
    def load(cls, path: str | Path) -> ScpContainer:
        """Load container from file, auto-detecting format from extension.

        Parameters
        ----------
        path : str | Path
            Input file path. Extension determines format

        Returns
        -------
        ScpContainer
            Loaded container

        Raises
        ------
        ValueError
            If file extension is not recognized
        """
        from scptensor.io import load_hdf5

        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == ".h5" or suffix == ".hdf5":
            return load_hdf5(path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Use .h5")

    # ==========================================================================
    # Convenience methods
    # ==========================================================================

    def list_assays(self) -> list[str]:
        """Return list of assay names in the container.

        Returns
        -------
        list[str]
            List of assay names.

        Examples
        --------
        >>> container.list_assays()
        ['proteins', 'peptides']
        """
        return list(self.assays.keys())

    def list_layers(self, assay_name: str) -> list[str]:
        """Return list of layer names for a specified assay.

        Parameters
        ----------
        assay_name : str
            Name of the assay to query.

        Returns
        -------
        list[str]
            List of layer names in the specified assay.

        Raises
        ------
        KeyError
            If assay_name does not exist in the container.
            Includes fuzzy matching suggestions for similar assay names.

        Examples
        --------
        >>> container.list_layers("proteins")
        ['raw', 'log', 'normalized']
        """
        if assay_name not in self.assays:
            from scptensor.core.utils import _find_closest_match

            available = list(self.assays.keys())
            suggestion = _find_closest_match(assay_name, available)

            error_parts = [f"Assay '{assay_name}' not found."]
            if suggestion:
                error_parts.append(f"Did you mean '{suggestion}'?")
            else:
                available_formatted = ", ".join(f"'{k}'" for k in available)
                error_parts.append(f"Available assays: {available_formatted}.")
                error_parts.append("Use list_assays() to see all available assays.")

            raise KeyError(" ".join(error_parts))
        return list(self.assays[assay_name].layers.keys())

    def summary(self) -> str:
        """Return a formatted summary of the container contents.

        Returns
        -------
        str
            Multi-line string summarizing the container structure including
            sample count, assay count, and per-assay details.

        Examples
        --------
        >>> print(container.summary())
        ScpContainer with 100 samples
        Assays: 2
          - proteins: 5000 features, 3 layers
          - peptides: 20000 features, 1 layer
        """
        lines = [
            f"ScpContainer with {self.n_samples} samples",
            f"Assays: {len(self.assays)}",
        ]
        for name, assay in self.assays.items():
            layers_count = len(assay.layers)
            lines.append(f"  - {name}: {assay.n_features} features, {layers_count} layers")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """Return HTML representation for Jupyter notebook display.

        Returns
        -------
        str
            HTML string for rich display in Jupyter notebooks.

        Examples
        --------
        >>> container  # In Jupyter, displays rich HTML output
        """
        lines = ["<div style='font-family: monospace;'>"]
        lines.append(f"<strong>ScpContainer</strong> with {self.n_samples} samples<br>")
        lines.append(f"<strong>Assays:</strong> {len(self.assays)}<br>")
        for name, assay in self.assays.items():
            layers_count = len(assay.layers)
            lines.append(
                f"&nbsp;&nbsp;* {name}: {assay.n_features} features, {layers_count} layers<br>"
            )
        lines.append("</div>")
        return "".join(lines)
