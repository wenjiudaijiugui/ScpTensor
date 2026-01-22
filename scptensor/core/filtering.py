"""Type-safe filtering utilities for ScpTensor.

This module provides a clean, type-safe API for filtering samples and features
in ScpContainer and Assay objects, eliminating parameter ambiguity and reducing
code duplication.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from scptensor.core.structures import Assay, ScpContainer


@dataclass
class FilterCriteria:
    """
    Type-safe criteria for filtering samples or features.

    Provides a clear single-parameter API that eliminates ambiguity about
    which filtering method to use. Each factory method creates a criteria
    object for a specific filtering approach.

    Parameters
    ----------
    criteria_type : str
        Type of filtering criteria ("ids", "indices", "mask", "expression")
    value : object
        The filtering value (IDs, indices, mask, or expression)

    Raises
    ------
    ValueError
        If criteria_type is not one of the valid types

    Examples
    --------
    >>> # Filter by sample IDs
    >>> criteria = FilterCriteria.by_ids(["sample1", "sample2"])
    >>> container.filter_samples(criteria)

    >>> # Filter by positional indices
    >>> criteria = FilterCriteria.by_indices([0, 1, 2])
    >>> container.filter_samples(criteria)

    >>> # Filter by boolean mask
    >>> criteria = FilterCriteria.by_mask(np.array([True, False, True]))
    >>> container.filter_samples(criteria)

    >>> # Filter by Polars expression
    >>> criteria = FilterCriteria.by_expression(pl.col("n_detected") > 100)
    >>> container.filter_samples(criteria)

    >>> # Filter features by expression
    >>> criteria = FilterCriteria.by_expression(pl.col("n_detected") > 10)
    >>> container.filter_features("proteins", criteria)
    """

    criteria_type: str
    value: object

    def __post_init__(self):
        """Validate criteria type on initialization."""
        valid_types = {"ids", "indices", "mask", "expression"}
        if self.criteria_type not in valid_types:
            raise ValueError(
                f"Invalid criteria_type: {self.criteria_type}. Must be one of {valid_types}"
            )

    @classmethod
    def by_ids(cls, ids: Sequence[str] | np.ndarray | pl.Series) -> FilterCriteria:
        """
        Create criteria for filtering by sample/feature IDs.

        Parameters
        ----------
        ids : Sequence[str] | np.ndarray | pl.Series
            Sample or feature identifiers to keep

        Returns
        -------
        FilterCriteria
            Criteria object for ID-based filtering

        Examples
        --------
        >>> criteria = FilterCriteria.by_ids(["P123", "P456", "P789"])
        >>> container.filter_features("proteins", criteria)
        """
        return cls(criteria_type="ids", value=ids)

    @classmethod
    def by_indices(cls, indices: Sequence[int] | np.ndarray) -> FilterCriteria:
        """
        Create criteria for filtering by positional indices.

        Parameters
        ----------
        indices : Sequence[int] | np.ndarray
            Positional indices to keep

        Returns
        -------
        FilterCriteria
            Criteria object for index-based filtering

        Examples
        --------
        >>> criteria = FilterCriteria.by_indices([0, 5, 10])
        >>> container.filter_samples(criteria)
        """
        return cls(criteria_type="indices", value=indices)

    @classmethod
    def by_mask(cls, mask: np.ndarray | pl.Series) -> FilterCriteria:
        """
        Create criteria for filtering by boolean mask.

        Parameters
        ----------
        mask : np.ndarray | pl.Series
            Boolean mask where True indicates items to keep

        Returns
        -------
        FilterCriteria
            Criteria object for mask-based filtering

        Examples
        --------
        >>> mask = np.array([True, False, True, True, False])
        >>> criteria = FilterCriteria.by_mask(mask)
        >>> container.filter_samples(criteria)
        """
        return cls(criteria_type="mask", value=mask)

    @classmethod
    def by_expression(cls, expr: pl.Expr) -> FilterCriteria:
        """
        Create criteria for filtering by Polars expression.

        Parameters
        ----------
        expr : pl.Expr
            Polars expression that evaluates to a boolean series

        Returns
        -------
        FilterCriteria
            Criteria object for expression-based filtering

        Examples
        --------
        >>> criteria = FilterCriteria.by_expression(pl.col("n_detected") > 100)
        >>> container.filter_samples(criteria)
        >>> criteria = FilterCriteria.by_expression(pl.col("mean_intensity") > 1.0)
        >>> container.filter_features("proteins", criteria)
        """
        return cls(criteria_type="expression", value=expr)


def resolve_filter_criteria(
    criteria: FilterCriteria,
    target: ScpContainer | Assay,
    is_sample: bool = True,
) -> np.ndarray:
    """
    Resolve FilterCriteria to positional index array.

    This unified function replaces _resolve_sample_indices and
    _resolve_feature_indices, eliminating code duplication while
    maintaining full compatibility with existing behavior.

    Parameters
    ----------
    criteria : FilterCriteria
        Filtering criteria object
    target : ScpContainer | Assay
        Target container or assay to filter
    is_sample : bool, default=True
        If True, filter samples (use obs DataFrame)
        If False, filter features (use var DataFrame)

    Returns
    -------
    np.ndarray
        Positional indices of items to keep

    Raises
    ------
    ValueError
        - If expression does not produce boolean result
        - If mask is not boolean dtype
        - If IDs not found in metadata
        - If criteria_type is unknown

    DimensionError
        If mask length does not match number of items

    Examples
    --------
    >>> container = create_test_container(n_samples=100)
    >>> criteria = FilterCriteria.by_expression(pl.col("n_detected") > 50)
    >>> indices = resolve_filter_criteria(criteria, container, is_sample=True)
    >>> print(len(indices))  # Number of samples with n_detected > 50

    >>> assay = container.assays["proteins"]
    >>> criteria = FilterCriteria.by_indices([0, 5, 10])
    >>> indices = resolve_filter_criteria(criteria, assay, is_sample=False)
    >>> print(indices)  # [0, 5, 10]
    """
    from scptensor.core.exceptions import DimensionError

    # Get metadata DataFrame and dimensions based on filter target
    if is_sample:
        # Type narrowing: when is_sample=True, target must be ScpContainer
        metadata_df = target.obs  # type: ignore[attr-defined]
        n_items = target.n_samples  # type: ignore[attr-defined]
        id_col = target.sample_id_col  # type: ignore[attr-defined]
    else:
        # Type narrowing: when is_sample=False, use var for both
        metadata_df = target.var  # type: ignore[attr-defined]
        n_items = target.n_features  # type: ignore[attr-defined]
        id_col = "_index"

    # Resolve based on criteria type
    if criteria.criteria_type == "expression":
        # Type narrowing: value is pl.Expr when criteria_type is "expression"
        expr_value: pl.Expr = criteria.value  # type: ignore[assignment]
        # Evaluate Polars expression to boolean series
        mask_result = metadata_df.select(expr_value).to_series()
        if mask_result.dtype != pl.Boolean:
            raise ValueError(f"Expression must produce boolean result, got {mask_result.dtype}")
        return np.where(mask_result.to_numpy())[0]

    elif criteria.criteria_type == "mask":
        # Convert to numpy array if needed
        mask_arr = (
            criteria.value.to_numpy()  # type: ignore[attr-defined]
            if isinstance(criteria.value, pl.Series)
            else np.asarray(criteria.value)
        )

        # Validate mask dimensions
        if mask_arr.shape[0] != n_items:
            raise DimensionError(
                f"Mask length ({mask_arr.shape[0]}) does not match "
                f"number of {'samples' if is_sample else 'features'} ({n_items})"
            )

        # Validate mask dtype
        if mask_arr.dtype != bool:
            raise ValueError(f"Mask must be boolean array, got {mask_arr.dtype}")

        return np.where(mask_arr)[0]

    elif criteria.criteria_type == "indices":
        # Directly return indices as numpy array
        return np.asarray(criteria.value)

    elif criteria.criteria_type == "ids":
        # Convert IDs to list format
        ids = criteria.value
        if isinstance(ids, np.ndarray):
            id_list = ids.tolist()
        elif isinstance(ids, pl.Series):
            id_list = ids.to_list()
        else:
            id_list = list(ids)  # type: ignore[arg-type]

        # Build ID to index mapping
        all_ids = metadata_df[id_col].to_list()
        id_to_idx = {item_id: i for i, item_id in enumerate(all_ids)}

        # Lookup indices for requested IDs
        try:
            return np.array([id_to_idx[item_id] for item_id in id_list])
        except KeyError as e:
            missing = set(id_list) - set(all_ids)
            raise ValueError(
                f"{'Sample' if is_sample else 'Feature'} IDs not found: {missing}"
            ) from e

    else:
        # This should never happen due to __post_init__ validation
        raise ValueError(f"Unknown criteria type: {criteria.criteria_type}")
