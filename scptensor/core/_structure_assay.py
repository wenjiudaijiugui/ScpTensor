"""Assay-level core data structures for ScpTensor."""

from __future__ import annotations

import numpy as np
import polars as pl
import scipy.sparse as sp

from ._structure_matrix import ScpMatrix


def _copy_matrix_if_requested(
    matrix: np.ndarray | sp.spmatrix | None,
    copy_data: bool,
) -> np.ndarray | sp.spmatrix | None:
    """Return a copied matrix payload when requested."""
    if matrix is None or not copy_data:
        return matrix
    if isinstance(matrix, np.ndarray) or sp.issparse(matrix):
        return matrix.copy()
    return matrix


class Assay:
    """Feature-space object managing data under a specific feature type."""

    def __init__(
        self,
        var: pl.DataFrame,
        layers: dict[str, ScpMatrix] | None = None,
        feature_id_col: str = "_index",
        validate_on_init: bool = True,
    ) -> None:
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
        """Validate all layers have matching feature dimensions."""
        for name, matrix in self.layers.items():
            if matrix.X.shape[1] != self.n_features:
                raise ValueError(
                    f"Layer '{name}': Features {matrix.X.shape[1]} != Assay {self.n_features}",
                )

    def validate(self) -> None:
        """Manually validate assay integrity."""
        self._validate()

    @property
    def n_features(self) -> int:
        """Number of features in this assay."""
        return self.var.height

    @property
    def feature_ids(self) -> pl.Series:
        """Unique feature identifiers."""
        return self.var[self.feature_id_col]

    @property
    # Keep the uppercase shortcut to mirror the canonical layer name "X".
    def X(self) -> np.ndarray | sp.spmatrix | None:  # noqa: N802
        """Shortcut to access the 'X' layer matrix if it exists."""
        layer = self.layers.get("X")
        return layer.X if layer else None

    def add_layer(self, name: str, matrix: ScpMatrix) -> None:
        """Add a new data layer to this assay."""
        if matrix.X.shape[1] != self.n_features:
            raise ValueError(f"Layer has {matrix.X.shape[1]} features, Assay has {self.n_features}")
        self.layers[name] = matrix

    def __repr__(self) -> str:
        """Return string representation of the assay."""
        return f"<Assay n_features={self.n_features}, layers={list(self.layers.keys())}>"

    def subset(self, feature_indices: list[int] | np.ndarray, copy_data: bool = True) -> Assay:
        """Return a new Assay with a subset of features."""
        new_var = self.var[feature_indices, :]
        new_layers: dict[str, ScpMatrix] = {}

        for name, matrix in self.layers.items():
            new_x = _copy_matrix_if_requested(matrix.X[:, feature_indices], copy_data)
            new_m = _copy_matrix_if_requested(
                matrix.M[:, feature_indices] if matrix.M is not None else None,
                copy_data,
            )
            new_layers[name] = ScpMatrix(X=new_x, M=new_m)

        return Assay(var=new_var, layers=new_layers, feature_id_col=self.feature_id_col)


__all__ = ["Assay"]
