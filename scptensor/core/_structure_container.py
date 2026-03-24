"""Container-level core data structures for ScpTensor."""

from __future__ import annotations

import copy
from datetime import datetime

import numpy as np
import polars as pl
import scipy.sparse as sp

from scptensor.core.assay_alias import resolve_assay_name
from scptensor.core.exceptions import AssayNotFoundError
from scptensor.core.filtering import FilterCriteria, resolve_filter_criteria
from scptensor.core.types import ProvenanceParams

from ._structure_assay import Assay
from ._structure_matrix import AggregationLink, ProvenanceLog, ScpMatrix


def _copy_matrix_if_requested(
    matrix: np.ndarray | sp.spmatrix,
    copy_data: bool,
) -> np.ndarray | sp.spmatrix:
    """Copy matrix payload when requested."""
    if not copy_data:
        return matrix
    if isinstance(matrix, np.ndarray) or sp.issparse(matrix):
        return matrix.copy()
    return matrix


class ScpContainer:
    """Top-level container managing global sample metadata and assay registry."""

    def __init__(
        self,
        obs: pl.DataFrame,
        assays: dict[str, Assay] | None = None,
        links: list[AggregationLink] | None = None,
        history: list[ProvenanceLog] | None = None,
        sample_id_col: str = "_index",
        validate_on_init: bool = True,
    ) -> None:
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
    def sample_ids(self) -> pl.Series:
        """Unique sample identifiers."""
        return self.obs[self.sample_id_col]

    def assay_shape(self, assay_name: str) -> tuple[int, int]:
        """Return ``(n_samples, n_features)`` for an explicit assay."""
        resolved_assay_name = resolve_assay_name(self, assay_name)
        if resolved_assay_name not in self.assays:
            raise AssayNotFoundError(
                assay_name=assay_name,
                available_assays=list(self.assays.keys()),
            )

        assay = self.assays[resolved_assay_name]
        return (self.n_samples, assay.n_features)

    def _validate(self) -> None:
        """Validate all assays have matching sample dimensions."""
        for assay_name, assay in self.assays.items():
            for layer_name, matrix in assay.layers.items():
                if matrix.X.shape[0] != self.n_samples:
                    raise ValueError(
                        f"Assay '{assay_name}', Layer '{layer_name}': "
                        f"Samples {matrix.X.shape[0]} != {self.n_samples}",
                    )

    def validate(self) -> None:
        """Manually validate container integrity."""
        self._validate()
        if self.links:
            self.validate_links()

    def validate_links(self) -> None:
        """Validate that all links connect to existing assays and features."""
        for link in self.links:
            if link.source_assay not in self.assays:
                raise ValueError(f"Link source assay '{link.source_assay}' not found.")
            if link.target_assay not in self.assays:
                raise ValueError(f"Link target assay '{link.target_assay}' not found.")

            linkage_cols = set(link.linkage.columns)
            if "source_id" not in linkage_cols or "target_id" not in linkage_cols:
                raise ValueError(
                    "Linkage table must contain 'source_id' and 'target_id' columns. "
                    f"Got columns: {link.linkage.columns}",
                )

            source_assay = self.assays[link.source_assay]
            target_assay = self.assays[link.target_assay]
            source_ids = set(
                source_assay.feature_ids.cast(pl.Utf8, strict=False).fill_null("").to_list(),
            )
            target_ids = set(
                target_assay.feature_ids.cast(pl.Utf8, strict=False).fill_null("").to_list(),
            )

            source_series = link.linkage["source_id"].cast(pl.Utf8, strict=False).fill_null("")
            target_series = link.linkage["target_id"].cast(pl.Utf8, strict=False).fill_null("")

            missing_source = sorted(
                {val for val in source_series.to_list() if val not in source_ids},
            )
            missing_target = sorted(
                {val for val in target_series.to_list() if val not in target_ids},
            )

            if missing_source:
                preview = ", ".join(missing_source[:5])
                raise ValueError(
                    f"Link source_id values not found in assay '{link.source_assay}': {preview}",
                )
            if missing_target:
                preview = ", ".join(missing_target[:5])
                raise ValueError(
                    f"Link target_id values not found in assay '{link.target_assay}': {preview}",
                )

    def add_assay(self, name: str, assay: Assay) -> ScpContainer:
        """Register a new assay to the container."""
        if name in self.assays:
            raise ValueError(f"Assay '{name}' already exists.")

        for layer_name, matrix in assay.layers.items():
            if matrix.X.shape[0] != self.n_samples:
                raise ValueError(
                    f"New Assay '{name}', Layer '{layer_name}': "
                    f"Samples {matrix.X.shape[0]} != {self.n_samples}",
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
        """Record an operation to the provenance history."""
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
        """Copy the container."""
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
        criteria: FilterCriteria,
        *,
        copy: bool = True,
    ) -> ScpContainer:
        """Filter samples from the container."""
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
        """Filter features for a specific assay."""
        if assay_name not in self.assays:
            raise AssayNotFoundError(assay_name)

        assay = self.assays[assay_name]
        indices = resolve_filter_criteria(criteria, assay, is_sample=False)

        new_assays = self._filter_assay_features(assay_name, assay, indices, copy)
        new_links = self._filter_links_for_assay_features(
            assay_name,
            assay.feature_ids[indices].cast(pl.Utf8, strict=False).fill_null("").to_list(),
        )
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
            links=new_links,
            history=new_history,
            sample_id_col=self.sample_id_col,
        )

    def _filter_assays_samples(self, indices: np.ndarray, copy: bool) -> dict[str, Assay]:
        """Filter all assays to keep only specified samples."""
        new_assays: dict[str, Assay] = {}

        for assay_name, assay in self.assays.items():
            new_layers: dict[str, ScpMatrix] = {}

            for layer_name, matrix in assay.layers.items():
                new_x = _copy_matrix_if_requested(matrix.X[indices, :], copy)
                new_m = (
                    _copy_matrix_if_requested(matrix.M[indices, :], copy)
                    if matrix.M is not None
                    else None
                )
                new_layers[layer_name] = ScpMatrix(X=new_x, M=new_m)

            new_assays[assay_name] = Assay(
                var=assay.var.clone() if copy else assay.var,
                layers=new_layers,
                feature_id_col=assay.feature_id_col,
            )

        return new_assays

    def _filter_assay_features(
        self,
        assay_name: str,
        assay: Assay,
        indices: np.ndarray,
        copy: bool,
    ) -> dict[str, Assay]:
        """Filter specified assay to keep only specified features."""
        new_assays: dict[str, Assay] = {}

        for name, current_assay in self.assays.items():
            if name == assay_name:
                new_layers: dict[str, ScpMatrix] = {}

                for layer_name, matrix in current_assay.layers.items():
                    new_x = _copy_matrix_if_requested(matrix.X[:, indices], copy)
                    new_m = (
                        _copy_matrix_if_requested(matrix.M[:, indices], copy)
                        if matrix.M is not None
                        else None
                    )
                    new_layers[layer_name] = ScpMatrix(X=new_x, M=new_m)

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
        self,
        action: str,
        params: ProvenanceParams,
        description: str,
    ) -> list[ProvenanceLog]:
        """Create new history list with added log entry."""
        new_history = list(self.history)
        new_history.append(
            ProvenanceLog(
                timestamp=datetime.now().isoformat(),
                action=action,
                params=params,
                description=description,
            ),
        )
        return new_history

    def _filter_links_for_assay_features(
        self,
        assay_name: str,
        retained_feature_ids: list[str],
    ) -> list[AggregationLink]:
        """Filter link rows that reference features removed from one assay."""
        retained = set(retained_feature_ids)
        new_links: list[AggregationLink] = []

        for link in self.links:
            linkage = link.linkage

            if link.source_assay == assay_name:
                source_mask = (
                    linkage["source_id"].cast(pl.Utf8, strict=False).fill_null("").is_in(retained)
                )
                linkage = linkage.filter(source_mask)

            if link.target_assay == assay_name:
                target_mask = (
                    linkage["target_id"].cast(pl.Utf8, strict=False).fill_null("").is_in(retained)
                )
                linkage = linkage.filter(target_mask)

            if linkage is link.linkage:
                new_links.append(link)
            else:
                new_links.append(
                    AggregationLink(
                        source_assay=link.source_assay,
                        target_assay=link.target_assay,
                        linkage=linkage,
                    ),
                )

        return new_links

    def list_assays(self) -> list[str]:
        """Return list of assay names in the container."""
        return list(self.assays.keys())

    def list_layers(self, assay_name: str) -> list[str]:
        """Return list of layer names for a specified assay."""
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
        """Return a formatted summary of the container contents."""
        lines = [
            f"ScpContainer with {self.n_samples} samples",
            f"Assays: {len(self.assays)}",
        ]
        for name, assay in self.assays.items():
            layers_count = len(assay.layers)
            lines.append(f"  - {name}: {assay.n_features} features, {layers_count} layers")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """Return HTML representation for Jupyter notebook display."""
        lines = ["<div style='font-family: monospace;'>"]
        lines.append(f"<strong>ScpContainer</strong> with {self.n_samples} samples<br>")
        lines.append(f"<strong>Assays:</strong> {len(self.assays)}<br>")
        for name, assay in self.assays.items():
            layers_count = len(assay.layers)
            lines.append(
                f"&nbsp;&nbsp;* {name}: {assay.n_features} features, {layers_count} layers<br>",
            )
        lines.append("</div>")
        return "".join(lines)


__all__ = ["ScpContainer"]
