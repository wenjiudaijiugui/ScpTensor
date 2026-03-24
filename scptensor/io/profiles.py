"""Vendor profiles and column-resolution helpers for DIA quant-table import."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from scptensor.core.exceptions import ValidationError

Software = Literal["diann", "spectronaut", "auto"]
ResolvedSoftware = Literal["diann", "spectronaut"]
Level = Literal["protein", "peptide"]
TableFormat = Literal["auto", "matrix", "long"]

DIANN_PEPTIDE_MATRIX_RE = re.compile(
    r"(?:\[[^\]]+\]\s*)?(.+?)\.(?:PEP|Precursor)\.(?:Quantity|Intensity)$",
    re.IGNORECASE,
)

SPECTRONAUT_MATRIX_SUFFIXES = (
    "_Quantity",
    "_PeptideQuantity",
    "_PrecursorQuantity",
    "_TotalQuantity",
    "_Intensity",
    "_PeakArea",
    "_NormalizedPeakArea",
)


@dataclass(frozen=True)
class ImportProfile:
    """Column-resolution hints for a ``(software, level)`` pair."""

    feature_candidates: tuple[str, ...]
    quantity_candidates: tuple[str, ...]
    sample_candidates_long: tuple[str, ...]
    fdr_candidates: tuple[str, ...]
    metadata_candidates: tuple[str, ...]


@dataclass(frozen=True)
class ResolvedLongColumns:
    """Resolved long-table columns used by a single import run."""

    feature: str
    quantity: str
    sample: str
    fdr: str | None


PROFILES: dict[tuple[ResolvedSoftware, Level], ImportProfile] = {
    ("diann", "protein"): ImportProfile(
        feature_candidates=("Protein.Group", "Protein.Groups", "Protein.Ids"),
        quantity_candidates=("PG.MaxLFQ", "PG.TopN", "PG.Quantity", "PG.Normalised"),
        sample_candidates_long=("Run",),
        fdr_candidates=("PG.Q.Value", "Global.PG.Q.Value", "Lib.PG.Q.Value"),
        metadata_candidates=(
            "Protein.Group",
            "Protein.Groups",
            "Protein.Ids",
            "Protein.Names",
            "Genes",
            "First.Protein.Description",
            "PG.Q.Value",
            "Global.PG.Q.Value",
            "Lib.PG.Q.Value",
        ),
    ),
    ("diann", "peptide"): ImportProfile(
        feature_candidates=(
            "Precursor.Id",
            "EG.PrecursorId",
            "Modified.Sequence",
            "Stripped.Sequence",
        ),
        quantity_candidates=("Precursor.Normalised", "Precursor.Quantity"),
        sample_candidates_long=("Run",),
        fdr_candidates=("Q.Value", "Global.Q.Value", "Lib.Q.Value"),
        metadata_candidates=(
            "Precursor.Id",
            "Modified.Sequence",
            "Stripped.Sequence",
            "Precursor.Charge",
            "Protein.Group",
            "Protein.Ids",
            "Protein.Names",
            "Genes",
            "Q.Value",
            "Global.Q.Value",
            "Lib.Q.Value",
        ),
    ),
    ("spectronaut", "protein"): ImportProfile(
        feature_candidates=(
            "PG.ProteinGroups",
            "PG.ProteinAccessions",
            "ProteinGroups",
            "ProteinGroup",
        ),
        quantity_candidates=("PG.Quantity", "PG.Normalized", "PG.Normalised"),
        sample_candidates_long=("R.FileName", "R.File.Name"),
        fdr_candidates=("PG.Qvalue", "PG.QValue", "PG.Q.Value"),
        metadata_candidates=(
            "PG.ProteinGroups",
            "PG.ProteinAccessions",
            "PG.ProteinNames",
            "PG.Genes",
            "PG.Description",
            "PG.Qvalue",
            "PG.QValue",
            "PG.Q.Value",
        ),
    ),
    ("spectronaut", "peptide"): ImportProfile(
        feature_candidates=(
            "EG.PrecursorId",
            "PEP.GroupingKey",
            "EG.ModifiedSequence",
            "FG.PrecursorId",
            "FG.Id",
        ),
        quantity_candidates=(
            "EG.TotalQuantity",
            "PEP.Quantity",
            "FG.Quantity",
            "F.NormalizedPeakArea",
            "F.PeakArea",
        ),
        sample_candidates_long=("R.FileName", "R.File.Name"),
        fdr_candidates=(
            "EG.Qvalue",
            "EG.QValue",
            "PEP.Qvalue",
            "PEP.QValue",
            "Q.Value",
        ),
        metadata_candidates=(
            "EG.PrecursorId",
            "PEP.GroupingKey",
            "PEP.GroupingKeyType",
            "EG.ModifiedSequence",
            "FG.PrecursorId",
            "FG.Id",
            "PG.ProteinGroups",
            "PG.ProteinAccessions",
            "EG.ProteinId",
            "EG.Qvalue",
            "EG.QValue",
            "PEP.Qvalue",
            "PEP.QValue",
            "Q.Value",
        ),
    ),
}


def preview_columns(columns: list[str], limit: int = 12) -> str:
    """Return a compact preview string for a column list."""
    head = columns[:limit]
    suffix = "" if len(columns) <= limit else f", ... (+{len(columns) - limit} more)"
    return ", ".join(head) + suffix


def validate_table_format(table_format: str) -> None:
    """Validate runtime ``table_format`` input."""
    valid_formats = {"auto", "matrix", "long"}
    if table_format not in valid_formats:
        raise ValidationError(
            f"Unsupported table_format='{table_format}'. "
            "Supported values: 'auto', 'matrix', 'long'.",
        )


def detect_software(columns: list[str]) -> Literal["diann", "spectronaut", "unknown"]:
    """Detect vendor software from column names."""
    cols = set(columns)

    if {"R.FileName", "R.File.Name"} & cols:
        return "spectronaut"
    if "Run" in cols:
        return "diann"

    if "PG.ProteinGroups" in cols or "PEP.GroupingKey" in cols or "EG.PrecursorId" in cols:
        return "spectronaut"
    if "Protein.Group" in cols or "Precursor.Id" in cols or "Modified.Sequence" in cols:
        return "diann"

    if any(DIANN_PEPTIDE_MATRIX_RE.search(col) for col in columns):
        return "diann"
    if any(col.endswith(SPECTRONAUT_MATRIX_SUFFIXES) for col in columns):
        return "spectronaut"

    return "unknown"


def resolve_profile(software: ResolvedSoftware, level: Level) -> ImportProfile:
    """Resolve the vendor import profile for a `(software, level)` pair."""
    key = (software, level)
    if key not in PROFILES:
        raise ValidationError(f"Unsupported combination: software={software}, level={level}")
    return PROFILES[key]


def resolve_feature_column(columns: list[str], profile: ImportProfile, feature_column: str) -> str:
    """Resolve the feature ID column."""
    if feature_column != "auto":
        if feature_column not in columns:
            raise ValidationError(
                f"feature_column='{feature_column}' not found. "
                f"Available columns: {preview_columns(columns)}",
            )
        return feature_column

    for candidate in profile.feature_candidates:
        if candidate in columns:
            return candidate

    raise ValidationError(
        "Unable to auto-detect feature column. "
        f"Expected one of: {profile.feature_candidates}. "
        f"Available columns: {preview_columns(columns)}",
    )


def resolve_quantity_column(
    columns: list[str],
    profile: ImportProfile,
    quantity_column: str,
) -> str:
    """Resolve the long-table quantity column."""
    if quantity_column != "auto":
        if quantity_column not in columns:
            raise ValidationError(
                f"quantity_column='{quantity_column}' not found. "
                f"Available columns: {preview_columns(columns)}",
            )
        return quantity_column

    for candidate in profile.quantity_candidates:
        if candidate in columns:
            return candidate

    raise ValidationError(
        "Unable to auto-detect quantity column for long-format input. "
        f"Expected one of: {profile.quantity_candidates}. "
        f"Available columns: {preview_columns(columns)}",
    )


def resolve_sample_column_long(
    columns: list[str],
    profile: ImportProfile,
    sample_column: str,
) -> str:
    """Resolve the long-table sample column."""
    if sample_column != "auto":
        if sample_column not in columns:
            raise ValidationError(
                f"sample_column='{sample_column}' not found. "
                f"Available columns: {preview_columns(columns)}",
            )
        return sample_column

    for candidate in profile.sample_candidates_long:
        if candidate in columns:
            return candidate

    raise ValidationError(
        "Unable to auto-detect sample column for long-format input. "
        f"Expected one of: {profile.sample_candidates_long}. "
        f"Available columns: {preview_columns(columns)}",
    )


def resolve_fdr_column(columns: list[str], profile: ImportProfile) -> str | None:
    """Resolve an optional FDR column."""
    for candidate in profile.fdr_candidates:
        if candidate in columns:
            return candidate
    return None


def resolve_long_columns(
    columns: list[str],
    profile: ImportProfile,
    *,
    feature_column: str,
    quantity_column: str,
    sample_column: str,
) -> ResolvedLongColumns:
    """Resolve all required long-format columns once per import."""
    return ResolvedLongColumns(
        feature=resolve_feature_column(columns, profile, feature_column),
        quantity=resolve_quantity_column(columns, profile, quantity_column),
        sample=resolve_sample_column_long(columns, profile, sample_column),
        fdr=resolve_fdr_column(columns, profile),
    )


def resolve_software(software: Software, columns: list[str]) -> ResolvedSoftware:
    """Resolve the vendor software, honoring explicit overrides."""
    if software != "auto":
        if software not in ("diann", "spectronaut"):
            raise ValidationError(
                f"Unsupported software='{software}'. "
                "Supported values: 'diann', 'spectronaut', 'auto'.",
            )
        return software

    guessed = detect_software(columns)
    if guessed == "unknown":
        raise ValidationError(
            "Unable to detect software type from columns. "
            "Please set software='diann' or software='spectronaut' explicitly. "
            f"Columns preview: {preview_columns(columns)}",
        )
    return guessed


__all__ = [
    "DIANN_PEPTIDE_MATRIX_RE",
    "ImportProfile",
    "Level",
    "PROFILES",
    "ResolvedLongColumns",
    "ResolvedSoftware",
    "SPECTRONAUT_MATRIX_SUFFIXES",
    "Software",
    "TableFormat",
    "detect_software",
    "preview_columns",
    "resolve_fdr_column",
    "resolve_feature_column",
    "resolve_long_columns",
    "resolve_profile",
    "resolve_quantity_column",
    "resolve_sample_column_long",
    "resolve_software",
    "validate_table_format",
]
