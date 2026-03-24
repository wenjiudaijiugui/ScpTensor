"""Shared low-level readers and string normalization helpers for quant-table import."""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl

from scptensor.core.exceptions import ValidationError

SUPPORTED_SUFFIXES = {".tsv", ".txt", ".csv", ".parquet"}


def is_vendor_normalized_column(column: str) -> bool:
    """Return whether a resolved quantity column looks vendor-normalized."""
    normalized_tokens = ("normalized", "normalised")
    return any(token in column.lower() for token in normalized_tokens)


def clean_sample_name(name: str) -> str:
    """Normalize sample names from column values."""
    return re.sub(r"(?i)\.raw$", "", str(name).strip())


def make_unique(values: list[str]) -> list[str]:
    """Make values unique by suffixing repeated entries."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for val in values:
        n = seen.get(val, 0)
        if n == 0:
            out.append(val)
        else:
            out.append(f"{val}_{n}")
        seen[val] = n + 1
    return out


def infer_separator(path: Path, delimiter: str | None = None) -> str:
    """Infer delimiter for text tables."""
    if delimiter is not None:
        return delimiter
    if path.suffix.lower() in {".tsv", ".txt"}:
        return "\t"
    return ","


def read_table(path: Path, delimiter: str | None = None, n_rows: int | None = None) -> pl.DataFrame:
    """Read table as a Polars DataFrame (CSV/TSV/Parquet)."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValidationError(
            "Unsupported file extension for quant-table import: "
            f"'{suffix or '<none>'}'. Supported: {sorted(SUPPORTED_SUFFIXES)}",
        )

    try:
        if suffix == ".parquet":
            return pl.read_parquet(path) if n_rows is None else pl.read_parquet(path).head(n_rows)
        return pl.read_csv(
            path,
            separator=infer_separator(path, delimiter),
            n_rows=n_rows,
            null_values=["", "NA", "NaN", "nan", "NULL", "null"],
            infer_schema_length=0,
            ignore_errors=n_rows is not None,
        )
    except Exception as exc:
        raise ValidationError(
            f"Failed to read quant table '{path}'. "
            "Check delimiter/file encoding and ensure the export is a valid "
            "DIA-NN/Spectronaut table.",
        ) from exc


__all__ = [
    "SUPPORTED_SUFFIXES",
    "clean_sample_name",
    "infer_separator",
    "is_vendor_normalized_column",
    "make_unique",
    "read_table",
]
