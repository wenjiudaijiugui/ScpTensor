"""Peptide-level pivot report parser for ScpTensor.

This module provides functionality to parse peptide-level pivot reports from
DIA-NN BGS Factory and Spectronaut formats, converting them into ScpContainer
objects with proper obs/var structure and mask matrices.

Supported Formats:
    DIA-NN BGS Factory: Peptide reports with sample columns prefixed by [N]
    Spectronaut: Peptide pivot reports with sample-specific quantity columns

Main Functions:
    read_pivot_report: Main entry point for parsing pivot reports
    _detect_format: Auto-detect format type from column patterns
    _extract_sample_columns: Identify sample columns from column names
    _parse_diann_bgs: Parse DIA-NN BGS Factory format
    _parse_spectronaut_pivot: Parse Spectronaut pivot format
    _aggregate_to_protein: Optional peptide-to-protein aggregation

Example:
    >>> from scptensor.io import read_pivot_report
    >>> container = read_pivot_report("peptide_report.tsv")
    >>> # With protein aggregation
    >>> container = read_pivot_report("peptide_report.tsv", protein_agg=True)
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

from scptensor.core.exceptions import ValidationError
from scptensor.core.structures import (
    AggregationLink,
    Assay,
    ProvenanceLog,
    ScpContainer,
    ScpMatrix,
)

__all__ = [
    "load_peptide_pivot",
    "_detect_format",
    "_extract_sample_columns",
    "_load_diann_bgs",
    "_load_spectronaut_pivot",
    "_aggregate_to_protein",
]


# ============================================================================
# Format Detection
# ============================================================================


def _detect_format(columns: list[str]) -> Literal["diann_bgs", "spectronaut", "unknown"]:
    """Auto-detect pivot report format from column patterns.

    Parameters
    ----------
    columns : list[str]
        List of column names from the pivot report.

    Returns
    -------
    Literal["diann_bgs", "spectronaut", "unknown"]
        Detected format type or "unknown" if patterns don't match.
    """
    # DIA-NN BGS: [N] filename.PEP.Quantity pattern
    diann_pattern = re.compile(r"\[.*?\].*?\.PEP\.", re.IGNORECASE)
    # Spectronaut: SampleName_Quantity or SampleName_Fragments pattern
    spectronaut_pattern = re.compile(
        r".+_Quantity$|.+_Fragments$|.+_PeptideQuantity$",
        re.IGNORECASE
    )

    diann_count = sum(1 for col in columns if diann_pattern.search(col))
    spectronaut_count = sum(1 for col in columns if spectronaut_pattern.match(col))

    if diann_count > 0:
        return "diann_bgs"
    elif spectronaut_count > 0:
        return "spectronaut"
    return "unknown"


# ============================================================================
# Sample Column Extraction
# ============================================================================


def _extract_sample_columns(
    columns: list[str],
    format_type: Literal["diann_bgs", "spectronaut"],
) -> tuple[list[str], dict[str, str]]:
    """Identify and extract sample columns from pivot report columns.

    Parameters
    ----------
    columns : list[str]
        List of column names from the pivot report.
    format_type : Literal["diann_bgs", "spectronaut"]
        Format type for parsing sample column patterns.

    Returns
    -------
    sample_columns : list[str]
        List of column names containing sample quantification data.
    sample_names : dict[str, str]
        Mapping from column name to clean sample name.
    """
    sample_columns: list[str] = []
    sample_names: dict[str, str] = {}

    if format_type == "diann_bgs":
        # DIA-NN BGS: [N] SampleName.PEP.Quantity
        pattern = re.compile(
            r"\[.*?\]\s+(\S+?)\.(PEP|Precursor)\.(Quantity|Intensity)",
            re.IGNORECASE
        )
        for col in columns:
            match = pattern.search(col)
            if match:
                sample_columns.append(col)
                sample_name = re.sub(r"\.raw$", "", match.group(1), flags=re.IGNORECASE)
                sample_names[col] = sample_name

    elif format_type == "spectronaut":
        # Spectronaut: SampleName_Quantity, SampleName_Fragments, etc.
        quantity_suffixes = ["_Quantity", "_PeptideQuantity", "_PrecursorQuantity", "_Intensity"]
        for col in columns:
            col_lower = col.lower()
            for suffix in quantity_suffixes:
                if col_lower.endswith(suffix.lower()):
                    sample_columns.append(col)
                    sample_name = re.sub(r"\.raw$", "", col[:-len(suffix)], flags=re.IGNORECASE)
                    sample_names[col] = sample_name
                    break
    else:
        raise ValueError(f"Unsupported format type: {format_type}")

    return sample_columns, sample_names


# ============================================================================
# DIA-NN BGS Factory Parser
def _load_diann_bgs(path: Path, assay_name: str) -> ScpContainer:
    """Parse DIA-NN BGS Factory peptide pivot format."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    separator = "\t" if path.suffix in [".tsv", ".txt"] else ","
    df = pl.read_csv(
        path,
        separator=separator,
        null_values=["", "NA", "NaN", "nan", "NULL", "null"],
        ignore_errors=True,
    )

    if df.is_empty():
        raise ValidationError(f"Empty file: {path}")

    format_type = _detect_format(df.columns)
    if format_type != "diann_bgs":
        raise ValidationError(f"File does not appear to be DIA-NN BGS Factory format. Detected: {format_type}")

    sample_columns, sample_names = _extract_sample_columns(df.columns, "diann_bgs")

    if not sample_columns:
        raise ValidationError(f"No sample columns found in {path}")

    metadata_columns = [col for col in df.columns if col not in sample_columns]

    peptide_id_col = None
    for potential_id in ["EG.PrecursorId", "PEP.Group", "Peptide.Sequence", "Modified.Sequence"]:
        if potential_id in metadata_columns:
            peptide_id_col = potential_id
            break
    if peptide_id_col is None:
        peptide_id_col = metadata_columns[0]

    var = df.select(metadata_columns)

    if var[peptide_id_col].is_duplicated().any():
        var = var.with_row_index(name="_row_num")
        var = var.rename({peptide_id_col: "_original_id"})
        var = var.with_columns(
            pl.concat_str([
                pl.col("_original_id").cast(pl.Utf8),
                pl.lit("_"),
                pl.col("_row_num").cast(pl.Utf8),
            ]).alias("_index")
        )
        var = var.drop("_row_num", "_original_id")
    else:
        var = var.rename({peptide_id_col: "_index"})

    quantity_pattern = re.compile(r"\.(Quantity|Intensity)$", re.IGNORECASE)
    quantity_columns = [col for col in sample_columns if quantity_pattern.search(col)]

    x_t = df.select(quantity_columns).to_numpy()
    x_dense = np.nan_to_num(x_t, nan=0.0).astype(np.float64)
    mask_t = np.isnan(x_t).astype(np.int8)

    x = x_dense.T
    m = mask_t.T

    unique_sample_names = [sample_names[col] for col in quantity_columns]
    seen = set()
    ordered_samples = []
    for name in unique_sample_names:
        if name not in seen:
            seen.add(name)
            ordered_samples.append(name)

    obs = pl.DataFrame({"_index": ordered_samples})

    assay = Assay(
        var=var,
        layers={"X": ScpMatrix(X=x, M=m)},
        feature_id_col="_index",
    )

    return ScpContainer(
        obs=obs,
        assays={assay_name: assay},
        sample_id_col="_index",
    )


# ============================================================================
# Spectronaut Pivot Parser
def _load_spectronaut_pivot(path: Path, assay_name: str) -> ScpContainer:
    """Parse Spectronaut peptide pivot format."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    separator = "\t" if path.suffix in [".tsv", ".txt"] else ","
def _load_spectronaut_pivot(path: Path, assay_name: str) -> ScpContainer:
    """Parse Spectronaut peptide pivot format."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    separator = "\t" if path.suffix in [".tsv", ".txt"] else ","
    df = pl.read_csv(
        path,
        separator=separator,
        null_values=["", "NA", "NaN", "nan", "NULL", "null"],
        ignore_errors=True,
    )

    if df.is_empty():
        raise ValidationError(f"Empty file: {path}")

    format_type = _detect_format(df.columns)
    if format_type != "spectronaut":
        raise ValidationError(f"File does not appear to be Spectronaut format. Detected: {format_type}")

    sample_columns, sample_names = _extract_sample_columns(df.columns, "spectronaut")

    if not sample_columns:
        raise ValidationError(f"No sample columns found in {path}")

    metadata_columns = [col for col in df.columns if col not in sample_columns]

    # Prefer PG.ProteinGroups as ID if available (for protein aggregation)
    peptide_id_col = None
    for potential_id in ["PG.ProteinGroups", "EG.PrecursorId", "PeptideSequence", "ModifiedPeptide", "EG.ProteinId"]:
        if potential_id in metadata_columns:
            peptide_id_col = potential_id
            break
    if peptide_id_col is None:
        peptide_id_col = metadata_columns[0]

    var = df.select(metadata_columns)

    if var[peptide_id_col].is_duplicated().any():
        var = var.with_row_index(name="_row_num")
        var = var.rename({peptide_id_col: "_original_id"})
        var = var.with_columns(
            pl.concat_str([
                pl.col("_original_id").cast(pl.Utf8),
                pl.lit("_"),
                pl.col("_row_num").cast(pl.Utf8),
            ]).alias("_index")
        )
        var = var.drop("_row_num", "_original_id")
    else:
        var = var.rename({peptide_id_col: "_index"})

    x_t = df.select(sample_columns).to_numpy()
    x_dense = np.nan_to_num(x_t, nan=0.0).astype(np.float64)
    mask_t = np.isnan(x_t).astype(np.int8)

    x = x_dense.T
    m = mask_t.T

    unique_sample_names = [sample_names[col] for col in sample_columns]
    seen = set()
    ordered_samples = []
    for name in unique_sample_names:
        if name not in seen:
            seen.add(name)
            ordered_samples.append(name)

    obs = pl.DataFrame({"_index": ordered_samples})

    assay = Assay(
        var=var,
        layers={"X": ScpMatrix(X=x, M=m)},
        feature_id_col="_index",
    )

    return ScpContainer(
        obs=obs,
        assays={assay_name: assay},
        sample_id_col="_index",
    )


def _aggregate_to_protein(
    container: ScpContainer,
    source_assay: str,
    target_assay: str = "proteins",
    method: Literal["sum", "max", "mean"] = "sum",
) -> ScpContainer:
    """Aggregate peptides to proteins.

    Parameters
    ----------
    container : ScpContainer
        Container with peptide-level data.
    source_assay : str
        Name of the peptide assay to aggregate.
    target_assay : str, default "proteins"
        Name for the new protein assay.
    method : Literal["sum", "max", "mean"], default "sum"
        Aggregation method for combining peptides.

    Returns
    -------
    ScpContainer
        New container with both peptide and protein assays.

    Raises
    ------
    AssayNotFoundError
        If source_assay does not exist.
    ValidationError
        If protein mapping column is not found.
    """
    if source_assay not in container.assays:
        from scptensor.core.exceptions import AssayNotFoundError
        raise AssayNotFoundError(source_assay)

    peptide_assay = container.assays[source_assay]
    peptide_var = peptide_assay.var
    peptide_matrix = peptide_assay.layers["X"]

    # Find protein ID column
    protein_id_col = None
    for col in ["Protein.Group", "ProteinId", "Protein.Groups", "PG.ProteinGroups"]:
        if col in peptide_var.columns:
            protein_id_col = col
            break
    if protein_id_col is None:
        for col in peptide_var.columns:
            if "protein" in col.lower():
                protein_id_col = col
                break

    if protein_id_col is None:
        raise ValidationError(
            f"Cannot aggregate: no protein ID column found. "
            f"Available: {peptide_var.columns}"
        )

    # Get protein assignments
    protein_ids = peptide_var[protein_id_col].to_numpy()
    unique_proteins = np.unique(protein_ids[~pd_isna(protein_ids)])

    # Build protein matrix
    n_samples = container.n_samples
    n_proteins = len(unique_proteins)
    x_protein = np.zeros((n_samples, n_proteins), dtype=np.float64)
    m_protein = np.zeros((n_samples, n_proteins), dtype=np.int8)

    # Build linkage for AggregationLink
    peptide_ids = peptide_var["_index"].to_numpy()
    linkage_data = {"source_id": [], "target_id": []}

    for i, protein_id in enumerate(unique_proteins):
        mask = protein_ids == protein_id
        indices = np.where(mask)[0]

        # Add to linkage
        for idx in indices:
            linkage_data["source_id"].append(peptide_ids[idx])
            linkage_data["target_id"].append(protein_id)

        # Aggregate peptide values
        peptide_x = peptide_matrix.X[:, indices]
        peptide_m = peptide_matrix.get_m()[:, indices]

        if method == "sum":
            x_protein[:, i] = np.nansum(peptide_x, axis=1)
        elif method == "max":
            x_protein[:, i] = np.nanmax(peptide_x, axis=1)
        else:  # mean
            x_protein[:, i] = np.nanmean(peptide_x, axis=1)

        m_protein[:, i] = np.max(peptide_m, axis=1)

    # Create protein var - _index must be unique (each protein appears once)
    protein_var = pl.DataFrame({"_index": unique_proteins.tolist()})

    # Create protein assay with unique feature IDs
    protein_assay = Assay(
        var=protein_var,
        layers={"X": ScpMatrix(X=x_protein, M=m_protein)},
        feature_id_col="_index",
    )

    # Create aggregation link
    link = AggregationLink(
        source_assay=source_assay,
        target_assay=target_assay,
        linkage=pl.DataFrame(linkage_data),
    )

    # Build new container
    new_assays = dict(container.assays)
    new_assays[target_assay] = protein_assay

    new_history = list(container.history)
    new_history.append(
        ProvenanceLog(
            timestamp=datetime.now().isoformat(),
            action="aggregate_to_protein",
            params={
                "source_assay": source_assay,
                "target_assay": target_assay,
                "method": method,
            },
            description=f"Aggregated {source_assay} to {target_assay} using {method}",
        )
    )

    return ScpContainer(
        obs=container.obs.clone(),
        assays=new_assays,
        links=list(container.links) + [link],
        history=new_history,
        sample_id_col=container.sample_id_col,
    )


def pd_isna(arr: np.ndarray) -> np.ndarray:
    """Check for NaN values in numpy array.

    Parameters
    ----------
    arr : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Boolean mask indicating NaN values.
    """
    if arr.dtype.kind in ["f", "c"]:
        return np.isnan(arr)
    elif arr.dtype == object:
        return np.array([x is None or str(type(x)) == "<class 'polars._polars.NAType'>" for x in arr])
    return np.zeros(arr.shape, dtype=bool)


# ============================================================================
# Main Entry Point
# ============================================================================


def load_peptide_pivot(
    path: str | Path,
    *,
    assay_name: str = "peptides",
    protein_agg: bool = False,
    protein_assay_name: str = "proteins",
    agg_method: Literal["sum", "max", "mean"] = "sum",
) -> ScpContainer:
    """Load peptide-level pivot report and build ScpContainer.

    Automatically detects format type (DIA-NN BGS Factory or Spectronaut)
    and parses the file accordingly. Optionally aggregates peptides to proteins.

    Parameters
    ----------
    path : str | Path
        Path to the pivot report file (TSV or CSV).
    assay_name : str, default "peptides"
        Name for the peptide assay.
    protein_agg : bool, default False
        Whether to aggregate peptides to proteins.
    protein_assay_name : str, default "proteins"
        Name for the protein assay if aggregating.
    agg_method : Literal["sum", "max", "mean"], default "sum"
        Aggregation method for peptide-to-protein summarization.

    Returns
    -------
    ScpContainer
        Container with peptide-level data and optionally protein-level data.

    Raises
    ------
    FileNotFoundError
        If file does not exist.
    ValidationError
        If file format is not recognized or structure is invalid.
    """
    path = Path(path)

    separator = "\t" if path.suffix in [".tsv", ".txt"] else ","
    df = pl.read_csv(
        path,
        separator=separator,
        n_rows=10,
        null_values=["", "NA", "NaN", "nan", "NULL", "null"],
        ignore_errors=True,
    )

    format_type = _detect_format(df.columns)

    if format_type == "unknown":
        raise ValidationError(
            f"Unable to detect format for {path}. "
            f"Expected DIA-NN BGS Factory or Spectronaut format. "
            f"Columns: {df.columns[:5]}..."
        )

    if format_type == "diann_bgs":
        container = _load_diann_bgs(path, assay_name)
    else:
        container = _load_spectronaut_pivot(path, assay_name)

    if protein_agg:
        container = _aggregate_to_protein(
            container,
            source_assay=assay_name,
            target_assay=protein_assay_name,
            method=agg_method,
        )

    return container


# Backward compatibility aliases
read_pivot_report = load_peptide_pivot
load_spectronaut = load_peptide_pivot
load_diann_bgs = load_peptide_pivot
def load_peptide_pivot(
    path: str | Path,
    *,
    assay_name: str = "peptides",
    protein_agg: bool = False,
    protein_assay_name: str = "proteins",
    agg_method: Literal["sum", "max", "mean"] = "sum",
) -> ScpContainer:
    """Load peptide-level pivot report and build ScpContainer.

    Automatically detects format type (DIA-NN BGS Factory or Spectronaut)
    and parses the file accordingly. Optionally aggregates peptides to proteins.

    Parameters
    ----------
    path : str | Path
        Path to the pivot report file (TSV or CSV).
    assay_name : str, default "peptides"
        Name for the peptide assay.
    protein_agg : bool, default False
        Whether to aggregate peptides to proteins.
    protein_assay_name : str, default "proteins"
        Name for the protein assay if aggregating.
    agg_method : Literal["sum", "max", "mean"], default "sum"
        Aggregation method for peptide-to-protein summarization.

    Returns
    -------
    ScpContainer
        Container with peptide-level data and optionally protein-level data.

    Raises
    ------
    FileNotFoundError
        If file does not exist.
    ValidationError
        If file format is not recognized or structure is invalid.
    """
    path = Path(path)

    separator = "\t" if path.suffix in [".tsv", ".txt"] else ","
    df = pl.read_csv(
        path,
        separator=separator,
        n_rows=10,
        null_values=["", "NA", "NaN", "nan", "NULL", "null"],
        ignore_errors=True,
    )

    format_type = _detect_format(df.columns)

    if format_type == "unknown":
        raise ValidationError(
            f"Unable to detect format for {path}. "
            f"Expected DIA-NN BGS Factory or Spectronaut format. "
            f"Columns: {df.columns[:5]}..."
        )

    if format_type == "diann_bgs":
        container = _load_diann_bgs(path, assay_name)
    else:
        container = _load_spectronaut_pivot(path, assay_name)

    if protein_agg:
        container = _aggregate_to_protein(
            container,
            source_assay=assay_name,
            target_assay=protein_assay_name,
            method=agg_method,
        )

    return container