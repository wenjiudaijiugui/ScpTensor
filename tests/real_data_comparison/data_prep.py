"""
Data preparation module for converting Spectronaut TSV to h5ad format.

This module converts single-cell proteomics data from Spectronaut output
to Scanpy-compatible h5ad format for comparison testing.

Data Structure:
    - Input: Spectronaut protein group TSV report
    - Output: Scanpy AnnData object with:
        - X: Expression matrix (proteins x samples)
        - obs: Sample metadata (patient, cell_id, cell_type)
        - var: Protein metadata (gene names, descriptions)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import scanpy as sc


def parse_spectronaut_tsv(tsv_path: str | Path) -> tuple[np.ndarray, list[str], list[str], dict]:
    """Parse Spectronaut TSV file and extract expression matrix and metadata.

    Parameters
    ----------
    tsv_path : str | Path
        Path to Spectronaut TSV file

    Returns
    -------
    X : np.ndarray
        Expression matrix (n_proteins x n_samples)
    protein_names : list[str]
        Protein group names
    sample_names : list[str]
        Raw file names (full basename without extension)
    metadata : dict
        Additional metadata including genes and descriptions
    """
    tsv_path = Path(tsv_path)

    # Read TSV file
    df = pl.read_csv(tsv_path, separator="\t")

    # Identify data columns (PG.PEP values ending with "(Run-Wise)")
    data_cols = [col for col in df.columns if col.endswith("PG.PEP (Run-Wise)")]

    # Extract raw file names from column names
    # Format: [N] filename.raw.PG.PEP (Run-Wise)
    sample_names = []
    for col in data_cols:
        parts = col.split()
        if len(parts) >= 2:
            raw_file = parts[1]  # e.g., "20241107_..._sc_50_GD11.raw.PG.PEP"
            # Strip the .PG.PEP suffix to get the raw file name
            if raw_file.endswith(".PG.PEP"):
                raw_file = raw_file[:-7]  # Remove ".PG.PEP"
            sample_names.append(raw_file)

    # Extract expression matrix
    X = df.select(data_cols).to_numpy()

    # Extract protein metadata
    protein_names = df["PG.ProteinGroups"].to_list()
    genes = df["PG.Genes"].to_list()
    descriptions = df["PG.ProteinDescriptions"].to_list()

    metadata = {
        "genes": genes,
        "descriptions": descriptions,
        "n_proteins": len(protein_names),
        "n_samples": len(sample_names),
    }

    return X, protein_names, sample_names, metadata


def load_experimental_design(
    design_path: str | Path,
) -> pl.DataFrame:
    """Load experimental design template file.

    Parameters
    ----------
    design_path : str | Path
        Path to experimental design TSV file

    Returns
    -------
    design_df : pl.DataFrame
        Experimental design metadata with columns:
        - raw_file: Raw file name
        - patient: Patient identifier
        - cell: Cell number
        - cell_type: Cell type annotation
    """
    design_path = Path(design_path)
    return pl.read_csv(design_path, separator="\t")


def match_samples_to_design(
    sample_names: list[str],
    design_df: pl.DataFrame,
) -> dict[str, dict[str, str]]:
    """Match sample names to experimental design metadata.

    Parameters
    ----------
    sample_names : list[str]
        Raw file names from TSV file (e.g., "20241107_..._sc_50_GD11.raw")
    design_df : pl.DataFrame
        Experimental design metadata

    Returns
    -------
    sample_metadata : dict[str, dict[str, str]]
        Mapping from sample name to metadata dict with keys:
        - patient: Patient identifier
        - cell: Cell number
        - cell_type: Cell type
    """
    sample_metadata = {}

    # Create mapping from raw file name to metadata
    design_files = {}
    for row in design_df.iter_rows(named=True):
        raw_file = row["raw file"]
        design_files[raw_file] = row

    # Match samples
    for sample_name in sample_names:
        if sample_name in design_files:
            row = design_files[sample_name]
            sample_metadata[sample_name] = {
                "patient": str(row["patient"]),
                "cell": str(row["cell"]),
                "cell_type": row["cell type"],
            }
        else:
            # Default metadata for unmatched samples
            sample_metadata[sample_name] = {
                "patient": "unknown",
                "cell": "unknown",
                "cell_type": "unknown",
            }

    return sample_metadata


def convert_to_h5ad(
    tsv_path: str | Path,
    output_path: str | Path,
    design_path: Optional[str | Path] = None,
    min_nonzero: int = 3,
    min_samples_per_protein: int = 3,
    transpose: bool = True,
) -> sc.AnnData:
    """Convert Spectronaut TSV to h5ad format.

    Parameters
    ----------
    tsv_path : str | Path
        Path to input TSV file
    output_path : str | Path
        Path to output h5ad file
    design_path : str | Path, optional
        Path to experimental design TSV file
    min_nonzero : int
        Minimum non-zero values per sample to keep the sample
    min_samples_per_protein : int
        Minimum samples with detections per protein to keep the protein
    transpose : bool
        If True, transpose to have samples as rows (Scanpy convention).
        If False, keep proteins as rows.

    Returns
    -------
    adata : sc.AnnData
        Scanpy AnnData object with:
        - X: Expression matrix (samples x proteins if transpose=True)
        - obs: Sample metadata
        - var: Protein metadata
    """
    tsv_path = Path(tsv_path)
    output_path = Path(output_path)

    # Parse TSV file
    X, protein_names, sample_names, metadata = parse_spectronaut_tsv(tsv_path)

    print(f"Loaded data: {X.shape[0]} proteins x {X.shape[1]} samples")

    # Load experimental design if provided
    sample_metadata = {}
    if design_path is not None:
        design_df = load_experimental_design(design_path)
        sample_metadata = match_samples_to_design(sample_names, design_df)
        print(f"Loaded experimental design: {len(sample_metadata)} samples matched")
    else:
        # Default metadata
        for sample_name in sample_names:
            sample_metadata[sample_name] = {
                "patient": "unknown",
                "cell": sample_name,
                "cell_type": "unknown",
            }

    # Build obs DataFrame (sample metadata)
    obs_dict = {
        "sample_id": sample_names,
        "patient": [sample_metadata.get(s, {}).get("patient", "unknown") for s in sample_names],
        "cell": [sample_metadata.get(s, {}).get("cell", "unknown") for s in sample_names],
        "cell_type": [sample_metadata.get(s, {}).get("cell_type", "unknown") for s in sample_names],
        "n_proteins_detected": np.count_nonzero(X, axis=0).tolist(),
    }

    obs = pl.DataFrame(obs_dict)

    # Build var DataFrame (protein metadata)
    var_dict = {
        "protein_names": protein_names,
        "genes": metadata["genes"],
        "descriptions": metadata["descriptions"],
        "n_samples_detected": np.count_nonzero(X, axis=1).tolist(),
    }

    var = pl.DataFrame(var_dict)

    # Filter samples with too few detections
    valid_samples = obs["n_proteins_detected"] >= min_nonzero
    if not valid_samples.all():
        print(f"Filtering {(~valid_samples).sum()} samples with < {min_nonzero} detections")
        X = X[:, valid_samples]
        obs = obs.filter(valid_samples)
        sample_names = [s for s, v in zip(sample_names, valid_samples) if v]

    # Filter proteins with too few detections
    var_plasm = pl.DataFrame(var_dict)
    valid_proteins = var_plasm["n_samples_detected"] >= min_samples_per_protein
    if not valid_proteins.all():
        print(f"Filtering {(~valid_proteins).sum()} proteins detected in < {min_samples_per_protein} samples")
        X = X[valid_proteins.to_list(), :]
        var = var.filter(valid_proteins)
        protein_names = [p for p, v in zip(protein_names, valid_proteins) if v]

    print(f"After filtering: {X.shape[0]} proteins x {X.shape[1]} samples")

    # Transpose for Scanpy convention (samples as rows)
    if transpose:
        X = X.T

    # Create AnnData object
    adata = sc.AnnData(
        X=X,
        obs=obs.to_pandas(),
        var=var.to_pandas(),
    )

    # Save to h5ad
    adata.write_h5ad(output_path)
    print(f"Saved h5ad file to: {output_path}")

    return adata


def main() -> None:
    """Main entry point for data conversion."""
    # Define paths
    base_dir = Path(__file__).parent.parent / "data" / "PXD061065"
    tsv_path = base_dir / "20250204_112949_gbm_sc_full_fasta_19.4_Report.tsv"
    design_path = base_dir / "experimentalDesignTemplate.txt"
    output_dir = Path(__file__).parent / "real_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "PXD061065.h5ad"

    # Convert
    print("Converting Spectronaut TSV to h5ad format...")
    print(f"Input TSV: {tsv_path}")
    print(f"Design file: {design_path}")
    print(f"Output h5ad: {output_path}")
    print()

    adata = convert_to_h5ad(
        tsv_path=tsv_path,
        output_path=output_path,
        design_path=design_path,
        min_nonzero=100,  # Minimum proteins per sample
        min_samples_per_protein=3,  # Minimum samples per protein
        transpose=True,
    )

    print()
    print("Conversion complete!")
    print(f"AnnData shape: {adata.shape} (samples x proteins)")
    print(f"obs columns: {list(adata.obs.columns)}")
    print(f"var columns: {list(adata.var.columns)}")


if __name__ == "__main__":
    main()
