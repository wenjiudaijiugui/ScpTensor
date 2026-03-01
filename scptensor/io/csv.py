"""CSV format I/O for ScpContainer.

Provides save and load functions for CSV format, creating
a directory structure with separate files for obs, var, and layers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import scipy.sparse as sp

from scptensor.core.exceptions import ValidationError
from scptensor.core.structures import Assay, ProvenanceLog, ScpContainer, ScpMatrix

__all__ = [
    "save_csv",
    "load_csv",
    "load_diann",
    "save_diann",
]

def load_diann(
    path: str | Path,
    *,
    assay_name: str = "proteins",
    quantity_column: str = "auto",
) -> ScpContainer:
    """Import DIA-NN output.

    Supports:
    1. ``report.pg_matrix.tsv``: The protein group matrix file.
    2. ``report.parquet/tsv``: The main report file (long format).

    For main report files, performs:
    - Filtering: ``Q.Value <= 0.01`` and ``PG.Q.Value <= 0.01``.
    - Aggregation: Pivots quantity column to create the protein matrix.

    Parameters
    ----------
    path : str | Path
        Path to the file.
    assay_name : str, optional
        Name of the assay to create. Default is "proteins".
    quantity_column : str, optional
        Which quantity column to use. Default "auto" tries:
        "PG.MaxLFQ", "PG.Normalised", "PG.Quantity", "Precursor.Quantity".

    Returns
    -------
    ScpContainer
        Container with:
        - obs: Sample metadata (filenames).
        - assays[assay_name]:
            - X: Intensities (n_samples x n_features).
            - var: Protein metadata.

    Raises
    ------
    FileNotFoundError
        If file does not exist.
    ValidationError
        If file structure is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix == ".parquet":
        return _load_diann_parquet(path, assay_name, quantity_column)
    return _load_diann_tsv_or_long(path, assay_name, quantity_column)
    return _load_diann_tsv_or_long(path, assay_name, quantity_column)


def load_diann(
    path: str | Path,
    *,
    assay_name: str = "proteins",
    quantity_column: str = "auto",
) -> ScpContainer:
    """Import DIA-NN output.

    Supports:
    1. ``report.pg_matrix.tsv``: The protein group matrix file.
    2. ``report.parquet/tsv``: The main report file (long format).

    For main report files, performs:
    - Filtering: ``Q.Value <= 0.01`` and ``PG.Q.Value <= 0.01``.
    - Aggregation: Pivots quantity column to create the protein matrix.

    Parameters
    ----------
    path : str | Path
        Path to the file.
    assay_name : str, optional
        Name of the assay to create. Default is "proteins".
    quantity_column : str, optional
        Which quantity column to use. Default "auto" tries:
        "PG.MaxLFQ", "PG.Normalised", "PG.Quantity", "Precursor.Quantity".

    Returns
    -------
    ScpContainer
        Container with:
        - obs: Sample metadata (filenames).
        - assays[assay_name]:
            - X: Intensities (n_samples x n_features).
            - var: Protein metadata.

    Raises
    ------
    FileNotFoundError
        If file does not exist.
    ValidationError
        If file structure is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix == ".parquet":
        return _load_diann_parquet(path, assay_name, quantity_column)
    return _load_diann_tsv_or_long(path, assay_name, quantity_column)


def save_diann(
    container: ScpContainer,
    path: str | Path,
    *,
    assay_name: str = "proteins",
    layer_name: str = "MaxLFQ",
    format: Literal["tsv", "parquet"] = "tsv",
) -> None:
    """Export ScpContainer to DIA-NN format.

    Supports export to:
    1. TSV format: Protein group matrix file.
    2. Parquet format: Main report file.

    Parameters
    ----------
    container : ScpContainer
        Container to export.
    path : str | Path
        Output file path. Extension determines format:
        - ``.tsv`` for protein group matrix
        - ``.parquet`` for main report
    assay_name : str, optional
        Name of the assay to export. Default is "proteins".
    layer_name : str, optional
        Name of the layer to export. Default is "MaxLFQ".
    format : Literal["tsv", "parquet"], optional
        Output format. Default is "tsv".

    Raises
    ------
    NotImplementedError
        DIA-NN export is not yet implemented.
    ValidationError
        If assay_name or layer_name is not found.
    """
    raise NotImplementedError("DIA-NN export not yet implemented")

def _load_diann_parquet(path: Path, assay_name: str, quantity_column: str = "auto") -> ScpContainer:
    """Internal handler for report.parquet."""
    lf = pl.scan_parquet(path)
    schema_names = lf.collect_schema().names()

    if quantity_column == "auto":
        for col in ["PG.MaxLFQ", "PG.Normalised", "PG.Quantity", "Precursor.Quantity"]:
            if col in schema_names:
                quantity_column = col
                break
        if quantity_column == "auto":
            raise ValidationError(
                f"No quantity column found in {path}. "
                f"Expected one of: PG.MaxLFQ, PG.Normalised, PG.Quantity, Precursor.Quantity"
            )

    required_cols = {"Protein.Group", "Run", quantity_column, "Q.Value", "PG.Q.Value"}
    if not required_cols.issubset(set(schema_names)):
        missing = required_cols - set(schema_names)
        raise ValidationError(f"Missing required columns in {path}: {missing}")

    filtered_lf = lf.filter((pl.col("Q.Value") <= 0.01) & (pl.col("PG.Q.Value") <= 0.01))

    meta_cols = ["Protein.Group", "Protein.Ids", "Protein.Names", "Genes", "First.Protein.Description"]
    available_meta = [c for c in meta_cols if c in schema_names]

    cols_to_fetch = list(required_cols.union(set(available_meta)))
    df = filtered_lf.select(cols_to_fetch).collect()

    if df.is_empty():
        raise ValidationError("No data remains after FDR filtering.")

    matrix_df = df.pivot(
        index="Protein.Group",
        on="Run",
        values=quantity_column,
        aggregate_function="max",
    ).fill_null(0.0)

    var_df = df.select(available_meta).unique(subset=["Protein.Group"])
    aligned_df = matrix_df.join(var_df, on="Protein.Group", how="left")

    sample_cols = [c for c in matrix_df.columns if c != "Protein.Group"]
    obs = pl.DataFrame({"_index": sample_cols})
    var = aligned_df.select(available_meta).rename({"Protein.Group": "_index"})
    x = aligned_df.select(sample_cols).to_numpy().T.astype(np.float64)

    layer_name = quantity_column.replace(".", "_")

    assay = Assay(
        var=var,
        layers={layer_name: ScpMatrix(X=x)},
        feature_id_col="_index",
    )

    return ScpContainer(
        obs=obs,
        assays={assay_name: assay},
        sample_id_col="_index",
    )


def _load_diann_long_format(
    path: Path,
    assay_name: str,
    quantity_column: str = "auto"
) -> ScpContainer:
    """Internal handler for long format TSV (main report)."""
    df = pl.read_csv(
        path,
        separator="\t",
        null_values=["", "NA", "NaN", "nan"],
        infer_schema_length=0,
    )

    if df.is_empty():
        raise ValidationError(f"Empty file: {path}")

    if quantity_column == "auto":
        for col in ["PG.MaxLFQ", "PG.Normalised", "PG.Quantity", "Precursor.Quantity"]:
            if col in df.columns:
                quantity_column = col
                break
        if quantity_column == "auto":
            raise ValidationError(f"No quantity column found in {path}")

    required_cols = ["Protein.Group", "Run", quantity_column, "Q.Value", "PG.Q.Value"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValidationError(f"Missing required columns in {path}: {missing}")

    for col in [quantity_column, "Q.Value", "PG.Q.Value"]:
        df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))

    df_filtered = df.filter(
        (pl.col("Q.Value") <= 0.01) & (pl.col("PG.Q.Value") <= 0.01)
    )

    if df_filtered.is_empty():
        raise ValidationError("No data remains after FDR filtering.")

    df_filtered = df_filtered.with_columns(pl.col(quantity_column).fill_null(0.0))

    matrix_df = df_filtered.pivot(
        index="Protein.Group", on="Run", values=quantity_column, aggregate_function="max"
    ).fill_null(0.0)

    meta_cols_set = {"Protein.Group", "Protein.Ids", "Protein.Names", "Genes", "First.Protein.Description"}
    found_meta_cols = [c for c in df_filtered.columns if c in meta_cols_set]
    var_df = df_filtered.select(found_meta_cols).unique(subset=["Protein.Group"])

    aligned_df = matrix_df.join(var_df, on="Protein.Group", how="left")
    sample_cols = [c for c in matrix_df.columns if c != "Protein.Group"]

    obs = pl.DataFrame({"_index": sample_cols})
    var = aligned_df.select(found_meta_cols).rename({"Protein.Group": "_index"})
    x = aligned_df.select(sample_cols).to_numpy().T.astype(np.float64)

    layer_name = quantity_column.replace(".", "_")
    assay = Assay(var=var, layers={layer_name: ScpMatrix(X=x)}, feature_id_col="_index")

    return ScpContainer(obs=obs, assays={assay_name: assay}, sample_id_col="_index")


def _load_diann_matrix_format(path: Path, assay_name: str) -> ScpContainer:
    """Internal handler for report.pg_matrix.tsv."""
    df = pl.read_csv(
        path, separator="\t", null_values=["", "NA", "NaN", "nan"], infer_schema_length=0
    )

    if df.is_empty():
        raise ValidationError(f"Empty file: {path}")

    meta_cols_set = {"Protein.Group", "Protein.Ids", "Protein.Names", "Genes", "First.Protein.Description"}
    found_meta_cols = [c for c in df.columns if c in meta_cols_set]

    if "Protein.Group" not in found_meta_cols:
        raise ValidationError(f"Invalid DIA-NN matrix: 'Protein.Group' column not found in {path}")

    var = df.select(found_meta_cols).rename({"Protein.Group": "_index"})
    sample_cols = [c for c in df.columns if c not in found_meta_cols]

    if not sample_cols:
        raise ValidationError("No sample columns found in DIA-NN matrix.")

    x_df = df.select(sample_cols).with_columns(
        [pl.col(c).cast(pl.Float64, strict=False).fill_null(0.0) for c in sample_cols]
    )
    x = x_df.to_numpy().T.astype(np.float64)

    obs = pl.DataFrame({"_index": sample_cols})
    assay = Assay(var=var, layers={"MaxLFQ": ScpMatrix(X=x)}, feature_id_col="_index")

    return ScpContainer(obs=obs, assays={assay_name: assay}, sample_id_col="_index")


def _load_diann_tsv_or_long(
    path: Path, assay_name: str, quantity_column: str = "auto"
) -> ScpContainer:
    """Internal handler for TSV - detects format type."""
    df = pl.read_csv(
        path, separator="\t", n_rows=10, null_values=["", "NA", "NaN", "nan"], ignore_errors=True
    )

    if "Protein.Group" in df.columns:
        long_format_indicators = {"Run", "Q.Value", "PG.Q.Value"}
        if long_format_indicators.issubset(set(df.columns)):
            return _load_diann_long_format(path, assay_name, quantity_column)
        else:
            return _load_diann_matrix_format(path, assay_name)

    return _load_diann_long_format(path, assay_name, quantity_column)


def _read_diann_matrix_format(path: Path, assay_name: str) -> ScpContainer:
    """Internal handler for report.pg_matrix.tsv."""
    df = pl.read_csv(
        path, separator="\t", null_values=["", "NA", "NaN", "nan"], infer_schema_length=0
    )

    if df.is_empty():
        raise ValidationError(f"Empty file: {path}")

    meta_cols_set = {"Protein.Group", "Protein.Ids", "Protein.Names", "Genes", "First.Protein.Description"}
    found_meta_cols = [c for c in df.columns if c in meta_cols_set]

    if "Protein.Group" not in found_meta_cols:
        raise ValidationError(f"Invalid DIA-NN matrix: 'Protein.Group' column not found in {path}")

    var = df.select(found_meta_cols).rename({"Protein.Group": "_index"})
    sample_cols = [c for c in df.columns if c not in found_meta_cols]

    if not sample_cols:
        raise ValidationError("No sample columns found in DIA-NN matrix.")

    x_df = df.select(sample_cols).with_columns(
        [pl.col(c).cast(pl.Float64, strict=False).fill_null(0.0) for c in sample_cols]
    )
    x = x_df.to_numpy().T.astype(np.float64)

    obs = pl.DataFrame({"_index": sample_cols})
    assay = Assay(var=var, layers={"MaxLFQ": ScpMatrix(X=x)}, feature_id_col="_index")

    return ScpContainer(obs=obs, assays={assay_name: assay}, sample_id_col="_index")


def _read_diann_tsv_or_long(
    path: Path, assay_name: str, quantity_column: str = "auto"
) -> ScpContainer:
    """Internal handler for TSV - detects format type."""
    df = pl.read_csv(
        path, separator="\t", n_rows=10, null_values=["", "NA", "NaN", "nan"], ignore_errors=True
    )

    if "Protein.Group" in df.columns:
        long_format_indicators = {"Run", "Q.Value", "PG.Q.Value"}
        if long_format_indicators.issubset(set(df.columns)):
            return _read_diann_long_format(path, assay_name, quantity_column)
        else:
            return _read_diann_matrix_format(path, assay_name)

    return _read_diann_long_format(path, assay_name, quantity_column)


def save_csv(
    container: ScpContainer,
    path: str | Path,
    *,
    layer_name: str = "X",
    mask: bool = True,
) -> None:
    """Export ScpContainer to CSV files.

    Creates a directory with the following structure:
    ```
    path/
    ├── obs.csv              # Sample metadata
    ├── assay_<name>_var.csv # Feature metadata per assay
    ├── assay_<name>_<layer>.csv  # Data matrix per assay/layer
    ├── assay_<name>_<layer>_mask.csv  # Mask matrix (if mask=True)
    └── metadata.json        # Container metadata
    ```

    Parameters
    ----------
    container : ScpContainer
        Container to export.
    path : str | Path
        Directory path for output files. Will be created if it doesn't exist.
    layer_name : str, optional
        Name of the layer to export. Default is "X".
    mask : bool, optional
        Whether to export mask matrices. Default is True.

    Raises
    ------
    ValidationError
        If layer_name is not found in assays.
    IOError
        If unable to write files.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Save obs as CSV
    obs_path = path / "obs.csv"
    container.obs.write_csv(obs_path)

    # Save metadata
    metadata = {
        "version": "0.1.0",
        "n_samples": container.n_samples,
        "sample_id_col": container.sample_id_col,
        "assays": list(container.assays.keys()),
        "history": [
            {
                "timestamp": log.timestamp,
                "action": log.action,
                "params": log.params,
                "software_version": log.software_version,
                "description": log.description,
            }
            for log in container.history
        ],
    }
    metadata_path = path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Save assays
    for assay_name, assay in container.assays.items():
        # Save var
        var_path = path / f"assay_{assay_name}_var.csv"
        assay.var.write_csv(var_path)

        # Check if layer exists
        if layer_name not in assay.layers:
            raise ValidationError(f"Layer '{layer_name}' not found in assay '{assay_name}'")

        matrix = assay.layers[layer_name]

        # Save data matrix
        data_path = path / f"assay_{assay_name}_{layer_name}.csv"

        # Convert sparse to dense for CSV
        if sp.issparse(matrix.X):
            x_dense = matrix.X.toarray()  # type: ignore[union-attr]
        else:
            x_dense = matrix.X

        # Create DataFrame with sample and feature IDs
        feature_ids = assay.feature_ids.to_list()
        sample_ids = container.sample_ids.to_list()

        data_dict = {"_index": sample_ids}
        for i, fid in enumerate(feature_ids):
            data_dict[str(fid)] = x_dense[:, i]

        data_df = pl.DataFrame(data_dict)
        data_df.write_csv(data_path)

        # Save mask if present and requested
        if mask and matrix.M is not None:
            mask_path = path / f"assay_{assay_name}_{layer_name}_mask.csv"

            if sp.issparse(matrix.M):
                m_dense = matrix.M.toarray()  # type: ignore[union-attr]
            else:
                m_dense = matrix.M

            mask_dict = {"_index": sample_ids}
            for i, fid in enumerate(feature_ids):
                mask_dict[str(fid)] = m_dense[:, i]

            mask_df = pl.DataFrame(mask_dict)
            mask_df.write_csv(mask_path)


def load_csv(
    path: str | Path,
    *,
    layer_name: str = "X",
    mask: bool = True,
) -> ScpContainer:
    """Load ScpContainer from CSV files.

    Reads data previously saved by :func:`save_csv`.

    Parameters
    ----------
    path : str | Path
        Directory path containing the CSV files.
    layer_name : str, optional
        Name of the layer to load. Default is "X".
    mask : bool, optional
        Whether to load mask matrices. Default is True.

    Returns
    -------
    ScpContainer
        Loaded container.

    Raises
    ------
    ValidationError
        If required files are missing.
    IOError
        If unable to read files.
    """
    path = Path(path)

    # Load metadata
    metadata_path = path / "metadata.json"
    if not metadata_path.exists():
        raise ValidationError(f"metadata.json not found in {path}")

    with open(metadata_path) as f:
        metadata = json.load(f)

    # Load obs
    obs_path = path / "obs.csv"
    if not obs_path.exists():
        raise ValidationError(f"obs.csv not found in {path}")

    obs = pl.read_csv(obs_path)
    sample_id_col = metadata.get("sample_id_col", "_index")

    # Reconstruct history
    history = []
    for log_entry in metadata.get("history", []):
        history.append(
            ProvenanceLog(
                timestamp=log_entry["timestamp"],
                action=log_entry["action"],
                params=log_entry["params"],
                software_version=log_entry.get("software_version"),
                description=log_entry.get("description"),
            )
        )

    # Load assays
    assays: dict[str, Assay] = {}
    for assay_name in metadata.get("assays", []):
        # Load var
        var_path = path / f"assay_{assay_name}_var.csv"
        if not var_path.exists():
            raise ValidationError(f"assay_{assay_name}_var.csv not found in {path}")

        var = pl.read_csv(var_path)

        # Load data matrix
        data_path = path / f"assay_{assay_name}_{layer_name}.csv"
        if not data_path.exists():
            raise ValidationError(f"assay_{assay_name}_{layer_name}.csv not found in {path}")

        data_df = pl.read_csv(data_path)

        # Extract feature IDs from var
        feature_id_col = "_index" if "_index" in var.columns else var.columns[0]
        var[feature_id_col].to_list()

        # Build data matrix
        # First column is sample IDs, rest are features
        data_df.columns[0]
        x_data = data_df.select(data_df.columns[1:]).to_numpy().astype(np.float64)

        # Load mask if present
        m_data = None
        if mask:
            mask_path = path / f"assay_{assay_name}_{layer_name}_mask.csv"

            if mask_path.exists():
                mask_df = pl.read_csv(mask_path)
                m_data = mask_df.select(mask_df.columns[1:]).to_numpy().astype(np.int8)

        assays[assay_name] = Assay(
            var=var,
            layers={layer_name: ScpMatrix(X=x_data, M=m_data)},
            feature_id_col=feature_id_col,
        )

    return ScpContainer(
        obs=obs,
        assays=assays,
        history=history,
        sample_id_col=sample_id_col,
    )
