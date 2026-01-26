"""CSV format I/O for ScpContainer.

Provides save and load functions for CSV format, creating
a directory structure with separate files for obs, var, and layers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp

from scptensor.core.exceptions import ValidationError
from scptensor.core.structures import Assay, ProvenanceLog, ScpContainer, ScpMatrix

__all__ = [
    "save_csv",
    "load_csv",
    "read_diann",
]


def read_diann(
    path: str | Path,
    *,
    assay_name: str = "proteins",
) -> ScpContainer:
    """Import DIA-NN output.

    Supports:
    1. ``report.pg_matrix.tsv``: The protein group matrix file.
    2. ``report.parquet``: The main report file (Parquet format).

    For ``report.parquet``, performs:
    - Filtering: ``Q.Value <= 0.01`` and ``PG.Q.Value <= 0.01``.
    - Aggregation: Pivots ``PG.MaxLFQ`` to create the protein matrix.

    Parameters
    ----------
    path : str | Path
        Path to the file.
    assay_name : str, optional
        Name of the assay to create. Default is "proteins".

    Returns
    -------
    ScpContainer
        Container with:
        - obs: Sample metadata (filenames).
        - assays[assay_name]:
            - X: MaxLFQ intensities (n_samples x n_features).
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

    # Handle Parquet (Main Report)
    if path.suffix == ".parquet":
        return _read_diann_parquet(path, assay_name)

    # Handle TSV (Matrix)
    return _read_diann_tsv(path, assay_name)


def _read_diann_parquet(path: Path, assay_name: str) -> ScpContainer:
    """Internal handler for report.parquet."""
    # Lazy load and filter
    # Filter: Q.Value <= 0.01 (Precursor FDR) AND PG.Q.Value <= 0.01 (Protein FDR)
    lf = pl.scan_parquet(path)

    required_cols = {"Protein.Group", "Run", "PG.MaxLFQ", "Q.Value", "PG.Q.Value"}
    if not required_cols.issubset(lf.columns):
        missing = required_cols - set(lf.columns)
        raise ValidationError(f"Missing required columns in {path}: {missing}")

    filtered_lf = lf.filter((pl.col("Q.Value") <= 0.01) & (pl.col("PG.Q.Value") <= 0.01))

    # We need to pivot. Polars pivot requires Eager DataFrame.
    # Collect only necessary columns to minimize memory
    # Metadata columns to preserve
    meta_cols = [
        "Protein.Group",
        "Protein.Ids",
        "Protein.Names",
        "Genes",
        "First.Protein.Description",
    ]
    # Check which exist
    available_meta = [c for c in meta_cols if c in lf.columns]

    # Columns to fetch
    cols_to_fetch = list(required_cols.union(available_meta))

    df = filtered_lf.select(cols_to_fetch).collect()

    if df.is_empty():
        raise ValidationError("No data remains after FDR filtering.")

    # Pivot to Matrix (Protein.Group x Run)
    # PG.MaxLFQ should be identical for the same Protein.Group in the same Run
    # We use 'max' aggregator.
    matrix_df = df.pivot(
        index="Protein.Group",
        columns="Run",
        values="PG.MaxLFQ",
        aggregate_function="max",
    ).fill_null(0.0)

    # Extract Var (Protein Metadata)
    # Get unique metadata for each Protein.Group
    var_df = df.select(available_meta).unique(subset=["Protein.Group"])

    # Align Var with Matrix
    # Join var_df to matrix_df on Protein.Group
    # matrix_df has Protein.Group as the first column
    aligned_df = matrix_df.join(var_df, on="Protein.Group", how="left")

    # Extract Obs (Samples)
    # Sample columns are those in matrix_df excluding Protein.Group
    sample_cols = [c for c in matrix_df.columns if c != "Protein.Group"]
    obs = pl.DataFrame({"_index": sample_cols})

    # Extract Var
    var = aligned_df.select(available_meta).rename({"Protein.Group": "_index"})

    # Extract X (Data)
    # Transpose to (n_samples x n_features)
    X_T = aligned_df.select(sample_cols).to_numpy()
    X = X_T.T.astype(np.float64)

    # Create Assay
    assay = Assay(
        var=var,
        layers={"MaxLFQ": ScpMatrix(X=X)},
        feature_id_col="_index",
    )

    return ScpContainer(
        obs=obs,
        assays={assay_name: assay},
        sample_id_col="_index",
    )


def _read_diann_tsv(path: Path, assay_name: str) -> ScpContainer:
    """Internal handler for report.pg_matrix.tsv."""
    # Read TSV with Polars
    # DIA-NN output is tab-separated
    # Handle standard missing value representations
    df = pl.read_csv(path, separator="\t", null_values=["", "NA", "NaN", "nan"])

    # Identify metadata columns
    # Standard DIA-NN columns (Protein.Group is the primary ID)
    meta_cols_set = {
        "Protein.Group",
        "Protein.Ids",
        "Protein.Names",
        "Genes",
        "First.Protein.Description",
    }
    found_meta_cols = [c for c in df.columns if c in meta_cols_set]

    if "Protein.Group" not in found_meta_cols:
        raise ValidationError(
            f"Invalid DIA-NN matrix: 'Protein.Group' column not found in {path}. "
            "Ensure this is a valid report.pg_matrix.tsv file."
        )

    # Extract var (Features)
    var = df.select(found_meta_cols)
    # Rename Protein.Group to _index for ScpTensor compatibility
    var = var.rename({"Protein.Group": "_index"})

    # Extract data (Samples)
    # Any column not in metadata set is considered a sample
    sample_cols = [c for c in df.columns if c not in found_meta_cols]

    if not sample_cols:
        raise ValidationError("No sample columns found in DIA-NN matrix.")

    # Convert to numpy array (n_features x n_samples)
    # Fill nulls with 0.0 or NaN?
    # MaxLFQ implies valid quantification. Missing usually means < LOD or filtered.
    # ScpTensor usually handles sparse/dense.
    # We will keep as NaN for now to represent missingness accurately,
    # or 0.0 if we assume it's intensity.
    # DIA-NN "omits" zero quantities.
    # Let's fill with 0.0 to match standard proteomics matrix format where 0 = missing/undetected
    X_T = df.select(sample_cols).fill_null(0.0).to_numpy()

    # Transpose to (n_samples x n_features)
    X = X_T.T.astype(np.float64)

    # Create obs (Samples)
    obs = pl.DataFrame({"_index": sample_cols})

    # Create Assay
    assay = Assay(
        var=var,
        layers={"MaxLFQ": ScpMatrix(X=X)},
        feature_id_col="_index",
    )

    return ScpContainer(
        obs=obs,
        assays={assay_name: assay},
        sample_id_col="_index",
    )


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
            X_dense = matrix.X.toarray()  # type: ignore[union-attr]
        else:
            X_dense = matrix.X

        # Create DataFrame with sample and feature IDs
        feature_ids = assay.feature_ids.to_list()
        sample_ids = container.sample_ids.to_list()

        data_dict = {"_index": sample_ids}
        for i, fid in enumerate(feature_ids):
            data_dict[str(fid)] = X_dense[:, i]

        data_df = pl.DataFrame(data_dict)
        data_df.write_csv(data_path)

        # Save mask if present and requested
        if mask and matrix.M is not None:
            mask_path = path / f"assay_{assay_name}_{layer_name}_mask.csv"

            if sp.issparse(matrix.M):
                M_dense = matrix.M.toarray()  # type: ignore[union-attr]
            else:
                M_dense = matrix.M

            mask_dict = {"_index": sample_ids}
            for i, fid in enumerate(feature_ids):
                mask_dict[str(fid)] = M_dense[:, i]

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
        X_data = data_df.select(data_df.columns[1:]).to_numpy().astype(np.float64)

        # Load mask if present
        M_data = None
        if mask:
            mask_path = path / f"assay_{assay_name}_{layer_name}_mask.csv"

            if mask_path.exists():
                mask_df = pl.read_csv(mask_path)
                M_data = mask_df.select(mask_df.columns[1:]).to_numpy().astype(np.int8)

        assays[assay_name] = Assay(
            var=var,
            layers={layer_name: ScpMatrix(X=X_data, M=M_data)},
            feature_id_col=feature_id_col,
        )

    return ScpContainer(
        obs=obs,
        assays=assays,
        history=history,
        sample_id_col=sample_id_col,
    )
