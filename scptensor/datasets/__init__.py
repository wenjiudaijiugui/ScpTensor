"""Example datasets for ScpTensor tutorials and testing.

This module provides ready-to-use example datasets that users can load
for learning and testing the ScpTensor framework.

Available Datasets:
-------------------

1. **load_toy_example()** - Small synthetic dataset (~100 samples, ~50 features)
   - Multiple batches (3)
   - 20% missing values (mix of MNAR and MCAR)
   - Perfect for quick testing and documentation examples

2. **load_simulated_scrnaseq_like()** - Larger simulated dataset (~500 samples, ~200 features)
   - Multiple cell types (4 groups)
   - Batch effects
   - Realistic missing patterns
   - For comprehensive pipeline testing

3. **load_example_with_clusters()** - Data with known cluster labels
   - Pre-computed cluster labels
   - Distinct cell types for clustering tutorials
   - Includes quality control metrics

Example Usage:
--------------
>>> from scptensor.datasets import load_toy_example
>>> container = load_toy_example()
>>> print(container)
"""

from pathlib import Path
from typing import TYPE_CHECKING

from scptensor.datasets._example import (
    REPRODUCIBILITY_NOTE,
    DatasetSize,
    DatasetType,
    load_example_with_clusters,
    load_simulated_scrnaseq_like,
    load_toy_example,
)

if TYPE_CHECKING:
    from scptensor.core.structures import ScpContainer


def _load_raw_csv(data_path: Path, meta_path: Path | None = None) -> "ScpContainer":
    """Helper to load raw CSV datasets into ScpContainer."""
    import polars as pl

    from scptensor.core.structures import Assay, ScpContainer, ScpMatrix

    # Load data matrix (Features x Cells or Cells x Features)
    # Most of these datasets are Proteins/Peptides as rows, Cells as columns
    df = pl.read_csv(data_path, null_values=["NA", "nan", "NaN", ""])

    # Identify index columns (usually 'Protein' or 'Peptide' or the first column)
    index_col = df.columns[0]
    feature_ids = df[index_col].to_list()

    # Extract data matrix (excluding index column)
    data_cols = [c for c in df.columns if c != index_col]
    sample_ids = data_cols
    x = df.select(data_cols).to_numpy().T  # Transpose to Cells x Features

    # Create Assay
    var = pl.DataFrame({index_col: feature_ids})
    assay = Assay(var=var, layers={"X": ScpMatrix(X=x)}, feature_id_col=index_col)

    # Load metadata if provided
    if meta_path and meta_path.exists():
        obs = pl.read_csv(meta_path)
        # Ensure obs has index matching sample_ids if possible
        # Check if first column is index or if there's a matching column
        if obs.columns[0] != "_index" and "_index" not in obs.columns:
            # Rename first column to _index if it matches sample_ids
            obs = obs.rename({obs.columns[0]: "_index"})
    else:
        # Create minimal obs
        obs = pl.DataFrame({"_index": sample_ids})

    return ScpContainer(obs=obs, assays={"main": assay})


def load_dataset(name: str) -> "ScpContainer":
    """
    Load a real dataset from the data/raw directory.

    Parameters
    ----------
    name : str
        Name of the dataset (e.g., 'sccope', 'plexdia', 'pscope')

    Returns
    -------
    ScpContainer
        Loaded data container
    """
    from pathlib import Path

    # Base path for datasets
    base_path = Path(__file__).parents[2] / "data" / "raw"

    dataset_map = {
        "sccope": ("sccope/Peptides-raw.csv", "sccope/Cells.csv"),
        "plexdia": ("plexdia/plexDIA_data.csv", "plexdia/plexDIA_cells.csv"),
        "pscope_leduc": ("pscope/pSCoPE_Leduc_data.csv", "pscope/pSCoPE_Leduc_cells.csv"),
        "pscope_huffman": (
            "pscope/pSCoPE_Huffman_data.csv",
            "pscope/pSCoPE_Huffman_cells.csv",
        ),
        "nanopots": ("integration/nanoPOTS_data.csv", "integration/nanoPOTS_cells.csv"),
        "n2": ("integration/N2_data.csv", "integration/N2_cells.csv"),
        "scope2_leduc": (
            "integration/SCoPE2_Leduc_data.csv",
            "integration/SCoPE2_Leduc_cells.csv",
        ),
        "spatial": ("spatial/BaselTMA_SP41_83_X14Y2.csv", None),
        "cell_cycle": ("cell_cycle/T-SCP_data.csv", "cell_cycle/T-SCP_cells.csv"),
        "clinical": (
            "clinical/ECCITE_seq_processed_data.csv",
            "clinical/ECCITE_seq_processed_cells.csv",
        ),
    }

    if name not in dataset_map:
        raise ValueError(
            f"Dataset '{name}' not found. Available datasets: {list(dataset_map.keys())}"
        )

    data_rel, meta_rel = dataset_map[name]
    data_path = base_path / data_rel

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found at {data_path}. Please ensure data is correctly placed in data/raw/"
        )

    meta_path = base_path / meta_rel if meta_rel else None
    return _load_raw_csv(data_path, meta_path)


__all__ = [
    "load_toy_example",
    "load_simulated_scrnaseq_like",
    "load_example_with_clusters",
    "load_dataset",
    "DatasetType",
    "DatasetSize",
    "REPRODUCIBILITY_NOTE",
]

__version__ = "0.1.0"
