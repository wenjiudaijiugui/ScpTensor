# ScpTensor: DIA-Based Single-Cell Proteomics Preprocessing Toolkit

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

ScpTensor is a Python package for DIA-based single-cell proteomics analysis with a strong focus on
reliable vendor-table ingestion and end-to-end quantitative workflows.

Project rules and scope contract: [AGENTS.md](AGENTS.md)

Current product scope is centered on:
- DIA-NN quant output import
- Spectronaut quant output import
- peptide-to-protein aggregation
- transformation, normalization, imputation, dimensionality reduction, clustering, and visualization

## Installation

```bash
git clone https://github.com/yourusername/ScpTensor.git
cd ScpTensor

# Use uv-managed environment
uv venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

uv pip install -e .

# Optional: development tools
uv pip install -e ".[dev]"
```

## Quick Start (DIA-NN)

```python
from pathlib import Path

from scptensor.cluster import cluster_kmeans
from scptensor.dim_reduction import reduce_pca
from scptensor.io import aggregate_to_protein, load_diann
from scptensor.normalization import norm_median
from scptensor.transformation import log_transform
from scptensor.viz import plot_data_overview

# 1) Load DIA-NN long-format report (peptide level)
report = Path("data/dia/diann/PXD054343/1_SC_LF_report.tsv")
container = load_diann(report, level="peptide", table_format="long", assay_name="peptides")

# 2) Aggregate peptide -> protein
container = aggregate_to_protein(
    container,
    source_assay="peptides",
    source_layer="raw",
    target_assay="proteins",
    method="top_n",
)

# 3) Transform + normalize
container = log_transform(
    container,
    assay_name="proteins",
    source_layer="raw",
    new_layer_name="log2",
    base=2.0,
)
container = norm_median(
    container,
    assay_name="proteins",
    source_layer="log2",
    new_layer_name="norm",
)

# 4) Reduce + cluster
container = reduce_pca(container, assay_name="proteins", base_layer="norm", n_components=20)
container = cluster_kmeans(container, assay_name="pca", base_layer="X", n_clusters=6)

# 5) Workflow-level visualization
_ = plot_data_overview(container, assay_name="proteins", layer="norm", groupby="kmeans_k6")
```

## Supported Input Types (IO)

ScpTensor IO currently targets DIA-NN and Spectronaut only.

| Software | Quant Level | File Shape |
| --- | --- | --- |
| DIA-NN | Protein | long + matrix |
| DIA-NN | Peptide/Precursor | long + matrix |
| Spectronaut | Protein | long + matrix |
| Spectronaut | Peptide/Precursor | long + matrix |

Main APIs:
- `scptensor.io.load_quant_table`
- `scptensor.io.load_diann`
- `scptensor.io.load_spectronaut`
- `scptensor.io.load_peptide_pivot`
- `scptensor.io.aggregate_to_protein`

## Documentation

- [Docs index](docs/README.md)
- [DIA-NN / Spectronaut IO spec](docs/io_input_spec_diann_spectronaut.md)
- [Main tutorial notebook](docs/tutorial.ipynb)
- [AutoSelect tutorial](docs/autoselect_tutorial.ipynb)

## Benchmark Assets

- [Benchmark index](benchmark/README.md)
- [AutoSelect benchmark assets](benchmark/autoselect/README.md)

## Development

```bash
# Lint
uv run ruff check scptensor tests

# Tests
uv run pytest -q
```

## License

MIT License. See [LICENSE](LICENSE).
