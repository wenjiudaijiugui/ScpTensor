# ScpTensor: DIA-Based Single-Cell Proteomics Preprocessing Toolkit

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[English](README.md) | [简体中文](README.zh-CN.md)

ScpTensor is a Python package for DIA-based single-cell proteomics preprocessing.
It focuses on robust DIA quant-table ingestion and protein-level preprocessing workflows.

## Key Features

- **Robust I/O:** Direct import of DIA-NN and Spectronaut quantitative outputs (protein and peptide levels).
- **Comprehensive Preprocessing:** End-to-end protein-level processing including log transformation, normalization, imputation, and batch integration.
- **Aggregation:** Robust peptide/precursor to protein aggregation methods.
- **Contract-Driven:** Strictly defined data boundaries and compute contracts for reproducible results.

**Note:** ScpTensor explicitly does *not* support non-DIA software by default, nor does it perform downstream differential expression or feature selection natively. Downstream tasks like dimensionality reduction (`reduce_*`) and clustering (`cluster_*`) are provided as experimental helpers via `scptensor.experimental`.

## Installation

We recommend using [uv](https://github.com/astral-sh/uv) for fast and reliable environment management.

```bash
git clone https://github.com/wenjiudaijiugui/ScpTensor.git
cd ScpTensor

uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install the stable core preprocessing runtime
uv pip install -e .

# Alternatively, install with optional enhancements:
uv pip install -e ".[viz]"          # Visualization polish
uv pip install -e ".[accel]"        # Numba JIT acceleration
uv pip install -e ".[experimental]" # Downstream helpers (e.g. UMAP)
uv pip install -e ".[all,dev]"      # Full suite for development
```

## Quick Start

The canonical user entrypoint is the Python API. Below is a quick example of processing a DIA-NN report:

```python
from pathlib import Path
from scptensor.io import aggregate_to_protein, load_diann
from scptensor.normalization import norm_median
from scptensor.transformation import log_transform
from scptensor.viz import plot_data_overview

# 1. Load DIA-NN long-format report (peptide level)
report = Path("data/dia/diann/PXD054343/1_SC_LF_report.tsv")
container = load_diann(report, level="peptide", table_format="long", assay_name="peptides")

# 2. Aggregate peptide -> protein
container = aggregate_to_protein(
    container, source_assay="peptides", source_layer="raw", target_assay="proteins", method="top_n"
)

# 3. Transform & Normalize
container = log_transform(container, assay_name="proteins", source_layer="raw", new_layer_name="log", base=2.0)
container = norm_median(container, assay_name="proteins", source_layer="log", new_layer_name="norm")

# 4. Visualize
_ = plot_data_overview(container, assay_name="proteins", layer="norm")
```

For more detailed guides, see the [Stable User Workflows](docs/user_workflows.md) and the [Main Tutorial Notebook](tutorial/tutorial.ipynb).

## Documentation

- **[Full Documentation Site](docs/index.md):** (Run `uv run mkdocs serve` locally)
- **[User Workflows](docs/user_workflows.md):** Canonical workflow guides.
- **[API Reference](docs/api.md):** Complete module and function reference.
- **[Architecture Contracts](docs/README.md#contract):** Core data models, compute semantics, and I/O specifications.

## Community & Contributing

We welcome community contributions! Please review our guidelines before submitting a Pull Request or opening an Issue:

- **[Contributing Guide](CONTRIBUTING.md):** Setup instructions, coding standards, and PR process.
- **[Code of Conduct](CODE_OF_CONDUCT.md):** Our community standards and expectations.
- **[Security Policy](SECURITY.md):** How to responsibly report security vulnerabilities.

*For internal project governance, architectural reviews, and benchmarking, see the `docs/internal/` and `benchmark/` directories.*

## License

This project is licensed under the [MIT License](LICENSE).
