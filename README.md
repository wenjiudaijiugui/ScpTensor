# ScpTensor: DIA-Based Single-Cell Proteomics Preprocessing Toolkit

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

ScpTensor is a Python package for DIA-based single-cell proteomics preprocessing.
It focuses on robust DIA quant-table ingestion and protein-level preprocessing workflows.

Project scope contract: [AGENTS.md](AGENTS.md)

## Scope

Current supported scope:
- DIA-NN quant output import
- Spectronaut quant output import
- peptide/precursor to protein aggregation
- protein-level preprocessing: transform, normalize, impute, integration
- preprocessing-oriented visualization

Explicit non-goals in current package scope:
- differential expression analysis
- feature selection module
- non-DIA software input support by default

Release-boundary clarification:
- Dimensionality reduction (`reduce_*`) and clustering (`cluster_*`) are
  currently treated as **experimental downstream analysis helpers**, not core
  preprocessing deliverables for release acceptance.
- They are provided via `scptensor.experimental`.

## Installation

```bash
git clone https://github.com/wenjiudaijiugui/ScpTensor.git
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
    new_layer_name="log",
    base=2.0,
)
container = norm_median(
    container,
    assay_name="proteins",
    source_layer="log",
    new_layer_name="norm",
)

# 4) Preprocessing-level visualization
_ = plot_data_overview(container, assay_name="proteins", layer="norm")
```

## Experimental Modules

Experimental downstream dim-reduction and clustering APIs are available under:

```python
from scptensor.experimental import cluster_kmeans, reduce_pca
```

These APIs are intentionally excluded from core preprocessing release criteria.
Boundary contract: [docs/experimental_downstream_contract.md](docs/experimental_downstream_contract.md)

Experimental peptide/PSM QC helper remains in the same namespace for boundary
clarity, but it is **not** a downstream helper. Its role is an
experimental pre-aggregation / peptide-PSM QC module:

```python
from scptensor.experimental import qc_psm
```

Helper contract: [docs/qc_psm_contract.md](docs/qc_psm_contract.md)

## Supported Input Types (I/O)

ScpTensor I/O currently targets DIA-NN and Spectronaut only.

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

Index:
- [Docs index](docs/README.md)
- [Tutorial index](tutorial/README.md)

Contract Index:
- Full docs-level contract index: [docs/README.md#contract](docs/README.md#contract)

Scope / Foundation:
- [Project scope contract](AGENTS.md)
- [Core data contract](docs/core_data_contract.md)
- [Core compute contract](docs/core_compute_contract.md)
- [DIA-NN / Spectronaut I/O contract](docs/io_diann_spectronaut.md)

Stable Preprocessing Modules:
- [Aggregation contract](docs/aggregation_contract.md)
- [Transformation contract](docs/transformation_contract.md)
- [Normalization contract](docs/normalization_contract.md)
- [Standardization contract](docs/standardization_contract.md)
- [Imputation contract](docs/imputation_contract.md)
- [Integration contract](docs/integration_contract.md)
- [QC contract](docs/qc_contract.md)

Supporting Stable Packages:
- [AutoSelect contract](docs/autoselect_contract.md)
- [Utils contract](docs/utils_contract.md)
- [Visualization contract](docs/viz_contract.md)

Release-Boundary / Experimental:
- [Experimental downstream boundary contract](docs/experimental_downstream_contract.md)
- [Experimental PSM QC helper contract](docs/qc_psm_contract.md)

Review Registry:
- [Review manifest](docs/review_manifest_20260312.json)
- [Citation registry](docs/references/citations.json)
- [Citation usage map](docs/references/citation_usage.json)

Note:
- `review_manifest_20260312.json` is a review-only manifest for `review_*.md` files, not a contract manifest.
- Frozen implementation contracts are indexed in [docs/README.md#contract](docs/README.md#contract) and grouped above.

Background / Convergence:
- [Aggregation literature background](docs/aggregation_literature.md)
- [Experimental helper alignment plan](docs/experimental_downstream_alignment_plan.md)
- [Optimization execution checklist](docs/optimization_checklist.md)
- [Runtime baseline spec](docs/runtime_baseline.md)

Tutorial:
- [Main tutorial notebook](tutorial/tutorial.ipynb)
- [AutoSelect tutorial](tutorial/autoselect_tutorial.ipynb)

## Benchmark Assets

- [Benchmark index](benchmark/README.md)
- [Aggregation benchmark README](benchmark/aggregation/README.md)
- [Normalization benchmark README](benchmark/normalization/README.md)
- [Imputation benchmark README](benchmark/imputation/README.md)
- [Integration benchmark README](benchmark/integration/README.md)
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
