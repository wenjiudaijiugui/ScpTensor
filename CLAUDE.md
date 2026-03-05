# ScpTensor Project Instructions

**Project:** ScpTensor - DIA-Based Single-Cell Proteomics Preprocessing Toolkit
**Last Updated:** 2026-03-04

---

## Authoritative Contract

`AGENTS.md` at repository root is authoritative.
If this file conflicts with `AGENTS.md`, follow `AGENTS.md`.

---

## Project Scope (Current)

ScpTensor is a DIA-based single-cell proteomics preprocessing package built around
`ScpContainer -> Assay -> ScpMatrix`.

Primary supported scope:
- DIA-NN and Spectronaut quantitative table input
- peptide/precursor to protein aggregation
- protein-level preprocessing (transform, normalization, imputation, integration)
- dimensionality reduction, clustering, and visualization

Out of scope for current package positioning:
- differential expression module
- feature selection module
- non-DIA vendor/file compatibility by default

---

## Current Source Layout

```text
scptensor/
├── core/                 # Data structures, errors, sparse/JIT ops
├── io/                   # DIA-NN / Spectronaut table loading
├── aggregation/          # Peptide/precursor -> protein
├── transformation/       # Log transform
├── normalization/        # Protein-level normalization
├── impute/               # Missing value imputation
├── integration/          # Batch correction
├── dim_reduction/        # PCA / UMAP / t-SNE
├── cluster/              # KMeans / graph clustering
├── qc/                   # QC metrics and filters
├── viz/                  # Base + recipe visualizations
├── autoselect/           # Method evaluation / selection utilities
├── utils/                # Helper utilities
└── standardization/      # Legacy compatibility surface
```

---

## Environment and Commands

Use `uv` by default.

```bash
# Setup
uv venv
uv pip install -e .

# Dev checks
uv run ruff check scptensor tests
uv run mypy scptensor
uv run pytest -q
```

---

## Engineering Conventions

- Use full type hints for public APIs.
- Prefer explicit, actionable error messages.
- Preserve provenance and avoid hidden in-place mutation side effects.
- Keep user-facing docs aligned with actual module availability.
- Add tests for new user-facing code paths and error scenarios.

---

## Documentation Hygiene

When updating docs, verify against real repository state:
- existing directories under `scptensor/`
- exported symbols in `scptensor/__init__.py`
- existing tests under `tests/`
- active workflows under `.github/workflows/`

Avoid hardcoded stale counts (e.g., fixed test file totals) unless automatically generated.
