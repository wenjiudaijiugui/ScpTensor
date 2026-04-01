# ScpTensor Project Contract (Authoritative)

This file defines binding project scope and contributor rules.
If any document, prompt, or comment conflicts with this file, follow this file.

## 1. Project Positioning

- ScpTensor is a **DIA-based single-cell proteomics preprocessing** package.
- User-facing descriptions and documentation must keep both the **DIA** and
  **single-cell** scope explicit.

## 2. Supported Inputs

- Upstream software: **DIA-NN** and **Spectronaut** only.
- Quantification levels: **protein** and **peptide/precursor**.
- Table shapes: **long** and **matrix/pivot**.

## 3. Output and Data Boundary

- The required end product is a **complete protein-level quantitative matrix**.
- Peptide/precursor input must support aggregation to protein level.
- `scptensor.aggregation` is the only peptide/precursor -> protein conversion
  stage.
- All downstream preprocessing modules operate on **protein-level** data.
- Do not present peptide/precursor/run-level methods as default
  protein-normalization methods.

## 4. Canonical I/O Entrypoints

- `scptensor.io.load_quant_table`
- `scptensor.io.load_diann`
- `scptensor.io.load_spectronaut`
- `scptensor.io.load_peptide_pivot`
- `scptensor.io.aggregate_to_protein`

## 5. Repository Boundary

- Core data model: `ScpContainer -> Assay -> ScpMatrix`.
- Stable package areas include:
  `core`, `io`, `aggregation`, `transformation`, `normalization`, `impute`,
  `integration`, `qc`, `viz`, `autoselect`, `utils`, `standardization`.
- Experimental downstream areas may exist:
  `dim_reduction`, `cluster`, `experimental`.
- Experimental downstream helpers are allowed in the repository, but they are
  not part of the core preprocessing release contract.

## 6. Tooling

- Use **uv** as the default environment and package manager.
- Standard setup:
  - `uv venv`
  - `uv pip install -e .`
  - `uv pip install -e ".[dev]"` when only development tools are needed
  - `uv pip install -e ".[all,dev]"` or `uv sync --all-extras && uv sync --group dev` for CI-parity full-repo checks
- Standard checks:
  - `uv run ruff check scptensor tests`
  - `uv run mypy scptensor`
  - `uv run pytest -q`

## 7. Engineering Rules

- Prefer **Polars** for table operations and **NumPy** for numerical work.
- Add dependencies only when they provide clear, necessary value.
- Public APIs must have full type hints.
- I/O and preprocessing errors must be explicit and actionable.
- Preserve provenance; avoid hidden in-place mutation.
- Do not add new I/O compatibility for non-DIA software or unrelated omics
  formats by default.
- Legacy compatibility outside DIA-NN/Spectronaut should be removed or archived,
  not expanded.

## 8. Tests and Documentation

- New user-facing code paths must include tests for:
  - format detection
  - required column validation
  - common user-facing error scenarios
  - successful protein-matrix generation
- Documentation must match the real repository state, especially:
  - package directories under `scptensor/`
  - exports in `scptensor/__init__.py`
  - tests under `tests/`
  - workflows under `.github/workflows/`
- Avoid hardcoded stale counts unless they are generated automatically.

## 9. Evidence and Benchmark Admissibility

- Primary scope stays fixed at DIA-based single-cell proteomics preprocessing.
- Do not add TMT / iTRAQ / isobaric-channel compatibility to I/O paths,
  defaults, examples, or benchmark baselines.
- External literature can be used as benchmark reference only when all of the
  following are available and verifiable:
  - directly accessible article content
  - public code repository with executable analysis pipeline
  - public quantitative matrix plus sample metadata required for reproduction
  - enough method detail to map paper figures to reproducible steps
- If any requirement is missing, the source must be marked as
  `inadmissible_for_reference` and must not be used as a quantitative baseline.
- Inadmissible sources may still be recorded under `docs/article/` for
  improvement inspiration, but must be explicitly labeled as non-reproducible.
