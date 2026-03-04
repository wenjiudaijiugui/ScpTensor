# ScpTensor Project Contract (Authoritative)

This file defines the non-negotiable project scope and engineering conventions.
If other documents conflict with this file, **this file wins**.

## 1. Scope: DIA-Driven Single-Cell

- ScpTensor is a **DIA-based single-cell proteomics preprocessing** package.
- Keep single-cell context in user-facing description and documentation.
- Supported upstream software is limited to:
  - **DIA-NN**
  - **Spectronaut**
- Supported quantitative input levels:
  - **Protein-level**
  - **Peptide/Precursor-level**
- Supported table shapes:
  - **Long format**
  - **Matrix/Pivot format**

## 2. Input/Output Contract

- Input must be DIA-NN or Spectronaut quantitative result files.
- If input is peptide/precursor-level, workflow must support peptide-to-protein aggregation.
- The primary analysis target and final deliverable is a **complete protein-level quantitative matrix**.
- Data-level boundary for module design:
  - `scptensor.aggregation` is the dedicated peptide/precursor -> protein conversion stage.
  - Except for this aggregation stage, all downstream processing modules are defined at the **protein level**.
  - Methods that are inherently peptide/precursor/run-level should not be implemented as protein-layer normalization by default.
- Core I/O entrypoints are:
  - `scptensor.io.load_quant_table`
  - `scptensor.io.load_diann`
  - `scptensor.io.load_spectronaut`
  - `scptensor.io.load_peptide_pivot`
  - `scptensor.io.aggregate_to_protein`

## 3. Environment and Tooling

- Use **uv** as the default environment/package manager.
- Standard workflow:
  - `uv venv`
  - `uv pip install -e .`
  - `uv run pytest`
  - `uv run ruff check`

## 4. Core Dependencies

- Core data processing stack should be centered on:
  - **Polars** (table operations/metadata)
  - **NumPy** (matrix/numerical operations)
- Additional libraries are allowed only when they provide clear, necessary value.

## 5. Non-Goals

- Do not add new I/O compatibility for non-DIA software or unrelated omics formats by default.
- Legacy compatibility layers outside DIA-NN/Spectronaut should be removed or archived instead of expanded.

## 6. Implementation Quality Requirements

- Error messages in I/O and preprocessing must be explicit and actionable.
- New code paths must include tests covering:
  - format detection
  - required column validation
  - common user-facing error scenarios
  - successful protein-matrix generation path
