# Integration Module Improvement Plan (DIA Single-Cell Proteomics)

Date: 2026-03-04
Branch: `feat/integration-batch-improvement`

## 1) Objective

Improve `scptensor.integration` for DIA-based single-cell proteomics with:

- better method coverage for protein-level batch correction,
- clearer separation between matrix-level correction vs embedding-level integration,
- stronger diagnostics and method-selection logic,
- benchmark-driven defaults and guardrails.

This plan follows project contract in `AGENTS.md`:

- DIA-NN / Spectronaut scope only,
- downstream modules are protein-level by default,
- explicit and actionable errors,
- tests for user-facing success and failure paths.

## 2) Evidence Summary (from literature review)

### P1/P2/P3 framing

- P1: batch correction and integration methods
- P2: DIA / mass-spec proteomics context
- P3: single-cell benchmark and evaluation context

### Key findings used by this plan

1. No single method dominates all datasets in proteomics benchmarks.
2. Linear/EB methods (ComBat, limma-style adjustments) remain strong matrix-level baselines.
3. Embedding alignment methods (Harmony, Scanorama, MNN) are useful for representation alignment, but should not be treated as universal replacements for matrix-level correction in differential analysis pipelines.
4. Over-correction risk is real; a `no batch correction` baseline must always be part of method selection.

## 3) Current Module Status

Current methods in code:

- `integrate_combat`
- `integrate_harmony`
- `integrate_mnn`
- `integrate_scanorama`

Current diagnostics:

- batch ASW
- batch mixing score
- approximate LISI

Current gaps:

1. No explicit `noBC` baseline method in integration API.
2. No limma-style matrix-level method.
3. ComBat path does not expose clear parametric/non-parametric mode controls.
4. Embedding-level and matrix-level semantics are not explicit enough in API contracts.
5. No dataset-aware method recommendation path with "do not correct" fallback.

## 4) Method Strategy and Scope

### Keep and strengthen

- Keep: ComBat, Harmony, MNN, Scanorama.
- Strengthen: parameter validation, logging, diagnostics, and usage boundaries.

### Add

1. `integrate_none` (no correction baseline)
2. `integrate_limma` (linear-model residual batch adjustment)
3. ComBat mode options (parametric vs non-parametric EB)

### Explicit non-goal in this phase

- Do not add non-DIA I/O support.
- Do not add methods that are not defensible for SCP batch correction without evidence.

## 5) Execution Plan (Milestones)

## M0: API Baseline and Semantics (first)

Deliverables:

1. Add `integrate_none` method and register in integration registry.
2. Standardize default assay naming and document accepted assay names.
3. Add explicit method metadata:
   - `integration_level = matrix` or `embedding`
   - `recommended_for_de = true/false`

Acceptance criteria:

- Unified `integrate()` can run `none`, `combat`, `harmony`, `mnn`, `scanorama`.
- History log records method, level, and key parameters.

## M1: Matrix-Level Methods (high priority)

Deliverables:

1. Extend ComBat:
   - add `eb_mode` option (`parametric` / `nonparametric`),
   - improve confounding checks and error text.
2. Implement `integrate_limma`:
   - linear model based batch adjustment on protein matrix,
   - support optional biological covariates.
3. Add covariate handling consistency across matrix-level methods.

Acceptance criteria:

- Matrix-level methods run on dense and sparse inputs.
- On synthetic batch-shift fixtures, batch-separation metrics improve without obvious collapse of biological separation.

## M2: Embedding-Level Track (medium priority)

Deliverables:

1. Enforce clear contract for Harmony/MNN/Scanorama:
   - input layer type expectations,
   - warning when method output is used as matrix for DE.
2. Add optional standardized pre-step:
   - PCA prep helper for embedding methods.

Acceptance criteria:

- Methods produce deterministic output with fixed seeds where applicable.
- API docs clearly state "embedding integration" boundaries.

## M3: Diagnostics and Auto-Selection (high priority)

Deliverables:

1. Expand quality report to include:
   - batch mixing metrics (ASW, LISI),
   - biology preservation proxy (label ASW or clustering consistency when labels are provided),
   - optional DE stability proxy hooks.
2. Add `integrate_auto`:
   - evaluate candidate methods + `none`,
   - choose best by weighted score,
   - fallback to `none` if no robust improvement.

Acceptance criteria:

- Auto-selection always includes `none`.
- Output report stores full metric table and final selection reason.

## M4: Tests and Benchmarks (must-have)

Deliverables:

1. Unit tests:
   - registration, parameter validation, confounding checks,
   - sparse/dense behavior and shape invariants.
2. Integration tests:
   - synthetic with known batch effects,
   - over-correction guard case.
3. Benchmark scripts:
   - compare `none/combat/limma/harmony/mnn/scanorama` on shared fixtures.

Acceptance criteria:

- New code paths have tests for success path and user-facing failures.
- Benchmark outputs include summary table and selected method rationale.

## M5: Documentation and Migration Notes

Deliverables:

1. Update `README` and integration docstrings with method boundaries.
2. Add user guide section:
   - when to use matrix-level vs embedding-level methods,
   - how to interpret diagnostics.
3. Add changelog entry for new integration API and defaults.

## 6) Proposed Priority Order

1. M0 API baseline and semantics
2. M1 matrix-level methods (`none`, ComBat modes, limma)
3. M3 diagnostics + `integrate_auto`
4. M2 embedding-track cleanup
5. M4 benchmark hardening
6. M5 docs and release notes

## 7) Risks and Boundaries

1. Over-correction can erase biological signal:
   - mitigation: always compare with `none`.
2. Batch and biology confounding:
   - mitigation: strict diagnostics and explicit errors.
3. Sparse high-missing SCP matrices:
   - mitigation: method-level missing handling checks and warnings.
4. External dependency instability (`harmonypy`, `scanorama`):
   - mitigation: graceful dependency checks and fallback paths.

## 8) References (evidence used by plan)

- Nat Commun 2025 DIA-SCP workflow benchmark: https://www.nature.com/articles/s41467-025-59853-w
- Scientific Data 2023 MultiPro benchmark set: https://www.nature.com/articles/s41597-023-01975-7
- Mol Syst Biol 2021 proBatch tutorial: https://pmc.ncbi.nlm.nih.gov/articles/PMC7868445/
- J Proteome Res 2023 BIRCH DIA workflow: https://pmc.ncbi.nlm.nih.gov/articles/PMC10683783/
- Genome Biology 2025 scplainer: https://genomebiology.biomedcentral.com/articles/10.1186/s13059-025-03620-4
- Nat Commun 2021 SCP data integration with Scanorama usage: https://www.nature.com/articles/s41467-021-26592-2
- Nat Commun 2024 diaPASEF SCP with Harmony usage: https://pmc.ncbi.nlm.nih.gov/articles/PMC10884598/
- Briefings in Bioinformatics 2025 SCPline/BIB: https://academic.oup.com/bib/article/26/1/bbae652/7936576
- ComBat original: https://academic.oup.com/biostatistics/article/8/1/118/252073
- MNN original: https://www.nature.com/articles/nbt.4091
- Harmony original: https://www.nature.com/articles/s41592-019-0619-0
- Scanorama original: https://www.nature.com/articles/s41587-019-0113-3
