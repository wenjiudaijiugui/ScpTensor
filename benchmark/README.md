# ScpTensor Benchmarks

`benchmark/` stores executable benchmark scripts, benchmark-specific notes, and
generated assets. It is not the source of truth for stable implementation
contracts.

## Current Directories

- [`aggregation/README.md`](aggregation/README.md): peptide/precursor -> protein aggregation benchmark
- [`normalization/README.md`](normalization/README.md): protein-level normalization benchmark
- [`imputation/README.md`](imputation/README.md): protein-level imputation benchmark
- [`integration/README.md`](integration/README.md): protein-level batch correction benchmark
- [`autoselect/README.md`](autoselect/README.md): AutoSelect scoring and strategy assets

## Boundary

- `benchmark/` owns executable replay and scoring assets, not stable preprocessing contracts.
- If benchmark text conflicts with `AGENTS.md` or `docs/*_contract.md`, contracts win.
- Benchmark-local layer or assay names do not redefine canonical repository naming such as `raw / log / norm / imputed / zscore`.
- Experimental downstream helpers may appear as evaluation endpoints or sensitivity panels, but they do not become stable preprocessing release criteria.
- Public data entrypoints, evidence admissibility, and resource typing should be checked against `docs/internal/review_*.md` and the review manifest before changing benchmark scope.

## Evidence Taxonomy

- `论文证据`: benchmark, design, and method papers
- `数据入口`: stable public dataset or accession landing pages
- `模块规范 / 软件文档`: official manuals, API docs, benchmark module pages
- `资源包`: reusable package or resource distribution pages

Read boundary:

- paper pages justify task design
- dataset pages justify replayable public inputs
- module/spec pages justify interface or metric semantics
- package pages justify reusable assets, not benchmark ownership

Machine-readable manifest:

- [`review_manifest_20260312.json`](../docs/internal/review_manifest_20260312.json)
- [`citations.json`](../docs/references/citations.json)
- [`citation_usage.json`](../docs/references/citation_usage.json)

## Stage Map

- `aggregation`
  - stable boundary: `scptensor.aggregation` is the only peptide/precursor -> protein conversion stage
  - current implementation: `precursor-to-protein auxiliary board`
  - authority docs: `docs/aggregation_contract.md`, `docs/internal/review_aggregation_benchmark_20260312.md`
- `normalization`
  - stable boundary: stage-specific protein normalization benchmark
  - current implementation: `raw -> log_transform -> normalization`; `quantile / trqn` are compared on an explicit logged layer
  - authority docs: `docs/normalization_contract.md`, `docs/internal/review_normalization_20260307.md`, `docs/internal/review_log_scale_20260312.md`
- `imputation`
  - stable boundary: `masked-value recovery + downstream stability`, with `no-imputation` baseline
  - current implementation: `main` and `auxiliary` boards, plus state-aware masking strata
  - authority docs: `docs/imputation_contract.md`, `docs/internal/review_masked_imputation_20260312.md`, `docs/internal/review_state_metrics_20260312.md`
- `integration`
  - stable boundary: report by scenario, not one global winner
  - current implementation: `balanced`, `partially confounded`, and `fully confounded` guardrail scenario
  - authority docs: `docs/integration_contract.md`, `docs/internal/review_batch_correction_20260305.md`
- `autoselect`
  - stable boundary: selection-contract validation assets, not a community leaderboard
  - current implementation: literature score, synthetic normalization stress, and integration strategy comparison
  - authority docs: `docs/autoselect_contract.md`, `docs/internal/review_autoselect_scoring_20260312.md`

## Notes

- Main benchmark interpretation still ends at the protein-level matrix, matching project scope.
- `benchmark/**/data/` and `benchmark/**/outputs/` are normally ignored and should not be treated as tracked documentation assets.
- Historical output snapshots are archival only. Re-run scripts and trust current generated `run_metadata.json` plus score tables instead of stale prose summaries.
