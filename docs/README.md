# Docs Index

`docs/` stores user-facing documentation only.
Benchmark scripts and benchmark outputs should stay under `benchmark/`.

## Current Files

- `../tutorial/README.md`: tutorial index
- `../tutorial/tutorial.ipynb`: main tutorial
- `../tutorial/autoselect_tutorial.ipynb`: AutoSelect tutorial
- `io_diann_spectronaut.md`: DIA-NN / Spectronaut I/O contract
- `aggregation_literature.md`: peptide -> protein aggregation literature notes
- `review_imputation_20260304.md`: DIA single-cell imputation review notes
- `review_batch_correction_20260305.md`: DIA single-cell batch-correction review notes
- `review_normalization_20260307.md`: DIA single-cell normalization review notes
- `review_qc_filtering_20260312.md`: DIA single-cell QC and filtering review notes
- `review_batch_diagnostics_20260312.md`: DIA single-cell batch diagnostics metrics review notes
- `review_autoselect_scoring_20260312.md`: AutoSelect scoring and reporting framework review notes
- `review_missingness_20260312.md`: DIA single-cell missingness semantics and detection-state review notes
- `review_log_scale_20260312.md`: DIA single-cell log-transform and scale-contract review notes
- `review_public_benchmark_data_20260312.md`: DIA single-cell public benchmark datasets and task-design review notes
- `review_masked_imputation_20260312.md`: DIA single-cell masked-value imputation benchmark design review notes
- `review_batch_confounding_20260312.md`: DIA single-cell batch-confounding benchmark design review notes
- `review_io_state_mapping_20260312.md`: DIA-NN / Spectronaut importer state-mapping contract review notes
- `review_aggregation_benchmark_20260312.md`: DIA single-cell aggregation benchmark design review notes
- `review_state_metrics_20260312.md`: DIA single-cell state-aware completeness and uncertainty metrics review notes

## Evidence Taxonomy

All `review_*.md` reviews now use the same source typing contract:

- `论文证据`
  - journal articles, benchmark papers, method papers
- `数据入口`
  - stable dataset pages or accession landing pages
- `模块规范 / 软件文档`
  - official software manuals, API docs, benchmark module pages
- `资源包`
  - reusable package/resource distribution pages
- `本地实现上下文`
  - ScpTensor local code/docs used only to check alignment with external evidence

Entry rule:

- do not use a paper page as a substitute for a dataset entrypoint
- do not use a software manual as a substitute for benchmark evidence
- do not use a package page as a substitute for a public dataset

Machine-readable index:

- `review_manifest_20260312.json`: typed manifest for all `review_*.md` reviews

## Notes

- Keep `docs/` focused on readable documentation.
- Keep executable benchmarks, benchmark reports, and generated benchmark assets in `benchmark/`.
- Do not track notebook runtime artifacts under `docs/`.
- Release boundary: dimensionality reduction and clustering are documented as
  experimental downstream helpers, not core preprocessing deliverables.
- Prefer `scptensor.experimental` imports when documenting `reduce_*` and
  `cluster_*` workflows.
- Prefer `scptensor.viz` canonical `plot_*` APIs in documentation examples.
